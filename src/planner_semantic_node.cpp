// planner_semantic_node.cpp
// 语义地图批量规划节点 - 与 Python plan_nyby.py 对齐
// 读取语义拓扑 JSON，为每条边规划轨迹，并输出更新后的 JSON

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>

#include "ele_planner/offline_ele_planner.h"
#include "planner_common.hpp"
#include "semantic_io.hpp"
#include "tomogram_format.hpp"

namespace pctplanner {

class PlannerSemanticNode : public rclcpp::Node {
public:
  PlannerSemanticNode()
      : rclcpp::Node("pct_planner_semantic"),
        use_quintic_(declare_parameter<bool>("use_quintic", true)),
        max_heading_rate_(declare_parameter<double>("max_heading_rate", 10.0)),
        z_offset_(declare_parameter<double>("z_offset", 1.0)),
        tomogram_path_(declare_parameter<std::string>(
            "tomogram_path", "../../rsc/tomogram/scene_map.bin")),
        input_json_(declare_parameter<std::string>(
            "input_json", "../../rsc/semantic_topology/nyby_b5f3.json")),
        output_json_(declare_parameter<std::string>(
            "output_json",
            "../../rsc/semantic_topology/nyby_b5f3_updated.json")) {

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/pct_path3", 10);

    // 定时器：1 Hz 循环发布不同边的轨迹
    publish_timer_ =
        create_wall_timer(std::chrono::seconds(1),
                          std::bind(&PlannerSemanticNode::PublishLoop, this));

    // 初始化
    Initialize();
  }

private:
  void Initialize() {
    RCLCPP_INFO(get_logger(), "Loading tomogram from: %s",
                tomogram_path_.c_str());

    // 读取 tomogram 文件
    std::vector<uint8_t> tomogram_buffer;
    std::ifstream file(tomogram_path_, std::ios::binary);
    if (!file.is_open()) {
      RCLCPP_ERROR(get_logger(), "Failed to open tomogram file: %s",
                   tomogram_path_.c_str());
      return;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    tomogram_buffer.resize(size);
    file.read(reinterpret_cast<char *>(tomogram_buffer.data()), size);
    file.close();

    // 解析 tomogram
    try {
      ParseTomogram(tomogram_buffer, input_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Tomogram parse failed: %s", e.what());
      return;
    }

    RCLCPP_INFO(
        get_logger(),
        "Tomogram parsed: n_slice=%u dim=%ux%u res=%.3f center=[%.2f,%.2f]",
        input_.n_slice, input_.dim_x, input_.dim_y, input_.resolution,
        input_.center_x, input_.center_y);

    // 加载语义地图
    RCLCPP_INFO(get_logger(), "Loading semantic graph from: %s",
                input_json_.c_str());
    try {
      semantic_graph_ = load_semantic_json(input_json_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to load semantic JSON: %s", e.what());
      return;
    }

    RCLCPP_INFO(get_logger(), "Semantic graph loaded: %zu nodes, %zu edges",
                semantic_graph_.nodes.size(), semantic_graph_.edges.size());

    // 为所有边规划轨迹
    PlanAll();
  }

  void PlanAll() {
    RCLCPP_INFO(get_logger(), "Planning trajectories for all %zu edges...",
                semantic_graph_.edges.size());

    // 创建规划器
    OfflineElePlanner planner(max_heading_rate_, use_quintic_);
    Eigen::MatrixXd gateway_d = input_.gateway.cast<double>();
    planner.InitMap(20.0, 15.0, input_.resolution, input_.n_slice, 0.2,
                    input_.trav, input_.elev_g, input_.elev_c, gateway_d,
                    input_.trav_gy, -input_.trav_gx);

    int success_count = 0;
    for (size_t edge_idx = 0; edge_idx < semantic_graph_.edges.size();
         ++edge_idx) {
      Edge &edge = semantic_graph_.edges[edge_idx];
      edge.clear();

      Eigen::Vector3f start_pos = edge.start_pos;
      Eigen::Vector3f end_pos = edge.end_pos;

      RCLCPP_INFO(get_logger(),
                  "Planning edge %zu: [%.2f,%.2f,%.2f] -> [%.2f,%.2f,%.2f]",
                  edge_idx, start_pos.x(), start_pos.y(), start_pos.z(),
                  end_pos.x(), end_pos.y(), end_pos.z());

      // 添加 z 偏移
      geometry_msgs::msg::Point start_shifted, end_shifted;
      start_shifted.x = start_pos.x();
      start_shifted.y = start_pos.y();
      start_shifted.z = start_pos.z() + z_offset_;
      end_shifted.x = end_pos.x();
      end_shifted.y = end_pos.y();
      end_shifted.z = end_pos.z() + z_offset_;

      Eigen::Vector3i start_idx, goal_idx;
      if (!PosToIdx(start_shifted, input_, start_idx) ||
          !PosToIdx(end_shifted, input_, goal_idx)) {
        RCLCPP_WARN(get_logger(), "Edge %zu: start or goal out of bounds",
                    edge_idx);
        continue;
      }

      // 重新初始化规划器状态（或者创建新的 planner）
      Eigen::Ref<const Eigen::Vector3i> start_ref(start_idx);
      Eigen::Ref<const Eigen::Vector3i> goal_ref(goal_idx);
      if (!planner.Plan(start_ref, goal_ref, true)) {
        RCLCPP_WARN(get_logger(), "Edge %zu: planner failed", edge_idx);
        continue;
      }

      // 获取轨迹
      Eigen::MatrixXd traj_raw;
      Eigen::VectorXd heights;
      if (use_quintic_) {
        const auto &optimizer = planner.get_trajectory_optimizer_wnoj();
        traj_raw = optimizer.GetResultMatrix();
        heights = optimizer.GetResultHeight();
      } else {
        const auto &optimizer = planner.get_trajectory_optimizer();
        traj_raw = optimizer.GetResultMatrix();
        heights = optimizer.GetResultHeight();
      }

      const int y_idx = static_cast<int>(traj_raw.cols() / 2);
      for (int i = 0; i < traj_raw.rows(); ++i) {
        double x = traj_raw(i, 0);
        double y = traj_raw(i, y_idx);
        double z = heights(i);

        // 转换到地图坐标
        Eigen::Vector3d p(x, y, z);
        GridToMap(input_, p);

        edge.append_point(p.x(), p.y(), p.z());
      }

      RCLCPP_INFO(get_logger(), "Edge %zu: planned %d waypoints", edge_idx,
                  edge.trajectory.num_waypoints);
      success_count++;
    }

    RCLCPP_INFO(get_logger(), "Planning complete: %d/%zu edges successful",
                success_count, semantic_graph_.edges.size());

    // 写入更新后的 JSON
    try {
      write_semantic_json(semantic_graph_, output_json_);
      RCLCPP_INFO(get_logger(), "Wrote updated semantic graph to: %s",
                  output_json_.c_str());
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to write output JSON: %s", e.what());
    }
  }

  void PublishLoop() {
    if (semantic_graph_.edges.empty())
      return;

    current_edge_idx_ = (current_edge_idx_ + 1) % semantic_graph_.edges.size();
    nav_msgs::msg::Path path_msg =
        semantic_graph_.edges[current_edge_idx_].traj2ros();
    path_msg.header.stamp = now();
    path_pub_->publish(path_msg);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  bool use_quintic_;
  double max_heading_rate_;
  double z_offset_;
  std::string tomogram_path_;
  std::string input_json_;
  std::string output_json_;

  PlannerInput input_;
  SemanticMap semantic_graph_;
  size_t current_edge_idx_ = 0;
};

} // namespace pctplanner

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerSemanticNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
