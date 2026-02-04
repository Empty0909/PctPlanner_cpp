// planner_semantic_node.cpp
// 语义地图批量规划节点 - 与 Python plan_nyby.py 对齐
// 读取语义拓扑 JSON，为每条边规划轨迹，并输出更新后的 JSON

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "planner_common.hpp"
#include "planner_core.hpp"
#include "semantic_io.hpp"

namespace pctplanner {

class PlannerSemanticNode : public rclcpp::Node {
public:
  PlannerSemanticNode()
      : rclcpp::Node("pct_planner_semantic"),
        current_edge_idx_(0) {
    // 配置规划器
    PlannerCore::Config config;
    config.use_quintic = declare_parameter<bool>("use_quintic", true);
    config.max_heading_rate = declare_parameter<double>("max_heading_rate", 10.0);
    config.z_offset = declare_parameter<double>("z_offset", 1.0);
    planner_ = std::make_unique<PlannerCore>(config);

    tomogram_path_ = declare_parameter<std::string>(
        "tomogram_path", "../../rsc/tomogram/scene_map.bin");
    input_json_ = declare_parameter<std::string>(
        "input_json", "../../rsc/semantic_topology/nyby_b5f3.json");
    output_json_ = declare_parameter<std::string>(
        "output_json", "../../rsc/semantic_topology/nyby_b5f3_updated.json");

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/pct_path3", 10);

    // 定时器：1 Hz 循环发布不同边的轨迹
    publish_timer_ = create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&PlannerSemanticNode::PublishLoop, this));

    // 初始化
    Initialize();
  }

private:
  void Initialize() {
    RCLCPP_INFO(get_logger(), "Loading tomogram from: %s", tomogram_path_.c_str());

    // 加载 tomogram
    if (!planner_->LoadTomogramFromFile(tomogram_path_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load tomogram: %s",
                   planner_->GetLastError().c_str());
      return;
    }

    const auto& input = planner_->GetInput();
    RCLCPP_INFO(get_logger(),
                "Tomogram parsed: n_slice=%u dim=%ux%u res=%.3f center=[%.2f,%.2f]",
                input.n_slice, input.dim_x, input.dim_y, input.resolution,
                input.center_x, input.center_y);

    // 加载语义地图
    RCLCPP_INFO(get_logger(), "Loading semantic graph from: %s", input_json_.c_str());
    try {
      semantic_graph_ = load_semantic_json(input_json_);
    } catch (const std::exception& e) {
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

    int success_count = 0;
    for (size_t edge_idx = 0; edge_idx < semantic_graph_.edges.size(); ++edge_idx) {
      Edge& edge = semantic_graph_.edges[edge_idx];
      edge.clear();

      Eigen::Vector3f start_pos = edge.start_pos;
      Eigen::Vector3f end_pos = edge.end_pos;

      RCLCPP_INFO(get_logger(),
                  "Planning edge %zu: [%.2f,%.2f,%.2f] -> [%.2f,%.2f,%.2f]",
                  edge_idx, start_pos.x(), start_pos.y(), start_pos.z(),
                  end_pos.x(), end_pos.y(), end_pos.z());

      // 执行规划
      Eigen::Vector3d start_d(start_pos.x(), start_pos.y(), start_pos.z());
      Eigen::Vector3d end_d(end_pos.x(), end_pos.y(), end_pos.z());
      auto result = planner_->Plan(start_d, end_d);

      if (!result.success) {
        RCLCPP_WARN(get_logger(), "Edge %zu: %s", edge_idx, result.error_msg.c_str());
        continue;
      }

      // 保存轨迹到边
      for (const auto& p : result.trajectory) {
        edge.append_point(p.x(), p.y(), p.z());
      }

      RCLCPP_INFO(get_logger(), "Edge %zu: planned %d waypoints",
                  edge_idx, edge.trajectory.num_waypoints);
      success_count++;
    }

    RCLCPP_INFO(get_logger(), "Planning complete: %d/%zu edges successful",
                success_count, semantic_graph_.edges.size());

    // 写入更新后的 JSON
    try {
      write_semantic_json(semantic_graph_, output_json_);
      RCLCPP_INFO(get_logger(), "Wrote updated semantic graph to: %s",
                  output_json_.c_str());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to write output JSON: %s", e.what());
    }
  }

  void PublishLoop() {
    if (semantic_graph_.edges.empty())
      return;

    current_edge_idx_ = (current_edge_idx_ + 1) % semantic_graph_.edges.size();
    nav_msgs::msg::Path path_msg = semantic_graph_.edges[current_edge_idx_].traj2ros();
    path_msg.header.stamp = now();
    path_pub_->publish(path_msg);
  }

  std::unique_ptr<PlannerCore> planner_;
  std::string tomogram_path_;
  std::string input_json_;
  std::string output_json_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  SemanticMap semantic_graph_;
  size_t current_edge_idx_;
};

}  // namespace pctplanner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerSemanticNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
