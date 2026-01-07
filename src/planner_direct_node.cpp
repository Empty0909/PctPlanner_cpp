// 文件作用：直接从磁盘加载 tomogram（Binary），监听
// /start_pos、/end_pos，规划路径发布 /pct_path2，并周期性重发，带 ASCII PCD
// 输出。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>

#include "ele_planner/offline_ele_planner.h"
#include "planner_common.hpp"
#include "tomogram_format.hpp"

namespace pctplanner {

class PlannerDirectNode : public rclcpp::Node {
public:
  PlannerDirectNode()
      : rclcpp::Node("pct_planner_direct_cpp"), has_start_(false),
        has_end_(false),
        use_quintic_(declare_parameter<bool>("use_quintic", true)),
        max_heading_rate_(declare_parameter<double>("max_heading_rate", 10.0)),
        pcd_path_(declare_parameter<std::string>("pcd_path",
                                                 "trajectory_direct.pcd")),
        tomo_path_(declare_parameter<std::string>(
            "tomo_path", "../../rsc/tomogram/scene_map.bin")) {

    // 离线模式：启动即读 tomogram 文件，等待起终点后规划
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/pct_path2", 10);
    start_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/start_pos", 10,
        std::bind(&PlannerDirectNode::OnStart, this, std::placeholders::_1));

    end_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/end_pos", 10,
        std::bind(&PlannerDirectNode::OnEnd, this, std::placeholders::_1));

    // 周期重发路径，方便 RViz 持续可见
    timer_ = create_wall_timer(std::chrono::milliseconds(500),
                               std::bind(&PlannerDirectNode::OnTimer, this));

    if (!LoadTomogram(tomo_path_, tomogram_buffer_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load tomogram file: %s",
                   tomo_path_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Loaded tomogram file: %s (%zu bytes)",
                  tomo_path_.c_str(), tomogram_buffer_.size());
    }
  }

private:
  // 启动即读取 tomogram 文件到内存，便于离线规划重复使用。
  bool LoadTomogram(const std::string &path, std::vector<uint8_t> &buffer) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
      return false;
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    buffer.resize(static_cast<size_t>(size));
    ifs.read(reinterpret_cast<char *>(buffer.data()), size);
    return true;
  }

  void OnStart(const geometry_msgs::msg::Point::SharedPtr msg) {
    start_ = *msg;
    has_start_ = true;
    RCLCPP_INFO(get_logger(), "Received start_pos: [%.3f, %.3f, %.3f]",
                start_.x, start_.y, start_.z);
    MaybePlan();
  }

  void OnEnd(const geometry_msgs::msg::Point::SharedPtr msg) {
    end_ = *msg;
    has_end_ = true;
    RCLCPP_INFO(get_logger(), "Received end_pos: [%.3f, %.3f, %.3f]", end_.x,
                end_.y, end_.z);
    MaybePlan();
  }

  // 如果起终点和 tomogram 都准备好，运行规划。
  void MaybePlan() {
    if (!(has_start_ && has_end_ && !tomogram_buffer_.empty()))
      return;

    PlannerInput input;
    try {
      ParseTomogram(tomogram_buffer_, input);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Tomogram parse failed: %s", e.what());
      return;
    }

    geometry_msgs::msg::Point start_shifted = start_;
    geometry_msgs::msg::Point end_shifted = end_;
    start_shifted.z += 0.5;
    end_shifted.z += 0.5;

    Eigen::Vector3i start_idx, goal_idx;
    if (!PosToIdx(start_shifted, input, start_idx) ||
        !PosToIdx(end_shifted, input, goal_idx)) {
      RCLCPP_ERROR(get_logger(), "Start or goal out of map bounds");
      return;
    }

    // 直接用文件中的 tomogram 初始化规划器
    OfflineElePlanner planner(max_heading_rate_, use_quintic_);
    Eigen::MatrixXd gateway_d = input.gateway.cast<double>();
    planner.InitMap(20.0, 15.0, input.resolution, input.n_slice, 0.2,
                    input.trav, input.elev_g, input.elev_c, gateway_d,
                    input.trav_gy, -input.trav_gx);

    Eigen::Ref<const Eigen::Vector3i> start_ref(start_idx);
    Eigen::Ref<const Eigen::Vector3i> goal_ref(goal_idx);
    if (!planner.Plan(start_ref, goal_ref, true)) {
      RCLCPP_WARN(get_logger(), "Planner failed to find path");
      return;
    }

    Eigen::MatrixXd traj_raw;
    Eigen::VectorXd heights;
    // 轨迹优化：可选五次多项式
    if (use_quintic_) {
      const auto &optimizer = planner.get_trajectory_optimizer_wnoj();
      traj_raw = optimizer.GetResultMatrix();
      heights = optimizer.GetResultHeight();
    } else {
      const auto &optimizer = planner.get_trajectory_optimizer();
      traj_raw = optimizer.GetResultMatrix();
      heights = optimizer.GetResultHeight();
    }

    std::vector<Eigen::Vector3d> traj;
    traj.reserve(static_cast<size_t>(traj_raw.rows()));
    // 轨迹矩阵 -> 三维点，并转换到 map 坐标；y 列统一按 cols/2 获取，兼容
    // WNOJ/WNOA。
    const int y_idx = static_cast<int>(traj_raw.cols() / 2);
    for (int i = 0; i < traj_raw.rows(); ++i) {
      double x = traj_raw(i, 0);
      double y = traj_raw(i, y_idx);
      // 高度已是米制，保持原值。
      double z = heights(i);
      traj.emplace_back(x, y, z);
    }
    for (auto &p : traj) {
      GridToMap(input, p);
    }

    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "map";
    path_msg.header.stamp = now();
    for (const auto &p : traj) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path_msg.header;
      pose.pose.position.x = p.x();
      pose.pose.position.y = p.y();
      pose.pose.position.z = p.z();
      pose.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose);
    }
    last_path_ = path_msg;
    current_traj_.swap(traj);
    path_pub_->publish(path_msg);
    if (!WriteAsciiPCD(pcd_path_, current_traj_)) {
      RCLCPP_WARN(get_logger(), "Failed to write PCD to %s", pcd_path_.c_str());
    }
    RCLCPP_INFO(get_logger(), "Published path with %zu poses",
                path_msg.poses.size());
  }

  void OnTimer() {
    if (last_path_.poses.empty())
      return;
    last_path_.header.stamp = now();
    for (auto &pose : last_path_.poses) {
      pose.header.stamp = last_path_.header.stamp;
    }
    path_pub_->publish(last_path_);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr end_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<uint8_t> tomogram_buffer_;
  geometry_msgs::msg::Point start_{};
  geometry_msgs::msg::Point end_{};
  bool has_start_;
  bool has_end_;
  bool use_quintic_;
  double max_heading_rate_;
  std::string pcd_path_;
  std::string tomo_path_;

  std::vector<Eigen::Vector3d> current_traj_;
  nav_msgs::msg::Path last_path_;
};

} // namespace pctplanner

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerDirectNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
