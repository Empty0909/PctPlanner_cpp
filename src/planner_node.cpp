// 文件作用：订阅 /tomogram_data、/start_pos、/end_pos，解析 tomogram 后调用 C++
// 核心规划器生成路径并发布 /pct_path 与 ASCII PCD。
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

class PlannerNode : public rclcpp::Node {
public:
  PlannerNode()
      : rclcpp::Node("pct_planner_cpp"), has_tomogram_(false),
        has_start_(false), has_end_(false),
        use_quintic_(declare_parameter<bool>("use_quintic", true)),
        max_heading_rate_(declare_parameter<double>("max_heading_rate", 10.0)),
        pcd_path_(
            declare_parameter<std::string>("pcd_path", "trajectory.pcd")) {
    // 启动订阅：tomogram、起点、终点到齐后才触发规划
    tomogram_sub_ = create_subscription<std_msgs::msg::ByteMultiArray>(
        "/tomogram_data", 10,
        std::bind(&PlannerNode::OnTomogram, this, std::placeholders::_1));
    start_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/start_pos", 10,
        std::bind(&PlannerNode::OnStart, this, std::placeholders::_1));
    end_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/end_pos", 10,
        std::bind(&PlannerNode::OnEnd, this, std::placeholders::_1));
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/pct_path", 10);

    // 定时器：每 2 秒重发一次路径（如果有）
    republish_timer_ = create_wall_timer(
        std::chrono::seconds(2), std::bind(&PlannerNode::RepublishPath, this));

    RCLCPP_INFO(get_logger(), "Planner C++ node ready");
  }

private:
  void RepublishPath() {
    if (last_path_msg_.poses.empty())
      return;
    last_path_msg_.header.stamp = now();
    path_pub_->publish(last_path_msg_);
  }

  void OnTomogram(const std_msgs::msg::ByteMultiArray::SharedPtr msg) {
    tomogram_buffer_ = msg->data;
    has_tomogram_ = true;
    RCLCPP_INFO(get_logger(), "Received tomogram_data (%zu bytes)",
                tomogram_buffer_.size());
    MaybePlan();
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

  // 如果 tomogram、起点、终点都已到达，则解析地图并运行规划。
  void MaybePlan() {
    if (!(has_tomogram_ && has_start_ && has_end_))
      return;

    PlannerInput input;
    try {
      ParseTomogram(tomogram_buffer_, input);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Tomogram parse failed: %s", e.what());
      return;
    }

    RCLCPP_INFO(
        get_logger(),
        "Tomogram parsed: n_slice=%u dim=%ux%u res=%.3f center=[%.2f,%.2f]",
        input.n_slice, input.dim_x, input.dim_y, input.resolution,
        input.center_x, input.center_y);

    geometry_msgs::msg::Point start_shifted = start_;
    geometry_msgs::msg::Point end_shifted = end_;
    start_shifted.z += 0.5;
    end_shifted.z += 0.5;

    Eigen::Vector3i start_idx, goal_idx;
    if (!PosToIdx(start_shifted, input, start_idx) ||
        !PosToIdx(end_shifted, input, goal_idx)) {
      RCLCPP_ERROR(get_logger(),
                   "Start or goal out of map bounds. Start=[%.2f,%.2f,%.2f], "
                   "End=[%.2f,%.2f,%.2f]",
                   start_.x, start_.y, start_.z, end_.x, end_.y, end_.z);
      return;
    }

    RCLCPP_INFO(get_logger(), "Start idx: [%d,%d,%d], Goal idx: [%d,%d,%d]",
                start_idx.x(), start_idx.y(), start_idx.z(), goal_idx.x(),
                goal_idx.y(), goal_idx.z());

    // 创建规划器并喂入地图（含梯度/网关），梯度符号和 Python 保持一致
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

    const auto &path_finder = planner.get_path_finder();
    Eigen::MatrixXd path = path_finder.GetResultMatrix();

    Eigen::MatrixXd traj_raw;
    Eigen::VectorXd heights;
    // 轨迹优化：选择五次多项式或原始优化器
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
    // WNOJ/WNOA 均是 [x, vx, ax?, y, vy, ay?] 或 [x, vx, y, vy]，y 列可用
    // cols/2 抓取。
    const int y_idx = static_cast<int>(traj_raw.cols() / 2);
    for (int i = 0; i < traj_raw.rows(); ++i) {
      double x = traj_raw(i, 0);
      double y = traj_raw(i, y_idx);
      // 优化器输出已在米制（输入 elev_g/elev_c 为米），无需按分辨率缩放。
      double z = heights(i);
      traj.emplace_back(x, y, z);
    }
    // 转换到地图坐标（中心在 map 原点）
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
    path_pub_->publish(path_msg);
    last_path_msg_ = path_msg; // 保存用于重发
    RCLCPP_INFO(get_logger(), "Published path with %zu poses",
                path_msg.poses.size());

    if (!WriteAsciiPCD(pcd_path_, traj)) {
      RCLCPP_WARN(get_logger(), "Failed to write PCD to %s", pcd_path_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Wrote ASCII PCD to %s", pcd_path_.c_str());
    }

    has_start_ = false;
    has_end_ = false;
  }

  rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr tomogram_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr end_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  std::vector<uint8_t> tomogram_buffer_;
  geometry_msgs::msg::Point start_{};
  geometry_msgs::msg::Point end_{};
  bool has_tomogram_;
  bool has_start_;
  bool has_end_;
  bool use_quintic_;
  double max_heading_rate_;
  std::string pcd_path_;

  rclcpp::TimerBase::SharedPtr republish_timer_;
  nav_msgs::msg::Path last_path_msg_;
};

} // namespace pctplanner

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
