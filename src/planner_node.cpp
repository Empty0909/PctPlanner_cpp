// 文件作用：订阅 /tomogram_data、/start_pos、/end_pos，解析 tomogram 后调用
// PlannerCore 生成路径并发布 /pct_path 与 ASCII PCD。

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>

#include "planner_common.hpp"
#include "planner_core.hpp"

namespace pctplanner {

class PlannerNode : public rclcpp::Node {
public:
  PlannerNode()
      : rclcpp::Node("pct_planner_cpp"),
        has_tomogram_(false),
        has_start_(false),
        has_end_(false) {
    // 配置规划器
    PlannerCore::Config config;
    config.use_quintic = declare_parameter<bool>("use_quintic", true);
    config.max_heading_rate = declare_parameter<double>("max_heading_rate", 10.0);
    config.z_offset = 0.5;
    planner_ = std::make_unique<PlannerCore>(config);

    pcd_path_ = declare_parameter<std::string>("pcd_path", "trajectory.pcd");

    // 话题订阅/发布
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

    // 定时器：每 2 秒重发一次路径
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
    if (!planner_->SetTomogramBuffer(msg->data)) {
      RCLCPP_ERROR(get_logger(), "%s", planner_->GetLastError().c_str());
      return;
    }
    has_tomogram_ = true;
    RCLCPP_INFO(get_logger(), "Received tomogram_data (%zu bytes)", msg->data.size());
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
    RCLCPP_INFO(get_logger(), "Received end_pos: [%.3f, %.3f, %.3f]",
                end_.x, end_.y, end_.z);
    MaybePlan();
  }

  void MaybePlan() {
    if (!(has_tomogram_ && has_start_ && has_end_))
      return;

    const auto& input = planner_->GetInput();
    RCLCPP_INFO(get_logger(),
                "Tomogram: n_slice=%u dim=%ux%u res=%.3f center=[%.2f,%.2f]",
                input.n_slice, input.dim_x, input.dim_y, input.resolution,
                input.center_x, input.center_y);

    // 执行规划
    auto result = planner_->Plan(start_, end_);
    if (!result.success) {
      RCLCPP_WARN(get_logger(), "Planning failed: %s", result.error_msg.c_str());
      return;
    }

    // 发布路径
    last_path_msg_ = PlannerCore::TrajectoryToPath(result.trajectory, now());
    path_pub_->publish(last_path_msg_);
    RCLCPP_INFO(get_logger(), "Published path with %zu poses",
                last_path_msg_.poses.size());

    // 写 PCD
    if (!WriteAsciiPCD(pcd_path_, result.trajectory)) {
      RCLCPP_WARN(get_logger(), "Failed to write PCD to %s", pcd_path_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Wrote ASCII PCD to %s", pcd_path_.c_str());
    }

    has_start_ = false;
    has_end_ = false;
  }

  std::unique_ptr<PlannerCore> planner_;
  std::string pcd_path_;

  rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr tomogram_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr end_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr republish_timer_;

  geometry_msgs::msg::Point start_{};
  geometry_msgs::msg::Point end_{};
  bool has_tomogram_;
  bool has_start_;
  bool has_end_;
  nav_msgs::msg::Path last_path_msg_;
};

}  // namespace pctplanner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
