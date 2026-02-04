// 文件作用：直接从磁盘加载 tomogram（Binary），监听
// /start_pos、/end_pos，规划路径发布 /pct_path2，并周期性重发，带 ASCII PCD 输出。
// 同时发布 /tomogram 点云用于 RViz 可视化。

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "planner_common.hpp"
#include "planner_core.hpp"
#include "tomogram_viz.hpp"

namespace pctplanner {

class PlannerDirectNode : public rclcpp::Node {
public:
  PlannerDirectNode()
      : rclcpp::Node("pct_planner_direct_cpp"),
        has_start_(false),
        has_end_(false) {
    // 配置规划器
    PlannerCore::Config config;
    config.use_quintic = declare_parameter<bool>("use_quintic", true);
    config.max_heading_rate = declare_parameter<double>("max_heading_rate", 10.0);
    config.z_offset = 0.5;
    planner_ = std::make_unique<PlannerCore>(config);

    pcd_path_ = declare_parameter<std::string>("pcd_path", "trajectory_direct.pcd");
    tomo_path_ = declare_parameter<std::string>("tomo_path", "../../rsc/tomogram/scene_map.bin");

    // 话题发布
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/pct_path2", 10);
    path_pub_compat_ = create_publisher<nav_msgs::msg::Path>("/pct_path", 10);
    tomo_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/tomogram", 10);

    // 话题订阅
    start_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/start_pos", 10,
        std::bind(&PlannerDirectNode::OnStart, this, std::placeholders::_1));
    end_sub_ = create_subscription<geometry_msgs::msg::Point>(
        "/end_pos", 10,
        std::bind(&PlannerDirectNode::OnEnd, this, std::placeholders::_1));

    // 周期重发定时器
    timer_ = create_wall_timer(std::chrono::milliseconds(500),
                               std::bind(&PlannerDirectNode::OnTimer, this));

    // 加载 tomogram 文件
    if (!planner_->LoadTomogramFromFile(tomo_path_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load tomogram: %s",
                   planner_->GetLastError().c_str());
    } else {
      const auto& input = planner_->GetInput();
      RCLCPP_INFO(get_logger(),
                  "Loaded tomogram: %s (%zu bytes), dim=%ux%u, n_slice=%u",
                  tomo_path_.c_str(), planner_->GetTomogramBuffer().size(),
                  input.dim_x, input.dim_y, input.n_slice);
      
      // 生成可视化点云
      tomo_cloud_msg_ = CreateTomogramCloud(planner_->GetTomogramBuffer());
      tomo_cloud_msg_.header.stamp = now();
      tomo_pub_->publish(tomo_cloud_msg_);
      RCLCPP_INFO(get_logger(), "Published tomogram visualization with %u points",
                  tomo_cloud_msg_.width);
    }
  }

private:
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
    if (!(has_start_ && has_end_ && planner_->IsInitialized()))
      return;

    // 执行规划
    auto result = planner_->Plan(start_, end_);
    if (!result.success) {
      RCLCPP_WARN(get_logger(), "Planning failed: %s", result.error_msg.c_str());
      return;
    }

    // 发布路径
    last_path_ = PlannerCore::TrajectoryToPath(result.trajectory, now());
    path_pub_->publish(last_path_);
    RCLCPP_INFO(get_logger(), "Published path with %zu poses", last_path_.poses.size());

    // 写 PCD
    if (!WriteAsciiPCD(pcd_path_, result.trajectory)) {
      RCLCPP_WARN(get_logger(), "Failed to write PCD to %s", pcd_path_.c_str());
    }
  }

  void OnTimer() {
    // 周期重发路径
    if (!last_path_.poses.empty()) {
      last_path_.header.stamp = now();
      for (auto& pose : last_path_.poses) {
        pose.header.stamp = last_path_.header.stamp;
      }
      path_pub_->publish(last_path_);
      path_pub_compat_->publish(last_path_);
    }

    // 周期重发 tomogram 可视化
    if (tomo_cloud_msg_.data.size() > 0) {
      tomo_cloud_msg_.header.stamp = now();
      tomo_pub_->publish(tomo_cloud_msg_);
    }
  }

  std::unique_ptr<PlannerCore> planner_;
  std::string pcd_path_;
  std::string tomo_path_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_compat_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tomo_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr end_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::Point start_{};
  geometry_msgs::msg::Point end_{};
  bool has_start_;
  bool has_end_;
  nav_msgs::msg::Path last_path_;
  sensor_msgs::msg::PointCloud2 tomo_cloud_msg_;
};

}  // namespace pctplanner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerDirectNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
