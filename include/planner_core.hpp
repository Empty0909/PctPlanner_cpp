#pragma once

// 规划核心模块：封装 OfflineElePlanner 的统一接口
// 供 planner_node、planner_direct_node、planner_semantic_node 共用

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ele_planner/offline_ele_planner.h"
#include "planner_common.hpp"
#include "tomogram_format.hpp"

namespace pctplanner {

// 规划结果
struct PlanResult {
  bool success = false;
  std::string error_msg;
  std::vector<Eigen::Vector3d> trajectory; // 世界坐标系下的轨迹点
};

// 规划器核心类
class PlannerCore {
public:
  struct Config {
    double max_heading_rate;
    bool use_quintic;
    double z_offset; // 起终点 z 偏移

    Config() : max_heading_rate(10.0), use_quintic(true), z_offset(0.5) {}
  };

  explicit PlannerCore(const Config &config = Config())
      : config_(config), initialized_(false) {}

  // 从文件加载 tomogram
  bool LoadTomogramFromFile(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
      last_error_ = "Failed to open file: " + path;
      return false;
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    tomogram_buffer_.resize(static_cast<size_t>(size));
    ifs.read(reinterpret_cast<char *>(tomogram_buffer_.data()), size);
    return InitFromBuffer();
  }

  // 从内存 buffer 加载 tomogram
  bool SetTomogramBuffer(const std::vector<uint8_t> &buffer) {
    tomogram_buffer_ = buffer;
    return InitFromBuffer();
  }

  // 执行规划
  PlanResult Plan(const geometry_msgs::msg::Point &start,
                  const geometry_msgs::msg::Point &end) {
    PlanResult result;

    if (!initialized_) {
      result.error_msg = "Planner not initialized";
      return result;
    }

    // 添加 z 偏移
    geometry_msgs::msg::Point start_shifted = start;
    geometry_msgs::msg::Point end_shifted = end;
    start_shifted.z += config_.z_offset;
    end_shifted.z += config_.z_offset;

    // 转换到栅格索引
    Eigen::Vector3i start_idx, goal_idx;
    if (!PosToIdx(start_shifted, input_, start_idx)) {
      result.error_msg = "Start position out of map bounds";
      return result;
    }
    if (!PosToIdx(end_shifted, input_, goal_idx)) {
      result.error_msg = "Goal position out of map bounds";
      return result;
    }

    // 调试日志：输出栅格索引和代价值
    // 矩阵布局: (n_slice * dim_x, dim_y)，row = layer * dim_x + x, col = y
    {
      const size_t s_row =
          static_cast<size_t>(start_idx[0]) * input_.dim_x + start_idx[2];
      const size_t g_row =
          static_cast<size_t>(goal_idx[0]) * input_.dim_x + goal_idx[2];
      double s_trav = input_.trav(s_row, start_idx[1]);
      double g_trav = input_.trav(g_row, goal_idx[1]);
      double s_elev_g = input_.elev_g(s_row, start_idx[1]);
      double g_elev_g = input_.elev_g(g_row, goal_idx[1]);
      double s_elev_c = input_.elev_c(s_row, start_idx[1]);
      double g_elev_c = input_.elev_c(g_row, goal_idx[1]);
      std::cerr << "[PlannerCore] Start: world(" << start.x << ", " << start.y
                << ", " << start.z << ") -> shifted_z=" << start_shifted.z
                << " -> grid(layer=" << start_idx[0] << ", y=" << start_idx[1]
                << ", x=" << start_idx[2] << "), trav=" << s_trav
                << ", elev_g=" << s_elev_g << ", elev_c=" << s_elev_c
                << std::endl;
      std::cerr << "[PlannerCore] Goal:  world(" << end.x << ", " << end.y
                << ", " << end.z << ") -> shifted_z=" << end_shifted.z
                << " -> grid(layer=" << goal_idx[0] << ", y=" << goal_idx[1]
                << ", x=" << goal_idx[2] << "), trav=" << g_trav
                << ", elev_g=" << g_elev_g << ", elev_c=" << g_elev_c
                << std::endl;
      std::cerr << "[PlannerCore] Matrix dims: trav(" << input_.trav.rows()
                << "x" << input_.trav.cols() << "), n_slice=" << input_.n_slice
                << ", dim_x=" << input_.dim_x << ", dim_y=" << input_.dim_y
                << std::endl;
    }

    // 创建规划器并初始化地图
    OfflineElePlanner planner(config_.max_heading_rate, config_.use_quintic);
    Eigen::MatrixXd gateway_d = input_.gateway.cast<double>();
    planner.InitMap(20.0, 15.0, input_.resolution, input_.n_slice, 0.2,
                    input_.trav, input_.elev_g, input_.elev_c, gateway_d,
                    input_.trav_gy, -input_.trav_gx);

    // 执行规划
    Eigen::Ref<const Eigen::Vector3i> start_ref(start_idx);
    Eigen::Ref<const Eigen::Vector3i> goal_ref(goal_idx);
    if (!planner.Plan(start_ref, goal_ref, true)) {
      result.error_msg = "Planner failed to find path";
      return result;
    }

    // 获取轨迹
    Eigen::MatrixXd traj_raw;
    Eigen::VectorXd heights;
    if (config_.use_quintic) {
      const auto &optimizer = planner.get_trajectory_optimizer_wnoj();
      traj_raw = optimizer.GetResultMatrix();
      heights = optimizer.GetResultHeight();
    } else {
      const auto &optimizer = planner.get_trajectory_optimizer();
      traj_raw = optimizer.GetResultMatrix();
      heights = optimizer.GetResultHeight();
    }

    // 转换轨迹到世界坐标
    result.trajectory.reserve(static_cast<size_t>(traj_raw.rows()));
    const int y_idx = static_cast<int>(traj_raw.cols() / 2);
    for (int i = 0; i < traj_raw.rows(); ++i) {
      Eigen::Vector3d p(traj_raw(i, 0), traj_raw(i, y_idx), heights(i));
      GridToMap(input_, p);
      result.trajectory.push_back(p);
    }

    result.success = true;
    return result;
  }

  // 使用 Eigen::Vector3d 接口（便于语义规划节点使用）
  PlanResult Plan(const Eigen::Vector3d &start, const Eigen::Vector3d &end) {
    geometry_msgs::msg::Point start_pt, end_pt;
    start_pt.x = start.x();
    start_pt.y = start.y();
    start_pt.z = start.z();
    end_pt.x = end.x();
    end_pt.y = end.y();
    end_pt.z = end.z();
    return Plan(start_pt, end_pt);
  }

  // 轨迹转 ROS Path 消息
  static nav_msgs::msg::Path
  TrajectoryToPath(const std::vector<Eigen::Vector3d> &trajectory,
                   const rclcpp::Time &stamp,
                   const std::string &frame_id = "map") {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id;
    path.header.stamp = stamp;

    for (const auto &p : trajectory) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = p.x();
      pose.pose.position.y = p.y();
      pose.pose.position.z = p.z();
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    return path;
  }

  // 获取 PlannerInput（用于其他需要访问地图参数的场景）
  const PlannerInput &GetInput() const { return input_; }

  // 获取 tomogram buffer（用于可视化）
  const std::vector<uint8_t> &GetTomogramBuffer() const {
    return tomogram_buffer_;
  }

  bool IsInitialized() const { return initialized_; }
  const std::string &GetLastError() const { return last_error_; }

private:
  bool InitFromBuffer() {
    try {
      ParseTomogram(tomogram_buffer_, input_);
      initialized_ = true;
      return true;
    } catch (const std::exception &e) {
      last_error_ = std::string("Tomogram parse failed: ") + e.what();
      initialized_ = false;
      return false;
    }
  }

  Config config_;
  bool initialized_;
  std::string last_error_;
  std::vector<uint8_t> tomogram_buffer_;
  PlannerInput input_;
};

} // namespace pctplanner
