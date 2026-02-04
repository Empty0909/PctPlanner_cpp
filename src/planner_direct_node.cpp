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
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
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
    // 同时发布到 /pct_path 以兼容 RViz 配置
    path_pub_compat_ = create_publisher<nav_msgs::msg::Path>("/pct_path", 10);
    // 发布 tomogram 可视化点云
    tomo_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/tomogram", 10);

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
      // 解析并发布 tomogram 可视化
      PublishTomogramVisualization();
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
    path_pub_compat_->publish(last_path_);

    // 周期性重发 tomogram 可视化
    if (tomo_cloud_msg_.data.size() > 0) {
      tomo_cloud_msg_.header.stamp = now();
      tomo_pub_->publish(tomo_cloud_msg_);
    }
  }

  // 解析 tomogram 并发布可视化点云 - 与 tomography_node::PublishSurfaceCloud
  // 完全一致
  void PublishTomogramVisualization() {
    if (tomogram_buffer_.empty())
      return;

    // 直接从二进制解析，不使用 PlannerInput（其矩阵布局不同）
    auto view = tomogram_format::Deserialize(tomogram_buffer_);
    const auto &h = view.header;
    const auto mode = tomogram_format::GetPrecisionMode(h);
    const size_t scalar_bytes = tomogram_format::ScalarBytes(mode);

    const uint32_t n_slice = h.n_slice;
    const uint32_t dim_x = h.dim_x;
    const uint32_t dim_y = h.dim_y;
    const float resolution = h.resolution;
    const float cx = h.center_x;
    const float cy = h.center_y;
    const float slice_dh = h.slice_dh;

    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const size_t layer_stride =
        static_cast<size_t>(n_slice) * plane * scalar_bytes;

    // 读取 trav 和 elev_g 原始数组
    std::vector<float> trav(n_slice * plane);
    std::vector<float> elev_g(n_slice * plane);

    for (uint32_t s = 0; s < n_slice; ++s) {
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y) {
          // 文件布局: idx = s * plane + x * dim_y + y
          const size_t idx_plane = static_cast<size_t>(s) * plane +
                                   static_cast<size_t>(x) * dim_y + y;
          const size_t base_offset = idx_plane * scalar_bytes;

          // trav 在 layer 0, elev_g 在 layer 3
          float t_val, eg_val;
          if (mode == tomogram_format::PrecisionMode::FLOAT32) {
            std::memcpy(&t_val, view.data + base_offset + 0 * layer_stride,
                        sizeof(float));
            std::memcpy(&eg_val, view.data + base_offset + 3 * layer_stride,
                        sizeof(float));
          } else {
            uint16_t t_bits, eg_bits;
            std::memcpy(&t_bits, view.data + base_offset + 0 * layer_stride,
                        sizeof(uint16_t));
            std::memcpy(&eg_bits, view.data + base_offset + 3 * layer_stride,
                        sizeof(uint16_t));
            Eigen::half t_h, eg_h;
            std::memcpy(&t_h, &t_bits, sizeof(uint16_t));
            std::memcpy(&eg_h, &eg_bits, sizeof(uint16_t));
            t_val = static_cast<float>(t_h);
            eg_val = static_cast<float>(eg_h);
          }

          const size_t arr_idx = static_cast<size_t>(s) * plane +
                                 static_cast<size_t>(x) * dim_y + y;
          trav[arr_idx] = t_val;
          elev_g[arr_idx] = eg_val;
        }
      }
    }

    // 与 tomography_node::PublishSurfaceCloud 完全一致的遮挡处理
    std::vector<std::vector<float>> vis_g(n_slice, std::vector<float>(plane));
    std::vector<std::vector<float>> vis_t(n_slice, std::vector<float>(plane));

    for (uint32_t s = 0; s < n_slice; ++s) {
      std::copy(elev_g.begin() + static_cast<size_t>(s) * plane,
                elev_g.begin() + static_cast<size_t>(s + 1) * plane,
                vis_g[s].begin());
      std::copy(trav.begin() + static_cast<size_t>(s) * plane,
                trav.begin() + static_cast<size_t>(s + 1) * plane,
                vis_t[s].begin());
    }

    // 遮挡处理：下层被上层遮挡时设为NaN
    for (uint32_t s = 0; s + 1 < n_slice; ++s) {
      for (size_t idx = 0; idx < plane; ++idx) {
        const float dh = vis_g[s + 1][idx] - vis_g[s][idx];
        if (dh < slice_dh) {
          vis_g[s][idx] = std::numeric_limits<float>::quiet_NaN();
          vis_t[s + 1][idx] = std::min(vis_t[s][idx], vis_t[s + 1][idx]);
        }
      }
    }

    // 收集可视化点
    std::vector<float> buffer;
    buffer.reserve(n_slice * plane * 4);

    const float ox = static_cast<float>(dim_x) / 2.0f;
    const float oy = static_cast<float>(dim_y) / 2.0f;

    for (uint32_t s = 0; s < n_slice; ++s) {
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y) {
          const size_t idx = static_cast<size_t>(x) * dim_y + y;

          // 被遮挡的跳过
          if (std::isnan(vis_g[s][idx]))
            continue;

          // elev_g 为 NaN 或异常值跳过
          if (std::isnan(elev_g[s * plane + idx]) || vis_g[s][idx] < -90.0f)
            continue;

          const float wx = (static_cast<float>(x) - ox) * resolution + cx;
          const float wy = (static_cast<float>(y) - oy) * resolution + cy;
          const float wz = vis_g[s][idx];
          const float inten = std::min(vis_t[s][idx], 50.0f);

          buffer.push_back(wx);
          buffer.push_back(wy);
          buffer.push_back(wz);
          buffer.push_back(inten);
        }
      }
    }

    if (buffer.empty())
      return;

    // 创建 PointCloud2 消息
    tomo_cloud_msg_.header.frame_id = "map";
    tomo_cloud_msg_.header.stamp = now();
    tomo_cloud_msg_.height = 1;
    tomo_cloud_msg_.width = static_cast<uint32_t>(buffer.size() / 4);
    tomo_cloud_msg_.is_bigendian = false;
    tomo_cloud_msg_.is_dense = false;
    tomo_cloud_msg_.point_step = sizeof(float) * 4;
    tomo_cloud_msg_.row_step =
        tomo_cloud_msg_.point_step * tomo_cloud_msg_.width;

    tomo_cloud_msg_.fields.resize(4);
    tomo_cloud_msg_.fields[0].name = "x";
    tomo_cloud_msg_.fields[0].offset = 0;
    tomo_cloud_msg_.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    tomo_cloud_msg_.fields[0].count = 1;
    tomo_cloud_msg_.fields[1].name = "y";
    tomo_cloud_msg_.fields[1].offset = 4;
    tomo_cloud_msg_.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    tomo_cloud_msg_.fields[1].count = 1;
    tomo_cloud_msg_.fields[2].name = "z";
    tomo_cloud_msg_.fields[2].offset = 8;
    tomo_cloud_msg_.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    tomo_cloud_msg_.fields[2].count = 1;
    tomo_cloud_msg_.fields[3].name = "intensity";
    tomo_cloud_msg_.fields[3].offset = 12;
    tomo_cloud_msg_.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    tomo_cloud_msg_.fields[3].count = 1;

    tomo_cloud_msg_.data.resize(tomo_cloud_msg_.row_step);

    // 使用迭代器填充数据
    sensor_msgs::PointCloud2Iterator<float> iter_x(tomo_cloud_msg_, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(tomo_cloud_msg_, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(tomo_cloud_msg_, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_i(tomo_cloud_msg_,
                                                   "intensity");

    for (size_t i = 0; i < buffer.size();) {
      *iter_x = buffer[i++];
      *iter_y = buffer[i++];
      *iter_z = buffer[i++];
      *iter_i = buffer[i++];
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_i;
    }

    tomo_pub_->publish(tomo_cloud_msg_);
    RCLCPP_INFO(get_logger(), "Published tomogram visualization with %u points",
                tomo_cloud_msg_.width);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_compat_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tomo_pub_;
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
  sensor_msgs::msg::PointCloud2 tomo_cloud_msg_;
};

} // namespace pctplanner

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::PlannerDirectNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
