// 文件作用：订阅 /global_points 点云，GPU（CUDA）生成
// tomogram（膨胀代价/梯度/地面/顶部高程），序列化 float16 发布
// /tomogram_data，并写入磁盘；可视化膨胀后代价、梯度、间隙。
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>

#include "tomogram_format.hpp"
#include "tomography_cuda.hpp"

namespace pctplanner {

class TomographyNode : public rclcpp::Node {
public:
  TomographyNode()
      : rclcpp::Node("pct_tomography_cpp"),
        output_path_(declare_parameter<std::string>(
            "output_path", "../../rsc/tomogram/scene_map.bin")),
        resolution_(declare_parameter<double>("resolution", 0.15)),
        slice_dh_(declare_parameter<double>("slice_dh", 0.5)),
        // 间隙检测参数：与 Python 原版 scene.py 保持一致
        // interval_min: 最小可通行间隙高度
        interval_min_(declare_parameter<double>("interval_min", 0.5)),
        // interval_free: 无惩罚间隙高度阈值
        interval_free_(declare_parameter<double>("interval_free", 0.6)),
        // 坡度参数：slope_max 用于计算 step_stand
        slope_max_(declare_parameter<double>("slope_max", 1)),
        // 越障参数：step_max 控制可跨越的最大高度差
        step_max_(declare_parameter<double>("step_max", 0.7)),
        standable_ratio_(declare_parameter<double>("standable_ratio", 0.4)),
        // cost_barrier 与 Python 原版一致
        cost_barrier_(declare_parameter<double>("cost_barrier", 50.0)),
        // 膨胀参数：与 Python 原版一致
        safe_margin_(declare_parameter<double>("safe_margin", 0.1)),
        inflation_(declare_parameter<double>("inflation", 0.05)),
        kernel_size_(declare_parameter<int>("kernel_size", 5)),
        map_frame_(declare_parameter<std::string>("map_frame", "map")),
        surface_only_(declare_parameter<bool>("surface_only", true)),
        published_(false) {
    tomogram_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/tomogram", 10);
    tomogram_data_pub_ =
        create_publisher<std_msgs::msg::ByteMultiArray>("/tomogram_data", 10);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/global_points", 10,
        std::bind(&TomographyNode::OnCloud, this, std::placeholders::_1));

    // Create a timer that triggers every 1 second (1 Hz)
    // periodic_timer_ = create_wall_timer(
    // std::chrono::milliseconds(200), // Period of 0.2 second
    // std::bind(&TomographyNode::PublishTomogram, this));

    RCLCPP_INFO(get_logger(),
                "Tomography CUDA node ready, listening on /global_points");
  }

private:
  void PublishTomogram() {
    // 可视化：默认发布全量体素；如需仅地面可在参数 surface_only = true
    // if (surface_only_) {
    // PublishSurfaceCloud(n_slice, dim_x, dim_y, cx, cy,
    // static_cast<float>(resolution_), slice_heights_f32,
    // gpu_out.inflated_cost, gpu_out.elev_g, missing_ground,
    // tomogram_pub_);
    // } else {
    // PublishAllCloud(n_slice, dim_x, dim_y, cx, cy,
    // static_cast<float>(resolution_), slice_heights_f32,
    // gpu_out.inflated_cost, tomogram_pub_);
    // }
  }

  void OnCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (published_)
      return; // 仅处理首帧，避免重复计算
    published_ = true;

    std::vector<Eigen::Vector3f> points;
    ExtractXYZ(*msg, points);
    if (points.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty point cloud; skipping");
      return;
    }

    // 计算包围盒与切片参数
    Eigen::Vector3f pmin = points.front();
    Eigen::Vector3f pmax = points.front();
    for (const auto &p : points) {
      pmin = pmin.cwiseMin(p);
      pmax = pmax.cwiseMax(p);
    }
    pmin.z() = std::min(pmin.z(), pmax.z());
    const float slice_h0 = pmin.z() + static_cast<float>(slice_dh_);
    const float slice_dh = static_cast<float>(slice_dh_);

    const uint32_t dim_x =
        static_cast<uint32_t>(std::ceil((pmax.x() - pmin.x()) / resolution_)) +
        4;
    const uint32_t dim_y =
        static_cast<uint32_t>(std::ceil((pmax.y() - pmin.y()) / resolution_)) +
        4;
    const uint32_t n_slice = std::max<uint32_t>(
        1, static_cast<uint32_t>(std::ceil((pmax.z() - pmin.z()) / slice_dh_)));
    const float cx = 0.5f * (pmax.x() + pmin.x());
    const float cy = 0.5f * (pmax.y() + pmin.y());

    TomogramHeader header = tomogram_format::MakeHeader(
        n_slice, dim_x, dim_y, static_cast<float>(resolution_), cx, cy,
        slice_h0, slice_dh);

    // 记录地图范围，便于设置起终点：中心/尺寸/分辨率/切片高度
    RCLCPP_INFO(
        get_logger(),
        "Tomogram bbox min[%.2f %.2f %.2f] max[%.2f %.2f %.2f], center[%.2f "
        "%.2f], res=%.3f, dims=%ux%u, slices=%u, slice_h0=%.3f, dh=%.3f",
        pmin.x(), pmin.y(), pmin.z(), pmax.x(), pmax.y(), pmax.z(), cx, cy,
        static_cast<float>(resolution_), dim_x, dim_y, n_slice, slice_h0,
        slice_dh);

    std::vector<uint16_t> slice_heights_fp16(n_slice);
    std::vector<float> slice_heights_f32(n_slice);
    for (uint32_t i = 0; i < n_slice; ++i) {
      float h = slice_h0 + slice_dh * static_cast<float>(i);
      slice_heights_fp16[i] = Float32ToFp16(h);
      slice_heights_f32[i] = h;
    }

    // 填充核参数，交给 CUDA 生成 trav/elev 信息
    TravParams params{};
    params.resolution = resolution_;
    params.slice_dh = slice_dh_;
    params.interval_min = interval_min_;
    params.interval_free = interval_free_;
    params.slope_max = slope_max_;
    params.step_max = step_max_;
    params.standable_ratio = standable_ratio_;
    params.cost_barrier = cost_barrier_;
    params.safe_margin = safe_margin_;
    params.inflation = inflation_;
    params.kernel_size = kernel_size_;

    GpuTomographyOutput gpu_out;
    if (!RunTomographyCuda(points, n_slice, dim_x, dim_y, cx, cy, slice_h0,
                           params, gpu_out)) {
      RCLCPP_ERROR(get_logger(), "CUDA tomography failed");
      return;
    }

    // 记录缺失地面/顶部的掩码，渲染时可跳过，避免在最低层铺满一片平面
    std::vector<uint8_t> missing_ground(gpu_out.elev_g.size(), 0);
    std::vector<uint8_t> missing_ceiling(gpu_out.elev_c.size(), 0);

    // 底/顶面缺省值兜底用于规划，但掩码保留缺失点用于可视化过滤
    const float default_ground = slice_h0;
    const float default_ceiling =
        slice_h0 + slice_dh * static_cast<float>(n_slice);
    for (size_t i = 0; i < gpu_out.elev_g.size(); ++i) {
      if (gpu_out.elev_g[i] <= -9e5f) {
        missing_ground[i] = 1;
        gpu_out.elev_g[i] = default_ground;
      }
      if (gpu_out.elev_c[i] >= 9e5f) {
        missing_ceiling[i] = 1;
        gpu_out.elev_c[i] = default_ceiling;
      }
    }

    // 缺失地面视为不可行：将代价置为障碍，避免 A* 穿过未观测空洞
    for (size_t i = 0; i < missing_ground.size(); ++i) {
      if (missing_ground[i]) {
        gpu_out.inflated_cost[i] = static_cast<float>(cost_barrier_);
      }
    }

    // 注意：与 Python 原版一致，不进行硬障碍钳位（1e6）
    // A* 规划器会根据代价权重自然避开高代价区域（如陡坡），
    // 同时保留楼梯等跨层通道的可通行性（代价 = cost_barrier 但非无穷大）

    // 生成 trav 梯度（在膨胀代价上取中心差分）
    std::vector<float> trav_gx(gpu_out.inflated_cost.size(), 0.0f);
    std::vector<float> trav_gy(gpu_out.inflated_cost.size(), 0.0f);
    ComputeTravGradient(n_slice, dim_x, dim_y, gpu_out.inflated_cost, trav_gx,
                        trav_gy);

    // 序列化 tomogram（trav/gx/gy/elev_g/elev_c 全体素）
    const size_t voxels_per_layer =
        static_cast<size_t>(n_slice) * dim_x * dim_y;
    const size_t total_voxels =
        voxels_per_layer * tomogram_format::kLayersPerVoxel;
    std::vector<uint16_t> data_fp16(total_voxels, 0);
    FillTomogramPayload(n_slice, dim_x, dim_y, gpu_out.inflated_cost, trav_gx,
                        trav_gy, gpu_out.elev_g, gpu_out.elev_c, data_fp16);

    std::vector<uint8_t> payload =
        tomogram_format::Serialize(header, slice_heights_fp16, data_fp16);

    std_msgs::msg::ByteMultiArray msg_out;
    msg_out.data.assign(payload.begin(), payload.end());
    tomogram_data_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published tomogram_data (%zu bytes)",
                payload.size());

    // 可视化：默认发布全量体素；如需仅地面可在参数 surface_only=true
    if (surface_only_) {
      PublishSurfaceCloud(n_slice, dim_x, dim_y, cx, cy,
                          static_cast<float>(resolution_), slice_heights_f32,
                          gpu_out.inflated_cost, gpu_out.elev_g, missing_ground,
                          tomogram_pub_);
    } else {
      PublishAllCloud(n_slice, dim_x, dim_y, cx, cy,
                      static_cast<float>(resolution_), slice_heights_f32,
                      gpu_out.inflated_cost, tomogram_pub_);
    }

    WriteBinary(output_path_, payload);
    RCLCPP_INFO(get_logger(), "Wrote tomogram to %s", output_path_.c_str());
  }

  void ExtractXYZ(const sensor_msgs::msg::PointCloud2 &cloud,
                  std::vector<Eigen::Vector3f> &out) {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      out.emplace_back(*iter_x, *iter_y, *iter_z);
    }
  }

  static uint16_t Float32ToFp16(float v) {
    Eigen::half h = static_cast<Eigen::half>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
  }

  void ComputeTravGradient(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                           const std::vector<float> &cost,
                           std::vector<float> &gx, std::vector<float> &gy) {
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    for (uint32_t s = 0; s < n_slice; ++s) {
      const size_t offset = static_cast<size_t>(s) * plane;
      for (uint32_t y = 1; y + 1 < dim_y; ++y) {
        for (uint32_t x = 1; x + 1 < dim_x; ++x) {
          const size_t idx = offset + static_cast<size_t>(y) * dim_x + x;
          gx[idx] = cost[idx + 1] - cost[idx - 1];
          gy[idx] = cost[idx + dim_x] - cost[idx - dim_x];
        }
      }
    }
  }

  void FillTomogramPayload(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                           const std::vector<float> &trav,
                           const std::vector<float> &gx,
                           const std::vector<float> &gy,
                           const std::vector<float> &ground,
                           const std::vector<float> &ceiling,
                           std::vector<uint16_t> &data_fp16) {
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const size_t layer_stride = static_cast<size_t>(n_slice) * plane;
    for (uint32_t s = 0; s < n_slice; ++s) {
      const size_t offset = static_cast<size_t>(s) * plane;
      for (uint32_t y = 0; y < dim_y; ++y) {
        for (uint32_t x = 0; x < dim_x; ++x) {
          const size_t idx = offset + static_cast<size_t>(y) * dim_x + x;
          data_fp16[idx + 0 * layer_stride] = Float32ToFp16(trav[idx]);
          data_fp16[idx + 1 * layer_stride] = Float32ToFp16(gx[idx]);
          data_fp16[idx + 2 * layer_stride] = Float32ToFp16(gy[idx]);
          data_fp16[idx + 3 * layer_stride] = Float32ToFp16(ground[idx]);
          data_fp16[idx + 4 * layer_stride] = Float32ToFp16(ceiling[idx]);
        }
      }
    }
  }

  // 体素全量可视化（大点云，通常只用于调试）
  void PublishAllCloud(
      uint32_t n_slice, uint32_t dim_x, uint32_t dim_y, float cx, float cy,
      float resolution, const std::vector<float> &slice_heights,
      const std::vector<float> &trav,
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub) {
    if (!pub)
      return;
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const size_t total = static_cast<size_t>(n_slice) * plane;
    if (trav.size() < total || slice_heights.size() < n_slice)
      return;

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = map_frame_;
    cloud.header.stamp = now();
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(total);
    cloud.is_bigendian = false;
    cloud.is_dense = false;
    cloud.point_step = sizeof(float) * 4;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.fields.resize(4);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;
    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;
    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;
    cloud.fields[3].name = "intensity";
    cloud.fields[3].offset = 12;
    cloud.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[3].count = 1;
    cloud.data.resize(cloud.row_step);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_i(cloud, "intensity");

    const float ox = static_cast<float>(dim_x) / 2.0f;
    const float oy = static_cast<float>(dim_y) / 2.0f;

    for (uint32_t s = 0; s < n_slice; ++s) {
      float hz = slice_heights[s];
      const size_t offset = static_cast<size_t>(s) * plane;
      for (uint32_t y = 0; y < dim_y; ++y) {
        for (uint32_t x = 0; x < dim_x;
             ++x, ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
          *iter_x = (static_cast<float>(x) - ox) * resolution + cx;
          *iter_y = (static_cast<float>(y) - oy) * resolution + cy;
          *iter_z = hz;
          *iter_i = trav[offset + static_cast<size_t>(y) * dim_x + x];
        }
      }
    }

    pub->publish(cloud);
  }

  // Surface-only 可视化：只取地面高度+trav，过滤掉相邻切片高度差小于 slice_dh_
  // 的重复点
  void PublishSurfaceCloud(
      uint32_t n_slice, uint32_t dim_x, uint32_t dim_y, float cx, float cy,
      float resolution, const std::vector<float> &slice_heights,
      const std::vector<float> &trav, const std::vector<float> &elev_g,
      const std::vector<uint8_t> &missing_ground,
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub) {
    if (!pub)
      return;
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const size_t total = static_cast<size_t>(n_slice) * plane;
    if (trav.size() < total || elev_g.size() < total ||
        slice_heights.size() < n_slice)
      return;

    std::vector<float> buffer;
    buffer.reserve(total * 4);

    const float ox = static_cast<float>(dim_x) / 2.0f;
    const float oy = static_cast<float>(dim_y) / 2.0f;

    for (uint32_t s = 0; s < n_slice; ++s) {
      const float *trav_s = trav.data() + static_cast<size_t>(s) * plane;
      const float *elev_s = elev_g.data() + static_cast<size_t>(s) * plane;
      const float *elev_next =
          (s + 1 < n_slice) ? elev_g.data() + static_cast<size_t>(s + 1) * plane
                            : nullptr;
      const uint8_t *miss_s =
          missing_ground.data() + static_cast<size_t>(s) * plane;

      for (uint32_t y = 0; y < dim_y; ++y) {
        for (uint32_t x = 0; x < dim_x; ++x) {
          const size_t idx = static_cast<size_t>(y) * dim_x + x;
          if (elev_next) {
            const float dh = std::fabs(elev_next[idx] - elev_s[idx]);
            if (dh < static_cast<float>(slice_dh_))
              continue; // 相邻切片高度差太小，视作重复/遮挡
          }

          // 缺失地面处跳过渲染，避免铺底平面
          if (miss_s[idx])
            continue;

          const float wx = (static_cast<float>(x) - ox) * resolution + cx;
          const float wy = (static_cast<float>(y) - oy) * resolution + cy;
          const float wz = elev_s[idx];
          // 可视化用：将代价钳位到 RViz 可显示范围（0~50，与 Python 原版一致）
          // 内部规划仍用原始 inflated_cost（包含 1e6 障碍）
          const float raw_inten = trav_s[idx];
          const float inten = std::min(raw_inten, 50.0f);

          buffer.push_back(wx);
          buffer.push_back(wy);
          buffer.push_back(wz);
          buffer.push_back(inten);
        }
      }
    }

    if (buffer.empty())
      return;

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = map_frame_;
    cloud.header.stamp = now();
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(buffer.size() / 4);
    cloud.is_bigendian = false;
    cloud.is_dense = false;
    cloud.point_step = sizeof(float) * 4;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.fields.resize(4);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;
    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;
    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;
    cloud.fields[3].name = "intensity";
    cloud.fields[3].offset = 12;
    cloud.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[3].count = 1;
    cloud.data.resize(cloud.row_step);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_i(cloud, "intensity");

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

    pub->publish(cloud);
  }

  void WriteBinary(const std::string &path,
                   const std::vector<uint8_t> &buffer) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
      RCLCPP_WARN(get_logger(), "Failed to open %s for write", path.c_str());
      return;
    }
    ofs.write(reinterpret_cast<const char *>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tomogram_pub_;
  rclcpp::Publisher<std_msgs::msg::ByteMultiArray>::SharedPtr
      tomogram_data_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

  rclcpp::TimerBase::SharedPtr periodic_timer_;

  std::string output_path_;
  double resolution_;
  double slice_dh_;
  double interval_min_;
  double interval_free_;
  double slope_max_;
  double step_max_;
  double standable_ratio_;
  double cost_barrier_;
  double safe_margin_;
  double inflation_;
  int kernel_size_;
  std::string map_frame_;
  bool surface_only_;
  bool published_;
};

} // namespace pctplanner

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pctplanner::TomographyNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
