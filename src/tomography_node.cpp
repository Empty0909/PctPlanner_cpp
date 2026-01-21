// 文件作用：订阅 /global_points 点云，GPU（CUDA）生成
// tomogram（膨胀代价/梯度/地面/顶部高程），按可配置精度（float16/float32）
// 序列化发布 /tomogram_data 并写入磁盘；可视化膨胀后代价、梯度、间隙。
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
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
        // ground_h: 地面参考高度，与 Python scene.py 中 ground_h 一致
        // 用于覆盖点云最低高度，确保切片起始高度可控
        ground_h_(declare_parameter<double>("ground_h", 0.0)),
        // 间隙检测参数：与 Python 原版 scene.py 保持一致
        // interval_min: 最小可通行间隙高度
        interval_min_(declare_parameter<double>("interval_min", 0.50)),
        // interval_free: 无惩罚间隙高度阈值
        interval_free_(declare_parameter<double>("interval_free", 0.60)),
        // 坡度参数：slope_max 用于计算 step_stand
        slope_max_(declare_parameter<double>("slope_max", 1.0)),
        // 越障参数：step_max 控制可跨越的最大高度差
        step_max_(declare_parameter<double>("step_max", 0.70)),
        standable_ratio_(declare_parameter<double>("standable_ratio", 0.40)),
        // cost_barrier 与 Python 原版一致
        cost_barrier_(declare_parameter<double>("cost_barrier", 50.0)),
        // 膨胀参数：与 Python 原版一致
        safe_margin_(declare_parameter<double>("safe_margin", 0.10)),
        inflation_(declare_parameter<double>("inflation", 0.05)),
        kernel_size_(declare_parameter<int>("kernel_size", 5)),
        map_frame_(declare_parameter<std::string>("map_frame", "map")),
        precision_mode_(ResolvePrecisionMode(
            declare_parameter<std::string>("precision_mode", "float32"))),
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
    // 与 Python 版本一致：用 ground_h 覆盖点云最低高度，使切片起始可控
    // Python: self.points_min[-1] = self.ground_h
    pmin.z() = static_cast<float>(ground_h_);
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

    // 记录地图范围，便于设置起终点：中心/尺寸/分辨率/切片高度
    RCLCPP_INFO(
        get_logger(),
        "Tomogram bbox min[%.2f %.2f %.2f] max[%.2f %.2f %.2f], center[%.2f "
        "%.2f], res=%.3f, dims=%ux%u, slices=%u, slice_h0=%.3f, dh=%.3f",
        pmin.x(), pmin.y(), pmin.z(), pmax.x(), pmax.y(), pmax.z(), cx, cy,
        static_cast<float>(resolution_), dim_x, dim_y, n_slice, slice_h0,
        slice_dh);

    std::vector<float> slice_heights_f32(n_slice);
    for (uint32_t i = 0; i < n_slice; ++i) {
      float h = slice_h0 + slice_dh * static_cast<float>(i);
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

    // 记录缺失地面/顶部的掩码，仅用于可视化过滤；不改写原始高度，保持与原版一致
    // 阈值与 Python 版本一致：Python 使用 > -1e6 判断有效，这里用 <= -1e6 + 1
    // 判断缺失 原版使用 -9e5f 会导致 -999999 等值被错误标记为缺失
    std::vector<uint8_t> missing_ground(gpu_out.elev_g.size(), 0);
    std::vector<uint8_t> missing_ceiling(gpu_out.elev_c.size(), 0);
    for (size_t i = 0; i < gpu_out.elev_g.size(); ++i) {
      if (gpu_out.elev_g[i] <= -1e6f + 1.0f) {
        missing_ground[i] = 1;
      }
      if (gpu_out.elev_c[i] >= 1e6f - 1.0f) {
        missing_ceiling[i] = 1;
      }
    }

    // 层简化（与原版 idx_simp 一致）：仅保留“有差异且可通行”的切片
    std::vector<uint32_t> idx_simp;
    idx_simp.push_back(0);
    if (n_slice > 1) {
      uint32_t l_idx = 0;
      uint32_t m_idx = 1;
      while (m_idx < n_slice - 2) {
        bool keep = false;
        const size_t plane = static_cast<size_t>(dim_x) * dim_y;
        const size_t offset_l = static_cast<size_t>(l_idx) * plane;
        const size_t offset_m = static_cast<size_t>(m_idx) * plane;
        const size_t offset_u = static_cast<size_t>(m_idx + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
          const float g_l = gpu_out.elev_g[offset_l + i];
          const float g_m = gpu_out.elev_g[offset_m + i];
          const float cost_l = gpu_out.inflated_cost[offset_l + i];
          const float cost_m = gpu_out.inflated_cost[offset_m + i];
          const float diff_h = gpu_out.elev_g[offset_u + i] - g_m;
          const bool mask_l_g = (g_m - g_l) > 0.0f;
          const bool mask_l_t = cost_l > cost_m;
          const bool mask_u_g = diff_h > 0.0f;
          const bool mask_t = cost_m < static_cast<float>(cost_barrier_);
          if ((mask_l_g || mask_l_t) && mask_u_g && mask_t) {
            keep = true;
            break;
          }
        }
        if (keep) {
          idx_simp.push_back(m_idx);
          l_idx = m_idx;
        }
        ++m_idx;
      }
      idx_simp.push_back(m_idx); // 对应原版最后一次追加（n_slice-2）
    }

    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const uint32_t simp_layers = static_cast<uint32_t>(idx_simp.size());

    // 调试：输出层简化信息
    RCLCPP_INFO(get_logger(),
                "Layer simplification: init=%u, simp=%u, idx_simp=[%s]",
                n_slice, simp_layers,
                [&]() {
                  std::string s;
                  for (size_t i = 0; i < idx_simp.size(); ++i) {
                    if (i > 0)
                      s += ",";
                    s += std::to_string(idx_simp[i]);
                  }
                  return s;
                }()
                    .c_str());

    // 生成简化后的 trav / elev / 高度
    std::vector<float> trav_simp(static_cast<size_t>(simp_layers) * plane);
    std::vector<float> elev_g_simp(static_cast<size_t>(simp_layers) * plane);
    std::vector<float> elev_c_simp(static_cast<size_t>(simp_layers) * plane);
    std::vector<float> slice_heights_simp;
    slice_heights_simp.reserve(simp_layers);
    std::vector<uint8_t> missing_ground_simp(
        static_cast<size_t>(simp_layers) * plane, 0);
    std::vector<uint8_t> missing_ceiling_simp(
        static_cast<size_t>(simp_layers) * plane, 0);

    for (size_t i = 0; i < idx_simp.size(); ++i) {
      const uint32_t src = idx_simp[i];
      const size_t src_off = static_cast<size_t>(src) * plane;
      const size_t dst_off = static_cast<size_t>(i) * plane;
      std::copy_n(gpu_out.inflated_cost.begin() + src_off, plane,
                  trav_simp.begin() + dst_off);
      std::copy_n(gpu_out.elev_g.begin() + src_off, plane,
                  elev_g_simp.begin() + dst_off);
      std::copy_n(gpu_out.elev_c.begin() + src_off, plane,
                  elev_c_simp.begin() + dst_off);
      std::copy_n(missing_ground.begin() + src_off, plane,
                  missing_ground_simp.begin() + dst_off);
      // ceiling 缺失掩码也对齐简化层
      std::copy_n(missing_ceiling.begin() + src_off, plane,
                  missing_ceiling_simp.begin() + dst_off);
      slice_heights_simp.push_back(slice_heights_f32[src]);
    }

    // 将缺失处置为 NaN，保持与原版导出一致（只影响序列化与可视化）
    for (size_t i = 0; i < elev_g_simp.size(); ++i) {
      if (missing_ground_simp[i]) {
        elev_g_simp[i] = std::numeric_limits<float>::quiet_NaN();
      }
      if (missing_ceiling_simp[i]) {
        elev_c_simp[i] = std::numeric_limits<float>::quiet_NaN();
      }
    }

    // 生成 trav 梯度（在简化层上取中心差分）
    std::vector<float> trav_gx(static_cast<size_t>(simp_layers) * plane, 0.0f);
    std::vector<float> trav_gy(static_cast<size_t>(simp_layers) * plane, 0.0f);
    ComputeTravGradient(simp_layers, dim_x, dim_y, trav_simp, trav_gx, trav_gy);

    // 序列化 tomogram（简化后的层）
    TomogramHeader simp_header = tomogram_format::MakeHeader(
        simp_layers, dim_x, dim_y, static_cast<float>(resolution_), cx, cy,
        slice_heights_simp.front(), slice_dh, precision_mode_);

    std::vector<uint8_t> slice_heights_raw;
    EncodeScalars(slice_heights_simp, precision_mode_, slice_heights_raw);

    std::vector<uint8_t> data_bytes;
    FillTomogramPayload(simp_layers, dim_x, dim_y, trav_simp, trav_gx, trav_gy,
                        elev_g_simp, elev_c_simp, precision_mode_, data_bytes);

    std::vector<uint8_t> payload =
        tomogram_format::Serialize(simp_header, slice_heights_raw, data_bytes);

    std_msgs::msg::ByteMultiArray msg_out;
    msg_out.data.assign(payload.begin(), payload.end());
    tomogram_data_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published tomogram_data (%zu bytes)",
                payload.size());

    // 可视化：默认发布全量体素；如需仅地面可在参数 surface_only=true
    if (surface_only_) {
      PublishSurfaceCloud(simp_layers, dim_x, dim_y, cx, cy,
                          static_cast<float>(resolution_), slice_heights_simp,
                          trav_simp, elev_g_simp, missing_ground_simp,
                          tomogram_pub_);
    } else {
      PublishAllCloud(simp_layers, dim_x, dim_y, cx, cy,
                      static_cast<float>(resolution_), slice_heights_simp,
                      trav_simp, tomogram_pub_);
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

  static tomogram_format::PrecisionMode
  ResolvePrecisionMode(const std::string &mode_str) {
    std::string lower = mode_str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower == "float32" || lower == "fp32" || lower == "f32") {
      return tomogram_format::PrecisionMode::FLOAT32;
    }
    if (lower == "float16" || lower == "fp16" || lower == "f16") {
      return tomogram_format::PrecisionMode::FLOAT16;
    }
    throw std::runtime_error("Unsupported precision_mode: " + mode_str);
  }

  static void EncodeScalars(const std::vector<float> &values,
                            tomogram_format::PrecisionMode mode,
                            std::vector<uint8_t> &out) {
    const size_t bytes = tomogram_format::ScalarBytes(mode);
    out.resize(values.size() * bytes);
    for (size_t i = 0; i < values.size(); ++i) {
      uint8_t *dst = out.data() + i * bytes;
      if (mode == tomogram_format::PrecisionMode::FLOAT32) {
        float val = values[i];
        std::memcpy(dst, &val, sizeof(float));
      } else {
        uint16_t bits = Float32ToFp16(values[i]);
        std::memcpy(dst, &bits, sizeof(uint16_t));
      }
    }
  }

  // 计算 trav 梯度，与 Python 版本严格对齐
  // Python 代码:
  //   trav_grad_x = inflated_cost[:, 2:, :] - inflated_cost[:, :-2, :]
  //   trav_grad_y = inflated_cost[:, :, 2:] - inflated_cost[:, :, :-2]
  //   trav_gx[:, 1:-1, :] = trav_grad_x  # x=1 到 x=dim_x-2，所有 y
  //   trav_gy[:, :, 1:-1] = trav_grad_y  # 所有 x，y=1 到 y=dim_y-2
  //
  // 关键区别：
  // - trav_gx 在 x 边界为 0，但在 y 边界有值
  // - trav_gy 在 y 边界为 0，但在 x 边界有值
  // 布局: [slice][x][y]，索引 = dim_y * x + y
  void ComputeTravGradient(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                           const std::vector<float> &cost,
                           std::vector<float> &gx, std::vector<float> &gy) {
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    for (uint32_t s = 0; s < n_slice; ++s) {
      const size_t offset = static_cast<size_t>(s) * plane;

      // gx: x 方向梯度，x=1 到 x=dim_x-2，所有 y 位置
      for (uint32_t x = 1; x + 1 < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y) {
          const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
          // gx = cost[x+1, y] - cost[x-1, y]，偏移量 ±dim_y
          gx[idx] = cost[idx + dim_y] - cost[idx - dim_y];
        }
      }

      // gy: y 方向梯度，所有 x 位置，y=1 到 y=dim_y-2
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 1; y + 1 < dim_y; ++y) {
          const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
          // gy = cost[x, y+1] - cost[x, y-1]，偏移量 ±1
          gy[idx] = cost[idx + 1] - cost[idx - 1];
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
                           tomogram_format::PrecisionMode mode,
                           std::vector<uint8_t> &data_bytes) {
    const size_t plane = static_cast<size_t>(dim_x) * dim_y;
    const size_t scalar_bytes = tomogram_format::ScalarBytes(mode);
    const size_t layer_stride =
        static_cast<size_t>(n_slice) * plane * scalar_bytes;
    data_bytes.resize(static_cast<size_t>(n_slice) * plane *
                      tomogram_format::kLayersPerVoxel * scalar_bytes);

    auto write_value = [&](size_t idx_plane, size_t layer, float value) {
      const size_t offset =
          (idx_plane + layer * static_cast<size_t>(n_slice) * plane) *
          scalar_bytes;
      uint8_t *dst = data_bytes.data() + offset;
      if (mode == tomogram_format::PrecisionMode::FLOAT32) {
        float v = value;
        std::memcpy(dst, &v, sizeof(float));
      } else {
        uint16_t bits = Float32ToFp16(value);
        std::memcpy(dst, &bits, sizeof(uint16_t));
      }
    };

    // 遍历顺序与 Python 一致: [slice][x][y]
    // 这里直接按线性索引写入，因为输入数组已按正确布局排列
    for (uint32_t s = 0; s < n_slice; ++s) {
      const size_t offset = static_cast<size_t>(s) * plane;
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y) {
          // idx = dim_y * x + y
          const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
          write_value(idx, 0, trav[idx]);
          write_value(idx, 1, gx[idx]);
          write_value(idx, 2, gy[idx]);
          write_value(idx, 3, ground[idx]);
          write_value(idx, 4, ceiling[idx]);
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

    // 遍历顺序与数据布局一致: [slice][x][y]
    for (uint32_t s = 0; s < n_slice; ++s) {
      float hz = slice_heights[s];
      const size_t offset = static_cast<size_t>(s) * plane;
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y;
             ++y, ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
          *iter_x = (static_cast<float>(x) - ox) * resolution + cx;
          *iter_y = (static_cast<float>(y) - oy) * resolution + cy;
          *iter_z = hz;
          // idx = dim_y * x + y
          *iter_i = trav[offset + static_cast<size_t>(x) * dim_y + y];
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

    // 与Python一致：复制数据进行可视化处理
    // vis_g[s][idx] 为NaN表示跳过，vis_t是可能被修改的代价值
    std::vector<std::vector<float>> vis_g(n_slice, std::vector<float>(plane));
    std::vector<std::vector<float>> vis_t(n_slice, std::vector<float>(plane));

    // 复制原始数据
    for (uint32_t s = 0; s < n_slice; ++s) {
      std::copy(elev_g.begin() + static_cast<size_t>(s) * plane,
                elev_g.begin() + static_cast<size_t>(s + 1) * plane,
                vis_g[s].begin());
      std::copy(trav.begin() + static_cast<size_t>(s) * plane,
                trav.begin() + static_cast<size_t>(s + 1) * plane,
                vis_t[s].begin());
    }

    // 与Python完全一致的遮挡处理逻辑：
    // for i in range(n_slice - 1):
    //     mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
    //     vis_g[i, mask_h] = np.nan          # 下层被遮挡，设为nan
    //     vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1,
    //     mask_h])
    for (uint32_t s = 0; s + 1 < n_slice; ++s) {
      for (size_t idx = 0; idx < plane; ++idx) {
        const float dh = vis_g[s + 1][idx] - vis_g[s][idx];
        if (dh < static_cast<float>(slice_dh_)) {
          // 下层被遮挡，设为NaN跳过
          vis_g[s][idx] = std::numeric_limits<float>::quiet_NaN();
          // 关键：上层代价取下层和上层的最小值
          vis_t[s + 1][idx] = std::min(vis_t[s][idx], vis_t[s + 1][idx]);
        }
      }
    }

    std::vector<float> buffer;
    buffer.reserve(total * 4);

    const float ox = static_cast<float>(dim_x) / 2.0f;
    const float oy = static_cast<float>(dim_y) / 2.0f;

    for (uint32_t s = 0; s < n_slice; ++s) {
      const uint8_t *miss_s =
          missing_ground.data() + static_cast<size_t>(s) * plane;

      // 遍历顺序与数据布局一致: [slice][x][y]
      for (uint32_t x = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y) {
          const size_t idx = static_cast<size_t>(x) * dim_y + y;

          // 与Python一致：vis_g为NaN表示被遮挡，跳过
          if (std::isnan(vis_g[s][idx]))
            continue;

          // 缺失地面处跳过渲染，避免铺底平面
          if (miss_s[idx])
            continue;

          const float wx = (static_cast<float>(x) - ox) * resolution + cx;
          const float wy = (static_cast<float>(y) - oy) * resolution + cy;
          const float wz = vis_g[s][idx]; // 使用vis_g
          // 可视化用：将代价钳位到 RViz 可显示范围（0~50，与 Python 原版一致）
          const float raw_inten =
              vis_t[s][idx]; // 使用vis_t（已处理过遮挡代价传递）
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
  double ground_h_;
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
  tomogram_format::PrecisionMode precision_mode_;
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
