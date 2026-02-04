#pragma once

// Tomogram 可视化模块：生成用于 RViz 显示的 PointCloud2 消息
// 与 tomography_node::PublishSurfaceCloud 逻辑完全一致

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "tomogram_format.hpp"

namespace pctplanner {

// Tomogram 可视化配置
struct TomogramVizConfig {
  std::string frame_id = "map";
  float cost_clamp = 50.0f; // 代价钳位上限
};

// 从 tomogram 二进制生成可视化点云（Surface-only 模式）
inline sensor_msgs::msg::PointCloud2
CreateTomogramCloud(const std::vector<uint8_t> &tomogram_buffer,
                    const TomogramVizConfig &config = TomogramVizConfig()) {

  sensor_msgs::msg::PointCloud2 cloud;

  if (tomogram_buffer.empty()) {
    return cloud;
  }

  // 解析 tomogram header
  auto view = tomogram_format::Deserialize(tomogram_buffer);
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

  // 辅助函数：读取标量值
  auto read_scalar = [&](const uint8_t *base, size_t offset) -> float {
    if (mode == tomogram_format::PrecisionMode::FLOAT32) {
      float value;
      std::memcpy(&value, base + offset, sizeof(float));
      return value;
    } else {
      uint16_t bits;
      std::memcpy(&bits, base + offset, sizeof(uint16_t));
      Eigen::half h_val;
      std::memcpy(&h_val, &bits, sizeof(uint16_t));
      return static_cast<float>(h_val);
    }
  };

  // 读取 trav 和 elev_g 原始数组
  std::vector<float> trav(n_slice * plane);
  std::vector<float> elev_g(n_slice * plane);

  for (uint32_t s = 0; s < n_slice; ++s) {
    for (uint32_t x = 0; x < dim_x; ++x) {
      for (uint32_t y = 0; y < dim_y; ++y) {
        const size_t idx_plane =
            static_cast<size_t>(s) * plane + static_cast<size_t>(x) * dim_y + y;
        const size_t base_offset = idx_plane * scalar_bytes;

        // trav 在 layer 0, elev_g 在 layer 3
        trav[idx_plane] =
            read_scalar(view.data, base_offset + 0 * layer_stride);
        elev_g[idx_plane] =
            read_scalar(view.data, base_offset + 3 * layer_stride);
      }
    }
  }

  // 遮挡处理：与 tomography_node::PublishSurfaceCloud 完全一致
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

  // 遮挡处理：下层被上层遮挡时设为 NaN
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

        // elev_g 异常值跳过
        if (vis_g[s][idx] < -90.0f)
          continue;

        const float wx = (static_cast<float>(x) - ox) * resolution + cx;
        const float wy = (static_cast<float>(y) - oy) * resolution + cy;
        const float wz = vis_g[s][idx];
        const float inten = std::min(vis_t[s][idx], config.cost_clamp);

        buffer.push_back(wx);
        buffer.push_back(wy);
        buffer.push_back(wz);
        buffer.push_back(inten);
      }
    }
  }

  if (buffer.empty()) {
    return cloud;
  }

  // 创建 PointCloud2 消息
  cloud.header.frame_id = config.frame_id;
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

  // 使用迭代器填充数据
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

  return cloud;
}

} // namespace pctplanner
