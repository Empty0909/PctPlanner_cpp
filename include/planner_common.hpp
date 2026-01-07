#pragma once

// 公共规划工具：Tomogram 解析、网格/世界坐标转换、PCD 导出。
// 供 planner_node 与 planner_direct_node 共用，避免重复实现。

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>

#include "tomogram_format.hpp"

namespace pctplanner {

struct PlannerInput {
  uint32_t n_slice;
  uint32_t dim_x;
  uint32_t dim_y;
  double resolution;
  double center_x;
  double center_y;
  std::vector<float> slice_heights; // 每层高度（米）
  Eigen::MatrixXd trav;             // 膨胀代价
  Eigen::MatrixXd trav_gx;          // trav x 梯度
  Eigen::MatrixXd trav_gy;          // trav y 梯度
  Eigen::MatrixXd elev_g;           // 地面高度
  Eigen::MatrixXd elev_c;           // 顶部高度
  Eigen::MatrixXi gateway;          // 上/下层可穿越标记
};

inline float Fp16ToFloat(uint16_t bits) {
  Eigen::half h;
  std::memcpy(&h, &bits, sizeof(uint16_t));
  return static_cast<float>(h);
}

inline double ReadScalar(const uint8_t *base, size_t offset,
                         tomogram_format::PrecisionMode mode) {
  if (mode == tomogram_format::PrecisionMode::FLOAT32) {
    float value;
    std::memcpy(&value, base + offset, sizeof(float));
    return static_cast<double>(value);
  }
  uint16_t bits;
  std::memcpy(&bits, base + offset, sizeof(uint16_t));
  return static_cast<double>(Fp16ToFloat(bits));
}

// 反序列化 tomogram 二进制，填充 PlannerInput。
inline void ParseTomogram(const std::vector<uint8_t> &buffer,
                          PlannerInput &out) {
  auto view = tomogram_format::Deserialize(buffer);
  const auto &h = view.header;
  const auto mode = tomogram_format::GetPrecisionMode(h);
  const size_t scalar_bytes = tomogram_format::ScalarBytes(mode);
  out.n_slice = h.n_slice;
  out.dim_x = h.dim_x;
  out.dim_y = h.dim_y;
  out.resolution = h.resolution;
  out.center_x = h.center_x;
  out.center_y = h.center_y;

  out.slice_heights.resize(h.n_slice);
  for (uint32_t i = 0; i < h.n_slice; ++i) {
    const size_t offset = static_cast<size_t>(i) * scalar_bytes;
    out.slice_heights[i] =
        static_cast<float>(ReadScalar(view.slice_heights, offset, mode));
  }

  const size_t plane = static_cast<size_t>(h.dim_x) * h.dim_y;
  const size_t rows = static_cast<size_t>(h.n_slice) * h.dim_y;
  out.trav.resize(rows, h.dim_x);
  out.trav_gx.resize(rows, h.dim_x);
  out.trav_gy.resize(rows, h.dim_x);
  out.elev_g.resize(rows, h.dim_x);
  out.elev_c.resize(rows, h.dim_x);
  out.gateway.resize(rows, h.dim_x);

  const size_t layer_stride =
      static_cast<size_t>(h.n_slice) * plane * scalar_bytes;
  for (uint32_t s = 0; s < h.n_slice; ++s) {
    for (uint32_t y = 0; y < h.dim_y; ++y) {
      for (uint32_t x = 0; x < h.dim_x; ++x) {
        const size_t idx_plane = s * plane + y * h.dim_x + x;
        const size_t base_offset = idx_plane * scalar_bytes;
        double trav =
            ReadScalar(view.data, base_offset + 0 * layer_stride, mode);
        double gx = ReadScalar(view.data, base_offset + 1 * layer_stride, mode);
        double gy = ReadScalar(view.data, base_offset + 2 * layer_stride, mode);
        double eg = ReadScalar(view.data, base_offset + 3 * layer_stride, mode);
        double ec = ReadScalar(view.data, base_offset + 4 * layer_stride, mode);

        const size_t row = static_cast<size_t>(s) * h.dim_y + y;
        out.trav(row, x) = trav;
        out.trav_gx(row, x) = gx;
        out.trav_gy(row, x) = gy;
        out.elev_g(row, x) = eg;
        out.elev_c(row, x) = ec;
      }
    }
  }

  // 与 Python 版本一致：把缺失地面/顶面高度填成哨兵值，避免 NaN 进入规划器。
  out.elev_g =
      out.elev_g.unaryExpr([](double v) { return std::isnan(v) ? -100.0 : v; });
  out.elev_c =
      out.elev_c.unaryExpr([](double v) { return std::isnan(v) ? 1e6 : v; });

  // gateway 判定与 Python 一致：跨层代价突变且地面高度连续时视为“可穿越”。
  out.gateway.setZero();
  for (uint32_t s = 0; s + 1 < h.n_slice; ++s) {
    for (uint32_t y = 0; y < h.dim_y; ++y) {
      const size_t row = static_cast<size_t>(s) * h.dim_y + y;
      const size_t row_next = static_cast<size_t>(s + 1) * h.dim_y + y;
      for (uint32_t x = 0; x < h.dim_x; ++x) {
        double diff_t = out.trav(row_next, x) - out.trav(row, x);
        double diff_g = std::abs(out.elev_g(row_next, x) - out.elev_g(row, x));
        if (diff_t < -8.0 && diff_g < 0.1)
          out.gateway(row, x) = 2; // 通往上一层
        if (diff_t > 8.0 && diff_g < 0.1)
          out.gateway(row_next, x) = -2; // 通往下一层
      }
    }
  }
}

inline int ZToSlice(double z, const std::vector<float> &heights) {
  int idx = -1;
  for (size_t i = 0; i < heights.size(); ++i) {
    if (z < heights[i])
      break;
    idx = static_cast<int>(i);
  }
  if (idx < 0)
    idx = 0;
  if (idx >= static_cast<int>(heights.size()))
    idx = static_cast<int>(heights.size()) - 1;
  return idx;
}

// 世界坐标 → 栅格索引（layer, y, x）。返回 false 表示越界。
inline bool PosToIdx(const geometry_msgs::msg::Point &p, const PlannerInput &in,
                     Eigen::Vector3i &out_idx) {
  double y = p.y - in.center_y;
  double x = p.x - in.center_x;
  int ix = static_cast<int>(std::round(x / in.resolution) +
                            static_cast<double>(in.dim_x) / 2.0);
  int iy = static_cast<int>(std::round(y / in.resolution) +
                            static_cast<double>(in.dim_y) / 2.0);
  int layer = ZToSlice(p.z, in.slice_heights);
  if (ix < 0 || ix >= static_cast<int>(in.dim_x) || iy < 0 ||
      iy >= static_cast<int>(in.dim_y) || layer < 0 ||
      layer >= static_cast<int>(in.n_slice)) {
    return false;
  }
  out_idx = Eigen::Vector3i(layer, iy, ix);
  return true;
}

// 栅格坐标 → 世界坐标（原点在地图中心）。
inline void GridToMap(const PlannerInput &in, Eigen::Vector3d &p) {
  // 与 Python transTrajGrid2Map 保持一致：
  // world_x = (grid_y - dim_x/2) * res + center_x
  // world_y = (grid_x - dim_y/2) * res + center_y
  // world_z = grid_z + 0.5（高度已是米制，仅保留常数偏移）
  const double ox = static_cast<double>(in.dim_y) / 2.0;
  const double oy = static_cast<double>(in.dim_x) / 2.0;
  const double gx = (p.y() - oy) * in.resolution + in.center_x;
  const double gy = (p.x() - ox) * in.resolution + in.center_y;
  p.x() = gx;
  p.y() = gy;
  p.z() = p.z() + 0.5; // z 保持米制，只加常数偏置
}

// 写 ASCII PCD，成功返回 true。
inline bool WriteAsciiPCD(const std::string &path,
                          const std::vector<Eigen::Vector3d> &pts) {
  std::ofstream ofs(path);
  if (!ofs) {
    return false;
  }
  ofs << "# .PCD v0.7\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z\n";
  ofs << "SIZE 4 4 4\n";
  ofs << "TYPE F F F\n";
  ofs << "COUNT 1 1 1\n";
  ofs << "WIDTH " << pts.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << pts.size() << "\n";
  ofs << "DATA ascii\n";
  for (const auto &p : pts) {
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  }
  return true;
}

} // namespace pctplanner
