// 文件作用：定义 tomogram
// 二进制格式的头部与序列化/反序列化工具，供规划与建图节点共用。
//
// 数据布局（与 Python 版本完全一致）:
//   Header: 48 bytes
//   SliceHeights: n_slice × scalar_bytes
//   Data: 5 × n_slice × dim_x × dim_y × scalar_bytes
//         层顺序: [trav][trav_gx][trav_gy][elev_g][elev_c]
//         每层内存布局: [slice][x][y]，即 index = dim_y * x + y
//
// dim_x 对应世界坐标 x 轴（点云的 px），dim_y 对应 y 轴（py）
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pctplanner {

struct TomogramHeader {
  std::array<char, 4> magic; // 魔数 "TMG1"
  uint16_t version;          // 版本号
  uint16_t precision_mode;   // 精度模式（与 PrecisionMode 对应）
  uint32_t n_slice;          // 切片数量
  uint32_t dim_x;            // 栅格宽
  uint32_t dim_y;            // 栅格高
  float resolution;          // 分辨率
  float center_x;            // 地图中心 x
  float center_y;            // 地图中心 y
  float slice_h0;            // 首切片高度
  float slice_dh;            // 切片间距
};

namespace tomogram_format {

enum class PrecisionMode : uint16_t {
  FLOAT16 = 0,
  FLOAT32 = 1,
};
constexpr std::array<char, 4> kMagic = {'T', 'M', 'G', '1'};
constexpr uint16_t kVersion = 1;
constexpr size_t kHeaderBytes = sizeof(TomogramHeader);
constexpr size_t kLayersPerVoxel = 5; // trav, gx, gy, elev_g, elev_c

inline PrecisionMode GetPrecisionMode(const TomogramHeader &h) {
  if (h.precision_mode == static_cast<uint16_t>(PrecisionMode::FLOAT32))
    return PrecisionMode::FLOAT32;
  return PrecisionMode::FLOAT16;
}

inline size_t ScalarBytes(PrecisionMode mode) {
  return mode == PrecisionMode::FLOAT32 ? sizeof(float) : sizeof(uint16_t);
}

inline size_t SliceHeightsBytes(const TomogramHeader &h) {
  return static_cast<size_t>(h.n_slice) * ScalarBytes(GetPrecisionMode(h));
}

inline size_t VoxelLayerBytes(const TomogramHeader &h) {
  return static_cast<size_t>(h.n_slice) * h.dim_x * h.dim_y *
         ScalarBytes(GetPrecisionMode(h));
}

inline size_t DataBytes(const TomogramHeader &h) {
  return VoxelLayerBytes(h) * kLayersPerVoxel;
}

inline size_t TotalBytes(const TomogramHeader &h) {
  return kHeaderBytes + SliceHeightsBytes(h) + DataBytes(h);
}

inline void ValidatePrecision(uint16_t value) {
  const bool ok = (value == static_cast<uint16_t>(PrecisionMode::FLOAT16)) ||
                  (value == static_cast<uint16_t>(PrecisionMode::FLOAT32));
  if (!ok) {
    throw std::runtime_error("tomogram precision unsupported");
  }
}

inline TomogramHeader
MakeHeader(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y, float resolution,
           float center_x, float center_y, float slice_h0, float slice_dh,
           PrecisionMode precision = PrecisionMode::FLOAT16) {
  TomogramHeader h{};
  h.magic = kMagic;
  h.version = kVersion;
  h.precision_mode = static_cast<uint16_t>(precision);
  h.n_slice = n_slice;
  h.dim_x = dim_x;
  h.dim_y = dim_y;
  h.resolution = resolution;
  h.center_x = center_x;
  h.center_y = center_y;
  h.slice_h0 = slice_h0;
  h.slice_dh = slice_dh;
  return h;
}

inline void Validate(const TomogramHeader &h) {
  if (h.magic != kMagic)
    throw std::runtime_error("tomogram magic mismatch");
  if (h.version != kVersion)
    throw std::runtime_error("tomogram version unsupported");
  ValidatePrecision(h.precision_mode);
  if (h.n_slice == 0 || h.dim_x == 0 || h.dim_y == 0)
    throw std::runtime_error("tomogram dims must be positive");
}

inline std::vector<uint8_t>
Serialize(const TomogramHeader &h,
          const std::vector<uint8_t> &slice_heights_raw,
          const std::vector<uint8_t> &data_raw) {
  Validate(h);
  const size_t expected_heights = SliceHeightsBytes(h);
  const size_t expected_data = DataBytes(h);
  if (slice_heights_raw.size() != expected_heights)
    throw std::runtime_error("slice_heights size mismatch");
  if (data_raw.size() != expected_data)
    throw std::runtime_error("data size mismatch");

  std::vector<uint8_t> out;
  out.resize(TotalBytes(h));
  uint8_t *ptr = out.data();
  std::memcpy(ptr, &h, kHeaderBytes);
  ptr += kHeaderBytes;
  std::memcpy(ptr, slice_heights_raw.data(), SliceHeightsBytes(h));
  ptr += SliceHeightsBytes(h);
  std::memcpy(ptr, data_raw.data(), DataBytes(h));
  return out;
}

struct View {
  TomogramHeader header;
  const uint8_t *slice_heights;
  const uint8_t *data;
};

inline View Deserialize(const std::vector<uint8_t> &buffer) {
  if (buffer.size() < kHeaderBytes)
    throw std::runtime_error("buffer too small for header");
  View v{};
  std::memcpy(&v.header, buffer.data(), kHeaderBytes);
  Validate(v.header);
  const size_t need = TotalBytes(v.header);
  if (buffer.size() < need)
    throw std::runtime_error("buffer too small for payload");
  const uint8_t *ptr = buffer.data() + kHeaderBytes;
  v.slice_heights = ptr;
  ptr += SliceHeightsBytes(v.header);
  v.data = ptr;
  return v;
}

} // namespace tomogram_format

using PrecisionMode = tomogram_format::PrecisionMode;

} // namespace pctplanner
