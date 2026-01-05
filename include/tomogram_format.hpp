// 文件作用：定义 tomogram 二进制格式的头部与序列化/反序列化工具，供规划与建图节点共用。
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pctplanner {

struct TomogramHeader {
    std::array<char, 4> magic;   // 魔数 "TMG1"
    uint16_t version;            // 版本号
    uint16_t reserved;           // 保留字段
    uint32_t n_slice;            // 切片数量
    uint32_t dim_x;              // 栅格宽
    uint32_t dim_y;              // 栅格高
    float resolution;            // 分辨率
    float center_x;              // 地图中心 x
    float center_y;              // 地图中心 y
    float slice_h0;              // 首切片高度
    float slice_dh;              // 切片间距
};

namespace tomogram_format {
constexpr std::array<char, 4> kMagic = {'T', 'M', 'G', '1'};
constexpr uint16_t kVersion = 1;
constexpr size_t kHeaderBytes = sizeof(TomogramHeader);
constexpr size_t kLayersPerVoxel = 5;        // trav, gx, gy, elev_g, elev_c
constexpr size_t kFloat16Bytes = 2;          // uint16 承载 float16 位模式

inline size_t SliceHeightsBytes(uint32_t n_slice) {
    return static_cast<size_t>(n_slice) * kFloat16Bytes;
}

inline size_t VoxelLayerBytes(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y) {
    return static_cast<size_t>(n_slice) * dim_x * dim_y * kFloat16Bytes;
}

inline size_t DataBytes(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y) {
    return VoxelLayerBytes(n_slice, dim_x, dim_y) * kLayersPerVoxel;
}

inline size_t TotalBytes(const TomogramHeader& h) {
    return kHeaderBytes + SliceHeightsBytes(h.n_slice) + DataBytes(h.n_slice, h.dim_x, h.dim_y);
}

inline TomogramHeader MakeHeader(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                                 float resolution, float center_x, float center_y,
                                 float slice_h0, float slice_dh) {
    TomogramHeader h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.reserved = 0;
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

inline void Validate(const TomogramHeader& h) {
    if (h.magic != kMagic) throw std::runtime_error("tomogram magic mismatch");
    if (h.version != kVersion) throw std::runtime_error("tomogram version unsupported");
    if (h.n_slice == 0 || h.dim_x == 0 || h.dim_y == 0) throw std::runtime_error("tomogram dims must be positive");
}

inline std::vector<uint8_t> Serialize(const TomogramHeader& h,
                                      const std::vector<uint16_t>& slice_heights_fp16,
                                      const std::vector<uint16_t>& data_fp16) {
    Validate(h);
    const size_t expected_heights = h.n_slice;
    const size_t expected_data = kLayersPerVoxel * static_cast<size_t>(h.n_slice) * h.dim_x * h.dim_y;
    if (slice_heights_fp16.size() != expected_heights) throw std::runtime_error("slice_heights size mismatch");
    if (data_fp16.size() != expected_data) throw std::runtime_error("data size mismatch");

    std::vector<uint8_t> out;
    out.resize(TotalBytes(h));
    uint8_t* ptr = out.data();
    std::memcpy(ptr, &h, kHeaderBytes);
    ptr += kHeaderBytes;
    std::memcpy(ptr, slice_heights_fp16.data(), SliceHeightsBytes(h.n_slice));
    ptr += SliceHeightsBytes(h.n_slice);
    std::memcpy(ptr, data_fp16.data(), DataBytes(h.n_slice, h.dim_x, h.dim_y));
    return out;
}

struct View {
    TomogramHeader header;
    const uint16_t* slice_heights_fp16;
    const uint16_t* data_fp16;
};

inline View Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < kHeaderBytes) throw std::runtime_error("buffer too small for header");
    View v{};
    std::memcpy(&v.header, buffer.data(), kHeaderBytes);
    Validate(v.header);
    const size_t need = TotalBytes(v.header);
    if (buffer.size() < need) throw std::runtime_error("buffer too small for payload");
    const uint8_t* ptr = buffer.data() + kHeaderBytes;
    v.slice_heights_fp16 = reinterpret_cast<const uint16_t*>(ptr);
    ptr += SliceHeightsBytes(v.header.n_slice);
    v.data_fp16 = reinterpret_cast<const uint16_t*>(ptr);
    return v;
}

}  // namespace tomogram_format

}  // namespace pctplanner
