#include "tomography_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace pctplanner {
namespace {

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

__device__ inline float atomicMaxFloat(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ inline float atomicMinFloat(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

// 相对索引计算，与 Python kernels.py getIdxRelative 保持一致
// 布局: [slice][x][y]，即 index = dim_y * x + y
__device__ inline int getIdxRelative(int idx, int dx, int dy, int dim_x,
                                     int dim_y) {
  int plane_idx = idx % (dim_x * dim_y);
  int iy = plane_idx % dim_y; // y = plane_idx % dim_y
  int ix = plane_idx / dim_y; // x = plane_idx / dim_y
  int rx = ix + dx;
  int ry = iy + dy;
  if (rx < 0 || rx >= dim_x || ry < 0 || ry >= dim_y)
    return -1;
  // 偏移量: dx * dim_y + dy
  return idx + dx * dim_y + dy;
}

__global__ void ClearKernel(float *layers_g, float *layers_c,
                            float *grad_mag_sq, float *grad_mag_max,
                            float *trav_cost, float *inflated_cost,
                            float *interval, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  layers_g[idx] = -1e6f;
  layers_c[idx] = 1e6f;
  grad_mag_sq[idx] = 0.0f;
  grad_mag_max[idx] = 0.0f;
  trav_cost[idx] = 0.0f;
  inflated_cost[idx] = 0.0f;
  interval[idx] = 0.0f;
}

__global__ void TomographyKernel(const float3 *points, int num_points,
                                 float *layers_g, float *layers_c, float cx,
                                 float cy, float resolution, int dim_x,
                                 int dim_y, int n_slice, float slice_h0,
                                 float slice_dh) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points)
    return;
  float px = points[idx].x;
  float py = points[idx].y;
  float pz = points[idx].z;

  // 索引计算与 Python kernels.py 保持一致:
  // Python: idx_x = round((x - cx) / resolution) + n_row / 2
  //         idx_y = round((y - cy) / resolution) + n_col / 2
  //         index = n_col * idx_x + idx_y
  // 这里 n_row = dim_x, n_col = dim_y
  int ix = static_cast<int>(
      roundf((px - cx) / resolution + static_cast<float>(dim_x) / 2.0f));
  int iy = static_cast<int>(
      roundf((py - cy) / resolution + static_cast<float>(dim_y) / 2.0f));
  if (ix < 0 || ix >= dim_x || iy < 0 || iy >= dim_y)
    return;

  // 与 Python 一致: base = dim_y * ix + iy (Python: n_col * idx_x + idx_y)
  int base = dim_y * ix + iy;
  for (int s = 0; s < n_slice; ++s) {
    float slice = slice_h0 + slice_dh * static_cast<float>(s);
    int offset = s * dim_x * dim_y + base;
    if (pz <= slice) {
      atomicMaxFloat(&layers_g[offset], pz);
    } else {
      atomicMinFloat(&layers_c[offset], pz);
    }
  }
}

__global__ void GradIntervalKernel(const float *layers_g, const float *layers_c,
                                   float *grad_mag_sq, float *grad_mag_max,
                                   float *interval, int dim_x, int dim_y,
                                   int n_slice) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = dim_x * dim_y;
  int total = plane * n_slice;
  if (idx >= total)
    return;

  // 布局与 Python 一致: [slice][x][y]，即 index = dim_y * x + y
  int plane_idx = idx % plane;
  int y = plane_idx % dim_y;
  int x = plane_idx / dim_y;

  float lg = layers_g[idx];
  float lc = layers_c[idx];
  interval[idx] = lc - lg;

  // 如果当前体素没有地面点（未初始化），梯度设为0
  // 缺失地面的体素会在后续被标记为障碍
  const float missing_threshold = -9e5f;
  if (lg < missing_threshold) {
    grad_mag_sq[idx] = 0.0f;
    grad_mag_max[idx] = 0.0f;
    return;
  }

  if (x == 0 || x == dim_x - 1 || y == 0 || y == dim_y - 1) {
    grad_mag_sq[idx] = 0.0f;
    grad_mag_max[idx] = 0.0f;
    return;
  }

  // 相邻索引: y方向±1, x方向±dim_y（因为布局是 dim_y * x + y）
  int idx_ym = idx - 1;     // y - 1
  int idx_yp = idx + 1;     // y + 1
  int idx_xm = idx - dim_y; // x - 1
  int idx_xp = idx + dim_y; // x + 1

  // 获取相邻体素的地面高度，如果缺失则使用当前体素高度（梯度为0）
  float lg_xm = layers_g[idx_xm];
  float lg_xp = layers_g[idx_xp];
  float lg_ym = layers_g[idx_ym];
  float lg_yp = layers_g[idx_yp];

  // 如果相邻体素缺失地面，用当前高度替代（不产生梯度）
  if (lg_xm < missing_threshold)
    lg_xm = lg;
  if (lg_xp < missing_threshold)
    lg_xp = lg;
  if (lg_ym < missing_threshold)
    lg_ym = lg;
  if (lg_yp < missing_threshold)
    lg_yp = lg;

  float diff_x1 = lg - lg_xm;
  float diff_x2 = lg - lg_xp;
  float diff_y1 = lg - lg_ym;
  float diff_y2 = lg - lg_yp;

  float diff_x_sq = fmaxf(diff_x1 * diff_x1, diff_x2 * diff_x2);
  float diff_y_sq = fmaxf(diff_y1 * diff_y1, diff_y2 * diff_y2);

  grad_mag_sq[idx] = diff_x_sq + diff_y_sq;
  grad_mag_max[idx] = fmaxf(diff_x_sq, diff_y_sq);
}

__global__ void TravKernel(const float *interval, const float *grad_mag_sq,
                           const float *grad_mag_max, float *trav_cost,
                           int dim_x, int dim_y, int n_slice,
                           int half_kernel_size, float interval_min,
                           float interval_free, float step_cross_sq,
                           float step_stand_sq, int standable_th,
                           float cost_barrier) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = dim_x * dim_y;
  int total = plane * n_slice;
  if (idx >= total)
    return;

  float cost = 0.0f;
  float inter = interval[idx];
  if (inter < interval_min) {
    trav_cost[idx] = cost_barrier;
    return;
  }

  cost += fmaxf(0.0f, 20.0f * (interval_free - inter));

  float g_sq = grad_mag_sq[idx];
  float g_max = grad_mag_max[idx];
  if (g_sq <= step_stand_sq) {
    cost += 15.0f * g_sq / step_stand_sq;
    trav_cost[idx] = cost;
    return;
  }

  if (g_max <= step_cross_sq) {
    int standable = 0;
    for (int dy = -half_kernel_size; dy <= half_kernel_size; ++dy) {
      for (int dx = -half_kernel_size; dx <= half_kernel_size; ++dx) {
        int nbr = getIdxRelative(idx, dx, dy, dim_x, dim_y);
        if (nbr < 0)
          continue;
        if (grad_mag_sq[nbr] < step_stand_sq)
          standable++;
      }
    }
    if (standable < standable_th) {
      trav_cost[idx] = cost_barrier;
      return;
    }
    cost += 20.0f * g_max / step_cross_sq;
    trav_cost[idx] = cost;
    return;
  }

  trav_cost[idx] = cost_barrier;
}

__global__ void InflationKernel(const float *trav_cost,
                                const float *score_table, float *inflated_cost,
                                int dim_x, int dim_y, int n_slice,
                                int half_kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = dim_x * dim_y;
  int total = plane * n_slice;
  if (idx >= total)
    return;

  float acc = inflated_cost[idx];
  int counter = 0;
  for (int dy = -half_kernel_size; dy <= half_kernel_size; ++dy) {
    for (int dx = -half_kernel_size; dx <= half_kernel_size; ++dx) {
      int nbr = getIdxRelative(idx, dx, dy, dim_x, dim_y);
      if (nbr >= 0) {
        float val = trav_cost[nbr] * score_table[counter];
        if (val > acc)
          acc = val;
      }
      counter++;
    }
  }
  inflated_cost[idx] = acc;
}

} // namespace

bool RunTomographyCuda(const std::vector<Eigen::Vector3f> &points,
                       uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                       float cx, float cy, float slice_h0,
                       const TravParams &params, GpuTomographyOutput &out) {
  const int plane = static_cast<int>(dim_x * dim_y);
  const int total = plane * static_cast<int>(n_slice);
  const int threads = 256;
  const int blocks_points =
      static_cast<int>((points.size() + threads - 1) / threads);
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  float *d_layers_g = nullptr;
  float *d_layers_c = nullptr;
  float *d_grad_mag_sq = nullptr;
  float *d_grad_mag_max = nullptr;
  float *d_trav_cost = nullptr;
  float *d_inflated_cost = nullptr;
  float *d_interval = nullptr;
  float3 *d_points = nullptr;
  float *d_score = nullptr;

  CUDA_CHECK(cudaMalloc(&d_layers_g, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_layers_c, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_grad_mag_sq, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_grad_mag_max, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_trav_cost, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_inflated_cost, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_interval, sizeof(float) * total));
  CUDA_CHECK(cudaMalloc(&d_points, sizeof(float3) * points.size()));

  std::vector<float3> h_points(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    h_points[i] = make_float3(points[i].x(), points[i].y(), points[i].z());
  }
  CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
                        sizeof(float3) * h_points.size(),
                        cudaMemcpyHostToDevice));

  ClearKernel<<<blocks, threads>>>(d_layers_g, d_layers_c, d_grad_mag_sq,
                                   d_grad_mag_max, d_trav_cost, d_inflated_cost,
                                   d_interval, total);
  CUDA_CHECK(cudaDeviceSynchronize());

  TomographyKernel<<<blocks_points, threads>>>(
      d_points, static_cast<int>(points.size()), d_layers_g, d_layers_c, cx, cy,
      static_cast<float>(params.resolution), static_cast<int>(dim_x),
      static_cast<int>(dim_y), static_cast<int>(n_slice), slice_h0,
      static_cast<float>(params.slice_dh));
  CUDA_CHECK(cudaDeviceSynchronize());

  GradIntervalKernel<<<blocks, threads>>>(
      d_layers_g, d_layers_c, d_grad_mag_sq, d_grad_mag_max, d_interval,
      static_cast<int>(dim_x), static_cast<int>(dim_y),
      static_cast<int>(n_slice));
  CUDA_CHECK(cudaDeviceSynchronize());

  int half_trav_k = params.kernel_size / 2;
  float step_stand = 1.2f * static_cast<float>(params.resolution) *
                     std::tan(static_cast<float>(params.slope_max));
  float step_cross = static_cast<float>(params.step_max);
  int standable_th = static_cast<int>(params.standable_ratio *
                                      std::pow((2 * half_trav_k + 1), 2)) -
                     1;
  if (standable_th < 0)
    standable_th = 0;

  TravKernel<<<blocks, threads>>>(
      d_interval, d_grad_mag_sq, d_grad_mag_max, d_trav_cost,
      static_cast<int>(dim_x), static_cast<int>(dim_y),
      static_cast<int>(n_slice), half_trav_k,
      static_cast<float>(params.interval_min),
      static_cast<float>(params.interval_free), step_cross * step_cross,
      step_stand * step_stand, standable_th,
      static_cast<float>(params.cost_barrier));
  CUDA_CHECK(cudaDeviceSynchronize());

  int half_inf_k = static_cast<int>((params.safe_margin + params.inflation) /
                                    params.resolution);
  int score_size = (2 * half_inf_k + 1) * (2 * half_inf_k + 1);
  std::vector<float> h_score(score_size, 0.0f);
  for (int i = 0; i < 2 * half_inf_k + 1; ++i) {
    for (int j = 0; j < 2 * half_inf_k + 1; ++j) {
      float dx = params.resolution * static_cast<float>(i - half_inf_k);
      float dy = params.resolution * static_cast<float>(j - half_inf_k);
      float dist = std::sqrt(dx * dx + dy * dy);
      float val =
          1.0f -
          (dist - static_cast<float>(params.inflation)) /
              (static_cast<float>(params.safe_margin + params.resolution));
      if (val < 0.0f)
        val = 0.0f;
      if (val > 1.0f)
        val = 1.0f;
      h_score[i * (2 * half_inf_k + 1) + j] = val;
    }
  }
  CUDA_CHECK(cudaMalloc(&d_score, sizeof(float) * score_size));
  CUDA_CHECK(cudaMemcpy(d_score, h_score.data(), sizeof(float) * score_size,
                        cudaMemcpyHostToDevice));

  InflationKernel<<<blocks, threads>>>(
      d_trav_cost, d_score, d_inflated_cost, static_cast<int>(dim_x),
      static_cast<int>(dim_y), static_cast<int>(n_slice), half_inf_k);
  CUDA_CHECK(cudaDeviceSynchronize());

  out.trav_cost.resize(total);
  out.inflated_cost.resize(total);
  out.interval.resize(total);
  out.grad_mag_sq.resize(total);
  out.grad_mag_max.resize(total);
  out.elev_g.resize(total);
  out.elev_c.resize(total);

  CUDA_CHECK(cudaMemcpy(out.trav_cost.data(), d_trav_cost,
                        sizeof(float) * total, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.inflated_cost.data(), d_inflated_cost,
                        sizeof(float) * total, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.interval.data(), d_interval, sizeof(float) * total,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.grad_mag_sq.data(), d_grad_mag_sq,
                        sizeof(float) * total, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.grad_mag_max.data(), d_grad_mag_max,
                        sizeof(float) * total, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.elev_g.data(), d_layers_g, sizeof(float) * total,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.elev_c.data(), d_layers_c, sizeof(float) * total,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_layers_g);
  cudaFree(d_layers_c);
  cudaFree(d_grad_mag_sq);
  cudaFree(d_grad_mag_max);
  cudaFree(d_trav_cost);
  cudaFree(d_inflated_cost);
  cudaFree(d_interval);
  cudaFree(d_points);
  cudaFree(d_score);

  return true;
}

} // namespace pctplanner
