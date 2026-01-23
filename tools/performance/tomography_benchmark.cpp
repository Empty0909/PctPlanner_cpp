/**
 * @file tomography_benchmark.cpp
 * @brief 独立 Tomography CUDA 性能测试程序（无 ROS2 依赖）
 *
 * 编译：
 *   g++ -o tomography_benchmark tomography_benchmark.cpp \
 *       -I/path/to/eigen3 -I../include \
 *       -L../build -ltomography_cuda \
 *       -lpcl_common -lpcl_io -O3
 *
 * 运行：
 *   ./tomography_benchmark --pcd /path/to/map.pcd --runs 10
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// PCL headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Eigen
#include <Eigen/Core>

// CUDA tomography header
#include "tomography_cuda.hpp"

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct BenchmarkStats {
  std::vector<double> pcd_load;
  std::vector<double> cuda_init;
  std::vector<double> cuda_kernel;
  std::vector<double> cuda_total;
  std::vector<double> total;
};

void PrintStats(const std::string &name, const std::vector<double> &data) {
  if (data.empty())
    return;
  double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  double sq_sum =
      std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
  double std_dev =
      data.size() > 1
          ? std::sqrt((sq_sum - data.size() * mean * mean) / (data.size() - 1))
          : 0.0;
  double min_val = *std::min_element(data.begin(), data.end());
  double max_val = *std::max_element(data.begin(), data.end());

  std::cout << std::setw(25) << std::left << name << std::setw(12) << std::fixed
            << std::setprecision(2) << mean << std::setw(12) << std_dev
            << std::setw(12) << min_val << std::setw(12) << max_val
            << std::endl;
}

int main(int argc, char **argv) {
  std::string pcd_path = "../../rsc/pcd/map.pcd";
  int num_runs = 10;
  int warmup_runs = 3;

  // 简单参数解析
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--pcd" && i + 1 < argc) {
      pcd_path = argv[++i];
    } else if (arg == "--runs" && i + 1 < argc) {
      num_runs = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup_runs = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--pcd <path>] [--runs <N>] [--warmup <M>]" << std::endl;
      return 0;
    }
  }

  std::cout << "==============================================================="
               "=======\n";
  std::cout << "C++/CUDA Tomography 精确性能测试\n";
  std::cout << "==============================================================="
               "=======\n";
  std::cout << "PCD 文件: " << pcd_path << std::endl;
  std::cout << "预热次数: " << warmup_runs << std::endl;
  std::cout << "测试次数: " << num_runs << std::endl;

  // 设置 CUDA 参数
  pctplanner::TravParams params{};
  params.resolution = 0.15;
  params.slice_dh = 0.5;
  params.interval_min = 0.50;
  params.interval_free = 0.60;
  params.slope_max = 1.0;
  params.step_max = 0.70;
  params.standable_ratio = 0.40;
  params.cost_barrier = 50.0;
  params.safe_margin = 0.10;
  params.inflation = 0.05;
  params.kernel_size = 5;

  float ground_h = 0.0f;

  // 预热
  std::cout << "\nGPU 预热中 (" << warmup_runs << " 次)..." << std::endl;
  for (int w = 0; w < warmup_runs; ++w) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
      std::cerr << "错误：无法加载 PCD 文件: " << pcd_path << std::endl;
      return 1;
    }

    std::vector<Eigen::Vector3f> points;
    points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
        points.emplace_back(pt.x, pt.y, pt.z);
      }
    }

    Eigen::Vector3f pmin = points.front();
    Eigen::Vector3f pmax = points.front();
    for (const auto &p : points) {
      pmin = pmin.cwiseMin(p);
      pmax = pmax.cwiseMax(p);
    }
    pmin.z() = ground_h;
    float slice_h0 = pmin.z() + static_cast<float>(params.slice_dh);

    uint32_t dim_x = static_cast<uint32_t>(
                         std::ceil((pmax.x() - pmin.x()) / params.resolution)) +
                     4;
    uint32_t dim_y = static_cast<uint32_t>(
                         std::ceil((pmax.y() - pmin.y()) / params.resolution)) +
                     4;
    uint32_t n_slice =
        std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(
                                  (pmax.z() - pmin.z()) / params.slice_dh)));
    float cx = 0.5f * (pmax.x() + pmin.x());
    float cy = 0.5f * (pmax.y() + pmin.y());

    pctplanner::GpuTomographyOutput gpu_out;
    pctplanner::RunTomographyCuda(points, n_slice, dim_x, dim_y, cx, cy,
                                  slice_h0, params, gpu_out);
  }

  // 正式测试
  BenchmarkStats stats;
  std::cout << "\n开始正式测试 (" << num_runs << " 次)..." << std::endl;

  uint32_t point_count = 0;
  uint32_t dim_x = 0, dim_y = 0, n_slice = 0;

  for (int run = 0; run < num_runs; ++run) {
    // 1. PCD 加载计时
    auto t0 = Clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
      std::cerr << "错误：无法加载 PCD 文件" << std::endl;
      return 1;
    }
    auto t1 = Clock::now();
    double pcd_load_time = std::chrono::duration_cast<Ms>(t1 - t0).count();
    stats.pcd_load.push_back(pcd_load_time);

    // 转换点云
    std::vector<Eigen::Vector3f> points;
    points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
        points.emplace_back(pt.x, pt.y, pt.z);
      }
    }
    point_count = static_cast<uint32_t>(points.size());

    // 计算边界框
    Eigen::Vector3f pmin = points.front();
    Eigen::Vector3f pmax = points.front();
    for (const auto &p : points) {
      pmin = pmin.cwiseMin(p);
      pmax = pmax.cwiseMax(p);
    }
    pmin.z() = ground_h;
    float slice_h0 = pmin.z() + static_cast<float>(params.slice_dh);

    dim_x = static_cast<uint32_t>(
                std::ceil((pmax.x() - pmin.x()) / params.resolution)) +
            4;
    dim_y = static_cast<uint32_t>(
                std::ceil((pmax.y() - pmin.y()) / params.resolution)) +
            4;
    n_slice =
        std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(
                                  (pmax.z() - pmin.z()) / params.slice_dh)));
    float cx = 0.5f * (pmax.x() + pmin.x());
    float cy = 0.5f * (pmax.y() + pmin.y());

    // 2. CUDA 计算计时 (包含 GPU 传输和 kernel)
    auto t2 = Clock::now();
    pctplanner::GpuTomographyOutput gpu_out;
    bool ok = pctplanner::RunTomographyCuda(points, n_slice, dim_x, dim_y, cx,
                                            cy, slice_h0, params, gpu_out);
    auto t3 = Clock::now();
    double cuda_total_time = std::chrono::duration_cast<Ms>(t3 - t2).count();
    stats.cuda_total.push_back(cuda_total_time);

    if (!ok) {
      std::cerr << "CUDA tomography 失败" << std::endl;
      return 1;
    }

    double total_time = pcd_load_time + cuda_total_time;
    stats.total.push_back(total_time);

    std::cout << "  Run " << (run + 1) << "/" << num_runs << ": "
              << "PCD=" << std::fixed << std::setprecision(1) << pcd_load_time
              << "ms, "
              << "CUDA=" << cuda_total_time << "ms, "
              << "Total=" << total_time << "ms" << std::endl;
  }

  // 打印统计结果
  std::cout << "\n============================================================="
               "=========\n";
  std::cout << "C++/CUDA 统计结果\n";
  std::cout << "==============================================================="
               "=======\n";
  std::cout << "点云数量: " << point_count << std::endl;
  std::cout << "地图维度: " << dim_x << "×" << dim_y << "×" << n_slice
            << std::endl;
  std::cout << "总像素数: " << (static_cast<size_t>(dim_x) * dim_y * n_slice)
            << std::endl;
  std::cout << "\n";

  std::cout << std::setw(25) << std::left << "阶段" << std::setw(12)
            << "平均(ms)" << std::setw(12) << "标准差" << std::setw(12)
            << "最小" << std::setw(12) << "最大" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "-------\n";
  PrintStats("PCD 加载", stats.pcd_load);
  PrintStats("CUDA 总时间", stats.cuda_total);
  PrintStats("端到端总时间", stats.total);

  // 计算平均值用于对比
  double avg_pcd =
      std::accumulate(stats.pcd_load.begin(), stats.pcd_load.end(), 0.0) /
      stats.pcd_load.size();
  double avg_cuda =
      std::accumulate(stats.cuda_total.begin(), stats.cuda_total.end(), 0.0) /
      stats.cuda_total.size();
  double avg_total =
      std::accumulate(stats.total.begin(), stats.total.end(), 0.0) /
      stats.total.size();

  std::cout << "\n============================================================="
               "=========\n";
  std::cout << "性能总结\n";
  std::cout << "==============================================================="
               "=======\n";
  std::cout << "PCD 加载: " << std::fixed << std::setprecision(2) << avg_pcd
            << " ms\n";
  std::cout << "CUDA 计算: " << avg_cuda << " ms\n";
  std::cout << "端到端: " << avg_total << " ms\n";

  return 0;
}
