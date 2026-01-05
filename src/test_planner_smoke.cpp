// 文件作用：离线最小地图的冒烟测试，构造 1 个切片 10x10 的
// tomogram，调用规划器验证能生成轨迹。
#include <Eigen/Dense>
#include <cstring>
#include <iostream>
#include <vector>

#include "ele_planner/offline_ele_planner.h"
#include "tomogram_format.hpp"

using namespace pctplanner;

static uint16_t fp16(float v) {
  // float -> fp16 辅助
  Eigen::half h = static_cast<Eigen::half>(v);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(uint16_t));
  return bits;
}

int main() {
  const uint32_t n_slice = 1;
  const uint32_t dim_x = 10;
  const uint32_t dim_y = 10;
  const float res = 0.2f;
  // 构造最小 header + 全零代价，顶面=1.0m
  TomogramHeader h = tomogram_format::MakeHeader(n_slice, dim_x, dim_y, res,
                                                 0.0f, 0.0f, 0.0f, 0.2f);
  std::vector<uint16_t> slice_heights_fp16(n_slice, fp16(0.0f));
  const size_t plane = dim_x * dim_y;
  std::vector<uint16_t> data_fp16(plane * tomogram_format::kLayersPerVoxel,
                                  fp16(0.0f));
  for (size_t idx = 0; idx < plane; ++idx)
    data_fp16[idx + 4 * plane] = fp16(1.0f);

  auto payload = tomogram_format::Serialize(h, slice_heights_fp16, data_fp16);
  auto view = tomogram_format::Deserialize(payload);

  Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(n_slice * dim_y, dim_x);
  Eigen::MatrixXd gx = Eigen::MatrixXd::Zero(n_slice * dim_y, dim_x);
  Eigen::MatrixXd gy = Eigen::MatrixXd::Zero(n_slice * dim_y, dim_x);
  Eigen::MatrixXd eg = Eigen::MatrixXd::Zero(n_slice * dim_y, dim_x);
  Eigen::MatrixXd ec = Eigen::MatrixXd::Constant(n_slice * dim_y, dim_x, 1.0);
  Eigen::MatrixXi gateway = Eigen::MatrixXi::Zero(n_slice * dim_y, dim_x);

  // 初始化规划器并规划从 (1,mid)->(dim_x-2,mid)
  OfflineElePlanner planner(10.0, true);
  Eigen::MatrixXd gateway_d = gateway.cast<double>();
  planner.InitMap(20.0, 15.0, res, n_slice, 0.2, cost, eg, ec, gateway_d, gy,
                  -gx);
  Eigen::Vector3i s(0, dim_y / 2, 1);
  Eigen::Vector3i g(0, dim_y / 2, dim_x - 2);
  Eigen::Ref<const Eigen::Vector3i> s_ref(s);
  Eigen::Ref<const Eigen::Vector3i> g_ref(g);
  if (!planner.Plan(s_ref, g_ref, true)) {
    std::cerr << "Plan failed" << std::endl;
    return 1;
  }
  auto traj = planner.get_trajectory_optimizer_wnoj().GetResultMatrix();
  if (traj.rows() == 0) {
    std::cerr << "Empty trajectory" << std::endl;
    return 1;
  }
  std::cout << "Smoke test OK, traj rows: " << traj.rows() << std::endl;
  return 0;
}
