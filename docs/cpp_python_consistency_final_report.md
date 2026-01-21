# PctPlanner C++/Python 版本一致性修复 — 完整报告

**日期**: 2026年1月21日  
**版本**: v1.1 (Final)  
**状态**: ✅ 已完成

---

## 📋 执行摘要

本报告记录了 PctPlanner C++ 版本与 Python 版本代价地图（Tomogram）一致性问题的完整排查与修复过程。经过多轮调试，已实现 **100% 数值一致性**。

### 最终测试结果

| 层名 | 匹配率 | 最大差异 | 状态 |
|------|--------|----------|------|
| trav (通行代价) | 100.0000% | 3.9e-03 | ✅ |
| trav_gx (x方向梯度) | 100.0000% | 1.6e-02 | ✅ |
| trav_gy (y方向梯度) | 100.0000% | 3.1e-02 | ✅ |
| elev_g (地面高度) | 100.0000% | 0 | ✅ |
| elev_c (顶面高度) | 100.0000% | 0 | ✅ |

**总体匹配率: 35,129,666 / 35,129,666 = 100.0000%**

---

## 🔍 发现的问题

### 问题 1: TomographyKernel 半精度减法精度问题

#### 问题描述

C++ 的 CUDA kernel 使用 `__hsub()` 进行半精度 (half/float16) 减法，而 Python/CuPy 的 `float16` 类在进行算术运算时会**隐式提升到 float32 精度**。

#### 根本原因

CuPy 的 `float16` 类定义中包含隐式类型转换运算符：

```cpp
// cupy/_core/include/cupy/carray.cuh
class float16 {
    __device__ operator float() const { return float(data_); }
};
```

当执行 `x - center` 时：

- **Python/CuPy**: 触发 `operator float()`，转换为 `float(x) - float(center)` → float32 精度
- **C++ (修复前)**: 使用 `__hsub(px_h, cx_h)` → half 精度（仅 10 位尾数）

#### 精度差异示例

| 输入值 | Python 结果 | C++ 结果 (修复前) | 索引偏差 |
|--------|-------------|-------------------|----------|
| px=-23.83, cx=41.51 | -65.328125 | -65.3125 | 1 |
| px=48.13, cx=-33.40 | 81.53125 | 81.5000 | 1 |

约 **6.45%** 的点受此影响，被分配到错误的网格索引。

#### 修复方案

**文件**: `src/tomography_cuda.cu`  
**修改**: 将半精度减法改为先转换到 float 再相减

```cuda
// 修复前
float diff_x = __half2float(__hsub(px_h, cx_h));
float diff_y = __half2float(__hsub(py_h, cy_h));

// 修复后
float diff_x = __half2float(px_h) - __half2float(cx_h);
float diff_y = __half2float(py_h) - __half2float(cy_h);
```

---

### 问题 2: 梯度计算边界条件错误

#### 问题描述

修复问题 1 后，`trav_gx` 和 `trav_gy` 层在边界位置仍存在差异：

- `trav_gx`: 在 y 边界 (y=0 和 y=dim_y-1) 错误地为 0
- `trav_gy`: 在 x 边界 (x=0 和 x=dim_x-1) 错误地为 0

#### 根本原因

Python 代码的梯度计算逻辑：

```python
trav_grad_x = inflated_cost[:, 2:, :] - inflated_cost[:, :-2, :]
trav_grad_y = inflated_cost[:, :, 2:] - inflated_cost[:, :, :-2]

trav_gx[:, 1:-1, :] = trav_grad_x  # x边界为0，y边界有值
trav_gy[:, :, 1:-1] = trav_grad_y  # y边界为0，x边界有值
```

C++ 修复前的代码（错误）：

```cpp
// 错误：同时限制 x 和 y 到内部区域
for (uint32_t x = 1; x + 1 < dim_x; ++x) {
    for (uint32_t y = 1; y + 1 < dim_y; ++y) {  // ❌ y 也被限制了
        gx[idx] = cost[idx + dim_y] - cost[idx - dim_y];
        gy[idx] = cost[idx + 1] - cost[idx - 1];
    }
}
```

#### 修复方案

**文件**: `src/tomography_node.cpp`  
**函数**: `ComputeTravGradient`

```cpp
// 修复后：分离 gx 和 gy 的循环，各自有独立的边界条件

// gx: x=1 到 x=dim_x-2，所有 y 位置（包括 y 边界）
for (uint32_t x = 1; x + 1 < dim_x; ++x) {
    for (uint32_t y = 0; y < dim_y; ++y) {
        const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
        gx[idx] = cost[idx + dim_y] - cost[idx - dim_y];
    }
}

// gy: 所有 x 位置（包括 x 边界），y=1 到 y=dim_y-2
for (uint32_t x = 0; x < dim_x; ++x) {
    for (uint32_t y = 1; y + 1 < dim_y; ++y) {
        const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
        gy[idx] = cost[idx + 1] - cost[idx - 1];
    }
}
```

---

### 问题 3: 项目构建系统混乱

#### 问题描述

项目中存在两套 build/install 目录：

- `/home/lzy/PctPlanner/build/` — 父目录级别（错误位置）
- `/home/lzy/PctPlanner/PctPlanner_Cpp/build/` — 项目目录内（正确位置）

ROS2 launch 使用 `install/` 目录中的可执行文件，但开发时在错误的位置编译，导致修复后的代码未被正确使用。

#### 修复方案

1. 删除父目录的 build/install/log 目录
2. 统一在 `PctPlanner_Cpp/` 目录内使用 `colcon build`
3. 更新 README 文档，明确编译规范

---

## 📁 修改的文件清单

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `src/tomography_cuda.cu` | Bug Fix | 半精度减法改为 float 减法 |
| `src/tomography_node.cpp` | Bug Fix | 梯度边界条件分离处理 |
| `README_CN.md` | 文档更新 | 项目结构说明完善 |
| `.gitignore` | 无变更 | 已正确配置 |

---

## 🔧 编译与验证步骤

### 编译命令

```bash
cd ~/PctPlanner/PctPlanner_Cpp

# 清理旧编译（可选）
rm -rf build/ install/ log/

# 编译
colcon build --packages-select pct_planner_cpp_port \
  --cmake-args \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.8

# 加载环境
source install/setup.bash
export LD_LIBRARY_PATH=$PWD/planner_lib:$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
```

### 验证命令

```bash
# 生成代价地图
ros2 launch pct_planner_cpp_port pct_all.launch.py \
  params_file:=$PWD/config/scene_default.yaml \
  publish_start_end:=true \
  start_x:=5.63 start_y:=15 start_z:=0 \
  end_x:=-9.68 end_y:=6.95 end_z:=0

# 对比测试
cd tools/rsc_compare
python3 tomogram_compare.py \
  --bin ../../rsc/tomogram/scene_map.bin \
  --pickle /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle
```

---

## 📊 技术深度分析

### CuPy float16 隐式转换机制

CuPy 的 `float16` 类（位于 `cupy/_core/include/cupy/carray.cuh`）设计了隐式转换运算符：

```cpp
class float16 {
private:
    half data_;
public:
    __device__ float16() {}
    __device__ float16(float v) : data_(v) {}
    __device__ operator float() const { return float(data_); }
    // 注意：没有重载 operator-，所以减法会触发隐式转换
};
```

当执行 `a - b`（两个 float16）时，编译器的隐式转换规则：

1. 没有 `float16 operator-(float16, float16)`
2. 存在 `operator float()`，可以转换到 float
3. 使用 `float operator-(float, float)`

这导致所有 float16 算术运算实际上是在 float32 精度下进行的。

### CUDA 半精度运算精度对比

| 精度类型 | 尾数位数 | 精度范围 | 累计误差风险 |
|----------|----------|----------|--------------|
| half (__hsub) | 10 bits | ~0.001 | 高 |
| float | 23 bits | ~1e-7 | 低 |
| double | 52 bits | ~1e-16 | 极低 |

---

## ✅ 验收标准

1. ✅ 所有 5 个层 100% 匹配
2. ✅ NaN 分布完全一致
3. ✅ 边界值处理正确
4. ✅ 编译流程统一规范
5. ✅ 文档更新完成

---

## 📝 经验教训与建议

### 教训

1. **跨语言移植要注意隐式类型转换**：Python/CuPy 的运算符重载可能有隐藏的类型提升行为
2. **构建系统要统一**：多个 build 目录会导致混乱
3. **边界条件要独立测试**：梯度计算的边界处理容易出错

### 建议

1. 添加自动化回归测试，确保 C++/Python 版本一致性
2. 考虑在 CI/CD 中加入对比测试
3. 关键算法添加单元测试覆盖边界情况

---

## 📎 相关文档

- [tomography_precision_fix_report.md](tomography_precision_fix_report.md) — 精度修复详细报告
- [README_CN.md](../README_CN.md) — 项目使用说明

---

**报告完成**  
如有问题，请联系开发团队。
