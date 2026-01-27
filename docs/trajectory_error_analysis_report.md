# PctPlanner C++/Python 版本轨迹误差分析报告

**日期**: 2026年1月26日  
**分析者**: GitHub Copilot  
**测试规模**: 10,000 组规划用例

---

## 1. 问题概述

在对比测试 C++ 版本和 Python 版本 PctPlanner 规划器时，发现部分测试用例存在**最大 109 米的轨迹误差**。经过深入调试，定位到问题根源。

### 1.1 现象描述

| 指标 | Python 版本 | C++ 版本 |
|------|-------------|----------|
| 成功率 | 64.06% | 64.21% |
| 平均时间 | 176.19 ms | 173.79 ms |
| **轨迹误差** | - | **最大 109.40 m** |

异常轨迹的 Z 坐标呈现 **-99.4** 的特征值，明显不合理。

---

## 2. 根本原因分析

### 2.1 直接原因：优化器返回越界坐标

以测试用例 **#6084** 为例：

| 版本 | heights 范围 | 异常点数量 | 优化 cost |
|------|--------------|------------|-----------|
| Python (pickle) | [-99.90, 9.86] | 11 | 529,219,188 |
| C++ (bin) | [-1.00, 9.66] | 0 | 223,198,770 |

Python 版本在轨迹索引 259-272 处出现异常：

```
索引 259: traj_raw = [130.99, ..., -20.14, ...]  ← y坐标为负数，越界！
索引 260: traj_raw = [127.48, ..., -21.33, ...]  ← y坐标为负数，越界！
索引 261: traj_raw = [120.98, ..., -16.19, ...]  ← y坐标为负数，越界！
```

**越界后果**：

- `GetHeight()` 调用 `index_y_safe(y)` 将负数 y 钳制到边界值 0
- 边界位置的 `elev_g` 为 NaN（被替换为 -100）
- 最终返回 **-99.9** 作为无效高度标记
- 转换为世界坐标后：`-99.9 / 0.15 * 0.15 + 0.5 ≈ -99.4`

### 2.2 根本原因：Tomogram 数据不一致

pickle 文件和 bin 文件虽然存储相同场景，但存在**微小的数值差异**：

| 对比项 | 结果 |
|--------|------|
| 总元素数 | 48,943,980 |
| 不同的 fp16 值 | **8,214 个** (0.017%) |
| 差异幅度 | 1 个最低有效位 (LSB) |

差异示例：

```
位置 [trav,0,158,612]: pickle=0x267a (0.0253) vs bin=0x2679 (0.0252)
位置 [trav,0,322,96]:  pickle=0x2e7a (0.1012) vs bin=0x2e79 (0.1011)
```

### 2.3 差异放大链

```
Tomogram 数据微小差异 (0.017%)
    ↓
A* 路径搜索选择不同路径 (9 个点差异)
    ↓
GTSAM 优化器收敛到不同局部最优
    ↓
部分轨迹坐标越界 (y < 0)
    ↓
GetHeight() 返回 -99.9 (无效标记)
    ↓
最终轨迹误差 109 米
```

---

## 3. 验证实验

### 3.1 控制变量实验

使用**相同的 bin 文件**分别在 Python 和 C++ 环境运行：

| 数据源 | 异常点 | heights 范围 | cost |
|--------|--------|--------------|------|
| C++ bin 文件 (Python 环境) | **0** | [-1.00, 9.66] | 223,198,770 |
| C++ bin 文件 (C++ 环境) | **0** | [-1.00, 9.66] | 223,198,770 |
| Python pickle 文件 (Python 环境) | **11** | [-99.90, 9.86] | 529,219,188 |

**结论**：使用相同数据时，结果完全一致。问题在于 pickle 和 bin 文件数据不一致。

### 3.2 库文件一致性验证

| 库文件 | MD5 (Python) | MD5 (C++) | 一致 |
|--------|--------------|-----------|------|
| ele_planner.cpython-38*.so | 1d27e69f... | 1d27e69f... | ✅ |
| a_star.cpython-38*.so | c6f62787... | c6f62787... | ✅ |
| traj_opt.cpython-38*.so | 0426d381... | 0426d381... | ✅ |

库文件完全一致，排除代码差异。

---

## 4. 问题分类

### 4.1 数据同步问题

- **原因**：pickle 和 bin 文件由不同版本的 CUDA tomography 代码生成，fp32→fp16 转换时舍入方式不同
- **影响**：8,214 个数值存在 1 LSB 差异
- **后果**：A* 搜索结果不同

### 4.2 边界处理问题

`gpmp_optimizer_wnoa.cc` 中的边界检查逻辑：

```cpp
// dense_elevation_map.cc
if (std::isnan(elev_g)) {
    lower_height = -99;  // 无效标记
    upper_height = -99;
}
```

当轨迹越界时，返回 -99.9 但未做进一步处理，导致输出轨迹包含不合理的 Z 值。

---

## 5. 解决方案

### 5.1 短期方案（推荐）

**确保 pickle 和 bin 文件从同一源生成**：

```bash
# 使用转换工具从 bin 生成 pickle
python3 tools/rsc_compare/tomogram_convert.py bin2pickle \
    PctPlanner_Cpp/rsc/tomogram/scene_map.bin \
    PctPlanner_py/rsc/tomogram/scene_map.pickle

# 或者反向转换
python3 tools/rsc_compare/tomogram_convert.py pickle2bin \
    PctPlanner_py/rsc/tomogram/scene_map.pickle \
    PctPlanner_Cpp/rsc/tomogram/scene_map.bin
```

### 5.2 中期方案

**添加轨迹边界检查**：

在 `gpmp_optimizer_wnoa.cc` 中优化后添加边界检查：

```cpp
// 检查优化后的轨迹是否越界
for (int i = 0; i < N; ++i) {
    double x = trajectory_(i, 0);
    double y = trajectory_(i, 2);
    if (x < 0 || x >= max_x_ || y < 0 || y >= max_y_) {
        // 钳制到有效范围或标记为失败
        trajectory_(i, 0) = std::clamp(x, 0.0, max_x_ - 1.0);
        trajectory_(i, 2) = std::clamp(y, 0.0, max_y_ - 1.0);
    }
}
```

### 5.3 长期方案

**统一 Tomogram 生成流程**：

1. 由 C++ 版本 CUDA tomography 生成 bin 文件
2. 使用格式转换工具生成 pickle 文件供 Python 版本使用
3. 保证两个文件的数据完全一致

---

## 6. 测试建议

### 6.1 A* 路径对比测试

使用 `--no-optimize` 选项跳过优化器，单独验证 A* 搜索一致性：

```bash
# 关闭优化器测试
python3 run_batch.py --version cpp --no-optimize \
    --input data/test_cases_valid_ground.csv \
    --output results/cpp_astar_results.json --limit 10000

python3 run_batch.py --version python --no-optimize \
    --input data/test_cases_valid_ground.csv \
    --output results/python_astar_results.json --limit 10000
```

### 6.2 相同数据测试

使用相同的代价地图进行测试：

```bash
# Python 版本使用 C++ 的 bin 文件
python3 run_batch.py --version python \
    --tomo /path/to/scene_map.bin \
    --input data/test_cases_valid_ground.csv \
    --output results/python_with_bin.json --limit 10000
```

---

## 7. 结论

| 问题 | 状态 | 说明 |
|------|------|------|
| 库代码一致性 | ✅ 已验证 | MD5 完全一致 |
| Tomogram 数据一致性 | ❌ 有差异 | 8,214 个 fp16 值不同 |
| A* 路径一致性 | ❌ 有差异 | 数据差异导致路径不同 |
| 优化器边界处理 | ⚠️ 待优化 | 越界时返回 -99.9 |

**核心结论**：轨迹误差的根本原因是 **pickle 和 bin 文件数据不一致**，而非代码问题。通过统一数据源可彻底解决。

---

## 附录：关键代码位置

| 文件 | 行号 | 功能 |
|------|------|------|
| `dense_elevation_map.cc` | 98-99 | 定义无效高度标记 -99 |
| `gpmp_optimizer_wnoa.cc` | 213-215 | 调用 GetHeight() 获取高度 |
| `planner_wrapper.py` | 193-231 | Python 版本 plan() 实现 |
| `tomogram_convert.py` | - | 格式转换工具 |
