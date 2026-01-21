# PctPlanner C++ 版本问题分析报告

本文档总结了 C++ 版本与 Python 版本对比测试中发现的问题，供后续改进参考。

## 一、测试环境

- **测试用例数量**: 2000
- **Python 版本成功率**: 55.56%
- **C++ 版本成功率**: 55.62%
- **双方都成功**: 720 例

## 二、代价地图 (Tomography) 差异

### 2.1 整体统计

| 指标 | 值 |
|------|-----|
| 精确匹配 (<0.01) | 95.91% |
| 小差异 (0.01-1) | 1.90% |
| 中差异 (1-10) | 0.47% |
| 大差异 (10-50) | **1.71%** |
| 屏障差异 (>=50) | **0.01%** (4,564个体素) |
| NaN 不一致 | C++独有NaN: 21,983 / Py独有NaN: 21,081 |

### 2.2 按层类型统计

| 层 | 最大差异 | RMS | 匹配率 |
|----|---------|-----|--------|
| trav | 50.0 | 2.72 | 97.08% |
| trav_gx | 79.9 | 3.35 | 95.29% |
| trav_gy | 82.4 | 3.43 | 95.16% |
| elev_g | 13.3 | 0.17 | 97.04% |
| elev_c | 13.0 | 0.09 | 92.03% |

### 2.3 关键差异位置分析

#### 位置 (slice=0, x=99, y=660)
- **C++ trav = 50.0** (障碍)
- **Python trav = 0.0008** (可通行)
- **C++ elev_g = NaN**, Python elev_g = -0.98
- **原因**: C++ 认为该格子缺失地面点，标记为障碍

#### 位置 (slice=0, x=101, y=131)
- **C++ trav_gx = -49.97**, Python = +29.95
- **差异**: 符号相反，且绝对值不同

#### 位置 (slice=26, x=176, y=108)
- **C++ elev_g = 0.12**, Python = 13.46
- **差异**: 约 13m 的高程差

## 三、潜在问题根源

### 3.1 TomographyKernel (点云投影)

**文件**: `src/tomography_cuda.cu` 第 75-120 行

**可能问题**:
1. **半精度计算差异**: C++ 使用 `__half` 类型计算索引，Python 使用 `float16`
   - 不同硬件/驱动可能有舍入差异
   - 建议验证: 使用相同点云，对比两个版本的栅格投影结果

2. **边界条件处理**: 
   ```cpp
   if (ix < 0 || ix >= dim_x || iy < 0 || iy >= dim_y)
     return;
   ```
   - Python 版本可能有不同的边界判断

### 3.2 GradIntervalKernel (梯度计算)

**文件**: `src/tomography_cuda.cu` 第 120-170 行

**可能问题**:
1. **边界处理差异**:
   - C++ 在边界 (x=0, x=dim_x-1, y=0, y=dim_y-1) 设梯度为 0
   - Python 使用 NumPy 切片 `[:, 1:-1, 1:-1]`，自然跳过边界
   - 如果边界定义略有不同，可能导致差异

2. **缺失值处理**:
   - C++ 直接使用 `-1e6` 值参与梯度计算，产生巨大梯度
   - Python 在建图后将 `-1e6` 转为 NaN，但梯度计算可能不同

### 3.3 TravKernel (通行代价计算)

**文件**: `src/tomography_cuda.cu` 第 173-224 行

**逻辑对比**: 两个版本的 TravKernel 逻辑基本一致，主要差异来自输入数据

### 3.4 层简化 (Layer Simplification)

**C++ 位置**: `src/tomography_node.cpp` 第 175-225 行
**Python 位置**: `tomography/scripts/tomogram.py` 第 155-170 行

**可能问题**:
- idx_simp 选择逻辑可能有微小差异
- 导致简化后的层数或层索引不完全一致

## 四、规划器 (Planner) 差异

### 4.1 heights 数组异常 (8 个用例)

**受影响用例**: 355, 651, 1029, 1162, 1398, 1601, 1695, 1971

**现象**:
- 轨迹在某个点后 Z 值突变
- 例: 用例 1029，Python Z 范围 [-99.40, -99.40]，C++ Z 范围 [-99.40, 2.09]

**相关代码**: `tools/benchmark/run_cpp_version.py` 第 260-265 行
```python
heights = optimizer.get_heights()
traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
```

**可能原因**:
1. `optimizer.get_heights()` 返回的数组长度与轨迹点数不匹配
2. 轨迹优化器在某些路径上没有正确填充 heights 数组
3. C++ 轨迹优化器的内部实现与 Python 有差异

### 4.2 批次崩溃

**崩溃批次**: 4-5, 10-11, 13, 17-19 等

**可能原因**:
1. 特定用例的输入坐标导致越界访问
2. C++ 库的内存泄漏在多次规划后导致 OOM
3. 某些边界条件未正确处理

## 五、建议排查步骤

### 5.1 建图层排查

1. **单点验证**:
   ```bash
   # 选取一个差异显著的格子坐标，验证点云投影
   python3 tools/debug_single_cell.py --slice 0 --x 99 --y 660
   ```

2. **半精度计算验证**:
   - 在 C++ TomographyKernel 中添加调试输出
   - 对比 Python 版本的索引计算

3. **梯度计算验证**:
   - 导出两个版本的 layers_g 原始值
   - 手工计算梯度验证

### 5.2 规划层排查

1. **heights 数组验证**:
   ```cpp
   // 在 C++ 轨迹优化器中添加
   printf("traj_length=%d, heights_length=%d\n", traj.rows(), heights.size());
   ```

2. **崩溃用例定位**:
   - 使用二分法缩小崩溃用例范围
   - 启用 AddressSanitizer 编译检测内存问题

### 5.3 端到端验证

1. **使用相同代价地图**:
   - 让 C++ 规划器读取 Python 生成的 pickle 文件
   - 如果结果一致，说明问题在建图层
   - 如果仍有差异，说明问题在规划层

## 六、关键文件清单

| 功能 | C++ 文件 | Python 文件 |
|------|----------|-------------|
| 点云投影 | `src/tomography_cuda.cu` TomographyKernel | `tomography/scripts/kernels.py` tomographyKernel |
| 梯度计算 | `src/tomography_cuda.cu` GradIntervalKernel | `tomography/scripts/tomogram.py` (NumPy切片) |
| 通行代价 | `src/tomography_cuda.cu` TravKernel | `tomography/scripts/kernels.py` travKernel |
| 代价膨胀 | `src/tomography_cuda.cu` InflationKernel | `tomography/scripts/kernels.py` inflationKernel |
| 层简化 | `src/tomography_node.cpp` | `tomography/scripts/tomogram.py` |
| 输出序列化 | `src/tomography_node.cpp` FillTomogramPayload | `tomography/scripts/tomogram.py` |
| 规划入口 | `planner_lib/ele_planner.so` | `planner/lib/ele_planner.so` |

## 七、后续工作

1. [ ] 修复 heights 数组长度不匹配问题
2. [ ] 统一 NaN 处理逻辑
3. [ ] 验证半精度计算的一致性
4. [ ] 添加更详细的单元测试
5. [ ] 实现端到端一致性验证工具
