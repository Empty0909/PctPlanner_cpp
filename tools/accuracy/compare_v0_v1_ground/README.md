# v0 vs v1 地图路径规划对比测试

对比 `nyby_ground_cost_map_v0.bin` 和 `nyby_ground_cost_map_v1.bin` 两个版本地图的规划结果差异。

## 重要发现

通过分析，两张地图的坐标系关系如下：

### 坐标变换参数（基于3点手动标记）

| 参数 | 值 | 说明 |
|------|-----|------|
| θ | 15.778° | 旋转角度 |
| tx | -11.537 m | X 方向平移 |
| ty | -2.216 m | Y 方向平移 |
| kx_z | 0.001921 | Z 与 x0 的系数 |
| ky_z | -0.003825 | Z 与 y0 的系数 |
| c_z | -0.141557 | Z 偏移常数 |

### 变换公式

```
XY 变换: v1 = R(θ) @ v0 + T
  x1 = cos(θ) * x0 - sin(θ) * y0 + tx
  y1 = sin(θ) * x0 + cos(θ) * y0 + ty

Z 变换 (平面拟合):
  z1 = z0 + kx * x0 + ky * y0 + c_z
```

### 对齐统计

| 指标 | 无变换 | 应用变换后 |
|------|--------|-----------|
| 二值一致率 | 67.0% | **82.3%** |
| Traversability 精确匹配 (|Δt|<1) | 58.8% | **79.2%** |
| Traversability RMSE | 21.46 | **8.99** |

### 路径长度对比

| 指标 | 值 |
|------|-----|
| v0 平均路径长度 | 72.34 m |
| v1 平均路径长度 | 74.83 m |
| v1/v0 平均比值 | 1.033 (v1 比 v0 长 3.3%) |
| 路径长度相关系数 | 0.9956 |

## 目录结构

```
compare_v0_v1_ground/
├── README.md                      # 本文档
├── batch_compare_simple.py        # 主脚本：v0 vs v1 对比测试
├── compute_transform_from_points.py  # 从标记点计算变换
├── generate_v0_test_cases.py      # 辅助脚本：生成测试用例
└── results/
    └── compare_3point_transform.json  # 最终测试结果
```

## 快速开始

### 运行 v0 vs v1 对比测试

```bash
cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy

# 设置库路径
export LD_LIBRARY_PATH=$PWD/../../planner_lib:$PWD/../../planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH

# 运行对比测试（应用坐标变换）
python3 compare_v0_v1_ground/batch_compare_simple.py \
    --csv data/nyby_v0_test_pairs.csv \
    --v0 ../../rsc/tomogram/nyby_ground_cost_map_v0.bin \
    --v1 ../../rsc/tomogram/nyby_ground_cost_map_v1.bin \
    --transform \
    --output compare_v0_v1_ground/results/comparison.json
```

### 生成新的测试用例（可选）

```bash
python3 compare_v0_v1_ground/generate_v0_test_cases.py \
    --num-cases 100 \
    --output data/new_test_cases.csv
```

## 脚本说明

### batch_compare_simple.py

**功能**：在 v0 和 v1 地图上分别规划，对比成功率和路径长度

**参数**：

| 参数 | 说明 | 必需 |
|------|------|------|
| `--csv` | 测试用例 CSV 文件 | 是 |
| `--v0` | v0 地图 bin 文件路径 | 是 |
| `--v1` | v1 地图 bin 文件路径 | 是 |
| `--transform` | 应用 v0->v1 坐标变换 | 否 |
| `--output` | 输出 JSON 文件 | 否 |
| `--limit` | 限制测试用例数（0=全部） | 否 |

**最新测试结果（应用变换后）**：

```
总用例数: 110
v0 成功: 110 (100.0%)
v1 成功: 110 (100.0%)
两者都成功: 110

路径长度对比:
  v0 平均: 72.34 m
  v1 平均: 74.83 m
  v1/v0 比值: 1.034
```

### compute_transform_from_points.py

**功能**：根据用户手动标记的对应点计算 v0 -> v1 坐标变换

**方法**：使用 Procrustes analysis (SVD) 计算最优刚性变换

**参考点**（在 rviz2 中标记）：

| 点 | v0 坐标 | v1 坐标 |
|---|---------|---------|
| 1 | (-47.3, -1.94, -0.101) | (-56.5, -16.8, -0.326) |
| 2 | (9.86, -105.0, 0.416) | (26.6, -101.0, 0.695) |
| 3 | (40.6, 6.65, -0.276) | (25.6, 15.5, -0.365) |

### generate_v0_test_cases.py

**功能**：在 v0 地图上随机采样可通行栅格生成测试用例

## 结论

1. **v0 和 v1 地图存在坐标变换**：旋转 15.78° + 平移 + Z 平面倾斜
2. **应用变换后规划成功率一致**：两者都达到 100%
3. **路径长度高度相关**（r=0.9956），v1 平均比 v0 长 3.3%
4. **地图内容一致性约 82%**，剩余差异来自不同采集/处理
