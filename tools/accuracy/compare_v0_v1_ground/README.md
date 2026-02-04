# v0 vs v1 地图路径规划对比测试

本目录包含用于对比 `nyby_ground_cost_map_v0.bin` 和 `nyby_ground_cost_map_v1.bin` 两个版本地图规划结果差异的工具脚本。

## 目录结构

```
compare_v0_v1_ground/
├── README.md                     # 本文档
├── run_v0_v1_compare_batch.py    # 主脚本：分批运行 v0 vs v1 对比测试
├── plan_on_map_batch.py          # 工作脚本：单批次规划（被主脚本调用）
└── generate_v0_test_cases.py     # 辅助脚本：在 v0 上生成有效测试用例
```

## 快速开始

### 1. 使用现有测试用例对比 v0 和 v1

```bash
cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy

python3 compare_v0_v1_ground/run_v0_v1_compare_batch.py \
    --input data/nyby_ground_test_cases.csv \
    --v0-bin /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v0.bin \
    --v1-bin /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v1.bin \
    --output results/v0_v1_comparison.json \
    --batch-size 50 \
    --limit 1000
```

### 2. 生成新的测试用例（可选）

如果需要在 v0 地图上生成新的测试用例（随机采样可通行栅格）：

```bash
cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy/compare_v0_v1_ground

python3 generate_v0_test_cases.py \
    --num-cases 1000 \
    --output ../data/v0_test_cases.csv
```

## 脚本说明

### run_v0_v1_compare_batch.py（主脚本）

**功能**：分批对比测试 v0 和 v1 地图的规划结果

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入测试用例 CSV 文件 | `data/nyby_ground_test_cases.csv` |
| `--v0-bin` | v0 地图 bin 文件路径 | `.../nyby_ground_cost_map_v0.bin` |
| `--v1-bin` | v1 地图 bin 文件路径 | `.../nyby_ground_cost_map_v1.bin` |
| `--output` | 输出 JSON 结果文件 | 自动生成带时间戳的文件名 |
| `--batch-size` | 每批处理的用例数 | 20 |
| `--limit` | 最大测试用例数 | 全部 |

**输出**：

- JSON 格式的详细结果文件
- Markdown 格式的统计报告（自动生成）

**工作原理**：

1. 将测试用例分成小批次
2. 每批在独立子进程中运行（避免内存泄漏导致 OOM）
3. 分别在 v0 和 v1 地图上规划
4. 合并结果并生成统计报告

---

### plan_on_map_batch.py（工作脚本）

**功能**：在指定地图上规划单批测试用例

**注意**：此脚本通常由 `run_v0_v1_compare_batch.py` 自动调用，一般不需要手动执行。

**参数**：

| 参数 | 说明 |
|------|------|
| `--input` | 输入 CSV 文件 |
| `--output` | 输出 JSON 文件 |
| `--bin` | 地图 bin 文件路径 |

---

### generate_v0_test_cases.py（辅助脚本）

**功能**：在 v0 地图上随机采样可通行栅格生成测试用例（不验证路径可达性）

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-cases` | 生成的用例数 | 1000 |
| `--output` | 输出 CSV 文件路径 | `../data/v0_test_cases.csv` |
| `--bin` | 地图 bin 文件路径 | `.../nyby_ground_cost_map_v0.bin` |
| `--min-distance` | 起终点最小距离（米） | 20.0 |
| `--max-distance` | 起终点最大距离（米） | 300.0 |
| `--seed` | 随机种子 | 42 |

**输出格式**（CSV）：

```
case_id,start_x,start_y,start_z,start_yaw,goal_x,goal_y,goal_z,goal_yaw
```

## 测试用例格式

输入 CSV 文件需要包含以下列：

- `case_id`：用例 ID
- `start_x`, `start_y`, `start_z`：起点坐标
- `start_yaw`：起点朝向（弧度）
- `goal_x`, `goal_y`, `goal_z`：终点坐标
- `goal_yaw`：终点朝向（弧度）

可选列：

- `v0_length`：v0 上的路径长度（如果已经计算过）

## 结果解读

统计报告包含以下指标：

| 指标 | 说明 |
|------|------|
| 两者都成功 | v0 和 v1 都规划成功的用例数 |
| 仅 v0 成功 | 只有 v0 规划成功（v1 失败）的用例数 |
| 仅 v1 成功 | 只有 v1 规划成功（v0 失败）的用例数 |
| 两者都失败 | v0 和 v1 都规划失败的用例数 |
| 平均路径长度差异 | 两者都成功时，v1 路径长度 - v0 路径长度的平均值 |

## 注意事项

1. **C++ 库验证**：脚本会自动验证加载的是 C++ 版本的规划库（位于 `PctPlanner_Cpp/planner_lib/`），而非 Python 版本。如果加载错误版本，脚本会报错退出。

2. **内存使用**：脚本采用分批子进程方式运行，避免内存泄漏导致 OOM。建议 `--batch-size` 保持在 20-50 之间。

3. **运行时间**：每批次约需 30-60 秒，1000 个用例约需 10-20 分钟。

4. **测试用例来源**：测试用例是在地图上随机采样可通行区域生成的，**没有验证路径可达性**。因此：
   - 规划成功率预期约为 10-20%
   - 这是正常现象，因为随机起终点不保证存在可行路径

5. **无需设置环境变量**：`generate_v0_test_cases.py` 只读取 bin 文件，不需要加载规划库。
