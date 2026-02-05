# PctPlanner 精度测试工具

本目录包含用于测试和对比 PctPlanner 规划器的工具集。

## 目录结构

```
accuracy/
├── README.md                     # 本说明文件
├── run_batch.py                  # 批量运行 Python/C++ 版本测试
├── generate_test_cases.py        # 测试用例生成器
├── run_python_version.py         # Python 版本规划测试（被 run_batch.py 调用）
├── run_cpp_version.py            # C++ 版本规划测试（被 run_batch.py 调用）
├── compare_versions.py           # Python vs C++ 结果对比分析
├── compare_v0_v1_ground/         # v0 vs v1 地图对比测试
│   ├── batch_compare_simple.py   # v0 vs v1 对比主脚本
│   ├── generate_v0_test_cases.py # 生成测试用例
│   └── results/                  # 测试结果
├── data/                         # 测试数据目录
│   └── *.csv                     # 测试用例文件
└── results/                      # 测试结果目录
    └── *.json, *.md
```

## 测试类型

### 1. Python vs C++ 版本对比

对比 Python 版本和 C++ 版本规划器的结果差异（成功率、轨迹、性能）。

详见下方 "批量测试" 部分。

### 2. v0 vs v1 地图对比

对比不同版本地图的规划成功率。详见 [compare_v0_v1_ground/README.md](compare_v0_v1_ground/README.md)。

```bash
cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy
export LD_LIBRARY_PATH=$PWD/../../planner_lib:$PWD/../../planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
python3 compare_v0_v1_ground/batch_compare_simple.py \
    --csv data/nyby_v0_test_pairs.csv \
    --v0 ../../rsc/tomogram/nyby_ground_cost_map_v0.bin \
    --v1 ../../rsc/tomogram/nyby_ground_cost_map_v1.bin
```

## 快速开始

### 1. 生成测试用例

```bash
python3 generate_test_cases.py \
    --pickle /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle \
    --num-cases 10000 \
    --output-csv data/test_cases.csv
```

### 2. 运行批量测试

```bash
# C++ 版本测试
python3 run_batch.py --version cpp \
    --input data/test_cases.csv \
    --output results/cpp_results.json \
    --tomo /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/scene_map.bin

# Python 版本测试
python3 run_batch.py --version python \
    --input data/test_cases.csv \
    --output results/python_results.json \
    --tomo /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle
```

### 3. 对比结果

```bash
python3 compare_versions.py \
    --python results/python_results.json \
    --cpp results/cpp_results.json \
    --output results/comparison.md
```

## 核心功能

### 批量测试 (run_batch.py)

推荐使用批量模式运行测试，避免内存泄漏导致 OOM：

```bash
# C++ 版本测试
python3 run_batch.py --version cpp \
    --input data/nyby_underground_test_cases.csv \
    --output results/nyby_underground_cpp_results.json \
    --limit 10000 \
    --tomo /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_underground_cost_map.bin

# Python 版本测试
python3 run_batch.py --version python \
    --input data/nyby_underground_test_cases.csv \
    --output results/nyby_underground_python_results.json \
    --limit 10000 \
    --tomo /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/nyby_underground_cost_map.pickle
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--version` | 测试版本：`cpp` 或 `python` |
| `--input` | 输入测试用例 CSV 文件 |
| `--output` | 输出结果 JSON 文件 |
| `--limit` | 限制测试用例数量 |
| `--batch-size` | 每批处理的用例数（默认 50） |
| `--tomo` | 自定义代价地图路径（.bin 或 .pickle） |
| `--no-optimize` | **关闭轨迹优化器，仅返回 A* 路径** |

### 关闭优化器模式

使用 `--no-optimize` 选项可以跳过 GTSAM 轨迹优化，仅对比 A* 路径搜索结果：

```bash
# C++ 版本 - 仅 A* 路径
python3 run_batch.py --version cpp \
    --input data/nyby_underground_test_cases.csv \
    --output results/nyby_underground_cpp_astar_results.json \
    --limit 10000 --no-optimize \
    --tomo /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_underground_cost_map.bin

# Python 版本 - 仅 A* 路径
python3 run_batch.py --version python \
    --input data/nyby_underground_test_cases.csv \
    --output results/nyby_underground_python_astar_results.json \
    --limit 10000 --no-optimize \
    --tomo /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/nyby_underground_cost_map.pickle

# 对比 A* 路径结果
python3 compare_versions.py \
    --python results/nyby_underground_python_results.json \
    --cpp results/nyby_underground_cpp_results.json \
    --output results/nyby_underground_comparison.md
```

这对于**定位差异来源**非常有用：

- 如果 A* 路径一致但优化后不一致 → 问题在 GTSAM 优化器
- 如果 A*路径就不一致 → 问题在代价地图数据或 A* 搜索

### 使用自定义代价地图

两个版本都支持使用 `--tomo` 参数指定代价地图路径：

```bash
# Python 版本使用 C++ 的 bin 格式代价地图
python3 run_batch.py --version python \
    --input data/test_cases_valid_ground.csv \
    --output results/python_with_cpp_map.json \
    --tomo /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/scene_map.bin \
    --limit 1000

# C++ 版本使用 Python 的 pickle 格式代价地图
python3 run_batch.py --version cpp \
    --input data/test_cases_valid_ground.csv \
    --output results/cpp_with_py_map.json \
    --tomo /home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle \
    --limit 1000
```

这可以验证**相同代价地图下，两个版本规划结果的一致性**。

### 结果对比 (compare_versions.py)

```bash
python3 compare_versions.py \
    --python results/python_results.json \
    --cpp results/cpp_results.json \
    --output results/comparison_report.md
```

## 生成测试用例

```bash
python3 generate_test_cases.py \
    --pickle /path/to/scene_map.pickle \
    --num-cases 10000 \
    --output-csv data/test_cases.csv
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--pickle` | Python 版本代价地图路径 |
| `--num-cases` | 生成的测试用例数量 |
| `--output-csv` | 输出 CSV 文件路径 |
| `--min-distance` | 起终点最小距离（默认 2m） |
| `--max-distance` | 起终点最大距离（默认 30m） |

## 重要说明

1. **批量模式**：由于库存在内存泄漏，测试脚本默认采用批量模式（每 50 个用例一个子进程）
2. **代价地图差异**：pickle 和 bin 格式的代价地图可能存在微小差异（fp16 量化误差），建议使用相同地图进行对比
3. **A* 路径对比**：使用 `--no-optimize` 可以排除优化器影响，单独验证 A* 搜索一致性

## 验收标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 成功率 | ≥ 60% | 两版本成功率应接近 |
| 轨迹误差 | ≤ 0.1 m | 使用相同代价地图时 |
| A* 路径 | 完全一致 | 使用相同代价地图时 |

## 测试原理

| 版本 | 库路径 | 代价地图格式 |
|------|--------|--------------|
| Python | `PctPlanner_py/planner/lib/` | `.pickle` |
| C++ | `PctPlanner_Cpp/planner_lib/` | `.bin` |

两个版本的库是独立编译的，通过对比测试验证 C++ 移植版本与原 Python 版本的一致性。

## 输出格式

### 结果 JSON

```json
{
  "version": "cpp",
  "lib_path": "库路径",
  "tomogram_path": "代价地图路径",
  "summary": {
    "total": 10000,
    "success": 6500,
    "failed": 3500,
    "success_rate": 65.0,
    "avg_time_ms": 185.5
  },
  "results": {
    "1": {
      "success": true,
      "time_ms": 180.5,
      "num_points": 325,
      "trajectory": [[x, y, z], ...]
    }
  }
}
```

### 对比报告 (Markdown)

生成的报告包括：

- 成功率对比
- 轨迹误差分析（总体、X/Y/Z 轴）
- 性能对比
- 异常用例列表
