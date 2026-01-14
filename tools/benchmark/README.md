# PctPlanner 基准测试工具

本目录包含用于对比测试 C++ 版本和 Python 版本 PctPlanner 规划器的工具集。

## 目录结构

```
benchmark/
├── README.md                 # 本说明文件
├── run_benchmark.sh          # 一键执行测试脚本
├── run_batch.py              # 批量运行（避免内存泄漏）
├── generate_test_cases.py    # 测试用例生成器
├── run_python_version.py     # Python 版本规划测试
├── run_cpp_version.py        # C++ 版本规划测试
├── compare_versions.py       # 结果对比分析
├── data/                     # 测试数据目录
│   ├── test_cases_10k.csv    # CSV 格式测试用例
│   └── test_cases_10k.json   # JSON 格式测试用例
└── results/                  # 测试结果目录
    ├── python_version_results.json
    ├── cpp_version_results.json
    └── version_comparison_report.md
```

## 快速开始

### 一键执行完整测试

```bash
./run_benchmark.sh
```

### 限制测试数量（推荐 100-200 个用于快速测试）

```bash
./run_benchmark.sh --clean --limit 100
```

### 重新生成测试用例

```bash
./run_benchmark.sh --generate 5000
```

### 清理旧结果后测试

```bash
./run_benchmark.sh --clean --limit 150
```

## 重要说明

由于 C++ 库存在内存泄漏问题，测试脚本采用**批量模式**运行（每 50 个用例一个子进程），避免 OOM。

某些测试用例可能导致程序崩溃（段错误），建议使用 `--limit` 参数限制测试数量。

## 手动执行

### 1. 生成测试用例

```bash
python3 generate_test_cases.py --num-cases 1000 \
    --output-csv data/test_cases.csv \
    --output-json data/test_cases.json
```

### 2. 运行 Python 版本测试（批量模式）

```bash
export LD_LIBRARY_PATH=/home/lzy/PctPlanner/PctPlanner_py/planner/lib:/home/lzy/PctPlanner/PctPlanner_py/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
python3 run_batch.py --version python --input data/test_cases_10k.csv --output results/python_results.json --limit 100
```

### 3. 运行 C++ 版本测试（批量模式）

```bash
export LD_LIBRARY_PATH=/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib:/home/lzy/PctPlanner_Cpp_0/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
python3 run_batch.py --version cpp --input data/test_cases_10k.csv --output results/cpp_results.json --limit 100
```

### 4. 对比分析

```bash
python3 compare_versions.py \
    --python results/python_results.json \
    --cpp results/cpp_results.json \
    --output results/comparison_report.md
```

## 验收标准

| 指标 | 目标值 | 当前状态 |
|------|--------|----------|
| Python 成功率 | ≥ 60% | ✅ 61.33% |
| 轨迹误差 | ≤ 0.01 m | ❌ 0.119 m (需优化) |

## 测试结果示例

基于 150 个测试用例：

| 版本 | 成功率 | 平均时间 |
|------|--------|----------|
| Python | 61.33% | 185.48 ms |
| C++ | 62.00% | 183.05 ms |

轨迹误差：

- 平均误差: 0.119 m
- 最大误差: 3.0 m
- Z 轴误差: 0.002 m (非常小)

## 测试原理

- **Python 版本**: 使用 `PctPlanner_py/planner/lib/` 下的库 + pickle 格式 tomogram
- **C++ 版本**: 使用 `PctPlanner_Cpp/planner_lib/` 下的库 + bin 格式 tomogram

两个版本的库是独立编译的，通过对比测试验证 C++ 移植版本与原 Python 版本的一致性。

## 输出说明

### 结果 JSON 格式

```json
{
  "config": {
    "lib_path": "库路径",
    "tomogram_path": "tomogram 文件路径"
  },
  "summary": {
    "total": 1000,
    "success": 650,
    "failed": 350,
    "success_rate": 0.65,
    "avg_time_ms": 185.5
  },
  "results": [
    {
      "case_id": 1,
      "success": true,
      "time_ms": 180.5,
      "trajectory": [[x, y, z], ...]
    }
  ]
}
```

### 对比报告

生成 Markdown 格式的详细对比报告，包括：

- 成功率对比
- 轨迹误差分析（总体、X/Y/Z 轴分别）
- 性能对比
- 异常用例列表
