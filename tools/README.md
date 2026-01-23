# PctPlanner 测试工具

本目录包含用于测试 C++ 和 Python 版本的各类工具脚本。

## 目录结构

```
tools/
├── accuracy/           # 精度一致性测试（验证 C++/Python 结果一致）
│   ├── data/           # 测试用例数据
│   ├── results/        # 一致性对比报告
│   ├── run_batch.py    # 批量运行测试
│   └── compare_versions.py  # 对比两版本结果
│
├── performance/        # 性能测试（测量运行速度）
│   ├── benchmark_python.py       # Python/CuPy 精确性能测试
│   ├── tomography_benchmark.cpp  # C++/CUDA 精确性能测试
│   └── README.md                 # 详细使用说明
│
└── rsc_compare/        # 资源文件对比（tomogram 等）
    ├── tomogram_compare.py   # 对比 tomogram 数据
    └── tomogram_convert.py   # 格式转换工具
```

## 快速使用

### 性能测试

```bash
# 1. C++/CUDA 性能测试
cd ~/PctPlanner/PctPlanner_Cpp
./build/cmake_build/tomography_benchmark \
  --pcd ~/PctPlanner/PctPlanner_py/rsc/pcd/map.pcd --runs 10

# 2. Python/CuPy 性能测试
cd tools/performance
python3 benchmark_python.py --runs 10
```

### 精度一致性测试

```bash
cd tools/accuracy
python3 run_batch.py --version python --input data/test_cases.csv --output results/python_results.json
python3 run_batch.py --version cpp --input data/test_cases.csv --output results/cpp_results.json
python3 compare_versions.py --python results/python_results.json --cpp results/cpp_results.json
```

### Tomogram 文件对比

```bash
cd tools/rsc_compare
python3 tomogram_compare.py \
  --bin ~/PctPlanner/PctPlanner_Cpp/rsc/tomogram/scene_map.bin \
  --pickle ~/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle
```
