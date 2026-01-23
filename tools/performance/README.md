# Tomography 性能对比测试

本目录包含 C++ 和 Python 版本 Tomography（代价地图生成）的精确性能测试工具。

## 快速开始

### 1. 测试 C++ 版本

```bash
# 首先构建 C++ 项目（在 PctPlanner_Cpp 目录下）
cd /home/lzy/PctPlanner/PctPlanner_Cpp
./build.sh

# 运行 C++ 性能测试
./build/cmake_build/tomography_benchmark --runs 10

# 指定 PCD 文件
./build/cmake_build/tomography_benchmark --pcd /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/pcd/map.pcd --runs 10
```

### 2. 测试 Python 版本

```bash
# 在 performance 目录下运行
cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/performance

# 运行 Python 性能测试
python3 benchmark_python.py --runs 10

# 指定 PCD 文件
python3 benchmark_python.py --pcd /home/lzy/PctPlanner/PctPlanner_Cpp/rsc/pcd/map.pcd --runs 10
```

## 测试工具说明

| 文件 | 描述 |
|------|------|
| `tomography_benchmark.cpp` | C++/CUDA 精确性能测试（编译为可执行文件） |
| `benchmark_python.py` | Python/CuPy 精确性能测试脚本 |

## 测试参数

两个测试工具使用相同的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pcd` | `PctPlanner_py/rsc/pcd/map.pcd` | 输入点云文件 |
| `--runs` | 10 | 测试运行次数 |
| `--warmup` | 2-3 | GPU 预热次数 |

## 测试输出

两个测试工具输出格式一致，便于对比：

```
======================================================================
C++/CUDA 统计结果
======================================================================
点云数量: 24,800,000
地图维度: 466×778×29

阶段                平均(ms)    标准差      最小        最大
--------------------------------------------------------------------
PCD 加载            272.00      5.00        265.00      280.00
CUDA 计算           239.00      3.00        235.00      245.00
端到端总时间        511.00      6.00        500.00      520.00
```

## 性能对比

完整测试流程：

```bash
# 1. 测试 C++ 版本
cd /home/lzy/PctPlanner/PctPlanner_Cpp
./build/cmake_build/tomography_benchmark --runs 10 > cpp_result.txt

# 2. 测试 Python 版本
cd tools/performance
python3 benchmark_python.py --runs 10 > python_result.txt

# 3. 对比结果
echo "=== C++ 结果 ===" && cat ../../cpp_result.txt
echo "=== Python 结果 ===" && cat python_result.txt
```

## 测试环境要求

### C++ 版本

- CUDA Toolkit
- PCL (Point Cloud Library)
- Eigen3

### Python 版本

- Python 3.8+
- CuPy (`pip install cupy-cuda11x` 或对应 CUDA 版本)
- Open3D (`pip install open3d`)

## 注意事项

1. **GPU 预热**：首次运行会初始化 CUDA 上下文，预热运行不计入统计
2. **公平对比**：确保两个测试使用相同的 PCD 文件和参数
3. **精确计时**：两个版本都使用 CUDA Event 进行 GPU 计时，确保精确性
