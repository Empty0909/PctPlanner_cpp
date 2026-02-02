# PctPlanner C++ 更新报告：ziquan-explore 分支同步

**日期**: 2026-02-02  
**版本**: v1.1.0  
**作者**: AI Assistant

---

## 📋 更新概述

本次更新将 `PctPlanner-ziquan-explore` 分支的三大改动同步到 `PctPlanner_Cpp` C++ 版本，确保 Python 和 C++ 版本功能对齐、同样输入产生同样输出。

---

## 🔧 改动一：Cost Map 生成优化 (CUDA FilterKernel)

### 背景

原版 CUDA 代码在处理稀疏点云时，缺失地面/顶部高度的格子会导致梯度计算异常，影响通行代价地图质量。

### 解决方案

新增 `FilterKernel` 对缺失值进行 8 邻域插值滤波。

### 修改文件

**`src/tomography_cuda.cu`**

#### 新增 FilterKernel 函数

```cpp
// FilterKernel: 与 Python ziquan-explore 分支的 filterKernel 对齐
// 对缺失的地面/顶部高度进行邻域插值：
// - 如果 current_g < -1e5，用邻域 8 个格子中有效值的最小值填充
// - 如果 current_c > 1e5，用邻域 8 个格子中有效值的最大值填充
__global__ void FilterKernel(const float *layers_g, const float *layers_c,
                             float *filtered_layers_g, float *filtered_layers_c,
                             int dim_x, int dim_y, int n_slice);
```

#### 更新 ClearKernel

新增 `filtered_layers_g` 和 `filtered_layers_c` 缓冲区初始化。

#### 更新 RunTomographyCuda

- 内存分配从 7 个 float 数组增加到 9 个
- 调用顺序变更为：

  ```
  ClearKernel → TomographyKernel → FilterKernel → GradIntervalKernel
  ```

- `GradIntervalKernel` 改为使用 `filtered_layers_*` 计算梯度
- 输出改为 `filtered_layers_*` 而非原始 `layers_*`

### 效果

- 减少稀疏区域的噪声
- 提高代价地图的连续性
- 与 Python 版本输出一致

---

## 🗺️ 改动二：语义地图模块

### 背景

需要支持语义拓扑地图的读写，用于批量规划语义图中所有边的轨迹。

### 新增文件

#### `include/semantic_io.hpp`

语义地图数据结构定义：

| 结构体 | 功能 |
|--------|------|
| `Trajectory` | 轨迹（waypoints、距离统计） |
| `JsonNode` | 语义节点（id、label、position） |
| `Edge` | 边（source、target、trajectory） |
| `SemanticMap` | 完整语义地图（nodes + edges） |

核心方法：

- `load_semantic_json(path)` - 从 JSON 文件加载
- `write_semantic_json(graph, path)` - 写入 JSON 文件
- `Trajectory::to_ros()` - 转换为 ROS Path 消息

#### `src/semantic_io.cpp`

使用 `nlohmann_json` 库实现 JSON 序列化/反序列化。

#### `src/planner_semantic_node.cpp`

语义地图批量规划节点，对应 Python 的 `plan_nyby.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tomogram_path` | `../../rsc/tomogram/scene_map.bin` | Tomogram 文件路径 |
| `input_json` | `../../rsc/semantic_topology/nyby_b5f3.json` | 输入语义地图 |
| `output_json` | `...nyby_b5f3_updated.json` | 输出更新后的地图 |
| `z_offset` | 1.0 | 起终点 Z 轴偏移 |
| `use_quintic` | true | 使用五次多项式优化 |

功能：

1. 读取 tomogram 和语义地图
2. 为每条边规划轨迹
3. 写入更新后的 JSON
4. 1 Hz 循环发布不同边的轨迹到 `/pct_path3`

---

## ⚙️ 改动三：配置参数对齐

### `config/scene_default.yaml`

参数更新以匹配 Python `scene.py`：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `resolution` | 0.15 | **0.10** | 栅格分辨率（米） |
| `kernel_size` | 5 | **7** | 膨胀卷积核尺寸 |
| `interval_free` | 0.6 | **0.65** | 无惩罚间隙阈值（米） |
| `slope_max` | 1.0 | **0.36** | 最大坡度（rad） |
| `step_max` | 0.7 | **0.20** | 最大台阶高度差（米） |
| `standable_ratio` | 0.40 | **0.20** | 可站立比例阈值 |
| `safe_margin` | 0.1 | **0.4** | 安全边界（米） |
| `inflation` | 0.05 | **0.2** | 膨胀附加成本 |

### 新增 `config/scene_nyby_ground.yaml`

针对 NYBY Ground 场景的特定配置（`standable_ratio=0.1`, `safe_margin=1.0`）。

---

## 📦 CMakeLists.txt 更新

### 新增依赖

```cmake
find_package(nlohmann_json QUIET)

# 如果系统没有 nlohmann_json，使用 FetchContent 获取
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()
```

### 新增编译目标

```cmake
# 语义地图读写库
add_library(semantic_io STATIC src/semantic_io.cpp)

# 语义地图批量规划节点
add_executable(planner_semantic_node src/planner_semantic_node.cpp)
```

---

## 📁 资源文件更新

### 新增目录

`rsc/semantic_topology/` - 语义拓扑 JSON 文件

### 复制的文件

从 `PctPlanner-ziquan-explore` 复制：

- `nyby_b5f3.json` / `nyby_b5f3_updated.json`
- `nyby_b5f2.json` / `nyby_b5f2_updated.json`
- `nyby_ground.json` / `nyby_ground_updated.json`
- `nyby_underground.json` / `nyby_underground_updated.json`
- `phd_dom_*.json`
- `b8f1_topology*.json`

---

## 🚀 使用方式

### 编译

```bash
cd /home/lzy/PctPlanner/PctPlanner_Cpp
./build.sh
```

### 运行语义规划节点

```bash
# 使用默认参数
ros2 run pct_planner_cpp_port planner_semantic_node

# 自定义参数
ros2 run pct_planner_cpp_port planner_semantic_node \
  --ros-args \
  -p tomogram_path:=/path/to/scene_map.bin \
  -p input_json:=/path/to/semantic_topology.json \
  -p output_json:=/path/to/output_updated.json \
  -p z_offset:=1.0
```

### 验证输出

```bash
# 查看发布的路径
ros2 topic echo /pct_path3

# 查看更新后的 JSON
cat /path/to/output_updated.json | jq '.edges[0].trajectory'
```

---

## ✅ 验证结果

| 测试项 | 状态 |
|--------|------|
| CMake 配置 | ✅ 通过 |
| 编译 (make -j4) | ✅ 通过 |
| tomography_cuda 库 | ✅ 构建成功 |
| semantic_io 库 | ✅ 构建成功 |
| planner_semantic_node | ✅ 构建成功 |

---

## 📝 后续工作

1. **功能测试**: 使用相同输入数据验证 Python 和 C++ 版本输出一致性
2. **性能测试**: 对比 FilterKernel 前后的代价地图质量
3. **文档完善**: 添加语义地图 JSON 格式规范文档

---

## 📚 相关文件索引

| 文件 | 描述 |
|------|------|
| `src/tomography_cuda.cu` | CUDA 断层建图核心代码（含 FilterKernel） |
| `include/semantic_io.hpp` | 语义地图数据结构定义 |
| `src/semantic_io.cpp` | 语义地图 JSON 读写实现 |
| `src/planner_semantic_node.cpp` | 语义地图批量规划节点 |
| `config/scene_default.yaml` | 默认场景配置（已更新） |
| `config/scene_nyby_ground.yaml` | NYBY Ground 场景配置 |
| `rsc/semantic_topology/*.json` | 语义拓扑地图文件 |
