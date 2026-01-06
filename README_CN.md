# PctPlanner C++（ROS2 Foxy）

基于 CUDA 的点云断层建图 + 全局路径规划，全 C++ 版本，面向 ROS2 Foxy。输入点云生成 tomogram（代价/梯度/高度），再规划无碰路径，提供一键启动。

## 原理与功能

- **断层建图**：订阅 `/global_points`，栅格化得到 trav 代价、梯度、地面/顶面高度，做与原 CuPy 版一致的切片简化，发布 fp16 二进制 tomogram 与“仅地面”可视化。
- **在线规划**（`planner_node`）：监听 `/tomogram_data` + `/start_pos` + `/end_pos`，A*+轨迹优化，输出 `/pct_path` 和 ASCII PCD。
- **离线规划**（`planner_direct_node`）：从 `tomo_path` 直接读 tomogram，等起终点后规划，发布 `/pct_path2` 并周期重发，便于 RViz。
- **工具**：示例点云发布器（`pcd_publisher`），最小冒烟测试。

## 环境要求

- Ubuntu 20.04，ROS2 Foxy
- CUDA 12.8（nvcc 12.8.x）
- CMake ≥ 3.16（已测 4.2.1）
- PCL、Eigen3、ament_cmake、pcl_conversions

## 目录速览

- `src/`：C++ 节点（tomography_node / planner_node / planner_direct_node / pcd_publisher / test_planner_smoke）
- `include/`：头文件（tomogram_format.hpp、tomography_cuda.hpp）
- `launch/`：`pct_all.launch.py`（可开关节点，自动起终点可选）
- `planner_lib/`：预编译规划库与内置 gtsam 4.1.1（install 好）
- `rsc/`：示例 PCD、RViz 配置、示例 tomogram

## 零基础上手（一步步）

1) 安装 ROS2 Foxy 和 CUDA 12.8，确认 `nvcc --version` 显示 12.8.x。
2) 进入工程目录：

   ```bash
   cd ~/PctPlanner_Cpp
   ```

3) 编译（显式指定 CUDA 路径）：

   ```bash
   colcon build --packages-select pct_planner_cpp_port \
     --cmake-args -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
   ```

4) 每个终端先加载环境：

   ```bash
   source install/setup.bash
   export LD_LIBRARY_PATH=$PWD/planner_lib:$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
   ```

5) **先**打开 RViz2，使用随包配置（已设置 Fixed Frame=map，并预置常用话题）：

   ```bash
   rviz2 -d $PWD/rsc/rviz/pct_ros.rviz
   ```

6) 一键启动（建图 + 可选规划 + 示例点云）。默认参数文件（对齐原 scene.py）见 `config/scene_default.yaml`：

   ```bash
   ros2 launch pct_planner_cpp_port pct_all.launch.py \
     output_path:=$PWD/rsc/tomogram/scene_map.bin \
     tomo_path:=$PWD/rsc/tomogram/scene_map.bin \
     pcd_path:=$PWD/trajectory.pcd \
     params_file:=$PWD/config/scene_default.yaml \
     publish_start_end:=true \
     start_x:=5.63 start_y:=15 start_z:=0 \
     end_x:=-9.68 end_y:=6.95 end_z:=0
   ```

## Launch 开关（可组合）

- `enable_planner` / `enable_planner_direct` / `enable_pcd_publisher`
- `publish_start_end` 与 `start_x/y/z`、`end_x/y/z`
- `surface_only`（tomography_node）：默认 true 仅发布地面，false 发布全体素（点数巨大）
- `use_quintic`、`max_heading_rate`：轨迹优化配置
- `params_file`：tomography_node 的 YAML 参数文件，默认与原 scene.py 数值一致（见 `config/scene_default.yaml`）

### 常用启动组合示例

1) 仅建图 + 示例点云（不跑规划）

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   enable_planner:=false enable_planner_direct:=false enable_pcd_publisher:=true
```

1) 全流程在线规划（建图+规划，自动发起起终点）

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   output_path:=$PWD/rsc/tomogram/scene_map.bin \
   pcd_path:=$PWD/trajectory.pcd \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=5.0 end_y:=5.0 end_z:=0.0
```

1) 仅离线规划（读取已有 tomogram，不发点云）

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   enable_planner:=false enable_pcd_publisher:=false enable_planner_direct:=true \
   tomo_path:=$PWD/rsc/tomogram/scene_map.bin pcd_path:=$PWD/trajectory_offline.pcd \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=5.0 end_y:=5.0 end_z:=0.0
```

1) 仅点云与建图，手动发布起终点

- 保持 `publish_start_end:=false`，需要时用 `ros2 topic pub --once /start_pos ...`、`/end_pos ...` 手动发布。
ros2 topic pub --once /start_pos geometry_msgs/msg/Point "{x: 4.45, y: 21.4, z: -0.267}"
ros2 topic pub --once /end_pos   geometry_msgs/msg/Point "{x: -8.03, y: 6.6, z: -0.153}"

## 冒烟测试

```bash
cd ~/PctPlanner_Cpp
source install/setup.bash
./install/pct_planner_cpp_port/lib/pct_planner_cpp_port/test_planner_smoke
```

## 常见问题

- 找不到 GTSAM：`export GTSAM_DIR=$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib/cmake/GTSAM` 后重编译。
- Launch 开关像没生效：重编译或直接用源码路径启动 `ros2 launch ./launch/pct_all.launch.py ...`。
- RViz 卡顿：保持 `surface_only:=true` 或隐藏 `/tomogram`，只看 `/global_points`。

## 提交/发布前

- 保留 `planner_lib` 中的 `.so` 及 `3rdparty/gtsam-4.1.1/install/`，避免用户重建 gtsam。
- 提交前清理 `build/`、`install/`、`log/`。
