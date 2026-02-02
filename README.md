# 🚀 PctPlanner C++ — ROS2 Foxy 版

> **Point Cloud Tomography (PCT) 三维路径规划器**  
> 输入三维点云 → 生成多层可通行代价地图 → 规划三维无碰撞路径

---

## 📖 目录

1. [项目简介](#-项目简介)
2. [核心原理](#-核心原理)
3. [系统架构](#-系统架构)
4. [功能特性](#-功能特性)
5. [环境要求](#-环境要求)
6. [安装指南（零基础版）](#-安装指南零基础版)
7. [快速上手](#-快速上手)
8. [配置参数说明](#-配置参数说明)
9. [ROS2 话题参考](#-ros2-话题参考)
10. [常见问题 FAQ](#-常见问题-faq)
11. [项目结构](#-项目结构)
12. [版权与致谢](#-版权与致谢)

---

## 🎯 项目简介

**PctPlanner** 是一款专为**复杂三维结构化环境**（如多层停车场、立体仓库、多楼层建筑等）设计的全局路径规划器。

### 核心问题

传统二维规划算法（如 ROS Navigation Stack）只能处理平面地图，面对**多层结构**时束手无策。机器人需要能够：

- 在坡道之间穿梭
- 跨楼层导航
- 处理复杂的三维障碍物

### 解决方案

PctPlanner 采用 **Point Cloud Tomography（点云断层扫描）** 技术，将三维点云转化为**多层可通行性地图**，实现真正的三维路径规划。

```
┌─────────────────────────────────────────────────────────────────┐
│                     PctPlanner 工作流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐  │
│   │ 三维点云     │──────▶│ 断层建图     │──────▶│ 路径规划    │  │
│   │ (LiDAR/RGB-D)│       │ (Tomography)│       │ (A* + 优化) │  │
│   └─────────────┘       └─────────────┘       └─────────────┘  │
│                                │                      │         │
│                                ▼                      ▼         │
│                    ┌─────────────────┐    ┌─────────────────┐  │
│                    │ 多层代价地图      │    │ 平滑三维路径     │  │
│                    │ (Tomogram .bin) │    │ (PCD + ROS Path)│  │
│                    └─────────────────┘    └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 核心原理

### 什么是 Point Cloud Tomography？

灵感来源于医学 CT（计算机断层扫描）：

| 医学 CT | Point Cloud Tomography |
|---------|------------------------|
| X 射线穿透人体 | 点云切片分析 |
| 生成人体横截面图像 | 生成每层可通行性信息 |
| 用于诊断病变 | 用于判断机器人能否通过 |

### 算法流程

```
Step 1: 点云栅格化
─────────────────
将稀疏点云投影到规则三维栅格中，统计每个体素的点密度


Step 2: 水平切片
─────────────────
按高度将栅格划分为多个水平层（如每 0.5m 一层）


Step 3: 可通行性分析
─────────────────
对每层计算：
  • 地面高度 (floor height)
  • 顶面高度 (ceiling height)  
  • 通行代价 (traversal cost)
  • 坡度梯度 (slope gradient)


Step 4: 代价膨胀
─────────────────
根据机器人尺寸，对障碍物区域进行膨胀处理


Step 5: 多层融合
─────────────────
生成连通的多层代价地图 (Tomogram)
```

### 可通行性判定标准

一个格子被判定为**可通行**需要满足：

| 指标 | 阈值（默认值） | 含义 |
|------|---------------|------|
| `interval_free` | ≥ 0.65m | 头顶净空高度 |
| `slope_max` | ≤ 0.36 rad | 最大坡度 |
| `step_max` | ≤ 0.20m | 相邻格子高度差 |

---

## 🏗 系统架构

```
                          ROS2 Topics
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ pcd_publisher │    │   外部传感器   │    │  /start_pos   │
│ (示例点云)     │    │ LiDAR/RGB-D  │    │  /end_pos     │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────┬───────┘                    │
                     ▼                            │
            ┌─────────────────┐                   │
            │ tomography_node │                   │
            │   (CUDA 加速)    │                   │
            └────────┬────────┘                   │
                     │ /tomogram_data             │
                     ▼                            ▼
            ┌─────────────────────────────────────────┐
            │            planner_node                 │
            │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
            │  │ A* 搜索  │─▶│ 轨迹优化 │─▶│ 路径输出 │ │
            │  │(Gateway) │  │ (GTSAM) │  │         │ │
            │  └─────────┘  └─────────┘  └─────────┘ │
            └────────────────────┬────────────────────┘
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
            /pct_path     trajectory.pcd   /path_marker
            (ROS Path)    (可视化点云)     (RViz Marker)
```

### 节点说明

| 节点名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `pcd_publisher` | 发布示例点云 | PCD 文件 | `/global_points` |
| `tomography_node` | CUDA 断层建图 | `/global_points` | `/tomogram_data`, `.bin` 文件 |
| `planner_node` | 在线路径规划 | `/tomogram_data`, 起终点 | `/pct_path`, PCD |
| `planner_direct_node` | 离线路径规划 | `.bin` 文件, 起终点 | `/pct_path2`, PCD |
| `planner_semantic_node` | 语义地图批量规划 | `.bin` 文件, 语义 JSON | `/pct_path3`, 更新后的 JSON |

---

## ✨ 功能特性

### 核心功能

- ✅ **CUDA 加速断层建图** — 百万级点云实时处理（含 FilterKernel 邻域滤波）
- ✅ **语义地图批量规划** — 自动为语义拓扑图中所有边生成轨迹
- ✅ **多层 A* 路径搜索** — 支持跨层路径规划
- ✅ **Gateway 检测** — 自动识别层间连接点（坡道、楼梯等）
- ✅ **GTSAM 轨迹优化** — 平滑、自然的机器人路径
- ✅ **ROS2 原生支持** — 完美集成 ROS2 生态

### 输出格式

- 📍 **ROS Path** — 标准 `nav_msgs/Path`，可直接用于导航控制
- 📁 **PCD 文件** — 可视化轨迹点云
- 🗺️ **Tomogram 二进制** — 可保存/加载的代价地图

### 精度模式

| 模式 | 存储 | 精度 | 推荐场景 |
|------|------|------|----------|
| `float16` | 文件小 | 厘米级 | 日常使用 |
| `float32` | 文件大 | 毫米级 | 高精度需求 |

---

## 💻 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 x86_64 | 8 核以上 |
| 内存 | 8 GB | 16 GB 以上 |
| GPU | NVIDIA GTX 1060 | RTX 3060 以上 |
| 显存 | 4 GB | 8 GB 以上 |
| 硬盘 | 20 GB 可用空间 | SSD 推荐 |

### 软件要求

| 软件 | 版本 | 必须 |
|------|------|------|
| Ubuntu | 20.04 LTS | ✅ |
| ROS2 | Foxy Fitzroy | ✅ |
| CUDA | 12.x (推荐 12.8) | ✅ |
| CMake | ≥ 3.16 | ✅ |
| GCC | 9.x (Ubuntu 20.04 默认) | ✅ |

### 依赖库

| 库 | 说明 | 安装方式 |
|----|------|---------|
| PCL | 点云处理 | apt |
| Eigen3 | 线性代数 | apt |
| GTSAM 4.1.1 | 轨迹优化 | **已内置** |
| pcl_conversions | ROS-PCL 转换 | apt |

---

## 📦 安装指南（零基础版）

> ⚠️ 请严格按照顺序执行每一步，遇到问题先查看 [FAQ](#-常见问题-faq)

### Step 1: 安装 ROS2 Foxy

```bash
# 1.1 设置语言环境
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8


# 1.2 添加 ROS2 软件源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo 
$UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null


# 1.3 安装 ROS2 Foxy 桌面版（包含 RViz2）
sudo apt update
sudo apt install ros-foxy-desktop python3-argcomplete -y


# 1.4 设置环境变量（添加到 ~/.bashrc）
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc


# 1.5 验证安装
ros2 --version
# 应显示: ros2 0.9.x
```

### Step 2: 安装 CUDA Toolkit

```bash
# 2.1 检查显卡驱动
nvidia-smi
# 应显示 NVIDIA 驱动版本和 GPU 信息


# 2.2 下载 CUDA 12.8（以 Ubuntu 20.04 为例）
# 访问: https://developer.nvidia.com/cuda-12-8-0-download-archive
# 选择: Linux > x86_64 > Ubuntu > 20.04 > deb (local)


# 或使用命令行:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2004-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8


# 2.3 配置环境变量（添加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


# 2.4 验证安装
nvcc --version
# 应显示: nvcc: NVIDIA (R) Cuda compiler driver
#         Cuda compilation tools, release 12.8, V12.8.xxx
```

### Step 3: 安装依赖库

```bash
# 3.1 PCL 和 Eigen
sudo apt install libpcl-dev libeigen3-dev -y


# 3.2 ROS2 PCL 转换包
sudo apt install ros-foxy-pcl-conversions ros-foxy-pcl-ros -y


# 3.3 colcon 构建工具
sudo apt install python3-colcon-common-extensions -y


# 3.4 其他依赖
sudo apt install build-essential cmake -y
```

### Step 4: 编译项目

#### 方式一：使用 build.sh 脚本（推荐，用于开发测试）

```bash
# 4.1 进入项目目录
cd ~/PctPlanner/PctPlanner_Cpp


# 4.2 使用构建脚本（自动配置 CUDA 路径）
./build.sh              # Release 构建（默认）
./build.sh debug        # Debug 构建
./build.sh clean        # 清理构建目录


# 4.3 验证编译成功
ls build/cmake_build/
# 应看到: tomography_node, planner_node, planner_direct_node, pcd_publisher, tomography_benchmark
```

**构建目录说明**：

| 目录 | 用途 | 推荐场景 |
|------|------|----------|
| `build/cmake_build/` | CMake 直接构建输出 | 开发测试、性能测试 |
| `build/pct_planner_cpp_port/` | Colcon ROS2 构建输出 | ROS2 节点运行 |

#### 方式二：使用 colcon（用于 ROS2 集成）

```bash
# 4.1 进入项目目录
cd ~/PctPlanner/PctPlanner_Cpp


# 4.2 清理旧的编译产物（可选）
rm -rf build/ install/ log/


# 4.3 编译（指定 CUDA 路径）
colcon build --packages-select pct_planner_cpp_port \
  --cmake-args \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.8


# 4.4 验证编译成功
ls install/pct_planner_cpp_port/lib/pct_planner_cpp_port/
# 应看到: tomography_node, planner_node, planner_direct_node, pcd_publisher
```

### Step 5: 配置环境

在**每个新终端**中运行（或添加到 `~/.bashrc`）：

```bash
# 5.1 加载 ROS2 工作空间
source ~/PctPlanner/PctPlanner_Cpp/install/setup.bash


# 5.2 配置库路径
export LD_LIBRARY_PATH=$HOME/PctPlanner/PctPlanner_Cpp/planner_lib:$HOME/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
```

**一键配置脚本**（推荐添加到 `~/.bashrc`）:

```bash
# PctPlanner 环境配置
alias pct_env='source ~/PctPlanner/PctPlanner_Cpp/install/setup.bash && export LD_LIBRARY_PATH=$HOME/PctPlanner/PctPlanner_Cpp/planner_lib:$HOME/PctPlanner/
PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH && echo "PctPlanner 环境已加载 ✓"'
```

之后只需输入 `pct_env` 即可完成环境配置。

---

## 🚀 快速上手

### 方式一：一键启动（推荐新手）

```bash
# 1. 打开终端，配置环境
cd ~/PctPlanner/PctPlanner_Cpp
source install/setup.bash
export LD_LIBRARY_PATH=$PWD/planner_lib:$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH


# 2. 先启动 RViz2 可视化（新终端）
rviz2 -d $PWD/rsc/rviz/pct_ros.rviz


# 3. 一键启动全流程
ros2 launch pct_planner_cpp_port pct_all.launch.py \
  params_file:=$PWD/config/scene_default.yaml \
  pcd_path:=$PWD/rsc/pcd/map.pcd \
  publish_start_end:=true \
  start_x:=5.63 start_y:=15.0 start_z:=0.0 \
  end_x:=-9.68 end_y:=6.95 end_z:=0.0
```

**预期效果**：

1. RViz 显示彩色点云（`/global_points`）
2. 几秒后显示代价地图（`/tomogram`）
3. 显示规划路径（红色线条 `/pct_path`）

### 方式二：分步执行（理解流程）

> ⚠️ **重要**：每个新终端都需要先执行环境配置！

```bash
# 【每个终端都要先执行】环境配置
cd ~/PctPlanner/PctPlanner_Cpp
source install/setup.bash
export LD_LIBRARY_PATH=$PWD/planner_lib:$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
```

```bash
# 终端 1: 发布示例点云
ros2 run pct_planner_cpp_port pcd_publisher \
  --ros-args -p pcd_path:=$PWD/rsc/pcd/nyby_ground_pct_v0.pcd


# 终端 2: 运行断层建图
ros2 run pct_planner_cpp_port tomography_node \
  --ros-args --params-file $PWD/config/scene_default.yaml \
  -p output_path:=$PWD/rsc/tomogram/nyby_cost_map.bin


# 终端 3: 运行路径规划
ros2 run pct_planner_cpp_port planner_node \
  --ros-args -p pcd_path:=$PWD/trajectory.pcd


# 终端 4: 发布起终点
ros2 topic pub --once /start_pos geometry_msgs/msg/Point "{x: 5.63, y: 15.0, z: 0.0}"
ros2 topic pub --once /end_pos geometry_msgs/msg/Point "{x: -9.68, y: 6.95, z: 0.0}"
```

### 方式三：离线规划（使用已有 Tomogram）

如果已经有保存好的 `.bin` 文件，可以跳过建图直接规划：

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
  enable_planner:=false \
  enable_pcd_publisher:=false \
  enable_planner_direct:=true \
  tomo_path:=$PWD/rsc/tomogram/scene_map.bin \
  pcd_path:=$PWD/trajectory_offline.pcd \
  publish_start_end:=true \
  start_x:=5.63 start_y:=15.0 start_z:=0.0 \
  end_x:=-9.68 end_y:=6.95 end_z:=0.0
```

### 方式四：语义地图批量规划

对语义拓扑图中的所有边自动生成轨迹：

```bash
# 使用默认参数
ros2 run pct_planner_cpp_port planner_semantic_node

# 自定义参数
ros2 run pct_planner_cpp_port planner_semantic_node \
  --ros-args \
  -p tomogram_path:=$PWD/rsc/tomogram/scene_map.bin \
  -p input_json:=$PWD/rsc/semantic_topology/nyby_b5f3.json \
  -p output_json:=$PWD/rsc/semantic_topology/nyby_b5f3_updated.json \
  -p z_offset:=1.0 \
  -p use_quintic:=true
```

**功能说明**：

1. 从 `input_json` 读取语义拓扑图（节点、边）
2. 为每条边自动规划从 source 到 target 的轨迹
3. 将规划结果写入 `output_json`
4. 以 1Hz 频率循环发布各边轨迹到 `/pct_path3` 话题

---

## ⚙️ 配置参数说明

### 断层建图参数 (`config/scene_default.yaml`)

```yaml
pct_tomography_cpp:
  ros__parameters:
    # === 栅格化参数 ===
    resolution: 0.10        # 栅格分辨率 (m)，越小精度越高，计算量越大
    kernel_size: 7          # 滤波核大小，影响地面检测平滑度
    
    # === 可通行性判定 ===
    interval_free: 0.65     # 最小头顶净空 (m)，需大于机器人高度
    slope_max: 0.36         # 最大坡度 (rad)，约 20°
    step_max: 0.20          # 最大台阶高度 (m)
    
    # === 代价膨胀 ===
    safe_margin: 0.4        # 安全边距 (m)，机器人半径 + 余量
    inflation: 0.2          # 障碍物膨胀距离 (m)
    
    # === 输出设置 ===
    precision_mode: float16 # 精度模式: float16 或 float32
```

### Launch 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `params_file` | string | scene_default.yaml | 断层建图参数文件 |
| `enable_planner` | bool | true | 启用在线规划节点 |
| `enable_planner_direct` | bool | false | 启用离线规划节点 |
| `enable_pcd_publisher` | bool | true | 启用示例点云发布 |
| `publish_start_end` | bool | false | 自动发布起终点 |
| `start_x/y/z` | float | 0.0 | 起点坐标 |
| `end_x/y/z` | float | 0.0 | 终点坐标 |
| `surface_only` | bool | true | 仅发布地面（减少 RViz 负载） |
| `precision_mode` | string | float16 | 精度模式 |
| `output_path` | string | - | Tomogram 输出路径 |
| `tomo_path` | string | - | Tomogram 输入路径（离线模式） |
| `pcd_path` | string | - | 轨迹 PCD 输出路径 |
| `use_quintic` | bool | true | 使用五次多项式轨迹优化 |

---

## 📡 ROS2 话题参考

### 订阅话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/global_points` | `sensor_msgs/PointCloud2` | 输入点云（需包含 XYZ） |
| `/start_pos` | `geometry_msgs/Point` | 起点坐标 |
| `/end_pos` | `geometry_msgs/Point` | 终点坐标 |

### 发布话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/tomogram` | `sensor_msgs/PointCloud2` | 可通行区域可视化 |
| `/tomogram_data` | `std_msgs/UInt8MultiArray` | 二进制 Tomogram 数据 |
| `/pct_path` | `nav_msgs/Path` | 规划路径（在线模式） |
| `/pct_path2` | `nav_msgs/Path` | 规划路径（离线模式） |
| `/pct_path3` | `nav_msgs/Path` | 规划路径（语义地图模式，循环发布各边轨迹） |
| `/path_marker` | `visualization_msgs/Marker` | 路径可视化标记 |

---

## ❓ 常见问题 FAQ

### Q1: 编译时找不到 CUDA

**错误信息**: `Could not find CUDA` 或 `nvcc not found`

**解决方法**:

```bash
# 确认 CUDA 安装路径
ls /usr/local/ | grep cuda


# 编译时显式指定
colcon build --packages-select pct_planner_cpp_port \
  --cmake-args \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
```

### Q2: 运行时找不到 GTSAM

**错误信息**: `libgtsam.so.4: cannot open shared object file`

**解决方法**:

```bash
# 设置库路径
export LD_LIBRARY_PATH=$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH


# 或编译时指定
export GTSAM_DIR=$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib/cmake/GTSAM
colcon build --packages-select pct_planner_cpp_port
```

### Q3: RViz 非常卡顿

**原因**: 点云数据量过大

**解决方法**:

1. 保持 `surface_only:=true`（默认已开启）
2. 在 RViz 中隐藏 `/tomogram` 话题
3. 降低 Decay Time 参数

### Q4: Launch 参数不生效

**解决方法**:

```bash
# 方法1: 清理后重新编译
rm -rf build/ install/ log/
colcon build --packages-select pct_planner_cpp_port


# 方法2: 直接使用源码路径启动
ros2 launch ./launch/pct_all.launch.py ...
```

### Q5: 规划失败，显示 "No valid path found"

**可能原因**:

1. 起点或终点在障碍物内
2. 起终点之间确实不可达
3. 参数设置过于严格

**解决方法**:

1. 检查起终点坐标是否在点云范围内
2. 在 RViz 中查看 `/tomogram` 确认可通行区域
3. 尝试调整参数：

   ```yaml
   slope_max: 0.5      # 放宽坡度限制
   step_max: 0.3       # 放宽台阶限制
   safe_margin: 0.2    # 减小安全边距
   ```

---

## 📁 项目结构

```
PctPlanner_Cpp/
├── CMakeLists.txt          # CMake 构建配置
├── package.xml             # ROS2 包配置
├── README_CN.md            # 中文文档（本文件）
├── README.md               # 英文文档
├── .gitignore              # Git 忽略规则
│
├── docs/
│   └── ZIQUAN_EXPLORE_SYNC_REPORT.md  # ziquan-explore 分支同步报告
│
├── config/
│   ├── scene_default.yaml  # 默认参数配置
│   └── scene_nyby_ground.yaml  # NYBY Ground 场景配置
│
├── include/                # 头文件
│   ├── planner_common.hpp  # 规划器公共定义
│   ├── semantic_io.hpp     # 语义地图数据结构与 JSON I/O
│   ├── tomogram_format.hpp # Tomogram 二进制格式
│   └── tomography_cuda.hpp # CUDA 断层建图接口
│
├── src/                    # C++ 源代码
│   ├── tomography_cuda.cu  # CUDA 断层建图实现（含 FilterKernel）
│   ├── tomography_node.cpp # 断层建图 ROS2 节点
│   ├── planner_node.cpp    # 在线规划节点
│   ├── planner_direct_node.cpp  # 离线规划节点
│   ├── planner_semantic_node.cpp # 语义地图批量规划节点
│   ├── semantic_io.cpp     # 语义地图 JSON 读写实现
│   ├── pcd_publisher.cpp   # 点云发布器
│   └── test_planner_smoke.cpp   # 冒烟测试
│
├── launch/
│   └── pct_all.launch.py   # 一键启动脚本
│
├── planner_lib/            # 预编译规划库（已包含，无需重建）
│   ├── lib*.so             # 规划器核心库
│   └── 3rdparty/
│       └── gtsam-4.1.1/    # GTSAM 预编译库
│
├── rsc/                    # 资源文件
│   ├── pcd/               # 示例点云
│   ├── tomogram/          # 生成的 Tomogram 文件
│   ├── semantic_topology/ # 语义拓扑地图 JSON 文件
│   └── rviz/              # RViz 配置文件
│
├── tools/                  # 辅助工具
│   ├── accuracy/          # 精度一致性测试
│   ├── performance/       # 性能对比测试
│   │   ├── benchmark_python.py      # Python 性能测试
│   │   ├── tomography_benchmark.cpp # C++ 性能测试
│   │   └── README.md                # 使用说明
│   └── rsc_compare/       # 资源文件对比工具
│
├── build.sh                # 🔧 一键构建脚本
├── build/                  # 🔧 构建输出目录（已 gitignore）
│   ├── cmake_build/       # CMake 直接构建（./build.sh）
│   └── pct_planner_cpp_port/  # Colcon ROS2 构建
├── install/                # 📦 colcon 安装目录（已 gitignore）
└── log/                    # 📝 colcon 编译日志（已 gitignore）
```

> ⚠️ **注意**：`build/`、`install/`、`log/` 目录为编译产物，已添加到 `.gitignore`，不会提交到版本控制。

---

## 📄 版权与致谢

本项目基于开源许可证发布，详见 LICENSE 文件。

### 依赖项目

- [ROS2 Foxy](https://docs.ros.org/en/foxy/) — 机器人操作系统
- [PCL](https://pointclouds.org/) — 点云处理库
- [GTSAM](https://gtsam.org/) — 因子图优化库
- [CUDA](https://developer.nvidia.com/cuda-toolkit) — GPU 并行计算

---

📝 **如有问题，欢迎提交 Issue 或 Pull Request！**
