# PctPlanner C++ (ROS2 Foxy)

CUDA-based point cloud tomography + global path planner, fully in C++ for ROS2 Foxy. Generates 3D tomograms from point clouds, plans collision-free trajectories, and ships a one-click launch.

## What it does (principle)

- **Tomography**: consume `/global_points` point clouds, build a 3D grid (trav cost, gradients, ground/ceiling heights). Uses slice simplification identical to the original CuPy pipeline, publishes binary tomogram with selectable precision (fp16/fp32) and a surface-only visualization.
- **Online planner** (`planner_node`): listens to `/tomogram_data` + `/start_pos` + `/end_pos`, runs A*/trajectory optimizer, outputs `/pct_path` and an ASCII PCD.
- **Offline planner** (`planner_direct_node`): loads a tomogram file (`tomo_path`), waits for start/end, publishes `/pct_path2` and ASCII PCD; re-publishes periodically for RViz.
- **Utilities**: sample point-cloud publisher (`pcd_publisher`) and a tiny smoke test.

## Requirements

- Ubuntu 20.04, ROS2 Foxy
- CUDA 12.8 (nvcc 12.8.x)
- CMake ≥ 3.16 (tested 4.2.1)
- PCL, Eigen3, ament_cmake, pcl_conversions

## Repository layout

- `src/`: C++ nodes (`tomography_node`, `planner_node`, `planner_direct_node`, `pcd_publisher`, `test_planner_smoke`)
- `include/`: headers (`tomogram_format.hpp`, `tomography_cuda.hpp`)
- `launch/`: `pct_all.launch.py` (toggle nodes; optional auto start/end publisher)
- `planner_lib/`: prebuilt planning libs + vendored gtsam 4.1.1 install
- `rsc/`: sample PCD, RViz config, sample tomogram

## Step-by-step (zero to run)

1) Install ROS2 Foxy and CUDA 12.8. Ensure `nvcc --version` shows 12.8.x.
2) Clone this repo (or copy the prepared folder) and enter it:

   ```bash
   cd ~/PctPlanner_Cpp
   ```

3) Build (specify CUDA paths explicitly):

   ```bash
   colcon build --packages-select pct_planner_cpp_port \
     --cmake-args -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
   ```

4) Source the workspace and set runtime libs (once per shell):

   ```bash
   source install/setup.bash
   export LD_LIBRARY_PATH=$PWD/planner_lib:$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
   ```

5) Start RViz2 **before** launching nodes, using the provided config (preloads displays and Fixed Frame=map):

   ```bash
   rviz2 -d $PWD/rsc/rviz/pct_ros.rviz
   ```

6) One-click launch (build tomogram, optionally plan, and publish sample cloud). A default parameter file aligned with the original scene.py is at `config/scene_default.yaml` (see **Precision modes** if you need float32 maps):

   ```bash
   ros2 launch pct_planner_cpp_port pct_all.launch.py \
     output_path:=$PWD/rsc/tomogram/scene_map.bin \
     tomo_path:=$PWD/rsc/tomogram/scene_map.bin \
     pcd_path:=$PWD/trajectory.pcd \
   params_file:=$PWD/config/scene_default.yaml \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=1.88 end_y:=-4.98 end_z:=3.72
   ```

   ros2 launch pct_planner_cpp_port pct_all.launch.py \
   output_path:=$PWD/rsc/tomogram/scene_map.bin \
   tomo_path:=$PWD/rsc/tomogram/scene_map.bin \
   pcd_path:=$PWD/trajectory.pcd \
   params_file:=$PWD/config/scene_default.yaml \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=1.88 end_y:=-4.98 end_z:=3.72

## Launch switches (mix and match)

- `enable_planner` / `enable_planner_direct` / `enable_pcd_publisher`
- `publish_start_end` with `start_x/y/z`, `end_x/y/z`
- `surface_only` (tomography_node): default true for surface visualization; false publishes full volume (very dense)
- `precision_mode` (tomography_node): `float16` (default) or `float32` to control on-wire tomogram precision
- `use_quintic`, `max_heading_rate` (planner trajectory options)
- `params_file`: YAML for tomography_node (defaults match original scene.py; see `config/scene_default.yaml`)

### Common launch recipes

1) Map only + sample cloud (no planners)

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   enable_planner:=false enable_planner_direct:=false enable_pcd_publisher:=true
```

1) Full online pipeline (build tomogram and plan; auto publish start/end)

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   output_path:=$PWD/rsc/tomogram/scene_map.bin \
   pcd_path:=$PWD/trajectory.pcd \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=5.0 end_y:=5.0 end_z:=0.0
```

1) Offline planning only (load existing tomogram; no cloud publisher)

```bash
ros2 launch pct_planner_cpp_port pct_all.launch.py \
   enable_planner:=false enable_pcd_publisher:=false enable_planner_direct:=true \
   tomo_path:=$PWD/rsc/tomogram/scene_map.bin pcd_path:=$PWD/trajectory_offline.pcd \
   publish_start_end:=true start_x:=0.0 start_y:=0.0 start_z:=0.0 end_x:=5.0 end_y:=5.0 end_z:=0.0
```

1) Map + cloud only, publish start/end manually

- Keep `publish_start_end:=false`, then run `ros2 topic pub --once /start_pos ...` and `/end_pos ...` when ready.

## Precision modes

- Tomogram headers now carry a `precision_mode` flag so planners know whether payload values are fp16 or fp32. The default stay-on-the-wire size is fp32.
- To force 32-bit floats (higher fidelity at twice the size) pass `precision_mode:=float32` when launching `tomography_node`, or edit `config/scene_default.yaml` (`pct_tomography_cpp.ros__parameters.precision_mode`).
- Any new `.bin` produced with fp32 still loads transparently in the C++ planners; older fp16 binaries keep working. After changing precision you must regenerate the tomogram once because the file format changed.

## Tips & troubleshooting

- If `GTSAM_DIR` is not found: `export GTSAM_DIR=$PWD/planner_lib/3rdparty/gtsam-4.1.1/install/lib/cmake/GTSAM` then rebuild.
- If launch toggles seem ignored, rebuild and use the launch from the source tree: `ros2 launch ./launch/pct_all.launch.py ...`.
- RViz laggy? Keep `surface_only:=true` (default) or hide `/tomogram` and view `/global_points`.

## Housekeeping for GitHub

- Keep `planner_lib` prebuilt `.so` and `3rdparty/gtsam-4.1.1/install/` so users can run without rebuilding gtsam.
- Clean before committing: remove `build/`, `install/`, `log/`.
