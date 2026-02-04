#!/usr/bin/env python3
"""
快速生成高质量测试用例（验证路径可达性）

在单进程中运行规划验证，定期做垃圾回收避免 OOM。
比 subprocess 方式快得多。

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy/compare_v0_v1_ground
    python3 generate_valid_cases_fast.py \
        --bin /path/to/map.bin \
        --num-cases 1000 \
        --output ../data/valid_cases.csv
"""

import os
import sys
import csv
import gc
import struct
import argparse
import numpy as np
from datetime import datetime

# 配置
DEFAULT_BIN = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v0.bin"
DEFAULT_OUTPUT = "../data/valid_cases.csv"
TRAVERSABLE_THRESHOLD = 10.0

# 设置库路径
CPP_LIB = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"
os.environ['LD_LIBRARY_PATH'] = f"{CPP_LIB}:{CPP_GTSAM}:" + os.environ.get('LD_LIBRARY_PATH', '')
sys.path.insert(0, CPP_LIB)


def verify_cpp_library():
    """确认使用的是 C++ 版本的库"""
    global a_star, ele_planner, traj_opt
    import a_star
    import ele_planner
    import traj_opt
    
    for name, mod in [('a_star', a_star), ('ele_planner', ele_planner), ('traj_opt', traj_opt)]:
        module_path = getattr(mod, '__file__', '')
        if 'PctPlanner_Cpp' not in module_path:
            raise ImportError(f"警告: {name} 不是 C++ 版本: {module_path}")
        print(f"  ✓ 使用 C++ 库 {name}: {module_path}")


def load_tomogram_bin(bin_path: str) -> dict:
    """加载 tomogram bin 文件"""
    with open(bin_path, 'rb') as f:
        magic = f.read(4).decode('ascii')
        if magic != 'TMG1':
            raise ValueError(f"Invalid magic: {magic}")
        
        version = struct.unpack('H', f.read(2))[0]
        precision_mode = struct.unpack('H', f.read(2))[0]
        n_slice = struct.unpack('I', f.read(4))[0]
        dim_x = struct.unpack('I', f.read(4))[0]
        dim_y = struct.unpack('I', f.read(4))[0]
        resolution = struct.unpack('f', f.read(4))[0]
        center_x = struct.unpack('f', f.read(4))[0]
        center_y = struct.unpack('f', f.read(4))[0]
        slice_h0 = struct.unpack('f', f.read(4))[0]
        slice_dh = struct.unpack('f', f.read(4))[0]
        
        scalar_size = 2 if precision_mode == 0 else 4
        dtype = np.float16 if precision_mode == 0 else np.float32
        
        slice_heights_bytes = f.read(n_slice * scalar_size)
        slice_heights = np.frombuffer(slice_heights_bytes, dtype=dtype).astype(np.float32)
        
        data_size = 5 * n_slice * dim_x * dim_y * scalar_size
        data_bytes = f.read(data_size)
        data = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
        data = data.reshape(5, n_slice, dim_x, dim_y)
    
    return {
        'n_slice': n_slice,
        'dim_x': dim_x,
        'dim_y': dim_y,
        'resolution': float(resolution),
        'center': np.array([center_x, center_y]),
        'slice_heights': slice_heights.tolist(),
        'data': data
    }


def create_planner(tomo: dict, use_quintic=True, max_heading_rate=10.0):
    """创建规划器"""
    resolution = tomo['resolution']
    n_slice = tomo['n_slice']
    
    data = tomo['data']
    trav = data[0]
    trav_gx = data[1]
    trav_gy = data[2]
    elev_g = np.nan_to_num(data[3], nan=-100)
    elev_c = np.nan_to_num(data[4], nan=1e6)
    
    diff_t = trav[1:] - trav[:-1]
    diff_g = np.abs(elev_g[1:] - elev_g[:-1])

    gateway_up = np.zeros_like(trav, dtype=bool)
    mask_t = diff_t < -8.0
    mask_g = (diff_g < 0.1) & (~np.isnan(data[3][1:]))
    gateway_up[:-1] = np.logical_and(mask_t, mask_g)

    gateway_dn = np.zeros_like(trav, dtype=bool)
    mask_t = diff_t > 8.0
    mask_g = (diff_g < 0.1) & (~np.isnan(data[3][:-1]))
    gateway_dn[1:] = np.logical_and(mask_t, mask_g)
    
    gateway = np.zeros_like(trav, dtype=np.int32)
    gateway[gateway_up] = 2
    gateway[gateway_dn] = -2

    planner = ele_planner.OfflineElePlanner(
        max_heading_rate=max_heading_rate, use_quintic=use_quintic
    )
    
    planner.init_map(
        20, 15, resolution, n_slice, 0.2,
        trav.reshape(-1, trav.shape[-1]).astype(np.double),
        elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
        elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
        gateway.reshape(-1, gateway.shape[-1]),
        trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
        -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
    )
    
    return planner


def plan_and_validate(planner, tomo: dict, start: tuple, goal: tuple) -> tuple:
    """
    执行规划并返回 (success, path_length)
    """
    resolution = tomo['resolution']
    center = tomo['center']
    n_slice = tomo['n_slice']
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    slice_heights = tomo['slice_heights']
    offset = np.array([dim_x // 2, dim_y // 2], dtype=np.int32)
    
    def z2slice(z):
        idx = np.searchsorted(slice_heights, z, side='right') - 1
        return max(0, idx)
    
    def pos2idx(pos):
        pos = pos - center
        idx = np.round(pos / resolution).astype(np.int32) + offset
        return np.array([idx[1], idx[0]], dtype=np.int32)
    
    sx, sy, sz = start[:3]
    ex, ey, ez = goal[:3]
    
    start_slice = np.clip(z2slice(sz + 0.5), 0, n_slice - 1).astype(np.int32)
    end_slice = np.clip(z2slice(ez + 0.5), 0, n_slice - 1).astype(np.int32)
    
    start_idx = np.zeros(3, dtype=np.int32)
    end_idx = np.zeros(3, dtype=np.int32)
    
    start_idx[0] = start_slice
    start_idx[1:] = pos2idx(np.array([sx, sy]))
    
    end_idx[0] = end_slice
    end_idx[1:] = pos2idx(np.array([ex, ey]))
    
    try:
        planner.plan(start_idx, end_idx, True)
    except Exception:
        return False, None
    
    path_finder = planner.get_path_finder()
    path = path_finder.get_result_matrix()
    if len(path) == 0:
        return False, None
    
    optimizer = planner.get_trajectory_optimizer_wnoj()
    traj_raw = optimizer.get_result_matrix()
    heights = optimizer.get_heights()
    
    y_idx = (traj_raw.shape[-1] - 1) // 2
    traj_3d = np.stack([traj_raw[:, 0], traj_raw[:, y_idx], heights / resolution], axis=1)
    
    offset_world = np.array([dim_y // 2, dim_x // 2, 0])
    traj_3d = traj_3d - offset_world
    traj_3d[:, :2] = traj_3d[:, [1, 0]] * resolution + center
    traj_3d[:, 2] = heights
    
    diffs = np.diff(traj_3d, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    return True, path_length


def generate_and_validate_cases(tomo: dict, planner, num_cases: int,
                                 min_distance: float, max_distance: float,
                                 seed: int, gc_interval: int = 50) -> list:
    """生成并验证测试用例"""
    np.random.seed(seed)
    
    # 获取可通行坐标
    trav = tomo['data'][0][0]
    elev_g = tomo['data'][3][0]
    valid_mask = (trav < TRAVERSABLE_THRESHOLD) & ~np.isnan(elev_g) & (elev_g > -50)
    coords = np.argwhere(valid_mask)
    
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    resolution = tomo['resolution']
    center = tomo['center']
    slice_heights = tomo['slice_heights']
    
    def grid_to_world(r, c):
        wx = (r - dim_x // 2) * resolution + center[0]
        wy = (c - dim_y // 2) * resolution + center[1]
        wz = slice_heights[0]
        return wx, wy, wz
    
    print(f"  可通行点数: {len(coords)}")
    
    valid_cases = []
    attempts = 0
    validated = 0
    max_attempts = num_cases * 100  # 假设成功率约 10-30%
    
    while len(valid_cases) < num_cases and attempts < max_attempts:
        attempts += 1
        
        # 随机选择两个点
        idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
        r1, c1 = coords[idx1]
        r2, c2 = coords[idx2]
        
        start = grid_to_world(r1, c1)
        goal = grid_to_world(r2, c2)
        
        # 检查距离
        dist = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        if dist < min_distance or dist > max_distance:
            continue
        
        # 验证路径
        validated += 1
        start_pose = (*start, 0.0)  # x, y, z, yaw
        goal_pose = (*goal, 0.0)
        
        success, path_length = plan_and_validate(planner, tomo, start_pose, goal_pose)
        
        if success:
            valid_cases.append({
                'case_id': len(valid_cases) + 1,
                'start_x': float(start[0]),
                'start_y': float(start[1]),
                'start_z': float(start[2]),
                'start_yaw': 0.0,
                'goal_x': float(goal[0]),
                'goal_y': float(goal[1]),
                'goal_z': float(goal[2]),
                'goal_yaw': 0.0,
                'path_length': path_length,
            })
            
            # 输出进度
            if len(valid_cases) % 10 == 0:
                rate = len(valid_cases) / validated * 100
                print(f"  进度: {len(valid_cases)}/{num_cases} 有效 ({validated} 已验证, 成功率 {rate:.1f}%)")
        
        # 定期垃圾回收
        if validated % gc_interval == 0:
            gc.collect()
    
    final_rate = len(valid_cases) / validated * 100 if validated > 0 else 0
    print(f"\n  总结: {len(valid_cases)} 有效用例 / {validated} 已验证 (成功率 {final_rate:.1f}%)")
    
    return valid_cases


def save_cases(cases: list, output_path: str):
    """保存用例到 CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fieldnames = ['case_id', 'start_x', 'start_y', 'start_z', 'start_yaw',
                  'goal_x', 'goal_y', 'goal_z', 'goal_yaw', 'path_length']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    
    print(f"\n已保存 {len(cases)} 个用例到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='快速生成高质量测试用例（验证路径可达性）')
    parser.add_argument('--bin', default=DEFAULT_BIN, help='地图 bin 文件路径')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='输出 CSV 路径')
    parser.add_argument('--num-cases', type=int, default=1000, help='目标有效用例数')
    parser.add_argument('--min-distance', type=float, default=20.0, help='起终点最小距离（米）')
    parser.add_argument('--max-distance', type=float, default=300.0, help='起终点最大距离（米）')
    parser.add_argument('--gc-interval', type=int, default=50, help='垃圾回收间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("快速生成高质量测试用例（单进程版）")
    print("=" * 60)
    print(f"  地图: {args.bin}")
    print(f"  目标用例数: {args.num_cases}")
    print(f"  距离范围: [{args.min_distance}, {args.max_distance}] m")
    print(f"  垃圾回收间隔: {args.gc_interval}")
    print()
    
    # 验证 C++ 库
    verify_cpp_library()
    
    # 加载地图
    print("\n加载地图...")
    tomo = load_tomogram_bin(args.bin)
    print(f"  维度: {tomo['dim_x']} x {tomo['dim_y']} x {tomo['n_slice']}")
    
    # 创建规划器
    print("\n创建规划器...")
    planner = create_planner(tomo)
    print("  ✓ 规划器创建成功")
    
    # 生成并验证用例
    print(f"\n开始生成 {args.num_cases} 个有效用例...")
    start_time = datetime.now()
    
    valid_cases = generate_and_validate_cases(
        tomo, planner,
        num_cases=args.num_cases,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        seed=args.seed,
        gc_interval=args.gc_interval
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n生成耗时: {elapsed:.1f} 秒")
    
    # 保存结果
    if valid_cases:
        save_cases(valid_cases, args.output)
    else:
        print("\n警告: 没有生成有效用例!")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
