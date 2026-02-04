#!/usr/bin/env python3
"""
在单个地图上规划一批测试用例（子进程工作脚本）

由 run_v0_v1_compare_batch.py 调用，每批在独立进程中运行以避免 OOM。

使用方法（通常由主脚本调用）:
    python3 plan_on_map_batch.py --input batch.csv --output result.json --bin map.bin
"""

import os
import sys
import csv
import json
import struct
import argparse
import time
import numpy as np

# ============================================================
# 关键：确保使用 C++ 版本的规划库
# ============================================================
CPP_PLANNER_ROOT = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM_LIB = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"

# 清除可能存在的 Python 版本路径，确保 C++ 版本优先
sys.path = [p for p in sys.path if 'PctPlanner_py' not in p]
sys.path.insert(0, CPP_PLANNER_ROOT)

# 设置动态库搜索路径
os.environ['LD_LIBRARY_PATH'] = f"{CPP_PLANNER_ROOT}:{CPP_GTSAM_LIB}:" + os.environ.get('LD_LIBRARY_PATH', '')

# 导入 C++ 版本的库
import a_star
import ele_planner
import traj_opt

# 验证加载的是 C++ 版本
def verify_cpp_libs():
    """验证加载的是 C++ 版本的库"""
    for name, mod in [('a_star', a_star), ('ele_planner', ele_planner), ('traj_opt', traj_opt)]:
        mod_path = getattr(mod, '__file__', None)
        if mod_path is None:
            raise RuntimeError(f"无法获取 {name} 模块路径")
        if 'PctPlanner_Cpp' not in mod_path:
            raise RuntimeError(f"{name} 不是从 C++ 版本加载的: {mod_path}")
    return True

verify_cpp_libs()


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


def create_planner(tomo, use_quintic=True, max_heading_rate=10.0):
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


def plan_and_get_length(planner, meta, start, end):
    """执行规划并返回路径长度"""
    resolution = meta['resolution']
    center = meta['center']
    n_slice = meta['n_slice']
    dim_x = meta['dim_x']
    dim_y = meta['dim_y']
    slice_heights = meta['slice_heights']
    offset = np.array([dim_x // 2, dim_y // 2], dtype=np.int32)
    
    def z2slice(z):
        idx = np.searchsorted(slice_heights, z, side='right') - 1
        return max(0, idx)
    
    def pos2idx(pos):
        pos = pos - center
        idx = np.round(pos / resolution).astype(np.int32) + offset
        return np.array([idx[1], idx[0]], dtype=np.int32)
    
    sx, sy, sz = start
    ex, ey, ez = end
    
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
        return None
    
    path_finder = planner.get_path_finder()
    path = path_finder.get_result_matrix()
    if len(path) == 0:
        return None
    
    optimizer = planner.get_trajectory_optimizer_wnoj()
    traj_raw = optimizer.get_result_matrix()
    heights = optimizer.get_heights()
    
    y_idx = (traj_raw.shape[-1] - 1) // 2
    traj_3d = np.stack([traj_raw[:, 0], traj_raw[:, y_idx], heights / resolution], axis=1)
    
    # 坐标转换
    offset_world = np.array([dim_y // 2, dim_x // 2, 0])
    traj_3d = traj_3d - offset_world
    traj_3d[:, :2] = traj_3d[:, [1, 0]] * resolution + center
    traj_3d[:, 2] = heights
    
    # 计算长度
    diffs = np.diff(traj_3d, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    return path_length


def main():
    parser = argparse.ArgumentParser(description='在地图上规划批次')
    parser.add_argument('--input', required=True, help='输入 CSV')
    parser.add_argument('--output', required=True, help='输出 JSON')
    parser.add_argument('--bin', required=True, help='地图 bin 路径')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 加载地图
    tomo = load_tomogram_bin(args.bin)
    meta = {k: tomo[k] for k in ['n_slice', 'dim_x', 'dim_y', 'resolution', 'center', 'slice_heights']}
    
    # 创建规划器
    planner = create_planner(tomo)
    
    # 读取测试用例
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    
    # 规划
    results = {}
    success_count = 0
    
    for case in cases:
        # 兼容两种 CSV 格式：case_id/id, goal_x/end_x
        case_id = case.get('case_id') or case.get('id')
        start = (
            float(case['start_x']), 
            float(case['start_y']), 
            float(case['start_z'])
        )
        end = (
            float(case.get('goal_x') or case.get('end_x')), 
            float(case.get('goal_y') or case.get('end_y')), 
            float(case.get('goal_z') or case.get('end_z'))
        )
        
        t0 = time.time()
        path_length = plan_and_get_length(planner, meta, start, end)
        elapsed = (time.time() - t0) * 1000
        
        if path_length is not None:
            success_count += 1
            results[case_id] = {
                'success': True,
                'path_length': path_length,
                'time_ms': elapsed
            }
        else:
            results[case_id] = {
                'success': False,
                'path_length': None,
                'time_ms': elapsed
            }
    
    total_time = time.time() - start_time
    
    # 输出结果
    output = {
        'summary': {
            'total': len(cases),
            'success': success_count,
            'failed': len(cases) - success_count,
            'success_rate': success_count / len(cases) * 100 if cases else 0,
            'total_time_sec': total_time
        },
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
