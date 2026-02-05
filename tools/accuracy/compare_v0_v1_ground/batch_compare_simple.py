#!/usr/bin/env python3
"""
v0 vs v1 地图规划对比测试 - 简化版

直接加载地图，遍历所有测试用例，输出对比结果。

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy
    export LD_LIBRARY_PATH=$PWD/../../planner_lib:$PWD/../../planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
    python3 compare_v0_v1_ground/batch_compare_simple.py \
        --csv data/nyby_v0_test_pairs.csv \
        --v0 ../../rsc/tomogram/nyby_ground_cost_map_v0.bin \
        --v1 ../../rsc/tomogram/nyby_ground_cost_map_v1.bin

添加 --transform 参数可应用 v0 -> v1 坐标变换:
    python3 compare_v0_v1_ground/batch_compare_simple.py \
        --csv data/nyby_v0_test_pairs.csv \
        --v0 ../../rsc/tomogram/nyby_ground_cost_map_v0.bin \
        --v1 ../../rsc/tomogram/nyby_ground_cost_map_v1.bin \
        --transform
"""

import os
import sys
import csv
import json
import struct
import argparse
import time
from datetime import datetime
import numpy as np


# ============================================================================
# v0 -> v1 坐标变换参数 (基于用户手动标记的三组对应点精确计算)
# ============================================================================
# 用户在 rviz2 中标记的对应点:
#   点1: v0=(-47.3, -1.94, -0.101) -> v1=(-56.5, -16.8, -0.326)
#   点2: v0=(9.86, -105.0, 0.416) -> v1=(26.6, -101.0, 0.695)
#   点3: v0=(40.6, 6.65, -0.276) -> v1=(25.6, 15.5, -0.365)
# 
# XY 变换公式: v1 = R(θ) @ v0 + T
#   x1 = cos(θ) * x0 - sin(θ) * y0 + tx
#   y1 = sin(θ) * x0 + cos(θ) * y0 + ty
#
# Z 变换公式 (平面拟合，存在倾斜):
#   z1 = z0 + kx * x0 + ky * y0 + c_z
#
# 使用 Procrustes analysis + 平面拟合精确计算:
#   θ = 15.7781°, T = (-11.537, -2.216), XY RMSE = 0.32m
#   Z 变换: z1 = z0 + 0.001921*x0 - 0.003825*y0 - 0.141557, Z RMSE = 0m
V0_TO_V1_TRANSFORM = {
    'theta_deg': 15.778068,    # 旋转角度 (度)
    'theta_rad': 0.275379,     # 旋转角度 (弧度)
    'tx': -11.536994,          # x 方向平移 (m)
    'ty': -2.215985,           # y 方向平移 (m)
    'pivot_x': 0.0,            # 使用原点作为 pivot
    'pivot_y': 0.0,
    'cos_theta': 0.962322,     # cos(15.7781°)
    'sin_theta': 0.271912,     # sin(15.7781°)
    # Z 变换参数 (z1 = z0 + kx * x0 + ky * y0 + c_z)
    'kx_z': 0.001921,          # z 与 x0 的线性关系斜率
    'ky_z': -0.003825,         # z 与 y0 的线性关系斜率
    'c_z': -0.141557,          # z 偏移常数
}


def transform_v0_to_v1(x0: float, y0: float, z0: float = None) -> tuple:
    """
    将 v0 世界坐标变换到 v1 坐标系
    
    XY 变换公式 (pivot 为原点):
        x1 = cos(θ) * x0 - sin(θ) * y0 + tx
        y1 = sin(θ) * x0 + cos(θ) * y0 + ty
    
    Z 变换公式 (平面拟合):
        z1 = z0 + kx * x0 + ky * y0 + c_z
    
    Returns:
        如果 z0 为 None，返回 (x1, y1)
        否则返回 (x1, y1, z1)
    """
    T = V0_TO_V1_TRANSFORM
    cos_t, sin_t = T['cos_theta'], T['sin_theta']
    tx, ty = T['tx'], T['ty']
    
    x1 = cos_t * x0 - sin_t * y0 + tx
    y1 = sin_t * x0 + cos_t * y0 + ty
    
    if z0 is not None:
        kx_z, ky_z, c_z = T['kx_z'], T['ky_z'], T['c_z']
        z1 = z0 + kx_z * x0 + ky_z * y0 + c_z
        return x1, y1, z1
    
    return x1, y1

# 添加 C++ planner_lib 路径
CPP_PLANNER_ROOT = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
sys.path.insert(0, CPP_PLANNER_ROOT)

import ele_planner


def load_tomogram(bin_path: str) -> dict:
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
        'slice_heights': slice_heights,
        'data': data
    }


def create_planner(tomo: dict):
    """创建并初始化规划器"""
    data = tomo['data']
    trav = data[0]
    trav_gx = data[1]
    trav_gy = data[2]
    elev_g = np.nan_to_num(data[3], nan=-100)
    elev_c = np.nan_to_num(data[4], nan=1e6)
    
    # Gateway 计算
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

    planner = ele_planner.OfflineElePlanner(max_heading_rate=10.0, use_quintic=True)
    
    # init_map 需要 F-contiguous 的 float64 数组
    def to_f_contiguous(arr):
        arr = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
        return np.asfortranarray(arr)
    
    planner.init_map(
        20, 15, tomo['resolution'], tomo['n_slice'], 0.2,
        to_f_contiguous(trav),
        to_f_contiguous(elev_g),
        to_f_contiguous(elev_c),
        np.asfortranarray(gateway.reshape(-1, gateway.shape[-1]).astype(np.float64)),
        to_f_contiguous(trav_gy),
        to_f_contiguous(-trav_gx)
    )
    
    return planner


def plan_path(planner, tomo: dict, start: tuple, end: tuple) -> dict:
    """规划路径，返回结果字典"""
    center = tomo['center']
    offset = np.array([tomo['dim_x'] // 2, tomo['dim_y'] // 2], dtype=np.int32)
    slice_heights = tomo['slice_heights']
    n_slice = tomo['n_slice']
    resolution = tomo['resolution']
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    
    def z2slice(z):
        idx = np.searchsorted(slice_heights, z, side='right') - 1
        return max(0, min(idx, n_slice - 1))
    
    def pos2idx(pos):
        """世界坐标 -> 栅格索引 [y, x]，与 planner_wrapper.py 一致"""
        pos = pos - center
        idx = np.round(pos / resolution).astype(np.int32) + offset
        return np.array([idx[1], idx[0]], dtype=np.int32)  # [y, x]
    
    sx, sy, sz = start
    ex, ey, ez = end
    
    # z + 0.5 偏移（与 C++ 版 z_offset 一致）
    start_slice = z2slice(sz + 0.5)
    end_slice = z2slice(ez + 0.5)
    
    start_xy = pos2idx(np.array([sx, sy]))
    end_xy = pos2idx(np.array([ex, ey]))
    
    # plan() 需要 [3, 1] 形状的列向量
    start_idx = np.array([[start_slice], [start_xy[0]], [start_xy[1]]], dtype=np.int32)
    end_idx = np.array([[end_slice], [end_xy[0]], [end_xy[1]]], dtype=np.int32)
    
    # 边界检查（使用 flatten 后的索引）
    start_flat = start_idx.flatten()
    end_flat = end_idx.flatten()
    if (start_flat[1] < 0 or start_flat[1] >= dim_y or
        start_flat[2] < 0 or start_flat[2] >= dim_x or
        end_flat[1] < 0 or end_flat[1] >= dim_y or
        end_flat[2] < 0 or end_flat[2] >= dim_x):
        return {'success': False, 'error': 'out_of_bounds', 'path_length': None, 'time_ms': 0}
    
    t0 = time.time()
    try:
        success = planner.plan(start_idx, end_idx, True)
    except Exception as e:
        return {'success': False, 'error': str(e), 'path_length': None, 'time_ms': 0}
    
    elapsed_ms = (time.time() - t0) * 1000
    
    if not success:
        return {'success': False, 'error': 'path_not_found', 'path_length': None, 'time_ms': elapsed_ms}
    
    # 规划成功，尝试获取路径长度
    path_length = None
    num_points = 0
    try:
        # 使用 get_debug_path() 获取路径 (返回 Nx3 数组: [slice, x, y] 栅格坐标)
        debug_path = planner.get_debug_path()
        if debug_path.shape[0] > 0:
            num_points = debug_path.shape[0]
            # 计算栅格路径长度 (使用 xy 坐标)
            xy_path = debug_path[:, 1:3]  # 取 x, y 列
            diffs = np.diff(xy_path, axis=0)
            path_length = float(np.sum(np.linalg.norm(diffs, axis=1))) * resolution
    except Exception:
        pass  # 忽略轨迹获取错误，规划仍然成功
    
    return {'success': True, 'path_length': path_length, 'time_ms': elapsed_ms, 'num_points': num_points}


def load_test_cases(csv_path: str) -> list:
    """加载测试用例"""
    cases = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('id') or row.get('case_id')
            start = (float(row['start_x']), float(row['start_y']), float(row['start_z']))
            end_x = row.get('end_x') or row.get('goal_x')
            end_y = row.get('end_y') or row.get('goal_y')
            end_z = row.get('end_z') or row.get('goal_z')
            end = (float(end_x), float(end_y), float(end_z))
            cases.append({'id': case_id, 'start': start, 'end': end})
    return cases


def main():
    parser = argparse.ArgumentParser(description='v0 vs v1 地图规划对比测试')
    parser.add_argument('--csv', required=True, help='测试用例 CSV 文件')
    parser.add_argument('--v0', required=True, help='v0 地图 bin 文件')
    parser.add_argument('--v1', required=True, help='v1 地图 bin 文件')
    parser.add_argument('--output', help='输出 JSON 文件（可选）')
    parser.add_argument('--limit', type=int, default=0, help='限制测试用例数（0=全部）')
    parser.add_argument('--transform', action='store_true',
                        help='应用 v0->v1 坐标变换（旋转+平移）后再在 v1 地图上规划')
    args = parser.parse_args()
    
    print("=" * 60)
    print("v0 vs v1 地图规划对比测试")
    if args.transform:
        print("  [启用 v0->v1 坐标变换]")
        print(f"  旋转: {V0_TO_V1_TRANSFORM['theta_deg']:.3f}°")
        print(f"  平移: ({V0_TO_V1_TRANSFORM['tx']:.2f}, {V0_TO_V1_TRANSFORM['ty']:.2f}) m")
    print("=" * 60)
    
    # 加载测试用例
    cases = load_test_cases(args.csv)
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"测试用例: {len(cases)} 个")
    
    # 加载 v0 地图
    print(f"\n加载 v0 地图: {args.v0}")
    tomo_v0 = load_tomogram(args.v0)
    print(f"  dim: {tomo_v0['dim_x']}x{tomo_v0['dim_y']}, n_slice: {tomo_v0['n_slice']}")
    planner_v0 = create_planner(tomo_v0)
    print("  规划器初始化完成")
    
    # 加载 v1 地图
    print(f"\n加载 v1 地图: {args.v1}")
    tomo_v1 = load_tomogram(args.v1)
    print(f"  dim: {tomo_v1['dim_x']}x{tomo_v1['dim_y']}, n_slice: {tomo_v1['n_slice']}")
    planner_v1 = create_planner(tomo_v1)
    print("  规划器初始化完成")
    
    # 运行测试
    print(f"\n开始测试...")
    results = []
    v0_success = 0
    v1_success = 0
    both_success = 0
    both_fail = 0
    v0_only = 0
    v1_only = 0
    
    for i, case in enumerate(cases):
        case_id = case['id']
        start = case['start']
        end = case['end']
        
        # v0 规划（使用原始坐标）
        r0 = plan_path(planner_v0, tomo_v0, start, end)
        # 规划失败后重新创建规划器（规划器有状态问题）
        if not r0['success']:
            planner_v0 = create_planner(tomo_v0)
        
        # v1 规划（根据 --transform 决定是否变换坐标）
        if args.transform:
            # 变换 start 和 end 坐标 (包括 z)
            start_x1, start_y1, start_z1 = transform_v0_to_v1(start[0], start[1], start[2])
            end_x1, end_y1, end_z1 = transform_v0_to_v1(end[0], end[1], end[2])
            start_v1 = (start_x1, start_y1, start_z1)
            end_v1 = (end_x1, end_y1, end_z1)
            r1 = plan_path(planner_v1, tomo_v1, start_v1, end_v1)
        else:
            r1 = plan_path(planner_v1, tomo_v1, start, end)
        if not r1['success']:
            planner_v1 = create_planner(tomo_v1)
        
        # 统计
        if r0['success']:
            v0_success += 1
        if r1['success']:
            v1_success += 1
        
        if r0['success'] and r1['success']:
            both_success += 1
        elif r0['success'] and not r1['success']:
            v0_only += 1
        elif not r0['success'] and r1['success']:
            v1_only += 1
        else:
            both_fail += 1
        
        results.append({
            'id': case_id,
            'start': start,
            'end': end,
            'v0': r0,
            'v1': r1
        })
        
        # 进度
        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"  进度: {i+1}/{len(cases)}, v0成功: {v0_success}, v1成功: {v1_success}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总用例数: {len(cases)}")
    print(f"v0 成功: {v0_success} ({100*v0_success/len(cases):.1f}%)")
    print(f"v1 成功: {v1_success} ({100*v1_success/len(cases):.1f}%)")
    print(f"两者都成功: {both_success}")
    print(f"仅 v0 成功: {v0_only}")
    print(f"仅 v1 成功: {v1_only}")
    print(f"两者都失败: {both_fail}")
    
    # 路径长度对比（仅两者都成功的用例）
    if both_success > 0:
        v0_lengths = [r['v0']['path_length'] for r in results if r['v0']['success'] and r['v1']['success'] and r['v0']['path_length'] is not None]
        v1_lengths = [r['v1']['path_length'] for r in results if r['v0']['success'] and r['v1']['success'] and r['v1']['path_length'] is not None]
        print(f"\n路径长度对比（{both_success}个两者都成功的用例）:")
        if v0_lengths and v1_lengths:
            print(f"  v0 平均: {np.mean(v0_lengths):.2f} m")
            print(f"  v1 平均: {np.mean(v1_lengths):.2f} m")
            print(f"  v1/v0 比值: {np.mean(v1_lengths)/np.mean(v0_lengths):.3f}")
        else:
            print(f"  无法计算路径长度（path_length 为 None）")
    
    # 保存结果
    if args.output:
        output_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'csv': args.csv,
                'v0_bin': args.v0,
                'v1_bin': args.v1,
                'total_cases': len(cases),
                'transform_enabled': args.transform,
                'transform_params': V0_TO_V1_TRANSFORM if args.transform else None
            },
            'summary': {
                'total': len(cases),
                'v0_success': v0_success,
                'v1_success': v1_success,
                'both_success': both_success,
                'v0_only': v0_only,
                'v1_only': v1_only,
                'both_fail': both_fail,
                'v0_success_rate': 100 * v0_success / len(cases),
                'v1_success_rate': 100 * v1_success / len(cases)
            },
            'results': results
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n结果已保存: {args.output}")
    
    print("\n完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
