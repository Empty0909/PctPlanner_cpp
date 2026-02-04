#!/usr/bin/env python3
"""
在 v0 地图上生成测试用例（随机采样可通行栅格，不验证路径可达性）

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy/compare_v0_v1_ground
    python3 generate_v0_test_cases.py --num-cases 1000 --output ../data/v0_test_cases.csv
"""

import os
import sys
import csv
import struct
import argparse
import numpy as np

DEFAULT_V0_BIN = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v0.bin"
DEFAULT_OUTPUT_CSV = "../data/v0_test_cases.csv"
TRAVERSABLE_THRESHOLD = 10.0


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


def main():
    parser = argparse.ArgumentParser(description='在 v0 地图上生成测试用例（随机采样）')
    parser.add_argument('--bin', default=DEFAULT_V0_BIN, help='地图 bin 文件路径')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_CSV, help='输出 CSV 路径')
    parser.add_argument('--num-cases', type=int, default=1000, help='用例数量')
    parser.add_argument('--min-distance', type=float, default=20.0, help='起终点最小距离（米）')
    parser.add_argument('--max-distance', type=float, default=300.0, help='起终点最大距离（米）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("在地图上生成测试用例（随机采样可通行栅格）")
    print("=" * 60)
    
    # 加载地图
    print(f"\n加载地图: {args.bin}")
    tomo = load_tomogram_bin(args.bin)
    print(f"  维度: {tomo['dim_x']} x {tomo['dim_y']} x {tomo['n_slice']}")
    print(f"  分辨率: {tomo['resolution']} m")
    print(f"  中心: ({tomo['center'][0]:.2f}, {tomo['center'][1]:.2f})")
    print(f"  层高度: {tomo['slice_heights']}")
    
    # 获取可通行坐标（使用第一层）
    trav = tomo['data'][0][0]  # shape: [dim_x, dim_y]
    elev_g = tomo['data'][3][0]
    valid_mask = (trav < TRAVERSABLE_THRESHOLD) & ~np.isnan(elev_g) & (elev_g > -50)
    coords = np.argwhere(valid_mask)  # shape: [N, 2], 每行是 (row, col)
    print(f"  可通行栅格: {len(coords)}")
    
    # 坐标转换函数
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    resolution = tomo['resolution']
    center = tomo['center']
    slice_heights = tomo['slice_heights']
    
    def grid_to_world(r, c):
        """栅格坐标转世界坐标"""
        wx = (r - dim_x // 2) * resolution + center[0]
        wy = (c - dim_y // 2) * resolution + center[1]
        wz = slice_heights[0]
        return wx, wy, wz
    
    # 生成测试用例
    print(f"\n生成 {args.num_cases} 个测试用例...")
    print(f"  距离范围: [{args.min_distance}, {args.max_distance}] m")
    
    test_cases = []
    attempts = 0
    max_attempts = args.num_cases * 100
    
    while len(test_cases) < args.num_cases and attempts < max_attempts:
        attempts += 1
        
        # 随机选择两个可通行栅格
        idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
        r1, c1 = coords[idx1]
        r2, c2 = coords[idx2]
        
        start = grid_to_world(r1, c1)
        end = grid_to_world(r2, c2)
        
        # 检查距离
        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        if dist < args.min_distance or dist > args.max_distance:
            continue
        
        test_cases.append({
            'case_id': len(test_cases) + 1,
            'start_x': float(start[0]),
            'start_y': float(start[1]),
            'start_z': float(start[2]),
            'start_yaw': 0.0,
            'goal_x': float(end[0]),
            'goal_y': float(end[1]),
            'goal_z': float(end[2]),
            'goal_yaw': 0.0,
        })
        
        if len(test_cases) % 1000 == 0:
            print(f"  已生成 {len(test_cases)} 个用例...")
    
    print(f"\n共生成 {len(test_cases)} 个用例 (尝试 {attempts} 次)")
    
    # 统计信息
    if test_cases:
        xs = [c['start_x'] for c in test_cases] + [c['goal_x'] for c in test_cases]
        ys = [c['start_y'] for c in test_cases] + [c['goal_y'] for c in test_cases]
        print(f"\n坐标范围:")
        print(f"  X: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"  Y: [{min(ys):.2f}, {max(ys):.2f}]")
    
    # 保存 CSV
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['case_id', 'start_x', 'start_y', 'start_z', 'start_yaw',
                      'goal_x', 'goal_y', 'goal_z', 'goal_yaw']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_cases)
    
    print(f"\n已保存: {args.output}")
    print("完成!")


if __name__ == '__main__':
    main()
