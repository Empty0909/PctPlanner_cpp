#!/usr/bin/env python3
"""
生成 10K 组起终点测试数据

从 Tomogram 中采样可通行区域的坐标作为起终点，确保测试点在有效范围内。
输出 CSV 文件：id, start_x, start_y, start_z, end_x, end_y, end_z
"""

import os
import sys
import pickle
import argparse
import numpy as np
import csv

# 默认路径
DEFAULT_PICKLE_PATH = "/home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle"
DEFAULT_OUTPUT_CSV = "test_cases.csv"
DEFAULT_NUM_CASES = 10000

# 可通行代价阈值（小于此值认为可通行）
TRAVERSABLE_THRESHOLD = 10.0


def load_tomogram_pickle(pickle_path: str) -> dict:
    """加载 Python 版本的 tomogram pickle 文件"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_traversable_coords(data: dict, layer: int = 0) -> np.ndarray:
    """
    获取指定层的可通行栅格坐标
    
    Returns:
        coords: Nx2 数组，每行是 (row, col) 索引
    """
    tomogram = np.asarray(data['data'], dtype=np.float32)
    # tomogram shape: [5, n_slice, dim_x, dim_y]
    # layer 0 是 trav (可通行代价)
    # layer 3 是 elev_g (地面高度)
    trav = tomogram[0]  # shape: [n_slice, dim_x, dim_y]
    elev_g = tomogram[3]  # shape: [n_slice, dim_x, dim_y]
    
    # 获取指定层的可通行栅格
    trav_layer = trav[layer]  # shape: [dim_x, dim_y]
    elev_layer = elev_g[layer]
    
    # 找出代价小于阈值的栅格
    traversable = trav_layer < TRAVERSABLE_THRESHOLD
    
    # 增加地面高度有效性检查：排除无效高度（-99.4或NaN）
    valid_ground = ~np.isnan(elev_layer) & (elev_layer > -50)
    
    # 同时满足可通行和有效地面
    valid_mask = traversable & valid_ground
    
    # 获取坐标
    coords = np.argwhere(valid_mask)  # shape: [N, 2], (row, col)
    
    return coords, trav_layer.shape


def grid_to_world(row: int, col: int, z_layer: int, 
                  resolution: float, center: np.ndarray, 
                  dim: tuple, slice_heights: list) -> tuple:
    """
    栅格坐标转世界坐标
    
    从 argwhere 得到的 (row, col) 对应 tomogram[layer][row][col]
    tomogram shape: [n_slice, dim_x, dim_y]
    所以 row = dim_x 方向的索引, col = dim_y 方向的索引
    
    与 planner_wrapper.pos2idx 互逆：
    pos2idx: [x, y] -> [y_grid, x_grid] (注意交换)
    所以: row 对应 x_grid, col 对应 y_grid? 不对...
    
    实际上看 planner_wrapper:
    - map_dim = [dim_x, dim_y]
    - offset = [dim_x // 2, dim_y // 2]
    - pos2idx: idx = (pos - center) / res + offset, 然后 [idx[1], idx[0]]
    
    所以 pos2idx 输出是 [y_grid, x_grid]，其中:
    - y_grid = (y - center[1]) / res + offset[1] = (y - center_y) / res + dim_y/2
    - x_grid = (x - center[0]) / res + offset[0] = (x - center_x) / res + dim_x/2
    
    逆变换:
    - x = (x_grid - dim_x/2) * res + center_x
    - y = (y_grid - dim_y/2) * res + center_y
    
    由于 argwhere 返回 [row, col] = [dim_x_idx, dim_y_idx]:
    - row 对应 x_grid
    - col 对应 y_grid
    """
    dim_x, dim_y = dim
    
    # row 对应 x 方向，col 对应 y 方向
    world_x = (row - dim_x // 2) * resolution + center[0]
    world_y = (col - dim_y // 2) * resolution + center[1]
    
    # z 坐标从 slice_heights 获取
    if z_layer < len(slice_heights):
        world_z = slice_heights[z_layer]
    else:
        world_z = 0.0
    
    return world_x, world_y, world_z


def generate_test_cases(data: dict, num_cases: int, 
                        min_distance: float = 2.0,
                        max_distance: float = 30.0) -> list:
    """
    生成测试用例
    
    Args:
        data: tomogram 数据字典
        num_cases: 生成的用例数量
        min_distance: 起终点最小距离（米）
        max_distance: 起终点最大距离（米）
    
    Returns:
        test_cases: 列表，每个元素是 (id, start_x, start_y, start_z, end_x, end_y, end_z)
    """
    resolution = float(data['resolution'])
    center = np.asarray(data['center'], dtype=np.float64)
    slice_heights = data['slice_heights']
    n_slice = len(slice_heights)
    
    print(f"Tomogram 信息:")
    print(f"  分辨率: {resolution}")
    print(f"  中心: {center}")
    print(f"  层数: {n_slice}")
    print(f"  层高度: {slice_heights}")
    
    # 收集所有层的可通行坐标
    all_coords = []  # [(layer, row, col), ...]
    for layer in range(n_slice):
        coords, dim = get_traversable_coords(data, layer)
        for row, col in coords:
            all_coords.append((layer, row, col, dim))
    
    print(f"  可通行栅格总数: {len(all_coords)}")
    
    if len(all_coords) < 2:
        raise ValueError("可通行区域太少，无法生成测试用例")
    
    # 将坐标转换为世界坐标
    world_coords = []
    for layer, row, col, dim in all_coords:
        wx, wy, wz = grid_to_world(row, col, layer, resolution, center, dim, slice_heights)
        world_coords.append((wx, wy, wz, layer))
    world_coords = np.array(world_coords)
    
    print(f"  世界坐标范围:")
    print(f"    X: [{world_coords[:, 0].min():.2f}, {world_coords[:, 0].max():.2f}]")
    print(f"    Y: [{world_coords[:, 1].min():.2f}, {world_coords[:, 1].max():.2f}]")
    print(f"    Z: [{world_coords[:, 2].min():.2f}, {world_coords[:, 2].max():.2f}]")
    
    # 随机采样起终点
    test_cases = []
    attempts = 0
    max_attempts = num_cases * 100
    
    np.random.seed(42)  # 固定随机种子，保证可复现
    
    while len(test_cases) < num_cases and attempts < max_attempts:
        attempts += 1
        
        # 随机选择两个点
        idx1, idx2 = np.random.choice(len(world_coords), 2, replace=False)
        
        start = world_coords[idx1]
        end = world_coords[idx2]
        
        # 计算距离
        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        
        # 检查距离是否在范围内
        if min_distance <= dist <= max_distance:
            case_id = len(test_cases) + 1
            test_cases.append({
                'id': case_id,
                'start_x': float(start[0]),
                'start_y': float(start[1]),
                'start_z': float(start[2]),
                'end_x': float(end[0]),
                'end_y': float(end[1]),
                'end_z': float(end[2]),
            })
            
            if len(test_cases) % 1000 == 0:
                print(f"  已生成 {len(test_cases)} 个用例...")
    
    print(f"  共生成 {len(test_cases)} 个用例 (尝试 {attempts} 次)")
    
    return test_cases


def save_to_csv(test_cases: list, output_path: str):
    """保存为 CSV 格式"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'start_x', 'start_y', 'start_z', 
                                                'end_x', 'end_y', 'end_z'])
        writer.writeheader()
        writer.writerows(test_cases)
    print(f"已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成 PctPlanner 测试用例')
    parser.add_argument('--pickle', type=str, default=DEFAULT_PICKLE_PATH,
                        help='Python tomogram pickle 文件路径')
    parser.add_argument('--output-csv', type=str, default=DEFAULT_OUTPUT_CSV,
                        help='输出 CSV 文件路径')
    parser.add_argument('--num-cases', type=int, default=DEFAULT_NUM_CASES,
                        help='生成的测试用例数量')
    parser.add_argument('--min-distance', type=float, default=2.0,
                        help='起终点最小距离（米）')
    parser.add_argument('--max-distance', type=float, default=30.0,
                        help='起终点最大距离（米）')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果输出路径是相对路径，则相对于脚本目录
    if not os.path.isabs(args.output_csv):
        args.output_csv = os.path.join(script_dir, args.output_csv)
    
    print("=" * 60)
    print("PctPlanner 测试用例生成器")
    print("=" * 60)
    
    # 加载 tomogram
    print(f"\n加载 Tomogram: {args.pickle}")
    data = load_tomogram_pickle(args.pickle)
    
    # 生成测试用例
    print(f"\n生成 {args.num_cases} 个测试用例...")
    test_cases = generate_test_cases(
        data, args.num_cases,
        min_distance=args.min_distance,
        max_distance=args.max_distance
    )
    
    # 保存
    print("\n保存测试用例...")
    save_to_csv(test_cases, args.output_csv)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
