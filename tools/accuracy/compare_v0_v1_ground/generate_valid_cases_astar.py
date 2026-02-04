#!/usr/bin/env python3
"""
快速生成高质量测试用例（仅使用 A* 验证路径可达性）

只使用 A* 搜索来验证路径是否存在，不做轨迹优化。
这样速度更快，内存占用更小。

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy/compare_v0_v1_ground
    export LD_LIBRARY_PATH="/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib:/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH"
    python3 generate_valid_cases_astar.py \
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
sys.path.insert(0, CPP_LIB)


def verify_cpp_library():
    """确认使用的是 C++ 版本的库"""
    global a_star
    import a_star as a_star_module
    a_star = a_star_module
    
    module_path = getattr(a_star, '__file__', '')
    if 'PctPlanner_Cpp' not in module_path:
        raise ImportError(f"警告: a_star 不是 C++ 版本: {module_path}")
    print(f"  ✓ 使用 C++ 库 a_star: {module_path}")


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


def create_astar_planner(tomo: dict):
    """创建 A* 规划器"""
    n_slice = tomo['n_slice']
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    data = tomo['data']
    
    # 获取 cost 和 elevation 数据
    trav = data[0]  # traversability cost
    elev_g = np.nan_to_num(data[3], nan=-100)  # ground elevation
    
    # 创建 A* 实例
    astar = a_star.Astar()
    
    # 初始化 - 需要传入 cost 和 elevation 层
    # init(max_x, max_y, max_layers, cost_layer, ele_layer)
    astar.init(
        dim_y, dim_x, n_slice,
        trav.reshape(-1, trav.shape[-1]).astype(np.double),
        elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double)
    )
    
    return astar


def world_to_grid(pos, tomo: dict) -> np.ndarray:
    """世界坐标转栅格索引"""
    dim_x = tomo['dim_x']
    dim_y = tomo['dim_y']
    resolution = tomo['resolution']
    center = tomo['center']
    slice_heights = tomo['slice_heights']
    
    wx, wy, wz = pos[:3]
    
    # z 转换为 slice 索引
    slice_idx = np.searchsorted(slice_heights, wz + 0.5, side='right') - 1
    slice_idx = max(0, min(slice_idx, tomo['n_slice'] - 1))
    
    # xy 转换为栅格索引
    gx = int(round((wx - center[0]) / resolution + dim_x // 2))
    gy = int(round((wy - center[1]) / resolution + dim_y // 2))
    
    # 注意：A* 期望 (y, x) 顺序
    return np.array([slice_idx, gy, gx], dtype=np.int32)


def validate_path_exists(astar, tomo: dict, start: tuple, goal: tuple) -> tuple:
    """
    使用 A* 验证路径是否存在
    返回 (success, estimated_path_length)
    """
    start_idx = world_to_grid(start, tomo)
    goal_idx = world_to_grid(goal, tomo)
    
    try:
        # search(start_idx, goal_idx)
        # 返回值: 路径点列表或空
        result = astar.search(start_idx, goal_idx)
        
        if result is None or len(result) == 0:
            return False, None
        
        path = astar.get_result_matrix()
        if path is None or len(path) < 2:
            return False, None
        
        # 计算路径长度（栅格距离 * 分辨率）
        resolution = tomo['resolution']
        path_length = 0.0
        for i in range(1, len(path)):
            dx = (path[i][2] - path[i-1][2]) * resolution  # x
            dy = (path[i][1] - path[i-1][1]) * resolution  # y
            dz = (path[i][0] - path[i-1][0])  # slice - 不计入距离
            path_length += np.sqrt(dx*dx + dy*dy)
        
        return True, path_length
        
    except Exception as e:
        return False, None


def generate_and_validate_cases(tomo: dict, astar, num_cases: int,
                                 min_distance: float, max_distance: float,
                                 seed: int, gc_interval: int = 100) -> list:
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
    max_attempts = num_cases * 100
    
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
        start_pose = (*start, 0.0)
        goal_pose = (*goal, 0.0)
        
        success, path_length = validate_path_exists(astar, tomo, start_pose, goal_pose)
        
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
                'estimated_path_length': path_length,
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
                  'goal_x', 'goal_y', 'goal_z', 'goal_yaw', 'estimated_path_length']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    
    print(f"\n已保存 {len(cases)} 个用例到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='快速生成高质量测试用例（A* 验证）')
    parser.add_argument('--bin', default=DEFAULT_BIN, help='地图 bin 文件路径')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='输出 CSV 路径')
    parser.add_argument('--num-cases', type=int, default=1000, help='目标有效用例数')
    parser.add_argument('--min-distance', type=float, default=20.0, help='起终点最小距离（米）')
    parser.add_argument('--max-distance', type=float, default=300.0, help='起终点最大距离（米）')
    parser.add_argument('--gc-interval', type=int, default=100, help='垃圾回收间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("快速生成高质量测试用例（A* 验证版）")
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
    
    # 创建 A* 规划器
    print("\n创建 A* 规划器...")
    astar = create_astar_planner(tomo)
    print("  ✓ 规划器创建成功")
    
    # 生成并验证用例
    print(f"\n开始生成 {args.num_cases} 个有效用例...")
    start_time = datetime.now()
    
    valid_cases = generate_and_validate_cases(
        tomo, astar,
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
