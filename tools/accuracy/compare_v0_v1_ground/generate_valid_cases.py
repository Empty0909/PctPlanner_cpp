#!/usr/bin/env python3
"""
生成高质量测试用例（验证路径可达性）

分批在子进程中运行规划验证，只保留能成功规划的起终点对。
使用子进程隔离避免内存泄漏导致 OOM。

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy/compare_v0_v1_ground
    python3 generate_valid_cases.py \
        --bin /path/to/map.bin \
        --num-cases 1000 \
        --output ../data/valid_cases.csv
"""

import os
import sys
import csv
import json
import struct
import argparse
import subprocess
import tempfile
import numpy as np
from datetime import datetime

# 配置
DEFAULT_BIN = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v0.bin"
DEFAULT_OUTPUT = "../data/valid_cases.csv"
TRAVERSABLE_THRESHOLD = 10.0
CPP_LIB = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"


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


def generate_candidate_cases(tomo: dict, num_candidates: int, 
                             min_distance: float, max_distance: float,
                             seed: int) -> list:
    """生成候选测试用例（不验证路径）"""
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
    
    candidates = []
    attempts = 0
    max_attempts = num_candidates * 20
    
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        
        idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
        r1, c1 = coords[idx1]
        r2, c2 = coords[idx2]
        
        start = grid_to_world(r1, c1)
        end = grid_to_world(r2, c2)
        
        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        if dist < min_distance or dist > max_distance:
            continue
        
        candidates.append({
            'case_id': len(candidates) + 1,
            'start_x': float(start[0]),
            'start_y': float(start[1]),
            'start_z': float(start[2]),
            'start_yaw': 0.0,
            'goal_x': float(end[0]),
            'goal_y': float(end[1]),
            'goal_z': float(end[2]),
            'goal_yaw': 0.0,
        })
    
    return candidates


def validate_batch_in_subprocess(candidates: list, bin_path: str, batch_size: int = 50) -> list:
    """在子进程中分批验证候选用例，返回验证成功的用例"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, 'validate_cases_worker.py')
    
    valid_cases = []
    total_batches = (len(candidates) + batch_size - 1) // batch_size
    
    for batch_idx, batch_start in enumerate(range(0, len(candidates), batch_size)):
        batch = candidates[batch_start:batch_start + batch_size]
        
        print(f"    批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个用例)...", end=" ", flush=True)
        
        # 写入临时 CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_in:
            tmp_in_path = tmp_in.name
            fieldnames = ['case_id', 'start_x', 'start_y', 'start_z', 'start_yaw',
                          'goal_x', 'goal_y', 'goal_z', 'goal_yaw']
            writer = csv.DictWriter(tmp_in, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)
        
        tmp_out_path = tmp_in_path + '.result.json'
        
        batch_valid = 0
        try:
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = f"{CPP_LIB}:{CPP_GTSAM}:" + env.get('LD_LIBRARY_PATH', '')
            
            cmd = [sys.executable, worker_script,
                   '--input', tmp_in_path,
                   '--output', tmp_out_path,
                   '--bin', bin_path]
            
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600  # 增加超时时间
            )
            
            if result.returncode == 0 and os.path.exists(tmp_out_path):
                with open(tmp_out_path, 'r') as f:
                    batch_result = json.load(f)
                
                for case in batch:
                    case_id = str(case['case_id'])
                    if case_id in batch_result and batch_result[case_id].get('success'):
                        case['path_length'] = batch_result[case_id]['path_length']
                        valid_cases.append(case)
                        batch_valid += 1
            
            print(f"{batch_valid} 有效", flush=True)
        
        except subprocess.TimeoutExpired:
            print("超时", flush=True)
        except Exception as e:
            print(f"错误: {e}", flush=True)
        finally:
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    
    return valid_cases


def main():
    parser = argparse.ArgumentParser(description='生成高质量测试用例（验证路径可达性）')
    parser.add_argument('--bin', default=DEFAULT_BIN, help='地图 bin 文件路径')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='输出 CSV 路径')
    parser.add_argument('--num-cases', type=int, default=1000, help='目标有效用例数')
    parser.add_argument('--min-distance', type=float, default=20.0, help='起终点最小距离（米）')
    parser.add_argument('--max-distance', type=float, default=300.0, help='起终点最大距离（米）')
    parser.add_argument('--batch-size', type=int, default=10, help='每批验证的用例数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print("生成高质量测试用例（验证路径可达性）")
    print("=" * 60)
    print(f"  地图: {args.bin}")
    print(f"  目标用例数: {args.num_cases}")
    print(f"  距离范围: [{args.min_distance}, {args.max_distance}] m")
    print(f"  批次大小: {args.batch_size}")
    print()
    
    # 加载地图元数据
    print("加载地图...")
    tomo = load_tomogram_bin(args.bin)
    print(f"  维度: {tomo['dim_x']} x {tomo['dim_y']} x {tomo['n_slice']}")
    
    all_valid_cases = []
    iteration = 0
    seed = args.seed
    
    while len(all_valid_cases) < args.num_cases:
        iteration += 1
        remaining = args.num_cases - len(all_valid_cases)
        
        # 估计需要的候选数（假设成功率约 20%）
        num_candidates = min(remaining * 6, 5000)
        
        print(f"\n迭代 {iteration}: 生成 {num_candidates} 个候选用例...")
        candidates = generate_candidate_cases(
            tomo, num_candidates, 
            args.min_distance, args.max_distance,
            seed + iteration * 10000
        )
        print(f"  生成了 {len(candidates)} 个候选")
        
        print(f"  验证中（每批 {args.batch_size} 个）...")
        valid_cases = validate_batch_in_subprocess(
            candidates, args.bin, args.batch_size
        )
        print(f"  验证成功: {len(valid_cases)} 个 ({len(valid_cases)/len(candidates)*100:.1f}%)")
        
        # 重新编号并添加
        for case in valid_cases:
            case['case_id'] = len(all_valid_cases) + 1
            all_valid_cases.append(case)
            
            if len(all_valid_cases) >= args.num_cases:
                break
        
        print(f"  累计有效用例: {len(all_valid_cases)} / {args.num_cases}")
        
        if len(candidates) < num_candidates * 0.5:
            print("警告: 候选生成效率低，可能需要调整距离参数")
    
    # 截取到目标数量
    all_valid_cases = all_valid_cases[:args.num_cases]
    
    # 统计信息
    print(f"\n最终统计:")
    if all_valid_cases:
        lengths = [c.get('path_length', 0) for c in all_valid_cases if c.get('path_length')]
        xs = [c['start_x'] for c in all_valid_cases] + [c['goal_x'] for c in all_valid_cases]
        ys = [c['start_y'] for c in all_valid_cases] + [c['goal_y'] for c in all_valid_cases]
        print(f"  有效用例数: {len(all_valid_cases)}")
        print(f"  坐标范围: X [{min(xs):.2f}, {max(xs):.2f}], Y [{min(ys):.2f}, {max(ys):.2f}]")
        if lengths:
            print(f"  路径长度: 平均 {np.mean(lengths):.2f}, 最短 {min(lengths):.2f}, 最长 {max(lengths):.2f}")
    
    # 保存 CSV
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['case_id', 'start_x', 'start_y', 'start_z', 'start_yaw',
                      'goal_x', 'goal_y', 'goal_z', 'goal_yaw', 'path_length']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_valid_cases)
    
    print(f"\n已保存: {args.output}")
    print("完成!")


if __name__ == '__main__':
    main()
