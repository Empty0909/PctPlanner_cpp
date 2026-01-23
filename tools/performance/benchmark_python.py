#!/usr/bin/env python3
"""
Python/CuPy Tomography 精确性能测试

使用 CUDA Event 进行精确 GPU 计时，与 C++ 版本对比。

使用方法：
    python3 benchmark_python.py --runs 10
    python3 benchmark_python.py --pcd /path/to/map.pcd --runs 5
"""

import os
import sys
import time
import argparse
import numpy as np
import gc

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PCTPLANNER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PY_ROOT = os.path.join(PCTPLANNER_ROOT, 'PctPlanner_py')
DEFAULT_PCD = os.path.join(PY_ROOT, 'rsc', 'pcd', 'map.pcd')


def run_benchmark(pcd_path: str, runs: int, warmup: int = 2):
    """执行精确性能测试"""
    
    print("=" * 70)
    print("Python/CuPy Tomography 精确性能测试")
    print("=" * 70)
    print(f"PCD 文件: {pcd_path}")
    print(f"预热次数: {warmup}")
    print(f"测试次数: {runs}")
    
    # 检查依赖
    try:
        import cupy as cp
        import open3d as o3d
    except ImportError as e:
        print(f"错误: 缺少依赖 - {e}")
        print("请安装: pip install cupy-cuda11x open3d")
        return None
    
    # 导入 Python tomogram 模块
    sys.path.insert(0, os.path.join(PY_ROOT, 'tomography', 'scripts'))
    sys.path.insert(0, os.path.join(PY_ROOT, 'tomography'))
    
    try:
        from tomogram import Tomogram
        from config.scene_map import SceneMap as Config
    except ImportError as e:
        print(f"错误: 导入 tomogram 模块失败 - {e}")
        return None
    
    cfg = Config()
    
    # ========== GPU 预热 ==========
    print(f"\nGPU 预热中 ({warmup} 次)...")
    _ = cp.zeros((2000, 2000), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    
    # 预加载一次获取地图参数
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    point_count = len(points)
    
    points_max = np.max(points, axis=0)
    points_min = np.min(points, axis=0)
    points_min[-1] = 0.0  # ground_h
    resolution = 0.15
    slice_dh = 0.5
    
    map_dim_x = int(np.ceil((points_max[0] - points_min[0]) / resolution)) + 4
    map_dim_y = int(np.ceil((points_max[1] - points_min[1]) / resolution)) + 4
    n_slice = int(np.ceil((points_max[2] - points_min[2]) / slice_dh))
    center = (points_max[:2] + points_min[:2]) / 2
    slice_h0 = points_min[-1] + slice_dh
    
    print(f"点云数量: {point_count:,}")
    print(f"地图维度: {map_dim_x}×{map_dim_y}×{n_slice}")
    print(f"总像素数: {map_dim_x * map_dim_y * n_slice:,}")
    
    # 预热运行
    for w in range(warmup):
        tomogram = Tomogram(cfg)
        tomogram.initMappingEnv(center, map_dim_x, map_dim_y, n_slice, slice_h0)
        tomogram.point2map(points)
        del tomogram
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    # ========== 正式测试 ==========
    print(f"\n开始正式测试 ({runs} 次)...")
    
    pcd_load_times = []
    cuda_times = []
    total_times = []
    
    for run_idx in range(runs):
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        # 1. PCD 加载计时 (CPU)
        t0 = time.perf_counter()
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        t1 = time.perf_counter()
        pcd_load_time = (t1 - t0) * 1000
        pcd_load_times.append(pcd_load_time)
        
        # 2. Tomography 初始化 + CUDA 计算 (GPU Event 精确计时)
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        t2 = time.perf_counter()
        tomogram = Tomogram(cfg)
        tomogram.initMappingEnv(center, map_dim_x, map_dim_y, n_slice, slice_h0)
        
        start_event.record()
        tomogram.point2map(points)
        end_event.record()
        end_event.synchronize()
        t3 = time.perf_counter()
        
        cuda_time = cp.cuda.get_elapsed_time(start_event, end_event)
        cuda_times.append(cuda_time)
        
        total_time = pcd_load_time + cuda_time
        total_times.append(total_time)
        
        print(f"  Run {run_idx + 1}/{runs}: "
              f"PCD={pcd_load_time:.1f}ms, CUDA={cuda_time:.1f}ms, "
              f"Total={total_time:.1f}ms")
        
        del tomogram
    
    # ========== 统计结果 ==========
    print("\n" + "=" * 70)
    print("Python/CuPy 统计结果")
    print("=" * 70)
    
    def stats(data):
        return np.mean(data), np.std(data), np.min(data), np.max(data)
    
    pcd_avg, pcd_std, pcd_min, pcd_max = stats(pcd_load_times)
    cuda_avg, cuda_std, cuda_min, cuda_max = stats(cuda_times)
    total_avg, total_std, total_min, total_max = stats(total_times)
    
    print(f"\n{'阶段':<20} {'平均(ms)':<12} {'标准差':<12} {'最小':<12} {'最大':<12}")
    print("-" * 68)
    print(f"{'PCD 加载':<20} {pcd_avg:<12.2f} {pcd_std:<12.2f} {pcd_min:<12.2f} {pcd_max:<12.2f}")
    print(f"{'CUDA 计算':<20} {cuda_avg:<12.2f} {cuda_std:<12.2f} {cuda_min:<12.2f} {cuda_max:<12.2f}")
    print(f"{'端到端总时间':<20} {total_avg:<12.2f} {total_std:<12.2f} {total_min:<12.2f} {total_max:<12.2f}")
    
    print("\n" + "=" * 70)
    print("性能总结")
    print("=" * 70)
    print(f"PCD 加载: {pcd_avg:.2f} ms")
    print(f"CUDA 计算: {cuda_avg:.2f} ms")
    print(f"端到端: {total_avg:.2f} ms")
    
    return {
        'pcd_avg': pcd_avg,
        'cuda_avg': cuda_avg,
        'total_avg': total_avg,
        'point_count': point_count,
        'map_dims': (map_dim_x, map_dim_y, n_slice),
    }


def main():
    parser = argparse.ArgumentParser(description='Python/CuPy Tomography 精确性能测试')
    parser.add_argument('--pcd', type=str, default=DEFAULT_PCD, help='PCD 文件路径')
    parser.add_argument('--runs', type=int, default=10, help='测试次数')
    parser.add_argument('--warmup', type=int, default=2, help='预热次数')
    args = parser.parse_args()
    
    if not os.path.exists(args.pcd):
        print(f"错误: PCD 文件不存在: {args.pcd}")
        return 1
    
    result = run_benchmark(args.pcd, args.runs, args.warmup)
    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())
