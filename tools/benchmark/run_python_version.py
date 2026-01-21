#!/usr/bin/env python3
"""
Python 版本规划测试

使用 Python 版本的规划库
支持 Python 版本的 tomogram pickle 文件 或 C++ 版本的 tomogram bin 文件
"""

import os
import sys
import csv
import json
import time
import pickle
import struct
import argparse
import traceback

import numpy as np

# ============================================================
# 关键：使用 Python 版本的库路径
# ============================================================
PYTHON_PLANNER_ROOT = "/home/lzy/PctPlanner/PctPlanner_py/planner"
PYTHON_LIB_PATH = os.path.join(PYTHON_PLANNER_ROOT, "lib")
PYTHON_SCRIPTS_PATH = os.path.join(PYTHON_PLANNER_ROOT, "scripts")

# 确保 Python 版本的库优先加载
sys.path.insert(0, PYTHON_LIB_PATH)
sys.path.insert(0, PYTHON_SCRIPTS_PATH)
sys.path.insert(0, PYTHON_PLANNER_ROOT)

# 设置动态库搜索路径
os.environ['LD_LIBRARY_PATH'] = PYTHON_LIB_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# 导入 Python 版本的库
from lib import a_star, ele_planner, traj_opt
from scripts.planner_wrapper import TomogramPlanner, load_tomogram_binary
from config import Config

# 默认路径 - Python 版本的 tomogram
DEFAULT_PICKLE_PATH = "/home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle"
DEFAULT_INPUT_CSV = "test_cases.csv"
DEFAULT_OUTPUT_JSON = "python_version_results.json"


def load_test_cases(csv_path: str) -> list:
    """加载测试用例"""
    test_cases = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'id': int(row['id']),
                'start_x': float(row['start_x']),
                'start_y': float(row['start_y']),
                'start_z': float(row['start_z']),
                'end_x': float(row['end_x']),
                'end_y': float(row['end_y']),
                'end_z': float(row['end_z']),
            })
    return test_cases


class PythonVersionPlanner:
    """
    Python 版本规划器
    
    使用：
    - Python 版本的 ele_planner 库
    - Python 版本的 tomogram pickle 文件 或 C++ 版本的 bin 文件
    """
    
    def __init__(self, tomo_path: str):
        self.cfg = Config()
        self.planner = TomogramPlanner(self.cfg)
        
        # 自动识别格式并加载
        if tomo_path.endswith('.bin'):
            print(f"  加载 C++ 二进制格式 tomogram: {tomo_path}")
            data_dict = load_tomogram_binary(tomo_path)
        else:
            print(f"  加载 Python pickle 格式 tomogram: {tomo_path}")
            with open(tomo_path, 'rb') as f:
                data_dict = pickle.load(f)
        
        self.planner._initialize_from_dict(data_dict)
        
        # 获取库版本信息
        lib_path = ele_planner.__file__
        print(f"  使用 ele_planner 库: {lib_path}", file=sys.stderr)
    
    def plan(self, start_x, start_y, start_z, end_x, end_y, end_z):
        """执行规划"""
        start_pos = np.array([start_x, start_y], dtype=np.float64)
        end_pos = np.array([end_x, end_y], dtype=np.float64)
        
        # 与原始 plan.py 一致: z + 0.5
        traj = self.planner.plan(start_pos, end_pos, start_z + 0.5, end_z + 0.5)
        return traj


import gc


def run_tests(test_cases: list, tomo_path: str, output_path: str, verbose: bool = False) -> dict:
    """执行测试 - 增量保存结果以减少内存使用"""
    print("初始化 Python 版本规划器...", file=sys.stderr)
    planner = PythonVersionPlanner(tomo_path)
    
    results = {}
    success_count = 0
    
    # 每 50 个用例保存一次，避免内存积累
    save_interval = 50
    
    for i, case in enumerate(test_cases):
        if (i + 1) % 100 == 0 or (i + 1) == len(test_cases) or verbose:
            print(f"  进度: {i + 1}/{len(test_cases)} (成功: {success_count})", file=sys.stderr)
        
        try:
            start_time = time.time()
            traj = planner.plan(
                case['start_x'], case['start_y'], case['start_z'],
                case['end_x'], case['end_y'], case['end_z']
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if traj is not None and len(traj) > 0:
                success_count += 1
                # 只保存轨迹的首尾点和统计信息，减少内存使用
                results[str(case['id'])] = {
                    'success': True,
                    'trajectory': traj.tolist(),
                    'time_ms': elapsed_ms,
                    'num_points': len(traj)
                }
            else:
                results[str(case['id'])] = {
                    'success': False,
                    'error': 'No valid path found',
                    'time_ms': elapsed_ms
                }
                
            # 主动释放轨迹内存
            del traj
            
        except Exception as e:
            results[str(case['id'])] = {
                'success': False,
                'error': str(e),
            }
        
        # 定期强制垃圾回收
        if (i + 1) % save_interval == 0:
            gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Python 版本规划测试')
    parser.add_argument('--pickle', '--tomo', type=str, default=DEFAULT_PICKLE_PATH,
                        dest='tomo_path',
                        help='Tomogram 文件路径 (.pickle 或 .bin)')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_CSV,
                        help='输入测试用例 CSV 文件')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_JSON,
                        help='输出结果 JSON 文件')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制测试用例数量')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    print("=" * 60)
    print("Python 版本规划测试")
    print("=" * 60)
    print(f"  库路径: {PYTHON_LIB_PATH}")
    print(f"  Tomogram: {args.tomo_path}")
    
    # 加载测试用例
    print(f"\n加载测试用例: {args.input}", file=sys.stderr)
    test_cases = load_test_cases(args.input)
    print(f"  共 {len(test_cases)} 个用例", file=sys.stderr)
    
    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"  限制为前 {args.limit} 个用例", file=sys.stderr)
    
    # 执行测试
    print(f"\n开始测试...", file=sys.stderr)
    start_time = time.time()
    results = run_tests(test_cases, args.tomo_path, args.output, args.verbose)
    total_time = time.time() - start_time
    
    # 统计
    success_count = sum(1 for r in results.values() if r.get('success', False))
    fail_count = len(results) - success_count
    success_rate = success_count / len(results) * 100 if results else 0
    
    success_times = [r.get('time_ms', 0) for r in results.values() if r.get('success')]
    avg_time = np.mean(success_times) if success_times else 0
    
    print(f"\n测试完成!", file=sys.stderr)
    print(f"  总耗时: {total_time:.1f} 秒", file=sys.stderr)
    print(f"  成功: {success_count}/{len(results)} ({success_rate:.2f}%)", file=sys.stderr)
    print(f"  失败: {fail_count}", file=sys.stderr)
    print(f"  平均规划时间: {avg_time:.2f} ms", file=sys.stderr)
    
    # 保存结果
    print(f"\n保存结果: {args.output}", file=sys.stderr)
    with open(args.output, 'w') as f:
        json.dump({
            'version': 'python',
            'lib_path': PYTHON_LIB_PATH,
            'tomogram_path': args.tomo_path,
            'summary': {
                'total': len(results),
                'success': success_count,
                'failed': fail_count,
                'success_rate': success_rate,
                'avg_time_ms': avg_time,
                'total_time_sec': total_time
            },
            'results': results
        }, f, indent=2)
    
    print("\n完成!", file=sys.stderr)
    return 0 if success_rate >= 60 else 1


if __name__ == '__main__':
    sys.exit(main())
