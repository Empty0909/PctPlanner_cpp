#!/usr/bin/env python3
"""
批量运行测试 - 分批处理避免内存泄漏

这个脚本将测试用例分成小批次，每批在单独的进程中运行。
这样可以避免 C++ 库的内存泄漏导致 OOM。
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import csv
from pathlib import Path

# 默认每批大小
DEFAULT_BATCH_SIZE = 50

# 路径配置
PYTHON_LIB = "/home/lzy/PctPlanner/PctPlanner_py/planner/lib"
PYTHON_GTSAM = "/home/lzy/PctPlanner/PctPlanner_py/planner/lib/3rdparty/gtsam-4.1.1/install/lib"
CPP_LIB = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"


def count_csv_rows(csv_path: str) -> int:
    """计算 CSV 行数（不含表头）"""
    with open(csv_path, 'r') as f:
        return sum(1 for _ in f) - 1


def run_batch(version: str, csv_path: str, output_path: str, 
              skip: int, limit: int, pickle_or_bin: str, no_optimize: bool = False) -> dict:
    """运行单个批次"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if version == 'python':
        script = os.path.join(script_dir, 'run_python_version.py')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f"{PYTHON_LIB}:{PYTHON_GTSAM}:" + env.get('LD_LIBRARY_PATH', '')
        args = ['--pickle', pickle_or_bin]
    else:
        script = os.path.join(script_dir, 'run_cpp_version.py')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f"{CPP_LIB}:{CPP_GTSAM}:" + env.get('LD_LIBRARY_PATH', '')
        args = ['--bin', pickle_or_bin]
    
    # 添加 no_optimize 参数
    if no_optimize:
        args.append('--no-optimize')
    
    # 创建临时 CSV 文件，只包含需要处理的行
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        with open(csv_path, 'r') as src:
            reader = csv.reader(src)
            writer = csv.writer(tmp)
            
            # 写入表头
            header = next(reader)
            writer.writerow(header)
            
            # 跳过前 skip 行
            for _ in range(skip):
                next(reader, None)
            
            # 写入 limit 行
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                writer.writerow(row)
    
    try:
        # 运行子进程
        cmd = [sys.executable, script, 
               '--input', tmp_path, 
               '--output', output_path,
               *args]
        
        result = subprocess.run(
            cmd, 
            env=env, 
            stdout=subprocess.DEVNULL,  # 抑制 C++ 库的调试输出
            stderr=subprocess.PIPE,
            timeout=1200  # 20 分钟超时
        )
        
        # 检查子进程退出状态
        if result.returncode != 0:
            stderr_output = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
            # 只显示最后几行错误信息
            last_lines = stderr_output.strip().split('\n')[-3:] if stderr_output else []
            if last_lines:
                print(f" 子进程退出码 {result.returncode}", file=sys.stderr, end='')
        
        # 读取结果
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  批次超时！", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  批次执行失败: {e}", file=sys.stderr)
        return None
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def merge_results(results_list: list, version: str, lib_path: str, tomo_path: str) -> dict:
    """合并多个批次的结果"""
    merged = {
        'version': version,
        'lib_path': lib_path,
        'tomogram_path': tomo_path,
        'summary': {
            'total': 0,
            'success': 0,
            'failed': 0,
            'success_rate': 0,
            'avg_time_ms': 0,
            'total_time_sec': 0
        },
        'results': {}
    }
    
    all_times = []
    
    for batch_result in results_list:
        if batch_result is None:
            continue
            
        for case_id, result in batch_result.get('results', {}).items():
            merged['results'][case_id] = result
            merged['summary']['total'] += 1
            
            if result.get('success'):
                merged['summary']['success'] += 1
                if 'time_ms' in result:
                    all_times.append(result['time_ms'])
            else:
                merged['summary']['failed'] += 1
        
        merged['summary']['total_time_sec'] += batch_result.get('summary', {}).get('total_time_sec', 0)
    
    if merged['summary']['total'] > 0:
        merged['summary']['success_rate'] = merged['summary']['success'] / merged['summary']['total'] * 100
    if all_times:
        merged['summary']['avg_time_ms'] = sum(all_times) / len(all_times)
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='批量运行测试')
    parser.add_argument('--version', type=str, choices=['python', 'cpp'], required=True,
                        help='测试版本')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 CSV 文件')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 JSON 文件')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'每批大小 (默认 {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制总测试用例数量')
    parser.add_argument('--tomo', type=str, default=None,
                        help='指定代价地图路径 (.bin 或 .pickle)，覆盖默认路径')
    parser.add_argument('--no-optimize', action='store_true',
                        help='关闭轨迹优化器，仅返回 A* 路径')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    # 设置版本相关参数
    if args.version == 'python':
        default_tomo = "/home/lzy/PctPlanner/PctPlanner_py/rsc/tomogram/scene_map.pickle"
        lib_path = PYTHON_LIB
    else:
        default_tomo = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/scene_map.bin"
        lib_path = CPP_LIB
    
    # 使用指定的地图路径，或默认路径
    pickle_or_bin = args.tomo if args.tomo else default_tomo
    
    # 计算批次
    total_rows = count_csv_rows(args.input)
    if args.limit:
        total_rows = min(total_rows, args.limit)
    
    num_batches = (total_rows + args.batch_size - 1) // args.batch_size
    
    print(f"批量模式: {args.version} 版本", file=sys.stderr)
    print(f"  代价地图: {pickle_or_bin}", file=sys.stderr)
    if args.no_optimize:
        print(f"  [注意] 已关闭轨迹优化器，仅返回 A* 路径", file=sys.stderr)
    print(f"  总用例: {total_rows}", file=sys.stderr)
    print(f"  批大小: {args.batch_size}", file=sys.stderr)
    print(f"  批次数: {num_batches}", file=sys.stderr)
    
    results_list = []
    
    for batch_idx in range(num_batches):
        skip = batch_idx * args.batch_size
        batch_limit = min(args.batch_size, total_rows - skip)
        
        print(f"  处理批次 {batch_idx + 1}/{num_batches} (用例 {skip + 1}-{skip + batch_limit})...", 
              file=sys.stderr, end='')
        
        # 临时输出文件
        tmp_output = args.output + f'.batch{batch_idx}'
        
        batch_result = run_batch(
            args.version, args.input, tmp_output,
            skip, batch_limit, pickle_or_bin, args.no_optimize
        )
        
        if batch_result:
            results_list.append(batch_result)
            success = batch_result.get('summary', {}).get('success', 0)
            total = batch_result.get('summary', {}).get('total', 0)
            print(f" 完成 ({success}/{total} 成功)", file=sys.stderr)
        else:
            print(f" 失败", file=sys.stderr)
        
        # 清理临时文件
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
    
    # 合并结果
    merged = merge_results(results_list, args.version, lib_path, pickle_or_bin)
    
    # 保存最终结果
    with open(args.output, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n结果已保存: {args.output}", file=sys.stderr)
    print(f"  成功率: {merged['summary']['success_rate']:.2f}%", file=sys.stderr)
    print(f"  平均时间: {merged['summary']['avg_time_ms']:.2f} ms", file=sys.stderr)
    
    # 只要测试执行成功就返回 0，成功率检查在对比分析阶段进行
    return 0


if __name__ == '__main__':
    sys.exit(main())
