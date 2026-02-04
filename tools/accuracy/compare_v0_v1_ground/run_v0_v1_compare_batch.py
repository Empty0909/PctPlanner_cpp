#!/usr/bin/env python3
"""
v0 vs v1 地图路径长度对比测试 - 分批模式

使用现有测试用例（从 CSV 读取），分别在 v0 和 v1 地图上规划，
每批在独立子进程中运行以避免内存泄漏导致 OOM。

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy
    python3 run_v0_v1_compare_batch.py \\
        --input data/nyby_ground_test_cases.csv \\
        --v0-bin /path/to/v0.bin \\
        --v1-bin /path/to/v1.bin \\
        --output results/v0_v1_comparison.json \\
        --batch-size 20 \\
        --limit 100
"""

import os
import sys
import json
import csv
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# 默认配置
DEFAULT_BATCH_SIZE = 20
DEFAULT_V0_BIN = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v0.bin"
DEFAULT_V1_BIN = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/nyby_ground_cost_map_v1.bin"
DEFAULT_INPUT_CSV = "data/nyby_ground_test_cases.csv"

CPP_LIB = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"


def count_csv_rows(csv_path: str) -> int:
    """计算 CSV 行数（不含表头）"""
    with open(csv_path, 'r') as f:
        return sum(1 for _ in f) - 1


def run_batch_on_map(csv_path: str, output_path: str, skip: int, limit: int, bin_path: str) -> dict:
    """在指定地图上运行单个批次"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, 'plan_on_map_batch.py')
    
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{CPP_LIB}:{CPP_GTSAM}:" + env.get('LD_LIBRARY_PATH', '')
    
    # 创建临时 CSV 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        with open(csv_path, 'r') as src:
            reader = csv.reader(src)
            writer = csv.writer(tmp)
            
            header = next(reader)
            writer.writerow(header)
            
            for _ in range(skip):
                next(reader, None)
            
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                writer.writerow(row)
    
    try:
        cmd = [sys.executable, script, 
               '--input', tmp_path, 
               '--output', output_path,
               '--bin', bin_path]
        
        result = subprocess.run(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )
        
        if result.returncode != 0:
            # 调试：打印子进程错误
            # print(f"子进程失败: {result.stderr.decode()[:500]}", file=sys.stderr)
            return None
        
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def merge_map_results(results_list: list) -> dict:
    """合并多个批次的结果"""
    all_results = {}
    total_success = 0
    total_count = 0
    all_times = []
    all_lengths = []
    
    for batch_result in results_list:
        if batch_result is None:
            continue
        
        for case_id, result in batch_result.get('results', {}).items():
            all_results[case_id] = result
            total_count += 1
            
            if result.get('success'):
                total_success += 1
                if 'time_ms' in result:
                    all_times.append(result['time_ms'])
                if 'path_length' in result:
                    all_lengths.append(result['path_length'])
    
    import numpy as np
    stats = {
        'total': total_count,
        'success': total_success,
        'failed': total_count - total_success,
        'success_rate': total_success / total_count * 100 if total_count else 0,
    }
    
    if all_times:
        stats['avg_time_ms'] = float(np.mean(all_times))
    if all_lengths:
        stats['avg_length'] = float(np.mean(all_lengths))
    
    return {'summary': stats, 'results': all_results}


def compare_results(v0_results: dict, v1_results: dict) -> dict:
    """对比 v0 和 v1 的结果"""
    import numpy as np
    
    both_success_cases = []
    v0_only_cases = []
    v1_only_cases = []
    both_fail_cases = []
    
    v0_data = v0_results.get('results', {})
    v1_data = v1_results.get('results', {})
    
    all_case_ids = set(v0_data.keys()) | set(v1_data.keys())
    
    for case_id in all_case_ids:
        v0 = v0_data.get(case_id, {})
        v1 = v1_data.get(case_id, {})
        
        v0_success = v0.get('success', False)
        v1_success = v1.get('success', False)
        
        if v0_success and v1_success:
            both_success_cases.append({
                'id': case_id,
                'v0_length': v0['path_length'],
                'v1_length': v1['path_length'],
                'diff': v1['path_length'] - v0['path_length']
            })
        elif v0_success and not v1_success:
            v0_only_cases.append(case_id)
        elif not v0_success and v1_success:
            v1_only_cases.append(case_id)
        else:
            both_fail_cases.append(case_id)
    
    # 统计
    total = len(all_case_ids)
    stats = {
        'total_cases': total,
        'both_success': len(both_success_cases),
        'v0_only_success': len(v0_only_cases),
        'v1_only_success': len(v1_only_cases),
        'both_fail': len(both_fail_cases),
        'v0_success_rate': v0_results['summary']['success_rate'],
        'v1_success_rate': v1_results['summary']['success_rate'],
    }
    
    if both_success_cases:
        diffs = np.array([c['diff'] for c in both_success_cases])
        v0_lens = np.array([c['v0_length'] for c in both_success_cases])
        v1_lens = np.array([c['v1_length'] for c in both_success_cases])
        
        stats.update({
            'mean_v0_length': float(np.mean(v0_lens)),
            'mean_v1_length': float(np.mean(v1_lens)),
            'mean_diff': float(np.mean(diffs)),
            'std_diff': float(np.std(diffs)),
            'min_diff': float(np.min(diffs)),
            'max_diff': float(np.max(diffs)),
            'median_diff': float(np.median(diffs)),
            'mean_abs_diff': float(np.mean(np.abs(diffs))),
            'pct_v1_longer': float(np.sum(diffs > 0.01) / len(diffs) * 100),
            'pct_v1_shorter': float(np.sum(diffs < -0.01) / len(diffs) * 100),
            'pct_equal': float(np.sum(np.abs(diffs) <= 0.01) / len(diffs) * 100),
        })
    
    return {
        'summary': stats,
        'both_success_cases': both_success_cases[:100],  # 只保存前100个
        'v0_only_success': v0_only_cases[:50],
        'v1_only_success': v1_only_cases[:50],
    }


def generate_report(comparison: dict, v0_bin: str, v1_bin: str, input_csv: str, output_dir: str):
    """生成 Markdown 报告"""
    stats = comparison['summary']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    md_path = os.path.join(output_dir, f"v0_v1_comparison_{timestamp}.md")
    
    with open(md_path, 'w') as f:
        f.write("# v0 vs v1 地图路径长度对比测试\n\n")
        f.write(f"生成时间: {timestamp}\n\n")
        f.write(f"## 测试配置\n\n")
        f.write(f"- 测试用例: `{input_csv}`\n")
        f.write(f"- v0 地图: `{v0_bin}`\n")
        f.write(f"- v1 地图: `{v1_bin}`\n\n")
        
        f.write("## 总体统计\n\n")
        f.write(f"| 项目 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 总用例数 | {stats['total_cases']} |\n")
        f.write(f"| 两者都成功 | {stats['both_success']} |\n")
        f.write(f"| 仅 v0 成功 | {stats['v0_only_success']} |\n")
        f.write(f"| 仅 v1 成功 | {stats['v1_only_success']} |\n")
        f.write(f"| 两者都失败 | {stats['both_fail']} |\n")
        f.write(f"| v0 成功率 | {stats['v0_success_rate']:.2f}% |\n")
        f.write(f"| v1 成功率 | {stats['v1_success_rate']:.2f}% |\n")
        
        if 'mean_diff' in stats:
            f.write("\n## 路径长度对比（两者都成功的用例）\n\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 有效对比数 | {stats['both_success']} |\n")
            f.write(f"| v0 平均长度 | {stats['mean_v0_length']:.2f} |\n")
            f.write(f"| v1 平均长度 | {stats['mean_v1_length']:.2f} |\n")
            f.write(f"| 平均差异 (v1-v0) | {stats['mean_diff']:.2f} |\n")
            f.write(f"| 差异标准差 | {stats['std_diff']:.2f} |\n")
            f.write(f"| 最小差异 | {stats['min_diff']:.2f} |\n")
            f.write(f"| 最大差异 | {stats['max_diff']:.2f} |\n")
            f.write(f"| 中位数差异 | {stats['median_diff']:.2f} |\n")
            f.write(f"| 平均绝对差异 | {stats['mean_abs_diff']:.2f} |\n")
            f.write(f"| v1 更长 | {stats['pct_v1_longer']:.1f}% |\n")
            f.write(f"| v1 更短 | {stats['pct_v1_shorter']:.1f}% |\n")
            f.write(f"| 相等 | {stats['pct_equal']:.1f}% |\n")
        
        f.write("\n## 结论\n\n")
        if 'mean_diff' in stats:
            if abs(stats['mean_diff']) < 1.0:
                f.write("v0 和 v1 地图的平均路径长度差异很小，两者基本一致。\n")
            elif stats['mean_diff'] > 0:
                f.write(f"v1 地图的平均路径长度比 v0 长 {stats['mean_diff']:.2f} 单位。\n")
            else:
                f.write(f"v1 地图的平均路径长度比 v0 短 {-stats['mean_diff']:.2f} 单位。\n")
        
        if stats['v0_only_success'] > stats['total_cases'] * 0.1:
            f.write(f"\n**注意**: {stats['v0_only_success']} 个用例在 v0 成功但 v1 失败，说明 v1 部分区域可通行性发生变化。\n")
    
    return md_path


def main():
    parser = argparse.ArgumentParser(description='v0 vs v1 地图对比测试 - 分批模式')
    parser.add_argument('--input', default=DEFAULT_INPUT_CSV, help='输入测试用例 CSV')
    parser.add_argument('--v0-bin', default=DEFAULT_V0_BIN, help='v0 地图路径')
    parser.add_argument('--v1-bin', default=DEFAULT_V1_BIN, help='v1 地图路径')
    parser.add_argument('--output', default='results/v0_v1_comparison.json', help='输出 JSON')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='批次大小')
    parser.add_argument('--limit', type=int, default=100, help='测试用例数量限制')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算批次
    total_rows = min(count_csv_rows(args.input), args.limit)
    num_batches = (total_rows + args.batch_size - 1) // args.batch_size
    
    print("=" * 60, file=sys.stderr)
    print("v0 vs v1 地图对比测试 - 分批模式", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  输入 CSV: {args.input}", file=sys.stderr)
    print(f"  v0 地图: {args.v0_bin}", file=sys.stderr)
    print(f"  v1 地图: {args.v1_bin}", file=sys.stderr)
    print(f"  用例数: {total_rows}", file=sys.stderr)
    print(f"  批大小: {args.batch_size}", file=sys.stderr)
    print(f"  批次数: {num_batches}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # ========== 在 v0 上规划 ==========
    print("在 v0 地图上规划...", file=sys.stderr)
    v0_results_list = []
    
    for batch_idx in range(num_batches):
        skip = batch_idx * args.batch_size
        batch_limit = min(args.batch_size, total_rows - skip)
        
        print(f"  v0 批次 {batch_idx + 1}/{num_batches}...", file=sys.stderr, end='', flush=True)
        
        tmp_output = args.output + f'.v0.batch{batch_idx}'
        batch_result = run_batch_on_map(args.input, tmp_output, skip, batch_limit, args.v0_bin)
        
        if batch_result:
            v0_results_list.append(batch_result)
            s = batch_result['summary']['success']
            t = batch_result['summary']['total']
            print(f" 完成 ({s}/{t})", file=sys.stderr)
        else:
            print(" 失败", file=sys.stderr)
        
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
    
    v0_merged = merge_map_results(v0_results_list)
    print(f"  v0 完成: {v0_merged['summary']['success']}/{v0_merged['summary']['total']}", file=sys.stderr)
    
    # ========== 在 v1 上规划 ==========
    print("\n在 v1 地图上规划...", file=sys.stderr)
    v1_results_list = []
    
    for batch_idx in range(num_batches):
        skip = batch_idx * args.batch_size
        batch_limit = min(args.batch_size, total_rows - skip)
        
        print(f"  v1 批次 {batch_idx + 1}/{num_batches}...", file=sys.stderr, end='', flush=True)
        
        tmp_output = args.output + f'.v1.batch{batch_idx}'
        batch_result = run_batch_on_map(args.input, tmp_output, skip, batch_limit, args.v1_bin)
        
        if batch_result:
            v1_results_list.append(batch_result)
            s = batch_result['summary']['success']
            t = batch_result['summary']['total']
            print(f" 完成 ({s}/{t})", file=sys.stderr)
        else:
            print(" 失败", file=sys.stderr)
        
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
    
    v1_merged = merge_map_results(v1_results_list)
    print(f"  v1 完成: {v1_merged['summary']['success']}/{v1_merged['summary']['total']}", file=sys.stderr)
    
    # ========== 对比结果 ==========
    print("\n对比结果...", file=sys.stderr)
    comparison = compare_results(v0_merged, v1_merged)
    
    # 保存完整结果
    full_result = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'input_csv': args.input,
            'v0_bin': args.v0_bin,
            'v1_bin': args.v1_bin,
            'total_cases': total_rows,
            'batch_size': args.batch_size,
        },
        'comparison': comparison,
        'v0_summary': v0_merged['summary'],
        'v1_summary': v1_merged['summary'],
    }
    
    with open(args.output, 'w') as f:
        json.dump(full_result, f, indent=2)
    print(f"  JSON: {args.output}", file=sys.stderr)
    
    # 生成报告
    md_path = generate_report(comparison, args.v0_bin, args.v1_bin, args.input, output_dir)
    print(f"  报告: {md_path}", file=sys.stderr)
    
    # 打印统计
    stats = comparison['summary']
    print("\n" + "=" * 60, file=sys.stderr)
    print("统计结果", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"总用例: {stats['total_cases']}", file=sys.stderr)
    print(f"两者都成功: {stats['both_success']}", file=sys.stderr)
    print(f"仅 v0 成功: {stats['v0_only_success']}", file=sys.stderr)
    print(f"仅 v1 成功: {stats['v1_only_success']}", file=sys.stderr)
    print(f"两者都失败: {stats['both_fail']}", file=sys.stderr)
    
    if 'mean_diff' in stats:
        print(f"\n路径长度比较 (两者都成功):", file=sys.stderr)
        print(f"  v0 平均: {stats['mean_v0_length']:.2f}", file=sys.stderr)
        print(f"  v1 平均: {stats['mean_v1_length']:.2f}", file=sys.stderr)
        print(f"  平均差异: {stats['mean_diff']:.2f}", file=sys.stderr)
        print(f"  差异标准差: {stats['std_diff']:.2f}", file=sys.stderr)
        print(f"  v1 更长: {stats['pct_v1_longer']:.1f}%", file=sys.stderr)
        print(f"  v1 更短: {stats['pct_v1_shorter']:.1f}%", file=sys.stderr)
    
    print("\n完成!", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
