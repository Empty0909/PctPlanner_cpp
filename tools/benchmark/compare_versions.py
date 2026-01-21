#!/usr/bin/env python3
"""
PctPlanner 版本对比分析

对比 Python 版本和 C++ 版本的规划结果:
- 成功率
- 轨迹差异
- 性能差异
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff

# 默认路径
DEFAULT_PYTHON_RESULTS = "python_version_results.json"
DEFAULT_CPP_RESULTS = "cpp_version_results.json"
DEFAULT_REPORT = "version_comparison_report.md"


def load_results(json_path: str) -> Dict:
    """加载测试结果"""
    with open(json_path, 'r') as f:
        return json.load(f)


def interpolate_trajectory(traj: np.ndarray, num_points: int = 100) -> np.ndarray:
    """将轨迹插值到统一点数"""
    if len(traj) < 2:
        return traj
    
    # 计算累积弧长
    diffs = np.diff(traj, axis=0)
    arc_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(arc_lengths)])
    
    if cumulative[-1] == 0:
        return traj
    
    # 归一化
    cumulative_normalized = cumulative / cumulative[-1]
    
    # 插值
    target_s = np.linspace(0, 1, num_points)
    interpolated = np.zeros((num_points, 3))
    
    for dim in range(3):
        f = interp1d(cumulative_normalized, traj[:, dim], kind='linear', fill_value='extrapolate')
        interpolated[:, dim] = f(target_s)
    
    return interpolated


def compute_trajectory_error(traj1: np.ndarray, traj2: np.ndarray) -> Dict:
    """计算两条轨迹之间的误差"""
    # 插值到相同点数
    n_points = 100
    traj1_interp = interpolate_trajectory(traj1, n_points)
    traj2_interp = interpolate_trajectory(traj2, n_points)
    
    # 点对点误差
    point_errors = np.linalg.norm(traj1_interp - traj2_interp, axis=1)
    
    # Hausdorff 距离
    hausdorff_12 = directed_hausdorff(traj1, traj2)[0]
    hausdorff_21 = directed_hausdorff(traj2, traj1)[0]
    hausdorff = max(hausdorff_12, hausdorff_21)
    
    # 各轴误差
    x_errors = np.abs(traj1_interp[:, 0] - traj2_interp[:, 0])
    y_errors = np.abs(traj1_interp[:, 1] - traj2_interp[:, 1])
    z_errors = np.abs(traj1_interp[:, 2] - traj2_interp[:, 2])
    
    return {
        'mean_error': float(np.mean(point_errors)),
        'max_error': float(np.max(point_errors)),
        'std_error': float(np.std(point_errors)),
        'hausdorff': float(hausdorff),
        'x_mean_error': float(np.mean(x_errors)),
        'y_mean_error': float(np.mean(y_errors)),
        'z_mean_error': float(np.mean(z_errors)),
        'x_max_error': float(np.max(x_errors)),
        'y_max_error': float(np.max(y_errors)),
        'z_max_error': float(np.max(z_errors)),
    }


def compare_results(python_results: Dict, cpp_results: Dict) -> Dict:
    """对比两个版本的结果"""
    py_results = python_results.get('results', {})
    cpp_results_data = cpp_results.get('results', {})
    
    all_ids = set(py_results.keys()) | set(cpp_results_data.keys())
    
    comparison = {
        'both_success': [],
        'both_fail': [],
        'python_only': [],
        'cpp_only': [],
        'trajectory_errors': [],
    }
    
    for case_id in sorted(all_ids, key=lambda x: int(x)):
        py_r = py_results.get(case_id, {})
        cpp_r = cpp_results_data.get(case_id, {})
        
        py_success = py_r.get('success', False)
        cpp_success = cpp_r.get('success', False)
        
        if py_success and cpp_success:
            comparison['both_success'].append(case_id)
            
            # 计算轨迹误差
            py_traj = np.array(py_r['trajectory'])
            cpp_traj = np.array(cpp_r['trajectory'])
            
            error = compute_trajectory_error(py_traj, cpp_traj)
            error['case_id'] = case_id
            error['py_points'] = len(py_traj)
            error['cpp_points'] = len(cpp_traj)
            error['py_time_ms'] = py_r.get('time_ms', 0)
            error['cpp_time_ms'] = cpp_r.get('time_ms', 0)
            comparison['trajectory_errors'].append(error)
            
        elif not py_success and not cpp_success:
            comparison['both_fail'].append(case_id)
        elif py_success and not cpp_success:
            comparison['python_only'].append(case_id)
        else:
            comparison['cpp_only'].append(case_id)
    
    return comparison


def generate_report(python_results: Dict, cpp_results: Dict, 
                   comparison: Dict, output_path: str):
    """生成对比报告"""
    py_summary = python_results.get('summary', {})
    cpp_summary = cpp_results.get('summary', {})
    
    total = py_summary.get('total', 0)
    
    both_success = len(comparison['both_success'])
    both_fail = len(comparison['both_fail'])
    python_only = len(comparison['python_only'])
    cpp_only = len(comparison['cpp_only'])
    
    # 计算轨迹误差统计
    errors = comparison['trajectory_errors']
    if errors:
        mean_errors = [e['mean_error'] for e in errors]
        max_errors = [e['max_error'] for e in errors]
        hausdorff_distances = [e['hausdorff'] for e in errors]
        
        avg_mean_error = np.mean(mean_errors)
        avg_max_error = np.mean(max_errors)
        overall_max_error = max(max_errors)
        avg_hausdorff = np.mean(hausdorff_distances)
        
        # 各轴误差
        x_mean_errors = [e['x_mean_error'] for e in errors]
        y_mean_errors = [e['y_mean_error'] for e in errors]
        z_mean_errors = [e['z_mean_error'] for e in errors]
    else:
        avg_mean_error = avg_max_error = overall_max_error = avg_hausdorff = 0
        x_mean_errors = y_mean_errors = z_mean_errors = []
    
    # 性能对比
    py_times = [e['py_time_ms'] for e in errors]
    cpp_times = [e['cpp_time_ms'] for e in errors]
    
    report = f"""# PctPlanner 版本对比报告

## 测试配置

| 项目 | Python 版本 | C++ 版本 |
|------|------------|---------|
| 库路径 | `{python_results.get('lib_path', 'N/A')}` | `{cpp_results.get('lib_path', 'N/A')}` |
| Tomogram | `{python_results.get('tomogram_path', 'N/A')}` | `{cpp_results.get('tomogram_path', 'N/A')}` |

## 测试摘要

| 指标 | Python 版本 | C++ 版本 |
|------|------------|---------|
| 总用例数 | {py_summary.get('total', 0)} | {cpp_summary.get('total', 0)} |
| 成功数 | {py_summary.get('success', 0)} | {cpp_summary.get('success', 0)} |
| 失败数 | {py_summary.get('failed', 0)} | {cpp_summary.get('failed', 0)} |
| 成功率 | {py_summary.get('success_rate', 0):.2f}% | {cpp_summary.get('success_rate', 0):.2f}% |
| 平均规划时间 | {py_summary.get('avg_time_ms', 0):.2f} ms | {cpp_summary.get('avg_time_ms', 0):.2f} ms |

## 对比结果

| 类别 | 数量 | 占比 |
|------|------|------|
| 双方成功 | {both_success} | {both_success/total*100 if total > 0 else 0:.1f}% |
| 双方失败 | {both_fail} | {both_fail/total*100 if total > 0 else 0:.1f}% |
| 仅 Python 成功 | {python_only} | {python_only/total*100 if total > 0 else 0:.1f}% |
| 仅 C++ 成功 | {cpp_only} | {cpp_only/total*100 if total > 0 else 0:.1f}% |

## 轨迹误差分析

基于 {both_success} 个双方都成功的用例:

| 误差指标 | 平均值 | 最大值 |
|---------|--------|--------|
| 点对点误差 | {avg_mean_error:.6f} m | {overall_max_error:.6f} m |
| Hausdorff 距离 | {avg_hausdorff:.6f} m | - |
| X 轴误差 | {np.mean(x_mean_errors) if x_mean_errors else 0:.6f} m | {max(x_mean_errors) if x_mean_errors else 0:.6f} m |
| Y 轴误差 | {np.mean(y_mean_errors) if y_mean_errors else 0:.6f} m | {max(y_mean_errors) if y_mean_errors else 0:.6f} m |
| Z 轴误差 | {np.mean(z_mean_errors) if z_mean_errors else 0:.6f} m | {max(z_mean_errors) if z_mean_errors else 0:.6f} m |

## 性能对比

基于 {both_success} 个双方都成功的用例:

| 版本 | 平均时间 | 最小时间 | 最大时间 |
|------|---------|---------|---------|
| Python | {np.mean(py_times) if py_times else 0:.2f} ms | {min(py_times) if py_times else 0:.2f} ms | {max(py_times) if py_times else 0:.2f} ms |
| C++ | {np.mean(cpp_times) if cpp_times else 0:.2f} ms | {min(cpp_times) if cpp_times else 0:.2f} ms | {max(cpp_times) if cpp_times else 0:.2f} ms |

## 验收标准检查

| 标准 | 目标 | 实际 | 结果 |
|------|------|------|------|
| Python 成功率 | ≥ 60% | {py_summary.get('success_rate', 0):.2f}% | {'✅ 通过' if py_summary.get('success_rate', 0) >= 60 else '❌ 未通过'} |
| 轨迹误差 | ≤ 0.01 m | {avg_mean_error:.6f} m | {'✅ 通过' if avg_mean_error <= 0.01 else '❌ 未通过'} |

"""
    
    # 如果有差异较大的用例，列出详情
    if errors:
        large_error_cases = [e for e in errors if e['max_error'] > 0.01]
        if large_error_cases:
            report += f"""
## 误差较大的用例 (最大误差 > 0.01m)

| Case ID | 平均误差 | 最大误差 | X误差 | Y误差 | Z误差 |
|---------|---------|---------|-------|-------|-------|
"""
            for e in sorted(large_error_cases, key=lambda x: -x['max_error'])[:20]:
                report += f"| {e['case_id']} | {e['mean_error']:.4f} | {e['max_error']:.4f} | {e['x_max_error']:.4f} | {e['y_max_error']:.4f} | {e['z_max_error']:.4f} |\n"
    
    # 列出仅单方成功的用例
    if comparison['python_only']:
        report += f"""
## 仅 Python 成功的用例

共 {len(comparison['python_only'])} 个: {', '.join(comparison['python_only'][:20])}{'...' if len(comparison['python_only']) > 20 else ''}
"""
    
    if comparison['cpp_only']:
        report += f"""
## 仅 C++ 成功的用例

共 {len(comparison['cpp_only'])} 个: {', '.join(comparison['cpp_only'][:20])}{'...' if len(comparison['cpp_only']) > 20 else ''}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return {
        'py_success_rate': py_summary.get('success_rate', 0),
        'cpp_success_rate': cpp_summary.get('success_rate', 0),
        'both_success': both_success,
        'avg_mean_error': avg_mean_error,
        'overall_max_error': overall_max_error,
    }


def main():
    parser = argparse.ArgumentParser(description='PctPlanner 版本对比分析')
    parser.add_argument('--python', type=str, default=DEFAULT_PYTHON_RESULTS,
                        help='Python 版本结果 JSON')
    parser.add_argument('--cpp', type=str, default=DEFAULT_CPP_RESULTS,
                        help='C++ 版本结果 JSON')
    parser.add_argument('--output', type=str, default=DEFAULT_REPORT,
                        help='输出报告文件')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    if not os.path.isabs(args.python):
        args.python = os.path.join(script_dir, args.python)
    if not os.path.isabs(args.cpp):
        args.cpp = os.path.join(script_dir, args.cpp)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    print("=" * 60)
    print("PctPlanner 版本对比分析")
    print("=" * 60)
    
    # 加载结果
    print(f"\n加载 Python 结果: {args.python}")
    python_results = load_results(args.python)
    print(f"加载 C++ 结果: {args.cpp}")
    cpp_results = load_results(args.cpp)
    
    # 对比
    print(f"\n执行对比分析...")
    comparison = compare_results(python_results, cpp_results)
    
    # 生成报告
    print(f"\n生成报告...")
    summary = generate_report(python_results, cpp_results, comparison, args.output)
    
    print(f"\n=== 对比摘要 ===")
    print(f"Python 成功率: {summary['py_success_rate']:.2f}%")
    print(f"C++ 成功率: {summary['cpp_success_rate']:.2f}%")
    print(f"双方成功: {summary['both_success']}")
    print(f"平均轨迹误差: {summary['avg_mean_error']:.6f} m")
    print(f"最大轨迹误差: {summary['overall_max_error']:.6f} m")
    
    print(f"\n报告已保存: {args.output}")
    print("\n完成!")
    
    # 返回码：轨迹误差是否满足要求
    if summary['avg_mean_error'] <= 0.01 and summary['py_success_rate'] >= 60:
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
