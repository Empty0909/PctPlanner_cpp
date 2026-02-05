#!/usr/bin/env python3
"""
根据用户手动标记的对应点计算 v0 -> v1 坐标变换

使用方法:
    cd /home/lzy/PctPlanner/PctPlanner_Cpp/tools/accuracy
    python3 compare_v0_v1_ground/compute_transform_from_points.py

需要至少 2 个对应点来确定旋转+平移变换
"""

import numpy as np
from scipy.optimize import least_squares
import json
import os


def compute_rigid_transform(v0_points, v1_points):
    """
    计算从 v0 到 v1 的刚性变换 (旋转 + 平移)
    使用最小二乘法找到最优的 theta, tx, ty
    
    变换公式: v1 = R(theta) @ v0 + T
    """
    v0_points = np.array(v0_points)
    v1_points = np.array(v1_points)
    
    n = len(v0_points)
    if n < 2:
        raise ValueError("需要至少 2 个对应点")
    
    # 方法1: 使用 SVD 求解最优刚性变换 (Procrustes analysis)
    # 先计算质心
    v0_centroid = np.mean(v0_points, axis=0)
    v1_centroid = np.mean(v1_points, axis=0)
    
    # 去中心化
    v0_centered = v0_points - v0_centroid
    v1_centered = v1_points - v1_centroid
    
    # 计算 H 矩阵
    H = v0_centered.T @ v1_centered
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 确保是正确的旋转矩阵 (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移
    t = v1_centroid - R @ v0_centroid
    
    # 提取旋转角度
    theta = np.arctan2(R[1, 0], R[0, 0])
    
    # 计算残差
    v1_pred = (R @ v0_points.T).T + t
    residuals = np.linalg.norm(v1_pred - v1_points, axis=1)
    rmse = np.sqrt(np.mean(residuals**2))
    
    return {
        'theta_rad': float(theta),
        'theta_deg': float(np.degrees(theta)),
        'tx': float(t[0]),
        'ty': float(t[1]),
        'R': R.tolist(),
        'rmse': float(rmse),
        'residuals': residuals.tolist()
    }


def apply_transform(point, theta, tx, ty):
    """应用变换"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y = point
    x1 = cos_t * x - sin_t * y + tx
    y1 = sin_t * x + cos_t * y + ty
    return x1, y1


def main():
    print("=" * 70)
    print("根据对应点计算 v0 -> v1 坐标变换")
    print("=" * 70)
    
    # 已知对应点 (用户在 rviz2 中手动标记)
    # 格式: [(v0_x, v0_y), (v1_x, v1_y)]
    
    # 用户提供的第一个点
    point1_v0 = (-47.3, -1.94)
    point1_v1 = (-56.5, -16.8)
    
    print("\n请在 rviz2 中标记更多对应点（建议在地图的另一角）")
    print("已有对应点:")
    print(f"  点1: v0={point1_v0} -> v1={point1_v1}")
    
    # 尝试读取用户输入的额外对应点
    print("\n请输入第二个对应点（格式: v0_x v0_y v1_x v1_y，回车结束）:")
    print("例如: 50.5 -90.2 40.3 -105.1")
    
    additional_points = []
    while True:
        try:
            line = input("> ").strip()
            if not line:
                break
            parts = [float(x) for x in line.split()]
            if len(parts) == 4:
                additional_points.append({
                    'v0': (parts[0], parts[1]),
                    'v1': (parts[2], parts[3])
                })
                print(f"  已添加: v0=({parts[0]}, {parts[1]}) -> v1=({parts[2]}, {parts[3]})")
            else:
                print("  格式错误，请输入 4 个数字")
        except ValueError:
            print("  输入无效，请输入数字")
        except EOFError:
            break
    
    # 组合所有对应点
    v0_points = [point1_v0]
    v1_points = [point1_v1]
    
    for p in additional_points:
        v0_points.append(p['v0'])
        v1_points.append(p['v1'])
    
    print(f"\n共 {len(v0_points)} 个对应点")
    
    if len(v0_points) < 2:
        print("\n警告: 只有 1 个对应点，只能计算纯平移变换")
        delta = np.array(v1_points[0]) - np.array(v0_points[0])
        result = {
            'theta_rad': 0,
            'theta_deg': 0,
            'tx': float(delta[0]),
            'ty': float(delta[1]),
            'rmse': 0,
            'method': 'pure_translation'
        }
    else:
        # 计算刚性变换
        result = compute_rigid_transform(v0_points, v1_points)
        result['method'] = 'rigid_transform'
    
    # 打印结果
    print("\n" + "=" * 70)
    print("计算结果")
    print("=" * 70)
    
    print(f"\n旋转角度: {result['theta_deg']:.4f}° ({result['theta_rad']:.6f} rad)")
    print(f"平移向量: ({result['tx']:.4f}, {result['ty']:.4f}) m")
    print(f"RMSE: {result.get('rmse', 0):.4f} m")
    
    # 验证
    print("\n对应点验证:")
    theta, tx, ty = result['theta_rad'], result['tx'], result['ty']
    for i, (v0, v1) in enumerate(zip(v0_points, v1_points)):
        v1_pred = apply_transform(v0, theta, tx, ty)
        error = np.linalg.norm(np.array(v1_pred) - np.array(v1))
        print(f"  点{i+1}: v0={v0} -> 预测 v1={v1_pred[0]:.2f}, {v1_pred[1]:.2f}, 实际 v1={v1}, 误差={error:.3f}m")
    
    # 生成代码
    print("\n" + "=" * 70)
    print("Python 代码 (可直接复制到 batch_compare_simple.py)")
    print("=" * 70)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    code = f"""
V0_TO_V1_TRANSFORM = {{
    'theta_deg': {result['theta_deg']:.6f},
    'theta_rad': {result['theta_rad']:.6f},
    'tx': {result['tx']:.6f},
    'ty': {result['ty']:.6f},
    'pivot_x': 0.0,  # 使用原点作为 pivot
    'pivot_y': 0.0,
    'cos_theta': {cos_t:.6f},
    'sin_theta': {sin_t:.6f},
}}

def transform_v0_to_v1(x0, y0):
    T = V0_TO_V1_TRANSFORM
    cos_t, sin_t = T['cos_theta'], T['sin_theta']
    tx, ty = T['tx'], T['ty']
    x1 = cos_t * x0 - sin_t * y0 + tx
    y1 = sin_t * x0 + cos_t * y0 + ty
    return x1, y1
"""
    print(code)
    
    # 保存结果
    output_path = "results/transform_from_points.json"
    os.makedirs("results", exist_ok=True)
    
    result['v0_points'] = v0_points
    result['v1_points'] = v1_points
    result['cos_theta'] = float(cos_t)
    result['sin_theta'] = float(sin_t)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
