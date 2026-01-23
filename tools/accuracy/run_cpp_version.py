#!/usr/bin/env python3
"""
C++ 版本规划测试

使用 C++ 版本的规划库 + C++ 版本的 tomogram bin 文件

注意：C++ 版本的 ele_planner 库与 Python 版本是独立编译的，
虽然接口相同，但内部实现可能有差异。
"""

import os
import sys
import csv
import json
import time
import struct
import argparse

import numpy as np

# ============================================================
# 关键：使用 C++ 版本的库路径
# ============================================================
CPP_PLANNER_ROOT = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
GTSAM_LIB_PATH = "/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"

# 确保 C++ 版本的库优先加载
sys.path.insert(0, CPP_PLANNER_ROOT)

# 设置动态库搜索路径
os.environ['LD_LIBRARY_PATH'] = CPP_PLANNER_ROOT + ':' + GTSAM_LIB_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# 导入 C++ 版本的库
import a_star
import ele_planner
import traj_opt

# 默认路径 - C++ 版本的 tomogram
DEFAULT_BIN_PATH = "/home/lzy/PctPlanner/PctPlanner_Cpp/rsc/tomogram/scene_map.bin"
DEFAULT_INPUT_CSV = "test_cases.csv"
DEFAULT_OUTPUT_JSON = "cpp_version_results.json"


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


def load_tomogram_bin(bin_path: str) -> dict:
    """
    加载 C++ 版本的 tomogram bin 文件
    
    二进制格式（参考 tomogram_format.hpp）:
    struct TomogramHeader {
      char magic[4];          // "TMG1"
      uint16_t version;       // 版本号
      uint16_t precision_mode;// 精度模式 (0=fp16, 1=fp32)
      uint32_t n_slice;       // 切片数量
      uint32_t dim_x;         // 栅格宽
      uint32_t dim_y;         // 栅格高
      float resolution;       // 分辨率
      float center_x;         // 地图中心 x
      float center_y;         // 地图中心 y
      float slice_h0;         // 首切片高度
      float slice_dh;         // 切片间距
    };
    """
    with open(bin_path, 'rb') as f:
        # 读取 header (40 bytes)
        magic = f.read(4).decode('ascii')
        if magic != 'TMG1':
            raise ValueError(f"Invalid magic: {magic}")
        
        version = struct.unpack('H', f.read(2))[0]          # uint16_t
        precision_mode = struct.unpack('H', f.read(2))[0]   # uint16_t
        n_slice = struct.unpack('I', f.read(4))[0]          # uint32_t
        dim_x = struct.unpack('I', f.read(4))[0]            # uint32_t
        dim_y = struct.unpack('I', f.read(4))[0]            # uint32_t
        resolution = struct.unpack('f', f.read(4))[0]       # float
        center_x = struct.unpack('f', f.read(4))[0]         # float
        center_y = struct.unpack('f', f.read(4))[0]         # float
        slice_h0 = struct.unpack('f', f.read(4))[0]         # float
        slice_dh = struct.unpack('f', f.read(4))[0]         # float
        
        # 确定标量大小
        scalar_size = 2 if precision_mode == 0 else 4
        dtype = np.float16 if precision_mode == 0 else np.float32
        
        # 读取 slice_heights
        slice_heights_bytes = f.read(n_slice * scalar_size)
        slice_heights = np.frombuffer(slice_heights_bytes, dtype=dtype).astype(np.float32)
        
        # 读取数据: [5, n_slice, dim_x, dim_y]
        data_size = 5 * n_slice * dim_x * dim_y * scalar_size
        data_bytes = f.read(data_size)
        data = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
        data = data.reshape(5, n_slice, dim_x, dim_y)
    
    return {
        'version': version,
        'precision_mode': precision_mode,
        'n_slice': n_slice,
        'dim_x': dim_x,
        'dim_y': dim_y,
        'resolution': resolution,
        'center': np.array([center_x, center_y]),
        'slice_heights': slice_heights.tolist(),
        'slice_h0': slice_h0,
        'slice_dh': slice_dh,
        'data': data
    }


class CppVersionPlanner:
    """
    C++ 版本规划器
    
    使用：
    - C++ 版本的 ele_planner 库
    - C++ 版本的 tomogram bin 文件
    """
    
    def __init__(self, bin_path: str, use_quintic: bool = True, max_heading_rate: float = 10.0):
        self.use_quintic = use_quintic
        self.max_heading_rate = max_heading_rate
        
        # 加载 C++ 版本的 tomogram
        print(f"  加载 C++ tomogram: {bin_path}", file=sys.stderr)
        tomo = load_tomogram_bin(bin_path)
        
        self.resolution = float(tomo['resolution'])
        self.center = tomo['center'].astype(np.float64)
        self.n_slice = tomo['n_slice']
        self.slice_heights = tomo['slice_heights']
        
        # C++ bin 文件中的 dim_x, dim_y 定义
        # 注意：reshape 时的顺序要与 init_map 期望的一致
        self.dim_x = tomo['dim_x']  # 466
        self.dim_y = tomo['dim_y']  # 778
        self.map_dim = [self.dim_x, self.dim_y]
        self.offset = np.array([int(self.dim_x / 2), int(self.dim_y / 2)], dtype=np.int32)
        
        # 获取库版本信息
        lib_path = ele_planner.__file__
        print(f"  使用 ele_planner 库: {lib_path}", file=sys.stderr)
        print(f"  Tomogram 维度: dim_x={self.dim_x}, dim_y={self.dim_y}, n_slice={self.n_slice}", file=sys.stderr)
        
        # 解析 tomogram 数据
        # data shape: [5, n_slice, dim_x, dim_y]
        tomogram = tomo['data']
        trav = tomogram[0]      # [n_slice, dim_x, dim_y]
        trav_gx = tomogram[1]
        trav_gy = tomogram[2]
        elev_g = tomogram[3]
        elev_g = np.nan_to_num(elev_g, nan=-100)
        elev_c = tomogram[4]
        elev_c = np.nan_to_num(elev_c, nan=1e6)
        
        self._init_planner(trav, trav_gx, trav_gy, elev_g, elev_c)
    
    def _init_planner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        """初始化规划器"""
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[1:]))
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[:-1]))
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        
        # init_map 期望: trav.reshape(-1, trav.shape[-1])
        # trav shape: [n_slice, dim_x, dim_y] -> reshape to [n_slice*dim_x, dim_y]
        self.planner.init_map(
            20, 15, self.resolution, self.n_slice, 0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )
    
    def z2slice(self, z):
        """高度转切片索引"""
        slice_idx = np.searchsorted(self.slice_heights, z, side='right') - 1
        return np.maximum(0, slice_idx)
    
    def pos2idx(self, pos):
        """世界坐标转栅格索引"""
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        # 注意：C++ 版本的 x/y 定义可能与 Python 版本不同
        idx = np.array([idx[1], idx[0]], dtype=np.int32)
        return idx
    
    def plan(self, start_x, start_y, start_z, end_x, end_y, end_z):
        """执行规划"""
        start_pos = np.array([start_x, start_y], dtype=np.float64)
        end_pos = np.array([end_x, end_y], dtype=np.float64)
        
        start_z_adj = start_z + 0.5
        end_z_adj = end_z + 0.5
        
        start_slice_idx = np.clip(self.z2slice(start_z_adj), 0, self.n_slice - 1).astype(np.int32)
        end_slice_idx = np.clip(self.z2slice(end_z_adj), 0, self.n_slice - 1).astype(np.int32)
        
        start_idx = np.zeros(3, dtype=np.int32)
        end_idx = np.zeros(3, dtype=np.int32)
        
        start_idx[0] = start_slice_idx
        start_idx[1:] = self.pos2idx(start_pos)
        
        end_idx[0] = end_slice_idx
        end_idx[1:] = self.pos2idx(end_pos)
        
        try:
            self.planner.plan(start_idx, end_idx, True)
        except:
            return None
        
        path_finder = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None
        
        optimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )
        
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()
        
        # 与 Python 版本一致：先 concatenate layers，再计算 y_idx
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        
        traj_3d = self._trans_traj_grid2map(traj_3d)
        
        return traj_3d
    
    def _trans_traj_grid2map(self, traj_grid):
        """栅格坐标到世界坐标转换"""
        # C++ 版本的转换逻辑
        offset = np.array([self.dim_y // 2, self.dim_x // 2, 0])
        center_ = np.array([self.center[1], self.center[0], 0.5])
        
        traj_grid = (traj_grid - offset) * self.resolution + center_
        traj_map = np.stack([traj_grid[:, 1], traj_grid[:, 0], traj_grid[:, 2]], axis=1)
        
        return traj_map


import gc


def run_tests(test_cases: list, bin_path: str, verbose: bool = False) -> dict:
    """执行测试 - 增量保存结果以减少内存使用"""
    print("初始化 C++ 版本规划器...", file=sys.stderr)
    planner = CppVersionPlanner(bin_path)
    
    results = {}
    success_count = 0
    
    # 每 50 个用例强制垃圾回收，避免内存积累
    gc_interval = 50
    
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
        if (i + 1) % gc_interval == 0:
            gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='C++ 版本规划测试')
    parser.add_argument('--bin', type=str, default=DEFAULT_BIN_PATH,
                        help='C++ 版本 Tomogram bin 文件路径')
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
    
    print("=" * 60, file=sys.stderr)
    print("C++ 版本规划测试", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  库路径: {CPP_PLANNER_ROOT}", file=sys.stderr)
    print(f"  Tomogram: {args.bin}", file=sys.stderr)
    
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
    results = run_tests(test_cases, args.bin, args.verbose)
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
            'version': 'cpp',
            'lib_path': CPP_PLANNER_ROOT,
            'tomogram_path': args.bin,
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
