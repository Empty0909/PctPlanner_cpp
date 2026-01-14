#!/usr/bin/env python3
"""
Tomogram 格式转换工具

支持在 C++ 二进制格式和 Python pickle 格式之间互相转换，
用于验证两个版本规划器在相同代价地图下的行为一致性。

使用示例：
  # Python pickle -> C++ binary
  python tomogram_convert.py pickle2bin input.pickle output.bin
  
  # C++ binary -> Python pickle
  python tomogram_convert.py bin2pickle input.bin output.pickle
  
  # 验证转换正确性（往返转换后比较）
  python tomogram_convert.py verify input.pickle
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import types
from pathlib import Path
from typing import Dict, Any

import numpy as np

# 兼容旧版 NumPy pickle
if "numpy._core" not in sys.modules and hasattr(np, "core"):
    alias = types.ModuleType("numpy._core")
    alias.__dict__.update(np.core.__dict__)
    sys.modules["numpy._core"] = alias

# C++ 二进制格式常量
MAGIC = b"TMG1"
HEADER_FMT = "<4sHHIII5f"  # magic, version, precision, n_slice, dim_x, dim_y, 5 floats
HEADER_SIZE = struct.calcsize(HEADER_FMT)
LAYER_NAMES = ("trav", "trav_gx", "trav_gy", "elev_g", "elev_c")


def load_pickle(path: Path) -> Dict[str, Any]:
    """加载 Python pickle 格式的 tomogram"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    cube = np.asarray(data["data"], dtype=np.float32)
    if cube.ndim != 4 or cube.shape[0] != 5:
        raise ValueError(f"Expected shape (5, n_slice, dim_x, dim_y), got {cube.shape}")
    
    n_slice = cube.shape[1]
    dim_x = cube.shape[2]
    dim_y = cube.shape[3]
    
    resolution = float(data["resolution"])
    center = np.asarray(data["center"], dtype=np.float64)
    slice_h0 = float(data["slice_h0"])
    slice_dh = float(data["slice_dh"])
    
    if "slice_heights" in data:
        slice_heights = np.asarray(data["slice_heights"], dtype=np.float32)
    else:
        slice_heights = slice_h0 + np.arange(n_slice, dtype=np.float32) * slice_dh
    
    return {
        "data": cube,
        "n_slice": n_slice,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "resolution": resolution,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "slice_h0": slice_h0,
        "slice_dh": slice_dh,
        "slice_heights": slice_heights,
    }


def load_binary(path: Path) -> Dict[str, Any]:
    """加载 C++ 二进制格式的 tomogram"""
    data = path.read_bytes()
    
    if len(data) < HEADER_SIZE:
        raise ValueError("File too small for header")
    
    magic, version, precision, n_slice, dim_x, dim_y, resolution, cx, cy, h0, dh = (
        struct.unpack_from(HEADER_FMT, data, 0)
    )
    
    if magic != MAGIC:
        raise ValueError(f"Magic mismatch: expected {MAGIC!r}, got {magic!r}")
    if version != 1:
        raise ValueError(f"Unsupported version {version}")
    
    dtype = np.float16 if precision == 0 else np.float32
    scalar_bytes = np.dtype(dtype).itemsize
    offset = HEADER_SIZE
    
    # 读取 slice_heights
    slice_heights = np.frombuffer(data, dtype=dtype, count=n_slice, offset=offset).astype(np.float32)
    offset += n_slice * scalar_bytes
    
    # 读取体素数据
    voxels = 5 * n_slice * dim_x * dim_y
    cube = np.frombuffer(data, dtype=dtype, count=voxels, offset=offset).astype(np.float32)
    cube = cube.reshape(5, n_slice, dim_x, dim_y)
    
    return {
        "data": cube,
        "n_slice": n_slice,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "resolution": resolution,
        "center_x": cx,
        "center_y": cy,
        "slice_h0": h0,
        "slice_dh": dh,
        "slice_heights": slice_heights,
        "precision": precision,
    }


def save_pickle(tomo: Dict[str, Any], path: Path):
    """保存为 Python pickle 格式"""
    data = {
        "data": tomo["data"].astype(np.float32),
        "resolution": tomo["resolution"],
        "center": np.array([tomo["center_x"], tomo["center_y"]], dtype=np.float64),
        "slice_h0": tomo["slice_h0"],
        "slice_dh": tomo["slice_dh"],
        "slice_heights": tomo["slice_heights"].astype(np.float32),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved pickle: {path} ({path.stat().st_size} bytes)")


def save_binary(tomo: Dict[str, Any], path: Path, precision: int = 0):
    """保存为 C++ 二进制格式
    
    Args:
        tomo: tomogram 数据字典
        path: 输出路径
        precision: 0=float16, 1=float32
    """
    dtype = np.float16 if precision == 0 else np.float32
    
    # 构建 header
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        1,  # version
        precision,
        tomo["n_slice"],
        tomo["dim_x"],
        tomo["dim_y"],
        tomo["resolution"],
        tomo["center_x"],
        tomo["center_y"],
        tomo["slice_h0"],
        tomo["slice_dh"],
    )
    
    # 转换数据
    slice_heights = tomo["slice_heights"].astype(dtype)
    cube = tomo["data"].astype(dtype)
    
    with open(path, "wb") as f:
        f.write(header)
        f.write(slice_heights.tobytes())
        f.write(cube.tobytes())
    
    print(f"Saved binary: {path} ({path.stat().st_size} bytes, precision={'fp16' if precision == 0 else 'fp32'})")


def pickle_to_binary(pickle_path: Path, binary_path: Path, precision: int = 0):
    """将 pickle 转换为 C++ 二进制格式"""
    print(f"Converting: {pickle_path} -> {binary_path}")
    tomo = load_pickle(pickle_path)
    print(f"  n_slice={tomo['n_slice']}, dim=({tomo['dim_x']}, {tomo['dim_y']})")
    print(f"  resolution={tomo['resolution']}, center=({tomo['center_x']:.4f}, {tomo['center_y']:.4f})")
    save_binary(tomo, binary_path, precision)


def binary_to_pickle(binary_path: Path, pickle_path: Path):
    """将 C++ 二进制格式转换为 pickle"""
    print(f"Converting: {binary_path} -> {pickle_path}")
    tomo = load_binary(binary_path)
    print(f"  n_slice={tomo['n_slice']}, dim=({tomo['dim_x']}, {tomo['dim_y']})")
    print(f"  resolution={tomo['resolution']}, center=({tomo['center_x']:.4f}, {tomo['center_y']:.4f})")
    save_pickle(tomo, pickle_path)


def verify_roundtrip(path: Path):
    """验证往返转换的正确性"""
    import tempfile
    
    if path.suffix == ".pickle":
        original = load_pickle(path)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_bin = Path(tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as tmp:
            tmp_pickle = Path(tmp.name)
        
        save_binary(original, tmp_bin, precision=1)  # 使用 fp32 避免精度损失
        tomo = load_binary(tmp_bin)
        save_pickle(tomo, tmp_pickle)
        roundtrip = load_pickle(tmp_pickle)
        
        tmp_bin.unlink()
        tmp_pickle.unlink()
    else:
        original = load_binary(path)
        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as tmp:
            tmp_pickle = Path(tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_bin = Path(tmp.name)
        
        save_pickle(original, tmp_pickle)
        tomo = load_pickle(tmp_pickle)
        precision = original.get("precision", 1)
        save_binary(tomo, tmp_bin, precision)
        roundtrip = load_binary(tmp_bin)
        
        tmp_pickle.unlink()
        tmp_bin.unlink()
    
    # 比较
    print("\n=== 往返转换验证 ===")
    all_ok = True
    
    for key in ["n_slice", "dim_x", "dim_y", "resolution", "center_x", "center_y", "slice_h0", "slice_dh"]:
        orig_val = original[key]
        rt_val = roundtrip[key]
        diff = abs(orig_val - rt_val)
        ok = diff < 1e-5
        status = "✓" if ok else "✗"
        print(f"  {status} {key}: {orig_val} -> {rt_val} (diff={diff:.2e})")
        if not ok:
            all_ok = False
    
    # 比较数据
    orig_data = original["data"]
    rt_data = roundtrip["data"]
    
    valid_mask = ~(np.isnan(orig_data) | np.isnan(rt_data))
    if valid_mask.sum() > 0:
        diff = np.abs(orig_data[valid_mask] - rt_data[valid_mask])
        max_diff = diff.max()
        mean_diff = diff.mean()
        ok = max_diff < 1e-3
        status = "✓" if ok else "✗"
        print(f"  {status} data: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        if not ok:
            all_ok = False
    
    print(f"\n{'✓ 验证通过' if all_ok else '✗ 验证失败'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Tomogram 格式转换工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # pickle -> binary
    p2b = subparsers.add_parser("pickle2bin", help="将 pickle 转换为 C++ 二进制格式")
    p2b.add_argument("input", type=Path, help="输入 pickle 文件")
    p2b.add_argument("output", type=Path, help="输出 binary 文件")
    p2b.add_argument("--fp32", action="store_true", help="使用 float32 精度（默认 float16）")
    
    # binary -> pickle
    b2p = subparsers.add_parser("bin2pickle", help="将 C++ 二进制格式转换为 pickle")
    b2p.add_argument("input", type=Path, help="输入 binary 文件")
    b2p.add_argument("output", type=Path, help="输出 pickle 文件")
    
    # verify
    ver = subparsers.add_parser("verify", help="验证往返转换正确性")
    ver.add_argument("input", type=Path, help="输入文件（pickle 或 binary）")
    
    args = parser.parse_args()
    
    if args.command == "pickle2bin":
        precision = 1 if args.fp32 else 0
        pickle_to_binary(args.input, args.output, precision)
    elif args.command == "bin2pickle":
        binary_to_pickle(args.input, args.output)
    elif args.command == "verify":
        verify_roundtrip(args.input)


if __name__ == "__main__":
    main()
