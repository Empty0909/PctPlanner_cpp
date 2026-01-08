#!/usr/bin/env python3
"""Compare or convert tomogram files between the C++ binary format and Python pickle.

This script helps ensure the CUDA C++ pipeline (tomogram_format.hpp) stays
bitwise-aligned with the legacy Python/CuPy implementation that stores
``{'data': np.ndarray(5, n_slice, dim_y, dim_x), ...}`` pickles.
"""

from __future__ import annotations

import argparse
import math
import pickle
import struct
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# 旧版 NumPy pickle 会引用 numpy._core.* 模块名，PyPI 新版只保留 numpy.core.*。
# 若检测到缺失则创建别名，保证 pickle.load 能在现代环境运行。
if "numpy._core" not in sys.modules and hasattr(np, "core"):
    alias = types.ModuleType("numpy._core")
    alias.__dict__.update(np.core.__dict__)
    sys.modules["numpy._core"] = alias

MAGIC = b"TMG1"
HEADER_FMT = "<4sHHIII5f"  # magic, version, precision, n_slice, dim_x, dim_y, 5 floats
tHeader = struct.Struct(HEADER_FMT)
LAYER_NAMES = ("trav", "trav_gx", "trav_gy", "elev_g", "elev_c")
PRECISION_LUT = {0: np.float16, 1: np.float32}


class Tomogram:
    def __init__(self, header: Dict[str, float], slice_heights: np.ndarray, data: np.ndarray,
                 precision: int) -> None:
        self.header = header
        self.slice_heights = slice_heights.astype(np.float32, copy=False)
        self.data = data.astype(np.float32, copy=False)
        self.precision = precision

    @property
    def n_slice(self) -> int:
        return int(self.header["n_slice"])

    @property
    def dim_x(self) -> int:
        return int(self.header["dim_x"])

    @property
    def dim_y(self) -> int:
        return int(self.header["dim_y"])


def load_cpp_binary(path: Path) -> Tomogram:
    data = path.read_bytes()
    if len(data) < tHeader.size:
        raise ValueError("File too small to contain a tomogram header")

    magic, version, precision_mode, n_slice, dim_x, dim_y, resolution, center_x, center_y, slice_h0, slice_dh = (
        tHeader.unpack_from(data, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"Magic mismatch: expected {MAGIC!r}, got {magic!r}")
    if version != 1:
        raise ValueError(f"Unsupported version {version}")
    if precision_mode not in PRECISION_LUT:
        raise ValueError(f"Unsupported precision mode {precision_mode}")

    scalar_dtype = PRECISION_LUT[precision_mode]
    scalar_bytes = np.dtype(scalar_dtype).itemsize
    offset = tHeader.size

    slice_heights = np.frombuffer(data, dtype=scalar_dtype, count=n_slice, offset=offset).astype(np.float32)
    offset += n_slice * scalar_bytes

    layer_voxels = len(LAYER_NAMES) * n_slice * dim_x * dim_y
    expected_bytes = layer_voxels * scalar_bytes
    if offset + expected_bytes > len(data):
        raise ValueError("Payload truncated: file ended before all voxel layers were read")

    payload = np.frombuffer(data, dtype=scalar_dtype, count=layer_voxels, offset=offset)
    # C++ 和 Python 现在都使用相同的布局: [layers][n_slice][dim_x][dim_y]
    # 注意: header 中的 dim_x/dim_y 定义与 Python 一致
    cube = payload.reshape(len(LAYER_NAMES), n_slice, dim_x, dim_y).astype(np.float32)

    header = {
        "version": version,
        "precision_mode": precision_mode,
        "n_slice": n_slice,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "resolution": resolution,
        "center_x": center_x,
        "center_y": center_y,
        "slice_h0": slice_h0,
        "slice_dh": slice_dh,
    }
    return Tomogram(header, slice_heights, cube, precision_mode)


def load_python_pickle(path: Path) -> Tomogram:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if "data" not in payload:
        raise ValueError("Python pickle missing 'data' key")

    cube = np.asarray(payload["data"], dtype=np.float32)
    if cube.ndim != 4:
        raise ValueError("Expected data array with 4 dimensions (5 x n_slice x H x W or H x W x 5)")
    if cube.shape[0] != len(LAYER_NAMES) and cube.shape[-1] == len(LAYER_NAMES):
        cube = np.transpose(cube, (3, 0, 1, 2))
    if cube.shape[0] != len(LAYER_NAMES):
        raise ValueError("Data array must expose 5 layers")

    n_slice = cube.shape[1]
    # Python 布局: (5, n_slice, dim_x, dim_y)
    # shape[2] 是 dim_x, shape[3] 是 dim_y
    dim_x = cube.shape[2]
    dim_y = cube.shape[3]

    header = {
        "version": 1,
        "precision_mode": 1,  # pickles are always float32
        "n_slice": n_slice,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "resolution": float(payload.get("resolution", np.nan)),
        "center_x": float(np.asarray(payload.get("center", [np.nan, np.nan]))[0]),
        "center_y": float(np.asarray(payload.get("center", [np.nan, np.nan]))[1]),
        "slice_h0": float(payload.get("slice_h0", np.nan)),
        "slice_dh": float(payload.get("slice_dh", np.nan)),
    }

    if "slice_heights" in payload:
        slice_heights = np.asarray(payload["slice_heights"], dtype=np.float32)
    else:
        indices = np.arange(n_slice, dtype=np.float32)
        slice_heights = header["slice_h0"] + indices * header["slice_dh"]

    return Tomogram(header, slice_heights, cube, header["precision_mode"])


def compare_tomograms(cpp: Tomogram, py: Tomogram, abs_tol: float, rel_tol: float) -> bool:
    ok = True

    def check(condition: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        status = "OK" if condition else "FAIL"
        print(f"[{status}] {label}{(': ' + detail) if detail else ''}")
        if not condition:
            ok = False

    scalar_fields = ("n_slice", "dim_x", "dim_y", "resolution", "center_x",
                     "center_y", "slice_h0", "slice_dh")
    for field in scalar_fields:
        c_val = cpp.header[field]
        p_val = py.header[field]
        if math.isnan(c_val) or math.isnan(p_val):
            detail = f"C++={c_val}, Py={p_val} (skipping)"
            check(True, f"{field}", detail)
            continue
        diff = abs(c_val - p_val)
        detail = f"C++={c_val:g}, Py={p_val:g}, diff={diff:g}"
        check(diff <= abs_tol, field, detail)

    # 找到共同的切片高度进行比较
    common_heights = []
    cpp_indices = []
    py_indices = []
    for ci, ch in enumerate(cpp.slice_heights):
        for pi, ph in enumerate(py.slice_heights):
            if abs(ch - ph) < 0.01:  # 高度匹配
                common_heights.append(ch)
                cpp_indices.append(ci)
                py_indices.append(pi)
                break
    
    print(f"    DEBUG: C++ slice_heights[:5] = {list(cpp.slice_heights[:5])}")
    print(f"    DEBUG: Py  slice_heights[:5] = {list(py.slice_heights[:5])}")
    print(f"    DEBUG: C++ indices for common = {cpp_indices[:5]}")
    print(f"    DEBUG: Py  indices for common = {py_indices[:5]}")
    
    if len(cpp.slice_heights) == len(py.slice_heights):
        diff = np.max(np.abs(cpp.slice_heights - py.slice_heights))
        check(diff <= abs_tol, "slice_heights", f"max_abs={diff:g}")
    else:
        check(False, "slice_heights", f"length mismatch (C++={len(cpp.slice_heights)}, Py={len(py.slice_heights)}, common={len(common_heights)})")
        print(f"    Common slice heights: {common_heights[:5]}...{common_heights[-5:]}" if len(common_heights) > 10 else f"    Common slice heights: {common_heights}")
        # 找出哪些高度不同
        cpp_set = set(round(h, 2) for h in cpp.slice_heights)
        py_set = set(round(h, 2) for h in py.slice_heights)
        only_cpp = sorted(cpp_set - py_set)
        only_py = sorted(py_set - cpp_set)
        if only_cpp:
            print(f"    C++ only heights: {only_cpp}")
        if only_py:
            print(f"    Python only heights: {only_py}")

    if cpp.data.shape != py.data.shape:
        check(False, "data shape", f"C++={cpp.data.shape}, Py={py.data.shape}")
        # 如果只是 n_slice 不同，尝试比较共同切片
        if cpp.data.shape[0] == py.data.shape[0] and cpp.data.shape[2:] == py.data.shape[2:]:
            print("    Comparing common slices based on matching heights...")
            if len(cpp_indices) > 0:
                cpp_common = cpp.data[:, cpp_indices, :, :]
                py_common = py.data[:, py_indices, :, :]
                delta = cpp_common - py_common
                
                # 统计分析
                cpp_trav = cpp_common[0, 0, :, :].flatten()
                py_trav = py_common[0, 0, :, :].flatten()
                
                print(f"\n    DEBUG: Slice 0 trav statistics:")
                print(f"    C++: min={np.nanmin(cpp_trav):.3f}, max={np.nanmax(cpp_trav):.3f}, mean={np.nanmean(cpp_trav):.3f}, std={np.nanstd(cpp_trav):.3f}")
                print(f"    Py:  min={np.nanmin(py_trav):.3f}, max={np.nanmax(py_trav):.3f}, mean={np.nanmean(py_trav):.3f}, std={np.nanstd(py_trav):.3f}")
                
                # 统计值分布
                cpp_zeros = np.sum(cpp_trav == 0)
                py_zeros = np.sum(py_trav == 0)
                cpp_barriers = np.sum(cpp_trav >= 50)
                py_barriers = np.sum(py_trav >= 50)
                print(f"    C++: zeros={cpp_zeros}, barriers(>=50)={cpp_barriers}")
                print(f"    Py:  zeros={py_zeros}, barriers(>=50)={py_barriers}")
                
                # 检查非零值的分布
                cpp_nonzero = np.count_nonzero(cpp_common[0])
                py_nonzero = np.count_nonzero(py_common[0])
                print(f"\n    Slice 0 trav: C++ non-zero count = {cpp_nonzero}, Py non-zero count = {py_nonzero}")
                
                # 忽略 nan 进行比较
                delta_finite = np.where(np.isfinite(delta), delta, 0)
                max_abs = float(np.nanmax(np.abs(delta)))
                rms = float(np.sqrt(np.nanmean(delta_finite ** 2)))
                print(f"\n    Common slices comparison: max_abs={max_abs:.3e}, rms={rms:.3e}")
                for idx, name in enumerate(LAYER_NAMES):
                    layer_delta = delta[idx]
                    layer_max = float(np.nanmax(np.abs(layer_delta)))
                    layer_rms = float(np.sqrt(np.nanmean(np.where(np.isfinite(layer_delta), layer_delta, 0) ** 2)))
                    # 检查差异的位置
                    if layer_max > 1.0:
                        # 找出差异最大的位置
                        abs_delta = np.abs(layer_delta)
                        abs_delta = np.where(np.isfinite(abs_delta), abs_delta, 0)
                        max_idx = np.unravel_index(np.argmax(abs_delta), layer_delta.shape)
                        cpp_val = cpp_common[idx][max_idx]
                        py_val = py_common[idx][max_idx]
                        print(f"      Layer {idx} ({name}): max_abs={layer_max:.3e}, rms={layer_rms:.3e}, max_at={max_idx}, c++={cpp_val:.3f}, py={py_val:.3f}")
                    else:
                        print(f"      Layer {idx} ({name}): max_abs={layer_max:.3e}, rms={layer_rms:.3e}")
        return False

    delta = cpp.data - py.data
    max_abs = float(np.nanmax(np.abs(delta)))
    rms = float(np.sqrt(np.nanmean(np.where(np.isfinite(delta), delta, 0) ** 2)))
    denom = np.maximum(np.abs(py.data), 1e-12)
    max_rel = float(np.nanmax(np.abs(delta) / denom))
    condition = (max_abs <= abs_tol) or (max_rel <= rel_tol)
    check(condition, "voxel data", f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, rms={rms:.3e}")

    # Trav layer difference distribution
    trav_diff = delta[0]
    trav_diff_finite = np.where(np.isfinite(trav_diff), trav_diff, 0)
    abs_diff = np.abs(trav_diff_finite)
    exact_match = np.sum(abs_diff < 0.01)
    small_diff = np.sum((abs_diff >= 0.01) & (abs_diff < 1.0))
    medium_diff = np.sum((abs_diff >= 1.0) & (abs_diff < 10.0))
    large_diff = np.sum((abs_diff >= 10.0) & (abs_diff < 50.0))
    barrier_diff = np.sum(abs_diff >= 50.0)
    total = trav_diff.size
    
    print(f"\n    Trav layer difference distribution:")
    print(f"      Exact match (<0.01): {exact_match} ({100*exact_match/total:.2f}%)")
    print(f"      Small diff (0.01-1): {small_diff} ({100*small_diff/total:.2f}%)")
    print(f"      Medium diff (1-10): {medium_diff} ({100*medium_diff/total:.2f}%)")
    print(f"      Large diff (10-50): {large_diff} ({100*large_diff/total:.2f}%)")
    print(f"      Barrier diff (>=50): {barrier_diff} ({100*barrier_diff/total:.2f}%)")
    
    # 找一个差异大的位置看看
    if np.max(abs_diff) > 1:
        max_idx = np.unravel_index(np.argmax(abs_diff), trav_diff.shape)
        s, x, y = max_idx
        print(f"\n    Max trav diff at {max_idx}: C++={cpp.data[0][max_idx]:.3f}, Py={py.data[0][max_idx]:.3f}")
        print(f"    Neighborhood at slice {s}, x={x}, y={y}:")
        print(f"    C++ trav:\n{cpp.data[0, s, max(0,x-2):x+3, max(0,y-2):y+3]}")
        print(f"    Py  trav:\n{py.data[0, s, max(0,x-2):x+3, max(0,y-2):y+3]}")
        print(f"    C++ elev_g:\n{cpp.data[3, s, max(0,x-2):x+3, max(0,y-2):y+3]}")
        print(f"    Py  elev_g:\n{py.data[3, s, max(0,x-2):x+3, max(0,y-2):y+3]}")

    # Per-layer stats for easier inspection
    for idx, name in enumerate(LAYER_NAMES):
        layer_delta = delta[idx]
        layer_max = float(np.nanmax(np.abs(layer_delta)))
        layer_rms = float(np.sqrt(np.nanmean(np.where(np.isfinite(layer_delta), layer_delta, 0) ** 2)))
        print(f"    Layer {idx} ({name}): max_abs={layer_max:.3e}, rms={layer_rms:.3e}")

    return ok and condition


def export_pickle_from_cpp(cpp: Tomogram, out_path: Path) -> None:
    payload = {
        "data": cpp.data.astype(np.float32),
        "resolution": cpp.header["resolution"],
        "center": np.array([cpp.header["center_x"], cpp.header["center_y"]], dtype=np.float32),
        "slice_h0": cpp.header["slice_h0"],
        "slice_dh": cpp.header["slice_dh"],
        "slice_heights": cpp.slice_heights.astype(np.float32),
        "n_slice": cpp.header["n_slice"],
        "dim_x": cpp.header["dim_x"],
        "dim_y": cpp.header["dim_y"],
        "precision_mode": cpp.precision,
    }
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote pickle: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare/convert tomogram files")
    parser.add_argument("--bin", type=Path, required=True,
                        help="C++ tomogram .bin path")
    parser.add_argument("--pickle", type=Path,
                        help="Python pickle path for comparison")
    parser.add_argument("--export-pickle", type=Path,
                        help="Optional output path to dump the C++ file as pickle")
    parser.add_argument("--abs-tol", type=float, default=1e-4,
                        help="Absolute tolerance for numeric comparison")
    parser.add_argument("--rel-tol", type=float, default=1e-3,
                        help="Relative tolerance for numeric comparison")

    args = parser.parse_args()

    cpp = load_cpp_binary(args.bin)
    print(f"Loaded C++ tomogram: n_slice={cpp.n_slice}, dim=({cpp.dim_x},{cpp.dim_y}), "
          f"precision={'fp32' if cpp.precision == 1 else 'fp16'}")

    if args.export_pickle:
        export_pickle_from_cpp(cpp, args.export_pickle)

    if args.pickle:
        py = load_python_pickle(args.pickle)
        print(f"Loaded Python tomogram: n_slice={py.n_slice}, dim=({py.dim_x},{py.dim_y})")
        ok = compare_tomograms(cpp, py, args.abs_tol, args.rel_tol)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
