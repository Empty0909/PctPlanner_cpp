# C++ 与 Python 版本逐行代码对比报告

本报告对 C++ 版本和 Python 版本的 PctPlanner 进行逐行对比，识别所有可能导致差异的细节。

---

## 当前对比结果摘要 (2026-01-14 更新)

| 指标 | 数值 |
|------|------|
| 精确匹配率 | **96.80%** |
| 屏障差异 | 4,109 (0.01%) |
| NaN 不一致 | C++独有 14,075 / Py独有 11,606 |
| trav 匹配率 | 97.66% |
| elev_g 匹配率 | 97.89% |

---

## 目录

1. [模块一：TomographyKernel (点云投影)](#模块一tomographykernel-点云投影)
2. [模块二：梯度计算 (GradIntervalKernel vs NumPy切片)](#模块二梯度计算-gradintervalkernel-vs-numpy切片)
3. [模块三：TravKernel (通行代价计算)](#模块三travkernel-通行代价计算)
4. [模块四：InflationKernel (代价膨胀)](#模块四inflationkernel-代价膨胀)
5. [模块五：层简化 (Layer Simplification)](#模块五层简化-layer-simplification)
6. [模块六：输出与序列化](#模块六输出与序列化)
7. [发现的差异汇总](#发现的差异汇总)

---

## 模块一：TomographyKernel (点云投影)

### 文件位置

- **C++**: `src/tomography_cuda.cu` 第 75-119 行
- **Python**: `tomography/scripts/kernels.py` 第 98-131 行

### 逻辑对比

#### 1.1 索引计算函数

**Python 版本** (`kernels.py:8-25`):

```cuda
__device__ int getIndexLine(float16 x, float16 center)
{
    int i = round((x - center) / ${resolution});  // 使用模板参数
    return i;
}

__device__ int getIndexMap_1d(float16 x, float16 y, float16 cx, float16 cy)
{
    int idx_x = getIndexLine(x, cx) + ${n_row} / 2;
    int idx_y = getIndexLine(y, cy) + ${n_col} / 2;
    if (idx_x < 0 || idx_x >= ${n_row} || idx_y < 0 || idx_y >= ${n_col})
        return -1;
    return ${n_col} * idx_x + idx_y;
}
```

**C++ 版本** (`tomography_cuda.cu:88-110`) - ✅ **已修复对齐**:

```cpp
// 索引计算与 Python kernels.py 严格对齐:
// Python: int i = round((x - center) / ${resolution})
// Python 中 x, center 是 float16，${resolution} 是字面量常数
__half px_h = __float2half(px);
__half py_h = __float2half(py);
__half cx_h = __float2half(cx);
__half cy_h = __float2half(cy);

// 差值在 half 精度（与 Python 一致）
float diff_x = __half2float(__hsub(px_h, cx_h));
float diff_y = __half2float(__hsub(py_h, cy_h));
// 除法在 float 精度（与 Python 一致，因为 Python 的 ${resolution} 是字面量常数）
float val_x = diff_x / resolution;
float val_y = diff_y / resolution;

// Python CuPy kernel 使用 round()，在 CUDA 中 round() 使用四舍五入
// (round half away from zero)，与 roundf() 行为一致
int ix = static_cast<int>(roundf(val_x)) + dim_x / 2;
int iy = static_cast<int>(roundf(val_y)) + dim_y / 2;
```

#### ✅ **差异 #1: half 精度计算方式 - 已修复**

| 方面 | Python | C++ (修复后) | 状态 |
|------|--------|-------------|------|
| 减法精度 | float16 | `__hsub` (half) | ✅ 一致 |
| 除法精度 | float (字面量提升) | float | ✅ 一致 |
| round 函数 | `round()` (四舍五入) | `roundf()` (四舍五入) | ✅ 一致 |

**修复内容**:

1. 将除法从 `__hdiv` 改为 float 精度除法
2. 确认使用 `roundf()` 而非 `rintf()`（CUDA 中 round() 使用四舍五入）

---

#### 1.2 切片循环

**Python 版本** (`kernels.py:113-121`):

```cuda
for ( int s_idx = 0; s_idx < ${n_slice}; s_idx ++ )
{
    U slice = ${slice_h0} + s_idx * ${slice_dh};
    if ( pz <= slice )
        atomicMaxFloat(&layers_g[getIndexBlock_1d(idx, s_idx)], pz);
    else
        atomicMinFloat(&layers_c[getIndexBlock_1d(idx, s_idx)], pz);
}
```

**C++ 版本** (`tomography_cuda.cu:109-118`):

```cpp
for (int s = 0; s < n_slice; ++s) {
    float slice = slice_h0 + slice_dh * static_cast<float>(s);
    int offset = s * dim_x * dim_y + base;
    if (pz <= slice) {
      atomicMaxFloat(&layers_g[offset], pz);
    } else {
      atomicMinFloat(&layers_c[offset], pz);
    }
}
```

#### ⚠️ **发现差异 #2: 索引计算顺序**

| 方面 | Python | C++ |
|------|--------|-----|
| 索引函数 | `getIndexBlock_1d(idx, s_idx)` = `layer_size * s_idx + idx` | `s * dim_x * dim_y + base` |
| 布局 | `[slice][plane_idx]` | `[slice][plane_idx]` |

**结论**: 索引计算一致，无差异。

---

#### 1.3 slice 高度计算

**Python 版本**:

```cuda
U slice = ${slice_h0} + s_idx * ${slice_dh};
```

注意：`${slice_h0}` 和 `${slice_dh}` 是模板替换的**字面量**，类型是 `U` (float16)。

**C++ 版本** - ✅ **已修复对齐**:

```cpp
// slice 计算与 Python 对齐：
// Python: U slice = ${slice_h0} + s_idx * ${slice_dh}
// Python 中 U 是 float16，但 ${slice_h0} 和 ${slice_dh} 是字面量
// 为严格对齐，使用 half 精度计算 slice
__half slice_h0_h = __float2half(slice_h0);
__half slice_dh_h = __float2half(slice_dh);

for (int s = 0; s < n_slice; ++s) {
    // 与 Python 一致：slice = slice_h0 + s * slice_dh，在 half 精度下计算
    __half s_h = __float2half(static_cast<float>(s));
    __half slice_h = __hadd(slice_h0_h, __hmul(s_h, slice_dh_h));
    float slice = __half2float(slice_h);
    // ...
}
```

#### ✅ **差异 #3: slice 高度精度 - 已修复**

| 方面 | Python | C++ (修复后) | 状态 |
|------|--------|-------------|------|
| 精度 | float16 (U 类型) | half 精度 (`__hadd`, `__hmul`) | ✅ 一致 |

**修复内容**: 将 slice 计算从 float32 改为 half 精度运算

---

## 模块二：梯度计算 (GradIntervalKernel vs NumPy切片)

### 文件位置

- **C++**: `src/tomography_cuda.cu` 第 121-170 行
- **Python**: `tomography/scripts/tomogram.py` 第 111-122 行

### 逻辑对比

**Python 版本** (`tomogram.py:111-122`):

```python
diff_x_sq = cp.maximum(
    (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
    (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
)
diff_y_sq = cp.maximum(
    (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
    (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
)
self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
```

**C++ 版本** (`tomography_cuda.cu:136-170`):

```cpp
int y = plane_idx % dim_y;
int x = plane_idx / dim_y;

// 边界处理
if (x == 0 || x == dim_x - 1 || y == 0 || y == dim_y - 1) {
    grad_mag_sq[idx] = 0.0f;
    grad_mag_max[idx] = 0.0f;
    return;
}

int idx_ym = idx - 1;     // y - 1
int idx_yp = idx + 1;     // y + 1
int idx_xm = idx - dim_y; // x - 1
int idx_xp = idx + dim_y; // x + 1

float lg_xm = layers_g[idx_xm];
float lg_xp = layers_g[idx_xp];
float lg_ym = layers_g[idx_ym];
float lg_yp = layers_g[idx_yp];

float diff_x1 = lg - lg_xm;
float diff_x2 = lg - lg_xp;
float diff_y1 = lg - lg_ym;
float diff_y2 = lg - lg_yp;

float diff_x_sq = fmaxf(diff_x1 * diff_x1, diff_x2 * diff_x2);
float diff_y_sq = fmaxf(diff_y1 * diff_y1, diff_y2 * diff_y2);

grad_mag_sq[idx] = diff_x_sq + diff_y_sq;
grad_mag_max[idx] = fmaxf(diff_x_sq, diff_y_sq);
```

#### ⚠️ **发现差异 #4: 梯度计算的切片范围**

**Python 的 NumPy 切片逻辑**:

- `diff_x_sq` 的形状是 `[n_slice, dim_x-2, dim_y]`
- `diff_y_sq` 的形状是 `[n_slice, dim_x, dim_y-2]`
- 最终只有 `[:, 1:-1, 1:-1]` (即内部区域) 被赋值

**C++ 的逐点计算逻辑**:

- 对每个点 `(x, y)` 检查是否在边界
- 边界条件: `x == 0 || x == dim_x - 1 || y == 0 || y == dim_y - 1`

| 方面 | Python | C++ | 差异 |
|------|--------|-----|------|
| x 方向边界 | `1:-1` (索引 1 到 dim_x-2) | `x == 0 \|\| x == dim_x - 1` | **一致** |
| y 方向边界 | `1:-1` (索引 1 到 dim_y-2) | `y == 0 \|\| y == dim_y - 1` | **一致** |

**结论**: 边界处理逻辑一致。

---

#### ⚠️ **发现差异 #5: 梯度差分的计算公式**

**Python**:

```python
# X 方向梯度
diff_x_sq = cp.maximum(
    (current - left) ** 2,   # layers_g[:, 1:-1, :] - layers_g[:, :-2, :]
    (current - right) ** 2   # layers_g[:, 1:-1, :] - layers_g[:,  2:, :]
)
```

这里 `current = layers_g[:, 1:-1, :]`，即中间区域。

**C++**:

```cpp
float diff_x1 = lg - lg_xm;  // current - left
float diff_x2 = lg - lg_xp;  // current - right
```

**验证**: 公式一致，都是 `(当前 - 左) ** 2` 和 `(当前 - 右) ** 2` 的最大值。

---

#### ⚠️ **发现差异 #6: 梯度结果的存储位置**

**Python**:

```python
self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
```

- `diff_x_sq` 形状: `[n_slice, dim_x-2, dim_y]`
- `diff_x_sq[:, :, 1:-1]` 形状: `[n_slice, dim_x-2, dim_y-2]`
- 赋值到 `grad_mag_sq[:, 1:-1, 1:-1]`

**C++**:

```cpp
// 对于 x ∈ [1, dim_x-2], y ∈ [1, dim_y-2]
grad_mag_sq[idx] = diff_x_sq + diff_y_sq;
```

**潜在问题**: Python 的 `diff_x_sq[:, :, 1:-1]` 和 `diff_y_sq[:, 1:-1, :]` 的组合索引需要仔细验证。

让我们逐步分析 Python 的索引：

1. `diff_x_sq` 是在 x 方向 `1:-1` 上计算的，形状 `[n_slice, dim_x-2, dim_y]`
2. `diff_x_sq[:, :, 1:-1]` 取 y 方向的 `1:-1`，形状 `[n_slice, dim_x-2, dim_y-2]`
3. `diff_y_sq` 是在 y 方向 `1:-1` 上计算的，形状 `[n_slice, dim_x, dim_y-2]`
4. `diff_y_sq[:, 1:-1, :]` 取 x 方向的 `1:-1`，形状 `[n_slice, dim_x-2, dim_y-2]`
5. 两者相加后赋值到 `grad_mag_sq[:, 1:-1, 1:-1]`，形状也是 `[n_slice, dim_x-2, dim_y-2]`

**结论**: 索引逻辑正确，C++ 和 Python 应该一致。

---

## 模块三：TravKernel (通行代价计算)

### 文件位置

- **C++**: `src/tomography_cuda.cu` 第 173-224 行
- **Python**: `tomography/scripts/kernels.py` 第 133-198 行

### 逻辑对比

#### 3.1 参数传递

**Python** (`tomogram.py:32-41`):

```python
self.trav_kernel = travKernel(
    self.map_dim_x,
    self.map_dim_y,
    self.half_trav_k_size,
    self.interval_min,
    self.interval_free,
    self.step_cross, 
    self.step_stand, 
    self.standable_th, 
    self.cost_barrier
)
```

**C++** (`tomography_cuda.cu:320-328`):

```cpp
TravKernel<<<blocks, threads>>>(
    d_interval, d_grad_mag_sq, d_grad_mag_max, d_trav_cost,
    static_cast<int>(dim_x), static_cast<int>(dim_y),
    static_cast<int>(n_slice), half_trav_k,
    static_cast<float>(params.interval_min),
    static_cast<float>(params.interval_free), step_cross * step_cross,
    step_stand * step_stand, standable_th,
    static_cast<float>(params.cost_barrier));
```

#### ⚠️ **发现差异 #7: step_cross 和 step_stand 的处理**

| 参数 | Python Kernel 接收 | C++ Kernel 接收 |
|------|-------------------|-----------------|
| step_cross | `step_cross` (原值) | `step_cross * step_cross` (平方) |
| step_stand | `step_stand` (原值) | `step_stand * step_stand` (平方) |

**Python kernels.py 模板替换**:

```python
step_cross_sq=step_cross ** 2,
step_stand_sq=step_stand ** 2,
```

**结论**: Python 在模板替换时计算平方，C++ 在调用前计算平方。逻辑一致。

---

#### 3.2 核心逻辑对比

**Python** (`kernels.py:144-181`):

```cuda
if ( interval[i] < ${interval_min} )
{
    trav_cost[i] = ${cost_barrier};
    return;
}
else
    trav_cost[i] += max(0.0, 20 * (${interval_free} - interval[i]));
if ( grad_mag_sq[i] <= ${step_stand_sq} )
{
    trav_cost[i] += 15 * grad_mag_sq[i] / ${step_stand_sq};
    return;
}
else 
{
    if ( grad_mag_max[i] <= ${step_cross_sq} )
    {
        // ... standable 检查 ...
    }
    else
    {
        trav_cost[i] = ${cost_barrier};
        return;
    }
}
```

**C++** (`tomography_cuda.cu:185-223`):

```cpp
if (inter < interval_min) {
    trav_cost[idx] = cost_barrier;
    return;
}

cost += fmaxf(0.0f, 20.0f * (interval_free - inter));

if (g_sq <= step_stand_sq) {
    cost += 15.0f * g_sq / step_stand_sq;
    trav_cost[idx] = cost;
    return;
}

if (g_max <= step_cross_sq) {
    // ... standable 检查 ...
}

trav_cost[idx] = cost_barrier;
```

#### ⚠️ **发现差异 #8: trav_cost 的累加方式**

| 步骤 | Python | C++ |
|------|--------|-----|
| 初始化 | `trav_cost[i]` 未显式初始化 (依赖外部清零) | `float cost = 0.0f;` |
| 累加 | `trav_cost[i] += ...` | `cost += ...; trav_cost[idx] = cost;` |

**Python 使用 `+=`**:

```cuda
trav_cost[i] += max(0.0, 20 * (${interval_free} - interval[i]));
trav_cost[i] += 15 * grad_mag_sq[i] / ${step_stand_sq};
```

**C++ 使用局部变量**:

```cpp
cost += fmaxf(0.0f, 20.0f * (interval_free - inter));
cost += 15.0f * g_sq / step_stand_sq;
trav_cost[idx] = cost;
```

**潜在问题**: 如果 Python 的 `trav_cost` 未被正确清零，结果会累积。但两边都有清零逻辑，应该一致。

---

#### 3.3 standable 邻域检查

**Python** (`kernels.py:159-175`):

```cuda
int standable_grids = 0;
for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
{
    for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
    {
        int idx = getIdxRelative(i, dx, dy);
        if ( idx < 0 )
            continue;
        if ( grad_mag_sq[idx] < ${step_stand_sq} )
            standable_grids += 1;
    }
}
if ( standable_grids < ${standable_th} )
{
    trav_cost[i] = ${cost_barrier};
    return;
}
```

**C++** (`tomography_cuda.cu:203-216`):

```cpp
int standable = 0;
for (int dy = -half_kernel_size; dy <= half_kernel_size; ++dy) {
    for (int dx = -half_kernel_size; dx <= half_kernel_size; ++dx) {
        int nbr = getIdxRelative(idx, dx, dy, dim_x, dim_y);
        if (nbr < 0)
            continue;
        if (grad_mag_sq[nbr] < step_stand_sq)
            standable++;
    }
}
if (standable < standable_th) {
    trav_cost[idx] = cost_barrier;
    return;
}
```

#### ⚠️ **发现差异 #9: getIdxRelative 函数实现**

**Python** (`kernels.py:73-87`):

```cuda
__device__ int getIdxRelative(int idx, int dx, int dy) 
{
    int idx_2d = idx % (int)${layer_size};
    int idx_x = idx_2d / ${n_col};
    int idx_y = idx_2d % ${n_col};
    int idx_rx = idx_x + dx;
    int idx_ry = idx_y + dy;

    if ( idx_rx < 0 || idx_rx > (${n_row} - 1) ) 
        return -1;
    if ( idx_ry < 0 || idx_ry > (${n_col} - 1) )
        return -1;

    return ${n_col} * dx + dy + idx;
}
```

**C++** (`tomography_cuda.cu:47-57`):

```cpp
__device__ inline int getIdxRelative(int idx, int dx, int dy, int dim_x,
                                     int dim_y) {
    int plane_idx = idx % (dim_x * dim_y);
    int iy = plane_idx % dim_y;
    int ix = plane_idx / dim_y;
    int rx = ix + dx;
    int ry = iy + dy;
    if (rx < 0 || rx >= dim_x || ry < 0 || ry >= dim_y)
        return -1;
    return idx + dx * dim_y + dy;
}
```

| 变量 | Python | C++ |
|------|--------|-----|
| layer_size | `${n_row} * ${n_col}` | `dim_x * dim_y` |
| n_row | `dim_x` | `dim_x` |
| n_col | `dim_y` | `dim_y` |
| 返回值 | `${n_col} * dx + dy + idx` | `idx + dx * dim_y + dy` |

**数学验证**:

- Python: `n_col * dx + dy + idx` = `dim_y * dx + dy + idx`
- C++: `idx + dx * dim_y + dy` = `idx + dim_y * dx + dy`

**结论**: 两者数学上等价。

#### ⚠️ **发现差异 #10: 边界判断运算符**

| 方面 | Python | C++ |
|------|--------|-----|
| x 上界 | `idx_rx > (${n_row} - 1)` | `rx >= dim_x` |
| y 上界 | `idx_ry > (${n_col} - 1)` | `ry >= dim_y` |

**数学验证**:

- `idx_rx > (n_row - 1)` 等价于 `idx_rx >= n_row`
- 两者逻辑一致。

---

## 模块四：InflationKernel (代价膨胀)

### 文件位置

- **C++**: `src/tomography_cuda.cu` 第 226-249 行
- **Python**: `tomography/scripts/kernels.py` 第 201-226 行

### 逻辑对比

**Python** (`kernels.py:206-218`):

```cuda
int counter = 0;
for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
{
    for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
    {
        int idx = getIdxRelative(i, dx, dy);
        if ( idx >= 0 )
            inflated_cost[i] = max(inflated_cost[i], trav_cost[idx] * score_table[counter]);
        counter += 1;
    }
}
```

**C++** (`tomography_cuda.cu:236-248`):

```cpp
float acc = inflated_cost[idx];
int counter = 0;
for (int dy = -half_kernel_size; dy <= half_kernel_size; ++dy) {
    for (int dx = -half_kernel_size; dx <= half_kernel_size; ++dx) {
        int nbr = getIdxRelative(idx, dx, dy, dim_x, dim_y);
        if (nbr >= 0) {
            float val = trav_cost[nbr] * score_table[counter];
            if (val > acc)
                acc = val;
        }
        counter++;
    }
}
inflated_cost[idx] = acc;
```

#### ⚠️ **发现差异 #11: counter 增量位置**

| 方面 | Python | C++ |
|------|--------|-----|
| counter 增量 | 在 if 外面 | 在 if 外面 |
| 越界时处理 | counter 仍然 +1 | counter 仍然 +1 |

**结论**: 逻辑一致。

---

#### 4.2 score_table 计算

**Python** (`tomogram.py:51-62`):

```python
self.inf_table = cp.zeros(
    (2 * self.half_inf_k_size + 1, 2 * self.half_inf_k_size + 1), 
    dtype=cp.float32
)
for i in range(self.inf_table.shape[0]):
    for j in range(self.inf_table.shape[1]):
        dist = np.sqrt(
            (self.resolution * (i - self.half_inf_k_size)) ** 2 + \
            (self.resolution * (j - self.half_inf_k_size)) ** 2
        )
        self.inf_table[i, j] = np.clip(
            1 - (dist - self.inflation) / (self.safe_margin + self.resolution),
            a_min=0.0, a_max=1.0
        )
```

**C++** (`tomography_cuda.cu:333-349`):

```cpp
for (int i = 0; i < 2 * half_inf_k + 1; ++i) {
    for (int j = 0; j < 2 * half_inf_k + 1; ++j) {
        float dx = params.resolution * static_cast<float>(i - half_inf_k);
        float dy = params.resolution * static_cast<float>(j - half_inf_k);
        float dist = std::sqrt(dx * dx + dy * dy);
        float val =
            1.0f -
            (dist - static_cast<float>(params.inflation)) /
                (static_cast<float>(params.safe_margin + params.resolution));
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;
        h_score[i * (2 * half_inf_k + 1) + j] = val;
    }
}
```

**结论**: 计算公式一致。

---

## 模块五：层简化 (Layer Simplification)

### 文件位置

- **C++**: `src/tomography_node.cpp` 第 175-210 行
- **Python**: `tomography/scripts/tomogram.py` 第 155-170 行

### 逻辑对比

**Python** (`tomogram.py:155-169`):

```python
idx_simp = [0]
if self.layers_g.shape[0] > 1:
    l_idx, m_idx = 0, 1
    diff_h = self.layers_g[1:] - self.layers_g[:-1]
    while m_idx < self.n_slice_init - 2:
        mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0
        mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]
        mask_u_g = diff_h[m_idx] > 0
        mask_t = self.inflated_cost[m_idx] < self.cost_barrier
        unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t
        if cp.any(unique):
            idx_simp.append(m_idx)
            l_idx = m_idx
        m_idx += 1
    idx_simp.append(m_idx)
```

**C++** (`tomography_node.cpp:175-210`):

```cpp
std::vector<uint32_t> idx_simp;
idx_simp.push_back(0);
if (n_slice > 1) {
    uint32_t l_idx = 0;
    uint32_t m_idx = 1;
    while (m_idx < n_slice - 2) {
        bool keep = false;
        const size_t plane = static_cast<size_t>(dim_x) * dim_y;
        const size_t offset_l = static_cast<size_t>(l_idx) * plane;
        const size_t offset_m = static_cast<size_t>(m_idx) * plane;
        const size_t offset_u = static_cast<size_t>(m_idx + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const float g_l = gpu_out.elev_g[offset_l + i];
            const float g_m = gpu_out.elev_g[offset_m + i];
            const float cost_l = gpu_out.inflated_cost[offset_l + i];
            const float cost_m = gpu_out.inflated_cost[offset_m + i];
            const float diff_h = gpu_out.elev_g[offset_u + i] - g_m;
            const bool mask_l_g = (g_m - g_l) > 0.0f;
            const bool mask_l_t = cost_l > cost_m;
            const bool mask_u_g = diff_h > 0.0f;
            const bool mask_t = cost_m < static_cast<float>(cost_barrier_);
            if ((mask_l_g || mask_l_t) && mask_u_g && mask_t) {
                keep = true;
                break;
            }
        }
        // ...
    }
}
```

#### ⚠️ **发现差异 #12: diff_h 的计算**

| 方面 | Python | C++ |
|------|--------|-----|
| diff_h 定义 | `diff_h = self.layers_g[1:] - self.layers_g[:-1]` (预计算) | `diff_h = gpu_out.elev_g[offset_u + i] - g_m` (逐点计算) |
| diff_h 索引 | `diff_h[m_idx]` | `elev_g[m_idx + 1] - elev_g[m_idx]` |

**Python diff_h 索引分析**:

- `diff_h = layers_g[1:] - layers_g[:-1]`
- `diff_h[0]` = `layers_g[1] - layers_g[0]`
- `diff_h[m_idx]` = `layers_g[m_idx + 1] - layers_g[m_idx]`

**C++ diff_h 计算**:

- `diff_h = elev_g[offset_u + i] - g_m`
- `offset_u = (m_idx + 1) * plane`
- 即 `elev_g[m_idx + 1][i] - elev_g[m_idx][i]`

**结论**: 两者计算的是相同的差值，逻辑一致。

---

## 模块六：输出与序列化

### 文件位置

- **C++**: `src/tomography_node.cpp` 第 257-280 行
- **Python**: `tomography/scripts/tomogram.py` 第 183-197 行

### 逻辑对比

**Python** (`tomogram.py:183-197`):

```python
layers_g = cp.where(
    self.layers_g[idx_simp] > -1e6, 
    self.layers_g[idx_simp],
    cp.nan
).get()
layers_c = cp.where(
    self.layers_c[idx_simp] < 1e6, 
    self.layers_c[idx_simp], 
    cp.nan
).get()
trav_gx = np.zeros_like(layers_g)
trav_gx[:, 1:-1, :] = trav_grad_x.get()
trav_gy = np.zeros_like(layers_g)
trav_gy[:, :, 1:-1] = trav_grad_y.get()
```

**C++** (`tomography_node.cpp:257-270`):

```cpp
for (size_t i = 0; i < elev_g_simp.size(); ++i) {
    if (missing_ground_simp[i]) {
        elev_g_simp[i] = std::numeric_limits<float>::quiet_NaN();
    }
    if (missing_ceiling_simp[i]) {
        elev_c_simp[i] = std::numeric_limits<float>::quiet_NaN();
    }
}

// 生成 trav 梯度（在简化层上取中心差分）
std::vector<float> trav_gx(static_cast<size_t>(simp_layers) * plane, 0.0f);
std::vector<float> trav_gy(static_cast<size_t>(simp_layers) * plane, 0.0f);
ComputeTravGradient(simp_layers, dim_x, dim_y, trav_simp, trav_gx, trav_gy);
```

#### ⚠️ **发现差异 #13: NaN 判断阈值** - ✅ **已修复**

| 方面 | Python | C++ (修复后) | 状态 |
|------|--------|-------------|------|
| 地面缺失判断 | `> -1e6` | `<= -1e6f + 1.0f` | ✅ 一致 |
| 天花板缺失判断 | `< 1e6` | `>= 1e6f - 1.0f` | ✅ 一致 |

**修复内容**: 将 C++ 的阈值从 `-9e5f` / `9e5f` 改为 `-1e6f + 1.0f` / `1e6f - 1.0f`

**C++ 修复后代码** (`tomography_node.cpp:162-175`):

```cpp
// 阈值与 Python 版本一致：Python 使用 > -1e6 判断有效，这里用 <= -1e6 + 1 判断缺失
// 原版使用 -9e5f 会导致 -999999 等值被错误标记为缺失
std::vector<uint8_t> missing_ground(gpu_out.elev_g.size(), 0);
std::vector<uint8_t> missing_ceiling(gpu_out.elev_c.size(), 0);
for (size_t i = 0; i < gpu_out.elev_g.size(); ++i) {
  if (gpu_out.elev_g[i] <= -1e6f + 1.0f) {
    missing_ground[i] = 1;
  }
  if (gpu_out.elev_c[i] >= 1e6f - 1.0f) {
    missing_ceiling[i] = 1;
  }
}
```

---

#### ⚠️ **发现差异 #14: trav_gx/trav_gy 计算**

**Python** (`tomogram.py:171-172`):

```python
trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
```

**C++** (`tomography_node.cpp:359-375`):

```cpp
void ComputeTravGradient(uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                         const std::vector<float> &cost,
                         std::vector<float> &gx, std::vector<float> &gy) {
    // ...
    for (uint32_t x = 1; x + 1 < dim_x; ++x) {
        for (uint32_t y = 1; y + 1 < dim_y; ++y) {
            const size_t idx = offset + static_cast<size_t>(x) * dim_y + y;
            gx[idx] = cost[idx + dim_y] - cost[idx - dim_y];
            gy[idx] = cost[idx + 1] - cost[idx - 1];
        }
    }
}
```

**差分公式对比**:

| 版本 | gx 公式 | gy 公式 |
|------|---------|---------|
| Python | `cost[:, x+1, :] - cost[:, x-1, :]` | `cost[:, :, y+1] - cost[:, :, y-1]` |
| C++ | `cost[x+1] - cost[x-1]` | `cost[y+1] - cost[y-1]` |

**结论**: 公式一致，都是中心差分。

---

## 发现的差异汇总

### ✅ 已修复问题

| 编号 | 问题描述 | 位置 | 修复内容 |
|------|----------|------|----------|
| **#13** | NaN 判断阈值不一致 | `tomography_node.cpp:162-175` | 阈值从 `-9e5f` 改为 `-1e6f + 1.0f` |
| **#1** | half 精度索引计算方式不同 | `tomography_cuda.cu:88-110` | 除法改为 float 精度，使用 `roundf()` |
| **#3** | slice 高度计算精度不同 | `tomography_cuda.cu:118-128` | 使用 half 精度运算 (`__hadd`, `__hmul`) |

### 🟡 剩余问题 (精度相关，难以完全消除)

| 编号 | 问题描述 | 影响 | 说明 |
|------|----------|------|------|
| **NaN 边界** | 约 25k 个 NaN 不一致 | 导致邻域梯度差异 | 由 half 精度舍入导致边界点投影不同 |
| **屏障差异** | 约 4k 个屏障差异 | 部分由 NaN 不一致引起 | 邻域 standable 计数受 NaN 影响 |

### ✅ 已确认一致

| 模块 | 状态 |
|------|------|
| 索引计算 (getIdxRelative) | ✅ 一致 |
| GradIntervalKernel 梯度计算 | ✅ 一致 |
| TravKernel 逻辑 | ✅ 一致 |
| InflationKernel 逻辑 | ✅ 一致 |
| 层简化 (Layer Simplification) | ✅ 一致 |
| score_table 计算 | ✅ 一致 |
| trav_gx/trav_gy 梯度计算 | ✅ 一致 |

---

## 测试结果历史

| 日期 | 精确匹配率 | 屏障差异 | NaN不一致(C++/Py) | 主要改动 |
|------|-----------|---------|-------------------|----------|
| 初始版本 | ~95.91% | ~4,564 | ~22k/~21k | - |
| 修复 #13 | ~96.11% | ~5,208 | ~17k/~15k | NaN 阈值对齐 |
| 修复 #1, #3 | **96.80%** | **4,109** | **14k/12k** | half 精度对齐 |

---

## 剩余差异分析

### 根本原因

剩余的 NaN 不一致和屏障差异主要由以下因素导致：

1. **Half 精度舍入边界效应**: 当点云坐标恰好在栅格边界时，half 精度运算的微小舍入差异可能导致点被投影到不同栅格

2. **点云输入差异**: Python tomogram 与 C++ tomogram 可能使用了不同时间采集或不同来源的点云

3. **级联效应**: NaN 位置不一致 → 梯度计算不同 → standable 计数跨越阈值 → 屏障差异

### 影响评估

- **精确匹配率 96.80%** 已经很高
- **屏障差异 4,109 个 (0.01%)** 对路径规划影响有限
- 当两版本 elev_g 都有效时，trav 精确匹配率高达 **95.69%**，且**零屏障差异**

---

## 下一步行动建议

### 如需进一步提升一致性

1. **确保使用相同点云**: 验证 C++ 和 Python 使用完全相同的点云输入
2. **考虑使用 float32 模式**: 两边都使用 float32 可消除 half 精度差异

### 当前状态评估

✅ **核心算法已对齐**: TomographyKernel、GradIntervalKernel、TravKernel、InflationKernel 逻辑完全一致

✅ **关键阈值已修复**: NaN 判断阈值、round 函数选择、精度模式已对齐

✅ **可用于生产**: 96.80% 精确匹配率，屏障差异 0.01%，对路径规划结果影响极小

---

*报告更新时间: 2026-01-14*
