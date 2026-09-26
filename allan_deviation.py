"""固定速率单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集）。

本模块只实现题目契约内的数学子集：

* :func:`analyze` 用累积和计算重叠 Allan 方差（overlapping Allan variance）；
* 对差分样本数 >= 20 且方差严格为正的 (m, sigma) 点，在 log(tau)-log(sigma)
  平面上做连续（相邻区间无重叠、无缺口）分段直线最小二乘拟合；
* 分段数 1..3，每段至少 3 个点，分段罚项为模块常量 :data:`SEGMENT_PENALTY`，
  全局最优由动态规划穷举所有合法切分得到。

注意：拟合斜率只是对该批离线数据 log-log 形态的经验描述，不是任何临床结论或
硬件指标保证；本模块也不宣称完整兼容任何行业标准。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "analyze",
    "fit_temperature_polynomial",
    "analyze_temperature_compensation",
    "AllanPoint",
    "AllanResult",
    "Breakpoint",
    "Segment",
    "FitResult",
    "AllanError",
    "AllanTypeError",
    "AllanValueError",
    "FitInfeasibleError",
    "SearchBudgetExhaustedError",
    "TemperatureFitError",
    "TemperatureDesignError",
    "TemperatureConfoundingError",
    "TemperatureFit",
    "TemperatureCompensationResult",
    "MAX_SAMPLES",
    "MAX_CLUSTER_SIZES",
    "MIN_DIFFERENCES",
    "MAX_SEGMENTS",
    "MIN_SEGMENT_POINTS",
    "SEGMENT_PENALTY",
]

# ---------------------------------------------------------------------------
# 公开常量（契约边界与拟合参数）
# ---------------------------------------------------------------------------

MAX_SAMPLES = 8192
"""samples 的最大长度（接口边界）。"""

MAX_CLUSTER_SIZES = 8192
"""cluster_sizes 的最大长度（接口边界）。"""

MIN_DIFFERENCES = 20
"""只有相邻块均值差分样本数 >= 该值的 m 才参与 log-log 拟合。"""

MAX_SEGMENTS = 3
"""分段直线最多段数。"""

MIN_SEGMENT_POINTS = 3
"""每段至少包含的拟合点数。"""

SEGMENT_PENALTY = 1.0
"""每增加一段在总 SSE 上追加的固定罚项（单位：log(sigma) 残差平方）。

与数据规模无关的固定常量，使 "原始 SSE 改善量 > 罚项" 才接受更碎的切分。
"""

# 动态规划内层求值次数的硬上限。DP 是精确穷举，合法输入（K <= 8192、
# 最多 3 段）的求值次数有确定上界，该预算取该上界之上，因此对合法输入
# 不可达；保留它是为了在违反公开边界时给出与“证明无解”不同的错误类型。
_SEARCH_BUDGET = 8 * MAX_CLUSTER_SIZES * MAX_CLUSTER_SIZES


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllanPoint:
    """单个聚类大小 m 的 Allan 统计结果。

    属性:
        m: 聚类大小（样本数）。
        tau: 聚类时间 m * dt，单位秒。
        variance: 重叠 Allan 方差；差分样本数 < 2 时为 None（无法估计）。
        adev: Allan 标准差 ``sqrt(variance)``；variance 为 None 时同为 None。
        count: 实际参与计算的相邻等长块均值差分样本数，即 ``max(0, N - 2*m + 1)``；
            m 过大（2*m > N）时为 0。
        eligible_for_fit: count >= 20 且 variance > 0。
    """

    m: int
    tau: float
    variance: float | None
    adev: float | None
    count: int
    eligible_for_fit: bool


@dataclass(frozen=True)
class Breakpoint:
    """两段之间的断点，定义为右侧段第一个拟合点所在位置。"""

    m: int
    tau: float


@dataclass(frozen=True)
class Segment:
    """一段 log(tau)-log(sigma) 最小二乘直线 ``y = slope*x + intercept``。"""

    slope: float
    intercept: float
    m_start: int
    m_end: int
    tau_start: float
    tau_end: float
    n_points: int


@dataclass(frozen=True)
class FitResult:
    """分段直线拟合结果。

    属性:
        segments: 各段（按 tau 升序）。
        breakpoints: 内部断点（段数为 1 时为空）。
        slopes / intercepts: 与 segments 对齐的斜率、截距。
        sse: 全部拟合点的残差平方和。
        rmse: ``sqrt(sse / 拟合点数)``。
        residuals: 与 fitted_ms 对齐的逐点残差（观测 log(sigma) 减模型值）。
        fitted_ms: 参与拟合的 m，已去重、升序排列。
        segment_count: 实际选中的段数。
        objective: ``sse + (segment_count - 1) * SEGMENT_PENALTY``。
    """

    segments: tuple[Segment, ...]
    breakpoints: tuple[Breakpoint, ...]
    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]
    sse: float
    rmse: float
    residuals: tuple[float, ...]
    fitted_ms: tuple[int, ...]
    segment_count: int
    objective: float


@dataclass(frozen=True)
class AllanResult:
    """analyze 的完整返回值。"""

    points: tuple[AllanPoint, ...]
    fit: FitResult
    dt: float
    n_samples: int


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class AllanError(Exception):
    """本模块所有公开错误的基类。"""


class AllanTypeError(AllanError, TypeError):
    """参数类型非法（含对整数参数传入 bool）。"""


class AllanValueError(AllanError, ValueError):
    """参数取值非法（含 NaN/Infinity、越界、空序列等），信息可定位到参数。"""


@dataclass(frozen=True)
class TemperatureFit:
    """训练区间上的二阶温度多项式离线拟合结果。

    模型使用中心化且缩放后的温度 ``x=(temperature-center)/scale``：
    ``gyro = c0 + c1*x + c2*x**2 + residual``。
    """

    temperature_center: float
    temperature_scale: float
    coefficients: tuple[float, float, float]
    residuals: tuple[float, ...]
    predictions: tuple[float, ...]
    compensated: tuple[float, ...]
    train_start: int
    train_end: int


@dataclass(frozen=True)
class TemperatureCompensationResult:
    """原数据与温度补偿数据在训练/未来留出区间上的 Allan 对照结果。"""

    temperature_fit: TemperatureFit
    compensated: tuple[float, ...]
    timestamps: tuple[float, ...]
    train_start: int
    train_end: int
    holdout_start: int
    holdout_end: int
    original_training: AllanResult
    compensated_training: AllanResult
    original_holdout: AllanResult
    compensated_holdout: AllanResult
    dt: float
    n_samples: int


class TemperatureFitError(AllanValueError):
    """温度补偿输入不满足拟合契约的基类。"""


class TemperatureDesignError(TemperatureFitError):
    """温度设计矩阵列不足、列相关或训练样本不足。"""


class TemperatureConfoundingError(TemperatureFitError):
    """温度与时间仿射完全混淆，温度系数不能作因果解释。"""


class FitInfeasibleError(AllanError):
    """已证明不存在满足约束的分段方案（区别于搜索预算耗尽）。

    属性:
        reason: 人类可读的无解原因。
        points: 已算出的 AllanPoint 序列（纯函数无部分写入，这里仅作附带信息）。
    """

    def __init__(self, reason: str, points: Sequence[AllanPoint] = ()):
        super().__init__(reason)
        self.reason = reason
        self.points: tuple[AllanPoint, ...] = tuple(points)


class SearchBudgetExhaustedError(AllanError):
    """搜索预算耗尽：无法证明最优，但也没有证明无解（与 FitInfeasibleError 分开）。

    精确动态规划在公开输入边界内不会抛出该异常；它只可能在绕过公开边界
    （内部调用时给出超大 K）时出现。
    """

    def __init__(self, used: int, budget: int):
        super().__init__(f"search budget exhausted: used={used}, budget={budget}")
        self.used = used
        self.budget = budget


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _is_real(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate(samples: object, dt: object, cluster_sizes: object):
    if not isinstance(samples, Sequence) or isinstance(
        samples, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "samples: expected a finite sequence (list/tuple/...) of real numbers"
        )
    if not isinstance(cluster_sizes, Sequence) or isinstance(
        cluster_sizes, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "cluster_sizes: expected a finite sequence (list/tuple/range) of ints"
        )

    n = len(samples)
    if n == 0:
        raise AllanValueError("samples: must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise AllanValueError(
            f"samples: length {n} exceeds interface bound {MAX_SAMPLES}"
        )

    if not _is_real(dt):
        raise AllanTypeError("dt: expected int/float (bool is not accepted)")
    try:
        dt_f = float(dt)
    except OverflowError:
        raise AllanValueError(f"dt: {dt!r} cannot be represented as finite float") from None
    if not math.isfinite(dt_f):
        raise AllanValueError("dt: must be finite (NaN/Infinity rejected)")
    if dt_f <= 0.0:
        raise AllanValueError(f"dt: must be > 0, got {dt!r}")

    k_sizes = len(cluster_sizes)
    if k_sizes == 0:
        raise AllanValueError("cluster_sizes: must contain at least 1 cluster size")
    if k_sizes > MAX_CLUSTER_SIZES:
        raise AllanValueError(
            f"cluster_sizes: length {k_sizes} exceeds interface bound "
            f"{MAX_CLUSTER_SIZES}"
        )

    values: list[float] = [0.0] * n
    for i, v in enumerate(samples):
        if not _is_real(v):
            raise AllanTypeError(
                f"samples[{i}]: expected int/finite-float (bool/str/complex rejected)"
            )
        try:
            fv = float(v)
        except OverflowError:
            raise AllanValueError(
                f"samples[{i}]: {v!r} cannot be represented as finite float"
            ) from None
        if not math.isfinite(fv):
            raise AllanValueError(f"samples[{i}]: value must be finite (NaN/Infinity rejected)")
        values[i] = fv

    sizes: list[int] = [0] * k_sizes
    for i, m in enumerate(cluster_sizes):
        if isinstance(m, bool) or not isinstance(m, int):
            raise AllanTypeError(
                f"cluster_sizes[{i}]: cluster size must be an int (bool/float rejected)"
            )
        if m < 1:
            raise AllanValueError(
                f"cluster_sizes[{i}]: cluster size must be >= 1, got {m}"
            )
        # tau=m*dt 必须可表示为有限 float（超大 int m 在此被可定位地拒绝）。
        try:
            tau_test = float(m) * dt_f
        except OverflowError:
            tau_test = math.inf
        if not math.isfinite(tau_test):
            raise AllanValueError(
                f"cluster_sizes[{i}]: m*dt overflows finite float (m={m}, dt={dt_f})"
            )
        sizes[i] = m

    return values, dt_f, sizes


# ---------------------------------------------------------------------------
# 重叠 Allan 方差（累积和实现）
# ---------------------------------------------------------------------------


def _allan_points(values: Sequence[float], dt: float, sizes: Sequence[int]):
    n = len(values)

    # 先去除恒定偏置：相邻块均值之差对整体加常数数学不变，去偏置可避免大基线
    # （如 1e9 + 小噪声）在前缀和三项相减时灾难性抵消。均值用 fsum 精确求和。
    try:
        mean = math.fsum(values) / n
    except OverflowError:
        raise AllanValueError(
            "sample magnitudes overflow even the exact mean computation; "
            "inputs too large for the float64 computation"
        ) from None
    if not math.isfinite(mean):
        raise AllanValueError("sample mean is not finite; inputs out of float64 range")
    centered = [v - mean for v in values]

    # 样本前缀累积和 C[k] = sum(centered[0:k])，float64 顺序累加。
    cum = [0.0] * (n + 1)
    acc = 0.0
    for i, v in enumerate(centered):
        acc += v
        cum[i + 1] = acc

    cm = cum

    def compute(m: int) -> AllanPoint:
        count = n - 2 * m + 1
        tau = m * dt
        if count < 2:
            # 实际差分样本数不会为负：2m>N 时为 0；count=1 时无法做无偏估计。
            return AllanPoint(m, tau, None, None, max(0, count), False)
        # 相邻等长块均值之差：
        #   d_k = mean(w_{k+m}) - mean(w_k)
        #       = (C[k+2m] - 2*C[k+m] + C[k]) / m
        # 平方和用 math.fsum 求和，降低大块数时的累加舍入。
        try:
            q = math.fsum(
                ((cm[k + 2 * m] - 2.0 * cm[k + m] + cm[k]) / m) ** 2
                for k in range(count)
            )
        except OverflowError:
            q = math.inf
        variance = q / (2.0 * (count - 1))
        if not math.isfinite(variance):
            raise AllanValueError(
                f"allan variance overflowed finite float for m={m}; "
                "input magnitude too large for the computation"
            )
        adev = math.sqrt(variance)
        eligible = count >= MIN_DIFFERENCES and variance > 0.0
        return AllanPoint(m, tau, variance, adev, count, eligible)

    # 重复 m 只计算一次；输出顺序与出现次数仍与输入 cluster_sizes 完全一致。
    cache: dict[int, AllanPoint] = {}
    points: list[AllanPoint] = []
    for m in sizes:
        pt = cache.get(m)
        if pt is None:
            pt = compute(m)
            cache[m] = pt
        points.append(pt)
    return points


# ---------------------------------------------------------------------------
# log-log 连续分段直线拟合（精确动态规划）
# ---------------------------------------------------------------------------


def _ols_sums(px, py, pxx, pxy, pyy, i: int, j: int):
    """区间 [i, j] 闭区间上 OLS 直线的 (slope, intercept, sse)，量取自前缀和。"""
    nn = j - i + 1
    sx = px[j + 1] - px[i]
    sy = py[j + 1] - py[i]
    sxx = pxx[j + 1] - pxx[i] - sx * sx / nn
    sxy = pxy[j + 1] - pxy[i] - sx * sy / nn
    syy = pyy[j + 1] - pyy[i] - sy * sy / nn
    if sxx <= 0.0:
        # 去重升序后的 x 互不相同，理论上不会发生；保留防护以免除零。
        raise FitInfeasibleError("degenerate log-log segment (zero x variance)")
    slope = sxy / sxx
    intercept = (sy - slope * sx) / nn
    # OLS 的 SSE 数学上必非负；负值只能是舍入残差，截断到 0（与 DP 内层一致）。
    sse = syy - sxy * sxy / sxx
    if sse < 0.0:
        sse = 0.0
    return slope, intercept, sse


def _fit_piecewise(points: Sequence[AllanPoint]):
    eligible = [p for p in points if p.eligible_for_fit]
    # 去重（保留首次出现），按 m 升序。
    seen: dict[int, AllanPoint] = {}
    for p in eligible:
        if p.m not in seen:
            seen[p.m] = p
    ordered = sorted(seen.values(), key=lambda p: p.m)
    k = len(ordered)

    if k < MIN_SEGMENT_POINTS:
        raise FitInfeasibleError(
            f"only {k} log-log point(s) with >={MIN_DIFFERENCES} differences and "
            f"positive variance; need at least {MIN_SEGMENT_POINTS}",
            points,
        )

    xs = [math.log(p.tau) for p in ordered]
    ys = [0.5 * math.log(p.variance) for p in ordered]

    px = [0.0] * (k + 1)
    py = [0.0] * (k + 1)
    pxx = [0.0] * (k + 1)
    pxy = [0.0] * (k + 1)
    pyy = [0.0] * (k + 1)
    for i in range(k):
        xi = xs[i]
        yi = ys[i]
        px[i + 1] = px[i] + xi
        py[i + 1] = py[i] + yi
        pxx[i + 1] = pxx[i] + xi * xi
        pxy[i + 1] = pxy[i] + xi * yi
        pyy[i + 1] = pyy[i] + yi * yi

    def sse(i: int, j: int) -> float:
        nn = j - i + 1
        sx = px[j + 1] - px[i]
        sy = py[j + 1] - py[i]
        sxx = pxx[j + 1] - pxx[i] - sx * sx / nn
        sxy = pxy[j + 1] - pxy[i] - sx * sy / nn
        syy = pyy[j + 1] - pyy[i] - sy * sy / nn
        val = syy - sxy * sxy / sxx
        return 0.0 if val < 0.0 else val

    pmax = min(MAX_SEGMENTS, k // MIN_SEGMENT_POINTS)

    # dp[seg-1][b] = 覆盖点 0..b、恰为 seg 段时的最小原始 SSE。
    # back[seg-1][b] 记录最后一段的起点下标。
    inf = math.inf
    dp0 = [inf] * k
    back0 = [-1] * k
    used = 0
    for b in range(MIN_SEGMENT_POINTS - 1, k):
        dp0[b] = sse(0, b)
        back0[b] = 0
    dp_prev = dp0
    back_prev = back0
    layers = [(dp0, back0)]

    for seg in range(2, pmax + 1):
        dp_cur = [inf] * k
        back_cur = [-1] * k
        min_prev = MIN_SEGMENT_POINTS * (seg - 1)
        for b in range(MIN_SEGMENT_POINTS * seg - 1, k):
            best = inf
            best_a = -1
            # 最后一段为 [a, b]，前面 seg-1 段覆盖 0..a-1。
            for a in range(min_prev, b - MIN_SEGMENT_POINTS + 2):
                val = dp_prev[a - 1] + sse(a, b)
                used += 1
                if val < best:
                    best = val
                    best_a = a
            dp_cur[b] = best
            back_cur[b] = best_a
            if used > _SEARCH_BUDGET:  # 公开边界内不可达，见常量说明
                raise SearchBudgetExhaustedError(used, _SEARCH_BUDGET)
        layers.append((dp_cur, back_cur))
        dp_prev = dp_cur
        back_prev = back_cur

    # 选段数：目标 = SSE + (p-1)*罚项；严格小于才换更碎方案，故并列时段数更少者胜。
    best_p = 1
    best_obj = layers[0][0][k - 1]
    for seg in range(2, pmax + 1):
        obj = layers[seg - 1][0][k - 1] + (seg - 1) * SEGMENT_PENALTY
        if obj < best_obj:
            best_obj = obj
            best_p = seg

    # 回溯切分。
    bounds: list[int] = []
    b = k - 1
    for seg in range(best_p, 1, -1):
        a = layers[seg - 1][1][b]
        bounds.append(a)
        b = a - 1
    bounds.reverse()
    starts = [0, *bounds]
    ends = [*[a - 1 for a in bounds], k - 1]

    segments: list[Segment] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    residuals: list[float] = [0.0] * k
    total_sse = 0.0
    for i0, i1 in zip(starts, ends):
        slope, intercept, seg_sse = _ols_sums(
            px, py, pxx, pxy, pyy, i0, i1
        )
        total_sse += seg_sse
        p0 = ordered[i0]
        p1 = ordered[i1]
        segments.append(
            Segment(
                slope,
                intercept,
                p0.m,
                p1.m,
                p0.tau,
                p1.tau,
                i1 - i0 + 1,
            )
        )
        slopes.append(slope)
        intercepts.append(intercept)
        for idx in range(i0, i1 + 1):
            residuals[idx] = ys[idx] - (slope * xs[idx] + intercept)

    breakpoints = tuple(
        Breakpoint(ordered[a].m, ordered[a].tau) for a in bounds
    )

    return FitResult(
        segments=tuple(segments),
        breakpoints=breakpoints,
        slopes=tuple(slopes),
        intercepts=tuple(intercepts),
        sse=total_sse,
        rmse=math.sqrt(total_sse / k),
        residuals=tuple(residuals),
        fitted_ms=tuple(p.m for p in ordered),
        segment_count=best_p,
        objective=best_obj,
    )


# ---------------------------------------------------------------------------
# 二阶温度多项式补偿（QR 求解，离线子集）
# ---------------------------------------------------------------------------

_QR_RANK_TOLERANCE = 1e-12


def _validate_real_sequence(values: object, name: str, n: int | None = None) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(f"{name}: expected a finite sequence of real numbers")
    if n is not None and len(values) != n:
        raise AllanValueError(
            f"{name}: length {len(values)} does not align with gyroscope length {n}"
        )
    result: list[float] = []
    for i, value in enumerate(values):
        if not _is_real(value):
            raise AllanTypeError(
                f"{name}[{i}]: expected int/float (bool/str/complex rejected)"
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise AllanValueError(f"{name}[{i}]: value must be finite")
        result.append(fvalue)
    return result


def _validate_index_range(value: object, name: str, n: int) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise AllanTypeError(f"{name}: expected a [start, end] pair")
    start, end = value
    if isinstance(start, bool) or not isinstance(start, int):
        raise AllanTypeError(f"{name}[0]: start must be an int (bool rejected)")
    if isinstance(end, bool) or not isinstance(end, int):
        raise AllanTypeError(f"{name}[1]: end must be an int (bool rejected)")
    if not (0 <= start < end <= n):
        raise AllanValueError(
            f"{name}: expected 0 <= start < end <= {n}, got [{start}, {end})"
        )
    return start, end


def _qr_decompose(matrix: list[list[float]]):
    """返回方阵 R（就地 Householder 变换后的上三角）和 Householder 向量。

    调用方随后使用同一组变换作用于右端项，因此不保存/不假设正规方程。
    """
    rows = len(matrix)
    cols = len(matrix[0])
    work = [row[:] for row in matrix]
    vectors: list[tuple[int, list[float]]] = []
    diagonals: list[float] = []
    for col in range(cols):
        vector = [0.0] * rows
        for row in range(col, rows):
            vector[row] = work[row][col]
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm == 0.0:
            diagonals.append(0.0)
            vectors.append((col, vector))
            continue
        sign = -1.0 if vector[col] < 0.0 else 1.0
        vector[col] += sign * norm
        vnorm_sq = math.fsum(v * v for v in vector)
        if vnorm_sq > 0.0:
            for jcol in range(col, cols):
                dot = math.fsum(vector[k] * work[k][jcol] for k in range(col, rows))
                factor = 2.0 * dot / vnorm_sq
                for row in range(col, rows):
                    work[row][jcol] -= factor * vector[row]
        diagonals.append(work[col][col])
        vectors.append((col, vector))
    return work, vectors, diagonals


def _apply_q_transpose(vectors: Sequence[tuple[int, list[float]]], rhs: list[float]) -> list[float]:
    y = rhs[:]
    rows = len(rhs)
    for col, vector in vectors:
        vnorm_sq = math.fsum(v * v for v in vector)
        if vnorm_sq == 0.0:
            continue
        dot = math.fsum(vector[k] * y[k] for k in range(col, rows))
        factor = 2.0 * dot / vnorm_sq
        for row in range(col, rows):
            y[row] -= factor * vector[row]
    return y


def _back_substitute(r: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    cols = len(r[0])
    coefficients = [0.0] * cols
    for i in range(cols - 1, -1, -1):
        tail = math.fsum(r[i][j] * coefficients[j] for j in range(i + 1, cols))
        coefficients[i] = (y[i] - tail) / r[i][i]
    return coefficients


def _qr_solve(design: Sequence[Sequence[float]], rhs: Sequence[float],
              rank_scale: float) -> list[float]:
    r, vectors, diagonals = _qr_decompose([list(row) for row in design])
    threshold = _QR_RANK_TOLERANCE * max(1.0, rank_scale)
    for col, diagonal in enumerate(diagonals):
        if abs(diagonal) <= threshold:
            raise TemperatureDesignError(
                f"temperature design matrix is rank-deficient at column {col}; "
                "need independent constant, first-order and second-order temperature terms"
            )
    y = _apply_q_transpose(vectors, list(rhs))
    return _back_substitute(r, y)


def _temperature_design(value: float) -> tuple[float, float, float]:
    return 1.0, value, value * value


def _reject_temperature_time_confounding(train_t: Sequence[float],
                                         train_x: Sequence[float]) -> None:
    design = [[1.0, x] for x in train_x]
    r, vectors, diagonals = _qr_decompose(design)
    if abs(diagonals[1]) <= _QR_RANK_TOLERANCE:
        return
    y = _apply_q_transpose(vectors, list(train_t))
    coefficients = _back_substitute(r, y)
    time_span = max(train_t) - min(train_t)
    residual_scale = time_span if time_span > 0.0 else 1.0
    max_residual = max(
        abs(train_t[i] - (coefficients[0] + coefficients[1] * train_x[i]))
        for i in range(len(train_t))
    )
    if max_residual <= _QR_RANK_TOLERANCE * residual_scale:
        raise TemperatureConfoundingError(
            "temperature is exactly confounded with time in the training interval; "
            "temperature coefficients are not identifiable independently of time drift"
        )


def fit_temperature_polynomial(timestamps, gyro, temperatures, train_range) -> TemperatureFit:
    """仅用训练区间拟合 ``c0+c1*x+c2*x**2``，并补偿整段陀螺序列。

    半开区间 ``train_range=[start,end)``。时间戳只按样本位置逐点对齐；长度不一致
    或非严格递增均拒绝，本函数不做任何插值。
    """
    gyro_values = _validate_real_sequence(gyro, "gyro")
    n = len(gyro_values)
    if n == 0:
        raise AllanValueError("gyro: must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise AllanValueError(f"gyro: length {n} exceeds interface bound {MAX_SAMPLES}")
    timestamp_values = _validate_real_sequence(timestamps, "timestamps", n)
    temperature_values = _validate_real_sequence(temperatures, "temperatures", n)
    for i in range(1, n):
        if not timestamp_values[i] > timestamp_values[i - 1]:
            raise AllanValueError(
                f"timestamps[{i}]: timestamps must be strictly increasing; no interpolation is performed"
            )
    start, end = _validate_index_range(train_range, "train_range", n)
    count = end - start
    if count < 3:
        raise TemperatureDesignError(
            f"train_range: need at least 3 rows for constant/first/second terms, got {count}"
        )

    train_temperature = temperature_values[start:end]
    tmin, tmax = min(train_temperature), max(train_temperature)
    temperature_center = (tmin + tmax) / 2.0
    temperature_scale = (tmax - tmin) / 2.0
    if temperature_scale <= 0.0:
        raise TemperatureDesignError(
            "temperatures: training temperature is constant; first/second-order columns are unavailable"
        )
    train_x = [(v - temperature_center) / temperature_scale for v in train_temperature]
    train_t = timestamp_values[start:end]
    _reject_temperature_time_confounding(train_t, train_x)

    design = [_temperature_design(x) for x in train_x]
    rhs = gyro_values[start:end]
    coefficients = _qr_solve(design, rhs, 1.0)
    predictions: list[float] = []
    compensated: list[float] = []
    for temperature, observation in zip(temperature_values, gyro_values):
        x = (temperature - temperature_center) / temperature_scale
        prediction = coefficients[0] + coefficients[1] * x + coefficients[2] * x * x
        predictions.append(prediction)
        compensated.append(observation - prediction)
    train_residuals = tuple(
        gyro_values[start + i] - predictions[start + i] for i in range(count)
    )
    return TemperatureFit(
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
        coefficients=tuple(coefficients),
        residuals=train_residuals,
        predictions=tuple(predictions),
        compensated=tuple(compensated),
        train_start=start,
        train_end=end,
    )


def analyze_temperature_compensation(timestamps, gyro, temperatures, train_range,
                                     cluster_sizes, dt=None, holdout_range=None):
    """拟合并对原/补偿数据分别计算训练区间与留出区间的重叠 Allan 统计。"""
    gyro_values = _validate_real_sequence(gyro, "gyro")
    n = len(gyro_values)
    if n == 0:
        raise AllanValueError("gyro: must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise AllanValueError(f"gyro: length {n} exceeds interface bound {MAX_SAMPLES}")
    timestamp_values = _validate_real_sequence(timestamps, "timestamps", n)
    _validate_real_sequence(temperatures, "temperatures", n)
    start, end = _validate_index_range(train_range, "train_range", n)
    if holdout_range is None:
        if end >= n:
            raise AllanValueError(
                "holdout_range: no samples remain after train_range; specify a non-empty disjoint interval"
            )
        holdout = (end, n)
    else:
        holdout = _validate_index_range(holdout_range, "holdout_range", n)
    h0, h1 = holdout
    if max(start, h0) < min(end, h1):
        raise AllanValueError("holdout_range must not overlap train_range")

    if dt is None:
        if n < 2:
            raise AllanValueError("dt: cannot infer sample period from one timestamp")
        dt_value = (timestamp_values[-1] - timestamp_values[0]) / (n - 1)
    else:
        if not _is_real(dt):
            raise AllanTypeError("dt: expected int/float (bool is not accepted)")
        dt_value = float(dt)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise AllanValueError("dt: must be finite and > 0")
    _validate(gyro_values, dt_value, cluster_sizes)
    for i in range(1, n):
        measured = timestamp_values[i] - timestamp_values[i - 1]
        tolerance = 1e-12 * max(1.0, abs(measured), abs(dt_value))
        if abs(measured - dt_value) > tolerance:
            raise AllanValueError(
                f"timestamps[{i}]: fixed-rate spacing {measured!r} does not match dt {dt_value!r}; "
                "resampling/interpolation is not performed"
            )

    fit = fit_temperature_polynomial(
        timestamp_values, gyro_values, temperatures, (start, end)
    )
    original_training = analyze(gyro_values[start:end], dt_value, cluster_sizes)
    compensated_training = analyze(list(fit.compensated[start:end]), dt_value, cluster_sizes)
    original_holdout = analyze(gyro_values[h0:h1], dt_value, cluster_sizes)
    compensated_holdout = analyze(list(fit.compensated[h0:h1]), dt_value, cluster_sizes)
    return TemperatureCompensationResult(
        temperature_fit=fit,
        compensated=fit.compensated,
        timestamps=tuple(timestamp_values),
        train_start=start,
        train_end=end,
        holdout_start=h0,
        holdout_end=h1,
        original_training=original_training,
        compensated_training=compensated_training,
        original_holdout=original_holdout,
        compensated_holdout=compensated_holdout,
        dt=dt_value,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def analyze(samples, dt, cluster_sizes) -> AllanResult:
    """计算重叠 Allan 方差并做 log-log 连续分段直线分解。

    参数:
        samples: 定频单轴角速率序列，长度 1..8192，元素为 int/float 且有限；
            bool 元素按类型错误拒绝。
        dt: 采样周期（秒），int/float、有限且 > 0；bool 拒绝。
        cluster_sizes: 聚类大小 m 的有限序列（list/tuple/range），每个为
            真正的 int（bool、float 拒绝）且 m >= 1，长度 <= 8192；允许重复、
            非单调，输出点顺序与输入一致，拟合前去重升序。

    返回:
        AllanResult。无法拟合（可拟合点 < 3，例如常量零方差或短数据）时抛出
        FitInfeasibleError，其 .points 附带已算好的逐 m 统计。

    异常:
        AllanTypeError / AllanValueError: 非法输入，信息定位到具体参数与下标。
        FitInfeasibleError: 证明无满足约束的分段方案。
        SearchBudgetExhaustedError: 搜索预算耗尽（公开输入边界内不可达）。
    """
    values, dt_f, sizes = _validate(samples, dt, cluster_sizes)
    points = _allan_points(values, dt_f, sizes)
    fit = _fit_piecewise(points)
    return AllanResult(tuple(points), fit, dt_f, len(values))
