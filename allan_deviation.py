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
    "fit_temperature_compensation",
    "analyze_with_temperature",
    "AllanPoint",
    "AllanResult",
    "Breakpoint",
    "Segment",
    "FitResult",
    "TemperatureFit",
    "TemperatureAllanResult",
    "AllanSubsetReport",
    "SplitAllanReport",
    "AllanError",
    "AllanTypeError",
    "AllanValueError",
    "FitInfeasibleError",
    "SearchBudgetExhaustedError",
    "TemperatureFitError",
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


@dataclass(frozen=True)
class TemperatureFit:
    """训练区间上的离线二次温度模型与补偿序列。

    模型为 ``prediction = c0 + c1*u + c2*u**2``，其中
    ``u = (temperature - center) / scale``。常温数据退化为常数项模型，
    ``coefficients`` 仍返回三元组，一次、二次系数为 0。
    """

    center: float
    scale: float
    coefficients: tuple[float, float, float]
    residuals: tuple[float, ...]
    compensated_samples: tuple[float, ...]
    train_indices: tuple[int, ...]
    term_count: int
    sse: float
    rmse: float


@dataclass(frozen=True)
class AllanSubsetReport:
    """一个连续区间上的 Allan 点；若分段拟合无解则 fit 为 None。"""

    start_index: int
    end_index: int
    points: tuple[AllanPoint, ...]
    fit: FitResult | None
    infeasible_reason: str | None


@dataclass(frozen=True)
class SplitAllanReport:
    """同一区间上原数据与温度补偿数据的并列 Allan 结果。"""

    original: AllanSubsetReport | None
    compensated: AllanSubsetReport | None


@dataclass(frozen=True)
class TemperatureAllanResult:
    """温度拟合及训练/留出区间分别计算的 Allan 结果。"""

    temperature_fit: TemperatureFit
    training: SplitAllanReport
    holdout: SplitAllanReport | None
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


class TemperatureFitError(AllanError):
    """温度模型无法被唯一拟合（数据不足、温度列相关或温度与时间完全混淆）。"""


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


# ---------------------------------------------------------------------------
# 离线温度补偿（二次多项式，Householder QR；非在线标定）
# ---------------------------------------------------------------------------

_SEQUENCE_TYPES = (str, bytes, bytearray, memoryview)


def _flatten_chunks(values, name: str):
    """接受扁平实数序列或由连续块组成的序列，块边界不参与拟合。"""
    if not isinstance(values, Sequence) or isinstance(values, _SEQUENCE_TYPES):
        raise AllanTypeError(f"{name}: expected a finite sequence or sequence chunks")
    if not values:
        return values
    if not isinstance(values[0], Sequence) or isinstance(values[0], _SEQUENCE_TYPES):
        return values

    flattened = []
    for chunk_index, chunk in enumerate(values):
        if not isinstance(chunk, Sequence) or isinstance(chunk, _SEQUENCE_TYPES):
            raise AllanTypeError(
                f"{name}[{chunk_index}]: chunks must be finite real-number sequences"
            )
        flattened.extend(chunk)
    return flattened


def _validate_real_sequence(values, name: str, expected_len: int | None = None):
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(f"{name}: expected a finite sequence of real numbers")
    if expected_len is not None and len(values) != expected_len:
        raise AllanValueError(
            f"{name}: length {len(values)} does not align with required length "
            f"{expected_len}; interpolation is not performed"
        )
    result = []
    for i, value in enumerate(values):
        if not _is_real(value):
            raise AllanTypeError(
                f"{name}[{i}]: expected int/float (bool/str/complex rejected)"
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise AllanValueError(
                f"{name}[{i}]: value must be finite (NaN/Infinity rejected)"
            )
        result.append(fvalue)
    return result


def _train_bounds(n: int, train_range, timestamps: Sequence[float] | None):
    if not isinstance(train_range, Sequence) or isinstance(
        train_range, (str, bytes, bytearray, memoryview)
    ) or len(train_range) != 2:
        raise AllanTypeError("train_range: expected (start, end)")
    start_value, end_value = train_range
    if not _is_real(start_value) or not _is_real(end_value):
        raise AllanTypeError(
            "train_range: start/end must be finite int/float timestamp or index values"
        )
    start_float, end_float = float(start_value), float(end_value)
    if not math.isfinite(start_float) or not math.isfinite(end_float):
        raise AllanValueError("train_range: start/end must be finite")

    if timestamps is not None:
        try:
            start = timestamps.index(start_float)
            end = timestamps.index(end_float)
        except ValueError:
            raise TemperatureFitError(
                "train_range timestamp was not found exactly; refusing to interpolate"
            ) from None
    else:
        if isinstance(start_value, bool) or isinstance(end_value, bool):
            raise AllanTypeError("train_range: bool bounds are not accepted")
        if not isinstance(start_value, int) or not isinstance(end_value, int):
            raise AllanTypeError(
                "train_range: without timestamps, inclusive bounds must be Python ints"
            )
        start, end = start_value, end_value

    if start < 0 or end < 0 or start >= n or end >= n:
        raise AllanValueError(f"train_range: bounds must lie within [0,{n - 1}]")
    if end < start:
        raise AllanValueError("train_range: end precedes start")
    return start, end


def _qr_solve(matrix, rhs, rank_tolerance=1e-11):
    """用 Householder QR 解小型超定最小二乘，不构造正规方程。"""
    a = [list(map(float, row)) for row in matrix]
    b = list(map(float, rhs))
    rows = len(a)
    cols = len(a[0]) if rows else 0
    if rows < cols or cols == 0:
        raise TemperatureFitError(
            f"temperature fit needs at least {cols} independent training rows, got {rows}"
        )

    for k in range(cols):
        column_norm = math.sqrt(math.fsum(a[i][k] ** 2 for i in range(k, rows)))
        scale = max(1.0, max(abs(a[i][k]) for i in range(k, rows)))
        if column_norm <= rank_tolerance * scale:
            raise TemperatureFitError(
                "temperature design columns are correlated; quadratic model is not identifiable"
            )
        alpha = -math.copysign(column_norm, a[k][k])
        v0 = a[k][k] - alpha
        if v0 == 0.0:
            raise TemperatureFitError(
                "temperature design columns are correlated; quadratic model is not identifiable"
            )
        v = [0.0] * rows
        v[k] = v0
        for i in range(k + 1, rows):
            v[i] = a[i][k]
        vnorm_sq = math.fsum(x * x for x in v[k:])
        for j in range(k, cols):
            dot = math.fsum(v[i] * a[i][j] for i in range(k, rows))
            factor = 2.0 * dot / vnorm_sq
            for i in range(k, rows):
                a[i][j] -= factor * v[i]
        dot_b = math.fsum(v[i] * b[i] for i in range(k, rows))
        factor_b = 2.0 * dot_b / vnorm_sq
        for i in range(k, rows):
            b[i] -= factor_b * v[i]

    x = [0.0] * cols
    for i in range(cols - 1, -1, -1):
        numerator = b[i] - math.fsum(a[i][j] * x[j] for j in range(i + 1, cols))
        if abs(a[i][i]) <= rank_tolerance * max(1.0, abs(numerator)):
            raise TemperatureFitError(
                "temperature design columns are correlated; quadratic model is not identifiable"
            )
        x[i] = numerator / a[i][i]
    return x


def _exact_affine_with_time(values, times):
    if len(values) < 2:
        return False
    n = len(values)
    mean_t = math.fsum(times) / n
    mean_v = math.fsum(values) / n
    stt = math.fsum((t - mean_t) ** 2 for t in times)
    if stt == 0.0:
        return False
    stv = math.fsum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    slope = stv / stt
    intercept = mean_v - slope * mean_t
    scale = max(1.0, max(abs(v) for v in values))
    return all(
        abs(v - (intercept + slope * t)) <= 1e-12 * scale
        for t, v in zip(times, values)
    )


def fit_temperature_compensation(
    samples,
    temperatures,
    train_range,
    timestamps=None,
) -> TemperatureFit:
    """在调用方指定训练区间拟合温度模型，并对整段离线序列补偿。

    train_range 在无 timestamps 时为闭区间整数下标；提供 timestamps 时，
    起点和终点必须在时间戳序列中精确出现。函数绝不插值。
    """
    samples = _flatten_chunks(samples, "samples")
    temperatures = _flatten_chunks(temperatures, "temperatures")
    gyro = _validate_real_sequence(samples, "samples")
    n = len(gyro)
    temp = _validate_real_sequence(temperatures, "temperatures", n)
    times = None
    if timestamps is not None:
        timestamps = _flatten_chunks(timestamps, "timestamps")
        times = _validate_real_sequence(timestamps, "timestamps", n)
        if any(times[i + 1] <= times[i] for i in range(n - 1)):
            raise AllanValueError("timestamps: must be strictly increasing")

    start, end = _train_bounds(n, train_range, times)
    train_index = list(range(start, end + 1))
    train_count = len(train_index)
    if train_count == 0:
        raise TemperatureFitError("temperature fit training interval is empty")

    train_y = [gyro[i] for i in train_index]
    train_t = [temp[i] for i in train_index]
    train_time = [times[i] if times is not None else float(i) for i in train_index]

    center = math.fsum(train_t) / train_count
    temperature_span = max(train_t) - min(train_t)
    if temperature_span == 0.0:
        scale = 1.0
        design = [[1.0] for _ in train_t]
        beta = _qr_solve(design, train_y)
        coefficients = (beta[0], 0.0, 0.0)
        term_count = 1
    else:
        if train_count < 3:
            raise TemperatureFitError(
                "quadratic constant/linear/second-order fit needs at least 3 training rows"
            )
        if _exact_affine_with_time(train_t, train_time):
            raise TemperatureFitError(
                "temperature is exactly confounded with time in the training interval; "
                "temperature and time effects cannot be separated"
            )
        scale = temperature_span / 2.0
        u_values = [(t - center) / scale for t in train_t]
        design = [[1.0, u, u * u] for u in u_values]
        beta = _qr_solve(design, train_y)
        coefficients = (beta[0], beta[1], beta[2])
        term_count = 3

    residual_values = []
    for i in train_index:
        u = (temp[i] - center) / scale
        prediction = coefficients[0] + coefficients[1] * u + coefficients[2] * u * u
        residual_values.append(gyro[i] - prediction)

    compensated = []
    for i in range(n):
        u = (temp[i] - center) / scale
        prediction = coefficients[0] + coefficients[1] * u + coefficients[2] * u * u
        compensated.append(gyro[i] - prediction)

    sse = math.fsum(r * r for r in residual_values)
    rmse = math.sqrt(sse / train_count)
    return TemperatureFit(
        center=center,
        scale=scale,
        coefficients=coefficients,
        residuals=tuple(residual_values),
        compensated_samples=tuple(compensated),
        train_indices=tuple(train_index),
        term_count=term_count,
        sse=sse,
        rmse=rmse,
    )


def _subset_allan(values, dt, cluster_sizes, start, end):
    subset = values[start:end + 1]
    try:
        points = _allan_points(subset, dt, cluster_sizes)
        fit = _fit_piecewise(points)
        reason = None
    except FitInfeasibleError as exc:
        points = exc.points if exc.points else _allan_points(subset, dt, cluster_sizes)
        fit = None
        reason = exc.reason
    return AllanSubsetReport(
        start_index=start,
        end_index=end,
        points=tuple(points),
        fit=fit,
        infeasible_reason=reason,
    )


def analyze_with_temperature(
    samples,
    temperatures,
    dt,
    cluster_sizes,
    train_range,
    timestamps=None,
) -> TemperatureAllanResult:
    """拟合并对完整序列补偿，然后分别报告训练段与相邻留出段 Allan。"""
    samples = _flatten_chunks(samples, "samples")
    temperatures = _flatten_chunks(temperatures, "temperatures")
    gyro, dt_f, sizes = _validate(samples, dt, cluster_sizes)
    temp = _validate_real_sequence(temperatures, "temperatures", len(gyro))
    times = None
    if timestamps is not None:
        timestamps = _flatten_chunks(timestamps, "timestamps")
        times = _validate_real_sequence(timestamps, "timestamps", len(gyro))
        if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
            raise AllanValueError("timestamps: must be strictly increasing")
    start, end = _train_bounds(len(gyro), train_range, times)

    if start == 0 and end == len(gyro) - 1:
        raise AllanValueError(
            "train_range covers all samples; a nonempty holdout interval is required"
        )
    if start == 0:
        holdout_start, holdout_end = end + 1, len(gyro) - 1
    elif end == len(gyro) - 1:
        holdout_start, holdout_end = 0, start - 1
    else:
        raise AllanValueError(
            "train_range must be a prefix or suffix so the holdout interval is contiguous"
        )

    fit_result = fit_temperature_compensation(
        gyro, temp, train_range, timestamps
    )
    compensated = list(fit_result.compensated_samples)
    training = SplitAllanReport(
        original=_subset_allan(gyro, dt_f, sizes, start, end),
        compensated=_subset_allan(compensated, dt_f, sizes, start, end),
    )
    holdout = SplitAllanReport(
        original=_subset_allan(gyro, dt_f, sizes, holdout_start, holdout_end),
        compensated=_subset_allan(
            compensated, dt_f, sizes, holdout_start, holdout_end
        ),
    )
    return TemperatureAllanResult(
        temperature_fit=fit_result,
        training=training,
        holdout=holdout,
        dt=dt_f,
        n_samples=len(gyro),
    )
