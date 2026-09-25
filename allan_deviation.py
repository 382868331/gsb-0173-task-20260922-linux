"""固定速率单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集 + 在线累积器）。

本模块只实现题目契约内的数学子集：

* :func:`analyze` 用累积和计算重叠 Allan 方差（overlapping Allan variance）；
* :class:`OnlineAllan` 固定 dt 与 cluster_sizes 的在线累积器：feed 任意长度
  chunk，随时 query 当前完整前缀的逐尺度 Allan 方差、有效差分数与分段拟合；
  只保留最大尺度所需的滑动样本窗口与每尺度 O(1) 累积量，不缓存完整历史，
  样本数可远超批量上限 MAX_SAMPLES，不对不足窗口的尾部补零；
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
    "OnlineAllan",
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


def _check_dt(dt: object) -> float:
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
    return dt_f


def _check_sizes_type(cluster_sizes: object) -> None:
    if not isinstance(cluster_sizes, Sequence) or isinstance(
        cluster_sizes, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "cluster_sizes: expected a finite sequence (list/tuple/range) of ints"
        )


def _check_sizes_bounds(cluster_sizes: Sequence) -> None:
    k_sizes = len(cluster_sizes)
    if k_sizes == 0:
        raise AllanValueError("cluster_sizes: must contain at least 1 cluster size")
    if k_sizes > MAX_CLUSTER_SIZES:
        raise AllanValueError(
            f"cluster_sizes: length {k_sizes} exceeds interface bound "
            f"{MAX_CLUSTER_SIZES}"
        )


def _check_sizes_values(cluster_sizes: Sequence, dt_f: float) -> list[int]:
    sizes: list[int] = [0] * len(cluster_sizes)
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
    return sizes


def _validate(samples: object, dt: object, cluster_sizes: object):
    if not isinstance(samples, Sequence) or isinstance(
        samples, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "samples: expected a finite sequence (list/tuple/...) of real numbers"
        )
    _check_sizes_type(cluster_sizes)

    n = len(samples)
    if n == 0:
        raise AllanValueError("samples: must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise AllanValueError(
            f"samples: length {n} exceeds interface bound {MAX_SAMPLES}"
        )

    dt_f = _check_dt(dt)

    _check_sizes_bounds(cluster_sizes)

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

    sizes = _check_sizes_values(cluster_sizes, dt_f)

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
# 在线累积器：固定 dt 与 cluster_sizes，feed 任意长度 chunk，随时 query
# ---------------------------------------------------------------------------


def _check_chunk(chunk: object) -> list[float]:
    """整块校验 feed 的输入；任一元素非法则抛错，调用方保证此时状态未变。"""
    if not isinstance(chunk, Sequence) or isinstance(
        chunk, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "chunk: expected a finite sequence (list/tuple/...) of real numbers"
        )
    vals: list[float] = [0.0] * len(chunk)
    for i, v in enumerate(chunk):
        if not _is_real(v):
            raise AllanTypeError(
                f"chunk[{i}]: expected int/finite-float (bool/str/complex rejected)"
            )
        try:
            fv = float(v)
        except OverflowError:
            raise AllanValueError(
                f"chunk[{i}]: {v!r} cannot be represented as finite float"
            ) from None
        if not math.isfinite(fv):
            raise AllanValueError(
                f"chunk[{i}]: value must be finite (NaN/Infinity rejected)"
            )
        vals[i] = fv
    return vals


def _comp_add(total: float, comp: float, v: float):
    """Neumaier 补偿加法：把 v 累进 (total, comp)，返回新的 (total, comp)。

    用于滑动窗口和（交替加/减）与差分平方和的累积，使长流的舍入误差
    保持在 eps 量级而不随样本数线性放大。
    """
    t = total + v
    if abs(total) >= abs(v):
        comp += (total - t) + v
    else:
        comp += (v - t) + total
    return t, comp


class _ScaleState:
    """单个尺度 m 的 O(1) 在线累积状态。"""

    __slots__ = ("m", "head", "head_c", "tail", "tail_c", "sq", "sq_c", "count")

    def __init__(self, m: int):
        self.m = m
        self.head = 0.0    # 最近 m 个残差之和（补偿主值）
        self.head_c = 0.0  # head 的补偿项
        self.tail = 0.0    # 之前 m 个残差之和（补偿主值）
        self.tail_c = 0.0  # tail 的补偿项
        self.sq = 0.0      # 均值差平方的累积和（补偿主值）
        self.sq_c = 0.0    # sq 的补偿项
        self.count = 0     # 已累积的差分样本数 = max(0, N - 2m + 1)


class OnlineAllan:
    """固定 dt 与 cluster_sizes 的重叠 Allan 在线累积器。

    用法::

        acc = OnlineAllan(dt, cluster_sizes)
        acc.feed(chunk)          # 任意长度、任意次数；非法 chunk 整块拒绝
        points = acc.query_points()   # 当前前缀的逐 m 统计（不拟合）
        result = acc.query()          # 与 analyze(当前完整前缀) 相同形状的 AllanResult

    契约:

    * ``dt`` 与 ``cluster_sizes`` 仅在构造时校验并固定，创建后不可更改
      （只读属性，无 setter；cluster_sizes 以 tuple 保存）。
    * :meth:`feed` 先**整块**校验再消费：任一元素非法时抛
      AllanTypeError/AllanValueError，已累积状态完全不变（不会吃掉前几个
      合法元素）。空 chunk 是无副作用的 no-op，返回 0。
    * 只保留 ``min(2*m_max, 已见样本数)`` 个滑动样本（``m_max`` 为最大尺度）
      与每尺度 O(1) 累积量；内存不随历史长度增长，样本数可远超
      批量上限 :data:`MAX_SAMPLES`。
    * 不足 2m 的尾部**不补零**：尺度 m 的差分数恒为 ``max(0, N - 2m + 1)``，
      与 :func:`analyze` 对同一前缀的结果一致。
    * 数值稳定：进入累积前减去首个非空 chunk 的精确均值（差分对整体常数
      平移数学不变，去偏置避免大直流基线下的灾难性抵消）；窗口和与差分
      平方和均用 Neumaier 补偿累加。
    * :meth:`query` / :meth:`query_points` 为只读操作：分段拟合在累积量的
      快照副本上进行，不改变任何累积状态，可任意次数调用。
    """

    __slots__ = (
        "_dt", "_sizes", "_states", "_m_max", "_cap", "_buf", "_n", "_offset",
    )

    def __init__(self, dt, cluster_sizes):
        _check_sizes_type(cluster_sizes)
        dt_f = _check_dt(dt)
        _check_sizes_bounds(cluster_sizes)
        sizes = _check_sizes_values(cluster_sizes, dt_f)
        self._dt = dt_f
        self._sizes = tuple(sizes)
        uniq = sorted(set(sizes))
        self._states = [_ScaleState(m) for m in uniq]
        self._m_max = uniq[-1]
        self._cap = 2 * self._m_max
        self._buf: list[float] = []  # 惰性增长到 cap 后转为环形覆盖
        self._n = 0
        self._offset: float | None = None  # 首个非空 chunk 的精确均值

    # -- 只读属性（创建后不可改 dt 与尺度）--------------------------------

    @property
    def dt(self) -> float:
        """构造时校验后的采样周期（float），只读。"""
        return self._dt

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        """构造时校验后的尺度序列（tuple 副本，含重复与原顺序），只读。"""
        return self._sizes

    @property
    def n_samples(self) -> int:
        """已成功消费的样本总数（可远超 MAX_SAMPLES）。"""
        return self._n

    @property
    def retained_samples(self) -> int:
        """当前保留的滑动窗口样本数，上界为 2*max(cluster_sizes)。"""
        return len(self._buf)

    # -- 写入 --------------------------------------------------------------

    def feed(self, chunk) -> int:
        """消费一个任意长度的样本 chunk，返回消费的样本数。

        先整块校验再写入：任一元素非法（非数值、bool、NaN/Infinity、不可
        表示为有限 float 的超大 int）则抛 AllanTypeError/AllanValueError，
        累积状态保持调用前的样子。空 chunk 为 no-op，返回 0。
        """
        vals = _check_chunk(chunk)  # 失败时此处抛出，下方状态一行未动
        if not vals:
            return 0
        if self._offset is None:
            # 差分对整体常数平移不变：减去首个非空 chunk 的精确均值，
            # 使大直流偏置信号的后续累积在 O(噪声) 量级上进行。
            self._offset = math.fsum(vals) / len(vals)
        buf = self._buf
        cap = self._cap
        states = self._states
        off = self._offset
        n = self._n
        for v in vals:
            r = v - off
            t = n
            # 1) 先用旧窗口内容更新各尺度（读取最旧样本必须发生在覆盖之前）。
            for st in states:
                m = st.m
                if t >= m:
                    old = buf[(t - m) % cap]
                    hv = r - old  # head 窗口滑出 r_{t-m}
                    if t >= 2 * m:
                        tv = old - buf[(t - 2 * m) % cap]  # tail 滑入 r_{t-m}、滑出 r_{t-2m}
                    else:
                        tv = old
                    st.tail, st.tail_c = _comp_add(st.tail, st.tail_c, tv)
                else:
                    hv = r  # head 窗口尚未满，只进不出
                st.head, st.head_c = _comp_add(st.head, st.head_c, hv)
                if t >= 2 * m - 1:
                    # 新差分 d_k（k = t-2m+1）= 后 m 均值 - 前 m 均值。
                    d = ((st.head + st.head_c) - (st.tail + st.tail_c)) / m
                    st.sq, st.sq_c = _comp_add(st.sq, st.sq_c, d * d)
                    st.count += 1
            # 2) 写入新样本：窗口未满则追加，已满则环形覆盖最旧槽位。
            if t < cap:
                buf.append(r)
            else:
                buf[t % cap] = r
            n = t + 1
        self._n = n
        return len(vals)

    # -- 读取（不改变累积状态）--------------------------------------------

    def query_points(self) -> tuple[AllanPoint, ...]:
        """当前完整前缀的逐 m AllanPoint（顺序与构造时 cluster_sizes 一致）。

        不做分段拟合，永不抛 FitInfeasibleError；方差溢出有限 float 时与
        批量入口一样抛 AllanValueError。
        """
        dt = self._dt
        by_m: dict[int, AllanPoint] = {}
        for st in self._states:
            m = st.m
            count = st.count
            tau = m * dt
            if count < 2:
                # 差分样本数 < 2 无法做无偏估计（2m > N 时 count 为 0）。
                by_m[m] = AllanPoint(m, tau, None, None, count, False)
                continue
            variance = (st.sq + st.sq_c) / (2.0 * (count - 1))
            if not math.isfinite(variance):
                raise AllanValueError(
                    f"allan variance overflowed finite float for m={m}; "
                    "input magnitude too large for the computation"
                )
            adev = math.sqrt(variance)
            eligible = count >= MIN_DIFFERENCES and variance > 0.0
            by_m[m] = AllanPoint(m, tau, variance, adev, count, eligible)
        return tuple(by_m[m] for m in self._sizes)

    def query(self) -> AllanResult:
        """与 ``analyze(当前完整前缀, dt, cluster_sizes)`` 相同形状的结果。

        分段拟合在 query_points() 的返回值上进行（纯函数），不触碰累积
        状态；合格拟合点不足时与批量入口一样抛 FitInfeasibleError，其
        .points 附带当前逐 m 统计。
        """
        points = self.query_points()
        fit = _fit_piecewise(points)
        return AllanResult(tuple(points), fit, self._dt, self._n)
