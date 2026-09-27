"""固定速率单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集）。

本模块只实现题目契约内的数学子集：

* :func:`analyze` 用累积和计算重叠 Allan 方差（overlapping Allan variance）；
* 对差分样本数 >= 20 且方差严格为正的 (m, sigma) 点，在 log(tau)-log(sigma)
  平面上做连续（相邻区间无重叠、无缺口）分段直线最小二乘拟合；
* 分段数 1..3，每段至少 3 个点，分段罚项为模块常量 :data:`SEGMENT_PENALTY`，
  全局最优由动态规划穷举所有合法切分得到。
* :class:`StreamingAllan`（工厂 :func:`streaming_allan`）在构造时固定统计尺度，
  随后逐批 :meth:`~StreamingAllan.append` 样本：跨批次窗口不遗漏、不重复，
  只保留 ``2*m_max-1`` 个历史样本；样本不足的长尺度单独标记为不可计算。

注意：拟合斜率只是对该批离线数据 log-log 形态的经验描述，不是任何临床结论或
硬件指标保证；本模块也不宣称完整兼容任何行业标准。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "analyze",
    "streaming_allan",
    "StreamingAllan",
    "StreamingAllanPoint",
    "StreamingAllanResult",
    "StreamingClosedError",
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


@dataclass(frozen=True)
class StreamingAllanPoint:
    """流式累加器在某个时刻、单个聚类大小 m 的统计结果。

    字段含义与 :class:`AllanPoint` 一致，并额外给出:

    属性:
        computable: 是否已有足够样本得到 Allan 方差的无偏估计
            （``count >= 2``，即已见到 ``>= 2*m+1`` 个样本）。样本不足时
            ``variance``/``adev`` 为 ``None``，该尺度单独不可计算，不影响
            其他尺度；``count`` 与 ``tau`` 始终返回。
        n_samples: 截至当前该累加器已接收的样本总数（所有点相同）。
    """

    m: int
    tau: float
    variance: float | None
    adev: float | None
    count: int
    eligible_for_fit: bool
    computable: bool
    n_samples: int


@dataclass(frozen=True)
class StreamingAllanResult:
    """流式累加器一次快照（append/snapshot 的返回值）。

    属性:
        points: 与构造时 ``cluster_sizes`` 一一对应、顺序相同的逐尺度统计；
            包含重复尺度（每个重复各得一份相同结果）。
        fit: 截至当前数据的分段直线拟合结果；合格点不足 3 个时为 ``None``
            （原因见 ``fit_infeasible_reason``），此时逐尺度统计照常返回。
        fit_infeasible_reason: fit 为 None 时的原因字符串；fit 非 None 时为 None。
        dt: 采样周期（秒）。
        n_samples: 截至当前接收的样本总数。
        n_differences_by_m: 去重 m 到当前实际差分样本数的映射快照。
        retained_samples: 为计算下一批跨批次窗口而保留的样本数（含必要历史）。
        retained_limit: 保留历史的恒定上限 ``max(0, 2*m_max - 1)``。
    """

    points: tuple[StreamingAllanPoint, ...]
    fit: FitResult | None
    fit_infeasible_reason: str | None
    dt: float
    n_samples: int
    retained_samples: int
    retained_limit: int


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class AllanError(Exception):
    """本模块所有公开错误的基类。"""


class AllanTypeError(AllanError, TypeError):
    """参数类型非法（含对整数参数传入 bool）。"""


class StreamingClosedError(AllanError, RuntimeError):
    """对已结束（close）的流式累加器继续 append（区别于取值/类型错误）。"""


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


def _check_real_sequence(seq: object, name: str, *, allow_empty: bool = False,
                         max_len: int | None = None) -> list[float]:
    """校验一个实数序列并返回其 float 列表（拒绝 bool/NaN/Infinity）。"""
    if not isinstance(seq, Sequence) or isinstance(
        seq, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            f"{name}: expected a finite sequence (list/tuple/range) of real numbers"
        )
    n = len(seq)
    if n == 0 and not allow_empty:
        raise AllanValueError(f"{name}: must contain at least 1 value")
    if max_len is not None and n > max_len:
        raise AllanValueError(
            f"{name}: length {n} exceeds interface bound {max_len}"
        )
    values: list[float] = [0.0] * n
    for i, v in enumerate(seq):
        if not _is_real(v):
            raise AllanTypeError(
                f"{name}[{i}]: expected int/finite-float (bool/str/complex rejected)"
            )
        try:
            fv = float(v)
        except OverflowError:
            raise AllanValueError(
                f"{name}[{i}]: {v!r} cannot be represented as finite float"
            ) from None
        if not math.isfinite(fv):
            raise AllanValueError(
                f"{name}[{i}]: value must be finite (NaN/Infinity rejected)"
            )
        values[i] = fv
    return values


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


def _check_cluster_sizes(cluster_sizes: object, dt_f: float) -> list[int]:
    if not isinstance(cluster_sizes, Sequence) or isinstance(
        cluster_sizes, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "cluster_sizes: expected a finite sequence (list/tuple/range) of ints"
        )
    k_sizes = len(cluster_sizes)
    if k_sizes == 0:
        raise AllanValueError("cluster_sizes: must contain at least 1 cluster size")
    if k_sizes > MAX_CLUSTER_SIZES:
        raise AllanValueError(
            f"cluster_sizes: length {k_sizes} exceeds interface bound "
            f"{MAX_CLUSTER_SIZES}"
        )
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
    return sizes


def _validate(samples: object, dt: object, cluster_sizes: object):
    dt_f = _check_dt(dt)
    values = _check_real_sequence(samples, "samples", max_len=MAX_SAMPLES)
    sizes = _check_cluster_sizes(cluster_sizes, dt_f)
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


class _CompensatedSum:
    """Neumaier 补偿求和：跨批次累加顺序/分批方式变化时漂移可忽略。"""

    __slots__ = ("total", "comp")

    def __init__(self):
        self.total = 0.0
        self.comp = 0.0

    def add(self, x: float) -> None:
        t = self.total + x
        if abs(self.total) >= abs(x):
            self.comp += (self.total - t) + x
        else:
            self.comp += (x - t) + self.total
        self.total = t

    def value(self) -> float:
        return self.total + self.comp


class StreamingAllan:
    """固定尺度的增量（分批）重叠 Allan 统计累加器。

    构造时固定 ``dt`` 与一组统计尺度 ``cluster_sizes``（允许重复/非单调，
    输出顺序与构造输入一致），随后用 :meth:`append` 逐批喂入样本。每次
    append（或随时 :meth:`snapshot`）都返回截至当前全部已接收数据的统计结果。

    * **跨批次窗口不遗漏、不重复**：第 ``k`` 个相邻块均值差分只在其末端样本
      ``x[k+2m-1]`` 到达的那一批被计入一次；早到的样本只作为历史保存。
    * **只保留下一批确实需要的历史**：内部仅保留最近
      ``2*m_max - 1`` 个样本（``m_max=max(cluster_sizes)``），更早的样本对
      任何未来窗口差分都不再需要，故丢弃。
    * **长尺度单独不可计算**：样本不足以使某 ``m`` 得到 ``>=2`` 个差分时，
      该尺度 ``computable=False``、``variance/adev=None``，其他尺度照常返回。
    * **与离线一致**：同一数据不论分多少批喂入，统计与把全部样本一次性交给
      :func:`analyze` 的结果在 README 误差口径内一致（平方和用补偿求和）。
    * **失败无部分变更**：append 的输入先整体校验、新差分先全部算出，任一步
      失败则累加器状态保持在该批之前，不产生部分写入。

    长度无上限：离线 :func:`analyze` 的 8192 点边界不适用于流式总样本数；
    内存只与 ``2*m_max`` 成正比，与已接收样本总数无关。
    """

    def __init__(self, dt, cluster_sizes):
        dt_f = _check_dt(dt)
        sizes = _check_cluster_sizes(cluster_sizes, dt_f)
        self._dt = dt_f
        self._sizes: tuple[int, ...] = tuple(sizes)
        # 去重尺度及其在输出中的所有位置（重复尺度共享同一累加状态）。
        positions: dict[int, list[int]] = {}
        for i, m in enumerate(self._sizes):
            positions.setdefault(m, []).append(i)
        self._positions = {m: tuple(ps) for m, ps in positions.items()}
        self._unique_ms: tuple[int, ...] = tuple(sorted(positions))
        self._m_max = max(self._sizes)
        self._retain = 2 * self._m_max - 1
        self._n = 0
        # 以固定中心（首批精确均值）去偏置后的历史样本；窗口差分对加常数
        # 数学不变，固定中心使跨批结果与分批方式无关且对大基线稳健。
        self._history: list[float] = []
        self._center = 0.0
        self._count: dict[int, int] = {m: 0 for m in self._unique_ms}
        self._qsum: dict[int, _CompensatedSum] = {
            m: _CompensatedSum() for m in self._unique_ms
        }
        self._closed = False

    # -- 基本属性 ---------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        """构造时给定的尺度序列（含重复、原顺序）。"""
        return self._sizes

    @property
    def n_samples(self) -> int:
        """截至当前已成功接收的样本总数。"""
        return self._n

    @property
    def retained_samples(self) -> int:
        """当前为未来跨批次窗口保留的历史样本数。"""
        return len(self._history)

    @property
    def retained_limit(self) -> int:
        """保留历史的恒定上限 ``2*m_max - 1``。"""
        return self._retain

    @property
    def closed(self) -> bool:
        return self._closed

    # -- 增量喂入 ---------------------------------------------------------

    def append(self, batch) -> StreamingAllanResult:
        """追加一批样本，返回追加后截至当前的完整统计快照。

        参数:
            batch: 非空实数序列（list/tuple/range），元素 int/有限 float，
                bool/NaN/Infinity 拒绝；空批按取值错误拒绝。

        异常:
            AllanTypeError / AllanValueError: 本批非法，状态保持不变。
            StreamingClosedError: 累加器已 close。
        """
        if self._closed:
            raise StreamingClosedError(
                "this StreamingAllan accumulator has been closed; no more append()"
            )
        # 流式批次不设 8192 上限（总样本本来就无界）；但必须非空且全部有限。
        values = _check_real_sequence(batch, "batch", allow_empty=False)
        self._ingest(values)
        return self.snapshot()

    def _ingest(self, values: list[float]) -> None:
        old_n = self._n
        old_count = self._count
        if old_n == 0:
            try:
                self._center = math.fsum(values) / len(values)
            except OverflowError:
                raise AllanValueError(
                    "batch magnitudes overflow even the exact mean computation; "
                    "inputs too large for the float64 computation"
                ) from None
            if not math.isfinite(self._center):
                raise AllanValueError(
                    "batch mean is not finite; inputs out of float64 range"
                )
        combined = self._history + [v - self._center for v in values]
        new_n = old_n + len(values)

        # 对固定中心数据建前缀和；combined 起点对应的全局下标为 offset。
        offset = old_n - len(self._history)
        cum = [0.0] * (len(combined) + 1)
        acc = 0.0
        for i, v in enumerate(combined):
            acc += v
            cum[i + 1] = acc

        # 先把本批所有新差分算到临时累加器，全部成功后才提交，保证原子性。
        new_q: dict[int, _CompensatedSum] = {}
        new_count: dict[int, int] = {}
        for m in self._unique_ms:
            total_count = max(0, new_n - 2 * m + 1)
            prev = old_count[m]
            add = total_count - prev
            if add > 0:
                q = _CompensatedSum()
                # 新完备差分的全局起点 k = prev .. total_count-1（其末端样本
                # 落在本批）；更早的 k 已在历史批次计过，绝不重复。
                for k in range(prev, total_count):
                    j = k - offset
                    d = (cum[j + 2 * m] - 2.0 * cum[j + m] + cum[j]) / m
                    dd = d * d
                    if not math.isfinite(dd):
                        raise AllanValueError(
                            f"allan variance overflowed finite float for m={m}; "
                            "input magnitude too large for the computation"
                        )
                    q.add(dd)
                new_q[m] = q
            new_count[m] = total_count

        # 提交：合并平方和、更新计数、只保留确实需要的尾部历史。
        for m, q in new_q.items():
            self._qsum[m].add(q.value())
        self._count = new_count
        self._n = new_n
        if len(combined) > self._retain:
            combined = combined[len(combined) - self._retain:]
        self._history = combined

    # -- 当前结果 ---------------------------------------------------------

    def _build_points(self):
        n = self._n
        points: list[StreamingAllanPoint] = [None] * len(self._sizes)  # type: ignore[list-item]
        fit_points: list[AllanPoint] = []
        for m in self._unique_ms:
            count = self._count[m]
            tau = m * self._dt
            if count >= 2:
                variance = self._qsum[m].value() / (2.0 * (count - 1))
                if not math.isfinite(variance):
                    raise AllanValueError(
                        f"allan variance overflowed finite float for m={m}; "
                        "input magnitude too large for the computation"
                    )
                adev = math.sqrt(variance)
                eligible = count >= MIN_DIFFERENCES and variance > 0.0
            else:
                variance = None
                adev = None
                eligible = False
            computable = count >= 2
            sp = StreamingAllanPoint(
                m, tau, variance, adev, count, eligible, computable, n
            )
            ap = AllanPoint(m, tau, variance, adev, count, eligible)
            fit_points.append(ap)
            for pos in self._positions[m]:
                points[pos] = sp
        return tuple(points), fit_points

    def snapshot(self) -> StreamingAllanResult:
        """返回截至当前的统计快照，不改变累加器状态（未喂入数据前亦可调用）。"""
        points, fit_points = self._build_points()
        fit = None
        reason: str | None = None
        if self._n > 0:
            try:
                fit = _fit_piecewise(fit_points)
            except FitInfeasibleError as exc:
                reason = exc.reason
        else:
            reason = (
                "no samples appended yet; need at least "
                f"{MIN_SEGMENT_POINTS} eligible log-log points"
            )
        return StreamingAllanResult(
            points=points,
            fit=fit,
            fit_infeasible_reason=reason,
            dt=self._dt,
            n_samples=self._n,
            retained_samples=len(self._history),
            retained_limit=self._retain,
        )

    def close(self) -> None:
        """结束该累加器：之后 append 抛 StreamingClosedError，snapshot 仍可用。"""
        self._closed = True


def streaming_allan(dt, cluster_sizes) -> StreamingAllan:
    """创建固定尺度的流式 Allan 累加器（:class:`StreamingAllan` 的工厂）。"""
    return StreamingAllan(dt, cluster_sizes)


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
