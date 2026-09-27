"""固定速率单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集）。

本模块只实现题目契约内的数学子集：

* :func:`analyze` 一次性计算重叠 Allan 方差（overlapping Allan variance）；
* :class:`StreamingAllan` 在构造时固定全部统计尺度，可持续追加单样本或批次；
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
    "StreamingAllan",
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
_ROLLING_SUM_REBUILD_INTERVAL = 32


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
    fit: FitResult | None
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


class _CompensatedSum:
    """Neumaier 补偿求和，用于长序列状态递推。"""

    __slots__ = ("value", "compensation")

    def __init__(self):
        self.value = 0.0
        self.compensation = 0.0

    def add(self, x: float) -> None:
        new_value = self.value + x
        if abs(self.value) >= abs(x):
            self.compensation += (self.value - new_value + x)
        else:
            self.compensation += (x - new_value + self.value)
        self.value = new_value


_MAX_ACTIVE_M = MAX_SAMPLES // 2


class _ScaleState:
    __slots__ = (
        "m",
        "current_difference",
        "square_sum",
        "square_compensation",
        "count",
    )

    def __init__(self, m: int):
        self.m = m
        self.current_difference = _CompensatedSum()
        self.square_sum = 0.0
        self.square_compensation = 0.0
        self.count = 0


    def add_square(self, value: float) -> None:
        new_sum = self.square_sum + value
        if abs(self.square_sum) >= abs(value):
            self.square_compensation += (self.square_sum - new_sum + value)
        else:
            self.square_compensation += (value - new_sum + self.square_sum)
        self.square_sum = new_sum

    def rebuild_current_difference(
        self, history: Sequence[float], capacity: int, index: int
    ) -> None:
        start = index - 2 * self.m + 1
        terms = []
        for offset in range(2 * self.m):
            value = history[(start + offset) % capacity]
            if offset < self.m:
                terms.append(-value)
            else:
                terms.append(value)
        self.current_difference = _CompensatedSum()
        try:
            self.current_difference.value = math.fsum(terms)
        except OverflowError:
            raise AllanValueError(
                f"allan difference overflowed finite float for m={self.m}; "
                "input magnitude too large for the computation"
            ) from None



class StreamingAllan:
    """对预先固定的一组聚类尺度持续追加样本的有状态 Allan 统计。

    每次 :meth:`append` 都返回截至当前全部样本的统计结果。对象只保留最长尺度
    形成跨批次窗口所必需的 ``2*max(m)+1`` 个原始样本；每个新差分只计数一次。

    较长尺度样本不足时，其 ``variance``/``adev`` 为 ``None``，其他尺度仍正常
    计算。若合格拟合点不足 3 个，返回结果的 ``fit`` 为 ``None``，这不是异常。
    """

    def __init__(self, dt, cluster_sizes):
        self.dt, checked_sizes = _validate_fixed_scale_input(dt, cluster_sizes)
        self.cluster_sizes: tuple[int, ...] = tuple(checked_sizes)
        self._active_sizes = {m for m in set(checked_sizes) if m <= _MAX_ACTIVE_M}
        self.max_m = max(self._active_sizes, default=1)
        self._states = {m: _ScaleState(m) for m in self._active_sizes}
        capacity = 2 * self.max_m + 1 if self._active_sizes else 1
        self._history: list[float | None] = [None] * capacity
        self._history_capacity = capacity
        self._n = 0
        self._reference = 0.0
        self._total_sum = 0.0
        self._total_compensation = 0.0

    @property
    def n_samples(self) -> int:
        return self._n

    def append(self, samples) -> AllanResult:
        """追加一个或一批同频样本，并返回到当前为止的全部尺度统计。

        新样本先整体校验，通过后才会更新状态；因此非法批次不会造成部分追加。
        """
        batch = _validate_stream_batch(samples, self._n)
        total_sum = self._total_sum
        total_compensation = self._total_compensation
        try:
            for value in batch:
                new_sum = total_sum + value
                if abs(total_sum) >= abs(value):
                    total_compensation += total_sum - new_sum + value
                else:
                    total_compensation += value - new_sum + total_sum
                total_sum = new_sum
            mean = (total_sum + total_compensation) / (self._n + len(batch))
        except OverflowError:
            raise AllanValueError(
                "sample magnitudes overflow even the exact mean computation; "
                "inputs too large for the float64 computation"
            ) from None
        if not math.isfinite(mean):
            raise AllanValueError(
                "sample mean is not finite; inputs out of float64 range"
            )
        saved_history = self._history.copy()
        saved_states = {
            m: (
                state.current_difference.value,
                state.current_difference.compensation,
                state.square_sum,
                state.square_compensation,
                state.count,
            )
            for m, state in self._states.items()
        }
        saved_n = self._n
        saved_reference = self._reference
        saved_total = (self._total_sum, self._total_compensation)
        try:
            for value in batch:
                self._append_one(value)
            self._total_sum = total_sum
            self._total_compensation = total_compensation
            return self.snapshot()
        except Exception:
            for slot, old_value in enumerate(saved_history):
                self._history[slot] = old_value
            for m, saved in saved_states.items():
                state = self._states[m]
                state.current_difference.value = saved[0]
                state.current_difference.compensation = saved[1]
                state.square_sum = saved[2]
                state.square_compensation = saved[3]
                state.count = saved[4]
            self._n = saved_n
            self._reference = saved_reference
            self._total_sum, self._total_compensation = saved_total
            raise

    def snapshot(self) -> AllanResult:
        """返回当前已接收样本的统计，不改变状态。"""
        points = tuple(self._point_for(m) for m in self.cluster_sizes)
        try:
            fit = _fit_piecewise(points)
        except FitInfeasibleError:
            fit = None
        return AllanResult(points, fit, self.dt, self._n)

    def _append_one(self, raw_value: float) -> None:
        index = self._n
        if index == 0:
            self._reference = raw_value
        value = raw_value - self._reference
        self._history[index % self._history_capacity] = value

        for state in self._states.values():
            m = state.m
            if index >= 2 * m - 1:
                if (index - (2 * m - 1)) % _ROLLING_SUM_REBUILD_INTERVAL == 0:
                    state.rebuild_current_difference(
                        self._history, self._history_capacity, index
                    )
                else:
                    delta = value
                    middle = self._history[(index - m) % self._history_capacity]
                    delta -= middle
                    delta -= middle
                    delta += self._history[
                        (index - 2 * m) % self._history_capacity
                    ]
                    state.current_difference.value += delta
                difference = (
                    state.current_difference.value
                    + state.current_difference.compensation
                ) / m
                state.add_square(difference * difference)
                state.count += 1

        self._n += 1

    def _point_for(self, m: int) -> AllanPoint:
        state = self._states.get(m)
        count = state.count if state is not None else 0
        tau = m * self.dt
        if count < 2:
            return AllanPoint(m, tau, None, None, count, False)
        total_squares = state.square_sum + state.square_compensation
        variance = total_squares / (2.0 * (count - 1))
        if not math.isfinite(variance):
            raise AllanValueError(
                f"allan variance overflowed finite float for m={m}; "
                "input magnitude too large for the computation"
            )
        adev = math.sqrt(variance)
        eligible = count >= MIN_DIFFERENCES and variance > 0.0
        return AllanPoint(m, tau, variance, adev, count, eligible)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _is_real(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_sequence(value: object, name: str) -> int:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(f"{name}: expected a finite sequence")
    return len(value)


def _validate_dt(dt: object) -> float:
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


def _validate_cluster_sizes(cluster_sizes: object, dt_f: float):
    _validate_sequence(cluster_sizes, "cluster_sizes")
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


def _validate_fixed_scale_input(dt: object, cluster_sizes: object):
    dt_f = _validate_dt(dt)
    return dt_f, _validate_cluster_sizes(cluster_sizes, dt_f)


def _validate_sample_values(samples: object, existing_count: int = 0) -> list[float]:
    n = _validate_sequence(samples, "samples")
    if n == 0:
        raise AllanValueError("samples: must contain at least 1 sample")
    if existing_count + n > MAX_SAMPLES:
        raise AllanValueError(
            f"samples: cumulative length {existing_count + n} exceeds interface bound "
            f"{MAX_SAMPLES}"
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
            raise AllanValueError(
                f"samples[{i}]: value must be finite (NaN/Infinity rejected)"
            )
        values[i] = fv
    return values


def _validate_stream_batch(samples: object, existing_count: int) -> list[float]:
    if _is_real(samples):
        return _validate_sample_values([samples], existing_count)
    return _validate_sample_values(samples, existing_count)


def _validate(samples: object, dt: object, cluster_sizes: object):
    dt_f, sizes = _validate_fixed_scale_input(dt, cluster_sizes)
    return _validate_sample_values(samples), dt_f, sizes


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
    stream = StreamingAllan(dt_f, sizes)
    result = stream.append(values)
    if result.fit is None:
        _fit_piecewise(result.points)
    return result
