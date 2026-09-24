"""固定速率单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集）。

本模块只实现题目契约内的数学子集：

* :func:`analyze` 用累积和计算重叠 Allan 方差（overlapping Allan variance）；
* 对差分样本数 >= 20 且方差严格为正的 (m, sigma) 点，在 log(tau)-log(sigma)
  平面上做连续（相邻区间无重叠、无缺口）分段直线最小二乘拟合；
* 分段数 1..3，每段至少 3 个点，分段罚项为模块常量 :data:`SEGMENT_PENALTY`，
  全局最优由动态规划穷举所有合法切分得到。
* 通过仅关键字参数 ``timestamps`` / ``valid`` / ``min_window_samples`` 支持
  不规则采样与缺测：按**实际时间窗口**（锚点 t0、步长 dt、窗长 m*dt 的半开
  区间）取相邻窗均值之差，任一窗有效样本覆盖不足时该差分记为缺失；均匀完整
  输入的统计量与定频路径一致。

注意：拟合斜率只是对该批离线数据 log-log 形态的经验描述，不是任何临床结论或
硬件指标保证；本模块也不宣称完整兼容任何行业标准。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "analyze",
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
        m: 聚类大小（以采样间隔 dt 为单位的窗口长度，即窗长 m*dt 秒）。
        tau: 聚类时间 m * dt，单位秒。
        variance: 重叠 Allan 方差；有效差分样本数 < 2 时为 None（无法估计）。
        adev: Allan 标准差 ``sqrt(variance)``；variance 为 None 时同为 None。
        count: 实际参与统计的**有效**相邻窗均值差分样本数，即
            ``max(0, N - 2*m + 1)`` 减去落在数据末端/因缺测被丢弃的对数；
            定频完整输入下就是 ``max(0, N - 2*m + 1)``。
        eligible_for_fit: count >= 20 且 variance > 0。
        positions: 网格上枚举过的相邻窗对位置总数（含有效对与缺失对）。
            定频路径恒等于 ``max(0, N - 2*m + 1)``；时间窗口路径恒等于
            ``max(0, floor((t_last - t0)/dt) - 2*m + 2)``，
            即 ``count + missing``。
        missing: 覆盖不足（某一窗有效样本数 < min_window_samples）或落在
            末端半窗之外而缺失的窗对数；定频完整输入下为 0。
    """

    m: int
    tau: float
    variance: float | None
    adev: float | None
    count: int
    eligible_for_fit: bool
    positions: int = 0
    missing: int = 0


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


def _validate(samples, dt, cluster_sizes, timestamps=None, valid=None,
              min_window_samples=None):
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

    # ---- 不规则采样：时间戳 ----
    ts: list[float] | None = None
    if timestamps is not None:
        if not isinstance(timestamps, Sequence) or isinstance(
            timestamps, (str, bytes, bytearray, memoryview)
        ):
            raise AllanTypeError(
                "timestamps: expected a finite sequence (list/tuple/...) of real "
                "numbers aligned with samples"
            )
        if len(timestamps) != n:
            raise AllanValueError(
                f"timestamps: length {len(timestamps)} != samples length {n}"
            )
        ts = [0.0] * n
        prev = math.nan
        for i, t in enumerate(timestamps):
            if not _is_real(t):
                raise AllanTypeError(
                    f"timestamps[{i}]: expected int/finite-float "
                    "(bool/str/complex rejected)"
                )
            ft = float(t)
            if not math.isfinite(ft):
                raise AllanValueError(
                    f"timestamps[{i}]: must be finite (NaN/Infinity rejected)"
                )
            if i > 0 and not ft > prev:
                # 相等与倒退一并拒绝（严格递增），错误可定位到后一个下标。
                raise AllanValueError(
                    f"timestamps[{i}]: timestamps must be strictly increasing, "
                    f"got {ft!r} after {prev!r}"
                )
            ts[i] = ft
            prev = ft

    # ---- 缺测标志 ----
    flags: list[bool] | None = None
    if valid is not None:
        if not isinstance(valid, Sequence) or isinstance(
            valid, (str, bytes, bytearray, memoryview)
        ):
            raise AllanTypeError(
                "valid: expected a finite sequence (list/tuple/...) of bool flags "
                "aligned with samples"
            )
        if len(valid) != n:
            raise AllanValueError(
                f"valid: length {len(valid)} != samples length {n}"
            )
        flags = [False] * n
        for i, f in enumerate(valid):
            # 有效标志只接受真正的 bool（0/1 等 int 不隐式转换，避免误传）。
            if not isinstance(f, bool):
                raise AllanTypeError(
                    f"valid[{i}]: validity flag must be a bool "
                    f"(int 0/1 is not accepted implicitly)"
                )
            flags[i] = f

    # ---- 每窗最少有效样本数 ----
    min_count: int | None = None
    if min_window_samples is not None:
        if isinstance(min_window_samples, bool) or not isinstance(
            min_window_samples, int
        ):
            raise AllanTypeError(
                "min_window_samples: expected an int (bool/float rejected)"
            )
        if min_window_samples < 1:
            raise AllanValueError(
                "min_window_samples: must be >= 1, got "
                f"{min_window_samples}"
            )
        min_count = min_window_samples

    return values, dt_f, sizes, ts, flags, min_count


# ---------------------------------------------------------------------------
# 重叠 Allan 方差
#
# 两条计算路径：
#   * _allan_points_uniform —— 定频完整数据，前缀和一次构造、各 m 复用；
#   * _allan_points_windowed —— 给定时间戳/有效标志时按实际时间窗口取窗内
#     有效样本均值，覆盖不足的窗对记为缺失。
# ---------------------------------------------------------------------------


def _centered_cumsum(values: Sequence[float]):
    """去全局偏置后返回 (centered, prefix_cumsum)；均值用 fsum 精确求和。

    相邻窗均值之差对整体加常数数学不变，去偏置可避免大基线信号在差分三项
    相减时灾难性抵消。
    """
    n = len(values)
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
    cum = [0.0] * (n + 1)
    acc = 0.0
    for i, v in enumerate(centered):
        acc += v
        cum[i + 1] = acc
    return centered, cum


def _allan_points_uniform(values: Sequence[float], dt: float, sizes: Sequence[int]):
    """定频完整输入路径（旧行为）：按样本下标分块，全部窗对都完整覆盖。"""
    n = len(values)
    _centered, cm = _centered_cumsum(values)

    def compute(m: int) -> AllanPoint:
        count = max(0, n - 2 * m + 1)
        tau = m * dt
        # 定频完整数据没有“缺失”的窗对；count<2 只是无法做无偏估计。
        if count < 2:
            # 实际差分样本数不会为负：2m>N 时为 0；count=1 时无法做无偏估计。
            return AllanPoint(m, tau, None, None, count, False, count, 0)
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
        if not math.isfinite(q):
            raise AllanValueError(
                f"allan variance overflowed finite float for m={m}; "
                "input magnitude too large for the computation"
            )
        variance = q / (2.0 * (count - 1))
        adev = math.sqrt(variance)
        eligible = count >= MIN_DIFFERENCES and variance > 0.0
        return AllanPoint(m, tau, variance, adev, count, eligible, count, 0)

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


def _allan_points_windowed(values, dt: float, sizes, ts, flags, min_required_default):
    """按实际时间窗口取相邻窗均值之差（不规则采样 / 缺测路径）。

    窗口端点规则（固定）：以首时间戳 t0 为网格锚点，第 k 对窗口为半开区间
        W1(k) = [t0 + k*dt,       t0 + (k+m)*dt)
        W2(k) = [t0 + (k+m)*dt,   t0 + (k+2m)*dt)
    恰落在边界上的样本归入右侧窗口（bisect_left 语义）。枚举到右窗最后一个
    网格格点起点 t0+(k+2m-1)*dt 不晚于最后一个时间戳为止（均匀完整输入下
    恰好还原定频路径的 N-2m+1 对）；物理上仍被截断的末端半窗不补零、不外推，
    交给覆盖阈值判缺失。窗内只统计 valid=True 的样本；任一窗有效样本数
    < min_window_samples（默认 m）时该差分记为缺失（missing），不计入 count。
    """
    n = len(values)

    # 仅有效样本参与：构造其时间戳与（去偏置后的）前缀和，供 O(log n) 取窗均值。
    if flags is None:
        use_idx = list(range(n))
    else:
        use_idx = [i for i, f in enumerate(flags) if f]

    if use_idx:
        valid_vals = [values[i] for i in use_idx]
        _centered, _ = _centered_cumsum(valid_vals)
        tv = [ts[i] for i in use_idx]
        cv = [0.0] * (len(use_idx) + 1)
        acc = 0.0
        for j, v in enumerate(_centered):
            acc += v
            cv[j + 1] = acc
    else:
        # 整段缺测：没有任何可用样本，所有窗对均缺失，不需要构造均值。
        tv = []
        cv = [0.0]

    t0 = ts[0]
    t_last = ts[n - 1]

    def compute(m: int) -> AllanPoint:
        tau = m * dt
        req = m if min_required_default is None else min_required_default
        # 末端规则（与定频路径严格兼容）：枚举 k=0..k_last，使右窗的最后一个
        # 网格格点起点 t0+(k+2m-1)*dt 不晚于 t_last —— 即右窗在数据范围内
        # 至少“开始了它的最后一格”；物理上仍被截断的半窗交给覆盖阈值
        # （min_window_samples）判缺失，末端之外不补零、不外推。
        # floor 前加 1e-9 相对容差，吸收 i*dt 网格的 binary64 舍入
        # （如 3.99/0.01 = 398.9999...）；容差远小于半格 0.5，不会吞掉真实半窗。
        span_raw = (t_last - t0) / dt
        span_steps = math.floor(span_raw + 1e-9 * max(1.0, abs(span_raw)))
        positions = max(0, span_steps - 2 * m + 2)

        valid_sq: list[float] = []
        for k in range(positions):
            # 边界一律直接写成 t0 + j*dt（一次乘法），避免 e0+tau 的二次
            # 舍入在均匀网格上把恰在边界的样本错分半位。
            e0 = t0 + k * dt
            e1 = t0 + (k + m) * dt
            e2 = t0 + (k + 2 * m) * dt
            # 半开区间：边界样本归右窗（bisect_left）。
            i0 = bisect_left(tv, e0)
            i1 = bisect_left(tv, e1)
            i2 = bisect_left(tv, e2)
            n1 = i1 - i0
            n2 = i2 - i1
            if n1 < req or n2 < req:
                continue
            mean1 = (cv[i1] - cv[i0]) / n1
            mean2 = (cv[i2] - cv[i1]) / n2
            valid_sq.append((mean2 - mean1) ** 2)

        count = len(valid_sq)
        missing = positions - count
        if count < 2:
            return AllanPoint(m, tau, None, None, count, False, positions, missing)
        try:
            q = math.fsum(valid_sq)
        except OverflowError:
            q = math.inf
        if not math.isfinite(q):
            raise AllanValueError(
                f"allan variance overflowed finite float for m={m}; "
                "input magnitude too large for the computation"
            )
        variance = q / (2.0 * (count - 1))
        adev = math.sqrt(variance)
        eligible = count >= MIN_DIFFERENCES and variance > 0.0
        return AllanPoint(m, tau, variance, adev, count, eligible, positions, missing)

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


def analyze(samples, dt, cluster_sizes, *, timestamps=None, valid=None,
            min_window_samples=None) -> AllanResult:
    """计算重叠 Allan 方差并做 log-log 连续分段直线分解。

    参数:
        samples: 单轴角速率序列，长度 1..8192，元素为 int/float 且有限；
            bool 按类型错误拒绝。
        dt: 名义采样周期（秒），int/float、有限且 > 0；bool 拒绝。同时是
            时间窗口网格的步长与聚类时间换算单位（tau=m*dt）。
        cluster_sizes: 聚类大小 m 的有限序列（list/tuple/range），每个为
            真正的 int（bool、float 拒绝）且 m >= 1，长度 <= 8192；允许重复、
            非单调，输出点顺序与输入一致，拟合前去重升序。
        timestamps: 仅关键字，可选。与 samples 等长的严格递增有限实数序列
            （秒，单调，不要求均匀）；不给定时走定频路径（旧行为）。
        valid: 仅关键字，可选。与 samples 等长的真正 bool 序列；False 表示
            该样本缺测，窗内统计时跳过。可与 timestamps 组合使用。不给定时
            视为全部有效。
        min_window_samples: 仅关键字，可选。每个时间窗口要求的最少有效样本
            数，真正的 int 且 >= 1；默认等于 m（即每个 m*dt 窗至少 m 个有效
            样本才算充分覆盖）。

    时间窗口规则（``timestamps``/``valid`` 存在时）：
        网格锚点为首时间戳 t0，第 k 对窗口为半开区间
        ``[t0+k*dt, t0+(k+m)*dt)`` 与 ``[t0+(k+m)*dt, t0+(k+2m)*dt)``；
        恰落在边界上的样本归右窗。只枚举右窗末端不晚于最后时间戳的窗对，
        末端之外的半窗不补零、不外推。任一窗有效样本数 < min_window_samples
        时该差分记为缺失；``count`` 只计有效差分，``missing`` 为被跳过的对数，
        ``positions = count + missing``。

    返回:
        AllanResult。无法拟合（可拟合点 < 3，例如常量零方差或覆盖不足）时
        抛出 FitInfeasibleError，其 .points 附带已算好的逐 m 统计。

    异常:
        AllanTypeError / AllanValueError: 非法输入，信息定位到具体参数与下标。
        FitInfeasibleError: 有效统计点不足，证明无满足约束的分段方案。
        SearchBudgetExhaustedError: 网格/搜索预算耗尽（公开边界内不可达）。
    """
    values, dt_f, sizes, ts, flags, min_req = _validate(
        samples, dt, cluster_sizes, timestamps, valid, min_window_samples
    )
    if ts is None and flags is None:
        points = _allan_points_uniform(values, dt_f, sizes)
    else:
        if ts is None:
            ts = [i * dt_f for i in range(len(values))]
        points = _allan_points_windowed(values, dt_f, sizes, ts, flags, min_req)
    fit = _fit_piecewise(points)
    return AllanResult(tuple(points), fit, dt_f, len(values))
