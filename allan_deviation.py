"""单轴陀螺仪角速率记录的重叠 Allan 误差分解（离线子集）。

本模块只实现题目契约内的数学子集，提供两种等价的时间轴模式：

* **定频模式**：:func:`analyze(samples, dt, cluster_sizes)`，用累积和计算重叠
  Allan 方差（overlapping Allan variance），聚类大小以样本数 m 给出；
* **时间戳模式**：:func:`analyze(samples, timestamps=..., valid=...,
  cluster_durations=...)`，接受每个样本的实际时间戳与有效标志，聚类大小以
  **时间窗长（秒）**给出；相邻等长时窗均值之差按样本实际落入的时窗归属计算，
  任一半窗有效样本覆盖不足的相邻窗对记为缺失。时间戳严格等间隔且全部有效时，
  时间戳模式与定频模式的统计结果一致（仅有 binary64 舍入级差别）。

两种模式下，均对差分样本数 >= 20 且方差严格为正的 (tau, sigma) 点，在
log(tau)-log(sigma) 平面上做连续（相邻区间无重叠、无缺口）分段直线最小二乘
拟合；分段数 1..3，每段至少 3 个点，分段罚项为模块常量
:data:`SEGMENT_PENALTY`，全局最优由动态规划穷举所有合法切分得到。

注意：拟合斜率只是对该批离线数据 log-log 形态的经验描述，不是任何临床结论或
硬件指标保证；本模块也不宣称完整兼容任何行业标准。
"""

from __future__ import annotations

import math
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
    "MIN_WINDOW_FRACTION",
]

# ---------------------------------------------------------------------------
# 公开常量（契约边界与拟合参数）
# ---------------------------------------------------------------------------

MAX_SAMPLES = 8192
"""samples 的最大长度（接口边界）。"""

MAX_CLUSTER_SIZES = 8192
"""cluster_sizes / cluster_durations 的最大长度（接口边界）。"""

MIN_DIFFERENCES = 20
"""只有相邻窗均值差分样本数 >= 该值的窗才参与 log-log 拟合。"""

MAX_SEGMENTS = 3
"""分段直线最多段数。"""

MIN_SEGMENT_POINTS = 3
"""每段至少包含的拟合点数。"""

SEGMENT_PENALTY = 1.0
"""每增加一段在总 SSE 上追加的固定罚项（单位：log(sigma) 残差平方）。

与数据规模无关的固定常量，使 "原始 SSE 改善量 > 罚项" 才接受更碎的切分。
"""

MIN_WINDOW_FRACTION = 0.5
"""时间戳模式：时窗至少包含该比例的有效样本（按满窗周期点数算）才可用。

窗内有效样本数必须 ``>= ceil(fraction * w/dt_nom)``，其中 dt_nom 为相邻
有效样本时间间隔的中位数（稳健标称周期）；不足该覆盖度的时窗不参与均值与
差分（该相邻窗对记为缺失）。比例固定为 1/2，即“至少半窗有有效样本”。
"""

_GRID_TOL = 1e-9
"""样本吸附到标称整数网格的相对容差（按坐标量级缩放）。

均匀采样经 binary64 运算后的网格位置偏差约为机器 epsilon 量级，取 1e-9 足以
吸收该偏差、又远小于半格，不会把真正落在格子中间的样本错配到相邻格。
"""

_M_INTEGER_TOL = 1e-6
"""判定“窗长是标称周期整数倍”的相对容差。

比样本吸附容差宽：当时间戳绝对值很大（如 epoch 秒 1.7e9）时，相邻间隔的
中位数本身带 ~1e-6 相对的 binary64 舍入，用户按真实周期整数倍给出的窗长
折算 M 后会偏开整数同样量级。真实的非整数倍窗长至少偏离半格（0.5），远
大于该容差，故不会把真正的分数窗错判成整数窗。
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
    """单个聚类窗的 Allan 统计结果。

    属性:
        m: 聚类大小。定频模式为样本数（int）；时间戳模式为窗长（秒，float）。
        tau: 聚类时间。定频模式为 ``m * dt``；时间戳模式即窗长（秒）。
        variance: 重叠 Allan 方差；有效相邻窗对 < 2 时为 None（无法估计），
            包括所有窗对都因覆盖不足而缺失的情况。
        adev: Allan 标准差 ``sqrt(variance)``；variance 为 None 时同为 None。
        count: 实际参与统计的相邻等长窗均值差分样本数（双侧覆盖均达标）。
        eligible_for_fit: count >= 20 且 variance > 0。
        expected_count: 按固定端点规则考察的相邻窗对总数（含缺失）；
            定频模式等于 count（无缺失概念）。
        missing_count: 因任一侧时窗有效样本覆盖不足而跳过的窗对数，
            即 ``expected_count - count``。
    """

    m: "int | float"
    tau: float
    variance: "float | None"
    adev: "float | None"
    count: int
    eligible_for_fit: bool
    expected_count: int = 0
    missing_count: int = 0


@dataclass(frozen=True)
class Breakpoint:
    """两段之间的断点，定义为右侧段第一个拟合点所在位置。

    m/tau 在定频模式分别为样本数与秒；时间戳模式 m==tau（均为窗长秒数）。
    """

    m: "int | float"
    tau: float


@dataclass(frozen=True)
class Segment:
    """一段 log(tau)-log(sigma) 最小二乘直线 ``y = slope*x + intercept``。"""

    slope: float
    intercept: float
    m_start: "int | float"
    m_end: "int | float"
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
    mode: str = "fixed"
    """``"fixed"``（定频 + cluster_sizes）或 ``"timestamp"``（时间戳 + 窗长）。"""
    n_valid: int = 0
    """时间戳模式下有效样本数；定频模式等于 n_samples。"""


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


def _validate_timestamped(samples: object, timestamps: object, valid: object,
                          cluster_durations: object):
    """时间戳模式输入校验，返回 (values, ts, valid_mask, durations)。

    与定频模式相同的类型/取值口径：int/float 且有限、bool 一律拒绝；
    时间戳额外要求严格递增，窗长要求 > 0。
    """
    if not isinstance(samples, Sequence) or isinstance(
        samples, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "samples: expected a finite sequence (list/tuple/...) of real numbers"
        )
    if not isinstance(timestamps, Sequence) or isinstance(
        timestamps, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "timestamps: expected a finite sequence (list/tuple/...) of real numbers"
        )

    n = len(samples)
    if n == 0:
        raise AllanValueError("samples: must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise AllanValueError(
            f"samples: length {n} exceeds interface bound {MAX_SAMPLES}"
        )
    if len(timestamps) != n:
        raise AllanValueError(
            f"timestamps: length {len(timestamps)} != samples length {n}"
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

    ts: list[float] = [0.0] * n
    prev = -math.inf
    for i, t in enumerate(timestamps):
        if not _is_real(t):
            raise AllanTypeError(
                f"timestamps[{i}]: expected int/finite-float (bool/str rejected)"
            )
        try:
            ft = float(t)
        except OverflowError:
            raise AllanValueError(
                f"timestamps[{i}]: {t!r} cannot be represented as finite float"
            ) from None
        if not math.isfinite(ft):
            raise AllanValueError(
                f"timestamps[{i}]: timestamp must be finite (NaN/Infinity rejected)"
            )
        if i > 0 and not ft > prev:
            raise AllanValueError(
                f"timestamps[{i}]: timestamps must be strictly increasing "
                f"(got {ft!r} after {prev!r})"
            )
        ts[i] = ft
        prev = ft

    if valid is None:
        mask = [True] * n
    else:
        if not isinstance(valid, Sequence) or isinstance(
            valid, (str, bytes, bytearray, memoryview)
        ):
            raise AllanTypeError(
                "valid: expected a finite sequence of bool/0/1 flags"
            )
        if len(valid) != n:
            raise AllanValueError(
                f"valid: length {len(valid)} != samples length {n}"
            )
        mask = [False] * n
        for i, f in enumerate(valid):
            if isinstance(f, bool):
                mask[i] = f
            elif isinstance(f, int) and f in (0, 1):
                mask[i] = bool(f)
            else:
                raise AllanTypeError(
                    f"valid[{i}]: validity flag must be bool or 0/1 (got {f!r})"
                )

    if not isinstance(cluster_durations, Sequence) or isinstance(
        cluster_durations, (str, bytes, bytearray, memoryview)
    ):
        raise AllanTypeError(
            "cluster_durations: expected a finite sequence (list/tuple/range) of "
            "positive window durations (seconds)"
        )
    k_durations = len(cluster_durations)
    if k_durations == 0:
        raise AllanValueError("cluster_durations: must contain at least 1 duration")
    if k_durations > MAX_CLUSTER_SIZES:
        raise AllanValueError(
            f"cluster_durations: length {k_durations} exceeds interface bound "
            f"{MAX_CLUSTER_SIZES}"
        )
    durations: list[float] = [0.0] * k_durations
    for i, w in enumerate(cluster_durations):
        if not _is_real(w):
            raise AllanTypeError(
                f"cluster_durations[{i}]: expected int/finite-float (bool rejected)"
            )
        try:
            fw = float(w)
        except OverflowError:
            raise AllanValueError(
                f"cluster_durations[{i}]: {w!r} cannot be represented as finite float"
            ) from None
        if not math.isfinite(fw):
            raise AllanValueError(
                f"cluster_durations[{i}]: duration must be finite (NaN/Infinity rejected)"
            )
        if fw <= 0.0:
            raise AllanValueError(
                f"cluster_durations[{i}]: duration must be > 0, got {w!r}"
            )
        durations[i] = fw

    return values, ts, mask, durations


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
            # 定频模式没有缺失概念：考察窗对数 == 实际窗对数。
            return AllanPoint(m, tau, None, None, max(0, count), False,
                              expected_count=max(0, count), missing_count=0)
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
        return AllanPoint(m, tau, variance, adev, count, eligible,
                          expected_count=count, missing_count=0)

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
# 时间戳模式：固定端点时窗 + 有效标志覆盖判定
# ---------------------------------------------------------------------------


def _median(sorted_vals: Sequence[float]) -> float:
    k = len(sorted_vals)
    mid = k // 2
    if k % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _timestamped_points(values: Sequence[float], ts: Sequence[float],
                        valid: Sequence[bool], durations: Sequence[float]):
    """按实际时间戳把样本分入固定端点时窗，逐窗长计算 Allan 统计。

    标称网格与端点规则（固定）：
      标称步长 s 取相邻**有效**样本时间间隔的中位数，t0 为首个有效样本时间
      戳。每个有效样本按 q=(t-t0)/s 吸附到整数网格单元（与整数的差在
      ``_GRID_TOL`` 内时吸附，否则向下取整），单元号非递减。对窗长 w（折合
      M=w/s 个标称单元），时窗在规则时间网格上按一个步长滑动，第 g 个时窗
      覆盖单元 [g, g+M) 内的全部有效样本；配对差分为起点相隔**一个窗长**的
      两个时窗（与重叠 Allan 方差一致）：

          d_g = mean(单元 [g+M, g+2M)) - mean(单元 [g, g+M))

      g=0,1,...；时窗对并集越过记录末端网格（span+s）后不再考察，考察窗对
      总数记为 expected_count。时间戳严格等间隔、全部有效且 w 为 s 的整数倍
      时，时窗与定频模式的样本块逐块重合，统计结果一致。非整数倍窗长时，跨
      网格线的边界按同一吸附规则处理（贴左边界计入、贴右边界计出）。
    覆盖规则（固定）：
      时窗内有效样本数不足 ``need = ceil(MIN_WINDOW_FRACTION * w/s)``（至少
      1）即该侧时窗缺失；窗对任一侧缺失则跳过，计入 missing_count。
    统计：
      variance = sum(d^2) / (2*(count-1))，count 为双侧均达标的窗对数。
    """
    n = len(values)

    # 仅有效样本参与时间轴与均值；无效样本不影响间隔中位数，也不占窗。
    vidx = [i for i in range(n) if valid[i]]
    nv = len(vidx)
    vts = [ts[i] for i in vidx]

    if nv < 2:
        # 不足两个有效样本，无法形成任何间隔与窗对；所有窗长统一缺失。
        return [
            AllanPoint(float(w), float(w), None, None, 0, False,
                       expected_count=0, missing_count=0)
            for w in durations
        ], math.nan, nv

    gaps = sorted(vts[i] - vts[i - 1] for i in range(1, nv))
    step = _median(gaps)
    if not step > 0.0 or not math.isfinite(step):
        # 严格递增的有限时间戳使相邻间隔必为正有限；保留防护。
        raise AllanValueError(
            "timestamps: cannot derive a positive nominal period from valid samples"
        )

    # 去恒定偏置（与定频模式同一口径）：相邻窗均值之差对整体加常数不变。
    try:
        mean = math.fsum(values[i] for i in vidx) / nv
    except OverflowError:
        raise AllanValueError(
            "sample magnitudes overflow even the exact mean computation; "
            "inputs too large for the float64 computation"
        ) from None
    if not math.isfinite(mean):
        raise AllanValueError("sample mean is not finite; inputs out of float64 range")
    vvals = [values[i] - mean for i in vidx]

    t0 = vts[0]

    def snap_cell(q: float) -> int:
        """把样本位置 q（单元坐标）吸附为整数单元号。"""
        c = math.floor(q)
        if (c + 1) - q <= _GRID_TOL * max(1.0, c + 1):
            return c + 1
        return c

    # 每个有效样本的（非递减）网格单元号。
    cells = [snap_cell((t - t0) / step) for t in vts]
    if any(cells[i] < cells[i - 1] for i in range(1, nv)):
        # snap 容差远小于半格，非递减应恒成立；保留防护。
        raise AllanValueError("timestamps: failed to map samples onto a monotonic grid")
    cmax = cells[-1]

    # 快路径判定：有效样本恰好占满从 0 起的连续整数网格（均匀采样、无缺测、
    # 无跨格抖动）。此时整数倍窗长的时窗与定频模式的等长样本块逐块重合，
    # 直接用累积和公式，复杂度与定频模式相同。
    regular = cmax == nv - 1 and all(c == i for i, c in enumerate(cells))

    # 有效样本前缀和（顺序与 vvals 一致；cells 非递减，下标即样本序）。
    pc_sum = [0.0] * (nv + 1)
    acc = 0.0
    for i, v in enumerate(vvals):
        acc += v
        pc_sum[i + 1] = acc
    cm = pc_sum

    def grid_line(q: float) -> int:
        """单元坐标 q 处的网格线：容差内贴整数网格线则吸附，否则向上取整。

        时窗 [o, o+M) 含单元 c 当且仅当 o <= c < o+M；左/右边界都用本函数
        换算成整数单元线（含下界、不含上界）。
        """
        r = round(q)
        if abs(q - r) <= _GRID_TOL * max(1.0, abs(r)):
            return r
        return math.ceil(q)

    def compute(w: float) -> AllanPoint:
        m_raw = w / step
        mr = round(m_raw)
        m_is_int = abs(m_raw - mr) <= _M_INTEGER_TOL * max(1.0, abs(mr))
        m_int = int(mr)

        if regular and m_is_int:
            # 快路径：时窗即连续 m 个有效样本，与定频模式完全一致。
            m = m_int
            count = nv - 2 * m + 1
            if count < 2:
                return AllanPoint(w, w, None, None, max(0, count), False,
                                  expected_count=max(0, count), missing_count=0)
            try:
                q = math.fsum(
                    ((cm[g + 2 * m] - 2.0 * cm[g + m] + cm[g]) / m) ** 2
                    for g in range(count)
                )
            except OverflowError:
                q = math.inf
            variance = q / (2.0 * (count - 1))
            if not math.isfinite(variance):
                raise AllanValueError(
                    f"allan variance overflowed finite float for window={w}s; "
                    "input magnitude too large for the computation"
                )
            adev = math.sqrt(variance)
            eligible = count >= MIN_DIFFERENCES and variance > 0.0
            return AllanPoint(w, w, variance, adev, count, eligible,
                              expected_count=count, missing_count=0)

        # 标称满窗点数；need 至少为 1，保证空窗必然缺失。
        need = max(1, math.ceil(MIN_WINDOW_FRACTION * m_raw - 1e-12))

        # g+2M 不得越过末端网格线 cmax+1（均匀时即 g <= N-2M）。
        if m_is_int:
            g_max = cmax + 1 - 2 * m_int
        else:
            x = cmax + 1 - 2.0 * m_raw
            xr = round(x)
            if abs(x - xr) <= _GRID_TOL * max(1.0, abs(xr)):
                g_max = xr
            else:
                g_max = math.floor(x)
        expected = max(0, g_max + 1)
        if expected == 0:
            return AllanPoint(w, w, None, None, 0, False,
                              expected_count=0, missing_count=0)

        # 第 g 对时窗的三条边界（单元线）随 g 单调不减；整数 M 时中/右边界
        # 就是 g+M、g+2M，免去吸附运算。三个在 cells 上只前进的指针总计
        # O(nv + expected)。
        p_lo = 0
        p_mid = 0
        p_hi = 0

        def advance(p: int, line: int) -> int:
            while p < nv and cells[p] < line:
                p += 1
            return p

        diffs: list[float] = []
        missing = 0
        for g in range(expected):
            line_lo = g
            if m_is_int:
                line_mid = g + m_int
                line_hi = g + 2 * m_int
            else:
                line_mid = grid_line(g + m_raw)
                line_hi = grid_line(g + 2.0 * m_raw)
            p_lo = advance(p_lo, line_lo)
            p_mid = advance(p_mid, line_mid)
            p_hi = advance(p_hi, line_hi)
            c0 = p_mid - p_lo
            c1 = p_hi - p_mid
            if c0 < need or c1 < need:
                missing += 1
                continue
            d = (cm[p_hi] - cm[p_mid]) / c1 - (cm[p_mid] - cm[p_lo]) / c0
            diffs.append(d)

        count = len(diffs)
        if count < 2:
            return AllanPoint(w, w, None, None, count, False,
                              expected_count=expected, missing_count=missing)
        try:
            q = math.fsum(d * d for d in diffs)
        except OverflowError:
            q = math.inf
        variance = q / (2.0 * (count - 1))
        if not math.isfinite(variance):
            raise AllanValueError(
                f"allan variance overflowed finite float for window={w}s; "
                "input magnitude too large for the computation"
            )
        adev = math.sqrt(variance)
        eligible = count >= MIN_DIFFERENCES and variance > 0.0
        return AllanPoint(w, w, variance, adev, count, eligible,
                          expected_count=expected, missing_count=missing)

    cache: dict[float, AllanPoint] = {}
    points: list[AllanPoint] = []
    for w in durations:
        pt = cache.get(w)
        if pt is None:
            pt = compute(w)
            cache[w] = pt
        points.append(pt)
    return points, step, nv


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


def analyze(samples, dt=None, cluster_sizes=None, *, timestamps=None, valid=None,
            cluster_durations=None) -> AllanResult:
    """计算重叠 Allan 方差并做 log-log 连续分段直线分解。

    支持两种时间轴模式（互斥）：

    **定频模式**（原有调用，位置参数）::

        analyze(samples, dt, cluster_sizes)

    参数:
        samples: 定频单轴角速率序列，长度 1..8192，元素为 int/float 且有限；
            bool 元素按类型错误拒绝。
        dt: 采样周期（秒），int/float、有限且 > 0；bool 拒绝。
        cluster_sizes: 聚类大小 m（样本数）的有限序列（list/tuple/range），
            每个为真正的 int（bool、float 拒绝）且 m >= 1，长度 <= 8192；允许
            重复、非单调，输出点顺序与输入一致，拟合前去重升序。

    **时间戳模式**（关键字参数）::

        analyze(samples, timestamps=ts, valid=flags,
                cluster_durations=durations)

    参数:
        samples: 同定频模式的取值口径。
        timestamps: 每个样本的时间戳（秒），与 samples 等长、int/float 有限、
            严格递增；bool 拒绝。
        valid: 可选的有效标志序列（与 samples 等长），元素为 bool 或 0/1；
            缺省视为全部有效。无效样本既不参与时窗均值，也不参与标称周期估计。
        cluster_durations: 聚类时窗长度（秒）的有限序列，元素 int/float、有限
            且 > 0；允许重复、非单调。输出 ``AllanPoint.m`` 即窗长（float 秒）。

    时间戳模式下 ``result.dt`` 回显由相邻有效样本间隔中位数得到的标称周期，
    ``result.mode == "timestamp"``，``result.n_valid`` 为有效样本数。时窗按
    标称周期在固定时间网格上滑动（左闭右开），窗内有效样本不足半窗标称点数
    （``ceil(0.5*w/dt_nom)``）的相邻窗对记为缺失（``missing_count``），不参与
    均值与差分；有效窗对不足 2 时该窗长 ``variance=None``。

    返回:
        AllanResult。无法拟合（合格点 < 3，例如常量零方差、整段缺测或短数据）
        时抛出 FitInfeasibleError，其 .points 附带已算好的逐窗统计。

    异常:
        AllanTypeError / AllanValueError: 非法输入或混用两种模式的参数，
            信息定位到具体参数与下标。
        FitInfeasibleError: 证明无满足约束的分段方案。
        SearchBudgetExhaustedError: 搜索预算耗尽（公开输入边界内不可达）。
    """
    timestamp_mode = timestamps is not None or cluster_durations is not None
    if timestamp_mode:
        if dt is not None or cluster_sizes is not None:
            raise AllanValueError(
                "timestamp mode (timestamps=/cluster_durations=) cannot be combined "
                "with fixed-rate arguments dt/cluster_sizes"
            )
        if timestamps is None:
            raise AllanValueError(
                "cluster_durations requires timestamps= (timestamp mode)"
            )
        if cluster_durations is None:
            raise AllanValueError(
                "timestamps requires cluster_durations= (timestamp mode)"
            )
        values, ts, mask, durations = _validate_timestamped(
            samples, timestamps, valid, cluster_durations
        )
        points, dt_echo, n_valid = _timestamped_points(
            values, ts, mask, durations
        )
        fit = _fit_piecewise(points)
        return AllanResult(tuple(points), fit, dt_echo, len(values),
                           mode="timestamp", n_valid=n_valid)

    if dt is None or cluster_sizes is None:
        raise AllanValueError(
            "fixed-rate mode requires analyze(samples, dt, cluster_sizes)"
        )
    if valid is not None:
        raise AllanValueError(
            "valid= requires timestamps= (only available in timestamp mode)"
        )
    values, dt_f, sizes = _validate(samples, dt, cluster_sizes)
    points = _allan_points(values, dt_f, sizes)
    fit = _fit_piecewise(points)
    return AllanResult(tuple(points), fit, dt_f, len(values),
                       mode="fixed", n_valid=len(values))
