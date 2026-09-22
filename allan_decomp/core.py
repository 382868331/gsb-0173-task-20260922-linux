"""
allan_decomp.core
=================

固定速率单轴陀螺仪（角速率采样）的重叠 Allan 方差分解。

本模块只实现本题契约定义的有限数学子集：
  * 重叠 Allan 方差（相邻等长块均值之差，滑动 1 点）；
  * log10-log10 连续分段直线辨识（最多 3 段，每段至少 3 个点）；
  * 穷举动态规划求全局最优分段 + 分段罚项。

它不是任何行业标准（如 IEEE 1139）的完整兼容实现；经验斜率只是对合成/离线
数据的统计描述，不构成硬件指标或临床结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 公开契约常量
# ---------------------------------------------------------------------------

MAX_SAMPLES = 8192
"""输入样本点数上限（接口边界）。"""

MAX_SEGMENTS = 3
"""分段直线的最大段数。"""

MIN_POINTS_PER_SEGMENT = 3
"""每段至少包含的拟合点数。"""

MIN_DIFFERENCES = 20
"""参与拟合的最小相邻块差分样本数。"""

DEFAULT_SEARCH_BUDGET = 50_000_000
"""
动态规划候选段评估次数的默认预算。

在 N <= 8192 的接口边界内，可拟合点数 n <= (N - 19) // 2 = 4086；融合 DP
对每个候选区间只评估一次，总数约 n + n^2/2 < 8.4e6，因此默认预算一定覆盖
最大合法输入；该参数只在调用方主动收紧时才可能触发 SearchBudgetExhausted。
"""

_REASON_NO_ESTIMATOR = "no_estimator"
_REASON_TOO_FEW_DIFFERENCES = "too_few_differences"
_REASON_ZERO_VARIANCE = "zero_variance"
_REASON_INSUFFICIENT_POINTS = "insufficient_points"


# ---------------------------------------------------------------------------# 错误语义
# ---------------------------------------------------------------------------

class AllanAnalysisError(ValueError):
    """本模块所有错误的基类（仍是 ValueError）。"""


class InvalidInputError(AllanAnalysisError):
    """
    非法输入。错误可定位到参数名、（如有）下标与原因。

    属性
    ----
    parameter : str
        参数名，如 "samples" / "dt" / "cluster_sizes" / "segment_penalty"。
    index : int | None
        序列类参数中的下标位置；非序列参数为 None。
    reason : str
        机器可读的原因短语。
    bad_value : object
        触发错误的原始值（不静默纠正、不强制转换）。
    """

    def __init__(self, parameter: str, bad_value: object, reason: str,
                 index: Optional[int] = None) -> None:
        self.parameter = parameter
        self.bad_value = bad_value
        self.reason = reason
        self.index = index
        location = f"{parameter}[{index}]" if index is not None else parameter
        super().__init__(
            f"invalid value for {location}: {reason} (got {bad_value!r})"
        )


class SearchBudgetExhausted(AllanAnalysisError):
    """
    动态规划搜索预算耗尽。

    这与“证明无解”严格分开：拟合无解时 analyze 正常返回
    ``fit.feasible is False`` 并给出 reason；只有搜索在找到/证明任何结果
    之前用尽调用方给定的评估预算时才抛出本异常。analyze 是无状态纯计算，
    异常发生时不会产生任何部分输出或外部变更。
    """

    def __init__(self, evaluations: int, budget: int, n_points: int) -> None:
        self.evaluations = evaluations
        self.budget = budget
        self.n_points = n_points
        super().__init__(
            f"piecewise-fit search budget exhausted after {evaluations} "
            f"candidate evaluations (budget={budget}, fit_points={n_points}); "
            "no optimum was returned and no state was modified"
        )


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterPoint:
    """
    单个簇长 m 的 Allan 方差结果。

    字段
    ----
    m : int
        簇长（样本数），m >= 1。
    tau : float
        簇时长 tau = m * dt。
    avar : float | None
        重叠 Allan 方差；当 n_differences == 0（2m > N）时无估计量，为 None。
    adev : float | None
        Allan 标准差 sqrt(avar)；avar 为 None 时亦为 None；零方差为 0.0。
    n_differences : int
        实际参与平均的相邻等长块差分样本数，n = N - 2m + 1（2m <= N），
        否则为 0。
    """

    m: int
    tau: float
    avar: Optional[float]
    adev: Optional[float]
    n_differences: int


@dataclass(frozen=True)
class ExcludedPoint:
    """未进入 log-log 拟合的簇长及其排除原因。"""

    m: int
    n_differences: int
    reason: str  # no_estimator / too_few_differences / zero_variance


@dataclass(frozen=True)
class PiecewiseFit:
    """
    log10(avar) ~ log10(tau) 的连续分段直线拟合结果。

    feasible 为 False 时表示在本契约约束下**证明无解**（已穷举前置条件），
    此时 reason 取：
      * "too_few_differences"：没有任何 m 满足 n_differences >= 20；
      * "zero_variance"：满足差分样本数要求的点全部 avar == 0；
      * "insufficient_points"：正方差拟合点少于每段最少点数 3。
    这与 SearchBudgetExhausted（搜索预算耗尽）是两条完全不同的路径。

    可行时
    ------
    starts / ends : 每段在拟合点序列中的闭区间下标；
    knot_x : 内部连接点（断点）的 x 坐标 log10(tau)；
    slopes : 每段 log-log 斜率；
    intercepts : 每段等效截距（折线在 knot 处连续）；
    partition_sse : 动态规划目标中的各段独立 OLS 残差平方和；
    residual_sse : 返回的连续折线模型在全部拟合点上的残差平方和；
    objective : partition_sse + (段数-1) * segment_penalty（DP 最小化目标）；
    residuals : 与拟合点对齐的逐点残差 y - y_hat；
    evaluations : DP 实际消耗的候选段评估次数。
    """

    feasible: bool
    reason: Optional[str]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    knot_x: tuple[float, ...]
    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]
    partition_sse: Optional[float]
    residual_sse: Optional[float]
    objective: Optional[float]
    segment_penalty: float
    residuals: tuple[float, ...]
    evaluations: int
    budget: int


@dataclass(frozen=True)
class Segment:
    """分段在物理量（簇长/簇时长）上的区间与经验斜率。"""

    start_index: int
    end_index: int
    m_start: int
    m_end: int
    tau_start: float
    tau_end: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class Breakpoint:
    """两段之间的连续连接点（断点），位于左段最后一个拟合点的 x 坐标。"""

    left_segment_end_index: int
    m: int
    tau: float
    x: float
    y: float


@dataclass(frozen=True)
class AnalysisResult:
    """analyze() 的完整返回。"""

    n_samples: int
    dt: float
    curve: tuple[ClusterPoint, ...]
    """与 cluster_sizes 输入顺序一一对应（保留重复 m）。"""

    fit_points: tuple[ClusterPoint, ...]
    """去重、按 m 升序、n_differences >= 20 且 avar > 0 的实际拟合点。"""

    excluded: tuple[ExcludedPoint, ...]
    """去重后未参与拟合的簇长及原因，按 m 升序。"""

    fit: PiecewiseFit
    segments: tuple[Segment, ...]
    breakpoints: tuple[Breakpoint, ...]


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _reject_bool_or_non_number(value: object, parameter: str,
                               index: Optional[int]) -> None:
    # bool 是 int 的子类；契约要求整数参数拒绝 bool，浮点参数同样不接受 bool。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            parameter, value,
            "must be a real number (int or float); bool and non-numeric "
            "types are rejected", index,
        )


def _finite_float(value: object, parameter: str,
                  index: Optional[int] = None) -> float:
    _reject_bool_or_non_number(value, parameter, index)
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise InvalidInputError(
            parameter, value, "NaN and +/-Infinity are rejected", index
        )
    return result


def _validate_samples(samples: object) -> list[float]:
    if not isinstance(samples, Sequence) or isinstance(
        samples, (str, bytes, bytearray)
    ):
        raise InvalidInputError(
            "samples", type(samples).__name__,
            "must be a finite sequence (list/tuple/...) of real numbers",
        )
    n = len(samples)
    if n < 1:
        raise InvalidInputError("samples", n, "must contain at least 1 sample")
    if n > MAX_SAMPLES:
        raise InvalidInputError(
            "samples", n,
            f"length must be <= {MAX_SAMPLES} (interface bound)",
        )
    values: list[float] = [0.0] * n
    for i, raw in enumerate(samples):
        values[i] = _finite_float(raw, "samples", i)
    return values


def _validate_cluster_sizes(cluster_sizes: object) -> list[int]:
    if not isinstance(cluster_sizes, Sequence) or isinstance(
        cluster_sizes, (str, bytes, bytearray)
    ):
        raise InvalidInputError(
            "cluster_sizes", type(cluster_sizes).__name__,
            "must be a finite sequence (list/tuple/...) of positive integers",
        )
    if len(cluster_sizes) == 0:
        raise InvalidInputError(
            "cluster_sizes", 0, "must contain at least one cluster size"
        )
    result: list[int] = []
    for i, raw in enumerate(cluster_sizes):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise InvalidInputError(
                "cluster_sizes", raw,
                "each cluster size must be an exact int; bool and float "
                "(including integral floats) are rejected", i,
            )
        if raw < 1:
            raise InvalidInputError(
                "cluster_sizes", raw, "each cluster size must be >= 1", i
            )
        result.append(raw)
    return result


# ---------------------------------------------------------------------------
# 重叠 Allan 方差（累积和实现）
# ---------------------------------------------------------------------------

def _cumsum(samples: Sequence[float]) -> list[float]:
    cum = [0.0] * (len(samples) + 1)
    acc = 0.0
    for i, v in enumerate(samples):
        acc += v
        cum[i + 1] = acc
    return cum


def _overlapping_avar_from_cumsum(
    cum: Sequence[float], m: int
) -> tuple[Optional[float], int]:
    """与 overlapping_avar 相同的定义，但复用调用方的累积和。"""
    n = len(cum) - 1
    count = n - 2 * m + 1
    if count <= 0:
        return None, 0

    # 生成器让 math.fsum 以高精度串接全部 d_k^2。
    def squared_diffs():
        inv_m = 1.0 / m
        for k in range(count):
            d = (cum[k + 2 * m] - 2.0 * cum[k + m] + cum[k]) * inv_m
            yield d * d

    return 0.5 * math.fsum(squared_diffs()) / count, count


def overlapping_avar(samples: Sequence[float], m: int) -> tuple[Optional[float], int]:
    """
    返回 (avar, n_differences)。

    定义（本题契约）：对 k = 0 .. N-2m（共 n = N-2m+1 个滑动位置），
        d_k = mean(x[k+m:k+2m]) - mean(x[k:k+m])，
        avar = (1 / (2 n)) * sum_k d_k^2 。
    2m > N 时没有相邻块对，返回 (None, 0)。
    内部使用累积和 O(N) 求块和，平方和用 math.fsum 累加。
    """
    if isinstance(m, bool) or not isinstance(m, int):
        raise InvalidInputError(
            "m", m, "cluster size must be an exact int; bool rejected"
        )
    if m < 1:
        raise InvalidInputError("m", m, "cluster size must be >= 1")
    return _overlapping_avar_from_cumsum(_cumsum(samples), m)


# ---------------------------------------------------------------------------
# 一元 OLS 工具（前缀和，区间 SSE O(1)）
# ---------------------------------------------------------------------------

class _PrefixMoments:
    __slots__ = ("n", "sx", "sy", "sxx", "sxy", "syy")

    def __init__(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        self.n = len(xs)
        sx = [0.0] * (self.n + 1)
        sy = [0.0] * (self.n + 1)
        sxx = [0.0] * (self.n + 1)
        sxy = [0.0] * (self.n + 1)
        syy = [0.0] * (self.n + 1)
        for i, (x, y) in enumerate(zip(xs, ys)):
            sx[i + 1] = sx[i] + x
            sy[i + 1] = sy[i] + y
            sxx[i + 1] = sxx[i] + x * x
            sxy[i + 1] = sxy[i] + x * y
            syy[i + 1] = syy[i] + y * y
        self.sx, self.sy, self.sxx, self.sxy, self.syy = sx, sy, sxx, sxy, syy

    def fit(self, i: int, j: int) -> tuple[float, float, float]:
        """闭区间 [i, j] 的 OLS：返回 (slope, intercept, sse)。"""
        cnt = j - i + 1
        sx = self.sx[j + 1] - self.sx[i]
        sy = self.sy[j + 1] - self.sy[i]
        sxx = self.sxx[j + 1] - self.sxx[i]
        sxy = self.sxy[j + 1] - self.sxy[i]
        syy = self.syy[j + 1] - self.syy[i]
        den = cnt * sxx - sx * sx
        if den == 0.0:  # 全部 x 相同（契约上不会发生，防御性处理）
            slope = 0.0
            intercept = sy / cnt
        else:
            slope = (cnt * sxy - sx * sy) / den
            intercept = (sy - slope * sx) / cnt
        sse = syy - slope * sxy - intercept * sy
        # SSE 数学上必为非负；任何严格负值都是浮点消去噪声，钳为 0。
        return slope, intercept, sse if sse > 0.0 else 0.0


def _infeasible(reason: str, penalty: float, budget: int) -> PiecewiseFit:
    return PiecewiseFit(
        feasible=False, reason=reason,
        starts=(), ends=(), knot_x=(), slopes=(), intercepts=(),
        partition_sse=None, residual_sse=None, objective=None,
        segment_penalty=penalty, residuals=(), evaluations=0, budget=budget,
    )


# ---------------------------------------------------------------------------
# 连续折线最小二乘（铰链基），仅用于 DP 选定断点后的最终模型
# ---------------------------------------------------------------------------

def _solve_linear_system(matrix: list[list[float]],
                         rhs: list[float]) -> list[float]:
    """带部分选主元的高斯-若尔当消元，方阵满秩时返回解。"""
    p = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ArithmeticError("singular design matrix in hinge fit")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_row = a[col]
        pivot_value = pivot_row[col]
        for r in range(p):
            if r == col:
                continue
            factor = a[r][col] / pivot_value
            if factor == 0.0:
                continue
            a[r][col] = 0.0
            for c in range(col + 1, p + 1):
                a[r][c] -= factor * pivot_row[c]
    return [a[i][p] / a[i][i] for i in range(p)]


def _continuous_hinge_fit(
    xs: Sequence[float], ys: Sequence[float],
    starts: Sequence[int],
) -> tuple[tuple[float, ...], tuple[float, ...], float, tuple[float, ...]]:
    """
    在给定分段起点 starts（第一个为 0）的情况下，拟合连续折线：
        y = c0 + c1*x + sum_r  d_r * max(0, x - knot_r)
    内部断点 knot_r = xs[starts[r] - 1]（左段最后一个点），折线在该点连续。
    返回 (各段斜率, 各段等效截距, 残差SSE, 逐点残差)。
    """
    n = len(xs)
    knots = [xs[s - 1] for s in starts[1:]]
    p = 1 + len(starts)  # 截距 + 总斜率 + 每个内部断点一个斜率增量
    columns: list[list[float]] = [[1.0] * n, list(xs)]
    for knot in knots:
        columns.append([max(0.0, x - knot) for x in xs])

    gram = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for a in range(p):
        col_a = columns[a]
        rhs[a] = math.fsum(col_a[i] * ys[i] for i in range(n))
        for b in range(a, p):
            col_b = columns[b]
            value = math.fsum(col_a[i] * col_b[i] for i in range(n))
            gram[a][b] = value
            gram[b][a] = value
    coef = _solve_linear_system(gram, rhs)

    slopes = [coef[1]]
    for d in coef[2:]:
        slopes.append(slopes[-1] + d)
    intercepts: list[float] = []
    running = coef[0]
    for r, d in enumerate(coef[2:]):
        if r == 0:
            intercepts.append(coef[0])
        running -= d * knots[r]
        intercepts.append(running)
    if not intercepts:
        intercepts.append(coef[0])

    residuals = [
        ys[i] - math.fsum(coef[c] * columns[c][i] for c in range(p))
        for i in range(n)
    ]
    sse = math.fsum(e * e for e in residuals)
    return tuple(slopes), tuple(intercepts), (sse if sse > 0.0 else 0.0), tuple(residuals)


# ---------------------------------------------------------------------------
# 动态规划分段
# ---------------------------------------------------------------------------

def fit_piecewise_lines(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    segment_penalty: float = 0.0,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> PiecewiseFit:
    """
    对 log-log 点 (xs, ys)（xs 严格递增、全部有限）做连续分段直线拟合。

    目标 = 各段独立 OLS 残差平方和之和 + (段数-1) * segment_penalty，
    在“最多 3 段、每段至少 3 点”的全部连续分段上做**穷举**动态规划，
    因此返回的是该离散分段空间内的全局最优（非近似搜索）。
    平局规则（确定性）：优先段数少；段数相同优先更靠左的断点。
    得到最优分段后，再以铰链基最小二乘拟合最终的连续折线。
    """
    if not isinstance(xs, Sequence) or not isinstance(ys, Sequence):
        raise InvalidInputError("xs/ys", "non-sequence",
                                "xs and ys must be sequences")
    if isinstance(xs, (str, bytes)) or isinstance(ys, (str, bytes)):
        raise InvalidInputError("xs/ys", "string",
                                "xs and ys must be numeric sequences")
    if len(xs) != len(ys):
        raise InvalidInputError(
            "xs/ys", (len(xs), len(ys)),
            "xs and ys must have equal length",
        )
    for i, v in enumerate(xs):
        _finite_float(v, "xs", i)
    for i, v in enumerate(ys):
        _finite_float(v, "ys", i)
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            raise InvalidInputError(
                "xs", xs[i], "xs must be strictly increasing", i
            )
    if isinstance(segment_penalty, bool) or not isinstance(
        segment_penalty, (int, float)
    ):
        raise InvalidInputError(
            "segment_penalty", segment_penalty,
            "must be a real number; bool rejected",
        )
    if not math.isfinite(segment_penalty) or segment_penalty < 0.0:
        raise InvalidInputError(
            "segment_penalty", segment_penalty,
            "must be finite and >= 0 (units: squared log10 residual)",
        )
    if isinstance(search_budget, bool) or not isinstance(search_budget, int):
        raise InvalidInputError(
            "search_budget", search_budget,
            "must be an exact int; bool and float rejected",
        )
    if search_budget < 1:
        raise InvalidInputError(
            "search_budget", search_budget, "must be >= 1",
        )

    n = len(xs)
    penalty = float(segment_penalty)
    if n < MIN_POINTS_PER_SEGMENT:
        return _infeasible(_REASON_INSUFFICIENT_POINTS, penalty, search_budget)

    moments = _PrefixMoments(xs, ys)
    # 热循环直接使用前缀数组的局部引用，避免每次候选评估的属性/函数调用开销。
    px, py = moments.sx, moments.sy
    pxx, pxy, pyy = moments.sxx, moments.sxy, moments.syy
    inf = math.inf
    evaluations = 0

    def _exhausted() -> None:
        raise SearchBudgetExhausted(evaluations, search_budget, n)

    def seg_sse(a: int, b: int) -> float:
        """闭区间 [a,b] 的 OLS SSE（回溯/最终核算时用，非热路径）。"""
        return moments.fit(a, b)[2]

    min_len = MIN_POINTS_PER_SEGMENT
    max_seg = min(MAX_SEGMENTS, n // min_len)

    # --- 融合动态规划 ---------------------------------------------------
    # 每个候选区间 [a, j] 的 OLS SSE 只计算一次，并同时用于 2 段和 3 段的
    # 转移：
    #   cost1[j] = SSE(0, j)
    #   cost2[j] = min_a cost1[a-1] + SSE(a, j)
    #   cost3[j] = min_a cost2[a-1] + SSE(a, j)
    # 按 j 升序、a 升序单次扫描；evaluations 统计实际计算的候选区间数。
    cost1 = [inf] * n
    cost2 = [inf] * n
    cost3 = [inf] * n
    prev2 = [-1] * n
    prev3 = [-1] * n

    for j in range(min_len - 1, n):
        evaluations += 1
        if evaluations > search_budget:
            _exhausted()
        # 内联 SSE(0, j)。
        c = j + 1
        sx = px[j + 1]
        sy = py[j + 1]
        sxx = pxx[j + 1]
        sxy = pxy[j + 1]
        syy = pyy[j + 1]
        den = c * sxx - sx * sx
        if den == 0.0:
            cost1[j] = 0.0
        else:
            slope = (c * sxy - sx * sy) / den
            sse = syy - slope * sxy - (sy - slope * sx) / c * sy
            cost1[j] = sse if sse > 0.0 else 0.0

    do_three = max_seg >= 3
    for j in range(2 * min_len - 1, n):
        best2 = inf
        best2_a = -1
        best3 = inf
        best3_a = -1
        j1 = j + 1
        a_limit = j - min_len + 1  # 末段至少 min_len 点
        for a in range(min_len, a_limit + 1):
            evaluations += 1
            if evaluations > search_budget:
                _exhausted()
            # 内联 SSE(a, j)：一次区间 SSE 同时服务 2 段/3 段转移。
            c = j1 - a
            sx = px[j1] - px[a]
            sy = py[j1] - py[a]
            sxx = pxx[j1] - pxx[a]
            sxy = pxy[j1] - pxy[a]
            syy = pyy[j1] - pyy[a]
            den = c * sxx - sx * sx
            if den == 0.0:
                sse = 0.0
            else:
                slope = (c * sxy - sx * sy) / den
                sse = syy - slope * sxy - (sy - slope * sx) / c * sy
                if sse <= 0.0:
                    sse = 0.0
            left1 = cost1[a - 1]
            if left1 < inf:
                cand2 = left1 + sse
                # a 升序 + 严格小于：平局取更靠左的断点。
                if cand2 < best2:
                    best2 = cand2
                    best2_a = a
            if do_three:
                left2 = cost2[a - 1]
                if left2 < inf:
                    cand3 = left2 + sse
                    if cand3 < best3:
                        best3 = cand3
                        best3_a = a
        cost2[j] = best2
        prev2[j] = best2_a - 1  # 记录前段末端点（= 末段起点 a 减 1）
        if do_three:
            cost3[j] = best3
            prev3[j] = best3_a - 1

    dp_cost = [cost1]
    dp_prev = [[-1] * n]
    if max_seg >= 2:
        dp_cost.append(cost2)
        dp_prev.append(prev2)
    if max_seg >= 3:
        dp_cost.append(cost3)
        dp_prev.append(prev3)

    # 跨段数选全局最优；k 从小到大、严格小于实现“平局段数更少优先”。
    best_k = 1
    best_value = dp_cost[0][n - 1]
    for k in range(2, max_seg + 1):
        value = dp_cost[k - 1][n - 1] + (k - 1) * penalty
        if value < best_value:
            best_value = value
            best_k = k
    if best_k == 1:
        best_value = dp_cost[0][n - 1]  # 不带罚项

    # 回溯断点：dp_prev[k-1][j] 给出最后一段起点前一位（即前段末端点）。
    ends_rev: list[int] = [n - 1]
    cursor = n - 1
    for k in range(best_k, 1, -1):
        cursor = dp_prev[k - 1][cursor]
        ends_rev.append(cursor)
    ends_fwd = list(reversed(ends_rev))
    starts = [0] + [e + 1 for e in ends_fwd[:-1]]
    ends = ends_fwd

    partition_sse = 0.0
    for s, e in zip(starts, ends):
        partition_sse += seg_sse(s, e)

    slopes, intercepts, residual_sse, residuals = _continuous_hinge_fit(
        xs, ys, starts
    )
    knot_x = tuple(xs[s - 1] for s in starts[1:])

    return PiecewiseFit(
        feasible=True, reason=None,
        starts=tuple(starts), ends=tuple(ends), knot_x=knot_x,
        slopes=slopes, intercepts=intercepts,
        partition_sse=partition_sse, residual_sse=residual_sse,
        objective=best_value, segment_penalty=penalty,
        residuals=residuals, evaluations=evaluations, budget=search_budget,
    )


# ---------------------------------------------------------------------------
# 顶层接口
# ---------------------------------------------------------------------------

def analyze(
    samples: Sequence[float],
    dt: float,
    cluster_sizes: Sequence[int],
    *,
    segment_penalty: float = 0.0,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> AnalysisResult:
    """
    离线单轴固定速率角速率序列的重叠 Allan 方差分解。

    参数
    ----
    samples : sequence[real]，1 <= len <= 8192，元素全部有限（拒绝 NaN/Inf）。
    dt : 正有限实数，采样周期（秒）。bool 不作为数值接受。
    cluster_sizes : 正整数序列，允许重复与非单调输入：
                    * 输出 curve 严格按输入顺序（重复 m 原样保留）；
                    * 拟合使用去重后升序的点集。
    segment_penalty : >= 0 的有限实数，每增加一段在目标函数上的罚项
                    （单位为 log10 残差的平方），默认 0。
    search_budget : 正整数（拒绝 bool），DP 候选段评估次数预算；
                    默认值覆盖全部合法输入，收紧时可能抛 SearchBudgetExhausted。

    返回 AnalysisResult；当拟合在契约下无解时 fit.feasible is False 且
    fit.reason 标明证明性原因，这不抛异常，与预算耗尽异常严格区分。
    """
    values = _validate_samples(samples)
    dt_value = _finite_float(dt, "dt")
    if dt_value <= 0.0:
        raise InvalidInputError("dt", dt, "must be > 0")
    ms = _validate_cluster_sizes(cluster_sizes)
    # 关键字参数在任何计算之前校验，保证无解路径上非法输入同样被定位拒绝。
    penalty = _finite_float(segment_penalty, "segment_penalty")
    if penalty < 0.0:
        raise InvalidInputError(
            "segment_penalty", segment_penalty,
            "must be >= 0 (units: squared log10 residual)",
        )
    if isinstance(search_budget, bool) or not isinstance(search_budget, int):
        raise InvalidInputError(
            "search_budget", search_budget,
            "must be an exact int; bool and float rejected",
        )
    if search_budget < 1:
        raise InvalidInputError("search_budget", search_budget, "must be >= 1")
    # 派生量 tau = m*dt 也必须有限：在最大簇长处检查即可覆盖全部 m。
    if not math.isfinite(max(ms) * dt_value):
        raise InvalidInputError(
            "dt", dt,
            "dt * max(cluster_sizes) overflows to infinity; "
            "derived tau must stay finite",
        )

    n_samples = len(values)

    # 累积和只构造一次，所有簇长的块均值差均复用它。
    cum = _cumsum(values)
    if not all(math.isfinite(v) for v in cum):
        raise InvalidInputError(
            "samples", None,
            "partial cumulative sum is not finite; input magnitudes overflow",
        )

    # ---- curve：保留输入顺序与重复 ----
    cache: dict[int, ClusterPoint] = {}

    def point_for(m: int) -> ClusterPoint:
        if m in cache:
            return cache[m]
        avar, count = _overlapping_avar_from_cumsum(cum, m)
        if avar is not None and not math.isfinite(avar):
            raise InvalidInputError(
                "samples", None,
                f"overlapping Allan variance at m={m} is not finite; "
                "input magnitudes overflow the squared-difference sum",
            )
        if avar is None:
            point = ClusterPoint(
                m=m, tau=m * dt_value, avar=None, adev=None,
                n_differences=0,
            )
        else:
            point = ClusterPoint(
                m=m, tau=m * dt_value, avar=avar,
                adev=math.sqrt(avar), n_differences=count,
            )
        cache[m] = point
        return point

    curve = tuple(point_for(m) for m in ms)

    # ---- 去重升序，划分拟合点/排除点 ----
    unique_ms = sorted(set(ms))
    fit_points: list[ClusterPoint] = []
    excluded: list[ExcludedPoint] = []
    for m in unique_ms:
        p = cache[m]
        if p.n_differences == 0:
            excluded.append(ExcludedPoint(m, 0, _REASON_NO_ESTIMATOR))
        elif p.n_differences < MIN_DIFFERENCES:
            excluded.append(
                ExcludedPoint(m, p.n_differences,
                              _REASON_TOO_FEW_DIFFERENCES)
            )
        elif p.avar == 0.0:
            excluded.append(
                ExcludedPoint(m, p.n_differences, _REASON_ZERO_VARIANCE)
            )
        else:
            fit_points.append(p)

    # ---- 无解判定（证明性，区别于预算异常）----
    if not any(cache[m].n_differences >= MIN_DIFFERENCES for m in unique_ms):
        fit = _infeasible(_REASON_TOO_FEW_DIFFERENCES,
                          penalty, search_budget)
    elif not fit_points:
        fit = _infeasible(_REASON_ZERO_VARIANCE,
                          penalty, search_budget)
    else:
        xs = [math.log10(p.tau) for p in fit_points]
        ys = [math.log10(p.avar) for p in fit_points]
        if len(fit_points) < MIN_POINTS_PER_SEGMENT:
            fit = _infeasible(_REASON_INSUFFICIENT_POINTS,
                              penalty, search_budget)
        else:
            fit = fit_piecewise_lines(
                xs, ys,
                segment_penalty=segment_penalty,
                search_budget=search_budget,
            )

    segments: list[Segment] = []
    breakpoints: list[Breakpoint] = []
    if fit.feasible:
        for s, e, slope, intercept in zip(
            fit.starts, fit.ends, fit.slopes, fit.intercepts
        ):
            ps, pe = fit_points[s], fit_points[e]
            segments.append(Segment(
                start_index=s, end_index=e,
                m_start=ps.m, m_end=pe.m,
                tau_start=ps.tau, tau_end=pe.tau,
                slope=slope, intercept=intercept,
            ))
        for r, s in enumerate(fit.starts[1:]):
            knot_index = s - 1
            xk = fit.knot_x[r]
            yk = fit.slopes[r] * xk + fit.intercepts[r]
            p = fit_points[knot_index]
            breakpoints.append(Breakpoint(
                left_segment_end_index=knot_index,
                m=p.m, tau=p.tau, x=xk, y=yk,
            ))

    return AnalysisResult(
        n_samples=n_samples, dt=dt_value, curve=curve,
        fit_points=tuple(fit_points), excluded=tuple(excluded),
        fit=fit, segments=tuple(segments),
        breakpoints=tuple(breakpoints),
    )
