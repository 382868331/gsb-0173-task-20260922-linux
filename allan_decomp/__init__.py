"""
allan_decomp
============

固定速率单轴陀螺仪角速率采样的重叠 Allan 方差分解（本题契约的有限子集）。

公开接口：
    analyze(samples, dt, cluster_sizes, *, segment_penalty=0.0,
            search_budget=DEFAULT_SEARCH_BUDGET) -> AnalysisResult
    fit_piecewise_lines(xs, ys, *, segment_penalty=0.0, ...) -> PiecewiseFit
    overlapping_avar(samples, m) -> (avar | None, n_differences)

数据类型：AnalysisResult / ClusterPoint / ExcludedPoint / PiecewiseFit /
Segment / Breakpoint。
错误：AllanAnalysisError（基类）、InvalidInputError、SearchBudgetExhausted。

经验 log-log 斜率只是对离线数据的统计描述，不是硬件保证或临床结论。
"""

from .core import (
    DEFAULT_SEARCH_BUDGET,
    MAX_SAMPLES,
    MAX_SEGMENTS,
    MIN_DIFFERENCES,
    MIN_POINTS_PER_SEGMENT,
    AllanAnalysisError,
    AnalysisResult,
    Breakpoint,
    ClusterPoint,
    ExcludedPoint,
    InvalidInputError,
    PiecewiseFit,
    SearchBudgetExhausted,
    Segment,
    analyze,
    fit_piecewise_lines,
    overlapping_avar,
)

__all__ = [
    "analyze",
    "fit_piecewise_lines",
    "overlapping_avar",
    "AnalysisResult",
    "ClusterPoint",
    "ExcludedPoint",
    "PiecewiseFit",
    "Segment",
    "Breakpoint",
    "AllanAnalysisError",
    "InvalidInputError",
    "SearchBudgetExhausted",
    "MAX_SAMPLES",
    "MAX_SEGMENTS",
    "MIN_POINTS_PER_SEGMENT",
    "MIN_DIFFERENCES",
    "DEFAULT_SEARCH_BUDGET",
]
