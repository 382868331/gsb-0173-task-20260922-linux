"""
必要测试。

参考实现的独立性：
  * naive_avar()        —— 朴素双重循环窗口求和，不调用 allan_decomp 任何函数；
  * enumerate_segments()—— 穷举所有合法分段边界并独立 OLS，不调用被测拟合器；
  * 合成数据由本文件内固定种子的确定性 RNG 生成。

对被测核心的往返检查（连续性、残差重算等）只作为补充断言。
"""

import math
import random
import unittest
from itertools import combinations

from allan_decomp import (
    DEFAULT_SEARCH_BUDGET,
    MAX_SAMPLES,
    InvalidInputError,
    SearchBudgetExhausted,
    analyze,
    fit_piecewise_lines,
)


# ---------------------------------------------------------------------------
# 独立参考 1：朴素 O(N*m) 窗口求和 Allan 方差
# ---------------------------------------------------------------------------

def naive_avar(samples, m):
    """与契约相同的定义，直接对每个 k 重算两个相邻窗口的普通和。"""
    n = len(samples)
    count = n - 2 * m + 1
    if count <= 0:
        return None, 0
    total = 0.0
    for k in range(count):
        s1 = 0.0
        for i in range(k, k + m):
            s1 += samples[i]
        s2 = 0.0
        for i in range(k + m, k + 2 * m):
            s2 += samples[i]
        d = (s2 - s1) / m
        total += d * d
    return total / (2.0 * count), count


# ---------------------------------------------------------------------------
# 独立参考 2：枚举所有分段 + 独立 OLS
# ---------------------------------------------------------------------------

def ref_ols(xs, ys, i, j):
    """闭区间 [i, j] 一元最小二乘，返回 (slope, intercept, sse)。"""
    pts = [(xs[t], ys[t]) for t in range(i, j + 1)]
    cnt = len(pts)
    mx = sum(x for x, _ in pts) / cnt
    my = sum(y for _, y in pts) / cnt
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in pts)
    return slope, intercept, sse


def enumerate_segments(xs, ys, penalty=0.0, max_segments=3, min_len=3):
    """
    穷举全部合法连续分段，返回 (starts, ends, objective, partition_sse, k)。
    平局规则与被测契约一致：段数少优先；同段数按边界元组字典序（更靠左优先）。
    """
    n = len(xs)
    best = None
    for k in range(1, max_segments + 1):
        # k 升序枚举内部边界；严格小于实现“平局段数少/边界更左优先”，
        # 与被测 DP 的确定性平局规则一致。
        for bounds in combinations(range(1, n), k - 1):
            edges = (0,) + bounds + (n,)
            if all(edges[r + 1] - edges[r] >= min_len for r in range(k)):
                sse = 0.0
                for r in range(k):
                    sse += ref_ols(xs, ys, edges[r], edges[r + 1] - 1)[2]
                obj = sse + (k - 1) * penalty
                candidate = (edges[:-1], tuple(e - 1 for e in edges[1:]),
                             obj, sse, k)
                if best is None or obj < best[2]:
                    best = candidate
    return best


# ---------------------------------------------------------------------------
# 确定性合成数据
# ---------------------------------------------------------------------------

def white_noise(n, seed, sigma=0.05):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def noise_plus_ramp(n, dt, seed, sigma=0.05, ramp=0.0566):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) + ramp * t * dt
            for t in range(n)]


class _CloseMixin:
    def assertClose(self, a, b, rel=1e-11, abs_=1e-14):
        self.assertLessEqual(abs(a - b), rel * max(1.0, abs(a), abs(b)) + abs_)


class TestOverlappingAvarAgainstNaive(_CloseMixin, unittest.TestCase):
    def test_matches_naive_reference(self):
        rng = random.Random(4242)
        samples = [rng.gauss(0.1, 1.3) for _ in range(300)]
        for m in (1, 2, 3, 7, 19, 50, 100, 150):
            got = analyze(samples, 0.01, [m]).curve[0]
            ref_val, ref_n = naive_avar(samples, m)
            self.assertEqual(got.n_differences, ref_n)
            if ref_val is None:
                self.assertIsNone(got.avar)
                self.assertIsNone(got.adev)
            else:
                self.assertClose(got.avar, ref_val)
                self.assertClose(got.adev, math.sqrt(ref_val))
                self.assertEqual(got.tau, m * 0.01)

    def test_no_estimator_when_blocks_dont_fit(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        r = analyze(samples, 1.0, [2, 3])
        # 2m == N：恰好 1 个差分；2m > N：无估计量。
        self.assertEqual(r.curve[0].n_differences, 1)
        self.assertIsNotNone(r.curve[0].avar)
        self.assertEqual(r.curve[1].n_differences, 0)
        self.assertIsNone(r.curve[1].avar)
        self.assertIsNone(r.curve[1].adev)

    def test_difference_counts(self):
        n = 100
        r = analyze([0.0] * n, 1.0, [1, 10, 40, 50, 51])
        counts = [p.n_differences for p in r.curve]
        self.assertEqual(counts, [99, 81, 21, 1, 0])

    def test_integer_samples_accepted(self):
        r = analyze([1, 0, 1, 0] * 20, 1, [1])
        self.assertEqual(r.curve[0].n_differences, 79)


class TestConstantSignal(unittest.TestCase):
    def test_constant_nonzero_has_zero_variance(self):
        # 常量（含非零常量）每个块均值差都为 0 → avar 恒 0。
        r = analyze([3.25] * 200, 0.01, [1, 2, 5, 10])
        for p in r.curve:
            self.assertEqual(p.avar, 0.0)
            self.assertEqual(p.adev, 0.0)
        self.assertFalse(r.fit.feasible)
        self.assertEqual(r.fit.reason, "zero_variance")
        self.assertEqual(r.segments, ())
        self.assertEqual(r.breakpoints, ())
        self.assertTrue(all(e.reason == "zero_variance" for e in r.excluded))

    def test_all_zero_short_is_too_few_differences(self):
        # 短数据先撞到差分样本数门槛，而不是零方差。
        r = analyze([0.0] * 10, 1.0, [1])
        self.assertFalse(r.fit.feasible)
        self.assertEqual(r.fit.reason, "too_few_differences")
        self.assertEqual(r.excluded[0].reason, "too_few_differences")


class TestDuplicateAndNonMonotonic(unittest.TestCase):
    def test_order_preserved_and_dedup_for_fit(self):
        samples = white_noise(500, 7)
        ms = [8, 1, 8, 3, 1, 10]
        r = analyze(samples, 0.01, ms)
        self.assertEqual([p.m for p in r.curve], ms)
        # 重复 m 返回逐字段相同的结果对象。
        self.assertEqual(r.curve[0], r.curve[2])
        self.assertEqual(r.curve[1], r.curve[4])
        # 拟合点去重升序。
        fit_ms = [p.m for p in r.fit_points]
        self.assertEqual(fit_ms, sorted(set(fit_ms)))
        self.assertTrue(all(b >= a for a, b in zip(fit_ms, fit_ms[1:])))
        # 去重后数量正确：所有这些 m 都有 n_differences >= 20 且 avar > 0。
        self.assertEqual(len(r.fit_points), len({1, 3, 8, 10}))

    def test_excluded_mix_reasons_sorted(self):
        samples = white_noise(60, 11)
        # N=60：m=31 时 2m>N 无估计量；m=21 时差分样本数 19 < 20。
        r = analyze(samples, 1.0, [31, 1, 21, 31, 2])
        reasons = {e.m: e.reason for e in r.excluded}
        self.assertEqual(reasons[31], "no_estimator")
        self.assertEqual(reasons[21], "too_few_differences")
        self.assertEqual([e.m for e in r.excluded], sorted(reasons))
        self.assertEqual({p.m for p in r.fit_points}, {1, 2})
        # 只有 2 个正方差拟合点 < 每段 3 点 → 证明无解，而非异常。
        self.assertFalse(r.fit.feasible)
        self.assertEqual(r.fit.reason, "insufficient_points")


class TestPiecewiseFitEnumeration(_CloseMixin, unittest.TestCase):
    """与穷举参考逐点对照（多组确定性数据、两种罚项）。"""

    def _cases(self):
        cases = []
        for seed, n in ((1, 9), (2, 10), (3, 12), (4, 11)):
            rng = random.Random(1000 + seed)
            xs = [0.5 * i for i in range(n)]
            ys = [math.sin(xs[i]) + 0.05 * i
                  + (rng.gauss(0.0, 0.05) if i % 2 else 0.0)
                  for i in range(n)]
            cases.append((xs, ys))
        # 纯二次曲线 + 折线型数据。
        xs = [float(i) for i in range(12)]
        cases.append((xs, [0.01 * i * i for i in range(12)]))
        cases.append((xs, [i for i in range(5)] + [4 + 2 * (i - 4)
                                                   for i in range(5, 12)]))
        return cases

    def test_partitions_match_enumeration(self):
        for penalty in (0.0, 0.01):
            for xs, ys in self._cases():
                with self.subTest(penalty=penalty, n=len(xs)):
                    fit = fit_piecewise_lines(xs, ys, segment_penalty=penalty)
                    self.assertTrue(fit.feasible)
                    ref = enumerate_segments(xs, ys, penalty=penalty)
                    self.assertIsNotNone(ref)
                    ref_starts, ref_ends, ref_obj, ref_sse, ref_k = ref
                    self.assertEqual(fit.starts, ref_starts)
                    self.assertEqual(fit.ends, ref_ends)
                    self.assertClose(fit.objective, ref_obj,
                                     rel=1e-9, abs_=1e-11)
                    self.assertClose(fit.partition_sse, ref_sse,
                                     rel=1e-9, abs_=1e-11)
                    self.assertAlmostEqual(
                        fit.objective,
                        fit.partition_sse + (len(fit.starts) - 1) * penalty,
                        delta=1e-12,
                    )

    def test_exact_line_prefers_one_segment_on_ties(self):
        # 完美直线：任何切分 SSE 都为 0；罚项为 0 时平局取更少段数。
        xs = [0.1 * i for i in range(12)]
        ys = [2.0 - 0.7 * x for x in xs]
        fit = fit_piecewise_lines(xs, ys, segment_penalty=0.0)
        self.assertEqual(fit.starts, (0,))
        self.assertEqual(fit.ends, (11,))
        self.assertAlmostEqual(fit.slopes[0], -0.7, delta=1e-12)
        self.assertAlmostEqual(fit.residual_sse, 0.0, delta=1e-14)

    def test_min_points_constraint(self):
        xs = [float(i) for i in range(9)]
        ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        fit = fit_piecewise_lines(xs, ys)
        # 9 个点允许 3 段；每段必须恰有 3 点。
        self.assertEqual(len(fit.starts), 3)
        self.assertEqual(fit.starts, (0, 3, 6))
        self.assertEqual(fit.ends, (2, 5, 8))

    def test_penalty_collapses_to_one_segment(self):
        xs = [0.1 * i for i in range(15)]
        ys = [math.sin(x) for x in xs]
        free = fit_piecewise_lines(xs, ys, segment_penalty=0.0)
        self.assertEqual(len(free.starts), 3)  # 无罚项时 SSE 严格下降
        penalized = fit_piecewise_lines(xs, ys, segment_penalty=1e6)
        self.assertEqual(penalized.starts, (0,))

    def test_continuity_and_residuals_roundtrip(self):
        xs = [0.2 * i for i in range(14)]
        ys = [math.sin(0.5 * x) + 0.01 * x * x for x in xs]
        fit = fit_piecewise_lines(xs, ys)
        # 断点处左右两段给出相同 y。
        for r, xk in enumerate(fit.knot_x):
            y_left = fit.slopes[r] * xk + fit.intercepts[r]
            y_right = fit.slopes[r + 1] * xk + fit.intercepts[r + 1]
            self.assertAlmostEqual(y_left, y_right, delta=1e-12)
        # 逐点残差与斜率/截距重算一致。
        for i, (x, y, res) in enumerate(zip(xs, ys, fit.residuals)):
            seg = max(r for r, s in enumerate(fit.starts) if s <= i)
            yhat = fit.slopes[seg] * x + fit.intercepts[seg]
            self.assertAlmostEqual(res, y - yhat, delta=1e-12)
        self.assertClose(fit.residual_sse,
                         sum(e * e for e in fit.residuals),
                         rel=1e-12)

    def test_too_few_points_is_proven_infeasible(self):
        for n in (0, 1, 2):
            fit = fit_piecewise_lines([float(i) for i in range(n)],
                                      [0.0] * n)
            self.assertFalse(fit.feasible)
            self.assertEqual(fit.reason, "insufficient_points")
            self.assertEqual(fit.evaluations, 0)


class TestBudgetVsInfeasible(unittest.TestCase):
    def test_tiny_budget_raises_exhaustion_on_feasible_problem(self):
        xs = [0.1 * i for i in range(20)]
        ys = [math.sin(x) for x in xs]
        with self.assertRaises(SearchBudgetExhausted) as cm:
            fit_piecewise_lines(xs, ys, search_budget=1)
        self.assertEqual(cm.exception.budget, 1)
        self.assertGreater(cm.exception.evaluations, 1)
        self.assertEqual(cm.exception.n_points, 20)

    def test_infeasible_never_consumes_budget(self):
        # 证明无解的路径在任何小预算下都直接返回，不得抛预算异常。
        for budget in (1, 2):
            fit = fit_piecewise_lines([1.0, 2.0], [0.0, 0.1],
                                      search_budget=budget)
            self.assertFalse(fit.feasible)
            self.assertEqual(fit.reason, "insufficient_points")
            self.assertEqual(fit.budget, budget)

    def test_default_budget_covers_interface_bound(self):
        # 接口边界 N=8192 时可拟合点数上界（m 从 1 起）：
        n = (MAX_SAMPLES - 19) // 2
        self.assertLessEqual(n, 4086)
        # 评估次数上界约 n^2/2 < n^2 < 1.68e7 < 默认预算。
        self.assertLess(n * n, DEFAULT_SEARCH_BUDGET)


class TestSyntheticRegions(unittest.TestCase):
    """统计性辨识：白噪声 avar~tau^-1；漂移斜坡 avar~tau^2。"""

    def test_pure_white_noise_slope_minus_one(self):
        n, dt = 2048, 0.01
        samples = white_noise(n, 20260922)
        r = analyze(samples, dt, list(range(1, 41)),
                    segment_penalty=1e6)  # 大罚项 → 单一直线
        self.assertTrue(r.fit.feasible)
        self.assertEqual(len(r.segments), 1)
        # 角度随机游走/角速率白噪声：AVAR ∝ tau^{-1}。
        self.assertAlmostEqual(r.segments[0].slope, -1.0, delta=0.18)

    def test_ramp_long_region_slope_plus_two(self):
        # 交叉点 m* ≈ 92：需要比白噪声实验更长的簇长才能看清 +2 区间。
        n, dt = 4096, 0.01
        samples = noise_plus_ramp(n, dt, 20260923)
        r = analyze(samples, dt, list(range(1, 151)))
        self.assertTrue(r.fit.feasible)
        self.assertEqual(len(r.segments), 3)
        # 最长簇长区间由斜坡主导：AVAR ∝ tau^{+2}。
        self.assertGreater(r.segments[-1].slope, 1.7)
        self.assertLess(r.segments[-1].slope, 2.2)
        # 最短簇长区间仍以白噪声为主：斜率接近 -1。
        self.assertLess(r.segments[0].slope, -0.5)
        # 白噪声区与漂移区之间存在一个过渡段（斜率居中）。
        self.assertTrue(
            r.segments[0].slope < r.segments[1].slope
            < r.segments[2].slope
        )
        # 断点数量与位置合理：漂移区必须延伸到足够大的 m。
        self.assertEqual(len(r.breakpoints), 2)
        self.assertGreaterEqual(r.segments[-1].m_end, 150)
        self.assertTrue(all(3 <= b.m <= 148 for b in r.breakpoints))


class TestInputValidation(unittest.TestCase):
    def assertInvalid(self, ctx, parameter, index=None):
        self.assertEqual(ctx.exception.parameter, parameter)
        if index is not None:
            self.assertEqual(ctx.exception.index, index)

    def test_reject_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInputError) as ctx:
                    analyze([0.0, bad, 1.0], 0.01, [1])
                self.assertInvalid(ctx, "samples", 1)
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([0.0] * 30, float("nan"), [1])
        self.assertInvalid(ctx, "dt")

    def test_reject_bool_everywhere(self):
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([True, False] * 20, 0.01, [1])
        self.assertInvalid(ctx, "samples", 0)
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([0.0] * 40, True, [1])
        self.assertInvalid(ctx, "dt")
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([0.0] * 40, 0.01, [1, True, 2])
        self.assertInvalid(ctx, "cluster_sizes", 1)
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], segment_penalty=True)
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], search_budget=True)

    def test_reject_types_and_ranges(self):
        with self.assertRaises(InvalidInputError):
            analyze((x for x in range(40)), 0.01, [1])  # 生成器不接受
        with self.assertRaises(InvalidInputError):
            analyze("0000", 0.01, [1])
        with self.assertRaises(InvalidInputError):
            analyze([], 0.01, [1])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * (MAX_SAMPLES + 1), 0.01, [1])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.0, [1])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, -0.01, [1])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [0])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [-2])
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [2.0])  # 整参数拒绝浮点
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], segment_penalty=-1e-9)
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], segment_penalty=float("inf"))
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], search_budget=0)
        with self.assertRaises(InvalidInputError):
            analyze([0.0] * 40, 0.01, [1], search_budget=100.0)

    def test_interface_bound_accepted(self):
        # 上限规模是合法输入；只取一个簇长，计算量很小。
        r = analyze([0.001 * (i % 7) for i in range(MAX_SAMPLES)],
                    0.01, [1])
        self.assertEqual(r.n_samples, MAX_SAMPLES)
        self.assertEqual(r.curve[0].n_differences, MAX_SAMPLES - 1)

    def test_derived_overflow_rejected_locally(self):
        # dt * max(m) 溢出为 inf：在 dt 参数上定位拒绝。
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([0.1] * 40, 1e308, [10])
        self.assertEqual(ctx.exception.parameter, "dt")
        # 累积和溢出：定位到 samples。
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([1e308] * 40, 1.0, [1])
        self.assertEqual(ctx.exception.parameter, "samples")
        # 差分平方和溢出：同样定位到 samples，而不是返回 inf 结果。
        with self.assertRaises(InvalidInputError) as ctx:
            analyze([1e200 if i % 2 else -1e200 for i in range(40)],
                    1.0, [1])
        self.assertEqual(ctx.exception.parameter, "samples")

    def test_error_is_value_error_and_localizable(self):
        try:
            analyze([0.0] * 5, 0.01, [1, -9])
        except InvalidInputError as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertEqual(exc.parameter, "cluster_sizes")
            self.assertEqual(exc.index, 1)
            self.assertEqual(exc.bad_value, -9)
            self.assertIn("cluster_sizes[1]", str(exc))
        else:
            self.fail("expected InvalidInputError")


if __name__ == "__main__":
    unittest.main(verbosity=2)
