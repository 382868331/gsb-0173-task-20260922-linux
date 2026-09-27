"""allan_deviation 的必要测试。

参考实现完全独立于被测核心：
  * naive_allan_variance —— 朴素滑动窗口求和，双重循环直接按定义计算；
  * brute_force_fit      —— itertools 枚举所有合法切分，独立闭式 OLS 求最优。

被测核心只用于取得实际输出，期望值一律来自参考/解析结论。
"""

import itertools
import math
import os
import random
import sys
import unittest

# `python -m unittest discover -s tests` 以 tests/ 为顶层目录，需把仓库根加入路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402


# ---------------------------------------------------------------------------
# 独立参考 1：朴素窗口求和的重叠 Allan 方差
# ---------------------------------------------------------------------------


def naive_allan_variance(samples, m):
    """按定义双重循环计算，不使用任何前缀和/被测代码。

    窗口求和用 math.fsum（精确求和），使参考在大恒定偏置数据上仍可信。
    """
    n = len(samples)
    count = n - 2 * m + 1
    if count < 2:
        return None, count
    total = 0.0
    for k in range(count):
        d = math.fsum(samples[k + m:k + 2 * m]) / m - math.fsum(samples[k:k + m]) / m
        total += d * d
    return total / (2.0 * (count - 1)), count


# ---------------------------------------------------------------------------
# 独立参考 2：枚举所有合法连续分段，独立 OLS
# ---------------------------------------------------------------------------


def ref_ols(xs, ys, i, j):
    """区间 [i,j] 上独立实现的最小二乘直线与 SSE。"""
    pts = list(range(i, j + 1))
    nn = float(len(pts))
    mx = sum(xs[t] for t in pts) / nn
    my = sum(ys[t] for t in pts) / nn
    sxx = sum((xs[t] - mx) ** 2 for t in pts)
    sxy = sum((xs[t] - mx) * (ys[t] - my) for t in pts)
    syy = sum((ys[t] - my) ** 2 for t in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    sse = syy - sxy * sxy / sxx
    return slope, intercept, max(0.0, sse)


def brute_force_fit(taus, variances, penalty, max_segments=3, min_points=3):
    """枚举全部切分，返回 (best_sse_plus_penalty, best_bounds, per_segment_sse)。

    与核心相同的并列规则：严格更优才替换，故并列时段数更少者胜。
    """
    k = len(taus)
    xs = [math.log(t) for t in taus]
    ys = [0.5 * math.log(v) for v in variances]
    pmax = min(max_segments, k // min_points)
    best_obj = math.inf
    best_bounds = None
    best_sse = None
    for p in range(1, pmax + 1):
        # bounds: 右侧段起点，在 1..k-1 中选 p-1 个，段长均 >= min_points。
        for cuts in itertools.combinations(range(1, k), p - 1):
            bounds = list(cuts)
            starts = [0] + bounds
            ends = [b - 1 for b in bounds] + [k - 1]
            if any(j - i + 1 < min_points for i, j in zip(starts, ends)):
                continue
            sse_total = 0.0
            for i, j in zip(starts, ends):
                _, _, s = ref_ols(xs, ys, i, j)
                sse_total += s
            obj = sse_total + (p - 1) * penalty
            if obj < best_obj:
                best_obj = obj
                best_bounds = tuple(bounds)
                best_sse = sse_total
    return best_obj, best_bounds, best_sse


def synth_mixture(n, seed, noise_sd=1.0, ramp_per_sample=0.0, dt=1.0):
    rng = random.Random(seed)
    return [noise_sd * rng.gauss(0.0, 1.0) + ramp_per_sample * i
            for i in range(n)]


class TestNaiveAgreement(unittest.TestCase):
    def test_matches_naive_window_sum(self):
        rng = random.Random(4242)
        samples = [rng.gauss(0.0, 2.0) + 0.01 * i for i in range(300)]
        sizes = [1, 2, 3, 5, 8, 13, 50, 140]
        res = ad.analyze(samples, 0.01, sizes)
        self.assertEqual(len(res.points), len(sizes))
        for pt, m in zip(res.points, sizes):
            ref_var, ref_count = naive_allan_variance(samples, m)
            self.assertEqual(pt.count, ref_count)
            self.assertAlmostEqual(pt.tau, m * 0.01)
            rel = abs(pt.variance - ref_var) / ref_var
            # 前缀和(顺序累加)+fsum(平方) 与朴素双重循环：相对差应在 1e-12 内。
            self.assertLess(rel, 1e-12, (m, pt.variance, ref_var))
            self.assertAlmostEqual(pt.adev, math.sqrt(pt.variance))

    def test_large_baseline_stability(self):
        # 大恒定偏置：验证去偏置后不发生灾难性崩溃（不返回 0/乱值）。
        # binary64 把 1e9+O(1) 样本存入时低位已被量化（单次减法相对舍入
        # ~eps*基线/噪声），此误差来自输入表示本身，核心与朴素 fsum 参考
        # 都无法逾越；故按实测的表示极限分档断言，而非套用 O(1) 数据的 1e-12。
        rng = random.Random(77)
        noise = [rng.gauss(0.0, 1.0) for _ in range(400)]
        sizes = [1, 4, 16, 64]
        res0 = ad.analyze(noise, 0.01, sizes)
        for base, bound in ((1e6, 1e-9), (1e9, 1e-7)):
            samples = [base + g for g in noise]
            res = ad.analyze(samples, 0.01, sizes)
            for pt, m, p0 in zip(res.points, sizes, res0.points):
                ref_var, ref_count = naive_allan_variance(samples, m)
                self.assertEqual(pt.count, ref_count)
                self.assertGreater(pt.variance, 0.0)
                self.assertLess(abs(pt.variance - ref_var) / ref_var, bound)
                # 与无基线 O(1) 结果也处于同一量化误差带内。
                self.assertLess(abs(pt.variance - p0.variance) / p0.variance, bound)

    def test_counts_and_oversize_cluster(self):
        samples = synth_mixture(100, 1)
        # m=1 唯一合格点 -> 拟合无解，逐 m 统计随异常附带返回。
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze(samples, 1.0, [1, 50, 51, 100])
        points = cm.exception.points
        # 实际差分样本数不会为负：2m>N 时为 0；m=50 时恰为 1。
        self.assertEqual([p.count for p in points], [99, 1, 0, 0])
        self.assertIsNotNone(points[0].variance)
        # m=50 -> count=1，无法做无偏估计；m=51/100 -> count<=0。
        for idx in (1, 2, 3):
            self.assertIsNone(points[idx].variance)
            self.assertIsNone(points[idx].adev)
            self.assertFalse(points[idx].eligible_for_fit)


class TestPiecewiseFit(unittest.TestCase):
    def _check_against_enumeration(self, n, seed, ms, ramp):
        samples = synth_mixture(n, seed, ramp_per_sample=ramp)
        res = ad.analyze(samples, 1.0, ms)
        elig = [next(pp for pp in res.points if pp.m == m) for m in res.fit.fitted_ms]
        taus = [p.tau for p in elig]
        variances = [p.variance for p in elig]
        ref_obj, ref_bounds, ref_sse = brute_force_fit(
            taus, variances, ad.SEGMENT_PENALTY)
        scale = max(1.0, abs(ref_obj))
        self.assertAlmostEqual(res.fit.objective, ref_obj, delta=1e-9 * scale)
        self.assertAlmostEqual(res.fit.sse, ref_sse, delta=1e-9 * max(1.0, ref_sse))
        # 核心选出的切分必须与枚举最优之一一致（构造数据差距明显，不应并列）。
        core_bounds = tuple(res.fit.fitted_ms.index(bp.m) for bp in res.fit.breakpoints)
        self.assertEqual(core_bounds, ref_bounds, (seed, ramp, core_bounds, ref_bounds))
        # 段约束。
        self.assertLessEqual(len(res.fit.segments), 3)
        for seg in res.fit.segments:
            self.assertGreaterEqual(seg.n_points, 3)

    def test_dp_optimal_vs_enumeration(self):
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # 纯白噪声：应 1 段；白噪声+强漂移：应多段。多组种子。
        for seed in (1, 7, 99):
            self._check_against_enumeration(1500, seed, ms, 0.0)
            self._check_against_enumeration(1500, seed, ms, 0.01)

    def test_penalty_changes_segment_count(self):
        # 罚项极大时枚举参考与核心规则一致：本测试仅核对核心内部一致性：
        # SEGMENT_PENALTY 为模块公开常量，白噪声数据通常 1 段即可。
        samples = synth_mixture(2000, 5)
        res = ad.analyze(samples, 1.0, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(res.fit.segment_count, 1)
        # 残差与段直线对齐，且 sse == sum(residual^2)。
        self.assertEqual(len(res.fit.residuals), len(res.fit.fitted_ms))
        self.assertAlmostEqual(
            res.fit.sse, sum(r * r for r in res.fit.residuals), delta=1e-12)
        self.assertAlmostEqual(
            res.fit.rmse, math.sqrt(res.fit.sse / len(res.fit.fitted_ms)))


    def test_three_segment_layer_directly(self):
        # 直接构造 12 个合格 AllanPoint，真值为 3 段不同斜率（±0.2 噪声），
        # 覆盖 DP 第 3 层（通过 analyze 难以稳定构造出 3 区间真实数据）。
        rng = random.Random(2026)
        slopes = (-0.5, 0.0, 1.0)
        groups = ((1, 2, 4, 8), (16, 32, 64, 128), (256, 512, 1024, 2048))
        ms, taus, log_vars = [], [], []
        for slope, grp in zip(slopes, groups):
            for m in grp:
                ms.append(m)
                taus.append(float(m))
                log_vars.append(slope * math.log(m) + rng.gauss(0.0, 0.02))
        points = [
            ad.AllanPoint(m, float(m), math.exp(2.0 * lv), math.exp(lv), 100, True)
            for m, lv in zip(ms, log_vars)
        ]
        fit = ad._fit_piecewise(points)
        variances = [math.exp(2.0 * lv) for lv in log_vars]
        ref_obj, ref_bounds, ref_sse = brute_force_fit(
            taus, variances, ad.SEGMENT_PENALTY)
        self.assertEqual(fit.segment_count, 3)
        self.assertAlmostEqual(fit.objective, ref_obj, delta=1e-9 * max(1, ref_obj))
        core_bounds = tuple(fit.fitted_ms.index(bp.m) for bp in fit.breakpoints)
        self.assertEqual(core_bounds, ref_bounds)
        self.assertEqual([s.n_points for s in fit.segments], [4, 4, 4])
        for seg, expected in zip(fit.segments, slopes):
            self.assertAlmostEqual(seg.slope, expected, delta=0.25)

    def test_segment_feasibility_limits(self):
        def make(k):
            return [
                ad.AllanPoint(2 ** i, float(2 ** i), 1.0 + 0.01 * i,
                              math.sqrt(1.0 + 0.01 * i), 100, True)
                for i in range(k)
            ]
        # 9 点：3 段可行；8 点：最多 2 段；2 点：无解。
        self.assertEqual(ad._fit_piecewise(make(9)).segment_count <= 3, True)
        self.assertLessEqual(ad._fit_piecewise(make(8)).segment_count, 2)
        with self.assertRaises(ad.FitInfeasibleError):
            ad._fit_piecewise(make(2))


class TestSlopeIdentification(unittest.TestCase):
    def test_white_noise_then_drift_regions(self):
        # 离散交叉点 m*^3 = 2*sigma^2/a^2 = 2e4 -> m*≈27；小 m 斜率约 -1/2，大 m 约 +1。
        samples = synth_mixture(2048, 123, noise_sd=1.0, ramp_per_sample=0.01)
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        res = ad.analyze(samples, 1.0, ms)
        self.assertEqual(res.fit.segment_count, 2)
        seg_small = res.fit.segments[0]
        seg_large = res.fit.segments[-1]
        self.assertLess(seg_small.slope, -0.25)
        self.assertGreater(seg_small.slope, -0.75)
        self.assertGreater(seg_large.slope, 0.6)
        self.assertLess(seg_large.slope, 1.25)
        self.assertEqual(seg_small.m_start, 1)
        self.assertEqual(seg_large.m_end, 512)
        # 断点落在交叉点附近（16..64 之间）。
        self.assertTrue(16 <= res.fit.breakpoints[0].m <= 64)


class TestStreaming(unittest.TestCase):
    def test_arbitrary_batches_match_offline_and_naive(self):
        samples = synth_mixture(300, 4242, noise_sd=2.0, ramp_per_sample=0.01)
        sizes = [1, 2, 3, 5, 8, 13, 50, 140]
        stream = ad.StreamingAllan(0.01, sizes)
        cuts = [0, 1, 18, 71, 170, 237, len(samples)]
        final_result = None
        for left, right in zip(cuts, cuts[1:]):
            prefix = samples[:right]
            final_result = stream.append(samples[left:right])
            try:
                expected = ad.analyze(prefix, 0.01, sizes)
                expected_points = expected.points
            except ad.FitInfeasibleError as exc:
                expected_points = exc.points
            self.assertEqual(final_result.n_samples, right)
            self.assertEqual(
                [(p.m, p.count) for p in final_result.points],
                [(p.m, p.count) for p in expected_points])
            for actual, point in zip(final_result.points, expected_points):
                if point.variance is None:
                    self.assertIsNone(actual.variance)
                    self.assertIsNone(actual.adev)
                else:
                    self.assertEqual(actual.variance, point.variance)
                    self.assertEqual(actual.adev, point.adev)

        for point, m in zip(final_result.points, sizes):
            ref_variance, ref_count = naive_allan_variance(samples, m)
            self.assertEqual(point.count, ref_count)
            self.assertLess(
                abs(point.variance - ref_variance) / ref_variance, 1e-12)

    def test_cross_batch_windows_counted_once_without_gaps(self):
        samples = synth_mixture(120, 88)
        sizes = [3, 7, 20]
        stream = ad.StreamingAllan(1.0, sizes)
        cuts = [0, 5, 6, 13, 40, 41, 79, 120]
        for end in cuts[1:]:
            result = stream.append(samples[cuts[cuts.index(end) - 1]:end])
            for point, m in zip(result.points, sizes):
                expected_count = max(0, end - 2 * m + 1)
                self.assertEqual(point.count, expected_count, (end, m))
                if expected_count < 2:
                    self.assertIsNone(point.variance)
                    self.assertFalse(point.eligible_for_fit)

    def test_long_scale_unavailable_does_not_block_other_scales(self):
        samples = synth_mixture(80, 55)
        stream = ad.StreamingAllan(1.0, [1, 2, 3, 50])
        result = stream.append(samples)
        short_point, _, _, long_point = result.points
        self.assertGreaterEqual(short_point.count, 20)
        self.assertIsNotNone(short_point.variance)
        self.assertTrue(short_point.eligible_for_fit)
        self.assertEqual(long_point.count, 0)
        self.assertIsNone(long_point.variance)
        self.assertIsNone(long_point.adev)
        self.assertFalse(long_point.eligible_for_fit)
        self.assertIsNotNone(result.fit)

        more = synth_mixture(120, 56)
        result = stream.append(more)
        long_point = result.points[3]
        self.assertGreaterEqual(long_point.count, 20)
        self.assertIsNotNone(long_point.variance)
        self.assertIsNotNone(result.fit)

    def test_invalid_batch_is_atomic(self):
        samples = synth_mixture(80, 12)
        stream = ad.StreamingAllan(1.0, [1, 4])
        before = stream.append(samples)
        with self.assertRaises(ad.AllanValueError):
            stream.append([1.0, float("nan"), 2.0])
        self.assertEqual(stream.n_samples, 80)
        after = stream.snapshot()
        self.assertEqual(after.n_samples, before.n_samples)
        self.assertEqual(
            [p.variance for p in after.points],
            [p.variance for p in before.points])
        self.assertEqual(
            [p.count for p in after.points],
            [p.count for p in before.points])

    def test_duplicate_sizes_keep_fixed_order(self):
        stream = ad.StreamingAllan(0.5, [4, 1, 4])
        result = stream.append(synth_mixture(100, 13))
        self.assertEqual([p.m for p in result.points], [4, 1, 4])
        self.assertEqual(result.points[0], result.points[2])

    def test_append_accepts_one_scalar_but_not_bool(self):
        stream = ad.StreamingAllan(1.0, [1])
        result = stream.append(2.5)
        self.assertEqual(result.n_samples, 1)
        result = stream.append(3.5)
        self.assertEqual(result.n_samples, 2)
        self.assertEqual(result.points[0].count, 1)
        with self.assertRaises(ad.AllanTypeError):
            stream.append(True)

    def test_huge_always_inactive_scale_does_not_allocate_history(self):
        stream = ad.StreamingAllan(1e-300, [1, 10 ** 300])
        result = stream.append([1.0, 2.0])
        self.assertEqual(result.points[1].count, 0)
        self.assertIsNone(result.points[1].variance)
        self.assertEqual(len(stream._history), 3)


class TestDuplicatesOrderingAndShortData(unittest.TestCase):
    def test_duplicate_and_nonmonotonic_sizes(self):
        samples = synth_mixture(400, 9)
        sizes = [8, 2, 8, 1, 16, 2]
        res = ad.analyze(samples, 0.5, sizes)
        # 输出顺序、重复次数与输入完全一致，统计值相同。
        self.assertEqual([p.m for p in res.points], sizes)
        by_m = {}
        for p in res.points:
            by_m.setdefault(p.m, []).append(p)
        for m, pts in by_m.items():
            self.assertTrue(all(p.variance == pts[0].variance for p in pts))
            self.assertTrue(all(p.count == pts[0].count for p in pts))
        # 拟合点去重且升序。
        self.assertEqual(res.fit.fitted_ms, tuple(sorted(set(res.fit.fitted_ms))))
        self.assertLessEqual(len(res.fit.fitted_ms), len(sizes))

    def test_constant_zero_variance_infeasible(self):
        # 常量信号：所有差分为 0，方差为 0，log 未定义 -> 无可拟合点。
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze([3.5] * 500, 0.01, [1, 2, 4, 8])
        exc = cm.exception
        self.assertIsInstance(exc, ad.AllanError)
        # points 已附带，且方差确实为 0、count 正确。
        self.assertEqual(len(exc.points), 4)
        self.assertTrue(all(p.variance == 0.0 for p in exc.points))
        self.assertTrue(all(p.count >= 20 for p in exc.points))
        self.assertTrue(all(not p.eligible_for_fit for p in exc.points))

    def test_all_zero_constant(self):
        with self.assertRaises(ad.FitInfeasibleError):
            ad.analyze([0.0] * 100, 1.0, [1, 2, 3])

    def test_short_data_infeasible_but_counts_present(self):
        # 短数据：count 全部 < 20（N=20: count=N-2m+1=19,17,15），无解而非伪拟合。
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze(synth_mixture(20, 4), 1.0, [1, 2, 3])
        points = cm.exception.points
        self.assertEqual([p.count for p in points], [19, 17, 15])
        self.assertTrue(all(p.count < 20 for p in points))
        self.assertTrue(all(not p.eligible_for_fit for p in points))

    def test_only_two_eligible_points(self):
        # N=60: 合格阈值 m <= (N-19)/2 = 20；只有 m=1,2 合格 -> 无解（每段至少3点）。
        samples = synth_mixture(60, 4)
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze(samples, 1.0, [1, 2, 21, 22, 30])
        self.assertEqual(
            [p.eligible_for_fit for p in cm.exception.points],
            [True, True, False, False, False])


class TestInputValidation(unittest.TestCase):
    def test_reject_nonfinite(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, float("nan"), 2.0], 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, float("inf")], 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, float("-inf")], 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0], float("nan"), [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0], float("inf"), [1])

    def test_reject_bad_dt(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0], 0.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0], -1.0, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0], True, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0], "0.1", [1])

    def test_reject_bool_everywhere(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([True, False], 1.0, [1])  # bool 不当作数值样本
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1, 2], 1, [True])        # bool 不当作整数 m
        # 真正的 int 样本与 int dt、int m 是允许的（用非 2 幂周期的整数序列，
        # 保证 m=1,2,4 上方差为正）。
        res = ad.analyze([i % 7 for i in range(800)], 1, [1, 2, 4])
        self.assertEqual(res.dt, 1.0)

    def test_reject_cluster_size_types_and_values(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0] * 100, 1.0, [1.0, 2])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0] * 100, 1.0, ["2"])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 100, 1.0, [0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 100, 1.0, [-3])

    def test_reject_bad_sequences(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(123, 1.0, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze("abc", 1.0, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0] * 10, 1.0, 4)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([], 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 10, 1.0, [])

    def test_reject_huge_ints_and_overflow(self):
        # 超大 int 无法表示为有限 float，或使 m*dt 溢出：给可定位模块错误。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1] * 10, 10 ** 400, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([10 ** 400] * 10, 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 10, 1.0, [10 ** 310])
        # 单个 1e308 尖峰：精确均值不溢出，但相邻差分平方溢出 inf -> 拒绝非 inf/NaN。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1e308] + [0.0] * 49, 1.0, [1, 2, 4])
        # 大量极值连精确均值的中间求和都溢出（n*1e308 超 float 上限）。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1e308 if i % 2 else 0.0 for i in range(50)], 1.0, [1, 2, 4])
        # 极大常量若连精确均值的中间求和都溢出（n*1e308 超 float 上限），
        # 同样给可定位取值错误而非裸 OverflowError。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1e308] * 50, 1.0, [1, 2, 4])
        # 可表示的大常量：差分恒 0，零方差拟合无解（不是数值错误）。
        with self.assertRaises(ad.FitInfeasibleError):
            ad.analyze([1e150] * 50, 1.0, [1, 2, 4])
        # 不太大但 2m>N 的 m 是合法输入（count=0），只是无解。
        with self.assertRaises(ad.FitInfeasibleError):
            ad.analyze([1.0, 0.0, 1.0] * 100, 1.0, [10 ** 6])

    def test_length_bounds(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([0.0] * 8193, 1.0, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([0.0] * 10, 1.0, [1] * 8193)
        # 上界本身合法（常量数据无拟合点 -> 无解，不是边界错误）。
        with self.assertRaises(ad.FitInfeasibleError):
            ad.analyze([0.0] * 8192, 1.0, [1] * 8192)

    def test_error_messages_locatable(self):
        try:
            ad.analyze([1.0, float("nan")], 1.0, [1])
        except ad.AllanValueError as e:
            self.assertIn("samples[1]", str(e))
        try:
            ad.analyze([1.0] * 10, 1.0, [1, -2])
        except ad.AllanValueError as e:
            self.assertIn("cluster_sizes[1]", str(e))


class TestErrorTypeSeparation(unittest.TestCase):
    def test_infeasible_distinct_from_budget(self):
        # FitInfeasibleError 与 SearchBudgetExhaustedError 是不同类型。
        self.assertTrue(issubclass(ad.FitInfeasibleError, ad.AllanError))
        self.assertTrue(issubclass(ad.SearchBudgetExhaustedError, ad.AllanError))
        self.assertIsNot(ad.FitInfeasibleError, ad.SearchBudgetExhaustedError)
        # 公开边界内穷举精确完成，正常结果不带 budget 异常（K=10 点）。
        samples = synth_mixture(2048, 123, ramp_per_sample=0.01)
        res = ad.analyze(samples, 1.0,
                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertGreaterEqual(res.fit.objective, 0.0)

    def test_budget_exhaustion_is_distinct_path(self):
        # 公开边界内预算不可达；这里把预算压到极小，验证预算耗尽抛的是
        # SearchBudgetExhaustedError（带 used/budget），而不是 FitInfeasibleError。
        pts = [
            ad.AllanPoint(2 ** i, float(2 ** i), 1.0 + 0.1 * i,
                          math.sqrt(1.0 + 0.1 * i), 100, True)
            for i in range(12)
        ]
        saved = ad._SEARCH_BUDGET
        ad._SEARCH_BUDGET = 1
        try:
            with self.assertRaises(ad.SearchBudgetExhaustedError) as cm:
                ad._fit_piecewise(pts)
            self.assertGreater(cm.exception.used, cm.exception.budget)
        finally:
            ad._SEARCH_BUDGET = saved
        # 恢复后同一输入正常求解。
        self.assertIsNotNone(ad._fit_piecewise(pts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
