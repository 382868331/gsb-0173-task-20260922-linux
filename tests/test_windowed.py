"""不规则采样 / 缺测时间窗口路径的必要测试。

参考实现 ``naive_windowed`` 完全独立于被测核心：按半开区间定义直接枚举
每个网格窗对、用 math.fsum 求窗内有效样本均值，不调用核心任何函数。
被测核心只用于取得实际输出，期望值一律来自参考 / 手算解析结论。
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402


# ---------------------------------------------------------------------------
# 独立参考：按实际时间窗口直算（不调用被测核心）
# ---------------------------------------------------------------------------


def _grid_floor(span):
    """与核心端点容差同语义的独立写法：接近整数（1e-9 相对内）则取整。"""
    f = math.floor(span)
    if abs(span - (f + 1)) <= 1e-9 * max(1.0, abs(span)):
        return f + 1
    return f


def naive_windowed(samples, timestamps, dt, m, valid=None, req=None):
    """返回 (positions, count, missing, variance)，直接按半开区间定义枚举。"""
    n = len(samples)
    tv = [
        (timestamps[i], float(samples[i]))
        for i in range(n)
        if valid is None or valid[i]
    ]
    t0 = timestamps[0]
    t_last = timestamps[-1]
    span = _grid_floor((t_last - t0) / dt)
    positions = max(0, span - 2 * m + 2)
    need = m if req is None else req

    ds = []
    missing = 0
    for k in range(positions):
        lo1 = t0 + k * dt
        mid = t0 + (k + m) * dt
        hi2 = t0 + (k + 2 * m) * dt
        w1 = [v for t, v in tv if lo1 <= t < mid]
        w2 = [v for t, v in tv if mid <= t < hi2]
        if len(w1) < need or len(w2) < need:
            missing += 1
            continue
        d = math.fsum(w2) / len(w2) - math.fsum(w1) / len(w1)
        ds.append(d * d)
    count = len(ds)
    variance = math.fsum(ds) / (2.0 * (count - 1)) if count >= 2 else None
    return positions, count, missing, variance


def irregular_timestamps(n, dt, seed, jitter=0.45):
    """确定性生成严格递增、不均匀的时间戳：间隔在 dt*(1±jitter) 内。"""
    rng = random.Random(seed)
    ts = [0.0]
    for _ in range(n - 1):
        ts.append(ts[-1] + dt * rng.uniform(1.0 - jitter, 1.0 + jitter))
    return ts


def get_points(samples, dt, sizes, **kw):
    """analyze 成功返回 points；FitInfeasible 时取异常附带的 points。"""
    try:
        res = ad.analyze(samples, dt, sizes, **kw)
        return res.points
    except ad.FitInfeasibleError as exc:
        return exc.points


# ---------------------------------------------------------------------------
# 正常路径：与定频结果一致 / 与独立参考一致
# ---------------------------------------------------------------------------


class TestUniformEquivalence(unittest.TestCase):
    def test_exact_grid_matches_uniform_path(self):
        # 精确整数网格：窗口路径与定频路径只允许 1 ULP 级分组舍入差异。
        rng = random.Random(101)
        samples = [rng.gauss(0.0, 2.0) + 0.01 * i for i in range(400)]
        sizes = [1, 2, 3, 5, 8, 13, 50, 140]
        r_u = ad.analyze(samples, 1.0, sizes)
        r_t = ad.analyze(samples, 1.0, sizes, timestamps=[float(i) for i in range(400)])
        r_v = ad.analyze(samples, 1.0, sizes, valid=[True] * 400)
        for pu, pt, pv in zip(r_u.points, r_t.points, r_v.points):
            self.assertEqual(pt.count, pu.count)
            self.assertEqual(pt.positions, pu.count)
            self.assertEqual(pt.missing, 0)
            self.assertEqual(pv.positions, pu.count)
            self.assertEqual(pv.missing, 0)
            if pu.variance is None:
                self.assertIsNone(pt.variance)
                self.assertIsNone(pv.variance)
            else:
                self.assertLess(
                    abs(pt.variance - pu.variance) / pu.variance, 1e-14)
                self.assertLess(
                    abs(pv.variance - pu.variance) / pu.variance, 1e-14)

    def test_dt_grid_matches_uniform_path(self):
        # dt=0.01 的乘法网格（有二进制舍入）：端点容差保证对数与方差一致。
        rng = random.Random(102)
        n = 600
        samples = [rng.gauss(0.0, 1.0) + 0.002 * i for i in range(n)]
        sizes = [1, 2, 4, 8, 16, 32, 100, 250]
        r_u = ad.analyze(samples, 0.01, sizes)
        r_t = ad.analyze(
            samples, 0.01, sizes, timestamps=[i * 0.01 for i in range(n)])
        for pu, pt in zip(r_u.points, r_t.points):
            self.assertEqual((pt.count, pt.positions, pt.missing),
                             (pu.count, pu.count, 0))
            self.assertLess(
                abs(pt.variance - pu.variance) / pu.variance, 1e-12)

    def test_fit_chain_unchanged_on_windowed_path(self):
        # 均匀输入走窗口路径时，拟合链路结果与定频路径相同。
        rng = random.Random(103)
        samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(1500)]
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        r_u = ad.analyze(samples, 1.0, sizes)
        r_t = ad.analyze(samples, 1.0, sizes,
                         timestamps=[float(i) for i in range(1500)])
        self.assertEqual(r_t.fit.fitted_ms, r_u.fit.fitted_ms)
        self.assertEqual(r_t.fit.segment_count, r_u.fit.segment_count)
        self.assertEqual(
            [bp.m for bp in r_t.fit.breakpoints],
            [bp.m for bp in r_u.fit.breakpoints])
        self.assertAlmostEqual(
            r_t.fit.objective, r_u.fit.objective,
            delta=1e-9 * max(1.0, abs(r_u.fit.objective)))


class TestWindowedAgainstReference(unittest.TestCase):
    def test_irregular_timestamps_match_naive(self):
        rng = random.Random(202)
        n = 500
        dt = 0.02
        ts = irregular_timestamps(n, dt, 202)
        samples = [rng.gauss(0.0, 1.0) + 0.003 * i for i in range(n)]
        sizes = [1, 2, 3, 5, 8, 13, 21, 40]
        points = get_points(samples, dt, sizes, timestamps=ts)
        for p, m in zip(points, sizes):
            pos, cnt, miss, var = naive_windowed(samples, ts, dt, m)
            self.assertEqual(p.positions, pos)
            self.assertEqual(p.count, cnt)
            self.assertEqual(p.missing, miss)
            self.assertEqual(p.positions, p.count + p.missing)
            if var is None:
                self.assertIsNone(p.variance)
            else:
                self.assertLess(abs(p.variance - var) / var, 1e-12, (m,))

    def test_gaps_match_naive(self):
        rng = random.Random(203)
        n = 600
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        flags = [True] * n
        # 确定性挖出若干缺测段与散点。
        for i in range(120, 160):
            flags[i] = False
        for i in range(380, 400):
            flags[i] = False
        for i in (10, 500, 555):
            flags[i] = False
        sizes = [1, 2, 4, 8, 16, 32]
        points = get_points(samples, 0.01, sizes, valid=flags)
        ts = [i * 0.01 for i in range(n)]
        for p, m in zip(points, sizes):
            pos, cnt, miss, var = naive_windowed(
                samples, ts, 0.01, m, valid=flags)
            self.assertEqual((p.positions, p.count, p.missing), (pos, cnt, miss))
            self.assertGreater(p.missing, 0)  # 缺口确实造成缺失
            if var is not None:
                self.assertLess(abs(p.variance - var) / var, 1e-12)

    def test_irregular_and_gaps_combined_match_naive(self):
        rng = random.Random(204)
        n = 700
        dt = 0.05
        ts = irregular_timestamps(n, dt, 204, jitter=0.4)
        flags = [rng.random() > 0.08 for _ in range(n)]
        samples = [rng.gauss(0.0, 1.0) + 0.002 * i for i in range(n)]
        sizes = [1, 2, 4, 7, 12, 25]
        points = get_points(samples, dt, sizes, timestamps=ts, valid=flags)
        for p, m in zip(points, sizes):
            pos, cnt, miss, var = naive_windowed(
                samples, ts, dt, m, valid=flags)
            self.assertEqual((p.positions, p.count, p.missing), (pos, cnt, miss))
            if var is not None:
                self.assertLess(abs(p.variance - var) / var, 1e-11)

    def test_min_window_samples_threshold(self):
        rng = random.Random(205)
        n = 400
        dt = 0.01
        ts = irregular_timestamps(n, dt, 205, jitter=0.6)
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for req in (1, 2, 3):
            points = get_points(
                samples, dt, [1, 2, 4], timestamps=ts,
                min_window_samples=req)
            for p, m in zip(points, [1, 2, 4]):
                pos, cnt, miss, var = naive_windowed(
                    samples, ts, dt, m, req=req)
                self.assertEqual((p.positions, p.count, p.missing),
                                 (pos, cnt, miss), (req, m))
                # 放宽覆盖阈值只会让更多窗对有效。
            if req == 1:
                pts1 = points
        self.assertGreaterEqual(
            sum(p.count for p in pts1),
            sum(p.count for p in get_points(
                samples, dt, [1, 2, 4], timestamps=ts)))

    def test_windowed_fit_against_enumeration(self):
        # 不规则采样的长序列：拟合仍能进行，且与逐点枚举切分的独立参考一致。
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_allan import brute_force_fit

        rng = random.Random(206)
        n = 2000
        dt = 0.01
        ts = irregular_timestamps(n, dt, 206, jitter=0.2)
        samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(n)]
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        res = ad.analyze(samples, dt, sizes, timestamps=ts)
        elig = [next(pp for pp in res.points if pp.m == m)
                for m in res.fit.fitted_ms]
        self.assertGreaterEqual(len(elig), 3)
        ref_obj, ref_bounds, _ = brute_force_fit(
            [p.tau for p in elig], [p.variance for p in elig],
            ad.SEGMENT_PENALTY)
        self.assertAlmostEqual(
            res.fit.objective, ref_obj,
            delta=1e-9 * max(1.0, abs(ref_obj)))
        core_bounds = tuple(
            res.fit.fitted_ms.index(bp.m) for bp in res.fit.breakpoints)
        self.assertEqual(core_bounds, ref_bounds)


# ---------------------------------------------------------------------------
# 边界：端点规则、半开区间、整段缺测、覆盖不足、缓存
# ---------------------------------------------------------------------------


class TestWindowBoundaries(unittest.TestCase):
    def test_hand_computed_half_open_assignment(self):
        # t=[0,2,4], m=2, req=1:
        # k0: {0}=10 vs [2,4)={2}=30 -> d=20
        # k1: [1,3)={2}=30 vs [3,5)={4}=50 -> d=20 -> var=2*400/2=400
        p = get_points([10.0, 30.0, 50.0], 1.0, [2],
                       timestamps=[0.0, 2.0, 4.0],
                       min_window_samples=1)[0]
        self.assertEqual((p.positions, p.count, p.missing, p.variance),
                         (2, 2, 0, 400.0))
        # t=2 左移到 1.999：k0 的右窗 [2,4) 立即变空 -> 该对缺失；
        # k1: 30 vs 50 -> d=20，仅 1 个有效对 -> 无法估计方差。
        p = get_points([10.0, 30.0, 50.0], 1.0, [2],
                       timestamps=[0.0, 1.999, 4.0],
                       min_window_samples=1)[0]
        self.assertEqual((p.positions, p.count, p.missing, p.variance),
                         (2, 1, 1, None))

    def test_hand_computed_irregular_pair_means(self):
        # 窗内两个值、时间戳挪到半格：t_last=4.5 -> positions=2。
        # k0: mean(10,20)=15 vs mean(30,42)=36 -> 21
        # k1: mean(30,42)=36 vs mean(50,66)=58 -> 22
        # var = (441+484)/2 = 462.5
        p = get_points(
            [10.0, 20.0, 30.0, 42.0, 50.0, 66.0], 1.0, [2],
            timestamps=[0.0, 0.5, 2.0, 2.5, 4.0, 4.5])[0]
        self.assertEqual((p.positions, p.count, p.missing), (2, 2, 0))
        self.assertEqual(p.variance, 462.5)

    def test_endpoint_grid_extent(self):
        # 均匀 N=12, m=2 -> positions = N-2m+1 = 9（与定频一致）。
        p = get_points([float(i) for i in range(12)], 1.0, [2],
                       timestamps=[float(i) for i in range(12)])[0]
        self.assertEqual((p.positions, p.count, p.missing), (9, 9, 0))
        # t_last=10.7 不足第 12 个格点 -> positions=8，且 8 对都完整。
        p = get_points(
            [float(i) for i in range(11)] + [10.7], 1.0, [2],
            timestamps=[float(i) for i in range(11)] + [10.7])[0]
        self.assertEqual((p.positions, p.count, p.missing), (8, 8, 0))
        # 最后一个时间戳在末格内：positions 仍为 9，k=8 右窗拿到 2 个样本。
        p = get_points(
            [float(i) for i in range(12)] + [11.5], 1.0, [2],
            timestamps=[float(i) for i in range(12)] + [11.5])[0]
        self.assertEqual((p.positions, p.count, p.missing), (9, 9, 0))

    def test_float_grid_rounding_tolerance(self):
        # 3.99/0.01 在 binary64 下是 398.99999...：容差保证不丢最后一对。
        n = 400
        p = get_points([1.0] * n, 0.01, [2],
                       timestamps=[i * 0.01 for i in range(n)])
        # 常量信号 -> 方差 0 不 eligible，但位置数必须完整。
        self.assertEqual((p[0].positions, p[0].missing), (397, 0))

    def test_entire_record_missing(self):
        flags = [False] * 40
        points = get_points(
            [float(i) for i in range(40)], 1.0, [1, 2, 5], valid=flags)
        for p, m in zip(points, [1, 2, 5]):
            self.assertEqual(p.count, 0)
            self.assertEqual(p.variance, None)
            self.assertEqual(p.adev, None)
            self.assertEqual(p.positions, max(0, 40 - 2 * m + 1))
            self.assertEqual(p.missing, p.positions)
            self.assertFalse(p.eligible_for_fit)

    def test_under_coverage_is_infeasible_with_points(self):
        # 大量缺测使每个 m 的有效对都 < 20：证明无拟合点，逐 m 统计随异常返回。
        rng = random.Random(207)
        n = 300
        flags = [i % 3 == 0 for i in range(n)]  # 只剩 1/3 样本
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze(samples, 1.0, [1, 2, 4], valid=flags)
        points = cm.exception.points
        self.assertEqual(len(points), 3)
        self.assertTrue(all(p.count < 20 for p in points))
        self.assertTrue(all(p.missing > 0 for p in points))
        self.assertTrue(all(not p.eligible_for_fit for p in points))

    def test_duplicate_sizes_share_stats_on_windowed_path(self):
        rng = random.Random(208)
        n = 300
        ts = irregular_timestamps(n, 0.01, 208)
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        sizes = [4, 2, 4, 1, 2]
        res = ad.analyze(samples, 0.01, sizes, timestamps=ts)
        self.assertEqual([p.m for p in res.points], sizes)
        by_m = {}
        for p in res.points:
            by_m.setdefault(p.m, []).append(p)
        for pts in by_m.values():
            self.assertTrue(all(p.variance == pts[0].variance for p in pts))
            self.assertTrue(all(p.count == pts[0].count for p in pts))
            self.assertTrue(all(p.missing == pts[0].missing for p in pts))


# ---------------------------------------------------------------------------
# 失败：非法 timestamps / valid / min_window_samples，且不产生部分变更
# ---------------------------------------------------------------------------


class TestWindowedValidation(unittest.TestCase):
    def test_bad_timestamp_types(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], timestamps=123)
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], timestamps="012")
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, True, 2.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, "1", 2.0])

    def test_bad_timestamp_values(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, float("nan"), 2.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, float("inf"), 2.0])
        with self.assertRaises(ad.AllanValueError):  # 相等：非严格递增
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, 1.0, 1.0])
        with self.assertRaises(ad.AllanValueError):  # 倒退
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, 2.0, 1.5])

    def test_timestamp_length_mismatch(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], timestamps=[0.0, 1.0])

    def test_bad_valid_flags(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], valid=[1, 1, 0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       valid=[True, "true", False])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], valid=True)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], valid=[True, False])

    def test_bad_min_window_samples(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0] * 50, 1.0, [1], min_window_samples=1.0)
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([1.0] * 50, 1.0, [1], min_window_samples=True)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 50, 1.0, [1], min_window_samples=0)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0] * 50, 1.0, [1], min_window_samples=-2)

    def test_error_messages_locatable(self):
        try:
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       timestamps=[0.0, 2.0, 1.0])
        except ad.AllanValueError as e:
            self.assertIn("timestamps[2]", str(e))
        else:
            self.fail("expected AllanValueError")
        try:
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1],
                       valid=[True, True, 0])
        except ad.AllanTypeError as e:
            self.assertIn("valid[2]", str(e))
        else:
            self.fail("expected AllanTypeError")

    def test_inputs_not_mutated_on_error_and_success(self):
        samples = [1.0, float("nan"), 3.0]
        ts = [0.0, 1.0, 1.0]
        flags = [True, True]
        samples_copy = list(samples)
        ts_copy = list(ts)
        with self.assertRaises(ad.AllanError):
            ad.analyze(samples, 1.0, [1], timestamps=ts)
        with self.assertRaises(ad.AllanError):
            ad.analyze([1.0, 2.0, 3.0], 1.0, [1], valid=flags)
        self.assertEqual(samples, samples_copy)
        self.assertEqual(ts, ts_copy)
        # 成功路径同样不改输入。用确定性非周期序列（避免周期=4 使 m=4 差分恒 0）。
        rng = random.Random(209)
        good = [rng.gauss(0.0, 1.0) for _ in range(120)]
        good_ts = [float(i) for i in range(len(good))]
        good_flags = [True] * len(good)
        good_before = list(good)
        ad.analyze(good, 1.0, [1, 2, 4], timestamps=good_ts, valid=good_flags)
        self.assertEqual(good, good_before)
        self.assertEqual(good_ts, [float(i) for i in range(len(good))])
        self.assertEqual(good_flags, [True] * len(good))


if __name__ == "__main__":
    unittest.main(verbosity=2)
