"""时间戳 + 有效标志模式的必要测试。

与定频模式测试相同的纪律：期望值一律来自独立实现/手算，不调用被测核心产生：

  * ref_timestamped —— 完全按题面端点规则独立枚举每个网格时窗，直接求均值差分；
  * 小例子（线性序列 1..10、定向挖洞）用手算分数核对 count/missing/variance；
  * 均匀完整输入与旧的定频 analyze 互证（两个实现路径独立计算，只做交叉比对）。
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402

GRID_TOL = 1e-9
M_INTEGER_TOL = 1e-6
FRAC = ad.MIN_WINDOW_FRACTION


def ref_timestamped(samples, ts, valid, w):
    """按固定端点规则独立计算单个窗长：(expected, missing, count, variance)。

    不使用核心的任何内部函数：标称周期取有效样本间隔中位数，样本按容差吸附
    整数网格，时窗 [g, g+M) 左闭右开，配对窗相隔一个窗长，覆盖不足半窗即缺失。
    """
    vidx = [i for i in range(len(ts)) if valid[i]]
    vts = [ts[i] for i in vidx]
    vv = [samples[i] for i in vidx]
    gaps = sorted(vts[i] - vts[i - 1] for i in range(1, len(vts)))
    lg = len(gaps)
    step = gaps[lg // 2] if lg % 2 else 0.5 * (gaps[lg // 2 - 1] + gaps[lg // 2])
    t0 = vts[0]

    def snap(q):
        c = math.floor(q)
        if (c + 1) - q <= GRID_TOL * max(1.0, c + 1):
            return c + 1
        return c

    cells = [snap((t - t0) / step) for t in vts]
    m_raw = w / step
    mr = round(m_raw)
    assert abs(m_raw - mr) <= M_INTEGER_TOL * max(1.0, abs(mr)), (
        "reference cases use integer M"
    )
    m = int(mr)
    need = max(1, math.ceil(FRAC * m - 1e-12))
    x = cells[-1] + 1 - 2 * m
    xr = round(x)
    g_max = xr if abs(x - xr) <= GRID_TOL * max(1.0, abs(xr)) else math.floor(x)
    expected = max(0, g_max + 1)

    def members(lo, hi):
        return [vv[k] for k, c in enumerate(cells) if lo <= c < hi]

    diffs = []
    missing = 0
    for g in range(expected):
        a = members(g, g + m)
        b = members(g + m, g + 2 * m)
        if len(a) < need or len(b) < need:
            missing += 1
            continue
        diffs.append(sum(b) / len(b) - sum(a) / len(a))
    variance = (
        sum(d * d for d in diffs) / (2 * (len(diffs) - 1))
        if len(diffs) >= 2
        else None
    )
    return expected, missing, len(diffs), variance


def analyze_points(samples, ts, durations, valid=None):
    """取逐窗统计；拟合无解时从异常附带 points 取（纯函数，无部分写入）。"""
    try:
        res = ad.analyze(
            samples, timestamps=ts, valid=valid, cluster_durations=durations
        )
        return res.points
    except ad.FitInfeasibleError as exc:
        return exc.points


class TestUniformParity(unittest.TestCase):
    def test_uniform_complete_matches_fixed_mode(self):
        # 均匀、完整、窗长为周期整数倍：count/missing/方差与定频模式一致。
        for dt in (1.0, 0.1, 0.02, 0.001):
            rng = random.Random(20260924)
            n = 300
            samples = [rng.gauss(0.0, 1.5) + 0.004 * i for i in range(n)]
            ts = [i * dt for i in range(n)]
            sizes = [1, 2, 3, 5, 8, 13, 50]
            fixed = ad.analyze(samples, dt, sizes)
            ts_points = analyze_points(samples, ts, [m * dt for m in sizes])
            for m, pf, pt in zip(sizes, fixed.points, ts_points):
                self.assertEqual(pt.count, n - 2 * m + 1)
                self.assertEqual(pt.count, pf.count)
                self.assertEqual(pt.expected_count, pt.count)
                self.assertEqual(pt.missing_count, 0)
                self.assertAlmostEqual(pt.tau, m * dt)
                self.assertEqual(pt.m, m * dt)
                rel = abs(pt.variance - pf.variance) / pf.variance
                self.assertLess(rel, 1e-12, (dt, m, rel))
                self.assertTrue(pt.eligible_for_fit == pf.eligible_for_fit)

    def test_result_fields_in_timestamp_mode(self):
        rng = random.Random(3)
        n = 2048
        samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(n)]
        ts = [i * 0.01 for i in range(n)]
        durations = [0.01 * m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)]
        res = ad.analyze(samples, timestamps=ts, cluster_durations=durations)
        self.assertEqual(res.mode, "timestamp")
        self.assertEqual(res.n_valid, n)
        self.assertEqual(res.n_samples, n)
        self.assertAlmostEqual(res.dt, 0.01, delta=1e-15)
        fixed = ad.analyze(samples, 0.01, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(res.fit.segment_count, fixed.fit.segment_count)
        self.assertEqual(
            [bp.m for bp in res.fit.breakpoints],
            [bp.m * 0.01 for bp in fixed.fit.breakpoints],
        )
        for seg_t, seg_f in zip(res.fit.segments, fixed.fit.segments):
            self.assertAlmostEqual(seg_t.slope, seg_f.slope, delta=1e-10)


class TestHandComputed(unittest.TestCase):
    def test_ramp_differences_hand_value(self):
        # omega = 1..10，整数秒时间戳，w=2s（M=2, need=1）。
        # 时窗均值依次 1.5,2.5,...,8.5；配对相隔一个窗长（2 个单元），
        # 故每个差分恒为 2.0；count=7，var = 7*4/(2*6) = 7/3。
        vals = [float(i) for i in range(1, 11)]
        pts = analyze_points(vals, list(range(10)), [2.0])
        p = pts[0]
        self.assertEqual((p.expected_count, p.count, p.missing_count), (7, 7, 0))
        self.assertAlmostEqual(p.variance, 7.0 / 3.0)
        self.assertAlmostEqual(p.adev, math.sqrt(7.0 / 3.0))

    def test_half_window_threshold_creates_missing(self):
        # 挖掉单元 4,5,6（valid[4:7]=False，ts 4,5,6 缺测），w=4s（M=4, need=2），
        # cmax=11 -> g=0..4 共 5 对，逐对手算：
        #   g=0: 后窗[4,8) 仅 {7}（1<need）-> 缺失
        #   g=1: [1,5)={1,2,3} 与 [5,9)={7,8}：前3后2 -> d=7.5-2.0=5.5
        #   g=2: [2,6)={2,3}（2点）与 [6,10)={7,8,9} -> d=8.0-2.5=5.5
        #   g=3: 前窗[3,7) 仅 {3}（1<need）-> 缺失
        #   g=4: 前窗[4,8) 仅 {7}（1<need）-> 缺失
        # expected=5, missing=3, count=2, var = 2*5.5^2/(2*1) = 30.25。
        n = 12
        vals = [float(i) for i in range(n)]
        valid = [True] * n
        for i in (4, 5, 6):
            valid[i] = False
        pts = analyze_points(vals, list(range(n)), [4.0], valid)
        p = pts[0]
        self.assertEqual((p.expected_count, p.count, p.missing_count), (5, 2, 3))
        self.assertAlmostEqual(p.variance, 30.25)

    def test_count_below_two_is_missing_statistic(self):
        # 有效样本只在单元 0,1,2,9：网格跨度可考察 7 对 w=2s 时窗，但只有
        # g=0 一对双侧各含 >=1 个有效样本（count=1，无法做无偏估计），其余
        # 6 对至少一侧为空 -> variance=None（覆盖不足返回缺失，不抛、不给零）。
        vals = [float(i) for i in range(10)]
        valid = [i in (0, 1, 2, 9) for i in range(10)]
        pts = analyze_points(vals, list(range(10)), [2.0], valid)
        p = pts[0]
        self.assertEqual((p.expected_count, p.missing_count, p.count), (7, 6, 1))
        self.assertIsNone(p.variance)
        self.assertIsNone(p.adev)
        self.assertFalse(p.eligible_for_fit)

    def test_window_longer_than_record(self):
        # 2w 超过末端网格：没有可考察窗对，expected=0，直接缺失。
        vals = [float(i) for i in range(10)]
        pts = analyze_points(vals, list(range(10)), [5.0, 100.0])
        self.assertEqual(
            [(p.expected_count, p.count, p.variance) for p in pts],
            [(1, 1, None), (0, 0, None)],
        )

    def test_fractional_window_duration_boundary(self):
        # w=1.5s（M=1.5，非整数倍周期），锁定固定端点规则：起点相隔一个窗长，
        # 跨在两格之间的右边界向上取整。omega=0..19：
        #   g 偶/奇各对的前窗含 2 个连续样本（mean=g+0.5）、后窗含 1 个样本
        #   （值 g+2），差分恒为 1.5；cmax=19 -> expected=floor(20-3)+1=18。
        # var = 18*1.5^2/(2*17) = 81/68。
        vals = [float(i) for i in range(20)]
        pts = analyze_points(vals, list(range(20)), [1.5])
        p = pts[0]
        self.assertEqual((p.expected_count, p.count, p.missing_count), (18, 18, 0))
        self.assertAlmostEqual(p.variance, 81.0 / 68.0)

    def test_all_invalid_and_single_valid(self):
        vals = [float(i) for i in range(10)]
        pts = analyze_points(vals, list(range(10)), [1.0, 2.0], [False] * 10)
        for p in pts:
            self.assertEqual((p.expected_count, p.count, p.missing_count), (0, 0, 0))
            self.assertIsNone(p.variance)
        pts = analyze_points(vals, list(range(10)), [1.0], [True] + [False] * 9)
        self.assertEqual((pts[0].expected_count, pts[0].count), (0, 0))
        self.assertIsNone(pts[0].variance)

    def test_full_missing_segment_makes_fit_infeasible(self):
        # 中段 60 个样本整段缺测（单元 80..139 无有效样本），首尾各 80/60 个
        # 正常：网格仍到 cmax=199，大窗可以考察，但凡时窗跨越缺测中段的配对
        # 都因有效样本不足半窗而缺失。结果只有小窗 w=1s 有 >=20 对差分，
        # 合格拟合点 < 3 -> FitInfeasibleError，逐窗统计随 .points 附带。
        rng = random.Random(11)
        n = 200
        samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(n)]
        valid = [i < 80 or i >= 140 for i in range(n)]
        durations = [1.0, 40.0, 60.0, 120.0]
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            ad.analyze(
                samples,
                timestamps=list(range(n)),
                valid=valid,
                cluster_durations=durations,
            )
        points = cm.exception.points
        self.assertEqual(len(points), len(durations))
        p1, p40, p60, p120 = points
        self.assertGreaterEqual(p1.count, 20)
        self.assertIsNotNone(p1.variance)
        # w=1s 时落在 60 单元缺测段内的时窗为空，61 个起点与缺失格相邻，
        # 对应配对记缺失；有效配对仍有 138 个。
        self.assertEqual(p1.missing_count, 61)
        # w=40s 时仍有少量配对落在缺测段之外（22 对），但每个跨越中段的时窗
        # 都被记缺失（missing>0）；w=60s 双侧达标配对不足 20。
        self.assertGreater(p40.expected_count, 0)
        self.assertGreater(p40.missing_count, 0)
        self.assertEqual(p40.count + p40.missing_count, p40.expected_count)
        self.assertGreater(p60.expected_count, 0)
        self.assertGreater(p60.missing_count, 0)
        self.assertLess(p60.count, 20)
        self.assertIsNone(p60.variance)
        self.assertFalse(p60.eligible_for_fit)
        # 2w 越过末端网格：没有可考察窗对。
        self.assertEqual((p120.expected_count, p120.count), (0, 0))
        self.assertIsNone(p120.variance)
        # 合格点至多 2 个（w=1、w=40），不足每段 3 点 -> 无解。
        self.assertLess(sum(p.eligible_for_fit for p in points), 3)


class TestAgainstIndependentReference(unittest.TestCase):
    def test_irregular_timestamps(self):
        for seed in (1, 7, 99):
            rng = random.Random(seed)
            n = 200
            t = 0.0
            ts = []
            for _ in range(n):
                ts.append(t)
                t += 1.0 if rng.random() < 0.7 else 2.0
            samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(n)]
            for m in (1, 2, 3, 4, 8):
                p = analyze_points(samples, ts, [float(m)])[0]
                exp, miss, cnt, var = ref_timestamped(
                    samples, ts, [True] * n, float(m)
                )
                self.assertEqual(
                    (p.expected_count, p.missing_count, p.count),
                    (exp, miss, cnt),
                    (seed, m),
                )
                if var is None:
                    self.assertIsNone(p.variance)
                else:
                    self.assertLess(
                        abs(p.variance - var) / max(abs(var), 1e-300), 1e-12
                    )

    def test_irregular_with_random_gaps(self):
        rng = random.Random(55)
        n = 300
        t = 0.0
        ts = []
        for _ in range(n):
            ts.append(t)
            t += rng.choice([1.0, 1.0, 1.0, 2.0, 3.0])
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        valid = [rng.random() > 0.15 for _ in range(n)]
        for m in (1, 2, 3, 5, 10):
            p = analyze_points(samples, ts, [float(m)], valid)[0]
            exp, miss, cnt, var = ref_timestamped(
                samples, ts, valid, float(m)
            )
            self.assertEqual(
                (p.expected_count, p.missing_count, p.count), (exp, miss, cnt), m
            )
            if var is not None:
                self.assertLess(
                    abs(p.variance - var) / max(abs(var), 1e-300), 1e-12
                )
        # n_valid 计数正确。
        res = ad.analyze(
            samples[:60],
            timestamps=ts[:60],
            valid=valid[:60],
            cluster_durations=[1.0, 2.0, 4.0],
        )
        self.assertEqual(res.n_valid, sum(valid[:60]))

    def test_integer_zero_one_flags_accepted_like_bool(self):
        rng = random.Random(8)
        n = 120
        ts = list(float(i) for i in range(n))
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        flags = [1 if i % 5 else 0 for i in range(n)]
        bools = [bool(f) for f in flags]
        pa = analyze_points(samples, ts, [1.0, 2.0, 4.0], flags)
        pb = analyze_points(samples, ts, [1.0, 2.0, 4.0], bools)
        for a, b in zip(pa, pb):
            self.assertEqual(
                (a.count, a.expected_count, a.missing_count, a.variance),
                (b.count, b.expected_count, b.missing_count, b.variance),
            )


class TestTimestampScaleAndFractionalWindows(unittest.TestCase):
    def test_epoch_scale_integer_multiple_windows(self):
        # 绝对时间戳很大（epoch 秒 1.7e9）时，间隔中位数带 ~1e-6 相对的
        # binary64 舍入；用户按真实周期整数倍给的窗长仍必须按整数窗处理，
        # 计数与定频模式 N-2m+1 一致、方差相对误差 < 1e-12。
        rng = random.Random(2)
        n = 400
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        ts = [1.7e9 + i * 0.02 for i in range(n)]
        res = ad.analyze(
            samples, timestamps=ts,
            cluster_durations=[0.02, 0.04, 0.08, 0.16],
        )
        fixed = ad.analyze(samples, 0.02, [1, 2, 4, 8])
        for m, pt, pf in zip((1, 2, 4, 8), res.points, fixed.points):
            self.assertEqual(
                (pt.expected_count, pt.count, pt.missing_count),
                (n - 2 * m + 1, n - 2 * m + 1, 0),
            )
            self.assertLess(
                abs(pt.variance - pf.variance) / pf.variance, 1e-12
            )

    def test_true_fractional_window_not_snapped_to_integer(self):
        # w=1.5s 与标称周期 1s 相差半格，远宽于整数判定容差 -> 必须按分数
        # 窗边界处理（右边界向上取整），不得当成长度 2 的整数窗。
        n = 200
        samples = [0.0] * n
        ts = [float(i) for i in range(n)]
        pts = analyze_points(samples, ts, [1.5, 2.0])
        # 常量信号方差为 0；关键是计数边界不同：w=1.5 -> N-3，w=2 -> N-4+1?
        # cmax=199：expected(1.5)=floor(200-3)+1=198，expected(2)=200-4+1=197。
        self.assertEqual(pts[0].expected_count, 198)
        self.assertEqual(pts[1].expected_count, 197)


class TestDurationsOrdering(unittest.TestCase):
    def test_duplicate_and_nonmonotonic_durations(self):
        rng = random.Random(9)
        n = 400
        ts = [float(i) for i in range(n)]
        samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(n)]
        durations = [8.0, 2.0, 8.0, 1.0, 16.0, 2.0]
        res = ad.analyze(
            samples, timestamps=ts, cluster_durations=durations
        )
        self.assertEqual([p.m for p in res.points], durations)
        by_w = {}
        for p in res.points:
            by_w.setdefault(p.m, []).append(p)
        for w, ps in by_w.items():
            self.assertTrue(all(p.variance == ps[0].variance for p in ps))
            self.assertTrue(all(p.count == ps[0].count for p in ps))
        # 拟合去重升序，断点 m 为窗长（秒）。
        self.assertEqual(
            res.fit.fitted_ms, tuple(sorted(set(res.fit.fitted_ms)))
        )


class TestTimestampValidation(unittest.TestCase):
    def setUp(self):
        self.samples = [1.0, 2.0, 3.0, 4.0] * 50
        self.ts = [float(i) for i in range(200)]

    def test_reject_bad_timestamps(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts[:-1],
                       cluster_durations=[1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=[0.0] * 200,
                       cluster_durations=[1.0])
        ts = self.ts[:]
        ts[10] = ts[9] - 1.0
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=ts, cluster_durations=[1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=[float("nan")] * 200,
                       cluster_durations=[1.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps=[True] * 200,
                       cluster_durations=[1.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps="0123",
                       cluster_durations=[1.0])

    def test_reject_bad_valid(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts,
                       valid=[True] * 199, cluster_durations=[1.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps=self.ts,
                       valid=[1, 2] * 100, cluster_durations=[1.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps=self.ts,
                       valid=["Y"] * 200, cluster_durations=[1.0])

    def test_reject_bad_durations(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts, cluster_durations=[])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts, cluster_durations=[0.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts, cluster_durations=[-1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts,
                       cluster_durations=[float("inf")])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps=self.ts,
                       cluster_durations=[True])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze(self.samples, timestamps=self.ts, cluster_durations=4)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts,
                       cluster_durations=[1.0] * 8193)

    def test_reject_mixed_modes(self):
        # 时间戳参数与定频参数不得混用。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, 1.0, [1], timestamps=self.ts)
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, 1.0, [1], cluster_durations=[1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, timestamps=self.ts)  # 缺 cluster_durations
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples, [1.0], valid=[True] * 200)
        # 两个必选参数都缺：明确的定频模式缺参错误。
        with self.assertRaises(ad.AllanValueError):
            ad.analyze(self.samples)

    def test_error_messages_locatable(self):
        try:
            ad.analyze(self.samples, timestamps=self.ts[:-1] + [0.0],
                       cluster_durations=[1.0])
        except ad.AllanValueError as exc:
            self.assertIn("timestamps", str(exc))
        try:
            ad.analyze(self.samples, timestamps=self.ts,
                       valid=[True] * 199 + [2], cluster_durations=[1.0])
        except ad.AllanTypeError as exc:
            self.assertIn("valid[199]", str(exc))
        try:
            ad.analyze(self.samples, timestamps=self.ts,
                       cluster_durations=[1.0, -2.0])
        except ad.AllanValueError as exc:
            self.assertIn("cluster_durations[1]", str(exc))

    def test_samples_still_validated_in_timestamp_mode(self):
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([1.0, float("nan")] + [0.0] * 198,
                       timestamps=self.ts, cluster_durations=[1.0])
        with self.assertRaises(ad.AllanTypeError):
            ad.analyze([True] * 200, timestamps=self.ts,
                       cluster_durations=[1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([], timestamps=[], cluster_durations=[1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.analyze([0.0] * 8193,
                       timestamps=[float(i) for i in range(8193)],
                       cluster_durations=[1.0])


class TestNoMutation(unittest.TestCase):
    def test_inputs_not_mutated(self):
        rng = random.Random(12)
        n = 100
        samples = [rng.gauss(0.0, 1.0) for _ in range(n)]
        ts = [float(i) for i in range(n)]
        valid = [i % 7 != 0 for i in range(n)]
        durations = [4.0, 1.0, 4.0, 2.0]
        s0, t0, v0, d0 = (
            samples[:], ts[:], valid[:], durations[:]
        )
        ad.analyze(samples, timestamps=ts, valid=valid,
                   cluster_durations=durations)
        self.assertEqual(samples, s0)
        self.assertEqual(ts, t0)
        self.assertEqual(valid, v0)
        self.assertEqual(durations, d0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
