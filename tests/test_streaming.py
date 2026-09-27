"""StreamingAllan（分批/流式 Allan 统计）的必要测试。

期望值不来自被测流式核心：
  * 方差/计数用 tests.test_allan 中的朴素窗口求和独立参考 naive_allan_variance，
    或直接来自离线 analyze（离线本身另有独立参考覆盖）；
  * 跨批窗口“不遗漏、不重复”用不同分批方式 + 全量计数恒等式交叉验证。
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402
from test_allan import naive_allan_variance, synth_mixture  # noqa: E402


def _feed(dt, sizes, batches):
    acc = ad.streaming_allan(dt, sizes)
    res = None
    for b in batches:
        res = acc.append(b)
    return acc, res


class TestStreamingMatchesOffline(unittest.TestCase):
    def test_final_stats_match_naive_reference(self):
        rng = random.Random(551)
        data = [rng.gauss(0.0, 1.5) + 0.002 * i for i in range(600)]
        sizes = [1, 2, 3, 7, 13, 50, 200, 350]
        batches = [data[:1], data[1:9], data[9:250], data[250:301], data[301:]]
        _, res = _feed(0.01, sizes, batches)
        self.assertEqual(res.n_samples, 600)
        self.assertEqual([p.m for p in res.points], sizes)
        for p, m in zip(res.points, sizes):
            ref_var, ref_count = naive_allan_variance(data, m)
            self.assertEqual(p.count, max(0, ref_count))
            self.assertEqual(p.n_samples, 600)
            if ref_var is None:
                self.assertIsNone(p.variance)
                self.assertIsNone(p.adev)
                self.assertFalse(p.computable)
            else:
                self.assertTrue(p.computable)
                self.assertLess(abs(p.variance - ref_var) / ref_var, 1e-12)
                self.assertAlmostEqual(p.adev, math.sqrt(p.variance))

    def test_final_fit_matches_offline_analyze(self):
        data = synth_mixture(1200, 99, ramp_per_sample=0.01)
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        offline = ad.analyze(data, 1.0, sizes)
        _, res = _feed(1.0, sizes, [data[:3], data[3:17], data[17:800], data[800:]])
        self.assertIsNotNone(res.fit)
        self.assertEqual(res.fit.segment_count, offline.fit.segment_count)
        self.assertEqual(
            [bp.m for bp in res.fit.breakpoints],
            [bp.m for bp in offline.fit.breakpoints],
        )
        self.assertAlmostEqual(
            res.fit.objective, offline.fit.objective,
            delta=1e-9 * max(1.0, abs(offline.fit.objective)))


class TestCrossBatchWindows(unittest.TestCase):
    def test_no_omission_no_double_count_any_partition(self):
        data = synth_mixture(700, 7, noise_sd=1.0, ramp_per_sample=0.003)
        sizes = [1, 6, 29, 100, 350]
        # 多种分批方式（含每批 1 个、乱边界），最终计数与方差必须一致。
        partitions = [
            [data],
            [data[:1]] + [data[i:i + 1] for i in range(1, len(data))],
            [data[:2], data[2:301], data[301:699], data[699:]],
            [data[:13], data[13:14], data[14:699], data[699:]],
        ]
        ref = None
        for batches in partitions:
            _, res = _feed(1.0, sizes, batches)
            counts = tuple(p.count for p in res.points)
            # 恒等式：count(m) = max(0, N - 2m + 1)，与分批方式无关。
            self.assertEqual(
                counts, tuple(max(0, 700 - 2 * m + 1) for m in sizes))
            if ref is None:
                ref = tuple(p.variance for p in res.points)
            else:
                for v0, v1 in zip(ref, (p.variance for p in res.points)):
                    if v0 is None:
                        self.assertIsNone(v1)
                        continue
                    self.assertLess(abs(v0 - v1) / v0, 1e-12)

    def test_counts_grow_monotonically_after_each_append(self):
        data = synth_mixture(100, 3)
        sizes = [1, 10, 40, 60]
        acc = ad.streaming_allan(1.0, sizes)
        steps = [1, 4, 20, 30, 45]
        idx = 0
        for step in steps:
            res = acc.append(data[idx:idx + step])
            idx += step
            n = idx
            for p, m in zip(res.points, sizes):
                self.assertEqual(p.count, max(0, n - 2 * m + 1))
                self.assertEqual(p.n_samples, n)


class TestLongScalesIndependentlyIncomputable(unittest.TestCase):
    def test_individual_scale_marking(self):
        # N=100: m=1,2,3 可算；m=40 恰好 count=21 可算；m=49 count=3；
        # m=50 count=1 不可算；m=80 count=0 不可算。互不影响。
        data = synth_mixture(100, 5)
        sizes = [1, 2, 3, 40, 49, 50, 80]
        acc = ad.streaming_allan(1.0, sizes)
        res = acc.append(data)
        computable = [p.computable for p in res.points]
        self.assertEqual(computable, [True, True, True, True, True, False, False])
        # 不可算尺度仍给出 tau/count，只是方差为 None；可算尺度不受影响。
        for p, m in zip(res.points, sizes):
            self.assertAlmostEqual(p.tau, float(m))
            self.assertEqual(p.count, max(0, 100 - 2 * m + 1))
            if p.computable:
                self.assertIsNotNone(p.variance)
            else:
                self.assertIsNone(p.variance)
                self.assertIsNone(p.adev)
                self.assertFalse(p.eligible_for_fit)
        # 合格点（count>=20 且正方差）为 m=1,2,3,40 -> 4 点 >=3，可拟合。
        self.assertIsNotNone(res.fit)

    def test_fit_none_until_enough_eligible_but_stats_present(self):
        data = synth_mixture(30, 8)
        sizes = [1, 2, 20]
        acc = ad.streaming_allan(1.0, sizes)
        res = acc.append(data)
        # m=1,2 合格（count>=20），m=20 连一个无偏差分都没有；仅 2 个拟合点。
        self.assertIsNone(res.fit)
        self.assertIn("2", res.fit_infeasible_reason)
        self.assertEqual([p.computable for p in res.points], [True, True, False])
        # 继续追加后拟合变可行，之前统计无缝接上。
        res2 = acc.append(synth_mixture(500, 8))
        # 注意：两批是不同随机序列，仅验证状态推进与拟合恢复。
        self.assertEqual(res2.n_samples, 530)
        self.assertIsNotNone(res2.fit)
        self.assertTrue(all(p.computable for p in res2.points))

    def test_empty_snapshot_before_any_data(self):
        acc = ad.streaming_allan(1.0, [1, 2])
        res = acc.snapshot()
        self.assertEqual(res.n_samples, 0)
        self.assertTrue(all(not p.computable for p in res.points))
        self.assertTrue(all(p.count == 0 for p in res.points))
        self.assertIsNone(res.fit)
        # 之后仍可正常 append。
        res2 = acc.append(synth_mixture(100, 1))
        self.assertEqual(res2.n_samples, 100)


class TestRetainedHistory(unittest.TestCase):
    def test_only_required_history_is_kept(self):
        sizes = [2, 10, 33]
        acc = ad.streaming_allan(1.0, sizes)
        self.assertEqual(acc.retained_limit, 2 * 33 - 1)
        acc.append(synth_mixture(100, 1))
        self.assertLessEqual(acc.retained_samples, acc.retained_limit)
        self.assertEqual(acc.retained_samples, 65)  # N=100 > 65，裁到 2*mmax-1
        acc.append(synth_mixture(1000, 2))
        self.assertEqual(acc.retained_samples, 65)
        self.assertEqual(acc.n_samples, 1100)

    def test_history_sufficient_for_cross_batch_windows(self):
        # 历史裁剪后，跨批次窗口仍与朴素参考一致（不遗漏跨批差分）。
        rng = random.Random(31)
        data = [rng.gauss(0.0, 1.0) for _ in range(1500)]
        sizes = [1, 50, 200, 400]
        acc = ad.streaming_allan(0.1, sizes)
        for i in range(0, 1500, 333):
            acc.append(data[i:i + 333])
        res = acc.snapshot()
        for p, m in zip(res.points, sizes):
            ref_var, ref_count = naive_allan_variance(data, m)
            self.assertEqual(p.count, max(0, ref_count))
            self.assertLess(abs(p.variance - ref_var) / ref_var, 1e-12)

    def test_constant_signal_zero_variance(self):
        acc = ad.streaming_allan(1.0, [1, 2, 4])
        res = acc.append([3.5] * 200)
        self.assertTrue(all(p.computable for p in res.points))
        self.assertTrue(all(p.variance == 0.0 for p in res.points))
        self.assertTrue(all(not p.eligible_for_fit for p in res.points))
        self.assertIsNone(res.fit)


class TestAppendValidationAndAtomicity(unittest.TestCase):
    def test_invalid_batch_leaves_state_unchanged(self):
        acc = ad.streaming_allan(1.0, [1, 2])
        acc.append([1.0, 2.0, 3.0])
        before = acc.snapshot()
        with self.assertRaises(ad.AllanValueError):
            acc.append([1.0, float("nan")])
        with self.assertRaises(ad.AllanTypeError):
            acc.append([True, 2.0])
        with self.assertRaises(ad.AllanTypeError):
            acc.append("ab")
        with self.assertRaises(ad.AllanValueError):
            acc.append([])
        after = acc.snapshot()
        self.assertEqual(after.n_samples, before.n_samples)
        self.assertEqual(
            [p.count for p in after.points], [p.count for p in before.points])
        self.assertEqual(
            [p.variance for p in after.points], [p.variance for p in before.points])

    def test_constructor_validation(self):
        with self.assertRaises(ad.AllanTypeError):
            ad.streaming_allan(True, [1])
        with self.assertRaises(ad.AllanValueError):
            ad.streaming_allan(0.0, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.streaming_allan(1.0, [1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.streaming_allan(1.0, [0])
        with self.assertRaises(ad.AllanValueError):
            ad.streaming_allan(1.0, [])

    def test_close_blocks_append_but_snapshot_ok(self):
        acc = ad.streaming_allan(1.0, [1])
        acc.append(synth_mixture(50, 1))
        acc.close()
        self.assertTrue(acc.closed)
        with self.assertRaises(ad.StreamingClosedError):
            acc.append([1.0])
        self.assertEqual(acc.snapshot().n_samples, 50)
        self.assertIsInstance(ad.StreamingClosedError("x"), ad.AllanError)
        self.assertIsInstance(ad.StreamingClosedError("x"), RuntimeError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
