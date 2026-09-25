"""OnlineAllan 在线累积器的必要测试。

参考实现完全独立于被测核心：
  * naive_allan_variance —— 朴素滑动窗口求和，双重循环直接按定义计算
    （窗口求和用 math.fsum 精确求和），不调用任何被测代码。

被测核心只用于取得实际输出，期望值一律来自参考或与批量入口 analyze
对同一完整前缀的结果对照。
"""

import math
import os
import random
import sys
import tracemalloc
import unittest

# `python -m unittest discover -s tests` 以 tests/ 为顶层目录，需把仓库根加入路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402


# ---------------------------------------------------------------------------
# 独立参考：朴素窗口求和的重叠 Allan 方差（双重循环，不用任何前缀和/被测代码）
# ---------------------------------------------------------------------------


def naive_allan_variance(samples, m):
    n = len(samples)
    count = n - 2 * m + 1
    if count < 2:
        return None, max(0, count)
    total = 0.0
    for k in range(count):
        d = math.fsum(samples[k + m:k + 2 * m]) / m - math.fsum(samples[k:k + m]) / m
        total += d * d
    return total / (2.0 * (count - 1)), count


def synth(n, seed, noise_sd=1.0, ramp_per_sample=0.0):
    rng = random.Random(seed)
    return [noise_sd * rng.gauss(0.0, 1.0) + ramp_per_sample * i
            for i in range(n)]


def feed_in_chunks(acc, samples, chunk_sizes):
    """按循环的块尺寸序列喂入，返回每次喂入后的前缀长度列表。"""
    i = 0
    k = 0
    prefixes = []
    while i < len(samples):
        c = chunk_sizes[k % len(chunk_sizes)]
        k += 1
        acc.feed(samples[i:i + c])
        i += c
        prefixes.append(i)
    return prefixes


SIZES = [1, 2, 3, 5, 8, 13, 21]


class TestSinglePointEveryPrefix(unittest.TestCase):
    """同一合成信号逐点喂入：每个前缀都与独立双循环公式比较。"""

    def test_every_prefix_matches_naive(self):
        samples = synth(320, 11, ramp_per_sample=0.01)
        acc = ad.OnlineAllan(0.01, SIZES)
        for t, x in enumerate(samples):
            self.assertEqual(acc.feed([x]), 1)
            self.assertEqual(acc.n_samples, t + 1)
            for p, m in zip(acc.query_points(), SIZES):
                ref_var, ref_count = naive_allan_variance(samples[:t + 1], m)
                self.assertEqual(p.count, ref_count, (t, m))
                self.assertEqual(p.tau, m * 0.01)
                if ref_var is None:
                    self.assertIsNone(p.variance, (t, m))
                    self.assertIsNone(p.adev)
                else:
                    rel = abs(p.variance - ref_var) / ref_var
                    self.assertLess(rel, 1e-12, (t, m, p.variance, ref_var))
                    self.assertAlmostEqual(p.adev, math.sqrt(p.variance))


class TestChunkingEquivalence(unittest.TestCase):
    """同一信号按单点、素数块、整块送入：结果一致且与参考/批量入口一致。"""

    def setUp(self):
        self.samples = synth(700, 23, ramp_per_sample=0.01)
        self.sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    def _feed_all(self, mode):
        acc = ad.OnlineAllan(0.01, self.sizes)
        if mode == "single":
            for x in self.samples:
                acc.feed([x])
        elif mode == "prime":
            feed_in_chunks(acc, self.samples,
                           [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
        elif mode == "block":
            acc.feed(self.samples)
        else:
            raise AssertionError(mode)
        self.assertEqual(acc.n_samples, len(self.samples))
        return acc

    def test_prime_chunks_match_naive_at_each_boundary(self):
        acc = ad.OnlineAllan(0.01, self.sizes)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        i = 0
        k = 0
        while i < len(self.samples):
            c = primes[k % len(primes)]
            k += 1
            acc.feed(self.samples[i:i + c])
            i += c
            for p, m in zip(acc.query_points(), self.sizes):
                ref_var, ref_count = naive_allan_variance(self.samples[:i], m)
                self.assertEqual(p.count, ref_count, (i, m))
                if ref_var is None:
                    self.assertIsNone(p.variance)
                else:
                    rel = abs(p.variance - ref_var) / ref_var
                    self.assertLess(rel, 1e-12, (i, m, p.variance, ref_var))

    def test_three_feed_modes_agree(self):
        accs = [self._feed_all(mode) for mode in ("single", "prime", "block")]
        cols = [acc.query_points() for acc in accs]
        for triple in zip(*cols):
            p0 = triple[0]
            for p in triple[1:]:
                self.assertEqual(p.count, p0.count)
                if p0.variance is None:
                    self.assertIsNone(p.variance)
                else:
                    self.assertLess(
                        abs(p.variance - p0.variance) / p0.variance, 1e-12)

    def test_block_feed_matches_batch_analyze(self):
        acc = self._feed_all("block")
        res = ad.analyze(self.samples, 0.01, self.sizes)
        q = acc.query()
        self.assertEqual(q.n_samples, res.n_samples)
        self.assertEqual(q.dt, res.dt)
        for p, rp in zip(q.points, res.points):
            self.assertEqual(p.count, rp.count)
            self.assertEqual(p.eligible_for_fit, rp.eligible_for_fit)
            if rp.variance is None:
                self.assertIsNone(p.variance)
            else:
                self.assertLess(
                    abs(p.variance - rp.variance) / rp.variance, 1e-12)
        # 分段拟合结果与批量入口一致（断点完全相同，目标值在数值误差内）。
        self.assertEqual(q.fit.breakpoints, res.fit.breakpoints)
        self.assertEqual(q.fit.segment_count, res.fit.segment_count)
        self.assertAlmostEqual(
            q.fit.objective, res.fit.objective,
            delta=1e-9 * max(1.0, abs(res.fit.objective)))
        for s, rs in zip(q.fit.segments, res.fit.segments):
            self.assertAlmostEqual(s.slope, rs.slope, delta=1e-9)
            self.assertAlmostEqual(s.intercept, rs.intercept, delta=1e-9)


class TestConstantAndDCOffset(unittest.TestCase):
    def test_constant_signal_zero_variance(self):
        acc = ad.OnlineAllan(1.0, [1, 2, 4, 8])
        feed_in_chunks(acc, [3.5] * 500, [7, 13, 3])
        points = acc.query_points()
        self.assertEqual([p.count for p in points], [499, 497, 493, 485])
        # 常量信号：方差精确为 0（残差全为 0），无合格拟合点。
        self.assertTrue(all(p.variance == 0.0 for p in points))
        self.assertTrue(all(not p.eligible_for_fit for p in points))
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            acc.query()
        self.assertEqual(len(cm.exception.points), 4)
        # 与批量入口对同一前缀的行为一致。
        with self.assertRaises(ad.FitInfeasibleError):
            ad.analyze([3.5] * 500, 1.0, [1, 2, 4, 8])

    def test_large_dc_offset_with_small_variation(self):
        # 大直流偏置叠小变化：binary64 表示极限决定可达精度（同批量测试口径）。
        rng = random.Random(77)
        noise = [rng.gauss(0.0, 1.0) for _ in range(2000)]
        sizes = [1, 4, 16, 64]
        for base, bound in ((1e6, 1e-9), (1e9, 1e-7)):
            samples = [base + g for g in noise]
            acc = ad.OnlineAllan(0.01, sizes)
            feed_in_chunks(acc, samples, [7])
            for p, m in zip(acc.query_points(), sizes):
                ref_var, ref_count = naive_allan_variance(samples, m)
                self.assertEqual(p.count, ref_count)
                self.assertGreater(p.variance, 0.0)
                self.assertLess(abs(p.variance - ref_var) / ref_var, bound,
                                (base, m, p.variance, ref_var))


class TestBoundaryExactlyTwoM(unittest.TestCase):
    """刚够 2m 个点的边界：count=1 时方差为 None，不补零凑窗口。"""

    def test_exactly_2m_then_2m_plus_1(self):
        sizes = [3, 7]
        acc = ad.OnlineAllan(1.0, sizes)
        samples = [float(i % 5) for i in range(15)]
        acc.feed(samples[:14])  # 恰为 2*m_max
        points = acc.query_points()
        self.assertEqual([(p.m, p.count) for p in points], [(3, 9), (7, 1)])
        self.assertIsNotNone(points[0].variance)
        self.assertIsNone(points[1].variance)  # count=1 无法无偏估计
        self.assertIsNone(points[1].adev)
        acc.feed(samples[14:])  # 2*m_max + 1
        points = acc.query_points()
        self.assertEqual([(p.m, p.count) for p in points], [(3, 10), (7, 2)])
        for p, m in zip(points, sizes):
            ref_var, ref_count = naive_allan_variance(samples, m)
            self.assertEqual(p.count, ref_count)
            self.assertAlmostEqual(p.variance, ref_var, delta=1e-15 * ref_var)

    def test_no_zero_padding_of_short_tail(self):
        # N = 2m + 3：差分数恰为 N-2m+1=4，不足窗口的尾部不参与也不补零。
        acc = ad.OnlineAllan(1.0, [5])
        samples = synth(13, 3)
        acc.feed(samples)
        (p,) = acc.query_points()
        self.assertEqual(p.count, 13 - 10 + 1)
        ref_var, _ = naive_allan_variance(samples, 5)
        self.assertAlmostEqual(p.variance, ref_var, delta=1e-15 * ref_var)


class TestLongStreamBeyondBatchLimit(unittest.TestCase):
    """样本数远超单次批量上限 MAX_SAMPLES，内存不随历史增长。"""

    def test_long_stream_matches_naive(self):
        n = 60000
        self.assertGreater(n, ad.MAX_SAMPLES)
        samples = synth(n, 99, ramp_per_sample=0.005)
        sizes = [1, 2, 4, 8, 16, 32, 64]
        acc = ad.OnlineAllan(0.01, sizes)
        feed_in_chunks(acc, samples, [997])  # 素数块
        self.assertEqual(acc.n_samples, n)
        self.assertEqual(acc.retained_samples, 2 * max(sizes))
        for p, m in zip(acc.query_points(), sizes):
            ref_var, ref_count = naive_allan_variance(samples, m)
            self.assertEqual(p.count, ref_count)
            self.assertLess(abs(p.variance - ref_var) / ref_var, 1e-10,
                            (m, p.variance, ref_var))

    def test_prefix_8192_matches_batch_analyze(self):
        # 长流的前 MAX_SAMPLES 个样本与批量入口对同一前缀一致。
        samples = synth(ad.MAX_SAMPLES, 31, ramp_per_sample=0.005)
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        acc = ad.OnlineAllan(1.0, sizes)
        feed_in_chunks(acc, samples, [613])
        res = ad.analyze(samples, 1.0, sizes)
        q = acc.query()
        for p, rp in zip(q.points, res.points):
            self.assertEqual(p.count, rp.count)
            self.assertLess(abs(p.variance - rp.variance) / rp.variance, 1e-11)
        self.assertEqual(q.fit.breakpoints, res.fit.breakpoints)
        self.assertAlmostEqual(
            q.fit.objective, res.fit.objective,
            delta=1e-9 * max(1.0, abs(res.fit.objective)))

    def test_memory_does_not_grow_with_history(self):
        sizes = [1, 4, 16, 64, 256]
        acc = ad.OnlineAllan(0.01, sizes)
        rng = random.Random(5)
        chunk = [rng.gauss(0.0, 1.0) for _ in range(1000)]
        tracemalloc.start()
        try:
            for _ in range(20):
                acc.feed(chunk)
            base = tracemalloc.get_traced_memory()[0]
            retained = acc.retained_samples
            for _ in range(200):
                acc.feed(chunk)
            growth = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()
        self.assertEqual(acc.n_samples, 220000)
        # 保留窗口固定在 2*m_max，与历史长度无关。
        self.assertEqual(acc.retained_samples, retained)
        self.assertEqual(acc.retained_samples, 2 * 256)
        # 10 倍历史后 Python 堆当前占用不增长（允许微小分配器噪声）。
        self.assertLess(growth, 50_000)


class TestInvalidChunkAtomic(unittest.TestCase):
    """非法 chunk 先整块验证：失败不能吃掉其中前几项。"""

    def setUp(self):
        self.acc = ad.OnlineAllan(0.01, [1, 2, 4])
        self.acc.feed(synth(100, 8))
        self.before = self.acc.query_points()
        self.n0 = self.acc.n_samples

    def _assert_state_intact(self):
        self.assertEqual(self.acc.n_samples, self.n0)
        self.assertEqual(self.acc.query_points(), self.before)

    def test_nan_in_chunk_rejects_whole_chunk(self):
        with self.assertRaises(ad.AllanValueError) as cm:
            self.acc.feed([1.0, 2.0, float("nan"), 4.0])
        self.assertIn("chunk[2]", str(cm.exception))
        self._assert_state_intact()

    def test_inf_and_huge_int_rejected(self):
        with self.assertRaises(ad.AllanValueError):
            self.acc.feed([1.0, float("inf")])
        with self.assertRaises(ad.AllanValueError):
            self.acc.feed([10 ** 400])
        self._assert_state_intact()

    def test_type_errors_rejected(self):
        with self.assertRaises(ad.AllanTypeError):
            self.acc.feed([1.0, True])       # bool 不当作数值样本
        with self.assertRaises(ad.AllanTypeError):
            self.acc.feed([1.0, "2.0"])
        with self.assertRaises(ad.AllanTypeError):
            self.acc.feed("abc")             # str 不是样本序列
        with self.assertRaises(ad.AllanTypeError):
            self.acc.feed(3.5)               # 标量不是序列
        with self.assertRaises(ad.AllanTypeError):
            self.acc.feed(None)
        self._assert_state_intact()

    def test_empty_chunk_is_noop(self):
        self.assertEqual(self.acc.feed([]), 0)
        self._assert_state_intact()

    def test_feed_continues_after_rejection(self):
        try:
            self.acc.feed([1.0, float("nan")])
        except ad.AllanValueError:
            pass
        more = synth(50, 9)
        self.assertEqual(self.acc.feed(more), 50)
        self.assertEqual(self.acc.n_samples, self.n0 + 50)


class TestImmutability(unittest.TestCase):
    """创建后不能改 dt 和尺度。"""

    def test_dt_and_sizes_readonly(self):
        sizes = [1, 2, 4, 8]
        acc = ad.OnlineAllan(0.01, sizes)
        with self.assertRaises(AttributeError):
            acc.dt = 0.5
        with self.assertRaises(AttributeError):
            acc.cluster_sizes = [1, 2]
        with self.assertRaises(AttributeError):
            acc.n_samples = 0
        # cluster_sizes 以 tuple 保存，不可原地修改。
        self.assertIsInstance(acc.cluster_sizes, tuple)
        with self.assertRaises(AttributeError):
            acc.cluster_sizes.append(16)

    def test_constructor_copies_sizes(self):
        sizes = [1, 2, 4]
        acc = ad.OnlineAllan(0.01, sizes)
        sizes.append(999)
        sizes[0] = 777
        self.assertEqual(acc.cluster_sizes, (1, 2, 4))

    def test_constructor_validation_matches_batch(self):
        with self.assertRaises(ad.AllanValueError):
            ad.OnlineAllan(0.0, [1, 2])
        with self.assertRaises(ad.AllanTypeError):
            ad.OnlineAllan(True, [1, 2])
        with self.assertRaises(ad.AllanValueError):
            ad.OnlineAllan(1.0, [])
        with self.assertRaises(ad.AllanTypeError):
            ad.OnlineAllan(1.0, [1, True])
        with self.assertRaises(ad.AllanTypeError):
            ad.OnlineAllan(1.0, [1.0, 2])
        with self.assertRaises(ad.AllanValueError):
            ad.OnlineAllan(1.0, [0])
        with self.assertRaises(ad.AllanValueError):
            ad.OnlineAllan(1.0, [1] * (ad.MAX_CLUSTER_SIZES + 1))


class TestQueryDoesNotMutate(unittest.TestCase):
    """query / query_points（含分段拟合）不得改变累积状态。"""

    def test_repeated_queries_are_pure(self):
        samples = synth(400, 17, ramp_per_sample=0.01)
        acc = ad.OnlineAllan(1.0, [1, 2, 4, 8, 16])
        acc.feed(samples)
        p0 = acc.query_points()
        q1 = acc.query()
        q2 = acc.query()
        self.assertEqual(acc.query_points(), p0)
        self.assertEqual(q1, q2)
        self.assertEqual(acc.n_samples, 400)
        self.assertEqual(acc.retained_samples, 32)
        # 查询后继续喂入，结果仍与参考一致。
        extra = synth(100, 18)
        acc.feed(extra)
        full = samples + extra
        for p, m in zip(acc.query_points(), [1, 2, 4, 8, 16]):
            ref_var, ref_count = naive_allan_variance(full, m)
            self.assertEqual(p.count, ref_count)
            self.assertLess(abs(p.variance - ref_var) / ref_var, 1e-12)

    def test_infeasible_query_leaves_state_intact(self):
        acc = ad.OnlineAllan(1.0, [1, 2, 3])
        samples = synth(20, 4)
        acc.feed(samples)
        before = acc.query_points()
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            acc.query()
        # 异常附带当前逐 m 统计，且与 query_points 一致。
        self.assertEqual(cm.exception.points, before)
        self.assertEqual(acc.query_points(), before)
        self.assertEqual(acc.n_samples, 20)
        # 继续喂入后累积不中断：与一次性喂入的长前缀结果一致。
        more = synth(200, 6)
        acc.feed(more)
        acc2 = ad.OnlineAllan(1.0, [1, 2, 3])
        acc2.feed(samples + more)
        for p, q in zip(acc.query_points(), acc2.query_points()):
            self.assertEqual(p.count, q.count)
            self.assertAlmostEqual(p.variance, q.variance,
                                   delta=1e-15 * q.variance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
