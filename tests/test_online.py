"""AllanAccumulator 在线累积器的必要测试。

参考实现完全独立于被测核心：
  * naive_allan_variance —— 朴素滑动窗口 fsum 双重循环，直接按定义计算；
  * 离线 ad.analyze 仅作为"同一前缀两种入口互查"的补充对照，
    期望值一律来自朴素参考或解析结论。

覆盖：单点/素数块/整块三种送样在每个前缀对照朴素参考；常量信号、
大直流偏置叠小变化、刚够 2m 个点的边界；非法 chunk 整块拒绝且无部分
变更；dt/尺度创建后不可改；query 不改累积状态；内存不随历史长度增长。
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


def naive_allan_variance(samples, m):
    """按定义双重循环计算（与 tests/test_allan.py 的参考同一写法，独立于核心）。"""
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
    return [noise_sd * rng.gauss(0.0, 1.0) + ramp_per_sample * i for i in range(n)]


def primes_upto(limit):
    out = []
    for c in range(2, limit + 1):
        if all(c % p for p in out if p * p <= c):
            out.append(c)
    return out


def feed_in_chunks(acc, samples, chunk_sizes):
    """按给定块尺寸序列（循环使用）把 samples 送入 acc，返回每次 feed 后的前缀长度。"""
    prefixes = []
    i = 0
    k = 0
    while i < len(samples):
        c = chunk_sizes[k % len(chunk_sizes)]
        k += 1
        j = min(i + c, len(samples))
        acc.feed(samples[i:j])
        prefixes.append(j)
        i = j
    return prefixes


def current_points(acc):
    """取当前前缀的逐 m 统计；拟合无解时从异常附带信息取（与离线口径一致）。"""
    try:
        return acc.query().points
    except ad.FitInfeasibleError as exc:
        return exc.points


class TestPrefixConsistency(unittest.TestCase):
    """同一合成信号按单点、素数块、整块送入，逐前缀对照朴素双循环参考。"""

    SIZES = [1, 2, 3, 5, 8, 13, 50, 140]
    DT = 0.01

    def _check_prefix(self, points, prefix, n):
        for pt, m in zip(points, self.SIZES):
            ref_var, ref_count = naive_allan_variance(prefix, m)
            self.assertEqual(pt.count, ref_count, (n, m))
            self.assertEqual(pt.tau, m * self.DT)
            if ref_var is None:
                self.assertIsNone(pt.variance, (n, m))
                self.assertIsNone(pt.adev)
            else:
                rel = abs(pt.variance - ref_var) / ref_var
                # 补偿滑动窗口和 vs 朴素 fsum 双重循环：O(1) 数据相对差 1e-12 内。
                self.assertLess(rel, 1e-12, (n, m, pt.variance, ref_var))
                self.assertAlmostEqual(pt.adev, math.sqrt(pt.variance))

    def test_single_point_feed_every_prefix(self):
        samples = synth(300, 4242, noise_sd=2.0, ramp_per_sample=0.01)
        acc = ad.AllanAccumulator(self.DT, self.SIZES)
        for i, v in enumerate(samples):
            acc.feed([v])
            self.assertEqual(acc.n_samples, i + 1)
            self._check_prefix(current_points(acc), samples[:i + 1], i + 1)

    def test_prime_chunk_feed_every_prefix(self):
        samples = synth(300, 4242, noise_sd=2.0, ramp_per_sample=0.01)
        acc = ad.AllanAccumulator(self.DT, self.SIZES)
        chunk_sizes = primes_upto(50)
        i = 0
        k = 0
        while i < len(samples):
            j = min(i + chunk_sizes[k % len(chunk_sizes)], len(samples))
            k += 1
            acc.feed(samples[i:j])
            self.assertEqual(acc.n_samples, j)
            self._check_prefix(current_points(acc), samples[:j], j)
            i = j

    def test_whole_block_feed(self):
        samples = synth(300, 4242, noise_sd=2.0, ramp_per_sample=0.01)
        acc = ad.AllanAccumulator(self.DT, self.SIZES)
        acc.feed(samples)
        self._check_prefix(current_points(acc), samples, len(samples))

    def test_chunking_pattern_does_not_change_state(self):
        # 逐样本更新与分块方式无关：三种送法的最终累积结果逐位相同。
        samples = synth(300, 4242, noise_sd=2.0, ramp_per_sample=0.01)
        acc_single = ad.AllanAccumulator(self.DT, self.SIZES)
        for v in samples:
            acc_single.feed([v])
        acc_prime = ad.AllanAccumulator(self.DT, self.SIZES)
        feed_in_chunks(acc_prime, samples, primes_upto(50))
        acc_block = ad.AllanAccumulator(self.DT, self.SIZES)
        acc_block.feed(samples)
        results = [acc_single.query(), acc_prime.query(), acc_block.query()]
        for r in results[1:]:
            for pa, pb in zip(results[0].points, r.points):
                self.assertEqual(pa, pb)  # 不可变 dataclass 全字段相等（含浮点逐位）
            self.assertEqual(results[0].fit, r.fit)

    def test_matches_batch_analyze_on_prefixes(self):
        # 补充互查：在线 query 与离线 analyze 作用同一完整前缀（含拟合结果）。
        samples = synth(300, 99, noise_sd=2.0, ramp_per_sample=0.01)
        acc = ad.AllanAccumulator(self.DT, self.SIZES)
        acc.feed(samples)
        online = acc.query()
        batch = ad.analyze(samples, self.DT, self.SIZES)
        self.assertEqual(online.n_samples, batch.n_samples)
        self.assertEqual(online.dt, batch.dt)
        for po, pb in zip(online.points, batch.points):
            self.assertEqual(po.count, pb.count)
            self.assertLess(abs(po.variance - pb.variance) / pb.variance, 1e-12)
        self.assertLess(
            abs(online.fit.objective - batch.fit.objective),
            1e-9 * max(1.0, abs(batch.fit.objective)),
        )
        self.assertEqual(online.fit.breakpoints, batch.fit.breakpoints)
        self.assertEqual(online.fit.segment_count, batch.fit.segment_count)


class TestSpecialSignals(unittest.TestCase):
    def test_constant_signal_zero_variance(self):
        # 常量信号：所有差分精确为 0，方差精确 0，拟合无解（与离线口径一致）。
        acc = ad.AllanAccumulator(0.01, [1, 2, 4, 8])
        feed_in_chunks(acc, [3.5] * 500, [7, 128, 3])
        with self.assertRaises(ad.FitInfeasibleError) as cm:
            acc.query()
        points = cm.exception.points
        self.assertEqual(len(points), 4)
        self.assertTrue(all(p.variance == 0.0 for p in points))
        self.assertTrue(all(p.count >= 20 for p in points))
        self.assertTrue(all(not p.eligible_for_fit for p in points))

    def test_large_dc_bias_small_variation(self):
        # 大直流偏置叠小变化：与朴素 fsum 参考按表示极限分档（同离线测试口径）。
        rng = random.Random(77)
        noise = [rng.gauss(0.0, 1.0) for _ in range(400)]
        sizes = [1, 4, 16, 64]
        for base, bound in ((1e6, 1e-9), (1e9, 1e-7)):
            samples = [base + g for g in noise]
            acc = ad.AllanAccumulator(0.01, sizes)
            feed_in_chunks(acc, samples, [13, 7, 29])
            points = current_points(acc)
            for pt, m in zip(points, sizes):
                ref_var, ref_count = naive_allan_variance(samples, m)
                self.assertEqual(pt.count, ref_count)
                self.assertGreater(pt.variance, 0.0)
                self.assertLess(abs(pt.variance - ref_var) / ref_var, bound,
                                (base, m, pt.variance, ref_var))

    def test_exactly_2m_boundary(self):
        # 刚够 2m 个点：count=1，无法无偏估计，variance 为 None；再来一个点 count=2。
        samples = synth(9, 5)
        acc = ad.AllanAccumulator(1.0, [4])
        acc.feed(samples[:8])  # N = 2m
        (pt,) = current_points(acc)
        self.assertEqual(pt.count, 1)
        self.assertIsNone(pt.variance)
        self.assertIsNone(pt.adev)
        acc.feed(samples[8:9])  # N = 2m + 1
        (pt,) = current_points(acc)
        self.assertEqual(pt.count, 2)
        ref_var, _ = naive_allan_variance(samples[:9], 4)
        self.assertLess(abs(pt.variance - ref_var) / ref_var, 1e-12)

    def test_beyond_batch_sample_limit(self):
        # 样本数远超批量上限 8192：在线入口不受 MAX_SAMPLES 约束。
        samples = synth(9000, 11, noise_sd=1.5, ramp_per_sample=0.005)
        sizes = [1, 2, 8, 64, 512]
        self.assertGreater(len(samples), ad.MAX_SAMPLES)
        acc = ad.AllanAccumulator(0.02, sizes)
        feed_in_chunks(acc, samples, [997])
        self.assertEqual(acc.n_samples, 9000)
        for pt, m in zip(current_points(acc), sizes):
            ref_var, ref_count = naive_allan_variance(samples, m)
            self.assertEqual(pt.count, ref_count)
            self.assertLess(abs(pt.variance - ref_var) / ref_var, 1e-12, m)


class TestChunkValidationAtomic(unittest.TestCase):
    def setUp(self):
        self.samples = synth(120, 8)
        self.sizes = [1, 2, 4, 8]

    def test_invalid_chunk_rejected_as_whole(self):
        acc = ad.AllanAccumulator(1.0, self.sizes)
        acc.feed(self.samples[:60])
        before = [p.variance for p in current_points(acc)]
        n_before = acc.n_samples
        # NaN 在块尾：整块拒绝，前面的合法元素也不能被吃掉。
        with self.assertRaises(ad.AllanValueError) as cm:
            acc.feed([1.0, 2.0, float("nan")])
        self.assertIn("chunk[2]", str(cm.exception))
        self.assertEqual(acc.n_samples, n_before)
        self.assertEqual([p.variance for p in current_points(acc)], before)
        # 失败后再送合法数据，结果与从未发生失败完全一致（逐位）。
        acc.feed(self.samples[60:])
        fresh = ad.AllanAccumulator(1.0, self.sizes)
        fresh.feed(self.samples)
        self.assertEqual(current_points(acc), current_points(fresh))

    def test_chunk_type_and_value_errors(self):
        acc = ad.AllanAccumulator(1.0, self.sizes)
        acc.feed(self.samples[:40])
        n_before = acc.n_samples
        with self.assertRaises(ad.AllanTypeError):
            acc.feed([1.0, True])          # bool 不当作数值样本
        with self.assertRaises(ad.AllanTypeError):
            acc.feed([1.0, "2.0"])
        with self.assertRaises(ad.AllanTypeError):
            acc.feed("abc")                # 字符串不是样本序列
        with self.assertRaises(ad.AllanTypeError):
            acc.feed(3.5)                  # 非标量
        with self.assertRaises(ad.AllanValueError):
            acc.feed([float("inf")])
        with self.assertRaises(ad.AllanValueError):
            acc.feed([10 ** 400])          # 无法表示为有限 float
        self.assertEqual(acc.n_samples, n_before)

    def test_empty_chunk_is_noop(self):
        acc = ad.AllanAccumulator(1.0, self.sizes)
        acc.feed(self.samples[:10])
        self.assertEqual(acc.feed([]), 10)
        self.assertEqual(acc.n_samples, 10)


class TestImmutabilityAndStatelessQuery(unittest.TestCase):
    def test_dt_and_sizes_frozen_after_creation(self):
        sizes = [1, 2, 4]
        acc = ad.AllanAccumulator(0.5, sizes)
        sizes.append(8)  # 修改传入的 list 不影响累积器
        self.assertEqual(acc.cluster_sizes, (1, 2, 4))
        self.assertIsInstance(acc.cluster_sizes, tuple)
        self.assertEqual(acc.dt, 0.5)
        with self.assertRaises(AttributeError):
            acc.dt = 0.1
        with self.assertRaises(AttributeError):
            acc.cluster_sizes = (1,)

    def test_constructor_validation_matches_batch(self):
        with self.assertRaises(ad.AllanValueError):
            ad.AllanAccumulator(0.0, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.AllanAccumulator(True, [1])
        with self.assertRaises(ad.AllanTypeError):
            ad.AllanAccumulator(1.0, [True])
        with self.assertRaises(ad.AllanTypeError):
            ad.AllanAccumulator(1.0, [1.0])
        with self.assertRaises(ad.AllanValueError):
            ad.AllanAccumulator(1.0, [0])
        with self.assertRaises(ad.AllanValueError):
            ad.AllanAccumulator(1.0, [])
        with self.assertRaises(ad.AllanValueError):
            ad.AllanAccumulator(1.0, [10 ** 310])  # m*dt 溢出有限 float

    def test_query_does_not_mutate_state(self):
        samples = synth(200, 3, ramp_per_sample=0.02)
        acc = ad.AllanAccumulator(1.0, [1, 2, 4, 8, 16])
        acc.feed(samples)
        r1 = acc.query()
        r2 = acc.query()
        self.assertEqual(r1, r2)  # 连续两次 query 逐位相同
        # 交替 query/feed 与一路只 feed 的最终状态逐位一致。
        acc_a = ad.AllanAccumulator(1.0, [1, 2, 4, 8, 16])
        acc_b = ad.AllanAccumulator(1.0, [1, 2, 4, 8, 16])
        extra = synth(80, 6)
        for v in extra:
            acc_a.feed([v])
            try:
                acc_a.query()
            except ad.FitInfeasibleError:
                pass  # 早期前缀拟合无解是正常口径，query 本身不得改动状态
        acc_b.feed(extra)
        self.assertEqual(acc_a.query(), acc_b.query())


class TestMemoryBounded(unittest.TestCase):
    def test_memory_does_not_grow_with_history(self):
        # 只保留 2*max(m) 滑动样本与每尺度常数累积量：历史变长内存不变。
        sizes = [1, 2, 4, 8, 16, 32, 64]
        acc = ad.AllanAccumulator(0.01, sizes)
        rng = random.Random(2026)

        def feed_block(n):
            fed = 0
            for _ in range(n // 997):
                acc.feed([rng.gauss(0.0, 1.0) for _ in range(997)])
                fed += 997
            return fed

        n_warm = feed_block(50_000)  # 预热，环形缓冲等结构已就位
        self.assertEqual(len(acc._ring), 2 * max(sizes))
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        n_more = feed_block(100_000)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        growth = sum(s.size_diff for s in after.compare_to(before, "filename"))
        self.assertEqual(acc.n_samples, n_warm + n_more)
        self.assertGreater(acc.n_samples, ad.MAX_SAMPLES)
        self.assertEqual(len(acc._ring), 2 * max(sizes))  # 滑动窗口不随历史变长
        # 允许少量解释器内部簿记，但量级必须远小于历史长度对应的尺寸。
        self.assertLess(growth, 64 * 1024, f"memory grew by {growth} bytes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
