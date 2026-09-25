"""固定输入演示：合成角速率数据的 Allan 误差分解。

运行：python demo.py（约几秒，离线、确定性、无第三方依赖）。

信号模型（确定性代码生成，固定种子）：
    omega[i] = sigma * w[i] + a * i,  w[i] 为高斯白噪声，a*i 为恒定角加速度漂移。
理论形态（以样本数 m 表示的离散结果，tau = m*dt）：
    白噪声: AVAR(m) = sigma^2 / m          -> log-log 斜率 -1/2
    恒斜坡: AVAR(m) = a^2 * m^2 / 2        -> log-log 斜率 +1
交叉点 m* = (2*sigma^2/a^2)^(1/3)（连续时间写法：N^2/tau 与 alpha^2*tau^2/2，
其中 N=sigma*sqrt(dt)、alpha=a/dt）。斜率只是本批离线数据的经验形态描述，
不是硬件指标保证。
"""

import os
import random
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allan_deviation as ad  # noqa: E402

N = 2048
DT = 0.01            # 秒
SIGMA = 1.0          # 白噪声角速率标准差（任意角速度单位）
ACCEL = 0.01         # 恒定角加速度（同单位/样本斜率）
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def build_samples():
    rng = random.Random(20260922)
    return [SIGMA * rng.gauss(0.0, 1.0) + ACCEL * i for i in range(N)]


def main():
    samples = build_samples()
    t0 = time.perf_counter()
    result = ad.analyze(samples, DT, SIZES)
    elapsed = time.perf_counter() - t0

    print("=" * 72)
    print("固定速率单轴陀螺仪 Allan 误差分解（离线经验分解，非硬件指标保证）")
    print(f"样本数 N={N}, dt={DT}s, 时长={N * DT:.1f}s, 核心计算耗时 {elapsed:.3f}s")
    print("-" * 72)
    print(f"{'m':>5} {'tau(s)':>10} {'count':>6} {'Allan方差':>14} {'Allan标准差':>14} 入拟合")
    for p in result.points:
        if p.variance is None:
            print(f"{p.m:>5} {p.tau:>10.4g} {p.count:>6} {'--(差分<2)':>14} "
                  f"{'--':>14} {'否':>4}")
        else:
            mark = "是" if p.eligible_for_fit else "否(count<20)"
            print(f"{p.m:>5} {p.tau:>10.4g} {p.count:>6} {p.variance:>14.6g} "
                  f"{p.adev:>14.6g} {mark:>10}")

    fit = result.fit
    print("-" * 72)
    print(f"log-log 连续分段直线：{fit.segment_count} 段，"
          f"目标值 SSE+(p-1)*罚项 = {fit.objective:.6g} "
          f"(SSE={fit.sse:.6g}, RMSE={fit.rmse:.4g}, 罚项={ad.SEGMENT_PENALTY})")
    for i, seg in enumerate(fit.segments):
        print(f"  段{i + 1}: tau∈[{seg.tau_start:.3g},{seg.tau_end:.3g}]s "
              f"(m∈[{seg.m_start},{seg.m_end}], {seg.n_points}点)  "
              f"斜率={seg.slope:+.3f} 截距={seg.intercept:+.4f}")
    if fit.breakpoints:
        print("  断点（右侧段首点）: "
              + ", ".join(f"m={bp.m},tau={bp.tau:.3g}s" for bp in fit.breakpoints))

    small, large = fit.segments[0], fit.segments[-1]
    print("-" * 72)
    print("经验区间辨识（仅描述本批数据形态）:")
    print(f"  小 tau 段斜率 {small.slope:+.3f}（白噪声区理论约 -0.5，"
          f"实测偏差 {small.slope - (-0.5):+.3f}）")
    print(f"  大 tau 段斜率 {large.slope:+.3f}（漂移区理论约 +1.0，"
          f"实测偏差 {large.slope - 1.0:+.3f}）")
    cross_m = (2.0 * SIGMA * SIGMA / (ACCEL * ACCEL)) ** (1.0 / 3.0)
    if fit.breakpoints:
        print(f"  实测断点 m={fit.breakpoints[0].m} (tau={fit.breakpoints[0].tau:.3g}s)，"
              f"理论交叉点 m*≈{cross_m:.1f} (tau≈{cross_m * DT:.3f}s)")
    print("=" * 72)

    # ---- 边界：真实触发一个非法输入拒绝（不静默纠正、可定位）----
    print("\n边界演示：传入含 NaN 的样本与 bool 类型的 m，必须被拒绝：")
    try:
        ad.analyze([1.0, float("nan"), 3.0], DT, [1, 2])
    except ad.AllanValueError as exc:
        print(f"  NaN 样本 -> {type(exc).__name__}: {exc}")
    try:
        ad.analyze(samples[:100], DT, [1, True, 4])
    except ad.AllanTypeError as exc:
        print(f"  bool 类型 m -> {type(exc).__name__}: {exc}")

    # ---- 边界：短数据无法拟合（证明无解，区别于预算耗尽）----
    # N=20: m=1 时 count=N-2+1=19 < 20，没有任何合格点 -> 证明无解。
    try:
        ad.analyze(samples[:20], DT, [1, 2, 3])
    except ad.FitInfeasibleError as exc:
        print(f"  短数据(N=20, count<20) -> {type(exc).__name__}: {exc.reason}")
    print("（以上拒绝均为实际计算路径触发，非预录文本；无异常被吞掉。）")

    # ==================================================================
    # 在线累积器：同一信号分块流式送入，随时 query 当前完整前缀
    # ==================================================================
    print("\n" + "=" * 72)
    print("在线累积器 AllanAccumulator：分块送样、随时查询当前前缀")
    print("=" * 72)
    acc = ad.AllanAccumulator(DT, SIZES)
    t0 = time.perf_counter()
    half = len(samples) // 2
    acc.feed(samples[:half])           # 先送一半
    mid = acc.query()
    print(f"已送入 {acc.n_samples} 个样本即查询：m=1 方差={mid.points[0].variance:.6g} "
          f"(count={mid.points[0].count})，拟合段数={mid.fit.segment_count}")
    # 其余按 97 一块送入（块长任意，不必整除、不足窗口的尾部不补零）
    i = half
    while i < len(samples):
        j = min(i + 97, len(samples))
        acc.feed(samples[i:j])
        i = j
    online = acc.query()
    elapsed_online = time.perf_counter() - t0
    worst = max(
        abs(po.variance - pb.variance) / pb.variance
        for po, pb in zip(online.points, result.points)
        if pb.variance
    )
    print(f"全部 {acc.n_samples} 个样本送完（含中途查询，耗时 {elapsed_online:.3f}s）：")
    print(f"  与离线 analyze 同一前缀相比：各方差最大相对差 {worst:.3g}，"
          f"断点一致={online.fit.breakpoints == result.fit.breakpoints}，"
          f"段数一致={online.fit.segment_count == result.fit.segment_count}")

    # ---- 内存不随历史长度增长（只保留 2*max(m) 滑动样本 + 每尺度累积量）----
    long_n = 120_000
    block = 1000
    rng = random.Random(20260922)
    acc_long = ad.AllanAccumulator(DT, SIZES)

    def feed_random(blocks):
        for _ in range(blocks):
            acc_long.feed([rng.gauss(0.0, 1.0) + ACCEL * k
                           for k in range(block)])

    feed_random(long_n // (2 * block))  # 预热一半
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    feed_random(long_n // (2 * block))
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(s.size_diff for s in snap1.compare_to(snap0, "filename"))
    print("-" * 72)
    print(f"长流验证：累计 {acc_long.n_samples} 个样本（远超批量上限 "
          f"{ad.MAX_SAMPLES}），环形缓冲仅 {len(acc_long._ring)} 个滑动样本；")
    print(f"  后一半历史送入前后 tracemalloc 增长 {growth} 字节"
          f"（历史加倍而内存不增）；m=1 方差={acc_long.query().points[0].variance:.6g}")

    # ---- 边界：非法 chunk 整块拒绝且状态不变；dt/尺度创建后不可改 ----
    print("-" * 72)
    print("边界演示（在线入口）：")
    n_before = acc.n_samples
    try:
        acc.feed([1.0, 2.0, float("nan")])
    except ad.AllanValueError as exc:
        print(f"  含 NaN 的 chunk -> {type(exc).__name__}: {exc}")
    print(f"  拒绝后样本数仍为 {acc.n_samples}（要求 {n_before}，块内前两项未被吃掉）："
          f"{acc.n_samples == n_before}")
    try:
        acc.dt = 0.1
    except AttributeError as exc:
        print(f"  创建后改 dt -> AttributeError: {exc}")
    print("=" * 72)


if __name__ == "__main__":
    main()
