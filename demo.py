"""固定输入演示：合成角速率数据的 Allan 误差分解。

运行：python demo.py（约几秒，离线、确定性、无第三方依赖）。

两部分：

1. 定频模式（向后兼容）：omega[i] = sigma*w[i] + a*i，w 为高斯白噪声，
   a*i 为恒定角加速度漂移。
   理论形态（以样本数 m 表示的离散结果，tau = m*dt）：
       白噪声: AVAR(m) = sigma^2 / m          -> log-log 斜率 -1/2
       恒斜坡: AVAR(m) = a^2 * m^2 / 2        -> log-log 斜率 +1
   交叉点 m* = (2*sigma^2/a^2)^(1/3)。

2. 时间戳 + 有效标志模式（本次新增）：接受每样本实际时间戳与 valid 标志，
   按固定端点时窗与“至少半窗有效样本”规则统计；演示不规则采样与中段整段
   缺测下的缺失计数，并真实触发时间戳非递增、标志长度不符等拒绝。

斜率只是本批离线数据的经验形态描述，不是硬件指标保证。
"""

import os
import random
import sys
import time

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


# ---------------------------------------------------------------------------
# 时间戳模式：不规则采样 + 中段整段缺测（确定性生成）
# ---------------------------------------------------------------------------

TS_N = 1200          # 名义样本数
TS_DT = 0.02         # 标称周期 20ms
TS_GAP = range(500, 620)   # 中段 120 个样本整段缺测（2.4s）


def build_timestamped():
    """返回 (values, timestamps, valid)：间隔在 1/2/3 个周期间抖动，中段挖洞。"""
    rng = random.Random(20260924)
    values = []
    timestamps = []
    valid = []
    t = 0.0
    for i in range(TS_N):
        # 确定性的不规则间隔：约 80% 为 1 个周期，其余 2 或 3 个周期。
        r = rng.random()
        step_n = 1 if r < 0.8 else (2 if r < 0.95 else 3)
        values.append(rng.gauss(0.0, 1.0) + 0.004 * i)
        timestamps.append(t)
        valid.append(i not in TS_GAP)
        t += step_n * TS_DT
    return values, timestamps, valid


def demo_timestamp_mode():
    values, timestamps, valid = build_timestamped()
    durations = [TS_DT * m for m in (1, 2, 4, 8, 16, 32, 64, 128)]

    print("=" * 72)
    print("时间戳 + 有效标志模式：不规则采样、中段整段缺测")
    print(f"名义样本 {TS_N}，有效 {sum(valid)}，"
          f"缺测 {TS_N - sum(valid)}（单元 {TS_GAP.start}..{TS_GAP.stop - 1}），"
          f"标称周期 {TS_DT:g}s")
    t0 = time.perf_counter()
    try:
        result = ad.analyze(
            values, timestamps=timestamps, valid=valid,
            cluster_durations=durations,
        )
    except ad.FitInfeasibleError as exc:
        result = None
        points = exc.points
        infeasible = exc
    else:
        points = result.points
        infeasible = None
    elapsed = time.perf_counter() - t0
    dt_nom = _dt_echo(timestamps, valid)
    print(f"核心计算耗时 {elapsed:.3f}s，回显标称 dt≈{dt_nom:.4g}s "
          f"（有效样本间隔中位数）")
    print("-" * 72)
    print(f"{'窗长(s)':>8} {'tau(s)':>8} {'考察窗对':>8} {'有效差分':>8} "
          f"{'缺失':>6} {'Allan方差':>14} 入拟合")
    for p in points:
        var = "--(缺失)" if p.variance is None else f"{p.variance:>14.6g}"
        mark = "是" if p.eligible_for_fit else "否"
        print(f"{p.m:>8.3g} {p.tau:>8.3g} {p.expected_count:>8} {p.count:>8} "
              f"{p.missing_count:>6} {var:>14} {mark:>4}")
    if infeasible is not None:
        print("-" * 72)
        print(f"分段拟合：{type(infeasible).__name__}（{infeasible.reason}）")
        print("跨缺测段的大窗长有效差分不足 20，合格点不足以拟合——")
        print("逐窗统计仍在上方完整给出（缺失被计数而非静默补零）。")
    else:
        fit = result.fit
        print("-" * 72)
        print(f"分段直线：{fit.segment_count} 段，目标值={fit.objective:.6g}")
        for i, seg in enumerate(fit.segments):
            print(f"  段{i + 1}: tau∈[{seg.tau_start:.3g},{seg.tau_end:.3g}]s "
                  f"({seg.n_points}点) 斜率={seg.slope:+.3f}")
    print("=" * 72)

    # ---- 实际错误：时间戳非递增、valid 长度不符、两模式混用 ----
    print("\n边界演示（时间戳模式，均为真实校验路径触发）：")
    bad_ts = timestamps[:]
    bad_ts[10] = bad_ts[9] - 1.0
    try:
        ad.analyze(values, timestamps=bad_ts, valid=valid,
                   cluster_durations=durations)
    except ad.AllanValueError as exc:
        print(f"  时间戳非递增 -> {type(exc).__name__}: {exc}")
    try:
        ad.analyze(values, timestamps=timestamps, valid=valid[:-1],
                   cluster_durations=durations)
    except ad.AllanValueError as exc:
        print(f"  valid 长度不符 -> {type(exc).__name__}: {exc}")
    try:
        ad.analyze(values, TS_DT, [1, 2, 4], timestamps=timestamps)
    except ad.AllanValueError as exc:
        print(f"  混用两种模式 -> {type(exc).__name__}: {exc}")
    try:
        ad.analyze(values, timestamps=timestamps,
                   cluster_durations=[0.02, -0.04])
    except ad.AllanValueError as exc:
        print(f"  非法窗长 -> {type(exc).__name__}: {exc}")
    print("（以上拒绝均为实际计算路径触发，非预录文本；无异常被吞掉。）")


def _dt_echo(timestamps, valid):
    vidx = [i for i, f in enumerate(valid) if f]
    gaps = sorted(timestamps[vidx[i]] - timestamps[vidx[i - 1]]
                  for i in range(1, len(vidx)))
    lg = len(gaps)
    return gaps[lg // 2] if lg % 2 else 0.5 * (gaps[lg // 2 - 1] + gaps[lg // 2])


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

    demo_timestamp_mode()


if __name__ == "__main__":
    main()
