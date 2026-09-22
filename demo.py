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


if __name__ == "__main__":
    main()
