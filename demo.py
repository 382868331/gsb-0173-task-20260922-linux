"""固定输入演示：合成角速率数据的 Allan 误差分解。

运行：python demo.py（约 1 秒内，离线、确定性、无第三方依赖）。

演示两部分：
  A. 定频完整记录（旧能力）：白噪声 + 恒定漂移，经验斜率与理论形态对照。
  B. 不规则采样 + 缺测记录（本次新增）：同一确定性信号，时间戳在名义间隔
     附近抖动并挖掉一段缺测，按实际时间窗口计算；展示有效/缺失窗对计数。

信号模型（确定性代码生成，固定种子）：
    omega[i] = sigma * w[i] + a * i,  w[i] 为高斯白噪声，a*i 为恒定角加速度漂移。
理论形态（以样本数 m 表示的离散结果，tau = m*dt）：
    白噪声: AVAR(m) = sigma^2 / m          -> log-log 斜率 -1/2
    恒斜坡: AVAR(m) = a^2 * m^2 / 2        -> log-log 斜率 +1
交叉点 m* = (2*sigma^2/a^2)^(1/3)。斜率只是本批离线数据的经验形态描述，
不是硬件指标保证。
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allan_deviation as ad  # noqa: E402

N = 2048
DT = 0.01            # 秒（名义采样周期，也是时间窗口网格步长）
SIGMA = 1.0          # 白噪声角速率标准差（任意角速度单位）
ACCEL = 0.01         # 恒定角加速度（同单位/样本斜率）
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def build_samples():
    rng = random.Random(20260922)
    return [SIGMA * rng.gauss(0.0, 1.0) + ACCEL * i for i in range(N)]


def print_points_table(result):
    print(f"{'m':>5} {'tau(s)':>10} {'有效count':>9} {'位置positions':>13} "
          f"{'缺失missing':>12} {'Allan方差':>14} 入拟合")
    for p in result.points:
        if p.variance is None:
            var_txt, mark = "--(有效对<2)", "否"
        else:
            var_txt = f"{p.variance:>14.6g}"
            mark = "是" if p.eligible_for_fit else "否(count<20)"
        print(f"{p.m:>5} {p.tau:>10.4g} {p.count:>9} {p.positions:>13} "
              f"{p.missing:>12} {var_txt} {mark:>10}")


def print_fit(fit):
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


def demo_uniform(samples):
    print("=" * 78)
    print("A. 定频完整记录（旧能力；重叠 Allan 方差 + 分段直线，非硬件指标保证）")
    t0 = time.perf_counter()
    result = ad.analyze(samples, DT, SIZES)
    print(f"样本数 N={N}, dt={DT}s, 时长={N * DT:.1f}s, "
          f"核心计算耗时 {time.perf_counter() - t0:.3f}s")
    print("-" * 78)
    print_points_table(result)
    print("-" * 78)
    print_fit(result.fit)
    small, large = result.fit.segments[0], result.fit.segments[-1]
    print("-" * 78)
    print("经验区间辨识（仅描述本批数据形态）:")
    print(f"  小 tau 段斜率 {small.slope:+.3f}（白噪声区理论约 -0.5，"
          f"实测偏差 {small.slope - (-0.5):+.3f}）")
    print(f"  大 tau 段斜率 {large.slope:+.3f}（漂移区理论约 +1.0，"
          f"实测偏差 {large.slope - 1.0:+.3f}）")
    cross_m = (2.0 * SIGMA * SIGMA / (ACCEL * ACCEL)) ** (1.0 / 3.0)
    if result.fit.breakpoints:
        print(f"  实测断点 m={result.fit.breakpoints[0].m} "
              f"(tau={result.fit.breakpoints[0].tau:.3g}s)，"
              f"理论交叉点 m*≈{cross_m:.1f} (tau≈{cross_m * DT:.3f}s)")


def build_irregular(samples):
    """确定性生成不规则时间戳与缺测标志（不改样本值本身）。"""
    rng = random.Random(20260924)
    ts = [0.0]
    for _ in range(N - 1):
        # 间隔在 DT*(1±35%) 内抖动，严格递增。
        ts.append(ts[-1] + DT * rng.uniform(0.65, 1.35))
    valid = [True] * N
    # 挖掉中间一整段缺测（600..679），再加若干散点缺测。
    for i in range(600, 680):
        valid[i] = False
    for i in (100, 1500, 1900):
        valid[i] = False
    return ts, valid


def demo_windowed(samples):
    print()
    print("=" * 78)
    print("B. 不规则采样 + 缺测记录（本次新增：按实际时间窗口取相邻均值差）")
    ts, valid = build_irregular(samples)
    print(f"样本数 N={N}, 名义 dt={DT}s, 实际跨度 {ts[-1]:.2f}s（间隔 ±35% 抖动）")
    print(f"缺测样本 {sum(1 for f in valid if not f)} 个（含 600..679 一整段）；"
          f"网格锚点 t0={ts[0]:.3f}s，窗为半开区间、边界样本归右窗")
    t0 = time.perf_counter()
    result = ad.analyze(samples, DT, SIZES, timestamps=ts, valid=valid)
    print(f"核心计算耗时 {time.perf_counter() - t0:.3f}s")
    print("-" * 78)
    print_points_table(result)
    print("-" * 78)
    print_fit(result.fit)
    total_missing = sum(p.missing for p in result.points)
    print("-" * 78)
    print(f"覆盖核对：positions=count+missing（全表合计缺失 {total_missing} 对，"
          "缺口附近的窗对被记为缺失而非补零）")
    assert all(p.positions == p.count + p.missing for p in result.points)
    assert total_missing > 0
    small = result.fit.segments[0]
    print(f"不规则采样下小 tau 段斜率 {small.slope:+.3f}（白噪声区理论约 -0.5）")

    # 边界：整段缺测导致有效对不足 -> 证明无拟合点（区别于预算耗尽）。
    all_bad = [False] * N
    try:
        ad.analyze(samples, DT, SIZES, valid=all_bad)
    except ad.FitInfeasibleError as exc:
        pts = exc.points
        print(f"  整段缺测 -> {type(exc).__name__}: {exc.reason}")
        print(f"    附带统计：m=1 时 positions={pts[0].positions}, "
              f"count={pts[0].count}, missing={pts[0].missing}, variance=None")


def demo_errors(samples):
    print()
    print("=" * 78)
    print("边界与实际错误（均为真实计算路径触发，非预录文本；无异常被吞掉）")

    def show(title, fn):
        try:
            fn()
        except ad.AllanError as exc:
            print(f"  {title} -> {type(exc).__name__}: {exc}")

    show("NaN 样本", lambda: ad.analyze(
        [1.0, float("nan"), 3.0], DT, [1, 2]))
    show("bool 类型的 m", lambda: ad.analyze(
        samples[:100], DT, [1, True, 4]))
    show("短数据(N=20, count<20)", lambda: ad.analyze(
        samples[:20], DT, [1, 2, 3]))
    show("时间戳非严格递增", lambda: ad.analyze(
        [1.0, 2.0, 3.0, 4.0], DT, [1], timestamps=[0.0, 1.0, 1.0, 3.0]))
    show("时间戳与样本不等长", lambda: ad.analyze(
        [1.0, 2.0, 3.0], DT, [1], timestamps=[0.0, 1.0]))
    show("valid 传 int 而非 bool", lambda: ad.analyze(
        [1.0, 2.0, 3.0, 4.0], DT, [1], valid=[1, 1, 0, 1]))
    show("min_window_samples<=0", lambda: ad.analyze(
        [1.0] * 100, DT, [1], min_window_samples=0))
    print("（以上错误先于返回抛出；analyze 为纯函数，不改输入、不留半更新状态。）")


def main():
    samples = build_samples()
    demo_uniform(samples)
    demo_windowed(samples)
    demo_errors(samples)
    print("=" * 78)


if __name__ == "__main__":
    main()
