#!/usr/bin/env python3
"""
demo.py —— 固定输入的离线演示（标准库，确定性，无网络/无第三方依赖）。

真实计算两部分：
  1) 正常结果：合成“角速率白噪声 + 恒定速率斜坡（漂移）”，计算重叠 Allan
     方差并用动态规划分段，辨识短簇长的白噪声区间与长簇长的漂移区间。
  2) 实际触发的拒绝边界：samples 中放入 NaN，被 InvalidInputError 定位拒绝。
     另用常量信号与 search_budget=1 现场演示“证明无解”与“搜索预算耗尽”
     是两条不同路径。

运行：python demo.py    （约 8 秒预算内，实测远低于此；无 sleep、无预录结果）
"""

import math
import random
import time

from allan_decomp import (
    InvalidInputError,
    SearchBudgetExhausted,
    analyze,
)

DT = 0.01            # 采样周期 10 ms
N = 4096             # 样本数（远低于 8192 接口上限）
SIGMA = 0.05         # 角速率白噪声标准差（rad/s 量级，单位仅作演示）
RAMP = 0.0566        # 恒定角加速斜坡（rad/s^2 量级，单位仅作演示）
SEED = 20260922
M_MAX = 150          # 演示使用的最大簇长


def build_signal():
    """仓库内确定性代码生成：白噪声叠加随时间线性增长的速率斜坡。"""
    rng = random.Random(SEED)
    return [rng.gauss(0.0, SIGMA) + RAMP * t * DT for t in range(N)]


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) 正常结果：白噪声 + 漂移斜坡的区间辨识
    # ------------------------------------------------------------------
    print_header("1) 正常结果：合成白噪声 + 速率斜坡的 Allan 误差分解")
    samples = build_signal()
    print(f"样本数 N={N}, dt={DT} s, 簇长 m=1..{M_MAX}, "
          f"种子={SEED}（确定性）")
    print(f"合成参数（仅作演示的量纲说法，非硬件/临床指标）："
          f"白噪声 sigma={SIGMA}, 斜坡={RAMP}")

    cluster_sizes = list(range(1, M_MAX + 1))
    r = analyze(samples, DT, cluster_sizes)

    print()
    print(f"{'m':>4} {'tau(s)':>9} {'AVAR':>13} {'n_differences':>14}")
    for m in (1, 2, 5, 10, 20, 50, 100, 150):
        p = r.curve[cluster_sizes.index(m)]
        print(f"{p.m:>4} {p.tau:>9.3f} {p.avar:>13.6e} "
              f"{p.n_differences:>14d}")

    print()
    print("log10(AVAR) ~ log10(tau) 连续分段直线（经验斜率，仅统计描述）：")
    for i, seg in enumerate(r.segments):
        print(f"  段{i + 1}: m={seg.m_start:>3}..{seg.m_end:<3}  "
              f"tau={seg.tau_start:6.3f}..{seg.tau_end:6.3f} s  "
              f"斜率={seg.slope:+.3f}")
    print("  断点（段间连续连接点）：")
    for bp in r.breakpoints:
        print(f"      m={bp.m:>3}, tau={bp.tau:7.3f} s, "
              f"log10(tau)={bp.x:+.4f}, log10(AVAR)={bp.y:+.4f}")

    short_slope = r.segments[0].slope
    long_slope = r.segments[-1].slope
    print()
    print("区间辨识（按本契约模型的经验形态，不宣称标准兼容或保证）：")
    print(f"  短簇长区间斜率 {short_slope:+.3f}（接近 -1：角速率白噪声/"
          f"角度随机游走形态，AVAR ∝ tau^-1）")
    print(f"  长簇长区间斜率 {long_slope:+.3f}（接近 +2：速率斜坡/"
          f"漂移形态，AVAR ∝ tau^+2）")
    print(f"  拟合残差平方和 partition_sse="
          f"{r.fit.partition_sse:.6e}, "
          f"连续折线 residual_sse={r.fit.residual_sse:.6e}")
    print(f"  DP 候选段评估次数={r.fit.evaluations} "
          f"(预算={r.fit.budget})，段数={len(r.segments)}，"
          f"罚项={r.fit.segment_penalty:g}")

    # ------------------------------------------------------------------
    # 2a) 实际触发的拒绝边界：NaN 被定位拒绝
    # ------------------------------------------------------------------
    print_header("2a) 拒绝边界（实际触发异常）：samples 含 NaN")
    bad = [0.01 * (i % 5) for i in range(60)]
    bad[7] = float("nan")
    try:
        analyze(bad, DT, [1, 2, 4])
    except InvalidInputError as exc:
        print(f"  捕获 {type(exc).__name__}（ValueError 子类）")
        print(f"  parameter={exc.parameter!r}, index={exc.index}, "
              f"reason={exc.reason!r}")
        print(f"  message: {exc}")
    else:
        raise AssertionError("NaN 必须被拒绝，而不是静默纠正")

    # ------------------------------------------------------------------
    # 2b) 证明无解（不是异常）：常量信号零方差
    # ------------------------------------------------------------------
    print_header("2b) 证明无解（正常返回，不抛异常）：常量信号")
    const_r = analyze([3.25] * 200, DT, [1, 2, 5, 10])
    print(f"  fit.feasible={const_r.fit.feasible}, "
          f"fit.reason={const_r.fit.reason!r}")
    print(f"  全部 m 的 AVAR={[p.avar for p in const_r.curve]}（恒为 0，"
          f"log10 无定义，按契约排除而非报错）")

    # ------------------------------------------------------------------
    # 2c) 搜索预算耗尽（异常，与证明无解严格分离）
    # ------------------------------------------------------------------
    print_header("2c) 搜索预算耗尽（实际触发异常，区别于 2b）")
    try:
        # 可行问题但预算收紧到 1 次评估：必然在完成前耗尽。
        analyze(samples, DT, list(range(1, 21)), search_budget=1)
    except SearchBudgetExhausted as exc:
        print(f"  捕获 {type(exc).__name__}")
        print(f"  evaluations={exc.evaluations}, budget={exc.budget}, "
              f"fit_points={exc.n_points}")
        print(f"  message: {exc}")
    else:
        raise AssertionError("预算 1 必须触发 SearchBudgetExhausted")

    elapsed = time.perf_counter() - t0
    print()
    print("-" * 72)
    print(f"演示完成，实际耗时 {elapsed:.3f} s（无 sleep、全部现场计算）。")
    print("经验斜率是对这批确定性合成数据的统计描述，"
          "不构成硬件指标保证或临床结论。")


if __name__ == "__main__":
    main()
