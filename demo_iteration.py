"""温度补偿迭代演示：训练区拟合，留出区单独报告 Allan。

运行：python demo_iteration.py
全程离线、确定性生成数据、无第三方依赖；常规规模下仅需数秒。
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allan_deviation as ad  # noqa: E402


N = 1024
DT = 0.01
TRAIN_END = 699
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def build_data():
    rng = random.Random(20260926)
    temperatures = []
    samples = []
    for i in range(N):
        temp = 25.0 + 4.0 * math.sin(i / 31.0) + 1.5 * math.sin(i / 97.0)
        centered = temp - 25.0
        drift = 0.35 * centered - 0.018 * centered * centered
        sample = 0.8 + drift + 0.08 * rng.gauss(0.0, 1.0)
        temperatures.append(temp)
        samples.append(sample)
    return samples, temperatures


def variance_at(report, m):
    point = next(point for point in report.points if point.m == m)
    if point.variance is None:
        return float("nan")
    return point.variance


def print_split(title, split):
    original = split.original
    compensated = split.compensated
    print(f"\n{title}：索引 {original.start_index}..{original.end_index}")
    print(f"{'m':>5} {'原AVAR':>14} {'补偿AVAR':>14} {'补偿/原':>12} 拟合")
    for original_point, compensated_point in zip(original.points, compensated.points):
        if original_point.variance is None or compensated_point.variance is None:
            continue
        ratio = compensated_point.variance / original_point.variance
        fit_mark = "是" if compensated.fit is not None else "否"
        print(
            f"{original_point.m:>5} {original_point.variance:>14.6g} "
            f"{compensated_point.variance:>14.6g} {ratio:>12.3f} "
            f"{fit_mark}"
        )
    if original.fit and compensated.fit:
        print(
            f"  原数据分段={original.fit.segment_count}，"
            f"补偿后分段={compensated.fit.segment_count}"
        )
    if compensated.infeasible_reason:
        print(f"  补偿后分段拟合无解：{compensated.infeasible_reason}")


def main():
    samples, temperatures = build_data()
    start = time.perf_counter()
    result = ad.analyze_with_temperature(
        samples,
        temperatures,
        DT,
        SIZES,
        (0, TRAIN_END),
    )
    elapsed = time.perf_counter() - start

    fit = result.temperature_fit
    print("=" * 76)
    print("离线温度补偿 Allan 演示（训练/留出分离；非真实传感器性能保证）")
    print(
        f"N={N}, dt={DT}s, 训练=0..{TRAIN_END}, 留出={TRAIN_END + 1}..{N - 1}, "
        f"现场计算耗时={elapsed:.3f}s"
    )
    print(
        f"温度中心化 center={fit.center:.6f}, scale={fit.scale:.6f}; "
        f"系数(c0,c1,c2)=" + "({:.6g}, {:.6g}, {:.6g})".format(*fit.coefficients)
    )
    print(f"训练残差 SSE={fit.sse:.6g}, RMSE={fit.rmse:.6g}")
    print_split("训练区间", result.training)
    print_split("留出区间", result.holdout)
    train_ratio = variance_at(result.training.compensated, 64) / variance_at(
        result.training.original, 64
    )
    holdout_ratio = variance_at(result.holdout.compensated, 64) / variance_at(
        result.holdout.original, 64
    )
    print("-" * 76)
    print(
        f"m=64：训练区补偿/原={train_ratio:.3f}，留出区补偿/原={holdout_ratio:.3f}。"
    )
    print("训练改善只说明训练区间拟合；留出结果才用于独立观察本次离线补偿稳定性。")

    print("\n异常路径：")
    confounded_temperature = [20.0 + 0.1 * i for i in range(N)]
    try:
        ad.fit_temperature_compensation(
            samples, confounded_temperature, (0, TRAIN_END)
        )
    except ad.TemperatureFitError as exc:
        print(f"  温度与时间完全混淆 -> {type(exc).__name__}: {exc}")

    timestamps = [DT * i for i in range(N)]
    try:
        ad.fit_temperature_compensation(
            samples, temperatures, (0.0, DT / 2.0), timestamps
        )
    except ad.TemperatureFitError as exc:
        print(f"  时间戳无法精确对齐 -> {type(exc).__name__}: {exc}")
    print("=" * 76)


if __name__ == "__main__":
    main()
