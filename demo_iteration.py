"""温度补偿增量演示：离线拟合二阶温漂并分别检查训练/留出 Allan。"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allan_deviation as ad

N = 1600
TRAIN_END = 800
DT = 0.01
SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
C0, C1, C2 = 0.7, -0.45, 0.18


def build_case():
    timestamps = [i * DT for i in range(N)]
    train_temperature = []
    for i in range(TRAIN_END):
        u = i / (TRAIN_END - 1)
        train_temperature.append(20.0 + 8.0 * (4.0 * u * (1.0 - u) - 1.0))
    holdout_temperature = [
        28.0 + 0.002 * i + 0.7 * math.sin(i / 23.0)
        for i in range(N - TRAIN_END)
    ]
    temperatures = train_temperature + holdout_temperature
    center = (min(train_temperature) + max(train_temperature)) / 2.0
    scale = (max(train_temperature) - min(train_temperature)) / 2.0
    gyro = []
    for i, temperature in enumerate(temperatures):
        x = (temperature - center) / scale
        if i < TRAIN_END:
            residual = 0.012 if i < TRAIN_END // 2 else -0.012
        else:
            residual = 0.008 * math.sin((i - TRAIN_END) * 0.37)
        gyro.append(C0 + C1 * x + C2 * x * x + residual)
    return timestamps, temperatures, gyro, (center, scale)


def variance_sum(result):
    return sum(p.variance for p in result.points if p.variance is not None)


def print_row(label, original, compensated):
    ratio = variance_sum(compensated) / variance_sum(original)
    print(f"  {label:<16} 原数据方差和={variance_sum(original):12.6g}  "
          f"补偿后={variance_sum(compensated):12.6g}  比值={ratio:8.4f}")


def main():
    timestamps, temperatures, gyro, reference_scaling = build_case()
    started = time.perf_counter()
    result = ad.analyze_temperature_compensation(
        timestamps, gyro, temperatures, [0, TRAIN_END], SIZES,
        dt=DT, holdout_range=[TRAIN_END, N]
    )
    elapsed = time.perf_counter() - started

    fit = result.temperature_fit
    print("=" * 78)
    print("二阶温度项离线补偿：训练拟合与未来留出 Allan 分开报告（非硬件性能保证）")
    print(f"N={N}, dt={DT}s, 训练=[0,{TRAIN_END}), 留出=[{TRAIN_END},{N}), "
          f"计算耗时={elapsed:.3f}s")
    print("-" * 78)
    print(f"中心化尺度: center={fit.temperature_center:.12g}, "
          f"scale={fit.temperature_scale:.12g}")
    print("参考独立构造: "
          f"center={reference_scaling[0]:.12g}, scale={reference_scaling[1]:.12g}")
    print("QR 系数 (c0,c1,c2): "
          + ", ".join(f"{value:+.12g}" for value in fit.coefficients))
    print(f"参考系数: ({C0:+.12g}, {C1:+.12g}, {C2:+.12g})")
    print(f"训练残差数={len(fit.residuals)}, 全序列补偿点数={len(result.compensated)}")
    print("-" * 78)
    print_row("训练区间", result.original_training, result.compensated_training)
    print_row("留出区间", result.original_holdout, result.compensated_holdout)
    print("训练改善只说明本批训练拟合；留出结果单独计算，不作为硬件指标保证。")
    print("=" * 78)

    print("\n反例：温度与时间在训练区间内完全线性混淆，必须明确拒绝：")
    confounded_temperature = [25.0 + 0.05 * i for i in range(TRAIN_END)]
    confounded_gyro = [C0 + 0.01 * confounded_temperature[i]
                       for i in range(TRAIN_END)]
    try:
        ad.fit_temperature_polynomial(
            timestamps[:TRAIN_END], confounded_gyro, confounded_temperature, [0, TRAIN_END]
        )
    except ad.TemperatureConfoundingError as exc:
        print(f"  {type(exc).__name__}: {exc}")

    print("反例：常温使一阶/二阶设计列不可用，也必须拒绝：")
    try:
        ad.fit_temperature_polynomial(
            timestamps[:20], gyro[:20], [25.0] * 20, [0, 20]
        )
    except ad.TemperatureDesignError as exc:
        print(f"  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
