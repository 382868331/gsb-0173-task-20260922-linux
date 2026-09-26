"""温度补偿增量测试；多项式期望值由独立参考直接构造/代回。"""

import math
import unittest

import allan_deviation as ad


def independent_poly(temperature, center, scale, c0, c1, c2):
    x = (temperature - center) / scale
    return c0 + c1 * x + c2 * x * x


def make_known_case(n=1600):
    dt = 0.01
    timestamps = [i * dt for i in range(n)]
    train_count = min(n, 800)
    train_temperature = []
    for i in range(train_count):
        u = i / (train_count - 1)
        train_temperature.append(20.0 + 8.0 * (4.0 * u * (1.0 - u) - 1.0))
    future_temperature = [28.0 + 0.002 * i + 0.7 * math.sin(i / 23.0)
                          for i in range(max(0, n - train_count))]
    temperatures = train_temperature + future_temperature
    c0, c1, c2 = 0.7, -0.45, 0.18
    center = (min(train_temperature) + max(train_temperature)) / 2.0
    scale = (max(train_temperature) - min(train_temperature)) / 2.0
    gyro = []
    for i, temperature in enumerate(temperatures):
        if i < train_count:
            deterministic_noise = 0.012 if i < train_count // 2 else -0.012
        else:
            deterministic_noise = 0.008 * math.sin((i - train_count) * 0.37)
        gyro.append(
            independent_poly(temperature, center, scale, c0, c1, c2)
            + deterministic_noise
        )
    return timestamps, temperatures, gyro, dt, (c0, c1, c2), (center, scale)


class TestTemperatureFit(unittest.TestCase):
    def test_qr_fit_recovers_known_polynomial_and_independent_residuals(self):
        timestamps, temperatures, gyro, _, coefficients_true, scaling = make_known_case()
        fit = ad.fit_temperature_polynomial(
            timestamps, gyro, temperatures, [0, 800]
        )
        c0, c1, c2 = fit.coefficients
        self.assertAlmostEqual(fit.temperature_center, scaling[0], places=12)
        self.assertAlmostEqual(fit.temperature_scale, scaling[1], places=12)
        for actual, expected in zip(fit.coefficients, coefficients_true):
            self.assertAlmostEqual(actual, expected, delta=2e-12)

        for index in range(0, 800, 37):
            expected_prediction = independent_poly(
                temperatures[index], fit.temperature_center,
                fit.temperature_scale, c0, c1, c2
            )
            residual_index = index
            self.assertAlmostEqual(
                fit.residuals[residual_index],
                gyro[index] - expected_prediction,
                delta=1e-13,
            )
            self.assertAlmostEqual(
                fit.compensated[index],
                gyro[index] - expected_prediction,
                delta=1e-13,
            )

        for index in range(800, 1600, 51):
            expected_prediction = independent_poly(
                temperatures[index], fit.temperature_center,
                fit.temperature_scale, c0, c1, c2
            )
            self.assertAlmostEqual(
                fit.compensated[index],
                gyro[index] - expected_prediction,
                delta=1e-13,
            )

    def test_training_and_holdout_allan_are_separate(self):
        timestamps, temperatures, gyro, dt, _, _ = make_known_case()
        sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        result = ad.analyze_temperature_compensation(
            timestamps, gyro, temperatures, [0, 800], sizes,
            dt=dt, holdout_range=[800, 1600]
        )
        self.assertEqual(result.original_training.n_samples, 800)
        self.assertEqual(result.compensated_training.n_samples, 800)
        self.assertEqual(result.original_holdout.n_samples, 800)
        self.assertEqual(result.compensated_holdout.n_samples, 800)
        self.assertEqual(len(result.compensated), 1600)

        def total_variance(allan_result):
            return sum(p.variance for p in allan_result.points if p.variance)

        self.assertLess(
            total_variance(result.compensated_holdout),
            total_variance(result.original_holdout),
        )
        self.assertNotAlmostEqual(
            result.original_training.points[-1].variance,
            result.original_holdout.points[-1].variance,
        )

    def test_exact_temperature_time_confounding_is_rejected(self):
        n = 100
        timestamps = [0.01 * i for i in range(n)]
        temperatures = [25.0 + 0.2 * i for i in range(n)]
        gyro = [1.0 + 0.03 * temperatures[i] for i in range(n)]
        with self.assertRaises(ad.TemperatureConfoundingError):
            ad.fit_temperature_polynomial(
                timestamps, gyro, temperatures, [0, 80]
            )

    def test_constant_temperature_is_rank_deficient(self):
        timestamps = [float(i) for i in range(20)]
        temperatures = [30.0] * 20
        gyro = [1.0 + 0.01 * i for i in range(20)]
        with self.assertRaises(ad.TemperatureDesignError):
            ad.fit_temperature_polynomial(
                timestamps, gyro, temperatures, [0, 20]
            )

    def test_insufficient_training_data(self):
        timestamps = [float(i) for i in range(10)]
        temperatures = [20.0 + i for i in range(10)]
        gyro = [float(i) for i in range(10)]
        with self.assertRaises(ad.TemperatureDesignError):
            ad.fit_temperature_polynomial(
                timestamps, gyro, temperatures, [0, 2]
            )

    def test_timestamps_must_align_and_no_interpolation(self):
        timestamps, temperatures, gyro, _, _, _ = make_known_case(20)
        with self.assertRaisesRegex(ad.AllanValueError, "length"):
            ad.fit_temperature_polynomial(
                timestamps[1:], gyro, temperatures, [0, 10]
            )
        bad_timestamps = timestamps[:]
        bad_timestamps[5] = bad_timestamps[4]
        with self.assertRaisesRegex(ad.AllanValueError, "strictly increasing"):
            ad.fit_temperature_polynomial(
                bad_timestamps, gyro, temperatures, [0, 10]
            )
        sizes = [1, 2, 4]
        with self.assertRaisesRegex(ad.AllanValueError, "does not match dt"):
            ad.analyze_temperature_compensation(
                timestamps, gyro, temperatures, [0, 10], sizes,
                dt=0.02, holdout_range=[10, 20]
            )

    def test_chunked_sequence_shapes_do_not_change_coefficients(self):
        timestamps, temperatures, gyro, dt, _, _ = make_known_case(400)
        tuple_timestamps = tuple(timestamps)
        tuple_gyro = tuple(gyro)
        tuple_temperatures = tuple(temperatures)
        first = ad.fit_temperature_polynomial(
            tuple_timestamps, tuple_gyro, tuple_temperatures, (0, 200)
        )
        second = ad.fit_temperature_polynomial(
            list(tuple_timestamps), list(tuple_gyro), list(tuple_temperatures),
            [0, 200]
        )
        self.assertEqual(first.coefficients, second.coefficients)
        self.assertEqual(first.compensated, second.compensated)
        self.assertEqual(first.residuals, second.residuals)

        sliced = ad.fit_temperature_polynomial(
            tuple_timestamps[:200], tuple_gyro[:200], tuple_temperatures[:200],
            (0, 200)
        )
        self.assertEqual(sliced.coefficients, first.coefficients)
        self.assertEqual(sliced.residuals, first.residuals)


if __name__ == "__main__":
    unittest.main()
