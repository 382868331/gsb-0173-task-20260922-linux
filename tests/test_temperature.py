"""温度补偿的独立参考与异常路径测试。"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allan_deviation as ad  # noqa: E402


def independent_predict(temperature, center, scale, coefficients):
    u = (temperature - center) / scale
    c0, c1, c2 = coefficients
    return c0 + c1 * u + c2 * u * u


def build_known_drift(n=400, train_end=239):
    temperatures = [20.0 + 3.0 * math.sin(i / 19.0) for i in range(n)]
    c0, c1_raw, c2_raw = 5.0, 0.4, -0.02
    samples = []
    for i, temp in enumerate(temperatures):
        deterministic_noise = 0.01 * math.sin(i / 7.0) + 0.005 * math.cos(i / 13.0)
        samples.append(
            c0
            + c1_raw * temp
            + c2_raw * temp * temp
            + deterministic_noise
        )
    return samples, temperatures, (0, train_end)


class TestTemperatureFit(unittest.TestCase):
    def test_qr_fit_and_independent_polynomial_round_trip(self):
        samples, temperatures, train_range = build_known_drift()
        fit = ad.fit_temperature_compensation(samples, temperatures, train_range)

        self.assertEqual(len(fit.residuals), 240)
        self.assertEqual(len(fit.compensated_samples), len(samples))
        self.assertEqual(fit.term_count, 3)
        self.assertEqual(fit.train_indices, tuple(range(240)))

        for local_index, global_index in enumerate(fit.train_indices):
            prediction = independent_predict(
                temperatures[global_index],
                fit.center,
                fit.scale,
                fit.coefficients,
            )
            residual = samples[global_index] - prediction
            self.assertAlmostEqual(fit.residuals[local_index], residual, delta=1e-12)
            self.assertAlmostEqual(
                fit.compensated_samples[global_index], residual, delta=1e-12
            )

        sse = sum(r * r for r in fit.residuals)
        self.assertAlmostEqual(fit.sse, sse, delta=1e-12)
        self.assertAlmostEqual(fit.rmse, math.sqrt(sse / 240), delta=1e-12)

    def test_reject_exact_temperature_time_confounding(self):
        samples = [0.01 * i + math.sin(i) for i in range(40)]
        temperatures = [20.0 + 0.5 * i for i in range(40)]
        with self.assertRaisesRegex(
            ad.TemperatureFitError, "exactly confounded with time"
        ):
            ad.fit_temperature_compensation(samples, temperatures, (0, 29))

    def test_reject_rank_deficient_nonconstant_temperature(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        temperatures = [20.0, 21.0, 20.0, 21.0]
        with self.assertRaisesRegex(ad.TemperatureFitError, "correlated"):
            ad.fit_temperature_compensation(samples, temperatures, (0, 3))

    def test_reject_insufficient_quadratic_training_rows(self):
        with self.assertRaisesRegex(ad.TemperatureFitError, "at least 3"):
            ad.fit_temperature_compensation(
                [1.0, 2.0, 3.0], [20.0, 21.0, 22.0], (0, 1)
            )

    def test_constant_temperature_degenerates_to_constant_term(self):
        rng = random.Random(910)
        noise = [rng.gauss(0.0, 0.01) for _ in range(40)]
        samples = [7.5 + value for value in noise]
        result = ad.analyze_with_temperature(
            samples,
            [23.0] * 40,
            1.0,
            [1, 2, 4],
            (0, 29),
        )
        fit = result.temperature_fit
        self.assertEqual(fit.term_count, 1)
        self.assertEqual(fit.coefficients[1:], (0.0, 0.0))
        train_mean = sum(noise[:30]) / 30
        self.assertAlmostEqual(fit.coefficients[0], 7.5 + train_mean)
        for i, value in enumerate(fit.compensated_samples):
            self.assertAlmostEqual(value, noise[i] - train_mean, delta=1e-12)

    def test_chunked_equivalent_inputs_do_not_change_fit(self):
        samples, temperatures, train_range = build_known_drift(60, 39)
        chunked_samples = [samples[:10], samples[10:30], samples[30:]]
        chunked_temperatures = [temperatures[:20], temperatures[20:]]
        flat_samples = [value for chunk in chunked_samples for value in chunk]
        flat_temperatures = [value for chunk in chunked_temperatures for value in chunk]

        direct = ad.fit_temperature_compensation(samples, temperatures, train_range)
        flattened = ad.fit_temperature_compensation(
            flat_samples, flat_temperatures, train_range
        )
        self.assertEqual(flattened, direct)


class TestTimestampAlignmentAndReports(unittest.TestCase):
    def test_timestamps_must_match_exactly_without_interpolation(self):
        samples, temperatures, _ = build_known_drift(40, 29)
        timestamps = [0.01 * i for i in range(40)]
        with self.assertRaisesRegex(ad.TemperatureFitError, "refusing to interpolate"):
            ad.fit_temperature_compensation(
                samples, temperatures, (0.0, 0.015), timestamps
            )

    def test_temperature_length_alignment_required(self):
        samples, _, _ = build_known_drift(40, 29)
        with self.assertRaisesRegex(ad.AllanValueError, "does not align"):
            ad.fit_temperature_compensation(
                samples, [20.0] * 39, (0, 29)
            )

    def test_training_and_holdout_reports_are_separate(self):
        samples, temperatures, train_range = build_known_drift(400, 239)
        result = ad.analyze_with_temperature(
            samples,
            temperatures,
            1.0,
            [1, 2, 4, 8, 16, 32, 64],
            train_range,
        )

        self.assertEqual(result.training.original.start_index, 0)
        self.assertEqual(result.training.original.end_index, 239)
        self.assertEqual(result.holdout.original.start_index, 240)
        self.assertEqual(result.holdout.original.end_index, 399)
        self.assertEqual(
            len(result.training.original.points),
            len(result.training.compensated.points),
        )
        self.assertEqual(
            len(result.holdout.original.points),
            len(result.holdout.compensated.points),
        )
        self.assertIsNotNone(result.training.compensated.fit)
        self.assertIsNotNone(result.holdout.compensated.fit)

        original_large_m = result.holdout.original.points[-1].variance
        compensated_large_m = result.holdout.compensated.points[-1].variance
        self.assertLess(compensated_large_m, original_large_m)

    def test_all_data_training_rejected_without_holdout(self):
        samples, temperatures, _ = build_known_drift(40, 29)
        with self.assertRaisesRegex(ad.AllanValueError, "holdout"):
            ad.analyze_with_temperature(
                samples, temperatures, 1.0, [1, 2], (0, 39)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
