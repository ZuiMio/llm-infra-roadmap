"""Unit tests for Week 03 Day 04 NumPy reduction exercises."""

import unittest

import numpy as np

from week03_numpy.day04.day04 import (
    center_features,
    mean_absolute_error,
    mean_features,
    sum_array,
)


class TestSumArray(unittest.TestCase):
    def test_sum_all_values(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])

        result = sum_array(values)

        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(float(result), 21.0)

    def test_sum_by_axis(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_array_equal(sum_array(values, axis=0), [5, 7, 9])
        np.testing.assert_array_equal(sum_array(values, axis=1), [6, 15])

    def test_keepdims_preserves_reduced_axis_as_one(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])

        by_columns = sum_array(values, axis=0, keepdims=True)
        by_rows = sum_array(values, axis=1, keepdims=True)

        self.assertEqual(by_columns.shape, (1, 3))
        self.assertEqual(by_rows.shape, (2, 1))
        np.testing.assert_array_equal(by_columns, [[5, 7, 9]])
        np.testing.assert_array_equal(by_rows, [[6], [15]])

    def test_accepts_negative_axis(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_array_equal(sum_array(values, axis=-1), [6, 15])

    def test_uses_float64_accumulator(self):
        values = np.array([100_000_000, 1, -100_000_000], dtype=np.float32)

        result = sum_array(values)

        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertAlmostEqual(float(result), 1.0)

    def test_rejects_invalid_axis(self):
        values = np.ones((2, 3))

        with self.assertRaises(ValueError):
            sum_array(values, axis=2)

        with self.assertRaises(ValueError):
            sum_array(values, axis=0.5)

        with self.assertRaises(ValueError):
            sum_array(values, axis=True)

    def test_rejects_zero_dimensional_input(self):
        with self.assertRaises(ValueError):
            sum_array(np.array(5))


class TestMeanFeatures(unittest.TestCase):
    def test_computes_mean_over_samples(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        original = features.copy()

        result = mean_features(features)

        np.testing.assert_allclose(result, [3.0, 14.0])
        self.assertEqual(result.shape, (2,))
        np.testing.assert_array_equal(features, original)

    def test_keepdims_keeps_a_single_sample_axis(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0]])

        result = mean_features(features, keepdims=True)

        np.testing.assert_allclose(result, [[2.0, 12.0]])
        self.assertEqual(result.shape, (1, 2))

    def test_rejects_non_2d_or_empty_features(self):
        with self.assertRaises(ValueError):
            mean_features(np.array([1.0, 2.0]))

        with self.assertRaises(ValueError):
            mean_features(np.empty((0, 3)))

        with self.assertRaises(ValueError):
            mean_features(np.empty((2, 0)))


class TestCenterFeatures(unittest.TestCase):
    def test_centers_each_feature_without_mutating_input(self):
        features = np.array([[1, 10], [3, 14]], dtype=np.int32)
        original = features.copy()

        result = center_features(features)

        np.testing.assert_allclose(result, [[-1.0, -2.0], [1.0, 2.0]])
        self.assertEqual(result.shape, features.shape)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(features, original)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_rejects_empty_or_non_2d_input(self):
        with self.assertRaises(ValueError):
            center_features(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            center_features(np.empty((0, 3)))


class TestMeanAbsoluteError(unittest.TestCase):
    def test_computes_mean_absolute_error(self):
        predictions = np.array([1, 2, 4])
        targets = np.array([2, 2, 1])
        predictions_before = predictions.copy()
        targets_before = targets.copy()

        result = mean_absolute_error(predictions, targets)

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 4.0 / 3.0)
        np.testing.assert_array_equal(predictions, predictions_before)
        np.testing.assert_array_equal(targets, targets_before)

    def test_supports_multidimensional_arrays(self):
        predictions = np.array([[1.0, 2.0], [4.0, 8.0]])
        targets = np.array([[2.0, 2.0], [1.0, 10.0]])

        result = mean_absolute_error(predictions, targets)

        self.assertAlmostEqual(result, 1.5)

    def test_uses_float64_arithmetic_for_float32_inputs(self):
        predictions = np.array([100_000_000, 1, -100_000_000], dtype=np.float32)
        targets = np.zeros(3, dtype=np.float32)

        result = mean_absolute_error(predictions, targets)

        self.assertAlmostEqual(result, 200_000_001 / 3, places=4)

    def test_rejects_mismatched_or_empty_inputs(self):
        with self.assertRaises(ValueError):
            mean_absolute_error(np.array([1, 2]), np.array([1]))

        with self.assertRaises(ValueError):
            mean_absolute_error(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()
