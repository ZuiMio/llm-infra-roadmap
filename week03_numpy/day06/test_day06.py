"""Unit tests for Week 03 Day 06 NumPy preprocessing exercises."""

import unittest

import numpy as np

from week03_numpy.day06.day06 import (
    compute_feature_stats,
    normalize_features,
    preprocess_features,
    validate_feature_matrix,
)


class TestValidateFeatureMatrix(unittest.TestCase):
    def test_accepts_array_like_and_returns_float64(self):
        result = validate_feature_matrix([[1, 2], [3, 4]])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_checks_expected_feature_count(self):
        result = validate_feature_matrix([[1, 2], [3, 4]], expected_n_features=2)

        self.assertEqual(result.shape[1], 2)
        with self.assertRaises(ValueError):
            validate_feature_matrix([[1, 2], [3, 4]], expected_n_features=3)

    def test_rejects_invalid_expected_feature_count(self):
        for invalid in (0, -1, 1.5, True, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_feature_matrix([[1, 2]], expected_n_features=invalid)

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            validate_feature_matrix([1, 2, 3])

    def test_rejects_empty_samples_or_features(self):
        with self.assertRaises(ValueError):
            validate_feature_matrix(np.empty((0, 3)))

        with self.assertRaises(ValueError):
            validate_feature_matrix(np.empty((2, 0)))

    def test_rejects_non_numeric_boolean_or_complex_values(self):
        invalid_inputs = (
            np.array([["a", "b"], ["c", "d"]]),
            np.array([[True, False]]),
            np.array([[1 + 2j, 3 + 4j]]),
        )
        for invalid in invalid_inputs:
            with self.subTest(dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    validate_feature_matrix(invalid)

    def test_rejects_nan_and_infinity(self):
        for invalid in (
            np.array([[1.0, np.nan]]),
            np.array([[1.0, np.inf]]),
            np.array([[-np.inf, 2.0]]),
        ):
            with self.subTest(values=invalid):
                with self.assertRaises(ValueError):
                    validate_feature_matrix(invalid)

    def test_does_not_mutate_input(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = features.copy()

        validate_feature_matrix(features)

        np.testing.assert_array_equal(features, original)


class TestComputeFeatureStats(unittest.TestCase):
    def test_computes_feature_wise_mean_and_population_std(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])

        mean, std = compute_feature_stats(features)

        np.testing.assert_allclose(mean, [3.0, 14.0])
        np.testing.assert_allclose(std, [np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)])
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertEqual(mean.dtype, np.dtype(np.float64))
        self.assertEqual(std.dtype, np.dtype(np.float64))

    def test_constant_feature_has_zero_standard_deviation(self):
        mean, std = compute_feature_stats([[1, 5], [2, 5], [3, 5]])

        np.testing.assert_allclose(mean, [2.0, 5.0])
        np.testing.assert_allclose(std, [np.sqrt(2.0 / 3.0), 0.0])

    def test_rejects_invalid_feature_matrix(self):
        with self.assertRaises(ValueError):
            compute_feature_stats(np.array([1.0, 2.0]))


class TestNormalizeFeatures(unittest.TestCase):
    def test_normalizes_features_using_broadcasting(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        mean = np.array([3.0, 14.0])
        std = np.array([2.0, 4.0])

        result = normalize_features(features, mean, std)

        np.testing.assert_allclose(result, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(result.shape, features.shape)
        self.assertEqual(result.dtype, np.dtype(np.float64))

    def test_handles_zero_std_without_division_by_zero(self):
        features = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        mean = np.array([2.0, 5.0])
        std = np.array([np.sqrt(2.0 / 3.0), 0.0])

        result = normalize_features(features, mean, std)

        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[:, 0], [-1.0, 0.0, 1.0])

    def test_zero_std_preserves_deviation_in_new_data(self):
        new_features = np.array([[7.0, 8.0]])
        mean = np.array([7.0, 5.0])
        std = np.array([0.0, 0.0])

        result = normalize_features(new_features, mean, std)

        np.testing.assert_allclose(result, [[0.0, 3.0]])

    def test_rejects_invalid_statistic_dimensions_or_lengths(self):
        features = np.ones((2, 3))

        invalid_statistics = (
            (np.ones((1, 3)), np.ones(3)),
            (np.ones(2), np.ones(3)),
            (np.ones(3), np.ones(2)),
        )
        for mean, std in invalid_statistics:
            with self.subTest(mean_shape=mean.shape, std_shape=std.shape):
                with self.assertRaises(ValueError):
                    normalize_features(features, mean, std)

    def test_rejects_non_finite_or_negative_statistics(self):
        features = np.ones((2, 2))
        invalid_statistics = (
            (np.array([0.0, np.nan]), np.array([1.0, 1.0])),
            (np.array([0.0, 0.0]), np.array([1.0, np.inf])),
            (np.array([0.0, 0.0]), np.array([1.0, -1.0])),
        )
        for mean, std in invalid_statistics:
            with self.subTest(mean=mean, std=std):
                with self.assertRaises(ValueError):
                    normalize_features(features, mean, std)

    def test_rejects_non_numeric_statistics(self):
        with self.assertRaises(ValueError):
            normalize_features(np.ones((2, 2)), ["0", "1"], [1.0, 1.0])

    def test_does_not_mutate_inputs(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0]])
        mean = np.array([2.0, 12.0])
        std = np.array([1.0, 2.0])
        before = (features.copy(), mean.copy(), std.copy())

        normalize_features(features, mean, std)

        np.testing.assert_array_equal(features, before[0])
        np.testing.assert_array_equal(mean, before[1])
        np.testing.assert_array_equal(std, before[2])


class TestPreprocessFeatures(unittest.TestCase):
    def test_runs_the_full_pipeline_and_returns_statistics(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        original = features.copy()

        normalized, mean, std = preprocess_features(features)

        np.testing.assert_allclose(mean, [3.0, 14.0])
        np.testing.assert_allclose(std, [np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)])
        np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=1e-12)
        np.testing.assert_array_equal(features, original)

    def test_pipeline_handles_constant_features(self):
        normalized, mean, std = preprocess_features([[1, 5], [2, 5], [3, 5]])

        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(normalized[:, 1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(std, [np.sqrt(2.0 / 3.0), 0.0])
        np.testing.assert_allclose(mean, [2.0, 5.0])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            preprocess_features([[1.0, np.nan], [2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
