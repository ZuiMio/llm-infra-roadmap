"""Unit tests for Week 03 Day 07 batch-feature pipeline exercises."""

import unittest

import numpy as np

from week03_numpy.day07.day07 import (
    fit_standardizer,
    linear_forward,
    predict_in_batches,
    transform_features,
)


class TestFitStandardizer(unittest.TestCase):
    def test_computes_feature_wise_mean_and_population_std(self):
        features = np.array([
            [1.0, 10.0, 5.0],
            [3.0, 14.0, 5.0],
            [5.0, 18.0, 5.0],
        ])

        mean, std = fit_standardizer(features)

        np.testing.assert_allclose(mean, [3.0, 14.0, 5.0])
        np.testing.assert_allclose(std, [np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0), 0.0])
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertEqual(mean.dtype, np.dtype(np.float64))
        self.assertEqual(std.dtype, np.dtype(np.float64))

    def test_accepts_array_like_input(self):
        mean, std = fit_standardizer([[1, 4], [3, 8]])

        np.testing.assert_allclose(mean, [2.0, 6.0])
        np.testing.assert_allclose(std, [1.0, 2.0])

    def test_rejects_invalid_feature_matrices(self):
        invalid_inputs = (
            np.array([1.0, 2.0]),
            np.empty((0, 2)),
            np.empty((2, 0)),
            np.array([[1.0, np.nan]]),
            np.array([[1.0, np.inf]]),
            np.array([["1", "2"]]),
            np.array([[True, False]]),
            np.array([[1.0 + 2.0j]]),
        )
        for features in invalid_inputs:
            with self.subTest(shape=features.shape, dtype=features.dtype):
                with self.assertRaises(ValueError):
                    fit_standardizer(features)

    def test_does_not_mutate_input(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        before = features.copy()

        fit_standardizer(features)

        np.testing.assert_array_equal(features, before)


class TestTransformFeatures(unittest.TestCase):
    def test_normalizes_using_broadcasting(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        mean = np.array([3.0, 14.0])
        std = np.array([2.0, 4.0])

        result = transform_features(features, mean, std)

        np.testing.assert_allclose(result, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(result.shape, features.shape)
        self.assertEqual(result.dtype, np.dtype(np.float64))

    def test_constant_training_feature_becomes_zero(self):
        features = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        mean = np.array([2.0, 5.0])
        std = np.array([np.sqrt(2.0 / 3.0), 0.0])

        result = transform_features(features, mean, std)

        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, 0.0])

    def test_zero_std_preserves_deviation_in_new_data(self):
        features = np.array([[7.0, 8.0]])
        mean = np.array([7.0, 5.0])
        std = np.array([0.0, 0.0])

        result = transform_features(features, mean, std)

        np.testing.assert_allclose(result, [[0.0, 3.0]])

    def test_rejects_invalid_statistics(self):
        features = np.ones((2, 2))
        invalid_pairs = (
            (np.ones(2), np.ones((1, 2))),
            (np.ones(3), np.ones(3)),
            (np.ones(2), np.array([1.0, -1.0])),
            (np.array([1.0, np.nan]), np.ones(2)),
            (np.ones(2), np.array([1.0, np.inf])),
            (np.array(["1", "2"]), np.ones(2)),
            (np.ones(2), np.array([True, False])),
            (np.ones(2, dtype=complex), np.ones(2)),
        )
        for mean, std in invalid_pairs:
            with self.subTest(mean_shape=mean.shape, std_shape=std.shape):
                with self.assertRaises(ValueError):
                    transform_features(features, mean, std)

    def test_rejects_invalid_feature_matrix(self):
        with self.assertRaises(ValueError):
            transform_features(np.array([1.0, 2.0]), np.ones(2), np.ones(2))

    def test_does_not_mutate_inputs(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([2.0, 3.0])
        std = np.array([1.0, 1.0])
        before = (features.copy(), mean.copy(), std.copy())

        transform_features(features, mean, std)

        np.testing.assert_array_equal(features, before[0])
        np.testing.assert_array_equal(mean, before[1])
        np.testing.assert_array_equal(std, before[2])


class TestLinearForward(unittest.TestCase):
    def test_computes_linear_forward_pass(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = np.array([1.0, 1.0, 1.0])

        result = linear_forward(inputs, weights, bias)

        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result, [[10.0, 13.0, 16.0], [20.0, 27.0, 34.0]])
        self.assertEqual(result.dtype, np.dtype(np.float64))

    def test_rejects_invalid_dimensions_and_shapes(self):
        invalid_cases = (
            (np.array([1.0, 2.0]), np.ones((2, 1)), np.ones(1)),
            (np.ones((2, 2)), np.ones((2, 1, 1)), np.ones(1)),
            (np.ones((2, 2)), np.ones((3, 1)), np.ones(1)),
            (np.ones((2, 2)), np.ones((2, 3)), np.ones(2)),
            (np.empty((0, 2)), np.ones((2, 1)), np.ones(1)),
            (np.ones((2, 2)), np.empty((2, 0)), np.ones(0)),
        )
        for inputs, weights, bias in invalid_cases:
            with self.subTest(input_shape=inputs.shape, weight_shape=weights.shape, bias_shape=bias.shape):
                with self.assertRaises(ValueError):
                    linear_forward(inputs, weights, bias)

    def test_rejects_non_finite_and_non_real_values(self):
        with self.assertRaises(ValueError):
            linear_forward(np.array([[np.nan]]), np.ones((1, 1)), np.zeros(1))
        with self.assertRaises(ValueError):
            linear_forward(np.ones((1, 1)), np.array([[1.0 + 1.0j]]), np.zeros(1))
        with self.assertRaises(ValueError):
            linear_forward(np.ones((1, 1)), np.ones((1, 1)), np.array([True]))

    def test_does_not_mutate_inputs(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0], [2.0]])
        bias = np.array([0.5])
        before = (inputs.copy(), weights.copy(), bias.copy())

        linear_forward(inputs, weights, bias)

        np.testing.assert_array_equal(inputs, before[0])
        np.testing.assert_array_equal(weights, before[1])
        np.testing.assert_array_equal(bias, before[2])


class TestPredictInBatches(unittest.TestCase):
    def setUp(self):
        self.features = np.array([
            [1.0, 10.0, 5.0],
            [3.0, 14.0, 5.0],
            [5.0, 18.0, 5.0],
            [7.0, 22.0, 5.0],
            [9.0, 26.0, 5.0],
        ])
        self.mean = np.array([3.0, 14.0, 5.0])
        self.std = np.array([np.sqrt(8.0), np.sqrt(32.0), 0.0])
        self.weights = np.array([[1.0, -1.0], [0.5, 2.0], [0.25, 0.0]])
        self.bias = np.array([0.1, -0.2])

    def test_batched_result_matches_full_matrix_result(self):
        expected = linear_forward(
            transform_features(self.features, self.mean, self.std),
            self.weights,
            self.bias,
        )

        result = predict_in_batches(
            self.features, self.mean, self.std, self.weights, self.bias, batch_size=2
        )

        self.assertEqual(result.shape, (5, 2))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(result, expected)

    def test_batch_size_larger_than_sample_count(self):
        result = predict_in_batches(
            self.features, self.mean, self.std, self.weights, self.bias, batch_size=10
        )
        expected = linear_forward(
            transform_features(self.features, self.mean, self.std),
            self.weights,
            self.bias,
        )

        np.testing.assert_allclose(result, expected)

    def test_batch_size_one_preserves_row_order(self):
        result = predict_in_batches(
            self.features, self.mean, self.std, self.weights, self.bias, batch_size=1
        )
        expected = linear_forward(
            transform_features(self.features, self.mean, self.std),
            self.weights,
            self.bias,
        )

        np.testing.assert_allclose(result, expected)

    def test_rejects_invalid_batch_size(self):
        for batch_size in (0, -1, 1.5, True, np.bool_(False)):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    predict_in_batches(
                        self.features, self.mean, self.std, self.weights, self.bias, batch_size
                    )

    def test_rejects_incompatible_feature_or_model_shapes(self):
        with self.assertRaises(ValueError):
            predict_in_batches(
                self.features, np.ones(2), np.ones(2), self.weights, self.bias, 2
            )
        with self.assertRaises(ValueError):
            predict_in_batches(
                self.features, self.mean, self.std, np.ones((2, 3)), np.ones(3), 2
            )
        with self.assertRaises(ValueError):
            predict_in_batches(
                self.features, self.mean, self.std, self.weights, np.ones(3), 2
            )

    def test_does_not_mutate_inputs(self):
        inputs = self.features.copy()
        mean = self.mean.copy()
        std = self.std.copy()
        weights = self.weights.copy()
        bias = self.bias.copy()
        before = tuple(x.copy() for x in (inputs, mean, std, weights, bias))

        predict_in_batches(inputs, mean, std, weights, bias, batch_size=2)

        for actual, expected in zip((inputs, mean, std, weights, bias), before):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
