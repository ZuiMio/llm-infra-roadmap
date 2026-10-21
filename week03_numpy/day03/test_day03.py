"""Unit tests for Week 03 Day 03 NumPy broadcasting exercises."""

import unittest

import numpy as np

from week03_numpy.day03.day03 import (
    add_feature_bias,
    scale_rows,
    standardize_features,
    vectorized_relu,
)


class TestAddFeatureBias(unittest.TestCase):
    def test_rejects_single_value_bias(self):
        with self.assertRaises(ValueError):
            add_feature_bias(
                np.ones((2, 3)),
                np.array([10]),
            )

    def test_broadcasts_bias_over_batch(self):
        features = np.array([[1, 2, 3], [4, 5, 6]])
        bias = np.array([10, 20, 30])
        features_before = features.copy()
        bias_before = bias.copy()

        result = add_feature_bias(features, bias)

        np.testing.assert_array_equal(
            result,
            [[11, 22, 33], [14, 25, 36]],
        )
        self.assertEqual(result.shape, features.shape)
        self.assertFalse(np.shares_memory(result, features))
        np.testing.assert_array_equal(features, features_before)
        np.testing.assert_array_equal(bias, bias_before)

    def test_rejects_non_2d_features(self):
        with self.assertRaises(ValueError):
            add_feature_bias(np.arange(3), np.arange(3))

    def test_rejects_non_1d_bias(self):
        with self.assertRaises(ValueError):
            add_feature_bias(np.ones((2, 3)), np.ones((1, 3)))

    def test_rejects_mismatched_feature_count(self):
        with self.assertRaises(ValueError):
            add_feature_bias(np.ones((2, 3)), np.ones(4))


class TestScaleRows(unittest.TestCase):
    def test_scales_each_row_independently(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        scales = np.array([2, 3])

        result = scale_rows(matrix, scales)

        np.testing.assert_array_equal(result, [[2, 4, 6], [12, 15, 18]])
        self.assertEqual(result.shape, matrix.shape)
        np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])

    def test_square_matrix_scales_rows_not_columns(self):
        matrix = np.array([[1, 10], [100, 1000]])

        result = scale_rows(matrix, [2, 3])

        np.testing.assert_array_equal(result, [[2, 20], [300, 3000]])

    def test_rejects_non_2d_matrix(self):
        with self.assertRaises(ValueError):
            scale_rows(np.arange(3), [1, 2, 3])

    def test_rejects_wrong_number_of_scales(self):
        with self.assertRaises(ValueError):
            scale_rows(np.ones((3, 2)), [1, 2])


class TestStandardizeFeatures(unittest.TestCase):
    def test_rejects_non_2d_features(self):
        with self.assertRaises(ValueError):
            standardize_features(
                np.array([1, 2, 3]),
                np.array([0, 0, 0]),
                np.array([1, 1, 1]),
            )
            
    def test_standardizes_each_feature_using_broadcasting(self):
        features = np.array([[1.0, 10.0], [3.0, 14.0]])
        original = features.copy()

        result = standardize_features(
            features,
            mean=[2.0, 12.0],
            std=[1.0, 2.0],
        )

        np.testing.assert_allclose(result, [[-1.0, -1.0], [1.0, 1.0]])
        self.assertEqual(result.shape, features.shape)
        np.testing.assert_array_equal(features, original)

    def test_rejects_wrong_statistic_shape(self):
        with self.assertRaises(ValueError):
            standardize_features(
                np.ones((2, 3)),
                mean=[0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            )

    def test_rejects_non_1d_statistics(self):
        with self.assertRaises(ValueError):
            standardize_features(
                np.ones((2, 2)),
                mean=[[0.0, 0.0]],
                std=[1.0, 1.0],
            )

    def test_rejects_zero_or_negative_standard_deviation(self):
        features = np.ones((2, 2))

        with self.assertRaises(ValueError):
            standardize_features(features, [0.0, 0.0], [1.0, 0.0])

        with self.assertRaises(ValueError):
            standardize_features(features, [0.0, 0.0], [1.0, -1.0])


class TestVectorizedRelu(unittest.TestCase):
    def test_applies_relu_to_one_dimensional_input(self):
        array = np.array([-2.0, 0.0, 3.0, -4.0])
        original = array.copy()

        result = vectorized_relu(array)

        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(array, original)
        self.assertFalse(np.shares_memory(result, array))

    def test_preserves_multidimensional_shape(self):
        array = np.array([[-1.0, 2.0], [0.0, -3.0]])

        result = vectorized_relu(array)

        np.testing.assert_array_equal(result, [[0.0, 2.0], [0.0, 0.0]])
        self.assertEqual(result.shape, array.shape)

    def test_accepts_array_like_input(self):
        result = vectorized_relu([[-2, 1], [4, -3]])

        np.testing.assert_array_equal(result, [[0, 1], [4, 0]])


if __name__ == "__main__":
    unittest.main()
