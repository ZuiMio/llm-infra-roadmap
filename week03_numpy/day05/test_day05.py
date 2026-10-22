"""Unit tests for Week 03 Day 05 matrix multiplication exercises."""

import unittest

import numpy as np

from week03_numpy.day05.day05 import (
    batched_matrix_multiply,
    linear_forward,
    matrix_multiply,
)


class TestMatrixMultiply(unittest.TestCase):
    def test_multiplies_two_matrices(self):
        left = np.array([[1, 2], [3, 4]])
        right = np.array([[10, 20], [30, 40]])

        result = matrix_multiply(left, right)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_array_equal(result, [[70, 100], [150, 220]])

    def test_accepts_array_like_inputs(self):
        result = matrix_multiply([[1, 2], [3, 4]], [[5], [6]])

        np.testing.assert_array_equal(result, [[17], [39]])

    def test_rejects_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            matrix_multiply(np.array([1, 2]), np.ones((2, 1)))

        with self.assertRaises(ValueError):
            matrix_multiply(np.ones((2, 2)), np.array([1, 2]))

    def test_rejects_incompatible_matrix_dimensions(self):
        with self.assertRaises(ValueError):
            matrix_multiply(np.ones((2, 3)), np.ones((2, 4)))

    def test_does_not_mutate_inputs(self):
        left = np.array([[1.0, 2.0], [3.0, 4.0]])
        right = np.array([[5.0, 6.0], [7.0, 8.0]])
        left_before = left.copy()
        right_before = right.copy()

        matrix_multiply(left, right)

        np.testing.assert_array_equal(left, left_before)
        np.testing.assert_array_equal(right, right_before)


class TestBatchedMatrixMultiply(unittest.TestCase):
    def test_multiplies_a_batch_by_one_shared_matrix(self):
        left = np.ones((4, 2, 3))
        right = np.ones((3, 5))

        result = batched_matrix_multiply(left, right)

        self.assertEqual(result.shape, (4, 2, 5))
        np.testing.assert_array_equal(result, np.full((4, 2, 5), 3.0))

    def test_multiplies_paired_batches(self):
        left = np.ones((4, 2, 3))
        right = np.ones((4, 3, 5))

        result = batched_matrix_multiply(left, right)

        self.assertEqual(result.shape, (4, 2, 5))
        np.testing.assert_array_equal(result, np.full((4, 2, 5), 3.0))

    def test_broadcasts_multiple_batch_dimensions(self):
        left = np.ones((2, 1, 3, 4))
        right = np.ones((1, 5, 4, 6))

        result = batched_matrix_multiply(left, right)

        self.assertEqual(result.shape, (2, 5, 3, 6))
        np.testing.assert_array_equal(result, np.full((2, 5, 3, 6), 4.0))

    def test_also_accepts_two_dimensional_matrices(self):
        result = batched_matrix_multiply(np.ones((2, 3)), np.ones((3, 4)))

        self.assertEqual(result.shape, (2, 4))
        np.testing.assert_array_equal(result, np.full((2, 4), 3.0))

    def test_rejects_inputs_with_fewer_than_two_dimensions(self):
        with self.assertRaises(ValueError):
            batched_matrix_multiply(np.array([1, 2, 3]), np.ones((3, 2)))

    def test_rejects_incompatible_matrix_dimensions(self):
        with self.assertRaises(ValueError):
            batched_matrix_multiply(np.ones((2, 3, 4)), np.ones((2, 5, 6)))

    def test_rejects_unbroadcastable_batch_dimensions(self):
        with self.assertRaises(ValueError):
            batched_matrix_multiply(np.ones((2, 2, 3)), np.ones((4, 3, 5)))

    def test_does_not_mutate_inputs(self):
        left = np.ones((2, 2, 3))
        right = np.ones((3, 4))
        left_before = left.copy()
        right_before = right.copy()

        batched_matrix_multiply(left, right)

        np.testing.assert_array_equal(left, left_before)
        np.testing.assert_array_equal(right, right_before)


class TestLinearForward(unittest.TestCase):
    def test_computes_linear_layer_forward_pass(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = np.array([1.0, 1.0, 1.0])

        result = linear_forward(inputs, weights, bias)

        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result, [[10.0, 13.0, 16.0], [20.0, 27.0, 34.0]])

    def test_supports_batch_and_sequence_dimensions(self):
        inputs = np.ones((2, 4, 3))
        weights = np.ones((3, 5))
        bias = np.arange(5, dtype=float)

        result = linear_forward(inputs, weights, bias)

        self.assertEqual(result.shape, (2, 4, 5))
        expected = np.full((2, 4, 5), 3.0) + bias
        np.testing.assert_allclose(result, expected)

    def test_rejects_inputs_with_fewer_than_two_dimensions(self):
        with self.assertRaises(ValueError):
            linear_forward(np.ones(3), np.ones((3, 2)), np.ones(2))

    def test_rejects_invalid_weight_or_bias_dimensions(self):
        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 3)), np.ones((1, 3, 2)), np.ones(2))

        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 3)), np.ones((3, 2)), np.ones((1, 2)))

    def test_rejects_input_feature_mismatch(self):
        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 4)), np.ones((3, 5)), np.ones(5))

    def test_rejects_bias_size_mismatch(self):
        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 3)), np.ones((3, 5)), np.ones(4))

    def test_does_not_mutate_inputs(self):
        inputs = np.ones((2, 3))
        weights = np.ones((3, 4))
        bias = np.ones(4)
        inputs_before = inputs.copy()
        weights_before = weights.copy()
        bias_before = bias.copy()

        linear_forward(inputs, weights, bias)

        np.testing.assert_array_equal(inputs, inputs_before)
        np.testing.assert_array_equal(weights, weights_before)
        np.testing.assert_array_equal(bias, bias_before)


if __name__ == "__main__":
    unittest.main()
