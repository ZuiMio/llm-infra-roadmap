"""Tests for supplemental NumPy neural-network operations."""

import unittest

import numpy as np

from week03_numpy.plus.math_ops import (
    cross_entropy_from_logits,
    layer_norm,
    linear_forward,
    relu,
    softmax,
)


class TestLinearForward(unittest.TestCase):
    def test_shape_values_and_bias_broadcasting(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([0.5, -0.5])

        result = linear_forward(x, w, b)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, [[4.5, 4.5], [10.5, 10.5]])

    def test_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 3)), np.ones((4, 2)), np.zeros(2))
        with self.assertRaises(ValueError):
            linear_forward(np.ones((2, 3)), np.ones((3, 2)), np.zeros(3))


class TestReLU(unittest.TestCase):
    def test_relu_is_elementwise_and_does_not_mutate_input(self):
        x = np.array([[-2.0, 0.0, 3.0], [1.5, -0.5, 4.0]])
        before = x.copy()

        result = relu(x)

        self.assertEqual(result.shape, x.shape)
        np.testing.assert_allclose(result, [[0.0, 0.0, 3.0], [1.5, 0.0, 4.0]])
        np.testing.assert_array_equal(x, before)


class TestSoftmax(unittest.TestCase):
    def test_rows_sum_to_one(self):
        probabilities = softmax([[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]], axis=1)

        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_stable_for_large_positive_and_negative_logits(self):
        logits = np.array([
            [10000.0, 10001.0, 10002.0],
            [-10000.0, -10001.0, -10002.0],
        ])

        probabilities = softmax(logits, axis=1)

        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
        np.testing.assert_allclose(probabilities[0], probabilities[1][::-1], rtol=1e-12)


class TestCrossEntropy(unittest.TestCase):
    def test_matches_known_two_class_value(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1])

        loss = cross_entropy_from_logits(logits, labels)

        self.assertAlmostEqual(loss, float(np.log1p(np.exp(-2.0))), places=12)

    def test_is_stable_for_large_logits(self):
        loss = cross_entropy_from_logits(
            np.array([[10000.0, 9999.0], [9998.0, 10000.0]]),
            np.array([0, 1]),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_rejects_invalid_labels(self):
        with self.assertRaises(ValueError):
            cross_entropy_from_logits(np.ones((2, 3)), np.array([0]))
        with self.assertRaises(ValueError):
            cross_entropy_from_logits(np.ones((2, 3)), np.array([0, 3]))
        with self.assertRaises(ValueError):
            cross_entropy_from_logits(np.ones((2, 3)), np.array([0.0, 1.0]))


class TestLayerNorm(unittest.TestCase):
    def test_normalizes_last_feature_axis(self):
        x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        output = layer_norm(x)

        self.assertEqual(output.shape, x.shape)
        np.testing.assert_allclose(output.mean(axis=-1), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(output.var(axis=-1), [1.0, 1.0], atol=1e-4)

    def test_supports_batch_sequence_feature_shape(self):
        x = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
        output = layer_norm(x)

        self.assertEqual(output.shape, (2, 4, 3))
        np.testing.assert_allclose(output.mean(axis=-1), np.zeros((2, 4)), atol=1e-12)

    def test_applies_gamma_and_beta(self):
        x = np.array([[1.0, 2.0, 3.0]])
        gamma = np.array([2.0, 1.0, 0.5])
        beta = np.array([0.5, -1.0, 2.0])

        output = layer_norm(x, gamma=gamma, beta=beta)
        expected = layer_norm(x) * gamma + beta

        np.testing.assert_allclose(output, expected)

    def test_rejects_wrong_affine_shape(self):
        with self.assertRaises(ValueError):
            layer_norm(np.ones((2, 3)), gamma=np.ones(2))


if __name__ == "__main__":
    unittest.main()
