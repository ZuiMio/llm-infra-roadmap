"""Tests for the NumPy-only tiny multiclass classifier."""

import unittest

import numpy as np

from week03_numpy.plus.math_ops import cross_entropy_from_logits, linear_forward
from week03_numpy.plus.tiny_nn import (
    linear_classifier_loss_and_gradients,
    make_synthetic_classification_data,
    predict_classes,
    train_linear_classifier,
)


class TestManualClassifierGradients(unittest.TestCase):
    def test_analytic_gradients_match_finite_differences(self):
        x = np.array([[0.2, -0.3], [1.0, 0.5], [-0.7, 1.1]], dtype=np.float64)
        labels = np.array([0, 1, 2], dtype=np.int64)
        weights = np.array([[0.1, -0.2, 0.05], [0.3, 0.15, -0.1]])
        bias = np.array([0.02, -0.04, 0.01])

        _, grad_w, grad_b = linear_classifier_loss_and_gradients(
            x, labels, weights, bias
        )

        eps = 1e-6
        numeric_grad_w = np.zeros_like(weights)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                w_plus = weights.copy()
                w_minus = weights.copy()
                w_plus[i, j] += eps
                w_minus[i, j] -= eps
                loss_plus = cross_entropy_from_logits(
                    linear_forward(x, w_plus, bias), labels
                )
                loss_minus = cross_entropy_from_logits(
                    linear_forward(x, w_minus, bias), labels
                )
                numeric_grad_w[i, j] = (loss_plus - loss_minus) / (2 * eps)

        numeric_grad_b = np.zeros_like(bias)
        for j in range(bias.size):
            b_plus = bias.copy()
            b_minus = bias.copy()
            b_plus[j] += eps
            b_minus[j] -= eps
            loss_plus = cross_entropy_from_logits(
                linear_forward(x, weights, b_plus), labels
            )
            loss_minus = cross_entropy_from_logits(
                linear_forward(x, weights, b_minus), labels
            )
            numeric_grad_b[j] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad_w, numeric_grad_w, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(grad_b, numeric_grad_b, rtol=1e-5, atol=1e-7)

    def test_gradient_function_does_not_mutate_parameters(self):
        x, labels = make_synthetic_classification_data()
        weights = np.zeros((2, 3))
        bias = np.zeros(3)
        weights_before = weights.copy()
        bias_before = bias.copy()

        linear_classifier_loss_and_gradients(x, labels, weights, bias)

        np.testing.assert_array_equal(weights, weights_before)
        np.testing.assert_array_equal(bias, bias_before)


class TestTinyClassifierTraining(unittest.TestCase):
    def test_loss_decreases_and_classification_is_correct(self):
        x, labels = make_synthetic_classification_data()
        weights, bias, history = train_linear_classifier(steps=300, learning_rate=0.2)
        predictions = predict_classes(x, weights, bias)
        accuracy = np.mean(predictions == labels)

        self.assertEqual(x.shape, (15, 2))
        self.assertEqual(labels.shape, (15,))
        self.assertEqual(weights.shape, (2, 3))
        self.assertEqual(bias.shape, (3,))
        self.assertEqual(len(history), 301)
        self.assertTrue(np.isfinite(history).all())
        self.assertLess(history[-1], history[0])
        self.assertGreaterEqual(accuracy, 0.95)

    def test_rejects_invalid_training_parameters(self):
        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    train_linear_classifier(steps=steps)
        for learning_rate in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    train_linear_classifier(learning_rate=learning_rate)


if __name__ == "__main__":
    unittest.main()
