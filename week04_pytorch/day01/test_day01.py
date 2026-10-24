"""Unit tests for Week 04 Day 01 PyTorch Tensor and Autograd exercises."""

import unittest

import torch

from week04_pytorch.day01.day01 import (
    linear_forward,
    make_scalar_square,
    mean_squared_error,
)


class TestScalarAutograd(unittest.TestCase):
    def test_creates_scalar_and_square_without_running_backward(self):
        x, y = make_scalar_square(3.0)

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, torch.Size([]))
        self.assertEqual(y.shape, torch.Size([]))
        self.assertTrue(x.requires_grad)
        self.assertTrue(x.is_leaf)
        self.assertIsNotNone(y.grad_fn)
        self.assertIsNone(x.grad)
        self.assertAlmostEqual(x.item(), 3.0)
        self.assertAlmostEqual(y.item(), 9.0)

    def test_backward_computes_expected_gradient(self):
        x, y = make_scalar_square(3.0)

        y.backward()

        self.assertIsNotNone(x.grad)
        self.assertAlmostEqual(x.grad.item(), 6.0)

    def test_gradient_matches_two_times_input_for_negative_value(self):
        x, y = make_scalar_square(-4.0)

        y.backward()

        self.assertAlmostEqual(x.grad.item(), -8.0)


class TestLinearForwardAndLoss(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
        self.bias = torch.tensor([0.25], requires_grad=True)

    def test_linear_forward_shape_and_values(self):
        predictions = linear_forward(self.inputs, self.weights, self.bias)

        self.assertEqual(predictions.shape, torch.Size([2, 1]))
        torch.testing.assert_close(
            predictions,
            torch.tensor([[-1.25], [-2.25]]),
        )

    def test_mse_is_a_scalar_with_expected_value(self):
        predictions = linear_forward(self.inputs, self.weights, self.bias)
        targets = torch.zeros_like(predictions)

        loss = mean_squared_error(predictions, targets)

        self.assertEqual(loss.shape, torch.Size([]))
        torch.testing.assert_close(loss, torch.tensor(3.3125))

    def test_backward_reaches_weights_and_bias(self):
        predictions = linear_forward(self.inputs, self.weights, self.bias)
        loss = mean_squared_error(predictions, torch.zeros_like(predictions))

        loss.backward()

        self.assertIsNotNone(self.weights.grad)
        self.assertIsNotNone(self.bias.grad)
        torch.testing.assert_close(
            self.weights.grad,
            torch.tensor([[-8.0], [-11.0]]),
        )
        torch.testing.assert_close(
            self.bias.grad,
            torch.tensor([-3.5]),
        )


if __name__ == "__main__":
    unittest.main()
