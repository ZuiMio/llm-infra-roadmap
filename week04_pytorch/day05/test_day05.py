"""Unit tests for Week 04 Day 05 loss and parameter gradients."""

import unittest

import torch

from week04_pytorch.day05.day05 import compute_loss_and_gradients


class TestLossAndParameterGradients(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
        self.bias = torch.tensor([0.25], requires_grad=True)
        self.targets = torch.zeros((2, 1))

    def test_loss_and_gradients_match_hand_calculation(self):
        predictions, loss, weight_grad, bias_grad = compute_loss_and_gradients(
            self.inputs, self.weights, self.bias, self.targets
        )

        torch.testing.assert_close(
            predictions, torch.tensor([[-1.25], [-2.25]])
        )
        self.assertEqual(loss.shape, torch.Size([]))
        torch.testing.assert_close(loss, torch.tensor(3.3125))
        torch.testing.assert_close(
            weight_grad, torch.tensor([[-8.0], [-11.5]])
        )
        torch.testing.assert_close(bias_grad, torch.tensor([-3.5]))

    def test_parameters_are_not_updated_by_backward(self):
        weights_before = self.weights.detach().clone()
        bias_before = self.bias.detach().clone()

        compute_loss_and_gradients(
            self.inputs, self.weights, self.bias, self.targets
        )

        torch.testing.assert_close(self.weights, weights_before)
        torch.testing.assert_close(self.bias, bias_before)

    def test_returned_gradients_are_copies_of_parameter_gradients(self):
        _, _, weight_grad, bias_grad = compute_loss_and_gradients(
            self.inputs, self.weights, self.bias, self.targets
        )

        self.assertIsNot(weight_grad, self.weights.grad)
        self.assertIsNot(bias_grad, self.bias.grad)
        torch.testing.assert_close(weight_grad, self.weights.grad)
        torch.testing.assert_close(bias_grad, self.bias.grad)


if __name__ == "__main__":
    unittest.main()
