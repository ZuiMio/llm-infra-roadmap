"""Unit tests for Week 04 Day 04 hand-written linear forward exercises."""

import unittest

import torch

from week04_pytorch.day04.day04 import linear_forward


class TestLinearForward(unittest.TestCase):
    def test_non_square_matrix_shapes_and_values(self):
        inputs = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        weights = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        bias = torch.tensor([0.5, -0.5])

        output = linear_forward(inputs, weights, bias)

        self.assertEqual(inputs.shape, torch.Size([2, 3]))
        self.assertEqual(weights.shape, torch.Size([3, 2]))
        self.assertEqual(bias.shape, torch.Size([2]))
        self.assertEqual(output.shape, torch.Size([2, 2]))
        torch.testing.assert_close(
            output,
            torch.tensor([
                [4.5, 4.5],
                [10.5, 10.5],
            ]),
        )

    def test_bias_is_broadcast_over_each_batch_row(self):
        inputs = torch.zeros((3, 2))
        weights = torch.ones((2, 4))
        bias = torch.tensor([1.0, 2.0, 3.0, 4.0])

        output = linear_forward(inputs, weights, bias)

        self.assertEqual(output.shape, torch.Size([3, 4]))
        expected = bias.unsqueeze(0).expand(3, 4)
        torch.testing.assert_close(output, expected)

    def test_autograd_graph_is_preserved(self):
        inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
        weights = torch.tensor([[2.0], [3.0]], requires_grad=True)
        bias = torch.tensor([0.5], requires_grad=True)

        output = linear_forward(inputs, weights, bias)

        self.assertTrue(output.requires_grad)
        self.assertIsNotNone(output.grad_fn)

        output.sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(bias.grad)
        torch.testing.assert_close(inputs.grad, torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(weights.grad, torch.tensor([[1.0], [2.0]]))
        torch.testing.assert_close(bias.grad, torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
