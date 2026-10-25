"""Unit tests for Week 04 Day 02 PyTorch autograd graph exercises."""

import unittest

import torch

from week04_pytorch.day02.day02 import (
    backward_with_retained_intermediate,
    build_computation_graph,
)


class TestComputationGraph(unittest.TestCase):
    def test_forward_values_and_tensor_metadata(self):
        x, middle, output = build_computation_graph(2.0)

        self.assertEqual(x.shape, torch.Size([]))
        self.assertEqual(middle.shape, torch.Size([]))
        self.assertEqual(output.shape, torch.Size([]))
        self.assertAlmostEqual(x.item(), 2.0)
        self.assertAlmostEqual(middle.item(), 6.0)
        self.assertAlmostEqual(output.item(), 36.0)

        self.assertTrue(x.requires_grad)
        self.assertTrue(middle.requires_grad)
        self.assertTrue(output.requires_grad)

        self.assertTrue(x.is_leaf)
        self.assertFalse(middle.is_leaf)
        self.assertFalse(output.is_leaf)

        self.assertIsNone(x.grad_fn)
        self.assertIsNotNone(middle.grad_fn)
        self.assertIsNotNone(output.grad_fn)

        self.assertIsNone(x.grad)

    def test_backward_populates_leaf_gradient(self):
        x, middle, output = build_computation_graph(2.0)

        output.backward()

        # output = (3*x)^2 = 9*x^2, so d(output)/dx = 18*x = 36 at x=2.
        self.assertIsNotNone(x.grad)
        torch.testing.assert_close(x.grad, torch.tensor(36.0))

    def test_retain_grad_keeps_intermediate_gradient(self):
        x, middle, output = backward_with_retained_intermediate(2.0)

        # d(output)/d(middle) = 2*middle = 12 at middle=6.
        self.assertIsNotNone(middle.grad)
        torch.testing.assert_close(middle.grad, torch.tensor(12.0))

        # Chain rule: d(output)/dx = d(output)/d(middle) * d(middle)/dx
        # = 12 * 3 = 36.
        self.assertIsNotNone(x.grad)
        torch.testing.assert_close(x.grad, torch.tensor(36.0))
        self.assertAlmostEqual(output.item(), 36.0)


if __name__ == "__main__":
    unittest.main()
