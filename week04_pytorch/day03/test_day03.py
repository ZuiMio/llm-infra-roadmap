"""Unit tests for Week 04 Day 03 gradient accumulation exercises."""

import unittest

import torch

from week04_pytorch.day03.day03 import (
    demonstrate_gradient_accumulation,
    demonstrate_gradient_reset,
)


class TestGradientAccumulation(unittest.TestCase):
    def test_second_backward_adds_to_existing_gradient(self):
        x, grad_after_first, grad_after_second = demonstrate_gradient_accumulation(2.0)

        self.assertEqual(x.shape, torch.Size([]))
        self.assertTrue(x.is_leaf)
        self.assertTrue(x.requires_grad)

        torch.testing.assert_close(grad_after_first, torch.tensor(4.0))
        torch.testing.assert_close(grad_after_second, torch.tensor(7.0))
        torch.testing.assert_close(x.grad, torch.tensor(7.0))

    def test_accumulation_uses_derivatives_of_both_losses(self):
        # At x=-1: d(x^2)/dx = -2, while d(3*x)/dx = 3.
        # The accumulated gradient is -2 + 3 = 1.
        x, grad_after_first, grad_after_second = demonstrate_gradient_accumulation(-1.0)

        torch.testing.assert_close(grad_after_first, torch.tensor(-2.0))
        torch.testing.assert_close(grad_after_second, torch.tensor(1.0))
        torch.testing.assert_close(x.grad, torch.tensor(1.0))


class TestGradientReset(unittest.TestCase):
    def test_clearing_gradient_prevents_old_gradient_from_carrying_over(self):
        x, grad_before_reset, grad_after_reset = demonstrate_gradient_reset(2.0)

        torch.testing.assert_close(grad_before_reset, torch.tensor(4.0))
        torch.testing.assert_close(grad_after_reset, torch.tensor(3.0))
        torch.testing.assert_close(x.grad, torch.tensor(3.0))

    def test_reset_behavior_with_negative_input(self):
        # At x=-2, the first derivative is -4; after reset, the second
        # loss 3*x contributes only its derivative, which is 3.
        x, grad_before_reset, grad_after_reset = demonstrate_gradient_reset(-2.0)

        torch.testing.assert_close(grad_before_reset, torch.tensor(-4.0))
        torch.testing.assert_close(grad_after_reset, torch.tensor(3.0))
        torch.testing.assert_close(x.grad, torch.tensor(3.0))


if __name__ == "__main__":
    unittest.main()
