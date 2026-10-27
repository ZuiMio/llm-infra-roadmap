"""Unit tests for Week 04 Day 06 manual parameter updates."""

import unittest

import torch

from week04_pytorch.day06.day06 import (
    manual_sgd_update,
    train_simple_relationship,
)


class TestManualSGDUpdate(unittest.TestCase):
    def test_updates_parameters_without_optimizer_and_clears_gradients(self):
        weights = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([2.0], requires_grad=True)
        weights.grad = torch.tensor([[2.0]])
        bias.grad = torch.tensor([4.0])

        manual_sgd_update(weights, bias, learning_rate=0.1)

        torch.testing.assert_close(weights, torch.tensor([[0.8]]))
        torch.testing.assert_close(bias, torch.tensor([1.6]))
        self.assertIsNone(weights.grad)
        self.assertIsNone(bias.grad)

    def test_requires_gradients_before_update(self):
        weights = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([0.0], requires_grad=True)
        weights.grad = torch.tensor([[1.0]])

        with self.assertRaises(RuntimeError):
            manual_sgd_update(weights, bias, learning_rate=0.1)

    def test_rejects_invalid_learning_rate(self):
        for learning_rate in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(learning_rate=learning_rate):
                weights = torch.tensor([[1.0]], requires_grad=True)
                bias = torch.tensor([0.0], requires_grad=True)
                weights.grad = torch.ones_like(weights)
                bias.grad = torch.ones_like(bias)
                with self.assertRaises(ValueError):
                    manual_sgd_update(weights, bias, learning_rate)


class TestTinyTrainingLoop(unittest.TestCase):
    def test_loss_decreases_and_parameters_approach_target_relation(self):
        weights, bias, losses = train_simple_relationship(
            steps=100, learning_rate=0.05
        )

        self.assertEqual(weights.shape, torch.Size([1, 1]))
        self.assertEqual(bias.shape, torch.Size([1]))
        self.assertEqual(len(losses), 101)
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
        self.assertLess(losses[-1], losses[0])

        torch.testing.assert_close(
            weights, torch.tensor([[2.0]]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            bias, torch.tensor([1.0]), atol=1e-3, rtol=1e-3
        )

    def test_rejects_invalid_step_count(self):
        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    train_simple_relationship(steps=steps)


if __name__ == "__main__":
    unittest.main()
