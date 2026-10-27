"""Integration tests for the Week 04 independent training-loop review."""

import unittest

import torch

from week04_pytorch.day07.weekly_review import train_linear_model


class TestWeeklyReview(unittest.TestCase):
    def test_rebuilds_training_loop_and_learns_known_relation(self):
        inputs = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        targets = 2.0 * inputs + 1.0
        initial_weights = torch.tensor([[-0.5]])
        initial_bias = torch.tensor([0.0])

        weights, bias, losses = train_linear_model(
            inputs,
            targets,
            initial_weights,
            initial_bias,
            steps=100,
            learning_rate=0.05,
        )

        self.assertEqual(weights.shape, torch.Size([1, 1]))
        self.assertEqual(bias.shape, torch.Size([1]))
        self.assertEqual(len(losses), 101)
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
        torch.testing.assert_close(
            weights, torch.tensor([[2.0]]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            bias, torch.tensor([1.0]), atol=1e-3, rtol=1e-3
        )

        # Inputs and initialization values must not be mutated.
        torch.testing.assert_close(inputs, torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]]))
        torch.testing.assert_close(initial_weights, torch.tensor([[-0.5]]))
        torch.testing.assert_close(initial_bias, torch.tensor([0.0]))

    def test_supports_multiple_input_and_output_features(self):
        inputs = torch.tensor([
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ])
        true_weights = torch.tensor([[1.5, -0.5], [0.75, 2.0]])
        true_bias = torch.tensor([0.25, -1.0])
        targets = inputs @ true_weights + true_bias

        weights, bias, losses = train_linear_model(
            inputs,
            targets,
            initial_weights=torch.zeros_like(true_weights),
            initial_bias=torch.zeros_like(true_bias),
            steps=300,
            learning_rate=0.03,
        )

        self.assertEqual(weights.shape, torch.Size([2, 2]))
        self.assertEqual(bias.shape, torch.Size([2]))
        self.assertLess(losses[-1], losses[0])
        torch.testing.assert_close(weights, true_weights, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bias, true_bias, atol=2e-2, rtol=2e-2)

    def test_rejects_invalid_steps_and_learning_rate(self):
        inputs = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [3.0]])
        initial_weights = torch.zeros((1, 1))
        initial_bias = torch.zeros(1)

        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    train_linear_model(
                        inputs, targets, initial_weights, initial_bias, steps=steps
                    )

        for learning_rate in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    train_linear_model(
                        inputs, targets, initial_weights, initial_bias,
                        learning_rate=learning_rate,
                    )

    def test_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            train_linear_model(
                inputs=torch.ones((3, 2)),
                targets=torch.ones((3, 1)),
                initial_weights=torch.ones((3, 1)),
                initial_bias=torch.ones(1),
            )

        with self.assertRaises(ValueError):
            train_linear_model(
                inputs=torch.ones((3, 2)),
                targets=torch.ones((3, 1)),
                initial_weights=torch.ones((2, 1)),
                initial_bias=torch.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
