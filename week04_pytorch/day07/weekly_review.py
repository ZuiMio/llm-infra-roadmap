"""
Week 04 - Day 07
Weekly review: independently rebuild a tiny trainable linear model.

Run from the repository root:
    python -m unittest week04_pytorch.day07.test_weekly_review -v

This is an integration challenge. Do not import the helper implementations
from day01, day04, day05, or day06. Rebuild the forward pass, MSE loss,
backward pass, manual no_grad update, and gradient reset here.
"""

from __future__ import annotations

import math

import torch


def train_linear_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: torch.Tensor,
    steps: int = 100,
    learning_rate: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Train a linear model from scratch and return trained tensors and loss.

    Shape contract:
        inputs:         [batch_size, in_features]
        targets:        [batch_size, out_features]
        initial_weights:[in_features, out_features]
        initial_bias:   [out_features]

    Requirements:
    - Validate the input/target/parameter shapes, steps, and learning rate.
    - Create new trainable leaf parameters from the initial tensors.
    - Do not use nn.Linear, torch.nn.functional.linear, or torch.optim.
    - Write the linear forward pass as inputs @ weights + bias.
    - Write MSE explicitly as ((predictions - targets) ** 2).mean().
    - Call loss.backward().
    - Update weights and bias under torch.no_grad().
    - Clear weights.grad and bias.grad after each update.
    - Record a loss before each update and one final loss after training.
    - Return detached copies of (weights, bias, loss_history).
    """
    # TODO:
    # 1. Validate steps and learning_rate.
    # 2. Validate tensor dimensions and the compatible shape contract above.
    # 3. Clone initial_weights and initial_bias into leaf tensors requiring grad.
    # 4. Implement the complete training loop from the docstring.
    # 5. Return trained parameter copies and the history.
    raise NotImplementedError("Implement train_linear_model.")


if __name__ == "__main__":
    inputs = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = 2.0 * inputs + 1.0
    initial_weights = torch.tensor([[-0.5]])
    initial_bias = torch.tensor([0.0])

    weights, bias, losses = train_linear_model(
        inputs, targets, initial_weights, initial_bias
    )
    print("trained weight:", weights)
    print("trained bias:", bias)
    print("initial loss:", losses[0])
    print("final loss:", losses[-1])
