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
    # 1. Validate steps and learning_rate.
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer.")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number.")

    # 2. Validate tensor dimensions and the compatible shape contract.
    if inputs.ndim != 2:
        raise ValueError("inputs must be a 2D tensor [batch_size, in_features].")
    if targets.ndim != 2:
        raise ValueError("targets must be a 2D tensor [batch_size, out_features].")
    if initial_weights.ndim != 2:
        raise ValueError("initial_weights must be a 2D tensor [in_features, out_features].")
    if initial_bias.ndim != 1:
        raise ValueError("initial_bias must be a 1D tensor [out_features].")

    batch_size, in_features = inputs.shape
    _, out_features = initial_weights.shape

    if targets.shape != (batch_size, out_features):
        raise ValueError("targets shape must match [batch_size, out_features].")
    if initial_weights.shape[0] != in_features:
        raise ValueError("initial_weights rows must match inputs' in_features.")
    if initial_bias.shape[0] != out_features:
        raise ValueError("initial_bias size must match initial_weights' out_features.")

    # 3. Clone initial_weights and initial_bias into leaf tensors requiring grad.
    weights = initial_weights.clone().detach().requires_grad_(True)
    bias = initial_bias.clone().detach().requires_grad_(True)

    loss_history: list[float] = []

    # 4. Implement the complete training loop.
    for _ in range(steps):
        # Forward pass & loss calculation
        predictions = inputs @ weights + bias
        loss = ((predictions - targets) ** 2).mean()
        
        # Record loss before the update
        loss_history.append(loss.item())

        # Compute gradients
        loss.backward()

        # Update parameters manually under no_grad() and clear gradients
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

    # Record one final loss after training
    with torch.no_grad():
        final_predictions = inputs @ weights + bias
        final_loss = ((final_predictions - targets) ** 2).mean()
        loss_history.append(final_loss.item())

    # 5. Return detached copies of (weights, bias, loss_history).
    return weights.detach().clone(), bias.detach().clone(), loss_history


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
