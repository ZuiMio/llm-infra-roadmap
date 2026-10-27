"""
Week 04 - Day 05
Compute a scalar loss and inspect gradients for weights and bias.

Run from the repository root:
    python -m unittest week04_pytorch.day05.test_day05 -v

Prerequisite: complete Day 04's linear_forward implementation.
Do not use nn.Linear or torch.optim.
"""

from __future__ import annotations

import torch

from week04_pytorch.day01.day01 import mean_squared_error
from week04_pytorch.day04.day04 import linear_forward


def compute_loss_and_gradients(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predictions, scalar MSE loss, weight gradient, and bias gradient.

    Clear old parameter gradients before this independent backward pass.
    Do not change the parameter values and do not use an optimizer.

    Returns:
        predictions
        loss
        weight_gradient
        bias_gradient
    """
    # TODO:
    # 1. Clear weights.grad and bias.grad.
    # 2. Use linear_forward() to calculate predictions.
    # 3. Use mean_squared_error() to calculate a scalar loss.
    # 4. Call loss.backward().
    # 5. Clone weights.grad and bias.grad.
    # 6. Return predictions, loss, and the two gradient copies.
    raise NotImplementedError("Implement compute_loss_and_gradients.")


if __name__ == "__main__":
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
    bias = torch.tensor([0.25], requires_grad=True)
    targets = torch.zeros((2, 1))

    predictions, loss, weight_grad, bias_grad = compute_loss_and_gradients(
        inputs, weights, bias, targets
    )
    print("predictions:", predictions)
    print("loss:", loss.item())
    print("weights.grad:", weight_grad)
    print("bias.grad:", bias_grad)
