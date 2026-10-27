"""
Week 04 - Day 06
Manual SGD update under torch.no_grad() and a tiny training loop.

Run from the repository root:
    python -m unittest week04_pytorch.day06.test_day06 -v

Prerequisite: complete Day 04's linear_forward and Day 01's MSE function.
Do not use torch.optim.
"""

from __future__ import annotations

import math

import torch

from week04_pytorch.day01.day01 import mean_squared_error
from week04_pytorch.day04.day04 import linear_forward


def manual_sgd_update(
    weights: torch.Tensor,
    bias: torch.Tensor,
    learning_rate: float,
) -> None:
    """Update weights and bias in place, without tracking the update.

    Preconditions: weights.grad and bias.grad have already been computed.
    Requirements:
    - Validate that learning_rate is a positive finite number.
    - Raise RuntimeError if either gradient is None.
    - Update parameters inside torch.no_grad().
    - Clear the consumed gradients by setting .grad = None.
    - Do not call torch.optim.
    """
    # TODO:
    # 1. Validate learning_rate.
    # 2. Ensure both tensors have gradients; otherwise raise RuntimeError.
    # 3. Under torch.no_grad(), subtract learning_rate * grad from each parameter.
    # 4. Set weights.grad and bias.grad to None.
    raise NotImplementedError("Implement manual_sgd_update.")


def train_simple_relationship(
    steps: int = 100,
    learning_rate: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Fit y = 2*x + 1 with a scalar-input, scalar-output linear model.

    Use the fixed samples x = [-2, -1, 0, 1, 2]. Initialize weight=-0.5
    and bias=0.0. For each step:
      1. forward pass
      2. MSE loss
      3. append the current loss to the history
      4. backward()
      5. manual_sgd_update()

    Append one final loss after all updates. Return detached copies of
    (weights, bias, loss_history). The list length should be steps + 1.
    """
    # TODO:
    # 1. Validate steps (positive integer; bool is not valid) and learning_rate.
    # 2. Create the fixed input and target tensors.
    # 3. Create trainable weight and bias leaf tensors.
    # 4. Implement the training loop in the docstring.
    # 5. Compute and append the final loss after the last update.
    # 6. Return detached copies of the trained parameters and the loss history.
    raise NotImplementedError("Implement train_simple_relationship.")


if __name__ == "__main__":
    weights, bias, losses = train_simple_relationship()
    print("trained weight:", weights)
    print("trained bias:", bias)
    print("initial loss:", losses[0])
    print("final loss:", losses[-1])
    print("number of recorded losses:", len(losses))
