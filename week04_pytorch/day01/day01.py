"""
Week 04 - Day 01
PyTorch tensors, a scalar computation graph, and a tiny linear forward pass.

Today's goals:
- Create a scalar tensor that tracks gradients.
- Understand y = x ** 2 and verify its derivative with backward().
- Compute inputs @ weights + bias using Tensor operations.
- Compute a scalar mean-squared-error loss and inspect parameter gradients.

Replace each TODO, then run:
    python -m unittest week04_pytorch.day01.test_day01 -v

Do not use nn.Linear or torch.optim in these exercises.
"""

from __future__ import annotations

import torch


def make_scalar_square(value: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a gradient-tracking scalar x and y = x ** 2.

    Do not call backward() here. The caller should inspect the graph and
    choose when to run backward propagation.
    """
    # TODO:
    # 1. Create a scalar floating-point tensor x from value.
    # 2. Make sure x tracks gradients.
    # 3. Compute y = x ** 2.
    # 4. Return (x, y).
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    y = x ** 2
    return x, y


def linear_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute inputs @ weights + bias without using nn.Linear.

    Expected shapes:
        inputs: [batch_size, in_features]
        weights: [in_features, out_features]
        bias: [out_features]
        output: [batch_size, out_features]
    """
    # TODO: Use matrix multiplication and broadcasting.
    # Do not detach the tensors; autograd must be able to track the operations.
    return inputs @ weights + bias


def mean_squared_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return the mean of (predictions - targets) squared.

    The returned loss should be a scalar tensor so loss.backward() is valid.
    """
    # TODO: Compute element-wise differences, square them, and take the mean.
    return ((predictions - targets) ** 2).mean()


if __name__ == "__main__":
    x, y = make_scalar_square()
    print("x:", x)
    print("y:", y)
    y.backward()
    print("x.grad:", x.grad)

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
    bias = torch.tensor([0.25], requires_grad=True)

    predictions = linear_forward(inputs, weights, bias)
    targets = torch.zeros_like(predictions)
    loss = mean_squared_error(predictions, targets)
    loss.backward()

    print("predictions:", predictions)
    print("loss:", loss)
    print("weights.grad:", weights.grad)
    print("bias.grad:", bias.grad)
