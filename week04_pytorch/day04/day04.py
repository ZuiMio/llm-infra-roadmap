"""
Week 04 - Day 04
A hand-written linear forward pass and tensor shape contracts.

Today's goals:
- Compute inputs @ weights + bias without nn.Linear.
- Predict matrix multiplication dimensions before running the code.
- Understand why a bias vector of shape [out_features] broadcasts across the batch.
- Preserve the autograd graph by not detaching inputs or parameters.

Run from the repository root:
    python -m unittest week04_pytorch.day04.test_day04 -v
"""

from __future__ import annotations

import torch


def linear_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute a linear layer using Tensor operations only.

    Shape contract:
        inputs:  [batch_size, in_features]
        weights: [in_features, out_features]
        bias:    [out_features]
        output:  [batch_size, out_features]

    Do not use nn.Linear, torch.nn.functional.linear, or detach().
    """
    # TODO:
    # 1. Compute inputs @ weights.
    # 2. Add bias using broadcasting.
    # 3. Return the result without breaking the computation graph.
    raise NotImplementedError("Implement linear_forward.")


if __name__ == "__main__":
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
    print("inputs.shape:", inputs.shape)
    print("weights.shape:", weights.shape)
    print("bias.shape:", bias.shape)
    print("output.shape:", output.shape)
    print(output)
