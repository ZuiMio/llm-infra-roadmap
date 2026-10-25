"""
Week 04 - Day 02
Leaf tensors, requires_grad, grad_fn, and retained intermediate gradients.

Today's goals:
- Distinguish leaf tensors from intermediate tensors.
- Inspect requires_grad and grad_fn before backward().
- See where gradients are stored after backward().
- Use retain_grad() when you explicitly need a non-leaf tensor's .grad.

Run from the repository root:
    python -m unittest week04_pytorch.day02.test_day02 -v
"""

from __future__ import annotations

import torch


def build_computation_graph(
    value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build x -> middle -> output, without running backward().

    The intended operations are:
        x = scalar(value), with requires_grad=True
        middle = 3 * x
        output = middle ** 2

    Return (x, middle, output). Do not call backward() here.
    """
    # TODO:
    # 1. Create x as a scalar float32 leaf tensor requiring gradients.
    # 2. Compute middle = 3 * x.
    # 3. Compute output = middle ** 2.
    # 4. Return all three tensors.
    raise NotImplementedError("Implement build_computation_graph.")


def backward_with_retained_intermediate(
    value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run backward() and retain the gradient for the middle tensor.

    Return (x, middle, output). The middle tensor is not a leaf, so call
    middle.retain_grad() before output.backward() if you want middle.grad
    to be populated. Do not detach tensors from the graph.
    """
    # TODO:
    # 1. Build the graph using build_computation_graph.
    # 2. Call middle.retain_grad() before backward().
    # 3. Call output.backward().
    # 4. Return (x, middle, output).
    raise NotImplementedError("Implement backward_with_retained_intermediate.")


if __name__ == "__main__":
    x, middle, output = build_computation_graph()
    print("Before backward:")
    print("x:", x, "| is_leaf:", x.is_leaf, "| requires_grad:", x.requires_grad)
    print("middle:", middle, "| is_leaf:", middle.is_leaf, "| grad_fn:", middle.grad_fn)
    print("output:", output, "| is_leaf:", output.is_leaf, "| grad_fn:", output.grad_fn)
    print("x.grad before backward:", x.grad)

    x, middle, output = backward_with_retained_intermediate()
    print("\nAfter backward with middle.retain_grad():")
    print("x.grad:", x.grad)
    print("middle.grad:", middle.grad)
