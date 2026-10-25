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
    # 1. Create x as a scalar float32 leaf tensor requiring gradients.
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    
    # 2. Compute middle = 3 * x.
    middle = 3 * x
    
    # 3. Compute output = middle ** 2.
    output = middle ** 2
    
    # 4. Return all three tensors.
    return x, middle, output


def backward_with_retained_intermediate(
    value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run backward() and retain the gradient for the middle tensor.

    Return (x, middle, output). The middle tensor is not a leaf, so call
    middle.retain_grad() before output.backward() if you want middle.grad
    to be populated. Do not detach tensors from the graph.
    """
    # 1. Build the graph using build_computation_graph.
    x, middle, output = build_computation_graph(value)
    
    # 2. Call middle.retain_grad() before backward().
    middle.retain_grad()
    
    # 3. Call output.backward().
    output.backward()
    
    # 4. Return (x, middle, output).
    return x, middle, output


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