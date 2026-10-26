"""
Week 04 - Day 03
Gradient accumulation and why gradients must be cleared.

Today's goals:
- Observe that backward() accumulates into a leaf tensor's .grad.
- Compare accumulation across two separate forward computations.
- Clear .grad before a new step and verify the new gradient is independent.
- Understand the relation to optimizer.zero_grad() in a training loop.

Run from the repository root:
    python -m unittest week04_pytorch.day03.test_day03 -v
"""

from __future__ import annotations

import torch


def demonstrate_gradient_accumulation(
    value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x, gradient after the first backward(), and after the second.

    Use two separate losses built from the same leaf x:
        first_loss = x ** 2
        second_loss = 3 * x

    At value=2, the first gradient is 4. The second backward() adds 3
    to the existing .grad, so the accumulated gradient is 7.

    Each loss must be computed separately; do not call backward() twice on
    the same graph.
    """
    # TODO:
    # 1. Create x as a scalar float32 tensor requiring gradients.
    # 2. Compute first_loss = x ** 2 and call backward().
    # 3. Clone and save x.grad as grad_after_first.
    # 4. Compute second_loss = 3 * x and call backward().
    # 5. Clone and save x.grad as grad_after_second.
    # 6. Return x, grad_after_first, grad_after_second.
    raise NotImplementedError("Implement demonstrate_gradient_accumulation.")


def demonstrate_gradient_reset(
    value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x, the first gradient, and the gradient after clearing .grad.

    Compute first_loss = x ** 2 and backpropagate. Then clear x.grad by
    assigning None. Compute second_loss = 3 * x and backpropagate again.

    At value=2, the returned gradients should be 4 and 3, not 4 and 7.
    """
    # TODO:
    # 1. Create x as a scalar float32 tensor requiring gradients.
    # 2. Compute first_loss = x ** 2 and call backward().
    # 3. Clone and save x.grad as grad_before_reset.
    # 4. Clear the old gradient by assigning x.grad = None.
    # 5. Compute second_loss = 3 * x and call backward().
    # 6. Clone and save x.grad as grad_after_reset.
    # 7. Return x, grad_before_reset, grad_after_reset.
    raise NotImplementedError("Implement demonstrate_gradient_reset.")


if __name__ == "__main__":
    x, first, accumulated = demonstrate_gradient_accumulation()
    print("Accumulation demo at x=2:")
    print("gradient after first backward:", first)
    print("gradient after second backward:", accumulated)
    print("current x.grad:", x.grad)

    x, first, after_reset = demonstrate_gradient_reset()
    print("\nReset demo at x=2:")
    print("gradient before reset:", first)
    print("gradient after reset and second backward:", after_reset)
    print("current x.grad:", x.grad)
