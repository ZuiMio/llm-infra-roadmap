"""
Week 03 - Day 05
NumPy matrix multiplication, batch dimensions, and shape contracts.

Goals:
- Distinguish element-wise multiplication from matrix multiplication.
- Validate matrix dimensions before multiplication.
- Understand batch dimensions and broadcasting in np.matmul.
- Implement a simple linear layer forward pass.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate input arrays.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day05.test_day05 -v
"""

import numpy as np


def matrix_multiply(left, right):
    """
    Multiply two 2D matrices.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - Both inputs must be 2D.
    - The number of columns in left must equal the number of rows in right.
    - Return left @ right without mutating either input.
    - Raise ValueError for invalid dimensions or incompatible matrix shapes.
    """
    # TODO 1: Convert inputs, validate their shapes, and multiply them.
    # raise NotImplementedError("Implement matrix_multiply")
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("invalid dimensions or incompatible matrix shapes.")
    return left @ right


def batched_matrix_multiply(left, right):
    """
    Multiply arrays using matrix multiplication on their final two dimensions.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - Both inputs must have at least two dimensions.
    - The final dimension of left must match the second-to-last dimension of right.
    - Leading batch dimensions must be broadcast-compatible.
    - Support ordinary 2D matrices and arrays with batch dimensions.
    - Raise ValueError for invalid or incompatible shapes.
    - Do not mutate either input.
    """
    # TODO 2: Validate matrix and batch dimensions, then use np.matmul.
    # raise NotImplementedError("Implement batched_matrix_multiply")
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim < 2:
        raise ValueError(f"left must have at least 2 dimensions, got {left.ndim}")
    if right.ndim < 2:
        raise ValueError(f"right must have at least 2 dimensions, got {right.ndim}")
    if left.shape[-1] != right.shape[-2]:
        raise ValueError(
            f"Final dimension of left ({left.shape[-1]}) must match "
            f"second-to-last dimension of right ({right.shape[-2]})."
        )
    left_batch = left.shape[:-2]
    right_batch = right.shape[:-2]
    try:
        np.broadcast_shapes(left_batch, right_batch)
    except ValueError as e:
        raise ValueError(
            f"Batch dimensions are not broadcast-compatible: {left_batch} and {right_batch}."
        ) from e
        
    return np.matmul(left, right)



def linear_forward(inputs, weights, bias):
    """
    Compute the forward pass of a linear layer.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - inputs must have at least two dimensions.
    - weights must be 2D with shape (in_features, out_features).
    - bias must be 1D with shape (out_features,).
    - The final input dimension must match weights.shape[0].
    - The bias length must match weights.shape[1].
    - Return inputs @ weights + bias using broadcasting.
    - Raise ValueError for invalid dimensions or mismatched feature sizes.
    - Do not mutate any input.
    """
    # TODO 3: Validate dimensions, multiply inputs by weights, and add bias.
    # raise NotImplementedError("Implement linear_forward")
    inputs = np.asarray(inputs)
    weights = np.asarray(weights)
    bias = np.asarray(bias)
    if (
        inputs.ndim < 2
        or weights.ndim != 2
        or bias.ndim != 1
        or inputs.shape[-1] != weights.shape[0]
        or bias.shape[0] != weights.shape[1]
    ):
        raise ValueError("invalid dimensions or mismatched feature sizes.")
    return inputs @ weights + bias


if __name__ == "__main__":
    left = np.array([[1, 2], [3, 4]])
    right = np.array([[10, 20], [30, 40]])
    print("Element-wise multiplication:")
    print(left * right)
    print("Matrix multiplication:")
    print(left @ right)

    batch = np.ones((4, 2, 3))
    weights = np.ones((3, 5))
    print("Batched matrix multiplication shape:", (batch @ weights).shape)
