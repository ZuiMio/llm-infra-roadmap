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
    raise NotImplementedError("Implement matrix_multiply")


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
    raise NotImplementedError("Implement batched_matrix_multiply")


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
    raise NotImplementedError("Implement linear_forward")


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
