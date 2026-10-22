# Day 5 — Matrix Multiplication and Batch Dimensions

## Learning goals

- Distinguish element-wise multiplication (`*`) from matrix multiplication (`@`).
- Predict output shapes before running matrix operations.
- Understand how the final two dimensions represent matrices and leading dimensions represent batches.
- Use batch broadcasting with NumPy's `matmul`.
- Understand the shape contract of a simple linear layer.

Use NumPy only for this week's exercises; do not use PyTorch yet.

## 1. Element-wise multiplication versus matrix multiplication

For arrays with matching shapes, `left * right` multiplies corresponding elements. The `left @ right` operator performs matrix multiplication.

For two 2D matrices:

`left.shape == (M, K)` and `right.shape == (K, N)`

The inner dimensions must match (`K`). The result shape is `(M, N)`.

```python
import numpy as np

left = np.array([[1, 2], [3, 4]])
right = np.array([[10, 20], [30, 40]])

print(left * right)
print(left @ right)
```

For the top-left value of `left @ right`, calculate the dot product of the first row of `left` and the first column of `right`: `1 * 10 + 2 * 30 = 70`.

## 2. Shape rule for batched matrix multiplication

For arrays with at least two dimensions, NumPy treats the final two dimensions as matrix dimensions and all leading dimensions as batch dimensions.

`(..., M, K) @ (..., K, N) -> (..., M, N)`

The leading batch dimensions must be broadcast-compatible. Each pair of matrices is multiplied after their batch dimensions are aligned.

| Left shape | Right shape | Result shape |
| --- | --- | --- |
| `(2, 3)` | `(3, 4)` | `(2, 4)` |
| `(4, 2, 3)` | `(3, 5)` | `(4, 2, 5)` |
| `(4, 2, 3)` | `(4, 3, 5)` | `(4, 2, 5)` |
| `(2, 1, 3, 4)` | `(1, 5, 4, 6)` | `(2, 5, 3, 6)` |

For the last row, the batch dimensions `(2, 1)` and `(1, 5)` broadcast to `(2, 5)`, and the matrix dimensions produce `(3, 6)`.

Two common failures:

- `(2, 3) @ (2, 4)` fails because `3 != 2` for the inner matrix dimensions.
- `(2, 2, 3) @ (4, 3, 5)` fails because batch sizes `2` and `4` are not broadcast-compatible.

## 3. Linear layer shape contract

A linear layer can be expressed as:

`output = inputs @ weights + bias`

For inputs shaped `(batch_size, seq_len, in_features)`, weights shaped `(in_features, out_features)`, and bias shaped `(out_features,)`:

`(B, S, H) @ (H, O) -> (B, S, O)`

The bias is broadcast over the batch and sequence dimensions. This is the basic shape pattern behind a projection layer in a neural network.

## 4. Exercises

Implement the three functions in `day05.py`:

1. `matrix_multiply(left, right)`: two 2D matrices; validate dimensions and inner-size compatibility.
2. `batched_matrix_multiply(left, right)`: support 2D and higher-dimensional inputs; validate matrix dimensions and batch broadcasting.
3. `linear_forward(inputs, weights, bias)`: validate the input feature dimension, weight shape, and bias length; compute `inputs @ weights + bias`.

For all three functions:

- Convert array-like inputs using `np.asarray()`.
- Raise `ValueError` for invalid dimensions or incompatible shapes.
- Do not mutate the input arrays.
- Keep the provided function signatures.

## 5. Run the tests

Run this command from the repository root:

```bash
python3.14 -m unittest week03_numpy.day05.test_day05 -v
```

The starter code contains TODOs, so the tests will not pass until you implement the functions.

## 6. Before coding, predict the shapes

1. `(5, 7) @ (7, 4)` -> `(5, 4)`
2. `(8, 5, 7) @ (7, 4)` -> `(8, 5, 4)`
3. `(2, 1, 5, 7) @ (1, 3, 7, 4)` -> `(2, 3, 5, 4)`
4. `(3, 5, 7) @ (4, 7, 2)` -> invalid: the batch dimensions `3` and `4` cannot broadcast.

Explain each answer by identifying the matrix dimensions and the batch dimensions separately.
