"""
Week 01 - Day 06
NumPy arrays, shapes, and broadcasting.

Topics:
- ndarray, shape, ndim, size, and dtype
- Array creation, indexing, and slicing
- reshape() and transpose()
- Element-wise operations versus matrix multiplication
- Reduction operations and axis
- Broadcasting
- Matrix multiplication with batched inputs

Rules:
- Solve each exercise without looking up full solutions.
- Use NumPy; do not use PyTorch.
- Keep the given function signatures.
- Use English comments and docstrings.
- Before running code, predict the output shape on paper.
- Add assertions for values, shapes, and edge cases.

Install (if needed):
    python -m pip install numpy

Run:
    python week01_python/day06.py
"""

import numpy as np


# ============================================================
# Exercise 1 - ndarray properties
# ============================================================

def describe_array(values):
    """
    Create a NumPy array from values and return a dictionary:
        {
            "shape": tuple of dimension sizes,
            "ndim": number of dimensions,
            "size": total number of elements,
            "dtype": dtype as a string,
        }

    Example:
        describe_array([[1, 2, 3], [4, 5, 6]])
        -> {
            "shape": (2, 3),
            "ndim": 2,
            "size": 6,
            "dtype": "int64" or another platform-dependent integer dtype,
        }

    Hint: use np.asarray() and str(array.dtype).
    """
    values_array = np.asarray(values)
    return {
        "shape": values_array.shape,
        "ndim": values_array.ndim,
        "size": values_array.size,
        "dtype": str(values_array.dtype),
    }


# ============================================================
# Exercise 2 - Array creation and reshape
# ============================================================

def create_grid(rows, cols):
    """
    Return an integer ndarray containing consecutive values
    from 0 to rows * cols - 1, arranged in (rows, cols).

    Example:
        create_grid(2, 3)
        -> array([[0, 1, 2],
                  [3, 4, 5]])

    Requirements:
    - Use np.arange() and reshape().
    - Assume rows and cols are positive integers.
    """
    return np.arange(rows * cols).reshape(rows, cols)


# ============================================================
# Exercise 3 - Indexing and slicing
# ============================================================

def select_columns(matrix):
    """
    Return every second column of a 2D NumPy array,
    starting from the first column (index 0).

    Example:
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
        -> [[1, 3],
            [5, 7]]

    Requirements:
    - Use NumPy slicing, not a Python loop.
    - Preserve the two-dimensional output.
    """
    return matrix[:, ::2]


# ============================================================
# Exercise 4 - Reshape
# ============================================================

def flatten_batch(x):
    """
    Reshape an array of shape (B, T, C) into (B, T * C).

    Example:
        x.shape == (2, 3, 4)
        flatten_batch(x).shape == (2, 12)

    Requirements:
    - Determine B from the input shape.
    - Do not hard-code any dimensions.
    - Do not modify the values in x.
    """
    B, T, C = x.shape
    return x.reshape(B, T * C)


# ============================================================
# Exercise 5 - Transpose
# ============================================================

def swap_time_and_channels(x):
    """
    Reorder axes of an ndarray from (B, T, C) to (B, C, T).

    Example:
        x.shape == (2, 3, 4)
        swap_time_and_channels(x).shape == (2, 4, 3)

    Requirements:
    - Use np.transpose() or ndarray.transpose().
    - Do not flatten or reshape the data to swap axes.
    """
    return x.transpose(0, 2, 1)


# ============================================================
# Exercise 6 - Reduction and axis
# ============================================================

def mean_over_time(x):
    """
    Compute the mean across the time dimension (axis 1).

    Input shape:  (B, T, C)
    Output shape: (B, C)

    Example:
        x = np.array([[[1, 2], [3, 4]]])
        mean_over_time(x)
        -> array([[2., 3.]])

    Requirements:
    - Use a NumPy reduction with the correct axis.
    - Assume T is nonzero.
    """
    return np.mean(x, axis=1)


# ============================================================
# Exercise 7 - Element-wise operations
# ============================================================

def scale_and_shift(x, scale, shift):
    """
    Compute an element-wise affine transformation.

    y = x * scale + shift

    Example:
        x = np.array([1, 2, 3])
        scale_and_shift(x, 2, 1)
        -> array([3, 5, 7])

    Requirements:
    - Support scalars and broadcast-compatible arrays.
    - Use element-wise operators (* and +), not matrix multiply.
    - Do not modify the inputs.
    """
    return x * scale + shift


# ============================================================
# Exercise 8 - Matrix multiplication
# ============================================================

def multiply_matrices(a, b):
    """
    Multiply two 2D arrays using matrix multiplication.

    Input shapes:
        a: (M, K)
        b: (K, N)

    Output shape:
        (M, N)

    Example:
        a = np.array([[1, 2]])
        b = np.array([[3], [4]])
        multiply_matrices(a, b) -> array([[11]])

    Requirements:
    - Use @ or np.matmul(), not element-wise *.
    - Let NumPy report incompatible dimensions.
    """
    return a @ b


# ============================================================
# Exercise 9 - Broadcasting a bias
# ============================================================

def add_bias(x, bias):
    """
    Add a feature bias to every sample.

    Input shapes:
        x:    (B, C)
        bias: (C,)

    Output shape:
        (B, C)

    Example:
        x = np.array([[1, 2], [3, 4]])
        bias = np.array([10, 20])
        add_bias(x, bias)
        -> array([[11, 22], [13, 24]])

    Requirements:
    - Use broadcasting.
    - Do not use explicit Python loops or np.tile().
    """
    return x + bias


# ============================================================
# Exercise 10 - Broadcasting in three dimensions
# ============================================================

def scale_features(x, feature_scale):
    """
    Scale each channel of a batched sequence.

    Input shapes:
        x:             (B, T, C)
        feature_scale: (C,)

    Output shape:
        (B, T, C)

    Requirements:
    - Use NumPy broadcasting.
    - Do not use explicit loops or np.tile().
    - Explain in your notes which axes are broadcast.
    """
    return x * feature_scale


# ============================================================
# Mini task - A shape-aware data pipeline
# ============================================================

def summarize_and_project(x, weight, bias):
    """
    Summarize sequence features and apply a matrix projection.

    Pipeline:
        x (B, T, C)
          -> mean over time (B, C)
          -> matrix multiply by weight (C, O)
          -> add bias (O,)
          -> output (B, O)

    Requirements:
    - Reuse your mean_over_time() and add_bias() functions.
    - Use @ for the matrix projection.
    - Return the projected output as an ndarray.
    - Do not use NumPy neural-network libraries or PyTorch.

    Example shapes:
        x.shape      == (2, 3, 4)
        weight.shape == (4, 5)
        bias.shape   == (5,)
        result.shape == (2, 5)
    """
    return add_bias(mean_over_time(x) @ weight, bias)


# ============================================================
# Shape prediction quiz
# ============================================================

# Write your answers in NOTES.md BEFORE executing the code.
#
# Q1: (4, 3) @ (3, 2)             -> (4, 2)
# Q2: (2, 5, 3) + (3,)            -> (2, 5, 3)
# Q3: (8, 6, 4).mean(axis=1)     -> (8, 4)
# Q4: (2, 3, 4).transpose(0,2,1) -> (2, 4, 3)
# Q5: (4, 3) * (3,)               -> (4, 3)
# Q6: (4, 3) @ (4, 3)             -> ValueError: 3 != 4
#      The inner dimensions of a 2D matrix product must match.
# Q7: (2, 1, 3) + (1, 4, 1)      -> (2, 4, 3)


# ============================================================
# Tests
# ============================================================

def run_tests():
    """Check shapes, values, broadcasting, and error cases."""

    # Exercise 1: ndarray metadata.
    values = [[1, 2, 3], [4, 5, 6]]
    info = describe_array(values)
    assert info == {
        "shape": (2, 3),
        "ndim": 2,
        "size": 6,
        "dtype": str(np.asarray(values).dtype),
    }
    assert describe_array([])["shape"] == (0,)

    # Exercise 2: array creation and reshaping.
    np.testing.assert_array_equal(
        create_grid(2, 3),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    assert create_grid(1, 4).shape == (1, 4)

    # Exercise 3: slicing must preserve two dimensions.
    matrix = np.arange(8).reshape(2, 4)
    np.testing.assert_array_equal(
        select_columns(matrix),
        np.array([[0, 2], [4, 6]]),
    )
    assert select_columns(matrix).ndim == 2

    # Exercise 4: flatten only the time and channel dimensions.
    x = np.arange(24).reshape(2, 3, 4)
    original_x = x.copy()
    flattened = flatten_batch(x)
    assert flattened.shape == (2, 12)
    np.testing.assert_array_equal(flattened, x.reshape(2, 12))

    # Exercise 5: transpose must change axis order, not element order.
    swapped = swap_time_and_channels(x)
    assert swapped.shape == (2, 4, 3)
    assert swapped[1, 3, 2] == x[1, 2, 3]

    # Exercise 6: reduce the sequence axis, not the batch or channels.
    sample = np.array([[[1, 2], [3, 4]]])
    np.testing.assert_allclose(
        mean_over_time(sample),
        np.array([[2.0, 3.0]]),
    )

    # Exercise 7: element-wise operations and scalar broadcasting.
    values = np.array([1, 2, 3])
    np.testing.assert_array_equal(
        scale_and_shift(values, 2, 1),
        np.array([3, 5, 7]),
    )

    # Exercise 8: matrix multiplication differs from element-wise *.
    a = np.array([[1, 2]])
    b = np.array([[3], [4]])
    np.testing.assert_array_equal(
        multiply_matrices(a, b),
        np.array([[11]]),
    )
    try:
        multiply_matrices(np.ones((4, 3)), np.ones((4, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched matrix dimensions must fail")

    # Exercise 9: add a one-dimensional bias to every row.
    inputs = np.array([[1, 2], [3, 4]])
    bias = np.array([10, 20])
    inputs_before = inputs.copy()
    bias_before = bias.copy()
    np.testing.assert_array_equal(
        add_bias(inputs, bias),
        np.array([[11, 22], [13, 24]]),
    )
    np.testing.assert_array_equal(inputs, inputs_before)
    np.testing.assert_array_equal(bias, bias_before)

    # Exercise 10: (C,) broadcasts across both B and T.
    feature_scale = np.array([2, 3, 4, 5])
    scaled = scale_features(x, feature_scale)
    assert scaled.shape == (2, 3, 4)
    np.testing.assert_array_equal(scaled[0, 0], x[0, 0] * feature_scale)

    # Mini task: reduce T, multiply (C, O), and broadcast the bias.
    inputs = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
    weights = np.array([[2, 0, 1], [0, 3, 1]])
    bias = np.array([1, -1, 2])
    originals = [array.copy() for array in (inputs, weights, bias)]
    output = summarize_and_project(inputs, weights, bias)
    assert output.shape == (2, 3)
    np.testing.assert_allclose(
        output,
        np.array([[5, 8, 7], [13, 20, 15]]),
    )
    for array, before in zip((inputs, weights, bias), originals):
        np.testing.assert_array_equal(array, before)

    # Check the shape quiz's broadcasting example.
    assert (np.ones((2, 1, 3)) + np.ones((1, 4, 1))).shape == (2, 4, 3)
    np.testing.assert_array_equal(x, original_x)

    print("All Day 06 tests passed!")


if __name__ == "__main__":
    run_tests()
