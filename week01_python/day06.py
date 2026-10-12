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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


# ============================================================
# Shape prediction quiz
# ============================================================

# Write your answers in NOTES.md BEFORE executing the code.
#
# Q1: (4, 3) @ (3, 2)            -> shape = ?
# Q2: (2, 5, 3) + (3,)           -> shape = ?
# Q3: (8, 6, 4).mean(axis=1)    -> shape = ?
# Q4: (2, 3, 4).transpose(0,2,1)-> shape = ?
# Q5: (4, 3) * (3,)              -> shape = ?
# Q6: (4, 3) @ (4, 3)            -> valid or error? Why?
# Q7: (2, 1, 3) + (1, 4, 1)     -> shape = ?


# ============================================================
# Tests
# ============================================================

def run_tests():
    """
    Add assertions for all ten exercises and the mini task.

    Suggested checks:
    - Correct shape, ndim, size, and dtype information.
    - Array creation, slicing, and transpose results.
    - Preserved element values after reshape.
    - Mean reduction on the intended axis.
    - Element-wise multiplication versus matrix multiplication.
    - Broadcasting with (C,) vectors over (B, C) and (B, T, C).
    - Shape and numeric results of summarize_and_project().
    - Invalid matrix multiplication dimensions raise ValueError.
    - Inputs remain unchanged by the functions.
    """
    pass


if __name__ == "__main__":
    run_tests()
