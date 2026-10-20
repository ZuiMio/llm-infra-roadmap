"""
Week 03 - Day 01
NumPy memory layout, strides, views, and copies.

Goals:
- Inspect shape, dtype, strides, and memory size.
- Understand when a basic slice shares memory with its source.
- Distinguish a transposed view from a contiguous copy.
- Predict array layout before running code.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate inputs inside the functions.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.test_day01 -v
"""

import numpy as np


def describe_array(array):
    """
    Return a dictionary describing an array's layout.

    Required keys:
        shape: tuple of dimension sizes
        ndim: number of dimensions
        dtype: dtype name as a string
        strides: tuple of byte steps for each axis
        nbytes: number of bytes used by the array's elements
        c_contiguous: whether elements follow C-contiguous layout

    Requirements:
    - Convert the input with np.asarray().
    - Do not force a copy just to inspect the array.
    - Return Python bool for c_contiguous.
    """
    # TODO 1: Use np.asarray() and return all six fields.
    raise NotImplementedError("Implement describe_array")


def slice_view_and_copy(matrix):
    """
    Return (view, copied_slice) for a 2D array.

    Requirements:
    - The input must be 2D and have at least three columns.
    - Raise ValueError if the input does not satisfy those requirements.
    - view must select columns 1 and 2 using basic slicing.
    - copied_slice must contain the same values but own independent memory.
    - Do not mutate matrix.
    """
    # TODO 2: Convert the input to an array and validate its shape.
    # Create the basic slice, then make an explicit copy.
    raise NotImplementedError("Implement slice_view_and_copy")


def transpose_layout(matrix):
    """
    Return (transposed_view, contiguous_copy) for a 2D array.

    Requirements:
    - Raise ValueError if matrix is not 2D.
    - transposed_view must be the transposed array.
    - contiguous_copy must have the same values as transposed_view.
    - contiguous_copy must use C-contiguous layout.
    - Do not mutate the input.
    """
    # TODO 3: Transpose the array and create an independent C-order copy.
    raise NotImplementedError("Implement transpose_layout")


if __name__ == "__main__":
    example = np.arange(12, dtype=np.float64).reshape(3, 4)
    print("Original:", describe_array(example))
    view, copied = slice_view_and_copy(example)
    print("View:", describe_array(view))
    print("Copy:", describe_array(copied))
    transposed, contiguous = transpose_layout(example)
    print("Transposed:", describe_array(transposed))
    print("Contiguous copy:", describe_array(contiguous))
