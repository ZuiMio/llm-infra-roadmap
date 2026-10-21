"""
Week 03 - Day 02
NumPy advanced indexing, boolean masks, and copy behavior.

Goals:
- Select elements with integer-array indexing.
- Filter elements with boolean masks.
- Replace values by condition without mutating the input.
- Select paired coordinates from a 2D matrix.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate input arrays.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day02.test_day02 -v
"""

import numpy as np


def select_by_indices(array, indices):
    """
    Select elements from a 1D array using integer-array indexing.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - The input array must be 1D.
    - indices must be a 1D array of integers.
    - Return an independent result that does not share memory with array.
    - Do not mutate the input array.
    - Raise ValueError for invalid dimensions or non-integer indices.
    """
    # TODO 1: Convert and validate the inputs, then select by integer indices.
    # raise NotImplementedError("Implement select_by_indices")
    array = np.asarray(array)
    indices = np.asarray(indices)
    if array.ndim != 1 or indices.ndim !=1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("invalid dimensions or non-integer indices.")
    return array[indices].copy()


def select_by_mask(array, mask):
    """
    Select elements whose corresponding boolean mask value is True.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - The input array must be 1D.
    - mask must contain booleans and have the same shape as array.
    - Return the selected elements as an independent result.
    - Do not mutate the input array.
    - Raise ValueError for an invalid array dimension, mask dtype, or shape.
    """
    # TODO 2: Convert and validate the inputs, then apply the boolean mask.
    # raise NotImplementedError("Implement select_by_mask")
    arr = np.asarray(array)
    msk = np.asarray(mask)
    
    # 1. 检查输入数组是否为一维
    if arr.ndim != 1:
        raise ValueError("The input array must be 1D.")
        
    # 2. 检查 mask 是否为布尔类型
    if msk.dtype != bool:
        raise ValueError("mask must contain booleans.")
        
    # 3. 检查形状是否一致
    if arr.shape != msk.shape:
        raise ValueError("mask must have the same shape as array.")
        
    # 4. 应用布尔索引并返回独立副本
    return arr[msk].copy()


def replace_above_threshold(array, threshold, replacement):
    """
    Return a copy where values strictly above threshold are replaced.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - The input array must be 1D.
    - Replace values satisfying array > threshold.
    - Values equal to threshold must remain unchanged.
    - Return a new array and do not mutate the input.
    - Raise ValueError if array is not 1D.
    """
    # TODO 3: Copy the array first, then replace values using a boolean mask.
    # raise NotImplementedError("Implement replace_above_threshold")
    array = np.asarray(array)
    
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    
    # np.where(condition, x, y) 会在 condition 为 True 的地方取 x，False 的地方取 y
    # 并且它会返回一个新的拷贝，不会突变原数组
    return np.where(array > threshold, replacement, array)



def select_matrix_coordinates(matrix, rows, cols):
    """
    Select paired (row, column) coordinates from a 2D matrix.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - matrix must be 2D.
    - rows and cols must be 1D integer arrays of equal length.
    - Use paired coordinates: rows[i] with cols[i].
    - Return an independent 1D result and do not mutate matrix.
    - Raise ValueError for invalid dimensions, index dtypes, or lengths.
    """
    # TODO 4: Validate the inputs and select the paired coordinates.
    # raise NotImplementedError("Implement select_matrix_coordinates")
    matrix = np.asarray(matrix)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if (
        matrix.ndim != 2
        or rows.ndim != 1
        or cols.ndim != 1
        or len(rows) != len(cols)
        or not np.issubdtype(rows.dtype, np.integer)
        or not np.issubdtype(cols.dtype, np.integer)
    ):
        raise ValueError(
            "Invalid dimensions, index dtypes, or lengths."
        )

    return matrix[rows, cols].copy()


if __name__ == "__main__":
    values = np.array([10, 20, 30, 40, 50])
    indices = [0, 2, 4]
    mask = values > 25

    print("By indices:", select_by_indices(values, indices))
    print("By mask:", select_by_mask(values, mask))
    print("After replacement:", replace_above_threshold(values, 25, 0))

    matrix = np.arange(9).reshape(3, 3)
    print("Paired coordinates:", select_matrix_coordinates(matrix, [0, 2], [1, 0]))
