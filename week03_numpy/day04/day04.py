"""
Week 03 - Day 04
NumPy reductions, axes, keepdims, and numerical precision.

Goals:
- Sum values along a selected axis.
- Compute per-feature means with optional kept dimensions.
- Center each feature using its mean.
- Compute mean absolute error with a float64 accumulator.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate input arrays.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day04.test_day04 -v
"""

import numpy as np


def sum_array(array, axis=None, keepdims=False):
    """
    Sum values over an axis or over the entire array.

    Requirements:
    - Convert array-like input with np.asarray().
    - Require at least one dimension.
    - axis may be None or a valid integer axis, including negative axes.
    - Reject booleans and non-integer axis values with ValueError.
    - Use a float64 accumulator for more careful summation.
    - Pass keepdims through to the reduction.
    - Do not mutate the input array.
    """
    # TODO 1: Validate dimensions and axis, then call np.sum.
    # raise NotImplementedError("Implement sum_array")
    array = np.asarray(array)
    if array.ndim < 1:
        raise ValueError("array requires at least one dimension.")
        
    if axis is not None:
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise ValueError("axis must be None or valid integer axis.")
            
    return np.sum(array, axis=axis, dtype=np.float64, keepdims=keepdims)


def mean_features(features, keepdims=False):
    """
    Compute the mean of each feature across the sample axis.

    Requirements:
    - Convert array-like input with np.asarray().
    - features must be a non-empty 2D array.
    - Compute the mean along axis 0.
    - Use a float64 accumulator for the mean.
    - If keepdims is False, return shape (n_features,).
    - If keepdims is True, return shape (1, n_features).
    - Do not mutate the input array.
    """
    # TODO 2: Validate the input and compute a feature-wise mean.
    # raise NotImplementedError("Implement mean_features")
    features = np.asarray(features)
    if features.ndim != 2 or features.size == 0:
        raise ValueError("features must be a non-empty 2D array.")
    return np.mean(features, axis=0, dtype=np.float64, keepdims=keepdims)


def center_features(features):
    """
    Center each feature by subtracting its mean across samples.

    Requirements:
    - Convert array-like input with np.asarray().
    - features must be a non-empty 2D numeric array.
    - Compute per-feature means using axis 0 and keepdims=True.
    - Convert calculations to float64 to make reduction precision explicit.
    - Return an array with the same shape and a mean close to zero per feature.
    - Do not mutate the input.
    """
    # TODO 3: Compute means with keepdims, then subtract them by broadcasting.
    # raise NotImplementedError("Implement center_features")
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        raise ValueError("features must be a non-empty 2D array.")
    return features - np.mean(features, axis=0, dtype=np.float64, keepdims=True)



def mean_absolute_error(predictions, targets):
    """
    Compute mean absolute error for arrays with matching shapes.

    Requirements:
    - Convert array-like inputs to float64 NumPy arrays.
    - predictions and targets must have identical, non-empty shapes.
    - Support arrays with any number of dimensions.
    - Compute mean(abs(predictions - targets)) with float64 arithmetic.
    - Return a Python float and do not mutate the inputs.
    - Raise ValueError for mismatched shapes or empty input.
    """
    # TODO 4: Validate shapes and size, then compute the mean absolute error.
    # raise NotImplementedError("Implement mean_absolute_error")
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError(
            "predictions and targets must have identical, non-empty shapes."
        )

    errors = np.abs(predictions - targets)
    return float(np.mean(errors, dtype=np.float64))


if __name__ == "__main__":
    values = np.array([[1, 2, 3], [4, 5, 6]])
    print("Sum over columns:", sum_array(values, axis=0))
    print("Sum over rows, keeping dimensions:", sum_array(values, axis=1, keepdims=True))

    print("Feature means:", mean_features(values))
    print("Feature means with keepdims:", mean_features(values, keepdims=True))
    print("Centered features:")
    print(center_features(values))

    predictions = np.array([1, 2, 4])
    targets = np.array([2, 2, 1])
    print("Mean absolute error:", mean_absolute_error(predictions, targets))
