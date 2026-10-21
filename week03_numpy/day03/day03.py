"""
Week 03 - Day 03
NumPy broadcasting and vectorized operations.

Goals:
- Add a feature-wise bias to a batch of examples.
- Scale each matrix row using one scale value per row.
- Standardize features using per-feature statistics.
- Apply ReLU without explicit Python loops.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate input arrays.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day03.test_day03 -v
"""

import numpy as np


def add_feature_bias(features, bias):
    """
    Add one bias value per feature to a 2D batch.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - features must be 2D with shape (n_samples, n_features).
    - bias must be 1D with shape (n_features,).
    - Return the broadcast result without mutating either input.
    - Raise ValueError for invalid dimensions or a mismatched feature count.
    """
    # TODO 1: Validate the shapes and add bias through broadcasting.
    raise NotImplementedError("Implement add_feature_bias")


def scale_rows(matrix, row_scales):
    """
    Multiply each row by its corresponding scale value.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - matrix must be 2D.
    - row_scales must be 1D with one value per matrix row.
    - Make the scale array broadcast along columns, not rows.
    - Return a new result and do not mutate either input.
    - Raise ValueError for invalid dimensions or a mismatched row count.
    """
    # TODO 2: Reshape row_scales so it broadcasts down the rows.
    raise NotImplementedError("Implement scale_rows")


def standardize_features(features, mean, std):
    """
    Standardize each feature using per-feature means and standard deviations.

    Requirements:
    - Convert array-like inputs with np.asarray().
    - features must be 2D; mean and std must be 1D.
    - mean and std must each have one value per feature.
    - Every value in std must be strictly positive.
    - Return (features - mean) / std using broadcasting.
    - Do not mutate any input; raise ValueError for invalid shapes or std.
    """
    # TODO 3: Validate shapes and standard deviations, then standardize.
    raise NotImplementedError("Implement standardize_features")


def vectorized_relu(array):
    """
    Apply ReLU element by element without explicit Python loops.

    Requirements:
    - Convert array-like input with np.asarray().
    - Support arrays with any number of dimensions.
    - Return an array with the same shape, replacing negative values with 0.
    - Do not mutate the input.
    """
    # TODO 4: Use a NumPy element-wise operation rather than a Python loop.
    raise NotImplementedError("Implement vectorized_relu")


if __name__ == "__main__":
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = np.array([10.0, 20.0, 30.0])

    print("Features plus bias:")
    print(add_feature_bias(features, bias))

    matrix = np.array([[1.0, 10.0], [100.0, 1000.0]])
    print("Scaled rows:")
    print(scale_rows(matrix, [2.0, 3.0]))

    print("Standardized features:")
    print(standardize_features(features, [2.5, 3.5, 4.5], [1.5, 1.5, 1.5]))

    print("ReLU:")
    print(vectorized_relu(np.array([[-2.0, 0.0], [3.0, -4.0]])))
