"""
Week 03 - Day 06
NumPy feature validation, statistics, and normalization.

Goals:
- Validate feature matrices before calculation.
- Compute feature-wise means and population standard deviations.
- Normalize features without mutating inputs.
- Compose the operations into a small preprocessing pipeline.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day06.test_day06 -v
"""

import numpy as np


def validate_feature_matrix(features, expected_n_features=None):
    """
    Validate a non-empty, finite, real-valued 2D feature matrix.

    Requirements:
    - Convert array-like input to a NumPy array.
    - Require a 2D matrix with at least one sample and one feature.
    - Require real numeric values; reject strings, booleans, and complex values.
    - Reject NaN and infinity.
    - If expected_n_features is provided, require a positive integer matching
      the matrix's feature count; reject booleans as integer values.
    - Return a float64 array without mutating the input.
    - Raise ValueError for invalid input.
    """
    # TODO 1: Validate the matrix, feature count, dtype, and finite values.
    raise NotImplementedError("Implement validate_feature_matrix")


def compute_feature_stats(features):
    """
    Compute per-feature means and population standard deviations.

    Requirements:
    - Validate features with validate_feature_matrix().
    - Compute along axis 0, using float64 calculations.
    - Use population standard deviation (ddof=0).
    - Return (mean, std), each with shape (n_features,).
    - A constant feature is valid and has standard deviation zero.
    - Do not mutate the input.
    """
    # TODO 2: Validate features, then compute feature-wise mean and std.
    raise NotImplementedError("Implement compute_feature_stats")


def normalize_features(features, mean, std):
    """
    Normalize features using previously computed per-feature statistics.

    Requirements:
    - Validate features with validate_feature_matrix().
    - mean and std must be finite, real-valued 1D arrays matching n_features.
    - Standard deviations must be non-negative; zero is allowed.
    - Compute (features - mean) / std using broadcasting.
    - To avoid division by zero, use a denominator of 1 for zero-std features.
      This makes a constant training feature normalize to zero while retaining
      deviations if a later dataset differs from its fitted constant value.
    - Return a float64 array without mutating any input.
    - Raise ValueError for invalid statistics.
    """
    # TODO 3: Validate statistics and normalize each feature by broadcasting.
    raise NotImplementedError("Implement normalize_features")


def preprocess_features(features):
    """
    Validate a feature matrix, fit its statistics, and normalize it.

    Requirements:
    - Validate features.
    - Compute per-feature means and population standard deviations.
    - Normalize using those statistics.
    - Return (normalized, mean, std), in that order.
    - Do not mutate the input.
    """
    # TODO 4: Compose the three earlier functions into one pipeline.
    raise NotImplementedError("Implement preprocess_features")


if __name__ == "__main__":
    features = np.array([
        [1.0, 10.0, 5.0],
        [3.0, 14.0, 5.0],
        [5.0, 18.0, 5.0],
    ])

    normalized, mean, std = preprocess_features(features)
    print("Mean:", mean)
    print("Population standard deviation:", std)
    print("Normalized features:")
    print(normalized)
