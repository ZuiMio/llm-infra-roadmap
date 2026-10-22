"""
Week 03 - Day 07
Mini project: a tested NumPy batch-feature pipeline.

Goals:
- Fit feature-wise standardization statistics from training data.
- Reuse the same statistics when transforming other data.
- Compute a linear forward pass with NumPy matrix multiplication.
- Process rows in batches and preserve the full output order.

Rules:
- Use NumPy, not PyTorch.
- Keep the function signatures.
- Use English comments and docstrings.
- Do not mutate input arrays.
- Replace each TODO and run the unit tests.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day07.test_day07 -v
"""

import numpy as np


def fit_standardizer(train_features):
    """
    Fit feature-wise means and population standard deviations on training data.

    Requirements:
    - Accept array-like input and convert it to a float64 NumPy array.
    - Require a non-empty 2D matrix of real numeric values.
    - Reject strings, booleans, complex values, NaN, and infinity.
    - Return (mean, std), each with shape (n_features,).
    - Compute statistics across samples (axis=0) using ddof=0.
    - Do not mutate the input.
    - Raise ValueError for invalid input or non-finite computed statistics.
    """
    # TODO 1: Validate the training matrix, then fit one mean and std per feature.
    raise NotImplementedError("Implement fit_standardizer")


def transform_features(features, mean, std):
    """
    Standardize a feature matrix using previously fitted statistics.

    Requirements:
    - Require a non-empty 2D real-valued finite feature matrix.
    - mean and std must be finite, real-valued 1D arrays of length n_features.
    - Reject strings, booleans, complex values, NaN, and infinity.
    - Standard deviations must be non-negative; zero is allowed.
    - Compute (features - mean) / std using NumPy broadcasting.
    - Replace zero standard deviations with a denominator of 1.0.
    - Return float64 without mutating any input.
    - Raise ValueError for invalid input or non-finite output.
    """
    # TODO 2: Validate data and statistics, then normalize by broadcasting.
    raise NotImplementedError("Implement transform_features")


def linear_forward(inputs, weights, bias):
    """
    Compute a linear layer forward pass for a 2D batch of inputs.

    Requirements:
    - inputs must be a non-empty 2D real-valued finite matrix.
    - weights must be a non-empty 2D real-valued finite matrix.
    - bias must be a 1D real-valued finite array.
    - weights has shape (in_features, out_features).
    - bias has shape (out_features,).
    - inputs.shape[1] must equal weights.shape[0].
    - Return inputs @ weights + bias as float64.
    - Do not mutate inputs, weights, or bias.
    - Raise ValueError for invalid dimensions, mismatched shapes, or invalid values.
    """
    # TODO 3: Validate shapes and values, then compute the linear forward pass.
    raise NotImplementedError("Implement linear_forward")


def predict_in_batches(features, mean, std, weights, bias, batch_size):
    """
    Transform features and compute linear outputs in row batches.

    The supplied mean and std must have been fitted on training data. Do not
    recalculate them from features inside this function.

    Requirements:
    - Validate features, fitted statistics, weights, bias, and batch_size.
    - batch_size must be a positive integer; reject booleans.
    - Normalize each batch with transform_features-equivalent behavior.
    - Compute each batch's outputs with linear_forward-equivalent behavior.
    - Concatenate results in the original row order.
    - Return a 2D float64 array with shape (n_samples, out_features).
    - Support a final batch smaller than batch_size and batch_size > n_samples.
    - Do not mutate any input.
    - Raise ValueError for invalid inputs or incompatible dimensions.
    """
    # TODO 4: Validate once, process row slices, and concatenate batch outputs.
    raise NotImplementedError("Implement predict_in_batches")


if __name__ == "__main__":
    train_features = np.array([
        [1.0, 10.0, 5.0],
        [3.0, 14.0, 5.0],
        [5.0, 18.0, 5.0],
    ])
    new_features = np.array([
        [2.0, 12.0, 5.0],
        [4.0, 16.0, 5.0],
        [6.0, 20.0, 5.0],
        [8.0, 24.0, 5.0],
    ])
    weights = np.array([
        [1.0, -1.0],
        [0.5, 2.0],
        [0.25, 0.0],
    ])
    bias = np.array([0.1, -0.2])

    mean, std = fit_standardizer(train_features)
    predictions = predict_in_batches(
        new_features, mean, std, weights, bias, batch_size=2
    )

    print("Training mean:", mean)
    print("Training population std:", std)
    print("Batched output shape:", predictions.shape)
    print(predictions)
