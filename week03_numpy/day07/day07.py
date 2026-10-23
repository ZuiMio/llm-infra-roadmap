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


# ---------- Shared validation helpers ----------

def _is_real_numeric_dtype(dtype):
    """Return whether dtype is an integer or floating-point type."""
    return (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _require_finite(array, name):
    """Reject NaN and infinity."""
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_float64_array(values, name, ndim):
    """Convert valid, non-empty real numeric data to float64."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a valid array-like object."
        ) from exc

    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not _is_real_numeric_dtype(array.dtype):
        raise ValueError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)

    return _require_finite(array, name)


def _validate_feature_statistics(features, mean, std):
    """Validate feature statistic shapes and standard deviation values."""
    expected_shape = (features.shape[1],)

    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            "The lengths of mean and std must match the number of features."
        )

    if np.any(std < 0):
        raise ValueError("std must contain non-negative values.")


def _validate_linear_shapes(inputs, weights, bias):
    """Validate the shape contract for a linear layer."""
    if inputs.shape[1] != weights.shape[0]:
        raise ValueError(
            "Input feature count must match the weight input dimension."
        )

    if bias.shape != (weights.shape[1],):
        raise ValueError(
            "Bias length must match the output feature count."
        )


# ---------- Shared computation helpers ----------

def _normalize_validated(features, mean, std):
    """Normalize already-validated arrays using broadcasting."""
    safe_std = np.where(std == 0.0, 1.0, std)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (features - mean) / safe_std

    return _require_finite(normalized, "normalized features")


def _linear_forward_validated(inputs, weights, bias):
    """Compute a linear forward pass on already-validated arrays."""
    with np.errstate(over="ignore", invalid="ignore"):
        output = inputs @ weights + bias

    return _require_finite(output, "linear output")


# ---------- Public functions ----------

def fit_standardizer(train_features):
    """Fit per-feature means and population standard deviations."""
    features = _as_float64_array(
        train_features, "train_features", ndim=2
    )

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(features, axis=0, dtype=np.float64)
        std = np.std(features, axis=0, dtype=np.float64, ddof=0)

    _require_finite(mean, "computed mean")
    _require_finite(std, "computed standard deviation")

    return mean, std


def transform_features(features, mean, std):
    """Normalize features with previously fitted statistics."""
    features = _as_float64_array(features, "features", ndim=2)
    mean = _as_float64_array(mean, "mean", ndim=1)
    std = _as_float64_array(std, "std", ndim=1)

    _validate_feature_statistics(features, mean, std)

    return _normalize_validated(features, mean, std)


def linear_forward(inputs, weights, bias):
    """Compute inputs @ weights + bias."""
    inputs = _as_float64_array(inputs, "inputs", ndim=2)
    weights = _as_float64_array(weights, "weights", ndim=2)
    bias = _as_float64_array(bias, "bias", ndim=1)

    _validate_linear_shapes(inputs, weights, bias)

    return _linear_forward_validated(inputs, weights, bias)


def predict_in_batches(features, mean, std, weights, bias, batch_size):
    """Normalize features and compute linear outputs in row batches."""
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    # Validate all inputs once.
    features = _as_float64_array(features, "features", ndim=2)
    mean = _as_float64_array(mean, "mean", ndim=1)
    std = _as_float64_array(std, "std", ndim=1)
    weights = _as_float64_array(weights, "weights", ndim=2)
    bias = _as_float64_array(bias, "bias", ndim=1)

    # Validate relationships between inputs.
    _validate_feature_statistics(features, mean, std)
    _validate_linear_shapes(features, weights, bias)

    batch_outputs = []
    batch_size = int(batch_size)

    for start in range(0, features.shape[0], batch_size):
        feature_batch = features[start:start + batch_size]

        normalized_batch = _normalize_validated(
            feature_batch, mean, std
        )
        output_batch = _linear_forward_validated(
            normalized_batch, weights, bias
        )

        batch_outputs.append(output_batch)

    return np.concatenate(batch_outputs, axis=0)


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
