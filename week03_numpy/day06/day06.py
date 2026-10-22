
"""
Week 03 - Day 06
NumPy feature validation, statistics, and normalization.

Goals:
- Validate feature matrices before calculation.
- Compute feature-wise means and population standard deviations.
- Normalize features without mutating inputs.
- Compose the operations into a small preprocessing pipeline.

Run from the repository root:
    python3.14 -m unittest week03_numpy.day06.test_day06 -v
"""

import numpy as np


def _is_real_numeric_dtype(dtype):
    """Return whether a dtype represents real integer or floating-point values."""
    return (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _to_float64_real_array(values, name):
    """
    Validate real numeric values and return a finite float64 copy.

    Boolean, string, object, and complex dtypes are rejected.
    """
    array = np.asarray(values)

    if not _is_real_numeric_dtype(array.dtype):
        raise ValueError(f"{name} must contain real numeric values.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float64, copy=True)

    # Conversion from a wider numeric type may overflow to infinity.
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} cannot be represented as finite float64 values.")

    return result


def validate_feature_matrix(features, expected_n_features=None):
    """
    Validate a non-empty, finite, real-valued 2D feature matrix.

    Return a float64 copy without modifying the input.
    Raise ValueError for invalid input.
    """
    features = np.asarray(features)

    if features.ndim != 2:
        raise ValueError(
            f"Features must be a 2D matrix, got {features.ndim} dimensions."
        )

    n_samples, n_features = features.shape

    if n_samples == 0 or n_features == 0:
        raise ValueError(
            "Features must contain at least one sample and one feature."
        )

    if expected_n_features is not None:
        if (
            isinstance(expected_n_features, (bool, np.bool_))
            or not isinstance(expected_n_features, (int, np.integer))
        ):
            raise ValueError("expected_n_features must be an integer.")

        if expected_n_features <= 0:
            raise ValueError("expected_n_features must be positive.")

        if n_features != expected_n_features:
            raise ValueError(
                f"Feature count mismatch: expected {expected_n_features}, "
                f"got {n_features}."
            )

    return _to_float64_real_array(features, "features")


def _compute_feature_stats_validated(features):
    """Compute per-feature means and population standard deviations."""
    mean = np.mean(features, axis=0, dtype=np.float64)
    std = np.std(features, axis=0, dtype=np.float64, ddof=0)
    return mean, std


def compute_feature_stats(features):
    """
    Compute per-feature means and population standard deviations.

    Return (mean, std), each with shape (n_features,).
    Constant features are valid and have standard deviation zero.
    """
    validated = validate_feature_matrix(features)
    return _compute_feature_stats_validated(validated)


def _validate_statistic_vector(values, name, n_features):
    """Validate a finite 1D statistic vector with the expected length."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array, got {array.ndim} dimensions."
        )

    if array.shape[0] != n_features:
        raise ValueError(
            f"{name} length ({array.shape[0]}) must match "
            f"feature count ({n_features})."
        )

    return _to_float64_real_array(array, name)


def _normalize_validated(features, mean, std):
    """Normalize validated features using validated statistics."""
    safe_std = np.where(std == 0.0, 1.0, std)
    return (features - mean) / safe_std


def normalize_features(features, mean, std):
    """
    Normalize features using supplied per-feature statistics.

    Uses broadcasting and a safe denominator for zero-standard-deviation
    features. Returns float64 without modifying any input.
    """
    validated_features = validate_feature_matrix(features)
    n_features = validated_features.shape[1]

    mean = _validate_statistic_vector(mean, "mean", n_features)
    std = _validate_statistic_vector(std, "std", n_features)

    if np.any(std < 0.0):
        raise ValueError("Standard deviations must be non-negative.")

    return _normalize_validated(validated_features, mean, std)


def preprocess_features(features):
    """
    Validate, fit feature statistics, and normalize a feature matrix.

    Return (normalized, mean, std), in that order.
    The input is validated only once and is never modified.
    """
    validated = validate_feature_matrix(features)
    mean, std = _compute_feature_stats_validated(validated)
    normalized = _normalize_validated(validated, mean, std)

    return normalized, mean, std


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
