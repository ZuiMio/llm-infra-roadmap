"""NumPy reference implementations for foundational neural-network operations.

All floating-point calculations use float64 for clear numerical behavior.
"""

from __future__ import annotations

import math

import numpy as np


def _as_real_array(values, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Convert real integer/floating data to float64 and validate it."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid array-like object.") from exc

    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain real numeric values.")

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as float64.") from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def linear_forward(inputs, weights, bias) -> np.ndarray:
    """Compute a dense layer: inputs @ weights + bias.

    Shapes:
        inputs:  [batch_size, in_features]
        weights: [in_features, out_features]
        bias:    [out_features]
        output:  [batch_size, out_features]
    """
    x = _as_real_array(inputs, "inputs", ndim=2)
    w = _as_real_array(weights, "weights", ndim=2)
    b = _as_real_array(bias, "bias", ndim=1)

    if min(x.shape) <= 0 or min(w.shape) <= 0:
        raise ValueError("inputs and weights must have non-zero dimensions.")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            "inputs.shape[1] must match weights.shape[0] for matrix multiplication."
        )
    if b.shape != (w.shape[1],):
        raise ValueError("bias must have shape (out_features,).")

    with np.errstate(over="ignore", invalid="ignore"):
        output = x @ w + b

    if not np.isfinite(output).all():
        raise ValueError("linear output contains non-finite values.")
    return output


def relu(values) -> np.ndarray:
    """Apply ReLU elementwise without mutating the input."""
    x = _as_real_array(values, "values")
    return np.maximum(x, 0.0)


def softmax(logits, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along an axis."""
    x = _as_real_array(logits, "logits")
    if x.ndim == 0:
        raise ValueError("logits must have at least one dimension.")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer.")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range for logits.")

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=axis, keepdims=True)
    probabilities = exp_values / denominator

    if not np.isfinite(probabilities).all():
        raise ValueError("softmax produced non-finite values.")
    return probabilities


def cross_entropy_from_logits(logits, labels) -> float:
    """Return mean multiclass cross-entropy from raw logits.

    logits must have shape [batch_size, num_classes].
    labels must be integer class IDs with shape [batch_size].
    This uses the log-sum-exp trick rather than applying log to softmax
    probabilities, which is more stable for large-magnitude logits.
    """
    x = _as_real_array(logits, "logits", ndim=2)
    if min(x.shape) <= 0:
        raise ValueError("logits dimensions must be non-zero.")

    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must have shape (batch_size,).")
    if not np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
        raise ValueError("labels must contain integer class IDs.")
    if np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ValueError("labels contain a class ID outside the valid range.")

    shifted = x - np.max(x, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    log_probabilities = shifted - log_normalizer[:, None]
    loss = -np.mean(log_probabilities[np.arange(x.shape[0]), y])

    if not math.isfinite(float(loss)):
        raise ValueError("cross-entropy loss is non-finite.")
    return float(loss)


def layer_norm(
    inputs,
    gamma=None,
    beta=None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalize along the final feature dimension, optionally affine.

    For each row/vector along the final axis:
        normalized = (x - mean) / sqrt(var + eps)
        output = normalized * gamma + beta

    gamma and beta, when provided, must each have shape [features].
    Variance is the population variance (ddof=0).
    """
    x = _as_real_array(inputs, "inputs")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("inputs must have a non-empty feature dimension.")
    if isinstance(eps, bool) or not isinstance(eps, (int, float, np.integer, np.floating)):
        raise ValueError("eps must be a positive finite number.")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a positive finite number.")

    feature_count = x.shape[-1]
    if gamma is None:
        scale = np.ones(feature_count, dtype=np.float64)
    else:
        scale = _as_real_array(gamma, "gamma", ndim=1)
        if scale.shape != (feature_count,):
            raise ValueError("gamma must have shape (features,).")

    if beta is None:
        shift = np.zeros(feature_count, dtype=np.float64)
    else:
        shift = _as_real_array(beta, "beta", ndim=1)
        if shift.shape != (feature_count,):
            raise ValueError("beta must have shape (features,).")

    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + float(eps))
    output = normalized * scale + shift

    if not np.isfinite(output).all():
        raise ValueError("layer_norm produced non-finite values.")
    return output
