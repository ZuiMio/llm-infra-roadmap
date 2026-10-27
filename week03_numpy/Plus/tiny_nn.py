"""Tiny multiclass linear classifier with gradients derived by hand."""

from __future__ import annotations

import math

import numpy as np

from week03_numpy.Plus.math_ops import (
    cross_entropy_from_logits,
    linear_forward,
    softmax,
)


def make_synthetic_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, linearly separable 2D, three-class dataset."""
    offsets = np.array(
        [
            [0.00, 0.00],
            [-0.25, 0.15],
            [0.20, 0.10],
            [-0.15, -0.20],
            [0.10, -0.15],
        ],
        dtype=np.float64,
    )
    centers = np.array(
        [
            [-2.0, -1.0],
            [2.0, -1.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    features = np.concatenate([center + offsets for center in centers], axis=0)
    labels = np.repeat(np.arange(len(centers), dtype=np.int64), len(offsets))
    return features, labels


def linear_classifier_loss_and_gradients(
    inputs,
    labels,
    weights,
    bias,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mean cross-entropy and analytical gradients for W and b.

    Forward:
        logits = inputs @ weights + bias
        probabilities = softmax(logits)
        loss = cross_entropy_from_logits(logits, labels)

    For integer class labels and mean cross-entropy:
        d_logits = probabilities
        d_logits[row, labels[row]] -= 1
        d_logits /= batch_size
        dW = inputs.T @ d_logits
        db = d_logits.sum(axis=0)

    No autograd is used; gradients are calculated explicitly with NumPy.
    """
    x = np.asarray(inputs)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("inputs must be a non-empty 2D matrix.")
    if not (
        np.issubdtype(x.dtype, np.integer)
        or np.issubdtype(x.dtype, np.floating)
    ):
        raise ValueError("inputs must contain real numeric values.")
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("inputs must contain finite values.")

    w = np.asarray(weights)
    b = np.asarray(bias)
    if not (
        np.issubdtype(w.dtype, np.integer)
        or np.issubdtype(w.dtype, np.floating)
    ):
        raise ValueError("weights must contain real numeric values.")
    if not (
        np.issubdtype(b.dtype, np.integer)
        or np.issubdtype(b.dtype, np.floating)
    ):
        raise ValueError("bias must contain real numeric values.")
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    logits = linear_forward(x, w, b)
    probabilities = softmax(logits, axis=1)
    loss = cross_entropy_from_logits(logits, labels)

    label_array = np.asarray(labels)
    d_logits = probabilities.copy()
    d_logits[np.arange(x.shape[0]), label_array] -= 1.0
    d_logits /= x.shape[0]

    grad_weights = x.T @ d_logits
    grad_bias = np.sum(d_logits, axis=0)
    return loss, grad_weights, grad_bias


def predict_classes(inputs, weights, bias) -> np.ndarray:
    """Return predicted class IDs for each input row."""
    logits = linear_forward(inputs, weights, bias)
    return np.argmax(softmax(logits, axis=1), axis=1)


def train_linear_classifier(
    steps: int = 300,
    learning_rate: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train the synthetic classifier using manual gradient descent.

    Loss history has steps+1 values: the initial loss, one loss before each
    subsequent update, and a final loss after the last update.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError("steps must be a positive integer.")
    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate, (int, float, np.integer, np.floating)
    ):
        raise ValueError("learning_rate must be a positive finite number.")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive finite number.")

    x, labels = make_synthetic_classification_data()
    weights = np.zeros((x.shape[1], int(labels.max()) + 1), dtype=np.float64)
    bias = np.zeros(weights.shape[1], dtype=np.float64)

    loss_history: list[float] = []
    for _ in range(int(steps)):
        loss, grad_weights, grad_bias = linear_classifier_loss_and_gradients(
            x, labels, weights, bias
        )
        loss_history.append(loss)
        weights -= float(learning_rate) * grad_weights
        bias -= float(learning_rate) * grad_bias

    final_loss, _, _ = linear_classifier_loss_and_gradients(x, labels, weights, bias)
    loss_history.append(final_loss)
    return weights, bias, loss_history
