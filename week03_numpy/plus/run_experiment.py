"""Run the Week 03 Plus shape, numerical stability, and classifier demo."""

from __future__ import annotations

import numpy as np

from week03_numpy.plus.math_ops import (
    cross_entropy_from_logits,
    layer_norm,
    linear_forward,
    relu,
    softmax,
)
from week03_numpy.plus.tiny_nn import (
    make_synthetic_classification_data,
    predict_classes,
    train_linear_classifier,
)


def main() -> None:
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bias = np.array([0.5, -0.5])

    linear = linear_forward(inputs, weights, bias)
    print("Linear: inputs", inputs.shape, "@ weights", weights.shape, "+ bias", bias.shape)
    print("  output shape:", linear.shape, "values:", linear.tolist())

    activations = relu(np.array([[-2.0, 0.0, 3.0]]))
    print("ReLU output shape:", activations.shape, "values:", activations.tolist())

    large_logits = np.array([[10000.0, 10001.0, 10002.0]])
    probabilities = softmax(large_logits, axis=1)
    print("Softmax probabilities:", probabilities.tolist())
    print("Softmax row sum:", probabilities.sum(axis=1).tolist())

    labels = np.array([2], dtype=np.int64)
    print("Cross-entropy for large logits:", cross_entropy_from_logits(large_logits, labels))

    tokens = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    normalized = layer_norm(tokens)
    print("LayerNorm input shape:", tokens.shape, "output shape:", normalized.shape)
    print("LayerNorm means along feature axis:", normalized.mean(axis=-1).round(8).tolist())

    x, y = make_synthetic_classification_data()
    trained_weights, trained_bias, losses = train_linear_classifier()
    predictions = predict_classes(x, trained_weights, trained_bias)
    accuracy = float(np.mean(predictions == y))
    print("Classifier data shape:", x.shape, "labels shape:", y.shape)
    print("Classifier parameter shapes:", trained_weights.shape, trained_bias.shape)
    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])
    print("Training accuracy:", accuracy)


if __name__ == "__main__":
    main()
