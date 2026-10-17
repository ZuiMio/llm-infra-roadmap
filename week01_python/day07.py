"""
Week 01 - Day 07
Mini project: NumPy neural-network forward pass.

Goal:
    Build and explain this pipeline without PyTorch:

        Input -> Linear -> ReLU -> Prediction -> MSELoss

Topics:
- Matrix multiplication and bias broadcasting
- Object composition and __call__()
- Element-wise activation functions
- Mean squared error
- Tensor/array shapes and numerical correctness
- Testing a small forward-only computation graph

Rules:
- Write your own solutions in place of pass.
- Use NumPy, but do not use PyTorch or scikit-learn.
- Do NOT implement gradients, an optimizer, or a training loop today.
- Keep the public class/function signatures as provided.
- Use English comments and docstrings.
- Record each intermediate array shape in NOTES.md BEFORE running.
- Add assertions that check values, shapes, and input immutability.

Run:
    python -m pip install numpy
    python week01_python/day07.py
"""

import io
from contextlib import redirect_stdout

import numpy as np


# ============================================================
# Exercise 1 - Linear layer
# ============================================================

class Linear:
    """
    Implement the forward pass of a fully connected layer.

    Formula:
        y = x @ weight + bias

    Shapes:
        x:      (B, C)
        weight: (C, O)
        bias:   (O,)
        y:      (B, O)

    Requirements:
    - __init__ stores the weight and bias as instance attributes.
    - __call__(x) performs matrix multiplication and bias addition.
    - Use NumPy broadcasting; do not write a Python loop over samples.
    - Do not modify x, weight, or bias in place.
    - Assume the inputs have compatible shapes.

    Example:
        x = np.array([[1.0, 2.0]])
        w = np.array([[2.0, -1.0], [1.0, 3.0]])
        b = np.array([0.0, 2.0])
        Linear(w, b)(x) -> np.array([[4.0, 7.0]])
    """

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return x @ self.weight + self.bias


# ============================================================
# Exercise 2 - ReLU activation
# ============================================================

class ReLU:
    """
    Compute the rectified linear unit element by element.

    Formula:
        ReLU(x) = max(0, x)

    Requirements:
    - Accept an ndarray of any shape.
    - Return an ndarray with the same shape.
    - Do not modify the input in place.

    Example:
        ReLU()(np.array([-2.0, 0.0, 3.0]))
        -> np.array([0.0, 0.0, 3.0])
    """

    def __call__(self, x):
        x = np.asarray(x)
        return np.maximum(x, 0)


# ============================================================
# Exercise 3 - Mean squared error loss
# ============================================================

class MSELoss:
    """
    Compute the mean squared error between prediction and target.

    Formula:
        MSE = mean((prediction - target) ** 2)

    Requirements:
    - prediction and target must have exactly the same shape.
    - Raise ValueError if the shapes do not match.
      Do not silently allow broadcasting of target labels.
    - Assume the inputs are nonempty numerical arrays.
    - Return a scalar (NumPy scalar or Python float).
    - Do not modify either input.

    Example:
        MSELoss()(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        -> 2.5
    """

    def __call__(self, prediction, target):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction has shape {prediction.shape}, "
                f"but target has shape {target.shape}."
            )

        diff = prediction - target          # creates a new array, no in-place modification
        return np.mean(diff ** 2)



# ============================================================
# Exercise 4 - Compose layers into a simple model
# ============================================================

class TinyNetwork:
    """
    Combine Linear and ReLU into one callable model.

    Forward pipeline:
        x -> Linear(weight, bias) -> ReLU -> prediction

    Requirements:
    - Create self.linear and self.activation in __init__.
    - Reuse the existing classes instead of copying their formulas.
    - __call__ returns the final prediction.

    Shapes:
        x:          (B, C)
        prediction: (B, O)
    """

    def __init__(self, weight, bias):
        self.linear = Linear(weight, bias)
        self.activation = ReLU()

    def __call__(self, x):
        return self.activation(self.linear(x))


# ============================================================
# Exercise 5 - Full forward pass and loss
# ============================================================

def forward_and_loss(model, loss_fn, x, target):
    """
    Run the model and compute the loss.

    Return a dictionary:
        {
            "prediction": ndarray of shape (B, O),
            "loss": scalar mean squared error,
        }

    Requirements:
    - Call model(x) to get a prediction.
    - Call loss_fn(prediction, target).
    - Do not compute the loss manually here.
    """
    prediction = model(x)
    loss = loss_fn(prediction, target)
    return {
        "prediction": prediction,
        "loss": loss,
    }


# ============================================================
# Exercise 6 - Build a deterministic toy dataset
# ============================================================

def make_toy_example():
    """
    Create a small, reproducible example of a forward pass.

    Return a tuple: (x, weight, bias, target).

    Required shapes:
        x:      (3, 2)
        weight: (2, 2)
        bias:   (2,)
        target: (3, 2)

    Requirements:
    - Use np.array() with explicit numeric values, not random numbers.
    - Choose at least one input that produces a negative value
      before ReLU, so you can test the activation.
    - Keep this example small enough to calculate by hand.
    """
    x = np.array([
        [ 1.0,  2.0],
        [-1.0,  3.0],
        [ 2.0, -1.0],
    ])  # shape (3, 2)

    weight = np.array([
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])  # shape (2, 2)

    bias = np.array([0.0, -1.0])  # shape (2,)

    target = np.array([
        [ 1.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0],
    ])  # shape (3, 2)

    return x, weight, bias, target


# ============================================================
# Mini project - Run an end-to-end forward-only model
# ============================================================

def run_miniproject():
    """
    Connect the whole NumPy model.

    Requirements:
    - Get x, weight, bias, target from make_toy_example().
    - Construct TinyNetwork and MSELoss.
    - Call forward_and_loss().
    - Print the input, weight, bias, prediction, target, and loss.
    - Print shapes for x, weight, bias, prediction, and target.
    - Return the result dictionary.

    Do not add backpropagation or parameter updates.
    """
    # 1. Get the toy dataset.
    x, weight, bias, target = make_toy_example()

    # 2. Construct the model and loss function.
    model = TinyNetwork(weight, bias)
    loss_fn = MSELoss()

    # 3. Run the forward pass and compute the loss.
    result = forward_and_loss(model=model, loss_fn=loss_fn, x=x, target=target)

    # 4. Print the values.
    print("x        =", x)
    print("weight   =", weight)
    print("bias     =", bias)
    print("prediction =", result["prediction"])
    print("target   =", target)
    print("loss     =", result["loss"])

    # 5. Print the shapes.
    print("x.shape        =", x.shape)
    print("weight.shape   =", weight.shape)
    print("bias.shape     =", bias.shape)
    print("prediction.shape =", result["prediction"].shape)
    print("target.shape   =", target.shape)

    # 6. Return the result dictionary.
    return result


# ============================================================
# Shape prediction quiz
# ============================================================

# Record your answers in NOTES.md before running the program.
#
# Q1: x (4, 3) @ W (3, 2)               -> output shape = ?
# Q2: x @ W has shape (4, 2), b is (2,) -> final shape = ?
# Q3: ReLU gets an array (4, 2)         -> output shape = ?
# Q4: prediction and target are (4, 2) -> MSE shape = ?
# Q5: x (4, 3) @ W (4, 2)               -> valid or error? Why?
# Q6: prediction (4, 2), target (2,)    -> should MSELoss accept this?
#
# Explain:
# - Why is the Linear # ============================================================
# Tests
# ============================================================

def run_tests():
    """Check all exercises, expected values, shapes, and input immutability."""

    # Exercise 1: Linear computes x @ weight + bias and broadcasts bias by row.
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([[2.0, -1.0], [1.0, 3.0]])
    bias = np.array([0.0, 2.0])
    x_before, weight_before, bias_before = x.copy(), weight.copy(), bias.copy()

    linear = Linear(weight, bias)
    linear_output = linear(x)
    np.testing.assert_allclose(
        linear_output,
        np.array([[4.0, 7.0], [10.0, 11.0]]),
    )
    assert linear_output.shape == (2, 2)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(weight, weight_before)
    np.testing.assert_array_equal(bias, bias_before)

    # Different Linear instances can hold independent parameters.
    first = Linear(np.eye(2), np.zeros(2))
    second = Linear(np.zeros((2, 2)), np.ones(2))
    first_output = first(np.array([[2.0, 3.0]]))
    second_output = second(np.array([[2.0, 3.0]]))
    np.testing.assert_allclose(first_output, np.array([[2.0, 3.0]]))
    np.testing.assert_allclose(second_output, np.array([[1.0, 1.0]]))
    assert first.weight is not second.weight

    # Exercise 2: ReLU handles negative, zero, and positive values.
    relu_input = np.array([[-2.0, 0.0], [3.0, -4.0]])
    relu_before = relu_input.copy()
    relu_output = ReLU()(relu_input)
    np.testing.assert_array_equal(
        relu_output,
        np.array([[0.0, 0.0], [3.0, 0.0]]),
    )
    assert relu_output.shape == relu_input.shape
    assert relu_output is not relu_input
    np.testing.assert_array_equal(relu_input, relu_before)

    # Exercise 3: MSE returns a scalar and refuses target broadcasting.
    loss_fn = MSELoss()
    prediction = np.array([1.0, 3.0])
    target = np.array([2.0, 1.0])
    prediction_before, target_before = prediction.copy(), target.copy()
    loss = loss_fn(prediction, target)
    np.testing.assert_allclose(loss, 2.5)
    assert np.ndim(loss) == 0
    np.testing.assert_array_equal(prediction, prediction_before)
    np.testing.assert_array_equal(target, target_before)

    try:
        loss_fn(np.ones((2, 2)), np.ones((2,)))
    except ValueError:
        pass
    else:
        raise AssertionError("MSELoss must reject mismatched shapes")

    # Exercise 4: TinyNetwork composes Linear and ReLU.
    network_weight = np.array([[1.0, -1.0], [1.0, 1.0]])
    network_bias = np.array([0.0, -1.0])
    network_input = np.array([[1.0, 2.0], [-1.0, 3.0], [2.0, -1.0]])
    network = TinyNetwork(network_weight, network_bias)
    network_prediction = network(network_input)
    np.testing.assert_allclose(
        network_prediction,
        np.array([[3.0, 0.0], [2.0, 3.0], [1.0, 0.0]]),
    )
    assert network_prediction.shape == (3, 2)

    # Exercise 5: forward_and_loss returns the prediction and scalar loss.
    result = forward_and_loss(network, loss_fn, network_input, np.zeros((3, 2)))
    assert set(result) == {"prediction", "loss"}
    np.testing.assert_allclose(result["prediction"], network_prediction)
    np.testing.assert_allclose(
        result["loss"],
        np.mean(network_prediction ** 2),
    )
    assert np.ndim(result["loss"]) == 0

    # Exercise 6: the deterministic toy data has the documented shapes and
    # produces at least one negative pre-activation before ReLU.
    toy_x, toy_weight, toy_bias, toy_target = make_toy_example()
    assert toy_x.shape == (3, 2)
    assert toy_weight.shape == (2, 2)
    assert toy_bias.shape == (2,)
    assert toy_target.shape == (3, 2)
    pre_activation = toy_x @ toy_weight + toy_bias
    assert np.any(pre_activation < 0)

    expected_toy_prediction = np.array([
        [3.0, 0.0],
        [2.0, 3.0],
        [1.0, 0.0],
    ])
    toy_network = TinyNetwork(toy_weight, toy_bias)
    toy_result = forward_and_loss(
        toy_network,
        loss_fn,
        toy_x,
        toy_target,
    )
    np.testing.assert_allclose(
        toy_result["prediction"],
        expected_toy_prediction,
    )
    np.testing.assert_allclose(toy_result["loss"], 13.0 / 6.0)

    # Mini project: verify that it prints the required information and
    # returns finite numeric results without polluting test output.
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        project_result = run_miniproject()

    printed_output = output_buffer.getvalue()
    for label in (
        "x        =",
        "weight   =",
        "bias     =",
        "prediction =",
        "target   =",
        "loss     =",
        "x.shape",
        "weight.shape",
        "bias.shape",
        "prediction.shape",
        "target.shape",
    ):
        assert label in printed_output, f"Missing project output: {label}"

    assert project_result["prediction"].shape == (3, 2)
    assert np.all(np.isfinite(project_result["prediction"]))
    assert np.isfinite(project_result["loss"])

    print("All Day 07 tests passed!")


def main():
    """Run the tests, then print the end-to-end mini-project example."""
    run_tests()
    print("\n--- Day 07 mini project ---")
    run_miniproject()


if __name__ == "__main__":
    main()
