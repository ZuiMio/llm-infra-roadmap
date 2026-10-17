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
        "prediction":prediction,
        "loss":loss
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
    # 1. 取 toy 数据
    x, weight, bias, target = make_toy_example()

    # 2. 构造模型和损失函数
    model = TinyNetwork(weight, bias)
    loss_fn = MSELoss()

    # 3. 前向 + 计算损失
    result = forward_and_loss(model=model, loss_fn=loss_fn, x=x, target=target)

    # 4. 打印数值
    print("x        =", x)
    print("weight   =", weight)
    print("bias     =", bias)
    print("prediction =", result["prediction"])
    print("target   =", target)
    print("loss     =", result["loss"])

    # 5. 打印形状
    print("x.shape        =", x.shape)
    print("weight.shape   =", weight.shape)
    print("bias.shape     =", bias.shape)
    print("prediction.shape =", result["prediction"].shape)
    print("target.shape   =", target.shape)

    # 6. 返回结果字典
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
# - Why is the Linear bias added to every row?
# - Why does ReLU not change the shape?
# - What does the mean in MSE average over?
# - Why can silent broadcasting be dangerous for loss functions?


# ============================================================
# Tests
# ============================================================

def run_tests():
    """
    Add assertions for all exercises and the mini project.

    Suggested checks:
    - Linear gives the expected output for known small matrices.
    - Bias is applied to each sample in a batch.
    - ReLU handles negative, zero, and positive values.
    - ReLU preserves shape and does not modify its input.
    - MSELoss returns the correct numeric value.
    - MSELoss rejects mismatched shapes.
    - Two different Linear objects keep independent parameters.
    - TinyNetwork returns the correct shape and values.
    - forward_and_loss returns both prediction and loss.
    - make_toy_example() returns all required shapes.
    - run_miniproject() works and returns finite numeric results.

    Hints:
    - Use np.testing.assert_array_equal for exact integer arrays.
    - Use np.testing.assert_allclose for floating-point results.
    - Use np.isfinite() to check numerical results.
    """
    pass


if __name__ == "__main__":
    # Complete run_tests() before considering Day 07 finished.
    run_tests()
