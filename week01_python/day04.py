"""
Week 01 - Day 04
Python classes and object-oriented programming.

Topics:
- class and object
- __init__ and self
- Instance attributes and instance methods
- Class attributes versus instance attributes
- Encapsulation through methods
- Inheritance and method overriding
- super()
- __call__ and callable objects
- Composition of simple layers

Rules:
- Implement each class independently.
- Keep the given class and method names.
- Do not use NumPy or PyTorch today.
- Use English comments and docstrings.
- Add assertions for normal cases, edge cases, and independent instances.
"""


# ============================================================
# Exercise 1 - __init__, self, and instance methods
# ============================================================

class Student:
    """
    Store a student's name and score.

    Student("Alice", 85).passed() -> True
    Student("Bob", 59).passed() -> False

    Requirements:
    - Store name and score as instance attributes.
    - passed() returns True if score >= 60.
    - describe() returns a string formatted as "Alice: 85".
    """

    def __init__(self, name, score):
        pass

    def passed(self):
        pass

    def describe(self):
        pass


# ============================================================
# Exercise 2 - Independent object state
# ============================================================

class Counter:
    """
    A counter that stores its own value.

    Example:
        a = Counter()
        b = Counter(start=10)
        a.increment()
        b.increment(3)
        a.value -> 1
        b.value -> 13

    Requirements:
    - Store the current value in an instance attribute.
    - increment(amount=1) increases the value and returns it.
    - reset() sets the value to zero.
    - Two Counter instances must not share state.
    """

    def __init__(self, start=0):
        pass

    def increment(self, amount=1):
        pass

    def reset(self):
        pass


# ============================================================
# Exercise 3 - Configuration objects
# ============================================================

class ModelConfig:
    """
    Store a tiny model's configuration.

    Example:
        config = ModelConfig("minigpt", batch_size=4, seq_len=128)
        config.tokens_per_batch() -> 512
        config.to_dict() -> {
            "model_name": "minigpt",
            "batch_size": 4,
            "seq_len": 128,
        }

    Requirements:
    - Keep all values as instance attributes.
    - to_dict() returns a NEW dictionary.
    """

    def __init__(self, model_name, batch_size=1, seq_len=32):
        pass

    def tokens_per_batch(self):
        pass

    def to_dict(self):
        pass


# ============================================================
# Exercise 4 - Mutable instance attributes
# ============================================================

class LossTracker:
    """
    Record training losses for one experiment.

    Example:
        tracker = LossTracker()
        tracker.add(1.0)
        tracker.add(0.5)
        tracker.count() -> 2
        tracker.average() -> 0.75

    Requirements:
    - Each instance owns its own list of losses.
    - average() returns None when no losses are recorded.
    - Do not use a shared class-level list.
    """

    def __init__(self):
        pass

    def add(self, loss):
        pass

    def count(self):
        pass

    def average(self):
        pass


# ============================================================
# Exercise 5 - Class attributes versus instance attributes
# ============================================================

class TrainingRun:
    """
    Track a training run.

    Requirements:
    - The shared class attribute framework must equal "pytorch".
    - name and step are instance attributes.
    - step starts at zero.
    - advance_step() adds one and returns the new step.

    Example:
        run = TrainingRun("baseline")
        run.framework -> "pytorch"
        run.advance_step() -> 1
    """

    framework = "pytorch"

    def __init__(self, name):
        pass

    def advance_step(self):
        pass


# ============================================================
# Exercise 6 - Inheritance, overriding, and super()
# ============================================================

class BaseTransform:
    """
    A base class with an identity transform.

    forward(x) returns x without modification.
    """

    def forward(self, x):
        pass


class ScaleTransform(BaseTransform):
    """
    Extend BaseTransform with a scaling factor.

    Requirements:
    - Store factor in __init__.
    - Override forward(x).
    - Call super().forward(x) inside the overridden method.
    - Return the base result multiplied by factor.

    Example:
        ScaleTransform(3).forward(4) -> 12
    """

    def __init__(self, factor):
        pass

    def forward(self, x):
        pass


# ============================================================
# Exercise 7 - __call__ (activation function)
# ============================================================

class ReLU:
    """
    Implement scalar ReLU as a callable object.

    ReLU(x) = max(0, x)

    Example:
        relu = ReLU()
        relu(-2) -> 0
        relu(3) -> 3

    Do not use NumPy or PyTorch.
    """

    def __call__(self, x):
        pass


# ============================================================
# Exercise 8 - __call__ (affine layer)
# ============================================================

class Affine:
    """
    Implement a scalar affine transformation.

    y = weight * x + bias

    Example:
        layer = Affine(weight=2, bias=3)
        layer(5) -> 13

    Requirements:
    - Store weight and bias as instance attributes.
    - Implement __call__(x).
    - Implement set_parameters(weight, bias).
    """

    def __init__(self, weight, bias):
        pass

    def __call__(self, x):
        pass

    def set_parameters(self, weight, bias):
        pass


# ============================================================
# Mini task - Compose a tiny neural network
# ============================================================

class TinyScalarNetwork:
    """
    Compose two Affine layers and a ReLU activation.

    Forward pipeline:
        x -> Affine(w1, b1) -> ReLU -> Affine(w2, b2) -> output

    Example:
        net = TinyScalarNetwork(
            w1=2, b1=-3,
            w2=4, b2=1,
        )
        net(1) -> 1
        net(3) -> 13

    Requirements:
    - Create layer1, activation, and layer2 in __init__.
    - Store them as instance attributes.
    - Implement __call__(x) using those objects.
    - Do not repeat the affine or ReLU formulas here.
    """

    def __init__(self, w1, b1, w2, b2):
        pass

    def __call__(self, x):
        pass


# ============================================================
# Tests
# ============================================================

def run_tests():
    """
    Write assertions for every exercise.

    Suggested checks:
    - The two Student examples.
    - Independent state for two Counters.
    - ModelConfig tokens_per_batch() and to_dict().
    - Empty and non-empty LossTracker.
    - Separate TrainingRun.step values.
    - ScaleTransform.forward() and its use of super().
    - ReLU with negative, zero, and positive inputs.
    - Affine before and after set_parameters().
    - TinyScalarNetwork with at least two inputs.
    """
    pass


if __name__ == "__main__":
    run_tests()
