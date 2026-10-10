
"""
Week 01 - Day 04
Python classes and object-oriented programming.

Topics:
- class and object
- __init__ and self
- Instance attributes and methods
- Class attributes
- Inheritance and method overriding
- super()
- __call__
- Object composition
"""


# ============================================================
# Exercise 1 - __init__, self, and instance methods
# ============================================================

class Student:
    """Represent a student with a name and score."""

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def passed(self):
        return self.score >= 60

    def describe(self):
        return f"{self.name}: {self.score}"


# ============================================================
# Exercise 2 - Independent object state
# ============================================================

class Counter:
    """Maintain an independent counter."""

    def __init__(self, start=0):
        self.value = start

    def increment(self, amount=1):
        self.value += amount
        return self.value

    def reset(self):
        self.value = 0


# ============================================================
# Exercise 3 - Configuration objects
# ============================================================

class ModelConfig:
    """Store configuration values for a model."""

    def __init__(self, model_name, batch_size=1, seq_len=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len

    def tokens_per_batch(self):
        return self.batch_size * self.seq_len

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
        }


# ============================================================
# Exercise 4 - Mutable instance attributes
# ============================================================

class LossTracker:
    """Track loss values for a training experiment."""

    def __init__(self):
        # Each instance creates its own list.
        self.losses = []

    def add(self, loss):
        self.losses.append(loss)

    def count(self):
        return len(self.losses)

    def average(self):
        if not self.losses:
            return None

        return sum(self.losses) / len(self.losses)


# ============================================================
# Exercise 5 - Class attributes versus instance attributes
# ============================================================

class TrainingRun:
    """Track the progress of a training run."""

    # This attribute is shared by instances unless overridden.
    framework = "pytorch"

    def __init__(self, name):
        self.name = name
        self.step = 0

    def advance_step(self):
        self.step += 1
        return self.step


# ============================================================
# Exercise 6 - Inheritance, overriding, and super()
# ============================================================

class BaseTransform:
    """Implement an identity transformation."""

    def forward(self, x):
        return x


class ScaleTransform(BaseTransform):
    """Scale the output of the base transformation."""

    def __init__(self, factor):
        self.factor = factor

    def forward(self, x):
        # Call the parent class implementation.
        base_output = super().forward(x)
        return base_output * self.factor


# ============================================================
# Exercise 7 - __call__ (activation function)
# ============================================================

class ReLU:
    """Implement scalar ReLU activation."""

    def __call__(self, x):
        return max(0, x)


# ============================================================
# Exercise 8 - __call__ (affine layer)
# ============================================================

class Affine:
    """Implement a scalar affine transformation."""

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return self.weight * x + self.bias

    def set_parameters(self, weight, bias):
        self.weight = weight
        self.bias = bias


# ============================================================
# Mini Task - Compose a tiny neural network
# ============================================================

class TinyScalarNetwork:
    """
    Implement a tiny network using object composition.

    Pipeline:
        Input -> Affine -> ReLU -> Affine -> Output
    """

    def __init__(self, w1, b1, w2, b2):
        self.layer1 = Affine(w1, b1)
        self.activation = ReLU()
        self.layer2 = Affine(w2, b2)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x


# ============================================================
# Tests
# ============================================================

def run_tests():
    """Verify the behavior of all exercises."""

    # Exercise 1
    alice = Student("Alice", 85)
    bob = Student("Bob", 59)

    assert alice.passed() is True
    assert bob.passed() is False
    assert alice.describe() == "Alice: 85"

    # Exercise 2
    counter_a = Counter()
    counter_b = Counter(start=10)

    assert counter_a.increment() == 1
    assert counter_b.increment(3) == 13

    assert counter_a.value == 1
    assert counter_b.value == 13

    counter_a.reset()
    assert counter_a.value == 0
    assert counter_b.value == 13

    # Exercise 3
    config = ModelConfig(
        "minigpt",
        batch_size=4,
        seq_len=128,
    )

    assert config.tokens_per_batch() == 512

    config_dict = config.to_dict()

    assert config_dict == {
        "model_name": "minigpt",
        "batch_size": 4,
        "seq_len": 128,
    }

    # Verify that a new dictionary is returned.
    config_dict["batch_size"] = 16
    assert config.batch_size == 4

    # Exercise 4
    tracker_a = LossTracker()
    tracker_b = LossTracker()

    assert tracker_a.average() is None
    assert tracker_a.count() == 0

    tracker_a.add(1.0)
    tracker_a.add(0.5)

    assert tracker_a.count() == 2
    assert tracker_a.average() == 0.75

    # Verify that instances do not share their lists.
    assert tracker_b.count() == 0
    assert tracker_a.losses is not tracker_b.losses

    # Exercise 5
    run_a = TrainingRun("baseline")
    run_b = TrainingRun("experiment")

    assert TrainingRun.framework == "pytorch"
    assert run_a.framework == "pytorch"

    assert run_a.advance_step() == 1
    assert run_a.advance_step() == 2
    assert run_b.step == 0

    # Exercise 6
    base = BaseTransform()
    scale = ScaleTransform(3)

    assert base.forward(4) == 4
    assert scale.forward(4) == 12

    # Exercise 7
    relu = ReLU()

    assert relu(-2) == 0
    assert relu(0) == 0
    assert relu(3) == 3

    # Exercise 8
    layer = Affine(weight=2, bias=3)

    assert layer(5) == 13

    layer.set_parameters(weight=3, bias=1)

    assert layer(5) == 16

    # Mini Task
    net = TinyScalarNetwork(
        w1=2,
        b1=-3,
        w2=4,
        b2=1,
    )

    assert net(1) == 1
    assert net(3) == 13

    print("All Day 04 tests passed!")


if __name__ == "__main__":
    run_tests()
