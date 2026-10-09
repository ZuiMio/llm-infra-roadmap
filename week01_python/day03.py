
"""
Week 01 - Day 03
Python functions: arguments, scope, and mutability.

Topics:
- Positional and keyword arguments
- Default argument values
- Positional-only and keyword-only parameters
- *args and **kwargs
- Local and global scope
- Mutable versus immutable objects
- Closures and returning functions
"""


# ============================================================
# Exercise 1 - Positional and keyword arguments
# ============================================================

def calculate_tokens(batch_size, sequence_length):
    """
    Return the number of tokens in a batch.

    Examples:
        calculate_tokens(4, 128) -> 512
        calculate_tokens(sequence_length=128, batch_size=4) -> 512
    """
    return batch_size * sequence_length


# ============================================================
# Exercise 2 - Default argument values
# ============================================================

def format_experiment_name(model_name, version="v1"):
    """
    Return an experiment name formatted as 'model_name-version'.

    Examples:
        format_experiment_name("minigpt") -> "minigpt-v1"
        format_experiment_name("minigpt", "v2") -> "minigpt-v2"
    """
    return f"{model_name}-{version}"


# ============================================================
# Exercise 3 - Positional-only arguments
# ============================================================

def effective_batch_size(
    micro_batch,
    accumulation_steps,
    /,
    data_parallel_size=1,
):
    """
    Calculate the effective global batch size.

    The first two parameters must be passed positionally.

    Example:
        effective_batch_size(8, 4, data_parallel_size=2) -> 64
    """
    return micro_batch * accumulation_steps * data_parallel_size


# ============================================================
# Exercise 4 - Keyword-only arguments
# ============================================================

def build_optimizer_config(
    *,
    learning_rate=0.001,
    weight_decay=0.01,
):
    """
    Return a new dictionary containing optimizer settings.

    Both parameters must be specified by keyword
    when overriding their default values.
    """
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


# ============================================================
# Exercise 5 - *args
# ============================================================

def average_losses(*losses):
    """
    Calculate the arithmetic mean of loss values.

    Return None if no losses are provided.

    Examples:
        average_losses(1.0, 2.0, 3.0) -> 2.0
        average_losses() -> None
    """
    if not losses:
        return None

    return sum(losses) / len(losses)


# ============================================================
# Exercise 6 - **kwargs
# ============================================================

def prefix_metrics(prefix, **metrics):
    """
    Add a prefix to every metric name.

    Example:
        prefix_metrics("train", loss=0.8, accuracy=0.9)

        -> {
            "train/loss": 0.8,
            "train/accuracy": 0.9,
        }
    """
    return {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
    }


# ============================================================
# Exercise 7 - Combined parameter rules
# ============================================================

def summarize_step(step, /, loss, *, learning_rate):
    """
    Return training step information.

    step: positional-only
    loss: positional or keyword
    learning_rate: keyword-only
    """
    return {
        "step": step,
        "loss": loss,
        "learning_rate": learning_rate,
    }


# ============================================================
# Exercise 8 - Mutable default arguments
# ============================================================

def add_experiment(name, history=None):
    """
    Return a new list containing the existing experiments
    followed by the new experiment name.

    Never modify the caller's original list.

    Examples:
        add_experiment("run1") -> ["run1"]
        add_experiment("run2", ["run1"]) -> ["run1", "run2"]
    """
    if history is None:
        history = []

    # List concatenation creates a new list.
    return history + [name]


# ============================================================
# Exercise 9 - Scope and immutable values
# ============================================================

DEFAULT_LEARNING_RATE = 0.001


def choose_learning_rate(learning_rate=None):
    """
    Return the default learning rate when no value is given.

    Do not modify the module-level constant.
    """
    if learning_rate is None:
        return DEFAULT_LEARNING_RATE

    return learning_rate


# ============================================================
# Exercise 10 - Closures and functions as values
# ============================================================

def make_multiplier(factor):
    """
    Return a function that multiplies its input by factor.

    The returned function captures factor from
    the enclosing scope.

    Example:
        double = make_multiplier(2)
        double(5) -> 10
    """
    return lambda x: x * factor


# ============================================================
# Mini Task - Training Report
# ============================================================

def create_training_report(config, *losses, **metadata):
    """
    Create a training report.

    Args:
        config:
            Dictionary containing training configuration.

        *losses:
            Variable number of training loss values.

        **metadata:
            Additional information about the experiment.

    Returns:
        A dictionary containing:
        - config: shallow copy of the configuration
        - num_steps: number of recorded losses
        - average_loss: mean loss or None
        - metadata: additional experiment information

    The original config dictionary is not modified.
    """
    return {
        "config": config.copy(),
        "num_steps": len(losses),
        "average_loss": average_losses(*losses),
        "metadata": metadata,
    }


# ============================================================
# Tests
# ============================================================

def run_tests():
    """Run assertions for all exercises."""

    # Exercise 1
    assert calculate_tokens(4, 128) == 512
    assert calculate_tokens(
        sequence_length=128,
        batch_size=4,
    ) == 512

    # Exercise 2
    assert format_experiment_name("minigpt") == "minigpt-v1"
    assert format_experiment_name("minigpt", "v2") == "minigpt-v2"

    # Exercise 3
    assert effective_batch_size(8, 4) == 32
    assert effective_batch_size(
        8, 4, data_parallel_size=2
    ) == 64

    # Exercise 4
    assert build_optimizer_config() == {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    }

    assert build_optimizer_config(learning_rate=0.01)[
        "learning_rate"
    ] == 0.01

    # Exercise 5
    assert average_losses(1.0, 2.0, 3.0) == 2.0
    assert average_losses(5.0) == 5.0
    assert average_losses() is None

    # Exercise 6
    assert prefix_metrics(
        "train", loss=0.8, accuracy=0.9
    ) == {
        "train/loss": 0.8,
        "train/accuracy": 0.9,
    }

    assert prefix_metrics("train") == {}

    # Exercise 7
    assert summarize_step(
        10, loss=1.25, learning_rate=0.001
    ) == {
        "step": 10,
        "loss": 1.25,
        "learning_rate": 0.001,
    }

    # Exercise 8
    assert add_experiment("run1") == ["run1"]
    assert add_experiment("run2") == ["run2"]

    history = ["run1"]
    result = add_experiment("run2", history)

    assert result == ["run1", "run2"]
    assert history == ["run1"]
    assert result is not history

    # Exercise 9
    assert choose_learning_rate() == 0.001
    assert choose_learning_rate(0.01) == 0.01
    assert choose_learning_rate(0) == 0
    assert DEFAULT_LEARNING_RATE == 0.001

    # Exercise 10
    double = make_multiplier(2)
    triple = make_multiplier(3)

    assert double(5) == 10
    assert triple(5) == 15
    assert double(5) == 10

    # Mini task
    config = {"batch_size": 8}

    report = create_training_report(
        config,
        1.0,
        0.8,
        0.6,
        device="cpu",
        experiment="baseline",
    )

    assert report["config"] == {"batch_size": 8}
    assert report["num_steps"] == 3
    assert abs(report["average_loss"] - 0.8) < 1e-9
    assert report["metadata"] == {
        "device": "cpu",
        "experiment": "baseline",
    }

    # Verify that config was copied.
    assert report["config"] is not config

    report["config"]["batch_size"] = 16
    assert config["batch_size"] == 8

    # Test empty losses.
    empty_report = create_training_report(config)

    assert empty_report["num_steps"] == 0
    assert empty_report["average_loss"] is None
    assert empty_report["metadata"] == {}

    print("All Day 03 tests passed!")


if __name__ == "__main__":
    run_tests()
