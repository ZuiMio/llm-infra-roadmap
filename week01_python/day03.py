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

Rules:
- Implement the functions independently.
- Do not change a function's signature unless the task asks you to.
- Write assertions for normal and edge cases.
- Use English for comments and docstrings.
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
    pass


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
    pass


# ============================================================
# Exercise 3 - Positional-only arguments
# ============================================================

def effective_batch_size(micro_batch, accumulation_steps, /, data_parallel_size=1):
    """
    Return micro_batch * accumulation_steps * data_parallel_size.

    The first two parameters must be passed positionally.

    Example:
        effective_batch_size(8, 4, data_parallel_size=2) -> 64
    """
    pass


# ============================================================
# Exercise 4 - Keyword-only arguments
# ============================================================

def build_optimizer_config(*, learning_rate=0.001, weight_decay=0.01):
    """
    Return a new dictionary containing:
        "learning_rate": learning_rate
        "weight_decay": weight_decay

    Both arguments must be supplied by keyword when overridden.
    """
    pass


# ============================================================
# Exercise 5 - *args
# ============================================================

def average_losses(*losses):
    """
    Return the arithmetic mean of any number of losses.

    Return None when no losses are provided.

    Examples:
        average_losses(1.0, 2.0, 3.0) -> 2.0
        average_losses() -> None
    """
    pass


# ============================================================
# Exercise 6 - **kwargs
# ============================================================

def prefix_metrics(prefix, **metrics):
    """
    Return a dictionary with prefix added to every metric name.

    Example:
        prefix_metrics("train", loss=0.8, accuracy=0.9)
        -> {"train/loss": 0.8, "train/accuracy": 0.9}
    """
    pass


# ============================================================
# Exercise 7 - Combined parameter rules
# ============================================================

def summarize_step(step, /, loss, *, learning_rate):
    """
    Return a dictionary containing step, loss, and learning_rate.

    'step' is positional-only.
    'loss' may be positional or keyword.
    'learning_rate' is keyword-only.

    Example:
        summarize_step(10, loss=1.25, learning_rate=0.001)
        -> {"step": 10, "loss": 1.25, "learning_rate": 0.001}
    """
    pass


# ============================================================
# Exercise 8 - Mutable default arguments
# ============================================================

def add_experiment(name, history=None):
    """
    Return a NEW list with name appended to history.

    When history is None, start from an empty list.
    Do not mutate a list supplied by the caller.

    Examples:
        add_experiment("run1") -> ["run1"]
        add_experiment("run2", ["run1"]) -> ["run1", "run2"]

    Do not use history=[] as a default parameter.
    """
    pass


# ============================================================
# Exercise 9 - Scope and immutable values
# ============================================================

DEFAULT_LEARNING_RATE = 0.001


def choose_learning_rate(learning_rate=None):
    """
    Return DEFAULT_LEARNING_RATE if learning_rate is None.
    Otherwise return learning_rate.

    Read the module-level constant without changing it.
    """
    pass


# ============================================================
# Exercise 10 - Closures and functions as values
# ============================================================

def make_multiplier(factor):
    """
    Return a function that multiplies its input by factor.

    Example:
        double = make_multiplier(2)
        double(5) -> 10

    The returned function must remember factor.
    """
    pass


# ============================================================
# Mini task - Training report
# ============================================================

def create_training_report(config, *losses, **metadata):
    """
    Build a small training report using function arguments.

    Return:
        {
            "config": a shallow copy of config,
            "num_steps": number of loss values,
            "average_loss": average of losses, or None if empty,
            "metadata": a dictionary of keyword arguments,
        }

    Do not modify the caller's config dictionary.

    Example:
        create_training_report(
            {"batch_size": 8},
            1.0, 0.8, 0.6,
            device="cpu",
            experiment="baseline",
        )
    """
    pass


if __name__ == "__main__":
    # Add your own assertions for every exercise.
    # Include tests for empty inputs and repeated function calls.
    pass
