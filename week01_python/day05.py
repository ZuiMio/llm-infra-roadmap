"""
Week 01 - Day 05
Files, exceptions, iterators, and generators.

Topics:
- try / except / else / finally
- raise and common exceptions
- with open(...) and UTF-8 text files
- File paths and context managers
- iterable, iterator, iter(), and next()
- __iter__(), __next__(), and StopIteration
- yield and generator functions
- Streaming data and mini-batches

Rules:
- Implement all exercises independently.
- Do not change the provided function or class signatures.
- Use English comments and docstrings.
- Do not use NumPy or PyTorch.
- Avoid reading entire files when streaming is required.
- Add assertions for normal and edge cases.
"""

from pathlib import Path
import tempfile


# ============================================================
# Exercise 1 - Raising exceptions
# ============================================================

def parse_positive_int(value):
    """
    Convert value to an integer and require it to be positive.

    Examples:
        parse_positive_int("12") -> 12
        parse_positive_int(3) -> 3

    Requirements:
    - Raise ValueError if the integer is zero or negative.
    - Let ValueError propagate for invalid integer strings.
    """
    pass


# ============================================================
# Exercise 2 - Handling exceptions
# ============================================================

def safe_divide(a, b):
    """
    Return a / b, or None when division by zero occurs.

    Examples:
        safe_divide(10, 2) -> 5.0
        safe_divide(5, 0) -> None

    Requirements:
    - Use try / except ZeroDivisionError.
    - Do not use a bare except.
    - Do not suppress unrelated exceptions.
    """
    pass


# ============================================================
# Exercise 3 - Writing text files with a context manager
# ============================================================

def save_lines(path, lines):
    """
    Write a sequence of strings to a UTF-8 text file.

    Requirements:
    - Use with open(..., "w", encoding="utf-8").
    - Write each input string on its own line, ending in "\n".
    - Assume input strings do not contain newline characters.
    - Overwrite the file if it already exists.

    Example:
        save_lines(path, ["hello", "world"])
        # File content: "hello\nworld\n"
    """
    pass


# ============================================================
# Exercise 4 - Reading text files
# ============================================================

def load_nonempty_lines(path):
    """
    Return nonempty lines from a UTF-8 text file.

    Requirements:
    - Remove leading and trailing whitespace from each line.
    - Ignore lines that become empty after stripping.
    - Use with open(..., encoding="utf-8").
    - Return a list of strings.

    Example file:
        hello

          world

    Result:
        ["hello", "world"]
    """
    pass


# ============================================================
# Exercise 5 - Reading a file incrementally
# ============================================================

def count_file_lines(path):
    """
    Count the total number of lines in a text file.

    Requirements:
    - Count blank lines as lines.
    - Iterate over the file object instead of readlines().
    - Do not load the whole file into memory.
    - Return 0 for an empty file.
    """
    pass


# ============================================================
# Exercise 6 - iter(), next(), and StopIteration
# ============================================================

def take_first(iterable, n):
    """
    Return up to n items from an iterable as a list.

    Requirements:
    - Explicitly use iter() and next().
    - Stop gracefully if the iterable is exhausted.
    - Raise ValueError if n is negative.
    - Do not use slicing or itertools.

    Examples:
        take_first([10, 20, 30], 2) -> [10, 20]
        take_first([10], 3) -> [10]
        take_first([10], 0) -> []
    """
    pass


# ============================================================
# Exercise 7 - A custom iterator
# ============================================================

class Countdown:
    """
    Create an iterator producing start, start-1, ..., 1.

    Requirements:
    - Store iteration state on the instance.
    - Implement __iter__() and __next__().
    - Raise StopIteration when the sequence is exhausted.

    Examples:
        list(Countdown(4)) -> [4, 3, 2, 1]
        list(Countdown(0)) -> []

    Note:
        This is a one-shot iterator, not a restartable container.
    """

    def __init__(self, start):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


# ============================================================
# Exercise 8 - Generators and yield
# ============================================================

def fibonacci_generator(n):
    """
    Yield the first n Fibonacci numbers, starting with 0 and 1.

    Requirements:
    - Use yield rather than building and returning a list.
    - Raise ValueError for negative n.

    Examples:
        list(fibonacci_generator(6)) -> [0, 1, 1, 2, 3, 5]
        list(fibonacci_generator(0)) -> []
    """
    pass


# ============================================================
# Exercise 9 - Streaming tokens from a file
# ============================================================

def token_stream(path):
    """
    Yield whitespace-separated tokens from a UTF-8 text file.

    Requirements:
    - Read one line at a time with a context manager.
    - Split each line on whitespace.
    - Yield tokens in their original order.
    - Do not load the entire file into memory.

    Example file:
        hello AI
        learn infra

    Output:
        "hello", "AI", "learn", "infra"
    """
    pass


# ============================================================
# Exercise 10 - Streaming mini-batches
# ============================================================

def batch_items(iterable, batch_size):
    """
    Yield lists of at most batch_size elements.

    Requirements:
    - Support any iterable, including generators.
    - Preserve item order.
    - Yield the final smaller batch if necessary.
    - Never yield an empty batch.
    - Raise ValueError if batch_size <= 0.

    Examples:
        list(batch_items(range(5), 2))
        -> [[0, 1], [2, 3], [4]]

        list(batch_items([], 3))
        -> []
    """
    pass


# ============================================================
# Mini task - Streaming training log analysis
# ============================================================

def iter_training_metrics(path):
    """
    Yield dictionaries from a simple training log file.

    File format (no header):
        1,1.2
        2,1.0
        3,0.8

    Each nonempty line contains:
        step,loss

    Example yielded value:
        {"step": 1, "loss": 1.2}

    Requirements:
    - Skip blank lines.
    - Read lazily, line by line.
    - Convert step to int and loss to float.
    - Raise ValueError for malformed nonempty lines.
    - Do not read the whole file into memory.
    """
    pass


def summarize_training_log(path):
    """
    Summarize the metrics produced by iter_training_metrics(path).

    Return:
        {
            "num_steps": number of nonempty metric records,
            "average_loss": average loss, or None for no records,
            "last_step": the final recorded step, or None,
        }

    Requirements:
    - Iterate over iter_training_metrics(path).
    - Use constant extra memory, regardless of file size.
    - Do not build a list of all records.

    Example:
        summarize_training_log(path) -> {
            "num_steps": 3,
            "average_loss": 1.0,
            "last_step": 3,
        }
    """
    pass


# ============================================================
# Tests
# ============================================================

def run_tests():
    """
    Write assertions for all exercises.

    Suggested tests:
    - Valid and invalid integer inputs.
    - Division by zero and ordinary division.
    - Write/read round trips, empty files, and blank lines.
    - Exhausted iterators and repeated next() calls.
    - Generator outputs and zero/negative n.
    - Batches from lists and generators.
    - Valid, empty, and malformed training logs.

    Tip:
        Use tempfile.TemporaryDirectory() and Path() for file tests.
        Keep the repository free of generated test data.
    """
    pass


if __name__ == "__main__":
    run_tests()
