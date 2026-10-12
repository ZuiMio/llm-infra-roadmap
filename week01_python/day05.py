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
    result = int(value)
    if result <= 0:
        raise ValueError("Input must be a positive integer")
    return result


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
    try:
        return a / b
    except ZeroDivisionError:
        return None


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
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


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
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f)


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
    if n < 0:
        raise ValueError("n must be non-negative")

    iterator = iter(iterable)
    result = []

    for _ in range(n):
        try:
            result.append(next(iterator))
        except StopIteration:
            break

    return result


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
        self.start = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        current = self.start
        self.start -= 1
        return current


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
    if n < 0:
        raise ValueError("n must be non-negative")

    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for token in line.split():
                yield token


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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"Malformed line: {line}")
            step_str, loss_str = parts
            try:
                step = int(step_str)
                loss = float(loss_str)
            except ValueError as e:
                raise ValueError(f"Invalid data types in line: {line}") from e
            yield {"step": step, "loss": loss}


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
    num_steps = 0
    total_loss = 0.0
    last_step = None

    for metrics in iter_training_metrics(path):
        num_steps += 1
        total_loss += metrics["loss"]
        last_step = metrics["step"]

    average_loss = total_loss / num_steps if num_steps > 0 else None

    return {
        "num_steps": num_steps,
        "average_loss": average_loss,
        "last_step": last_step,
    }


# ============================================================
# Tests
# ============================================================

def run_tests():
    """Check results, error handling, generators, and file I/O."""

    def expect_raises(error_type, callback, *args):
        """Assert that a call raises a specific exception."""
        try:
            callback(*args)
        except error_type:
            return
        raise AssertionError(f"Expected {error_type.__name__}")

    # Exercise 1: invalid strings and non-positive numbers.
    assert parse_positive_int("12") == 12
    assert parse_positive_int(3) == 3
    for value in ("abc", "0", -2):
        expect_raises(ValueError, parse_positive_int, value)

    # Exercise 2: catch only division-by-zero errors.
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(5, 0) is None
    expect_raises(TypeError, safe_divide, "2", "1")

    # Exercise 6: consume any iterable without overrunning it.
    assert take_first([10, 20, 30], 2) == [10, 20]
    assert take_first(iter([10]), 3) == [10]
    assert take_first([10], 0) == []
    expect_raises(ValueError, take_first, [1], -1)

    # Exercise 7: an iterator stores its own progress.
    countdown = Countdown(3)
    assert iter(countdown) is countdown
    assert list(countdown) == [3, 2, 1]
    assert list(countdown) == []
    expect_raises(StopIteration, next, countdown)
    assert list(Countdown(0)) == []

    # Exercise 8: generator execution is lazy.
    assert list(fibonacci_generator(6)) == [0, 1, 1, 2, 3, 5]
    assert list(fibonacci_generator(0)) == []
    expect_raises(ValueError, list, fibonacci_generator(-1))

    # Exercise 10: retain the final partial batch.
    assert list(batch_items(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batch_items((i for i in range(4)), 3)) == [
        [0, 1, 2], [3]
    ]
    assert list(batch_items([], 3)) == []
    expect_raises(ValueError, list, batch_items([1, 2], 0))

    # Test file operations without leaving data in the repository.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.txt"

        save_lines(path, ["hello", "  world  ", "", "AI"])
        assert path.read_text(encoding="utf-8") == (
            "hello\n  world  \n\nAI\n"
        )
        assert load_nonempty_lines(path) == ["hello", "world", "AI"]
        assert count_file_lines(path) == 4
        assert list(token_stream(path)) == ["hello", "world", "AI"]

        save_lines(path, [])
        assert count_file_lines(path) == 0
        assert load_nonempty_lines(path) == []
        assert list(token_stream(path)) == []

        metrics_path = Path(directory) / "training.log"
        metrics_path.write_text("1,1.2\n\n2,1.0\n3,0.8\n", encoding="utf-8")

        assert list(iter_training_metrics(metrics_path)) == [
            {"step": 1, "loss": 1.2},
            {"step": 2, "loss": 1.0},
            {"step": 3, "loss": 0.8},
        ]

        summary = summarize_training_log(metrics_path)
        assert summary["num_steps"] == 3
        assert abs(summary["average_loss"] - 1.0) < 1e-12
        assert summary["last_step"] == 3

        metrics_path.write_text("", encoding="utf-8")
        assert summarize_training_log(metrics_path) == {
            "num_steps": 0, "average_loss": None, "last_step": None
        }

        for malformed in ("bad-row\n", "1,not-a-number\n"):
            metrics_path.write_text(malformed, encoding="utf-8")
            expect_raises(ValueError, list, iter_training_metrics(metrics_path))

    print("All Day 05 tests passed!")


if __name__ == "__main__":
    run_tests()
