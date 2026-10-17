"""
Week 01 - Day 05 Additional Learning
Python exceptions, exception chaining, and exception groups.

Topics:
- Exception inheritance
- Exception arguments
- Exception chaining
- try / except / else / finally
- Context managers
- ExceptionGroup and except*

Requires Python 3.11 or newer.
"""

from pathlib import Path
from tempfile import TemporaryDirectory


# ============================================================
# 1. Exception Inheritance
# ============================================================

class B(Exception):
    pass


class C(B):
    pass


class D(C):
    pass


def exception_order_example():
    """Demonstrate how exception handler order matters."""

    print("Specific handlers first:")

    for exc_type in (B, C, D):
        try:
            raise exc_type()
        except D:
            print("D")
        except C:
            print("C")
        except B:
            print("B")

    print("Base-class handler first:")

    for exc_type in (B, C, D):
        try:
            raise exc_type()
        except B:
            print("B")
        except C:
            print("C")
        except D:
            print("D")


# ============================================================
# 2. Exception Arguments
# ============================================================

def exception_arguments_example():
    """Inspect exception type, arguments, and message."""

    try:
        raise Exception("spam", "eggs")

    except Exception as exc:
        print("Type:", type(exc).__name__)
        print("Arguments:", exc.args)
        print("Message:", str(exc))

        # Unpack the exception arguments.
        x, y = exc.args

        print("x =", x)
        print("y =", y)


# ============================================================
# 3. Exception Chaining
# ============================================================

def connect_to_database():
    """Simulate a connection failure."""

    raise ConnectionError("Connection refused")


def exception_chaining_example():
    """
    Demonstrate explicit exception chaining.

    The 'from' keyword preserves the original exception
    as the cause of the new exception.
    """

    try:
        try:
            connect_to_database()

        except ConnectionError as exc:
            raise RuntimeError(
                "Failed to open database"
            ) from exc

    except RuntimeError as exc:
        print("Raised:", exc)
        assert isinstance(exc.__cause__, ConnectionError)
        print("Original cause:", repr(exc.__cause__))


# ============================================================
# 4. finally and KeyboardInterrupt
# ============================================================

def finally_example():
    """
    Demonstrate that finally executes even when
    an exception is raised.
    """

    try:
        try:
            raise KeyboardInterrupt()

        finally:
            print("Goodbye, world!")

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught for this example")


# ============================================================
# 5. try / except / else / finally
# ============================================================

def divide(x, y):
    """
    Demonstrate all four exception-handling clauses.

    try:
        Execute code that may raise an exception.

    except:
        Handle a specific exception.

    else:
        Execute when no exception occurs in try.

    finally:
        Execute regardless of how the block exits.
    """

    try:
        result = x / y

    except ZeroDivisionError:
        print("Division by zero!")
        return None

    else:
        print("Result is", result)
        return result

    finally:
        print("Executing finally clause")


def division_example():
    """Test division with different input values."""

    assert divide(2, 1) == 2.0
    assert divide(2, 0) is None

    # String division raises TypeError,
    # which is not handled inside divide().
    try:
        divide("2", "1")

    except TypeError as exc:
        print("Caught TypeError:", exc)


# ============================================================
# 6. File I/O with Context Managers
# ============================================================

def file_example():
    """
    Demonstrate file operations with a context manager.

    A temporary directory avoids leaving sample files
    in the project repository.
    """

    with TemporaryDirectory() as directory:
        path = Path(directory) / "myfile.txt"

        # Create a temporary text file.
        path.write_text(
            "hello\nworld\n",
            encoding="utf-8",
        )

        # Read the file line by line.
        with open(path, encoding="utf-8") as file:
            for line in file:
                print(line, end="")

        # The context manager automatically closes the file.
        assert file.closed


# ============================================================
# 7. ExceptionGroup
# ============================================================

def raise_exception_group():
    """Raise multiple exceptions as a single group."""

    errors = [
        OSError("error 1"),
        SystemError("error 2"),
    ]

    raise ExceptionGroup("there were problems", errors)


def exception_group_example():
    """
    Compare handling a whole ExceptionGroup
    with handling its individual exception types.
    """

    # Method 1: Catch the entire ExceptionGroup.
    try:
        raise_exception_group()

    except Exception as exc:
        assert isinstance(exc, ExceptionGroup)
        print("Caught whole group:", type(exc).__name__)

        print(
            "Number of contained errors:",
            len(exc.exceptions),
        )

    # Method 2: except* splits the group by matching exception type.
    # Python 3.11 or newer is required.
    try:
        raise_exception_group()

    except* OSError as group:
        print(
            "OSError subgroup:",
            [str(exc) for exc in group.exceptions],
        )

    except* SystemError as group:
        print(
            "SystemError subgroup:",
            [str(exc) for exc in group.exceptions],
        )


# ============================================================
# Main
# ============================================================

def main():
    """Run all exception-handling examples."""

    print("\n=== 1. Exception Inheritance ===")
    exception_order_example()

    print("\n=== 2. Exception Arguments ===")
    exception_arguments_example()

    print("\n=== 3. Exception Chaining ===")
    exception_chaining_example()

    print("\n=== 4. finally ===")
    finally_example()

    print("\n=== 5. Division ===")
    division_example()

    print("\n=== 6. File I/O ===")
    file_example()

    print("\n=== 7. ExceptionGroup ===")
    exception_group_example()


if __name__ == "__main__":
    main()
