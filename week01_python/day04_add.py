
"""
Week 01 - Day 04 Additional Learning
Python Input and Output

Topics:
- Formatted string literals (f-strings)
- str() and repr()
- Numeric formatting
- str.format()
- File I/O with context managers
- JSON serialization and deserialization
"""

import json
import math
import tempfile
from pathlib import Path


# ============================================================
# 1. Formatted String Literals (f-strings)
# ============================================================

def f_string_example():
    """
    Demonstrate variable interpolation with f-strings.

    An f-string evaluates expressions inside curly braces.
    """

    year = 2016
    event = "国民投票"

    print(f"{year}年の{event}の結果")

    yes_votes = 42_572_654
    total_votes = 85_705_149

    percentage = yes_votes / total_votes

    # Format an integer with thousands separators.
    # Format a ratio as a percentage with two decimal places.
    print(f"{yes_votes:,} 賛成票  {percentage:.2%}")

    # The same formatting can be done with str.format().
    print("{:,} 賛成票  {:.2%}".format(yes_votes, percentage))


# ============================================================
# 2. str() and repr()
# ============================================================

def string_representation_example():
    """
    Compare human-readable and developer-oriented
    string representations.

    str():
        Returns a readable representation.

    repr():
        Returns a representation useful for debugging.
    """

    s = "Hello, world."

    print("str():", str(s))
    print("repr():", repr(s))

    # Convert a numeric value to a string.
    print("str(1/7):", str(1 / 7))

    x = 10 * 3.25
    y = 200 * 200

    message = (
        "The value of x is " + repr(x)
        + ", and y is " + repr(y) + "..."
    )

    print(message)

    # repr() makes escape sequences visible.
    hello = "hello, world\n"
    hello_repr = repr(hello)

    print("Original string:", hello)
    print("repr():", hello_repr)

    # repr() also works with tuples and other objects.
    values = (x, y, ("spam", "eggs"))

    print("Tuple representation:", repr(values))


# ============================================================
# 3. Numeric Formatting
# ============================================================

def numeric_formatting_example():
    """
    Demonstrate numeric formatting and alignment.

    :.3f  -> Three decimal places
    :10   -> Minimum field width of 10
    :10d  -> Integer with minimum width of 10
    :.2%  -> Percentage with two decimal places
    """

    # Display pi rounded to three decimal places.
    print(f"πの値はだいたい {math.pi:.3f} です。")

    table = {
        "Sjoerd": 4127,
        "Jack": 4098,
        "Dcab": 7678,
    }

    # Align names and phone numbers in columns.
    for name, phone in table.items():
        print(f"{name:10} ==> {phone:10d}")


# ============================================================
# 4. Conversion Flags in f-strings
# ============================================================

def conversion_flags_example():
    """
    Demonstrate !r and the debug format specifier (=).

    !r:
        Converts a value using repr().

    variable=:
        Displays both the expression and its value.
    """

    animals = "ウナギ"

    print(f"私のホバークラフトは {animals} でいっぱいです。")

    # !r applies repr() to the value.
    print(f"私のホバークラフトは {animals!r} でいっぱいです。")

    bugs = "roaches"
    count = 13
    area = "living room"

    # The = specifier is useful for quick debugging.
    print(f"Debugging {bugs=} {count=} {area=}")


# ============================================================
# 5. str.format()
# ============================================================

def format_method_example():
    """
    Demonstrate positional fields in str.format().

    {}:
        Uses arguments in order.

    {0}, {1}:
        Uses explicitly numbered arguments.
    """

    print(
        'We are the {} who say "{}!"'.format(
            "knights", "Ni"
        )
    )

    print("{0} and {1}".format("spam", "eggs"))

    # Reverse the argument order.
    print("{1} and {0}".format("spam", "eggs"))


# ============================================================
# 6. File I/O with Context Managers
# ============================================================

def file_io_example():
    """
    Demonstrate writing and reading a UTF-8 text file.

    with open(...) automatically closes the file,
    even if an exception occurs inside the block.

    A temporary directory prevents unwanted files
    from being left in the project repository.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "workfile"

        # Create a sample text file.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Hello, Python!\n")
            f.write("This is a sample file.\n")

        # Read the entire file.
        with open(file_path, "r", encoding="utf-8") as f:
            read_data = f.read()

        print("File content:")
        print(read_data)

        # The file is automatically closed after the block.
        print("Is file closed?", f.closed)

        assert f.closed is True


# ============================================================
# 7. JSON Serialization
# ============================================================

def json_example():
    """
    Demonstrate JSON serialization and deserialization.

    json.dumps():
        Convert a Python object to a JSON string.

    json.loads():
        Convert a JSON string to a Python object.
    """

    data = [1, "simple", "list"]

    # Serialize a Python list to a JSON string.
    json_string = json.dumps(data)

    print("Original object:", data)
    print("Serialized JSON:", json_string)

    # Deserialize the JSON string back to Python.
    restored_data = json.loads(json_string)

    print("Restored object:", restored_data)

    # Verify that the data is preserved.
    assert restored_data == data


# ============================================================
# Main
# ============================================================

def main():
    """Run all additional learning examples."""

    print("=== 1. f-strings ===")
    f_string_example()

    print("\n=== 2. str() and repr() ===")
    string_representation_example()

    print("\n=== 3. Numeric Formatting ===")
    numeric_formatting_example()

    print("\n=== 4. Conversion Flags ===")
    conversion_flags_example()

    print("\n=== 5. str.format() ===")
    format_method_example()

    print("\n=== 6. File I/O ===")
    file_io_example()

    print("\n=== 7. JSON ===")
    json_example()

    print("\nAll Day 04 additional examples completed!")


if __name__ == "__main__":
    main()
