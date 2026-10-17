
"""
Week 01 - Day 02 Additional Practice
Python functions and argument handling.

Topics:
- Positional and keyword arguments
- *args and **kwargs
- Positional-only parameters (/)
- Keyword-only parameters (*)
- Argument unpacking
- Lambda expressions
- Sorting with a key function
- Function annotations
"""


# ============================================================
# 1. *args and **kwargs
# ============================================================

def cheeseshop(kind, *arguments, **keywords):
    """
    Demonstrate variable-length positional and keyword arguments.

    *arguments collects extra positional arguments into a tuple.
    **keywords collects extra keyword arguments into a dictionary.
    """

    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)

    # Iterate over positional arguments.
    for arg in arguments:
        print(arg)

    print("-" * 40)

    # Iterate over keyword arguments.
    for key, value in keywords.items():
        print(key, ":", value)

    print("-" * 40)

    print("Positional arguments as a tuple:", arguments)
    print("Unpacked positional arguments:", *arguments)

    print("-" * 40)

    print("Keyword arguments as a dictionary:", keywords)

    # Unpacking a dictionary with * produces its keys.
    print("Dictionary keys:", *keywords)


# ============================================================
# 2. Function Parameter Types
# ============================================================

def standard_arg(arg):
    """
    Accept an argument by position or keyword.
    """
    print(arg)


def pos_only_arg(arg, /):
    """
    Accept an argument only by position.

    The / symbol marks positional-only parameters.
    """
    print(arg)


def kwd_only_arg(*, arg):
    """
    Accept an argument only by keyword.

    The * symbol marks keyword-only parameters.
    """
    print(arg)


def combined_example(pos_only, /, standard, *, kwd_only):
    """
    Combine positional-only, standard, and keyword-only parameters.
    """
    print(pos_only, standard, kwd_only)


# ============================================================
# 3. Positional-only Parameters and **kwargs
# ============================================================

def foo(name, /, **kwds):
    """
    Check whether 'name' exists in the extra keyword arguments.

    Because 'name' is positional-only, the keyword 'name'
    can also be collected separately in **kwds.
    """
    return "name" in kwds


# ============================================================
# 4. Variable-length Arguments and String Joining
# ============================================================

def concat(*args, sep="/"):
    """
    Join multiple strings using the specified separator.

    Args:
        *args: Strings to join.
        sep: Separator between strings.

    Returns:
        A single joined string.
    """
    return sep.join(args)


# ============================================================
# 5. Argument Unpacking with * and **
# ============================================================

def parrot(voltage, state="a stiff", action="voom"):
    """
    Demonstrate positional arguments, default arguments,
    and dictionary unpacking.
    """

    print("-- This parrot wouldn't", action, end=" ")
    print("if you put", voltage, "volts through it.", end=" ")
    print("E's", state, "!")


# ============================================================
# 6. Lambda Expressions
# ============================================================

def make_incrementor(n):
    """
    Return a function that adds n to its argument.

    The returned lambda captures the value of n
    from the enclosing function.
    """
    return lambda x: x + n


# ============================================================
# 7. Sorting with Lambda
# ============================================================

def sort_pairs_by_word(pairs):
    """
    Sort a list of (number, word) tuples by the word.

    The original list is modified in place.
    """

    pairs.sort(key=lambda pair: pair[1])

    return pairs


# ============================================================
# 8. Function Annotations
# ============================================================

def annotated_example(ham: str, eggs: str = "eggs") -> str:
    """
    Demonstrate parameter and return type annotations.

    Type annotations provide metadata but do not
    automatically enforce types at runtime.
    """

    print("Annotations:", annotated_example.__annotations__)
    print("Arguments:", ham, eggs)

    return ham + " and " + eggs


# ============================================================
# 9. Main Function
# ============================================================

def main():
    """Run all examples."""

    # --------------------------------------------------------
    # Example 1: *args and **kwargs
    # --------------------------------------------------------

    print("\n=== 1. Variable-length Arguments ===")

    cheeseshop(
        "Limburger",
        "It's very runny, sir.",
        "It's really very, VERY runny, sir.",
        shopkeeper="Michael Palin",
        client="John Cleese",
        sketch="Cheese Shop Sketch",
    )

    # --------------------------------------------------------
    # Example 2: Parameter Types
    # --------------------------------------------------------

    print("\n=== 2. Function Parameter Types ===")

    standard_arg(10)
    standard_arg(arg=10)

    pos_only_arg(20)

    kwd_only_arg(arg=30)

    combined_example(1, 2, kwd_only=3)
    combined_example(1, standard=2, kwd_only=3)

    # --------------------------------------------------------
    # Example 3: Positional-only and **kwargs
    # --------------------------------------------------------

    print("\n=== 3. Positional-only and **kwargs ===")

    result = foo(1, **{"name": 2})

    print("Does **kwds contain 'name'?", result)

    # --------------------------------------------------------
    # Example 4: String Joining
    # --------------------------------------------------------

    print("\n=== 4. String Joining ===")

    print(concat("earth", "mars", "venus"))

    print(concat("earth", "mars", "venus", sep="."))

    # --------------------------------------------------------
    # Example 5: Argument Unpacking
    # --------------------------------------------------------

    print("\n=== 5. Argument Unpacking ===")

    options = {
        "voltage": "four million",
        "state": "bleedin' demised",
        "action": "VOOM",
    }

    # ** unpacks dictionary entries as keyword arguments.
    parrot(**options)

    # * unpacks a sequence as positional arguments.
    range_args = [3, 6]

    print("Direct call:", list(range(3, 6)))
    print("Unpacked call:", list(range(*range_args)))

    # --------------------------------------------------------
    # Example 6: Lambda
    # --------------------------------------------------------

    print("\n=== 6. Lambda Expressions ===")

    increment_by_42 = make_incrementor(42)

    print(increment_by_42(0))
    print(increment_by_42(1))

    # --------------------------------------------------------
    # Example 7: Sorting with Lambda
    # --------------------------------------------------------

    print("\n=== 7. Sorting with Lambda ===")

    pairs = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
    ]

    print("Before sorting:", pairs)

    sorted_pairs = sort_pairs_by_word(pairs)

    print("After sorting:", sorted_pairs)

    # --------------------------------------------------------
    # Example 8: Function Annotations
    # --------------------------------------------------------

    print("\n=== 8. Function Annotations ===")

    result = annotated_example("spam")

    print("Return value:", result)


if __name__ == "__main__":
    main()
