
"""
Week 01 - Day 03 Additional Learning
Python Lists, Comprehensions, and Matrix Transposition

Topics:
- list.append()
- list.insert()
- list.extend()
- map() and lambda
- List comprehensions
- Nested list comprehensions
- zip() and argument unpacking
- Matrix transposition
"""


# ============================================================
# 1. List append() and insert()
# ============================================================

def append_insert_example():
    """
    Demonstrate append() and insert().

    append(x):
        Add x as a single element at the end of a list.

    insert(i, x):
        Insert x at index i.
    """

    a = [1, 2, 3]

    a.append(4)
    print("After append(4):", a)

    # The entire list [5, 6] is added as one element.
    a.append([5, 6])
    print("After append([5, 6]):", a)

    # Insert 0 at index 0.
    a.insert(0, 0)
    print("After insert(0, 0):", a)


# ============================================================
# 2. List extend()
# ============================================================

def extend_example():
    """
    Demonstrate extend().

    extend(iterable):
        Add each element from an iterable to the list.

    Difference:
        append() adds one object.
        extend() adds elements from an iterable.
    """

    b = [1, 2, 3]

    b.extend([4, 5])
    print("After extend([4, 5]):", b)

    # Strings are iterable.
    # Each character is added separately.
    b.extend("ab")
    print('After extend("ab"):', b)


# ============================================================
# 3. map() and List Comprehension
# ============================================================

def squares_example():
    """
    Generate the squares of integers from 0 to 9.

    Method 1:
        map() with lambda

    Method 2:
        List comprehension
    """

    squares_map = list(
        map(lambda x: x ** 2, range(10))
    )

    squares_comprehension = [
        x ** 2 for x in range(10)
    ]

    print("Using map():", squares_map)
    print("Using comprehension:", squares_comprehension)

    # Both methods should produce the same result.
    assert squares_map == squares_comprehension


# ============================================================
# 4. Nested List Comprehension
# ============================================================

def matrix_transpose_example():
    """
    Transpose a matrix using two different methods.

    Original matrix:
        3 rows x 4 columns

    Transposed matrix:
        4 rows x 3 columns
    """

    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]

    # Method 1: Nested list comprehension.
    # For each column, collect the corresponding
    # element from every row.
    transposed_comprehension = [
        [row[i] for row in matrix]
        for i in range(len(matrix[0]))
    ]

    # Method 2: zip() with argument unpacking.
    # zip(*matrix) groups elements column by column.
    # Each output row is a tuple.
    transposed_zip = list(zip(*matrix))

    # Convert each tuple to a list.
    transposed_list = [
        list(row) for row in transposed_zip
    ]

    print("Original matrix:", matrix)
    print("Nested comprehension:", transposed_comprehension)
    print("Using zip():", transposed_zip)

    # Verify that both approaches produce
    # the same matrix values.
    assert transposed_comprehension == transposed_list


# ============================================================
# Main
# ============================================================

def main():
    """Run all examples."""

    print("=== 1. append() and insert() ===")
    append_insert_example()

    print("\n=== 2. extend() ===")
    extend_example()

    print("\n=== 3. map() and comprehension ===")
    squares_example()

    print("\n=== 4. Matrix transposition ===")
    matrix_transpose_example()


if __name__ == "__main__":
    main()
