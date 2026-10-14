"""
Python Tutorial - 10: Arrays, queues, sorting, heaps, and Decimal.

Topics:
- array.array for compact numeric storage
- collections.deque for efficient queue operations
- bisect for inserting into sorted lists
- heapq for priority queues
- decimal.Decimal for precise base-10 arithmetic
"""

from array import array
from bisect import insort
from collections import deque
from decimal import Decimal, getcontext, localcontext
from heapq import heapify, heappop, heappush


def array_example():
    """Store unsigned short integers in a compact array."""
    numbers = array("H", [4000, 10, 700, 22222])
    print("Sum:", sum(numbers))
    print("Slice:", numbers[1:3])

    assert sum(numbers) == 26932
    assert numbers[1:3] == array("H", [10, 700])


def deque_example():
    """Use deque as a first-in, first-out task queue."""
    tasks = deque(["task1", "task2", "task3"])
    tasks.append("task4")
    first = tasks.popleft()
    print("Handling:", first)
    print("Remaining tasks:", list(tasks))

    assert first == "task1"
    assert list(tasks) == ["task2", "task3", "task4"]

    # Breadth-first search uses the same enqueue/dequeue pattern.
    # It repeatedly removes the next node and adds its neighbors.


def bisect_example():
    """Insert an item while preserving ascending sorted order."""
    scores = [
        (100, "perl"),
        (200, "tcl"),
        (400, "lua"),
        (500, "python"),
    ]
    insort(scores, (300, "ruby"))
    print("Sorted scores:", scores)

    assert scores[2] == (300, "ruby")
    assert scores == sorted(scores)


def heap_example():
    """Use a min-heap to retrieve the smallest items efficiently."""
    data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    heapify(data)
    heappush(data, -5)

    three_smallest = [heappop(data) for _ in range(3)]
    print("Three smallest values:", three_smallest)
    assert three_smallest == [-5, 0, 1]


def decimal_example():
    """Compare decimal arithmetic with binary floating-point arithmetic."""
    decimal_product = round(Decimal("0.70") * Decimal("1.05"), 2)
    float_product = round(0.70 * 1.05, 2)

    print("Decimal multiplication:", decimal_product)
    print("Float multiplication:", float_product)

    print("Decimal remainder:", Decimal("1.00") % Decimal(".10"))
    print("Float remainder:", 1.00 % 0.10)

    exact_tenth_sum = sum([Decimal("0.1")] * 10)
    float_tenth_sum = sum([0.1] * 10)
    print("Decimal tenths equal 1:", exact_tenth_sum == Decimal("1.0"))
    print("Float tenths equal 1:", float_tenth_sum == 1.0)

    assert decimal_product == Decimal("0.74")
    assert exact_tenth_sum == Decimal("1.0")

    # Use a temporary context so precision changes do not leak globally.
    old_precision = getcontext().prec
    with localcontext() as context:
        context.prec = 36
        print("1 / 7 at precision 36:", Decimal(1) / Decimal(7))
    assert getcontext().prec == old_precision


def main():
    """Run all examples."""
    print("=== array.array ===")
    array_example()
    print("\n=== collections.deque ===")
    deque_example()
    print("\n=== bisect ===")
    bisect_example()
    print("\n=== heapq ===")
    heap_example()
    print("\n=== Decimal ===")
    decimal_example()
    print("All module 10 examples passed!")


if __name__ == "__main__":
    main()
