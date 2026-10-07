"""
Week 01 - Day 01
Python basics: control flow and functions.

Rule:
- Try each exercise yourself.
- Do not look up a complete solution immediately.
- Keep functions small and readable.
"""

# Exercise 1
# Return True if n is even, otherwise False.
def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False
    


# Exercise 2
# Return the sum of integers from 1 to n.
# Do not use sum().
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total



# Exercise 3
# Return the largest number in values.
# Do not use max().
def find_max(values):
    if not values:
        return None  # Return None for empty list
    max_value = values[0]
    for value in values:
        if value > max_value:
            max_value = value
    return max_value


# Exercise 4
# Return the arithmetic mean of values.
def average(values):
    if not values:
        return None  # Return None for empty list
    total = sum(values)
    return total / len(values)


# Exercise 5
# Return a new list containing only values greater than threshold.
def filter_greater_than(values, threshold):
    return [value for value in values if value > threshold]


# Exercise 6
# Return n!.
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        for i in range(1, n):
            n *= i
        return n


# Exercise 7
# Return the nth Fibonacci number.
# Define clearly whether your sequence starts with F(0)=0, F(1)=1.
def fibonacci(n):
    if n < 0:
        raise ValueError("Input should be a non-negative integer.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


# Exercise 8
# Return True if n is prime.
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# Exercise 9
# Return text reversed.
def reverse_string(text):
    return text[::-1]


# Exercise 10
# Return a dictionary mapping each character to its frequency.
# Example:
# count_characters("hello")
# -> {"h": 1, "e": 1, "l": 2, "o": 1}
def count_characters(text):
    frequency = {}
    for char in text:
        frequency[char] = frequency.get(char, 0) + 1
    return frequency


if __name__ == "__main__":
    # Add your own test cases here.
    print(is_even(2))
    print(is_even(2.0))
    print(sum_to_n(2))
    print(factorial(3))
    print(average([1, 2, 3]))
    print(filter_greater_than([1, 2, 3, 4, 5], 3))
    print(fibonacci(5))
    print(is_prime(7))
    print(reverse_string("hello"))
    print(count_characters("hello"))