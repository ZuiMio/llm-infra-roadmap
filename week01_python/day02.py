"""
Week 01 - Day 02
Python data structures.

Topics:
- list
- tuple
- dict
- set
- slicing
- comprehensions
- enumerate
- zip

Rule:
- Try each exercise yourself.
- Do not look up a complete solution immediately.
- Keep functions small and readable.
"""


# Exercise 1
# Return:
# - first element
# - last element
# - number of elements
# Do not modify the input list.
# Return None for an empty list.
def list_summary(values):
    if not values:
        return None
    return values[0], values[-1], len(values)


# Exercise 2
# Return the middle three elements of a list.
# Assume the input list has exactly 5 elements.
# Use slicing.
def middle_three(values):
    mid = len(values) // 2
    return values[mid - 1:mid + 2]


# Exercise 3
# Return a list containing each value only once.
# Preserve the original order.
# Do not use list(set(values)).
def unique_values(values):
    seen = set()
    unique_list = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_list.append(value)
    return unique_list


# Exercise 4
# shape is a tuple: (B, T, C)
# Return a dictionary containing:
# - batch_size
# - sequence_length
# - hidden_size
def describe_shape(shape):
    return {
        "batch_size": shape[0],
        "sequence_length": shape[1],
        "hidden_size": shape[2]
    }


# Exercise 5
# Count the frequency of each word.
# Do not use collections.Counter.
def count_words(words):
    word_list = words.split()
    total = len(word_list)
    word_freq = {}
    for word in word_list:
        word_freq[word] = word_freq.get(word, 0) + 1 / total
    return word_freq


# Exercise 6
# Return a new dictionary containing only
# entries whose score is greater than or equal to threshold.
# Do not modify the input dictionary.
def filter_by_score(scores, threshold):
    return {k: v for k, v in scores.items() if v >= threshold}


# Exercise 7
# Return the elements that appear in both collections as a set.
def common_elements(a, b):
    return a & b


# Exercise 8
# Return a tuple containing:
# - union
# - intersection
# - elements only in a
def compare_sets(a, b):
    return (a | b, a & b, a - b)


# Exercise 9
# Return the square of every even number.
# Use a list comprehension.
def square_even_numbers(values):
    return [x ** 2 for x in values if x % 2 == 0]


# Exercise 10
# Return a dictionary mapping each integer
# from 1 to n to its square.
# Use a dict comprehension.
def create_square_dict(n):
    return {i: i ** 2 for i in range(1, n + 1)}


# Mini task
# Analyze a list of tokens and return:
# {
#     "total": total number of tokens,
#     "unique": number of unique tokens,
#     "frequency": frequency dictionary,
# }
def analyze_tokens(tokens):
    total = len(tokens)
    unique = len(set(tokens))
    frequency = {token: tokens.count(token) / total for token in set(tokens)}
    return {
        "total": total,
        "unique": unique,
        "frequency": frequency
    }


if __name__ == "__main__":
    # Add your own test cases here.
    list_summary([1, 2, 3, 4, 5])
    middle_three([1, 2, 3, 4, 5])
    unique_values([1, 2, 2, 3, 3, 4])
    describe_shape((10, 20, 30))
    count_words("hello world hello")
    filter_by_score({"Alice": 85, "Bob": 90, "Charlie": 75}, 80)
    common_elements({1, 2, 3}, {3, 4, 5})
    compare_sets({1, 2, 3}, {3, 4, 5})
    square_even_numbers([1, 2, 3, 4, 5])
    create_square_dict(5)
    analyze_tokens(["hello", "world", "hello"])