# Week 01 — Python Foundations

## Goal

By the end of this week, I should be able to write small Python programs independently and implement a simple neural-network-style pipeline with NumPy:

`Linear → ReLU → MSELoss`

The goal is not to finish all of Python. The goal is to make Python syntax stop being an obstacle before learning PyTorch.

## Daily Plan

### Day 1 — Basic Syntax and Control Flow

Learn:

- variables
- numbers and strings
- `if / elif / else`
- `for`
- `while`
- `range`
- basic functions

Exercises are in `day01.py`.

### Day 2 — Data Structures

Learn:

- list
- tuple
- dict
- set
- slicing
- list/dict comprehensions

### Day 3 — Functions

Learn:

- positional and keyword arguments
- default arguments
- `*args`
- `**kwargs`
- scope
- mutable vs immutable objects

### Day 4 — Classes

Learn:

- `class`
- `__init__`
- `self`
- attributes
- methods
- inheritance
- `__call__`

### Day 5 — Files and Iteration

Learn:

- exceptions
- `with open(...)`
- iterable / iterator
- `__iter__`
- `__next__`
- generators
- `yield`

### Day 6 — NumPy

Learn:

- ndarray
- shape
- ndim
- dtype
- reshape
- transpose
- matrix multiplication
- broadcasting

Main habit: write down tensor/array shapes before running the code.

### Day 7 — Mini Project

Implement with NumPy:

- `Linear`
- `ReLU`
- `MSELoss`

Then connect them:

`input → Linear → ReLU → prediction → MSELoss`

## Learning Notes

Do not copy textbook definitions here. Record things that required thought.

### Concepts I initially misunderstood

- 

### Bugs I encountered

#### Bug 1

**Symptom:**

**Cause:**

**Fix:**

**What I learned:**

## End-of-week Self Check

I should be able to explain without notes:

- the difference between list, tuple, dict and set
- mutable vs immutable
- what `*args` and `**kwargs` do
- what `self` means
- why `with open()` is useful
- what a generator does
- the difference between NumPy `*` and `@`
- how the shapes in `X @ W + b` are determined
