# Week 03 — NumPy and Tensor Thinking

## Goal

Use NumPy beyond basic syntax. Focus on array memory behavior, indexing, broadcasting, vectorized operations, reductions, batched matrix multiplication, and preprocessing. These ideas prepare for PyTorch tensors in Week 04.

## Seven-Day Plan

- Day 1: Array memory layout, strides, views, and copies.
- Day 2: Basic indexing, advanced indexing, boolean masks, and copy behavior.
- Day 3: Broadcasting rules and vectorization versus Python loops.
- Day 4: Reduction axes, keepdims, and numerically careful calculations.
- Day 5: Matrix multiplication, batch dimensions, and shape contracts.
- Day 6: Build a small preprocessing pipeline with normalization and validation.
- Day 7: Mini project — a tested NumPy batch-feature pipeline.

## Learning Routine

1. Predict shapes and whether an operation returns a view or a copy.
2. Implement the function without copying a complete solution.
3. Run the supplied tests.
4. Explain any mismatch between your prediction and the actual result.
5. Record one bug and what you learned.

Use NumPy only during this week. Do not use PyTorch yet.

## Day 1 — Memory and Layout

Starter code: [day01/day01.py](day01/day01.py)  
Tests: [day01/test_day01.py](day01/test_day01.py)  
Notes: [note.md](note.md)

```bash
python3.14 -m unittest week03_numpy.day01.test_day01 -v
```

## Day 2 — Indexing and Copy Behavior

Starter code: [day02/day02.py](day02/day02.py)  
Tests: [day02/test_day02.py](day02/test_day02.py)  
Notes: [note_day02.md](day02/note_day02.md)

```bash
python3.14 -m unittest week03_numpy.day02.test_day02 -v
```

## Day 3 — Broadcasting and Vectorization

Starter code: [day03/day03.py](day03/day03.py)  
Tests: [day03/test_day03.py](day03/test_day03.py)  
Notes: [day03/note_day03.md](day03/note_day03.md)

```bash
python3.14 -m unittest week03_numpy.day03.test_day03 -v
```

## Day 4 — Reduction Axes, keepdims, and Precision

Starter code: [day04/day04.py](day04/day04.py)  
Tests: [day04/test_day04.py](day04/test_day04.py)  
Notes: [day04/note_day04.md](day04/note_day04.md)

Today's exercises cover summation over axes, optional dimension retention, feature-wise means, feature centering, and mean absolute error. The reduction tasks use float64 accumulation to make numerical precision an explicit part of the exercise.

Run from the repository root:

```bash
python3.14 -m unittest week03_numpy.day04.test_day04 -v
```

## Day 5 — Matrix Multiplication and Batch Dimensions

Starter code: [day05/day05.py](day05/day05.py)  
Tests: [day05/test_day05.py](day05/test_day05.py)  
Notes: [day05/note_day05.md](day05/note_day05.md)

Today's exercises cover the difference between `*` and `@`, inner matrix dimensions, broadcast-compatible batch dimensions, and the forward pass of a linear layer.

Run from the repository root:

```bash
python3.14 -m unittest week03_numpy.day05.test_day05 -v
```

The Day 5 starter functions contain TODOs, so the tests are expected to fail until the implementations are complete.

## Day 6 — Feature Validation and Preprocessing

Starter code: [day06/day06.py](day06/day06.py)  
Tests: [day06/test_day06.py](day06/test_day06.py)  
Notes: [day06/note_day06.md](day06/note_day06.md)

Today's exercises cover numeric and finite-value validation, feature-wise means and population standard deviations, normalization with broadcasting, and handling constant-valued features without division by zero.

Run from the repository root:

```bash
python3.14 -m unittest week03_numpy.day06.test_day06 -v
```

The Day 6 starter functions contain TODOs, so the tests are expected to fail until the implementations are complete.


## Day 7 — Mini Project: Batch-Feature Pipeline

Starter code: [day07/day07.py](day07/day07.py)  
Tests: [day07/test_day07.py](day07/test_day07.py)  
Notes: [day07/note_day07.md](day07/note_day07.md)

Integrate Day 5 matrix multiplication and Day 6 feature preprocessing into one pipeline. Fit statistics on training data, reuse those statistics for new data, compute a linear forward pass, and process inference rows in batches. The batch output must match processing all rows at once.

~~~bash
python3.14 -m unittest week03_numpy.day07.test_day07 -v
~~~

The Day 7 starter functions contain TODOs, so the tests are expected to fail until the implementation is complete.

## Git Workflow — Direct Commits to main

For this personal learning repository, work directly on `main`; do not create a feature branch or Pull Request for daily exercises.

From the repository root:

```bash
git switch main
git pull origin main
# Edit and test your files.
git status
git diff
git add week03_numpy/day07 week03_numpy/README.md README.md
git diff --cached
git commit -m "Complete Week 03 Day 07 batch pipeline project"
git push origin main
```

Review the diff before committing, and run the day's tests before considering the work complete.

## Definition of Done for Day 1

- [ ] Describe an ndarray's shape, dtype, strides, size in bytes, and C-contiguous status.
- [ ] Demonstrate that a basic slice can share memory with its source.
- [ ] Demonstrate that a copied slice does not share memory with its source.
- [ ] Explain why a transpose may be non-contiguous.
- [ ] Pass all Day 1 tests.
- [ ] Complete the Day 1 notes.

## Definition of Done for Day 2

- [ ] Use integer-array indexing and explain why the selected result is a copy.
- [ ] Filter values with a boolean mask and validate mask shape and dtype.
- [ ] Replace values conditionally without modifying the source array.
- [ ] Select paired matrix coordinates using row and column index arrays.
- [ ] Pass all Day 2 tests.
- [ ] Complete [note_day02.md](day02/note_day02.md).

## Definition of Done for Day 3

- [ ] Explain why a vector with shape `(n_features,)` broadcasts over a batch matrix.
- [ ] Scale rows by adding a singleton axis to the row-scale vector.
- [ ] Standardize features using per-feature statistics.
- [ ] Apply ReLU without explicit Python loops.
- [ ] Pass all Day 3 tests.
- [ ] Complete [note_day03.md](day03/note_day03.md).

## Definition of Done for Day 4

- [ ] Predict output shapes for axis reductions with and without `keepdims`.
- [ ] Compute per-feature means across samples.
- [ ] Center every feature using broadcasting and `keepdims=True`.
- [ ] Explain why float64 accumulation can improve some reductions.
- [ ] Compute mean absolute error while preserving input arrays.
- [ ] Pass all Day 4 tests.
- [ ] Complete [note_day04.md](day04/note_day04.md).

## Definition of Done for Day 5

- [ ] Explain the difference between element-wise multiplication and matrix multiplication.
- [ ] Predict output shapes and validate inner matrix dimensions.
- [ ] Implement matrix multiplication for two 2D arrays.
- [ ] Implement batched matrix multiplication with compatible batch broadcasting.
- [ ] Implement a linear layer forward pass using `inputs @ weights + bias`.
- [ ] Check invalid dimensions and shape mismatches with `ValueError`.
- [ ] Pass all Day 5 tests without mutating inputs.
- [ ] Complete [note_day05.md](day05/note_day05.md).

## Definition of Done for Day 6

- [ ] Validate non-empty 2D real-valued feature matrices and reject NaN/inf.
- [ ] Check expected feature count when provided.
- [ ] Compute per-feature means and population standard deviations.
- [ ] Normalize with broadcasting and reuse supplied training statistics.
- [ ] Handle zero-standard-deviation features without division by zero.
- [ ] Build the end-to-end preprocessing function and preserve inputs.
- [ ] Pass all Day 6 tests.
- [ ] Complete [note_day06.md](day06/note_day06.md).

## Definition of Done for Day 7

- [ ] Fit feature-wise training statistics using population standard deviation.
- [ ] Transform another dataset with the same training statistics.
- [ ] Validate shapes and values for a linear forward pass.
- [ ] Process inference data in batches and preserve row order.
- [ ] Show that batched and full-matrix outputs match.
- [ ] Preserve all input arrays.
- [ ] Pass all Day 7 tests.
- [ ] Record a bug and its cause in [note_day07.md](day07/note_day07.md).
