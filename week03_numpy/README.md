# Week 03 — NumPy and Tensor Thinking

## Goal

Use NumPy beyond basic syntax. Focus on array memory behavior, indexing, broadcasting, vectorized operations, reductions, and batched matrix multiplication. These ideas prepare for PyTorch tensors in Week 04.

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
Notes: [day02/note_day02.md](day02/note_day02.md)

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

The starter tests are expected to fail until you replace the TODOs.

## Git Workflow

Create a task branch from the latest `main`, for example:

```bash
git switch main
git pull
git switch -c feature/week03-day04-reductions
```

When your work is ready:

```bash
git status
git diff
git add week03_numpy
git diff --cached
git commit -m "Complete Week 03 Day 04 reduction exercises"
git push -u origin feature/week03-day04-reductions
```

Open a Pull Request into `main` after the tests pass.

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
