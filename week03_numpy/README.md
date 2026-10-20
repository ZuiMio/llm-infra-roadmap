# Week 03 — NumPy and Tensor Thinking

## Goal

Use NumPy beyond basic syntax. Focus on array memory behavior, shape reasoning, vectorized operations, reductions, and batched matrix multiplication. These ideas prepare for PyTorch tensors in Week 04.

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

Starter code: day01.py
Tests: test_day01.py
Notes: note.md

From the repository root, run:

    python3.14 -m unittest week03_numpy.test_day01 -v

The starter tests are expected to fail until you replace the TODOs.

## Git Workflow

Use the development branch: feature/week03-numpy-tensor-thinking

When your work is ready:

    git status
    git diff
    git add week03_numpy
    git diff --cached
    git commit -m "Complete Week 03 Day 01 NumPy layout exercises"
    git push -u origin feature/week03-numpy-tensor-thinking

Open a Pull Request into main after the tests pass.

## Definition of Done for Day 1

- [ ] Describe an ndarray's shape, dtype, strides, size in bytes, and C-contiguous status.
- [ ] Demonstrate that a basic slice can share memory with its source.
- [ ] Demonstrate that a copied slice does not share memory with its source.
- [ ] Explain why a transpose may be non-contiguous.
- [ ] Pass all Day 1 tests.
- [ ] Complete note.md.
