# Week 04 — PyTorch Tensors and Autograd

## Goal

Understand how PyTorch builds a computation graph during the forward pass and uses the chain rule during backward propagation. Start with a scalar example, then compute a small linear layer and loss without using nn.Linear or an optimizer.

**Planned window:** 2026-10-27 to 2026-11-02  
**Study kickoff:** 2026-10-24, starting a few days early. Keep the learning order; adjust the calendar if needed.

## Seven-Day Plan

- Day 1: Tensor basics, a scalar computation graph, and the gradient of x squared.
- Day 2: Leaf tensors, requires_grad, grad_fn, and the .grad field.
- Day 3: Gradient accumulation and why gradients must be cleared between optimizer steps.
- Day 4: Hand-write a linear forward pass: inputs @ weights + bias.
- Day 5: Compute a loss and inspect gradients for weights and bias.
- Day 6: Update parameters manually inside torch.no_grad(); fit a simple relationship.
- Day 7: Rebuild the key parts without looking at the solution, run tests, and write a weekly review.

This is a practical day-by-day breakdown added to the 12-week plan. The source plan's required Week 04 deliverables remain the authority.

## Day 1 — Start Here

Starter code: [day01/day01.py](day01/day01.py)  
Tests: [day01/test_day01.py](day01/test_day01.py)  
Notes and questions: [day01/note_day01.md](day01/note_day01.md)

Run from the repository root:

~~~bash
python3.14 -m unittest week04_pytorch.day01.test_day01 -v
~~~

The starter functions intentionally raise NotImplementedError. Implement one function at a time, and rerun the tests after each function.

## Key Concepts

### Tensor and computation graph

- A Tensor has a shape, dtype, and device.
- A leaf tensor such as a trainable scalar can be created with requires_grad=True.
- Operations involving tracked tensors produce results with a gradient function, accessible through grad_fn.
- Calling backward() on a scalar result computes gradients for participating leaf tensors.
- The resulting gradient is stored in each leaf tensor's .grad field.

### Required scalar exercise

For x = 3 and y = x squared:

- Forward value: y = 9.
- Derivative: dy/dx = 2x.
- Expected gradient: x.grad = 6 after y.backward().

Predict the result on paper before running the code.

### Required linear exercise

For inputs shaped [B, I], weights shaped [I, O], and bias shaped [O]:

~~~python
predictions = inputs @ weights + bias
~~~

The output shape is [B, O]. Build the expression using Tensor operations rather than nn.Linear. Compute a scalar mean-squared-error loss, call backward(), then inspect weights.grad and bias.grad.

## Definition of Done

- [ ] Explain the forward value and derivative for y = x squared at x = 3.
- [ ] Explain requires_grad, grad_fn, leaf tensors, and .grad.
- [ ] Explain why backward() populates gradients on leaf tensors.
- [ ] Demonstrate gradient accumulation and clear gradients before a fresh update.
- [ ] Hand-write inputs @ weights + bias and a scalar loss without nn.Linear or an optimizer.
- [ ] Perform a manual parameter update under torch.no_grad().
- [ ] Fit a simple relationship with a training loop and show that the loss decreases.
- [ ] Pass the Week 04 tests and complete the learning notes.

## References

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Autograd: Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [torch.autograd reference](https://docs.pytorch.org/docs/stable/autograd.html)

## Git Workflow

This personal learning repository uses direct commits to main for daily exercises. Before committing, inspect the diff and run the tests.

~~~bash
git switch main
git pull origin main
python3.14 -m unittest week04_pytorch.day01.test_day01 -v
git status
git diff
~~~
