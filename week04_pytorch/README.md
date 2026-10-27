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


## Multi-Machine Environment Strategy

Use the same Git repository on each machine, but create a separate Python virtual environment on each OS. Never copy a `.venv` directory from one machine to another.

- **Mac:** run correctness tests on CPU first. MPS is an optional follow-up if the installed PyTorch build and Mac support it.
- **Remote Linux:** use CPU for correctness tests; use CUDA when a supported NVIDIA GPU and compatible PyTorch installation are available.
- **Windows + NVIDIA:** this is a useful local CUDA environment. Select the matching Windows / Pip / Python / CUDA option on the official [PyTorch installer page](https://pytorch.org/get-started/locally/) rather than copying an old install command.

Today's unit tests intentionally use CPU tensors. Keep them device-independent so test results are comparable across machines. GPU-specific experiments can be added as separate optional checks.

Check the installed environment:

~~~bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
~~~

On macOS, check MPS separately:

~~~bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
~~~

For the standard Week 04 exercises, GPU availability is not a prerequisite. Move work to CUDA or MPS only in a later optional experiment after the CPU tests pass.

## Day 1 — Start Here

Starter code: [day01/day01.py](day01/day01.py)  
Tests: [day01/test_day01.py](day01/test_day01.py)  
Notes and questions: [day01/note_day01.md](day01/note_day01.md)

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day01.test_day01 -v
~~~

Day 1 is implemented. Rerun its tests whenever you make changes, then continue to the Day 2 starter functions.


## Day 2 — Leaf Tensors and Gradient Storage

Starter code: [day02/day02.py](day02/day02.py)  
Tests: [day02/test_day02.py](day02/test_day02.py)  
Notes and questions: [day02/note_day02.md](day02/note_day02.md)

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day02.test_day02 -v
~~~

Build the graph x → middle → output with x = 2, middle = 3x, and output = middle². Predict leaf status, requires_grad, grad_fn, and values before running backward(). Then use middle.retain_grad() to inspect the intermediate gradient.

The Day 2 starter functions contain TODOs until you implement them.


## Day 3 — Gradient Accumulation and Resetting Gradients

Starter code: [day03/day03.py](day03/day03.py)  
Tests: [day03/test_day03.py](day03/test_day03.py)  
Notes and questions: [day03/note_day03.md](day03/note_day03.md)

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day03.test_day03 -v
~~~

Compare two separate backward() calls without clearing .grad against a second backward() after setting .grad = None. At x=2, the first derivative is 4, the second contribution is 3, and the accumulated gradient is 7; after clearing, the next gradient is 3. The Day 3 starter functions intentionally contain TODOs until you implement them.


## Day 4 — Hand-Written Linear Forward

Starter code: [day04/day04.py](day04/day04.py)  
Tests: [day04/test_day04.py](day04/test_day04.py)  
Notes and questions: [day04/note_day04.md](day04/note_day04.md)

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day04.test_day04 -v
~~~

Implement `inputs @ weights + bias` without using nn.Linear or torch.nn.functional.linear. Predict the output shape first, calculate one example by hand, verify bias broadcasting, and make sure autograd remains connected to inputs and parameters. The Day 4 starter function intentionally contains a TODO until you implement it.

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
python -m unittest week04_pytorch.day04.test_day04 -v
git status
git diff
~~~
