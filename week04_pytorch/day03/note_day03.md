# Day 3 — Gradient Accumulation and Resetting Gradients

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day03.test_day03 -v
~~~

The starter functions intentionally contain TODOs. Implement them yourself and run the tests before marking Day 3 complete.

## Learning objectives

- Explain why repeated backward() calls add to a leaf tensor's existing .grad.
- Distinguish accidental gradient carry-over from intentional gradient accumulation.
- Reset gradients before starting a new optimization step.
- Connect the scalar experiment to optimizer.zero_grad() in a model training loop.

## Exercise 1 — Observe accumulation

Use one leaf tensor:

~~~python
x = torch.tensor(2.0, requires_grad=True)

first_loss = x ** 2
first_loss.backward()
print(x.grad)  # 4

second_loss = 3 * x
second_loss.backward()
print(x.grad)  # 7
~~~

Why is the second result 7 instead of 3?

The first loss contributes d(x²)/dx = 2x = 4 at x=2. The second loss contributes d(3x)/dx = 3. PyTorch accumulates the new gradient into the existing .grad, so the stored result becomes 4 + 3 = 7.

Important: these are two separate forward computations and two separate computation graphs. Do not call backward() twice on the exact same graph unless you deliberately use retain_graph=True.

## Exercise 2 — Clear before the next step

Now insert a reset between the losses:

~~~python
x = torch.tensor(2.0, requires_grad=True)

first_loss = x ** 2
first_loss.backward()
print(x.grad)  # 4

x.grad = None

second_loss = 3 * x
second_loss.backward()
print(x.grad)  # 3
~~~

The second backward() now starts from an empty gradient field. It no longer adds to the first step's gradient.

In a model training loop, use the optimizer API:

~~~python
optimizer.zero_grad(set_to_none=True)
loss = model(inputs)
loss = criterion(loss, targets)
loss.backward()
optimizer.step()
~~~

This is a minimal illustration: the exact model and criterion depend on the task. The key ordering is to clear stale gradients before the backward pass for the next optimizer update.

## Exercise 3 — Accumulation can also be intentional

Do not conclude that gradient accumulation is always wrong. In gradient accumulation training, several micro-batch losses intentionally contribute to one parameter update. Gradients are cleared at the start of the accumulation window, several backward passes are run, and optimizer.step() occurs after the intended micro-batches have contributed.

Distinguish these situations:
- **Accidental carry-over:** a previous optimizer step's gradients remain and contaminate the next step.
- **Intentional accumulation:** multiple micro-batch gradients are combined before one optimizer update.

For today's scalar experiment, the second backward() is used to expose the default accumulation behavior, not to perform a useful optimizer update.

## Questions to answer in your own words

1. Why does x.grad become 7 after the second backward() without clearing?
2. Why does x.grad become 3 when it is set to None before the second backward()?
3. What is the difference between clearing gradients and changing the parameter value?
4. Why would stale gradients be a bug if you intended one optimizer update per batch?
5. Why do training systems sometimes accumulate gradients over several micro-batches on purpose?
6. Why must optimizer.step() happen after the gradients intended for that update have been computed?

## Common pitfalls

- Assuming backward() overwrites .grad rather than accumulating into it.
- Clearing gradients after backward() instead of before the next intended accumulation window.
- Reusing a freed computation graph for another backward() call.
- Calling optimizer.step() after every micro-batch when you intended to accumulate several micro-batches first.
- Confusing gradient accumulation with accumulating loss values; they are not the same operation.

## Definition of Done

- [ ] Implement both functions without using torch.optim.
- [ ] Verify that gradients 4 and 3 accumulate to 7 at x=2.
- [ ] Clear .grad and verify that the second gradient is 3 instead of 7.
- [ ] Explain why separate backward() calls here use separate forward computations.
- [ ] Explain when gradient accumulation is intentional in training.
- [ ] Pass all Day 3 tests.
- [ ] Record one bug or useful observation in the session log.

## Session log

Date: 2026-10-26

- Completed:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
