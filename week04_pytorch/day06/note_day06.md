# Day 6 — Manual Parameter Updates and a Tiny Training Loop

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day06.test_day06 -v
~~~

Prerequisites: complete Day 04's `linear_forward` and Day 01's `mean_squared_error`. The Day 06 functions intentionally contain TODOs.

## Learning objectives

- Update parameters manually without using torch.optim.
- Understand why the update must happen under torch.no_grad().
- Clear gradients after consuming them for an update.
- Train a tiny linear model and confirm the loss decreases.

## Exercise 1 — A single manual update

After backward(), the gradient contains the local slope of the loss with respect to each parameter. A simple SGD update is:

~~~text
parameter = parameter - learning_rate * parameter.grad
~~~

For today's implementation, use a context manager:

~~~python
with torch.no_grad():
    weights -= learning_rate * weights.grad
    bias -= learning_rate * bias.grad
~~~

This prevents the in-place update from being recorded as another differentiable operation. After updating, set both gradients to None so that the next backward pass starts a new update step.

The function should reject a non-positive or non-finite learning rate and should raise RuntimeError if a required gradient is missing.

## Exercise 2 — Fit y = 2*x + 1

Use x values [-2, -1, 0, 1, 2] and targets produced by y = 2*x + 1. Start from weight=-0.5 and bias=0.0. Train for 100 steps with learning_rate=0.05.

For every step:
1. Compute predictions using the linear forward function.
2. Compute mean-squared-error.
3. Record the current loss.
4. Call backward().
5. Update the weight and bias with manual_sgd_update().

After the final update, compute one more loss and append it. This makes loss_history have steps+1 entries, including both the initial and final loss.

## What to observe

- Loss should decrease substantially.
- The weight should approach 2.
- The bias should approach 1.
- The update itself should not create a new gradient graph.
- Gradients should be None between optimizer steps because manual_sgd_update() clears them.

## Questions

1. Why do we use torch.no_grad() during the parameter update?
2. Why does a large learning rate sometimes make training unstable?
3. Why should the loss be computed before calling backward()?
4. Why clear gradients after the parameter update in this helper?
5. In a production training loop, what responsibilities would torch.optim usually take over?

## Definition of Done

- [ ] Implement manual_sgd_update() with validation, no_grad(), and gradient clearing.
- [ ] Implement the complete tiny training loop without torch.optim.
- [ ] Verify that loss decreases and parameters approach weight=2, bias=1.
- [ ] Explain the roles of forward, loss, backward, update, and gradient clearing.
- [ ] Pass all Day 06 tests.
- [ ] Record the loss before and after training in the session log.

## Session log

Date: 2026-10-29

- Completed:
- Initial loss:
- Final loss:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
