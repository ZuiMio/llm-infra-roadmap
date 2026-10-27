# Day 5 — Loss and Parameter Gradients

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day05.test_day05 -v
~~~

Prerequisite: complete Day 04's `linear_forward`. The Day 05 starter function intentionally contains a TODO.

## Goals

- Calculate predictions using a hand-written linear layer.
- Compute mean-squared-error loss as a scalar.
- Call backward() and inspect the gradients of weights and bias.
- Confirm that backward() computes gradients but does not itself update the parameters.

## Exercise 1 — Forward pass

Use:

~~~python
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
bias = torch.tensor([0.25], requires_grad=True)
targets = torch.zeros((2, 1))
~~~

Expected predictions:

~~~text
tensor([[-1.25],
        [-2.25]])
~~~

## Exercise 2 — Loss

Mean squared error is the mean of all squared prediction errors:

~~~python
loss = ((predictions - targets) ** 2).mean()
~~~

For the supplied values:

~~~text
loss = (1.25² + 2.25²) / 2 = 3.3125
~~~

The loss is a scalar, so the simple `loss.backward()` call computes its gradients.

## Exercise 3 — Hand-calculate parameter gradients

For this exercise there is one output feature, so MSE averages over the N samples:

~~~text
dL/dW = (2/N) * X.T @ (predictions - targets)
dL/db = (2/N) * sum(predictions - targets, over the batch)
~~~

Here there is one output feature and two samples. Work out both entries of the weight gradient and the bias gradient by hand before running the test.

Expected gradients:

~~~text
weights.grad = [[ -8.0],
                [-11.5]]
bias.grad    = [-3.5]
~~~

Remember the distinction:
- `loss.backward()` calculates and stores gradients.
- An optimizer or a manual update changes the parameter values.
- This exercise deliberately does not update parameters.

The function clears old gradients before its independent backward pass. This prevents an earlier experiment from contaminating today's expected values; Day 03 already demonstrated what happens when gradients accumulate.

## Questions

1. Why must weights have `requires_grad=True` for weights.grad to be populated?
2. Why is loss a scalar here?
3. Why does the bias gradient sum the per-sample error contributions?
4. What does backward() change, and what does it leave unchanged?
5. Why are the returned gradient tensors cloned?

## Definition of Done

- [ ] Implement `compute_loss_and_gradients`.
- [ ] Derive the MSE value by hand.
- [ ] Derive both weight-gradient entries and the bias gradient.
- [ ] Explain why backward() is not a parameter update.
- [ ] Pass all Day 05 tests.
- [ ] Record one debugging observation in the session log.

## Session log

Date: 2026-10-28

- Completed:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
