# Day 7 — Week 04 Integration and Independent Rebuild

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day07.test_weekly_review -v
~~~

This is a final integration challenge. The starter function intentionally contains a TODO. Do not import the helper implementations from Day 01, Day 04, Day 05, or Day 06; rebuild the minimal training loop yourself.

## Required implementation

Implement `train_linear_model` in `weekly_review.py`.

Your code must:
1. Validate shapes, positive integer step count, and positive finite learning rate.
2. Create fresh leaf parameters from the supplied initial weights and bias.
3. Compute predictions with `inputs @ weights + bias`.
4. Compute MSE explicitly with `((predictions - targets) ** 2).mean()`.
5. Call `loss.backward()`.
6. Update weights and bias inside `torch.no_grad()`.
7. Set both parameter gradients to None after each update.
8. Record loss before each update and once after the final update.
9. Return detached copies of trained weights, bias, and the loss history.

Do not use nn.Linear, torch.nn.functional.linear, or torch.optim. Those tools would hide the mechanics that this week's tasks are designed to teach.

## Experiment A — Single-input relation

Train on x = [-2, -1, 0, 1, 2] with targets y = 2x + 1. Start with weight=-0.5 and bias=0.0.

With 100 steps and learning rate 0.05, the loss should drop and the learned parameters should approach 2 and 1. The history must contain 101 entries: the pre-update loss for each of 100 steps plus one final loss after the last update.

## Experiment B — Multiple features

The test suite also asks you to fit a linear relationship with two input and two output features. This confirms that your implementation generalizes beyond a single scalar output and that broadcasting and matrix dimensions are handled correctly.

## Week 04 final checklist

- [ ] Explain x=3, y=x², and why x.grad=6.
- [ ] Explain requires_grad, is_leaf, grad_fn, and .grad.
- [ ] Demonstrate how gradients accumulate across backward() calls.
- [ ] Explain when gradient accumulation is intentional.
- [ ] Hand-write inputs @ weights + bias and predict the shape.
- [ ] Compute MSE and derive weight and bias gradients for a small example.
- [ ] Update parameters manually under torch.no_grad().
- [ ] Clear gradients before the next intended optimization step.
- [ ] Train a tiny linear model and show that the loss decreases.
- [ ] Pass Day 05, Day 06, and Day 07 tests.
- [ ] Record initial/final loss, a bug, the evidence used to fix it, and remaining questions.

## Weekly review record

Week: 04 — PyTorch Tensor and Autograd

- Tests passed:
- Initial loss / final loss:
- One bug and its root cause:
- Evidence that the fix works:
- Concepts I can explain without notes:
- Concepts that still need practice:
- Next step:
