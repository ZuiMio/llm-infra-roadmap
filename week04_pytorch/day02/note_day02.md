# Day 2 — Leaf Tensors, grad_fn, and .grad

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day02.test_day02 -v
~~~

The starter functions intentionally contain TODOs. Implement them yourself before treating the tests as complete.

## Learning objectives

- Identify leaf and non-leaf tensors.
- Explain how requires_grad and grad_fn differ.
- Observe that a leaf tensor's .grad is None before backward().
- Understand why a non-leaf tensor's .grad is not retained by default.
- Use retain_grad() to inspect an intermediate tensor's gradient.

## Exercise 1 — Read the computation graph

Build this computation:

~~~python
x = torch.tensor(2.0, requires_grad=True)
middle = 3 * x
output = middle ** 2
~~~

Predict the value, is_leaf, requires_grad, and grad_fn for each tensor before executing any backward pass.

Expected forward values:
- x = 2
- middle = 6
- output = 36

Questions:
1. Why is x a leaf tensor, while middle and output are not?
2. Why does x.grad_fn normally equal None even though x.requires_grad is True?
3. What does middle.grad_fn tell you?
4. Why are all three tensors involved in autograd even though only x is a leaf?

## Exercise 2 — Follow the chain rule

The function is:

~~~python
output = (3 * x) ** 2 = 9 * x ** 2
~~~

Therefore:

~~~text
d(output)/dx = 18*x
~~~

At x = 2, the gradient is 36. Call output.backward() and inspect x.grad. Remember that backward() computes gradients; it does not change x's value.

## Exercise 3 — Why middle.grad is normally not saved

After building the graph, try:

~~~python
output.backward()
print(x.grad)
print(middle.grad)
~~~

By default, PyTorch accumulates gradients into leaf tensors that require gradients. The intermediate tensor middle is non-leaf; its .grad is normally not populated unless you call middle.retain_grad() before backward(). Accessing a non-leaf .grad can emit a warning, so use this as a deliberate experiment rather than routine logging.

Now try:

~~~python
x, middle, output = build_computation_graph(2.0)
middle.retain_grad()
output.backward()
print(x.grad)       # 36
print(middle.grad)  # 12
~~~

The intermediate gradient is:

~~~text
d(output)/d(middle) = 2 * middle = 12
~~~

And the chain rule gives:

~~~text
d(output)/dx = d(output)/d(middle) * d(middle)/dx
             = 12 * 3
             = 36
~~~

Note: retain_grad() is used for inspecting intermediate gradients. It is not usually needed for standard model parameter updates.

## Questions to answer in your own words

1. Is requires_grad the same thing as is_leaf? Explain using x and middle.
2. Why is x.grad None before backward()?
3. Why might middle.grad still be None after backward()?
4. What does retain_grad() change, and what does it not change?
5. How does the chain rule connect middle.grad to x.grad?

## Common pitfalls

- Assuming requires_grad=True means a tensor must be a leaf.
- Assuming every tensor involved in the graph will keep its .grad.
- Calling retain_grad() after backward(); it must be called before backward() for this experiment.
- Thinking backward() changes the tensor's data rather than computing gradients.
- Mixing up grad_fn (the tracked operation that produced a non-leaf result) and .grad (the gradient stored after backward()).

## Definition of Done

- [ ] Implement both functions without detaching tensors.
- [ ] Predict the metadata of x, middle, and output before running the code.
- [ ] Explain why x.grad_fn is None but middle.grad_fn is not.
- [ ] Compute x.grad = 36 by hand and verify it in PyTorch.
- [ ] Use retain_grad() and verify middle.grad = 12.
- [ ] Pass all Day 2 tests.
- [ ] Record one question or bug you encountered.

## Session log

Date: 2026-10-25

- Completed:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
