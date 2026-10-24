# Day 1 — Tensor Basics and Autograd

- `source .venv/bin/activate`

Run from the repository root:

~~~bash
python3.14 -m unittest week04_pytorch.day01.test_day01 -v
~~~

The starter functions intentionally contain TODOs. Implement them yourself before treating the tests as complete.

## Learning objectives

- Create a scalar floating-point tensor with requires_grad=True.
- Follow a forward computation and identify the computation graph.
- Use backward() to compute a derivative.
- Write a linear forward pass as inputs @ weights + bias.
- Compute a scalar mean-squared-error loss and inspect gradients for weights and bias.

## Exercise 1 — Scalar derivative

Start with:

~~~python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
~~~

Before executing backward(), predict x, y, x.requires_grad, x.is_leaf, y.grad_fn, and x.grad.

Math check:

- y = x squared.
- dy/dx = 2x.
- At x = 3, the forward value is 9 and the derivative is 6.
- After y.backward(), inspect x.grad. Before backward(), x.grad should be None.

Try other inputs, including a negative number. Explain why the derivative changes sign while the square remains non-negative.

## Exercise 2 — Manual linear forward

Use these tensors:

~~~python
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
weights = torch.tensor([[0.5], [-1.0]], requires_grad=True)
bias = torch.tensor([0.25], requires_grad=True)
~~~

Predict the shapes before computing:

- inputs: [2, 2]
- weights: [2, 1]
- bias: [1]
- predictions: [2, 1]

Calculate one row by hand, then implement the expression with matrix multiplication and broadcasting. Do not use nn.Linear.

## Exercise 3 — Loss and parameter gradients

Use a zero target with the predictions from Exercise 2. The mean-squared-error loss is the mean of every element in (predictions - targets) squared.

After loss.backward(), inspect weights.grad and bias.grad. The gradients are derivatives of the scalar loss with respect to the leaf parameter tensors.

For this fixed test case, verify the implementation against the test expectations. Then alter one weight and predict how the output and gradients will change.

## Questions to answer in your own words

1. What is the difference between a Tensor's data and its gradient?
2. Why does x.grad start as None?
3. What does grad_fn tell you about y?
4. Why must the loss be scalar for the simple loss.backward() call used here?
5. How do the input, weight, and bias shapes determine the output shape?
6. Why should the training loop avoid detaching weights from the computation graph?

## Common pitfalls

- Creating x without requires_grad=True, then expecting a gradient.
- Calling backward() before finishing the forward computation.
- Expecting gradients on every intermediate tensor; by default, gradients are retained on leaf tensors.
- Replacing matrix multiplication (@) with element-wise multiplication (*).
- Accidentally using a target shape that broadcasts to a larger tensor than intended.
- Assuming a second backward pass automatically clears existing gradients.

## Session log

Date: 2026-10-24

- Completed:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
