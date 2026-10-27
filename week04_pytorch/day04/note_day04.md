# Day 4 — Linear Forward and Tensor Shape Contracts

Run from the repository root:

~~~bash
python -m unittest week04_pytorch.day04.test_day04 -v
~~~

The starter function intentionally contains a TODO. Implement it without using nn.Linear, torch.nn.functional.linear, or detach().

## Learning objectives

- Derive the output shape of a matrix multiplication.
- Implement a linear forward pass with inputs @ weights + bias.
- Explain how a one-dimensional bias broadcasts across batch rows.
- Confirm that the forward result remains connected to autograd.

## Exercise 1 — Predict the shapes first

Use these shapes:

~~~text
inputs:  [2, 3]
weights: [3, 2]
bias:    [2]
output:  [2, 2]
~~~

Matrix multiplication contracts the last dimension of inputs (3) with the first dimension of weights (3). The remaining dimensions are batch_size=2 and out_features=2.

The bias has one value for each output feature. It is added to every row of the [2, 2] matrix via broadcasting.

Before running the script, explain why a bias of shape [2] is compatible with an output of shape [2, 2], while a bias with 3 values would not match this layer's output features.

## Exercise 2 — Calculate one sample by hand

Given:

~~~python
inputs = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
weights = torch.tensor([[1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]])
bias = torch.tensor([0.5, -0.5])
~~~

For the first row, the matrix product is:

~~~text
[1, 2, 3] @ [[1, 0], [0, 1], [1, 1]] = [4, 5]
~~~

Then add the bias:

~~~text
[4, 5] + [0.5, -0.5] = [4.5, 4.5]
~~~

Do the same for the second row. Implement only the expression in day04.py and compare the output with the test.

## Exercise 3 — Keep autograd connected

The forward pass should use regular Tensor operations:

~~~python
output = inputs @ weights + bias
~~~

If inputs, weights, and bias are tracked appropriately, the output has a computation graph. Avoid converting tracked tensors to Python numbers, detaching the result, or wrapping the operation in no_grad().

The test calls output.sum().backward() only to check that the graph reaches each input tensor and parameter. A later exercise will focus on loss-based parameter gradients.

## Questions to answer in your own words

1. Why must weights.shape[0] equal inputs.shape[1]?
2. What do batch_size, in_features, and out_features mean?
3. Why does bias.shape equal [out_features] instead of [batch_size, out_features]?
4. What would happen to autograd if you returned output.detach()?
5. Why is the weight matrix stored as [in_features, out_features] in this exercise?

## Common pitfalls

- Using element-wise multiplication (*) instead of matrix multiplication (@).
- Reversing the dimensions of the weight matrix.
- Assuming broadcasting means that any bias shape is valid.
- Using a helper layer when the point is to implement the operation manually.
- Accidentally detaching the result from the computation graph.

## Definition of Done

- [ ] Implement linear_forward with inputs @ weights + bias.
- [ ] Predict [2, 3] @ [3, 2] -> [2, 2] before running the test.
- [ ] Calculate both output rows by hand.
- [ ] Explain bias broadcasting across the batch.
- [ ] Verify that autograd remains connected to inputs, weights, and bias.
- [ ] Pass all Day 4 tests.
- [ ] Record one useful observation in the session log.

## Session log

Date: 2026-10-27

- Completed:
- Bugs encountered:
- How I verified the fix:
- Concepts I can explain without notes:
- Questions to revisit:
