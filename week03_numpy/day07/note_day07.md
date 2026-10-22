# Day 7 — Mini Project: NumPy Batch-Feature Pipeline

Run tests from the repository root:

~~~bash
python3.14 -m unittest week03_numpy.day07.test_day07 -v
~~~

## Learning goals

- Fit a standardizer using training data.
- Reuse the fitted mean and standard deviation for new data.
- Compute a linear layer with matrix multiplication and broadcasting.
- Process data in row batches without changing output order.
- Test the pipeline end to end and verify that batching gives the same result as processing all rows together.

Use NumPy only; do not use PyTorch yet.

## Project scenario

You have a training feature matrix with shape (n_samples, n_features). You fit one mean and population standard deviation per feature. Later, a validation or inference matrix arrives. It must be transformed with the training statistics, not statistics newly fitted on that later matrix.

After normalization, a provided linear model computes:

~~~python
scores = normalized_features @ weights + bias
~~~

The weight matrix has shape (n_features, n_outputs), and the bias has shape (n_outputs,). The result has shape (n_samples, n_outputs).

The model parameters are given to you. This project is about the data pipeline and shape correctness, not training a model.

## Functions to implement

### 1. fit_standardizer(train_features)

Validate a non-empty 2D real-valued finite matrix. Compute feature-wise means and population standard deviations with axis=0 and ddof=0. Return (mean, std), each shaped (n_features,).

A constant feature is allowed and has standard deviation zero. Reject invalid types, NaN, infinity, and invalid dimensions. Do not mutate the input.

### 2. transform_features(features, mean, std)

Validate the feature matrix and the supplied one-dimensional statistics. The number of statistics must match the feature count; all values must be real and finite; standard deviations must be non-negative.

Use broadcasting to calculate:

~~~python
normalized = (features - mean) / safe_std
~~~

where safe_std replaces each zero standard deviation with 1.0. This makes a constant training feature map to zero while allowing deviations in new data to remain visible.

Do not calculate new statistics inside this function.

### 3. linear_forward(inputs, weights, bias)

Validate the input and model parameter shapes. Compute inputs @ weights + bias. Use the last dimension of inputs as the input feature dimension, weights.shape[0] must match it, and bias.shape must be (weights.shape[1],).

Reject invalid dimensions and non-finite or non-real values. Do not mutate inputs.

### 4. predict_in_batches(features, mean, std, weights, bias, batch_size)

This is the end-to-end function. It should:

1. Validate the inputs, statistics, and model parameters.
2. Require batch_size to be a positive integer; booleans do not count as valid integers for this exercise.
3. Slice the feature matrix by rows.
4. Transform each batch using the same fitted mean and std.
5. Compute the linear output for each normalized batch.
6. Concatenate batch outputs in the original row order.

The output should have shape (n_samples, n_outputs). A final batch may be smaller than batch_size, and batch_size may be larger than the number of samples.

## Important design rules

- The training matrix is used to fit the statistics.
- Validation or inference data reuses those same statistics.
- Do not mutate input arrays.
- Use float64 outputs and calculations.
- Prefer validation helpers so the four public functions do not each need to duplicate every check.
- A zero standard deviation is valid; a negative standard deviation is not.
- Keep the supplied function signatures.

## Work order

1. Implement fit_standardizer.
2. Implement transform_features and run its tests.
3. Implement linear_forward and run its tests.
4. Implement predict_in_batches.
5. Compare the batched output against the result from transforming and evaluating the full matrix at once.
6. Run the complete test suite and record one bug you found.

## Predict before running

Given:

~~~python
train_features = np.array([
    [1.0, 10.0, 5.0],
    [3.0, 14.0, 5.0],
    [5.0, 18.0, 5.0],
])
~~~

1. What are the shapes of mean and std?
2. Why is the third feature's standard deviation zero?
3. If the model has 3 input features and 2 output features, what shape must weights have?
4. If features.shape == (7, 3) and batch_size == 3, how many batches are processed?
5. Should changing batch_size change the returned predictions, assuming the calculation is otherwise correct?

## Definition of Done

- [ ] All four functions are implemented.
- [ ] Feature statistics use axis 0 and population standard deviation.
- [ ] New data reuses fitted training statistics.
- [ ] Zero-std features do not cause division by zero.
- [ ] Linear output shapes and mismatches are handled correctly.
- [ ] Batched inference matches full-matrix inference.
- [ ] Inputs remain unchanged.
- [ ] All Day 7 tests pass.
- [ ] Record at least one bug and its cause in this note.
