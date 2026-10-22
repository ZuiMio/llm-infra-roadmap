# Week 03 — Day 03 Notes

## Shape Predictions Before Running

For `features.shape == (4, 3)` and `bias.shape == (3,)`:

- What shape should `features + bias` have?
- Which axis is aligned when NumPy broadcasts a 1D array against a 2D array?
- What happens if `bias.shape == (4,)` instead?

## Broadcasting a Feature Vector

- Why can a vector of shape `(n_features,)` be added to a matrix of shape `(n_samples, n_features)`?
- Does NumPy need to physically repeat the bias vector for every sample?
- What shape mismatch should be rejected by `add_feature_bias`?

## Broadcasting Along Rows

- If a matrix has shape `(n_rows, n_cols)`, why is `row_scales.shape == (n_rows,)` not generally the right shape for row-wise multiplication?
- What does `row_scales[:, np.newaxis]` change?
- Why is a square matrix a useful test for distinguishing row scaling from column scaling?

## Feature Standardization

- Which axis represents features in a batch matrix?
- How do per-feature mean and standard deviation vectors broadcast over samples?
- Why should zero or negative standard deviations be rejected?

## Vectorization

- How does a NumPy element-wise operation differ from looping through each element in Python?
- Does vectorization change the shape of the array?
- How can I verify that an operation did not mutate its input?

## Test Results

- Command used:
- Number of tests passed:
- Any failing test:

## Debugging Notes

- Bug or confusing behavior:
- Root cause:
- Fix:
- What I learned:
