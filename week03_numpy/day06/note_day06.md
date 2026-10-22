# Day 6 — Feature Validation and Preprocessing

## Learning goals

- Validate input data before doing numerical calculations.
- Compute per-feature means and population standard deviations.
- Normalize a matrix using broadcasting.
- Handle missing or non-finite numeric values by rejecting invalid data.
- Handle constant-valued features without division by zero.
- Compose reusable functions into one small preprocessing pipeline.

Use NumPy only; do not use PyTorch yet.

## 1. Why validate data first?

A NumPy array can have the right shape but still contain invalid values, such as NaN or infinity. Shape checks alone are not enough for a reliable preprocessing step.

The Day 6 validator requires:

- A 2D matrix with at least one row and one feature.
- Real numeric values (not strings, booleans, or complex values).
- Finite values only; no NaN or positive/negative infinity.
- An optional expected feature count for checking consistency between datasets.

The returned matrix is float64 so later statistical operations have a clear numeric type. The original input must not be modified.

## 2. Compute one statistic per feature

For a matrix with shape `(n_samples, n_features)`, compute along `axis=0`:

`mean.shape == (n_features,)`

`std.shape == (n_features,)`

The standard deviation in this exercise is the population standard deviation (`ddof=0`), which is NumPy's default.

Each feature is processed independently. For example:

```python
features = np.array([
    [1.0, 10.0],
    [3.0, 14.0],
    [5.0, 18.0],
])

mean = features.mean(axis=0)  # [3, 14]
```

## 3. Normalize with broadcasting

A common normalization formula is:

`normalized = (features - mean) / std`

If the feature matrix is shaped `(n_samples, n_features)`, and both `mean` and `std` are shaped `(n_features,)`, NumPy broadcasts the statistics across all samples.

Do not add a Python loop over rows or features; use broadcasting.

### What if a feature is constant?

A constant feature has standard deviation zero. Dividing by zero would produce invalid results.

For this exercise, the normalization function substitutes a denominator of `1.0` where `std == 0`. For data used to compute those statistics, the constant feature becomes all zeros. If a later dataset differs from the fitted constant value, its deviation from the fitted mean remains visible rather than being divided by zero.

## 4. Keep fitting and transforming conceptually separate

In real preprocessing, statistics should usually be computed on the training split only. Reuse the resulting `mean` and `std` to normalize validation and test data. Do not compute new statistics from validation/test data when you need a consistent transformation; doing so can leak information into the evaluation process.

The separate `compute_feature_stats` and `normalize_features` functions let you apply this pattern. `preprocess_features` is a convenience pipeline that validates a matrix, calculates statistics from that matrix, and returns the normalized result and the fitted statistics.

## 5. Exercises

Implement the four functions in `day06.py`:

1. `validate_feature_matrix(features, expected_n_features=None)`
2. `compute_feature_stats(features)`
3. `normalize_features(features, mean, std)`
4. `preprocess_features(features)`

Requirements:

- Raise `ValueError` for invalid shapes, non-numeric values, non-finite values, or mismatched statistics.
- Use float64 for calculations.
- Preserve input arrays.
- Return `(normalized, mean, std)` from the full pipeline.
- Keep the supplied function signatures.

## 6. Run the tests

From the repository root:

```bash
python3.14 -m unittest week03_numpy.day06.test_day06 -v
```

The starter functions contain TODOs, so the tests are expected to fail until the implementation is complete.

## 7. Before coding, predict

Given:

```python
features = np.array([
    [1.0, 5.0],
    [2.0, 5.0],
    [3.0, 5.0],
])
```

1. What are the feature means?
2. What are the population standard deviations?
3. What happens to the second feature during normalization?
4. Why should the code avoid dividing by its zero standard deviation?
5. If a test sample has a second feature value of `8.0`, and the fitted mean/std are `5.0` and `0.0`, what value will the specified safe-denominator rule produce?
