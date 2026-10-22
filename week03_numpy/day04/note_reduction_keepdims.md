# Week 03 — Day 04 Notes

## Shape Predictions Before Running

For `values.shape == (2, 3)`:

- What shape does `np.sum(values, axis=0)` return?
- What shape does `np.sum(values, axis=1)` return?
- How do the shapes change when `keepdims=True`?
- What does `axis=None` mean?

## Reduction Axes

- Which axis represents samples in a matrix shaped `(n_samples, n_features)`?
- Why does `axis=0` compute one statistic per feature?
- Why does `axis=1` compute one statistic per sample?
- How do negative axis indices work?

## keepdims and Broadcasting

- What is the difference between shapes `(n_features,)` and `(1, n_features)`?
- Why is `keepdims=True` useful when subtracting means from a matrix?
- Which dimension remains with size 1 after a reduction?

## Numerical Precision

- What is an accumulator dtype?
- Why can a float32 reduction lose small values when large and small values are combined?
- How does requesting a float64 accumulator change the calculation?
- Does float64 eliminate every possible floating-point error? Explain.

## Feature Centering and Error Metrics

- Why should centered features have means close to zero along axis 0?
- Why should predictions and targets have matching shapes for MAE?
- Why convert values before subtracting and taking absolute values?
- Why must empty inputs be handled explicitly?

## Test Results

- Command used:
- Number of tests passed:
- Any failing test:

## Debugging Notes

- Bug or confusing behavior:
- Root cause:
- Fix:
- What I learned:
