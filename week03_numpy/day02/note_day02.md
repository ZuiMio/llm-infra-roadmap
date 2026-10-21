# Week 03 — Day 02 Notes

## Predictions Before Running

For `array = np.array([10, 20, 30, 40, 50])`:

- What does `array[[0, 2, 4]]` return?
- What does `array[array > 25]` return?
- Do either of these results share memory with `array`?

## Integer-Array Indexing

- What shape does the selected result have?
- What happens if the index array is two-dimensional?
- What happens if the indices are floats?
- Does modifying the result change the source array?

## Boolean Masks

- What dtype must a boolean mask have?
- What must its shape be relative to the source array?
- How can I build a mask for values greater than zero?
- Why does selecting with a mask return a copy?

## Conditional Replacement

- Why should I copy the input before assigning through a mask?
- Does a value equal to the threshold get replaced?
- Does the function return independent memory when no values match?

## Paired Coordinates

- For `matrix[rows, cols]`, how are rows and columns paired?
- Why does selecting two coordinate pairs return a 1D result?
- How would selecting all combinations of rows and columns differ?

## Test Results

- Command used:
- Number of tests passed:
- Any failing test:

## Debugging Notes

- Bug or confusing behavior:
- Root cause:
- Fix:
- What I learned:
