# Week 03 Plus — NumPy Neural-Network Math and Tiny Classifier

This folder extends the original seven daily NumPy exercises to close the remaining Week 03 outcomes from the 12-week plan. Keep the original Day 01–07 work; this supplement extends it rather than replacing it.

## Gap audit against the 12-week PDF

The original Week 03 daily tasks already cover much of the NumPy foundation: memory/views, indexing, broadcasting, reductions, matrix multiplication, feature preprocessing, batched inference, a basic ReLU, and a linear forward pass. Comparing the existing tasks with the PDF's Week 03 exit criteria shows that several outcomes still need explicit, integrated evidence:

- A complete set of Linear, ReLU, Softmax, CrossEntropy, and LayerNorm operations with shape checks.
- A `tiny_nn` experiment that trains a linear classifier on 2D synthetic data.
- Manually derived and implemented classifier gradients; an MLP is optional.
- Tests that verify each softmax row sums to approximately 1 and that large-magnitude logits stay numerically stable.
- A visible record of the C++ side track introduced in Week 03.

These supplemental files cover the missing Python/NumPy items with a runnable reference implementation, tests, and an experiment script. Use the notes to explain the equations rather than treating passing tests as the only goal.

## Files

- `math_ops.py`: Linear, ReLU, numerically stable Softmax, cross-entropy from logits, LayerNorm.
- `tiny_nn.py`: deterministic 2D three-class dataset, explicit cross-entropy gradients, prediction, and manual gradient descent.
- `test_math_ops.py`: shapes, known outputs, broadcasting, large-magnitude stability, LayerNorm behavior, and invalid inputs.
- `test_tiny_nn.py`: finite-difference checks of manual gradients and an end-to-end training check.
- `run_experiment.py`: compact demo that prints shapes, softmax row sums, large-logit cross-entropy, LayerNorm statistics, loss change, and accuracy.
- `cpp/README.md` and `cpp/linear_forward.cpp`: introductory C++ side exercise using vectors and references.

## Start here

From the repository root:

~~~bash
python -m unittest week03_numpy.Plus.test_math_ops -v
python -m unittest week03_numpy.Plus.test_tiny_nn -v
python -m week03_numpy.Plus.run_experiment
~~~

No PyTorch is required for this folder. It uses NumPy to keep the forward computation and gradients visible.

## Suggested study sequence

### Plus 1 — Neural-network operations

Read and trace `linear_forward`, `relu`, `softmax`, `cross_entropy_from_logits`, and `layer_norm`. For each function, write down input shape, output shape, reduction axis, and numerical-stability decision.

### Plus 2 — Manual gradient derivation

For a linear classifier:

~~~text
logits = X @ W + b
P = softmax(logits)
L = cross_entropy_from_logits(logits, labels)
~~~

For each sample, subtract 1 from the probability of its correct class, then average over the batch:

~~~text
d_logits = P
d_logits[row, label[row]] -= 1
d_logits /= batch_size

dW = X.T @ d_logits
db = sum(d_logits, axis=0)
~~~

Derive why these formulas follow from softmax plus cross-entropy. The implementation calculates gradients directly; it does not use automatic differentiation.

The gradient test compares your analytical gradient implementation with central finite differences. This is a numerical check, not a replacement for deriving the equations.

### Plus 3 — Train and inspect

Run the experiment and confirm:
- The shapes match the documented contracts.
- Softmax produces finite probabilities and each row sums to 1, even with logits around 10,000.
- Cross-entropy remains finite for large-magnitude logits.
- LayerNorm preserves the input shape and normalizes over the final feature axis.
- The classifier loss decreases and the synthetic dataset is classified accurately.

### Plus 4 — C++ side track

Read `cpp/README.md`, compile and run the sample. It reinforces function arguments, `std::vector<float>`, const references, mutable references, and shape checks.

## Definition of Done

- [ ] All tests in both Plus test modules pass.
- [ ] Can explain every function's input/output shape without running it.
- [ ] Can explain why subtracting the maximum logit stabilizes softmax.
- [ ] Can derive the manual `dW` and `db` formulas.
- [ ] Finite-difference tests agree with analytical gradients.
- [ ] The classifier's final loss is lower than its initial loss.
- [ ] Can describe the dataset, parameters, loss history, and accuracy printed by the experiment.
- [ ] Compile and run the C++ side exercise, or record why the compiler is unavailable.

## Limits

This is a small educational reference implementation, not a production neural-network library. It uses a linear classifier rather than an MLP, does not implement backpropagation through LayerNorm/ReLU, and does not cover GPU execution. These are deliberate scope limits; the immediate goal is to close the specified Week 03 learning outcomes before progressing further.
