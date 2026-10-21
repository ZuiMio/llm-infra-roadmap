# LLM Infra Roadmap

A learning repository for building the foundations needed for LLM training and inference infrastructure.

## Goal

Build the engineering fundamentals required to work toward:

- PyTorch training systems
- CUDA and GPU performance
- Distributed training
- NCCL / DDP / FSDP
- Megatron Core
- SGLang / vLLM
- Post-training infrastructure

## Current Stage

### Week 1 — Python Foundations

- [ ] Day 1: Python syntax, control flow, functions
- [ ] Day 2: list / tuple / dict / set
- [ ] Day 3: functions, arguments, scope, mutable vs immutable
- [ ] Day 4: classes and objects
- [ ] Day 5: exceptions, file I/O, iterators, generators
- [ ] Day 6: NumPy and tensor-shape thinking
- [ ] Day 7: mini project — Linear → ReLU → MSE

### Current Focus — Week 3: NumPy and Tensor Thinking

Day 2 covers advanced indexing, boolean masks, conditional replacement, and copy behavior.

- [Week 3 learning guide](week03_numpy/README.md)
- [Day 2 starter code](week03_numpy/day02/day02.py)
- [Day 2 tests](week03_numpy/day02/test_day02.py)
- [Day 2 notes](week03_numpy/day02/note_day02.md)

Run the Day 2 tests from the repository root:

```bash
python3.14 -m unittest week03_numpy.day02.test_day02 -v
```

## Roadmap

- [ ] Week 1: Python foundations
- [ ] Week 2: Linux, Git, Python engineering
- [ ] Week 3: NumPy and tensor thinking
- [ ] Week 4: PyTorch tensors and autograd
- [ ] Week 5: nn.Module, optimizer, DataLoader
- [ ] Week 6: language modeling basics
- [ ] Week 7: attention from scratch
- [ ] Week 8: multi-head attention and Transformer block
- [ ] Week 9: MiniGPT
- [ ] Week 10: training system features
- [ ] Week 11: profiling and benchmarking
- [ ] Week 12: engineering cleanup and portfolio preparation

## Learning Rule

This repository is for active practice, not copied notes.

For each topic:

1. Read the concept.
2. Write code without copying.
3. Test the code.
4. Record bugs and what caused them.
5. Commit meaningful progress.

## Long-term Direction

Python → PyTorch → Transformer → CUDA → DDP/NCCL → FSDP → Megatron → SGLang → Post-training Infra
