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

## Current Focus — Week 4: PyTorch Tensors and Autograd

The 12-week plan schedules Week 4 for 2026-10-27 to 2026-11-02. Study is starting on 2026-10-24, a few days early; keep the planned topic order and adjust dates if necessary.

- [Week 4 learning guide](week04_pytorch/README.md)
- [Day 1 starter code](week04_pytorch/day01/day01.py)
- [Day 1 tests](week04_pytorch/day01/test_day01.py)
- [Day 1 code](week04_pytorch/day01/day01.py) · [Day 1 tests](week04_pytorch/day01/test_day01.py) · [Day 1 notes](week04_pytorch/day01/note_day01.md)
- [Day 2 starter code](week04_pytorch/day02/day02.py) · [Day 2 tests](week04_pytorch/day02/test_day02.py) · [Day 2 notes](week04_pytorch/day02/note_day02.md)

Run the Day 2 tests from the repository root:

~~~bash
python -m unittest week04_pytorch.day02.test_day02 -v
~~~

Day 1 covers the scalar derivative, a hand-written linear forward pass, MSE loss, and parameter gradients. Day 2 studies leaf tensors, requires_grad, grad_fn, and retained intermediate gradients. The Day 2 starter functions contain TODOs until you implement them. The Week 3 materials remain available for review.

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
