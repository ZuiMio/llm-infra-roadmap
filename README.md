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

- [Week 3 Plus catch-up guide](week03_numpy/Plus/README.md) — supplemental neural-network math, manual gradients, synthetic classifier, and C++ side exercise.
- [Week 4 learning guide](week04_pytorch/README.md)
- [Day 1 code](week04_pytorch/day01/day01.py) · [Day 1 tests](week04_pytorch/day01/test_day01.py) · [Day 1 notes](week04_pytorch/day01/note_day01.md)
- [Day 2 code](week04_pytorch/day02/day02.py) · [Day 2 tests](week04_pytorch/day02/test_day02.py) · [Day 2 notes](week04_pytorch/day02/note_day02.md)
- [Day 3 code](week04_pytorch/day03/day03.py) · [Day 3 tests](week04_pytorch/day03/test_day03.py) · [Day 3 notes](week04_pytorch/day03/note_day03.md)
- [Day 4 starter code](week04_pytorch/day04/day04.py) · [Day 4 tests](week04_pytorch/day04/test_day04.py) · [Day 4 notes](week04_pytorch/day04/note_day04.md)
- [Day 5 starter code](week04_pytorch/day05/day05.py) · [Day 5 tests](week04_pytorch/day05/test_day05.py) · [Day 5 notes](week04_pytorch/day05/note_day05.md)
- [Day 6 starter code](week04_pytorch/day06/day06.py) · [Day 6 tests](week04_pytorch/day06/test_day06.py) · [Day 6 notes](week04_pytorch/day06/note_day06.md)
- [Day 7 review starter](week04_pytorch/day07/weekly_review.py) · [Day 7 tests](week04_pytorch/day07/test_weekly_review.py) · [Weekly review notes](week04_pytorch/day07/note_weekly_review.md)

Run the Day 5–7 tests from the repository root:

~~~bash
python -m unittest week04_pytorch.day05.test_day05 -v
python -m unittest week04_pytorch.day06.test_day06 -v
python -m unittest week04_pytorch.day07.test_weekly_review -v
~~~

Day 1 covers scalar differentiation; Day 2 covers leaf tensors and gradient storage; Day 3 covers gradient accumulation and resetting; Day 4 covers linear forward and tensor shapes; Day 5 covers MSE and parameter gradients; Day 6 covers manual updates and loss reduction; Day 7 rebuilds a tiny training loop from scratch. The Day 4–7 exercises include starter functions to implement. The Week 3 materials remain available for review.

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
