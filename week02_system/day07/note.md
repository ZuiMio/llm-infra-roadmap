# Day 07 — Training Log Analyzer

## Concepts

- What does analyze_lines() count?
- Why is analyze_file() separate from analyze_lines()?
- Why should aggregate_reports() contain no file I/O?
- Why do we catch OSError around file reading?
- What belongs in a JSON report?

## Test Results

- Command used:
- Unit test result:
- Sample A totals:
- Sample B totals:
- Combined totals:

## Debugging Notes

- Bug or confusing behavior:
- Root cause:
- Fix:
- What I learned:

## Reflection

- Which functions can be reused in another project?
- How does the CLI communicate failure to a shell script?
- Which test was most useful, and why?


python3.14 -m week02_system.day07.log_analyzer \
  week02_system/day07/input/trainer_a.log \
  week02_system/day07/input/trainer_b.log \
  --output week02_system/day07/output/report.json \
  --log-level INFO

git status

git diff

git add week02_system/day07

git diff --cached

git commit -m "Complete Week 02 Day 07 log analyzer"

git push -u origin feature/day07-training-log-analyzer