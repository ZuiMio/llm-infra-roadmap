# Week 02 — Day 07: Training Log Analyzer

## Project Goal

Build a small command-line tool that analyzes one or more training log files. This capstone combines the Week 02 topics:

- Terminal paths and file operations
- Shell-style log analysis concepts
- Git branches and Pull Requests
- Python modules and file I/O
- argparse, logging, exceptions, and unittest

Use the Python standard library only. Do not install another package for this project.

## Input Data

Two sample logs are provided:

- input/trainer_a.log
- input/trainer_b.log

Run commands from the repository root unless a command explicitly says otherwise.

## Requirements

### 1. Implement analyze_lines(lines)

Return a dictionary with exactly these keys:

- total_lines: number of lines, including blank lines.
- error_count: number of lines containing the word ERROR, case-insensitively.
- warning_count: number of lines containing the word WARNING, case-insensitively.

Count each matching line once. A line containing both ERROR and WARNING contributes to both counts. Do not count occurrences within a line separately.

Example contract:

~~~python
analyze_lines([
    "INFO started\n",
    "error: out of memory\n",
    "WARNING: slow batch\n",
])
# Expected:
# {"total_lines": 3, "error_count": 1, "warning_count": 1}
~~~

An empty iterable should return zero for all three counters.

### 2. Implement analyze_file(path)

- Accept a string or a pathlib.Path.
- Open the file as UTF-8 text.
- Reuse analyze_lines() rather than duplicating its counting logic.
- Let file-related OSError exceptions propagate to the caller.

### 3. Implement aggregate_reports(reports)

Input is a list of report dictionaries. Each report contains a file field and the three counters.

Return a dictionary containing the summed counters only:

~~~python
{
    "total_lines": ...,
    "error_count": ...,
    "warning_count": ...,
}
~~~

For an empty list, return zeros for all counters.

### 4. Implement write_json_report(reports, total, output_path)

Write a UTF-8 JSON document with two top-level keys:

- files: the per-file reports.
- total: the aggregate counters.

Use an indentation of 2 spaces. Create the output parent directory if it does not exist. Do not silently ignore write errors.

### 5. Complete the CLI

The argument parser is provided. It must support:

- One or more positional log file paths.
- Optional --output PATH to save a JSON report.
- Optional --log-level with DEBUG, INFO, WARNING, or ERROR; default INFO.

The existing main() scaffold shows the intended control flow. Complete any TODOs and make sure that:

- Each successfully read file gets its own report.
- A missing or unreadable file is logged, but does not prevent other input files from being analyzed.
- Per-file counts and a TOTAL summary are printed.
- If --output is supplied, the JSON report is written.
- Return exit code 0 if all requested files were analyzed, 1 if any file failed or the report could not be written.
- If no input file can be analyzed, return 1.

### 6. Complete and run the unit tests

The tests in test_log_analyzer.py define expected behavior. Add at least one additional test of your own. Recommended edge cases:

- Mixed uppercase and lowercase markers
- Blank lines
- A line containing both ERROR and WARNING
- Empty files
- One missing file alongside one valid file

## Run the Program

Analyze one file:

~~~bash
python -m week02_system.day07.log_analyzer week02_system/day07/input/trainer_a.log
~~~

Analyze multiple files and write JSON:

~~~bash
python -m week02_system.day07.log_analyzer \
  week02_system/day07/input/trainer_a.log \
  week02_system/day07/input/trainer_b.log \
  --output week02_system/day07/output/report.json \
  --log-level INFO
~~~

Run tests:

~~~bash
python -m unittest week02_system.day07.test_log_analyzer -v
~~~

Expected sample-data totals:

- Total lines: 14
- ERROR lines: 3
- WARNING lines: 3

The tests must be passing before you consider the project complete.

## Git Workflow

Use the branch feature/day07-training-log-analyzer.

After implementation:

~~~bash
git status
git diff
git add week02_system/day07
git diff --cached
git commit -m "Complete Week 02 Day 07 log analyzer"
git push origin feature/day07-training-log-analyzer
~~~

Open a Pull Request into main, review the changed files, and merge only after the tests pass.

## Definition of Done

- [ ] All TODOs are implemented.
- [ ] Unit tests pass.
- [ ] Both sample logs are analyzed correctly.
- [ ] JSON output is valid and contains per-file and aggregate counts.
- [ ] Error handling works when one input file is missing.
- [ ] A note describing key concepts and one bug/lesson is completed.
- [ ] A feature branch and Pull Request are used.
