# Day 2 — Shell, Environment Variables, and Log Analysis

## 1. Learning Objectives

- Understand environment variables and `PATH`.
- Use pipes to connect commands.
- Redirect command output to files.
- Search and count log entries.
- Understand command exit statuses.
- Combine commands to perform simple log analysis.

## 2. Environment Variables

Environment variables store configuration values that can be accessed by the shell and programs.

Useful examples:

| Variable | Purpose |
|---|---|
| `HOME` | User's home directory |
| `SHELL` | Path to the user's shell |
| `PATH` | Directories used to locate executable commands |

Inspect the environment:

```bash
echo "$HOME"
echo "$SHELL"
echo "$PATH"
command -v python3
python3 --version
```

Display each directory in `PATH` on a separate line:

```bash
printf '%s\n' "$PATH" | tr ':' '\n'
```

On macOS, `PATH` is typically a colon-separated list of directories. The shell uses it to locate executable commands.

## 3. Pipes and Redirection

### Pipe: `|`

A pipe connects the standard output of one command to the standard input of another command.

Example:

```bash
grep 'ERROR' input/train.log | wc -l
```

This searches for lines containing `ERROR` and counts the matching lines.

### Output redirection: `>` and `>>`

```bash
grep 'ERROR' input/train.log > output/errors.log
```

- `>` creates or overwrites the destination file.
- `>>` appends output to the end of the destination file.

Be careful: `>` can overwrite existing content.

### `tee`

`tee` displays command output and saves it to a file at the same time.

```bash
grep 'ERROR' input/train.log | tee output/errors.log
```

## 4. Log Analysis Practice

Create the directories and enter the practice directory.

```bash
mkdir -p week02_system/day02/input
mkdir -p week02_system/day02/output
cd week02_system/day02
```

Create a sample training log:

```bash
cat > input/train.log <<'EOF'
INFO job=trainer event=start
INFO job=trainer event=load_data
WARNING job=trainer event=slow_batch
INFO job=trainer event=step_complete
ERROR job=trainer event=oom
WARNING job=trainer event=retry
INFO job=trainer event=restart
ERROR job=trainer event=timeout
INFO job=trainer event=cleanup
ERROR job=trainer event=oom
EOF
```

### Exercise A — Count log entries

```bash
wc -l < input/train.log
grep -c 'ERROR' input/train.log
grep -c 'WARNING' input/train.log
grep 'ERROR' input/train.log
```

Expected results:

- Total lines: `10`
- Lines containing `ERROR`: `3`
- Lines containing `WARNING`: `2`
- The three error events are `oom`, `timeout`, and `oom`.

`grep -c` counts the number of matching lines, not the number of occurrences of a word within a line.

### Exercise B — Save matching lines

```bash
grep 'ERROR' input/train.log | tee output/errors.log
wc -l < output/errors.log
cat output/errors.log
```

The file `output/errors.log` should contain three lines.

Search for both errors and warnings:

```bash
grep -E 'ERROR|WARNING' input/train.log > output/attention.log
cat output/attention.log
```

The file `output/attention.log` should contain five lines.

`grep -E` supports extended regular expressions. Here, `|` inside the quoted pattern means OR: match `ERROR` or `WARNING`.

### Exercise C — Export an environment variable

```bash
export LOG_FILE="input/train.log"

grep 'ERROR' "$LOG_FILE"
wc -l < "$LOG_FILE"
bash -c 'printf "%s\n" "$LOG_FILE"'
```

`export` makes the variable available to child processes launched by the shell.

Use double quotes around `"$LOG_FILE"` so that the variable is expanded safely, especially when paths contain spaces.

### Exercise D — Command exit status

```bash
grep -q 'ERROR' "$LOG_FILE"
echo $?
```

Expected exit status: `0`, because a matching line exists.

Now search for a missing string:

```bash
grep -q 'NOT_FOUND' "$LOG_FILE"
echo $?
```

Expected exit status: `1`, because no line matches.

For `grep`:

- `0`: at least one match was found.
- `1`: no match was found.
- A value greater than `1`: an error occurred.

`grep -q` suppresses normal matching output. The exit status is useful when deciding what a shell script should do next.

Run:

```bash
grep -q 'ERROR' "$LOG_FILE" && echo "Errors found"
```

The message is displayed only if the left-hand command succeeds.

## 5. Important Concepts

| Concept | Meaning |
|---|---|
| Standard input | Data read by a command |
| Standard output | Normal output produced by a command |
| Pipe `|` | Passes one command's output to another command |
| Input redirection `<` | Reads input from a file |
| Output redirection `>` | Writes output to a file, overwriting it |
| Append redirection `>>` | Appends output to a file |
| Environment variable | A named configuration value |
| Exit status | A numeric result indicating success or failure |
| `&&` | Runs the next command only if the previous command succeeds |

## 6. Key Takeaways

- `PATH` helps the shell locate executable commands.
- A pipe connects commands without needing an intermediate file.
- `>` overwrites, while `>>` appends.
- `tee` is useful when output should be both displayed and saved.
- `grep -c` counts matching lines.
- Exit statuses allow shell scripts to react to command results.
- Log analysis often involves combining simple commands rather than writing a new program.

## 7. Reflection

- What is the difference between `grep 'ERROR' file | wc -l` and `grep -c 'ERROR' file`?
- What is the difference between a pipe and output redirection?
- Why does `grep` return `1` when there is no matching line?
- Why might logs be saved to a separate file instead of only displayed in the terminal?
- Which command combination was the most useful today?
