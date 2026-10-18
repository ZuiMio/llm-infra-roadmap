# Day 1 — Terminal and File System

## 1. Learning Objectives

- Understand the current working directory.
- Navigate directories and manage files from the terminal.
- Read and search text files.
- Understand relative paths and absolute paths.

## 2. Essential Commands

| Command | Purpose |
|---|---|
| `pwd` | Print the current working directory |
| `ls` | List directory contents |
| `ls -la` | Show detailed information, including hidden files |
| `cd` | Change the current directory |
| `mkdir -p` | Create directories, including parent directories |
| `printf` | Format and print text |
| `find` | Find files and directories |
| `cat` | Display file contents |
| `wc -l` | Count lines |
| `grep` | Search for matching lines |
| `cp` | Copy files |
| `head -n 2` | Display the first two lines |
| `tail -n 2` | Display the last two lines |

## 3. Current Working Directory and Paths

The current working directory is the directory in which a command is executed.

- An **absolute path** starts from the root directory.
- A **relative path** is interpreted relative to the current working directory.
- `.` represents the current directory.
- `..` represents the parent directory.
- `~` represents the current user's home directory.

Examples:

```bash
pwd
cd ..
cd ~
```

## 4. Hands-on Practice

touch + 编辑器（最常用，适合写正式内容）
```
touch hello.py            # 创建一个空文件
nano hello.py             # 用 nano 打开编辑（推荐新手）
# 或者
vim hello.py              # 用 vim（会用的自然懂）
# 或者 VS Code
code hello.py
```
echo 一行写入（适合快速写入/追加）

```
# 覆盖写入（文件不存在会创建，存在则清空重写）
echo "print('hello')" > hello.py

# 追加写入（不覆盖原有内容）
echo "print('world')" >> hello.py

# 写多行，用 $'...' 处理换行
echo $'line1\nline2' > test.txt

# 或者用 printf（比 echo 更可控，不带尾部换行）
printf "hello" > note.txt
printf "line1\nline2\n" > note.txt
```

Create a practice directory and a log file.

```bash
mkdir -p week02_system/day01/input
mkdir -p week02_system/day01/output
cd week02_system/day01

printf 'INFO training started\nWARNING slow batch\nERROR out of memory\nERROR timeout\n' > input/train.log
```

Read and inspect the file:

```bash
find . -type f
cat input/train.log
wc -l input/train.log
grep 'ERROR' input/train.log
```

Expected results:

- `find . -type f` lists the files under the current directory.
- `cat input/train.log` displays all four log lines.
- `wc -l input/train.log` reports 4 lines.
- `grep 'ERROR' input/train.log` displays the two lines containing `ERROR`.

Copy and inspect the file:

```bash
cp input/train.log output/train-copy.log
head -n 2 input/train.log
tail -n 2 input/train.log
find . -type f
```

The `output` directory must exist before copying the file.

## 5. Understanding Input and Output

Compare these commands:

```bash
wc -l input/train.log
wc -l < input/train.log
```

Both count the lines in the file. The first command normally displays the line count and filename. The second redirects the file into standard input, so `wc` displays only the line count.

The `<` symbol is input redirection. The `>` symbol is output redirection.

## 6. Key Takeaways

- The current working directory determines how relative paths are interpreted.
- `mkdir -p` is useful when creating project directories.
- `grep` searches file contents rather than filenames.
- `head` and `tail` are useful for quickly inspecting logs.
- A destination directory must exist before copying a file into it.
- Understanding standard input and output makes shell commands easier to combine.

## 7. Reflection

- What took me the longest to understand?
- What happens when I run a command from a different working directory?
- Why should I check whether a destination directory exists before copying a file?
