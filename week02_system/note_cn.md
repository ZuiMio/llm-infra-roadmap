## 📁 一、 目录与文件基础操作

- `pwd`：显示当前所在的绝对路径（Print Working Directory）。
- `ls` / `ls -la`：列出当前目录下的文件。`-la` 可以显示包括隐藏文件在内的所有详细信息。
- `mkdir -p week02_system/input`：递归创建多级目录，如果父目录不存在会自动连同创建。
- `cd week02_system`：切换当前工作目录到指定文件夹。
- `find . -type f`：在当前目录下查找所有的普通文件。
- `cp input/train.log output/train-copy.log`：复制文件，将源文件复制并重命名/存放到目标路径。

---

## 📖 二、 文件查看与内容统计

- `cat input/train.log`：在终端一次性打印出文件的全部内容。
- `head -n 2 input/train.log`：查看文件的前 2 行。
- `tail -n 2 input/train.log`：查看文件的最后 2 行。
- `wc -l input/train.log`：统计文件的总行数（包含文件名）。
- `wc -l < input/train.log`：纯数字统计行数（通过输入重定向，不显示文件名）。

---

## 🔍 三、 文本搜索、过滤与重定向

- `printf ... > ...`：向文件中快速写入多行文本。
- `grep 'ERROR' input/train.log`：在文件中搜索包含 `'ERROR'` 关键词的行并打印。
- `grep -c 'ERROR' input/train.log`：直接统计文件中包含 `'ERROR'` 的行数。
- `grep 'ERROR' input/train.log | wc -l`：通过管道符将搜索结果传给 `wc -l` 来统计错误行数。
- 重定向规则 (`>` / `>>` / `<`)：
  - `>`：**覆盖**写入文件（如果文件已存在会清空原内容）。
  - `>>`：向文件**末尾追加**内容。
  - `<`：输入重定向（把文件内容“喂”给左侧的命令）。
- `grep 'ERROR' input/train.log > output/errors.log`：将搜索到的错误结果保存到新文件中。
- `grep 'ERROR' input/train.log | tee output/errors.log`：一边在屏幕上打印结果，一边写入文件。
- `grep -E 'ERROR|WARNING' input/train.log > output/attention.log`：使用 `-E` 开启扩展正则，同时搜索多个条件（`ERROR` 或 `WARNING`）。

---

## ⚙️ 四、 环境变量与 Shell 进阶

- `echo "$HOME" / "$SHELL" / "$PATH"`：分别查看用户根目录、当前 Shell 解释器、系统命令查找路径。
- `printf '%s\n' "$PATH" | tr ':' '\n'`：将连在一起的 `$PATH` 路径拆开，让每个路径独占一行显示。
- `command -v python3`：检查系统中是否存在 `python3` 命令，并返回其路径。
- `python3 --version`：查看 Python 3 的具体版本号。
- `cat > input/train.log <<'EOF' ... EOF`：使用 Here Document 语法在终端直接输入并写入多行文本。
- `export LOG_FILE="input/train.log"`：定义并导出环境变量，供后续命令复用。
- `bash -c 'printf "%s\n" "$LOG_FILE"'`：在子 Shell 中验证环境变量的继承性。
- `grep -q 'ERROR' "$LOG_FILE"`：静默搜索（Quiet 模式），不在屏幕打印，只通过状态码反馈。
- `echo $?`：查看上一条命令的退出状态码（`0` 表示成功/找到，`1` 表示失败/未找到）。
- `grep -q 'ERROR' "$LOG_FILE" && echo "Errors found"`：短路求职，如果找到了错误（返回 0），则执行后面的打印命令。

---

## 🌿 五、 Git 版本控制工作流

- `git status`：查看当前仓库的状态（哪些文件被修改或新增）。
- `git diff -- week02_system/day02/note.md`：查看指定文件的具体修改差异。
- `git add week02_system/day02`：将指定目录下的改动添加到暂存区。
- `git commit -m "..."`：将暂存区的改动正式提交到本地仓库，并附带注释信息。
- `git push origin main`：将本地仓库的提交推送到远端主分支（如 GitHub）。