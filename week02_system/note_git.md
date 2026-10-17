
① Working Tree — 工作区

② Staging Area — 暂存区

③ Local Repository — 本地仓库

④ Remote Repository — 远程仓库

## Git 版本控制工作流

- `git status`：查看当前仓库的状态（哪些文件被修改或新增）。
- `git diff -- week02_system/day02/note.md`：查看指定文件的具体修改差异。
- `git add week02_system/day02`：将指定目录下的改动添加到暂存区。
- `git commit -m "..."`：将暂存区的改动正式提交到本地仓库，并附带注释信息。
- `git push origin main`：将本地仓库的提交推送到远端主分支（如 GitHub）。


2. Exercise A：检查仓库状态
- `pwd`：当前终端所在的目录。
- `git rev-parse --show-toplevel`：当前 Git 仓库的根目录。
- `git branch --show-current`：当前分支名称。
- `git status`：当前文件有没有未提交的修改。
- `git status` --short：快速查看修改。
- `git remote -v`：远程仓库的地址。

3. Exercise B：创建一个练习文件
- `mkdir -p week02_system/day03`

- `cat > week02_system/day03/practice.txt <<'EOF'
job=trainer
event=start
status=running
EOF`

4. Exercise C：理解 `git add` 和 `git diff`
- `git add week02_system/day03/practice.txt`
- `git diff` 工作区与暂存区之间的差异
- `git diff --cached` 暂存区与最近一次提交之间的差异

5. Exercise D：创建提交并查看历史
- `git commit -m "......"` 创建一条新的本地提交记录。
- `git log --oneline -5` 显示最近五条提交，每条通常只占一行。
- `git status` 用于确认是否还有未提交的修改。

6. Exercise E：修改已有文件
- `printf 'step=1\n' >> week02_system/day03/practice.txt`
- `git diff -- week02_system/day03/practice.txt`
- `git diff --cached -- week02_system/day03/practice.txt` 暂时没有显示这次新增内容，因为它还没有加入暂存区。

7. Bonus：如何撤销修改？
- `git restore --staged week02_system/day03/practice.txt` 这会把文件从暂存区移出，但保留工作区里的修改。
- `git restore -- week02_system/day03/practice.txt` 撤销工作区中尚未暂存的修改

8. 最后：推送到 GitHub
- `git status`
- `git push origin main`