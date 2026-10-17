
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


## Branches and GitHub Workflow
- Branch（分支）：指向一条提交历史的可移动引用，方便你在不同开发线上工作。
- Remote branch（远程分支）：例如 GitHub 上的 origin/main，代表 Git 本地记录的远程分支状态。
- Pull Request（PR）：在 GitHub 上提出将一个分支的变更合并到另一个分支的请求，可以先检查代码再合并。

2. Exercise A：检查并更新 main
- `git status`
- `git branch --show-current`
- `git switch main`	切换到本地的 main 分支
- `git pull origin main`	获取远程 main 的更新并整合到当前 main
- `git branch`	列出本地分支，星号表示当前分支

3. Exercise B：创建自己的开发分支
- `git switch -c feature/day04-git-workflow` 这个命令会创建分支并立即切换过去。
- `git branch --show-current`
- `git status`


4. Exercise C：在新分支中创建笔记
- `mkdir -p week02_system/day04`
- `cat > week02_system/day04/note.md <<'EOF'......EOF`
- `git add week02_system/day04/note.md`
- `git diff --cached`
- `git commit -m "Add Week 02 Day 04 Git workflow notes"`
- `git log --oneline -5`
- `git status`

5. Exercise D：把分支 push 到 GitHub
- `git push -u origin feature/day04-git-workflow` 这里的 -u 用于设置本地分支的上游分支。

6. Exercise E：合并之后同步本地 main
- `git switch main`
- `git pull origin main`
- `git log --oneline -5`
- `git status`

如果要删除已经合并的本地开发分支，可以先执行：
- `git branch --merged`
- `git branch -d feature/day04-git-workflow`

7. 今天还必须理解的两个命令

`git fetch` 和 `git pull` 有什么区别？
- `git fetch origin`：获取远程仓库的最新信息，更新本地的远程跟踪分支，但一般不会自动把这些变更合入当前工作分支。
- `git pull origin main`：获取远程更新，并将指定分支的更新整合到当前分支。整合方式会受 Git 配置影响。


`git push` 和 `Pull Request` 有什么区别？
- `git push` 把本地提交上传到远程仓库。
- `Pull Request` 是在 GitHub 上请求将来源分支的变更合并到目标分支。

你可以先 push 开发分支，再创建 PR；整个过程无需直接向 main 提交新内容。

