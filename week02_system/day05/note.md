
### 2. Exercise A：创建虚拟环境

假设项目 A 需要 NumPy 的某个版本，项目 B 依赖另一个版本。如果都安装到同一个 Python 环境中，就可能产生依赖冲突。
虚拟环境为项目提供相对独立的 Python 解释器和第三方包安装位置。它通常放在项目目录下的 .venv/ 中，但不应该把整个 .venv/ 提交到 GitHub。Python 官方文档也建议通过依赖文件来重建环境，而不是移动或共享虚拟环境本身。

git switch -c feature/day05-python-engineering

python3.14 --version

python3.14 -m venv .venv

source .venv/bin/activate

which python

python --version

python -m pip --version

deactivate

### 3. Exercise B：管理项目依赖

numpy>=2.0,<3.0

echo "numpy>=2.0,<3.0" > requirements.txt

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m pip show numpy

4. Exercise C：编写 metrics.py

touch metrics.py

python -c "import app; print('app imported successfully')"

python -c "from metrics import summarize_step_times; print(summarize_step_times([0.5, 1.5]))"




