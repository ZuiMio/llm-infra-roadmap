


## After set up. Now need to install venv and pytorch

`cd ~/projects/llm-infra-roadmap`

`python3 --version`

`pwd`

`sudo apt update`

`sudo apt install -y python3-venv python3-pip`

`rm -rf .venv`  # 删掉旧虚拟环境

`python3 -m venv .venv` # 重新创建新虚拟环境

`source .venv/bin/activate`

`python -m pip install --upgrade pip`

```
python -m pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
```
验证 GPU 加速：
```python
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x

    print("Tensor device:", y.device)
    print("Test passed!")
else:
    print("CUDA unavailable. Please check the installation.")
PY
```


## 一、先理解最终的架构

MacBook Air M4

VS Code · 编辑代码 · 查看运行结果

VS Code Remote Tunnels

Windows 11 PC

WSL2 · Ubuntu 24.04 · Python · PyTorch

如有 NVIDIA 显卡，可以进一步使用 CUDA

我们选择 WSL2 + Ubuntu + VS Code Remote Tunnels。WSL2 能直接在 Windows 中运行 Linux 环境，Remote Tunnels 则允许 Mac 通过安全隧道连接，不需要自己配置 SSH 端口。

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn

+1



## 二、第一步：在 Windows 11 安装 Ubuntu

以下命令都在 Windows 11 PC 上执行，不是在 Mac 上执行。

### 1. 安装 WSL2

在 Windows 搜索栏输入 `PowerShell`，右键选择“以管理员身份运行”，执行：

```
wsl --install -d Ubuntu-24.04
```

如果提示需要重启，就重启 Windows。

如果提示找不到这个发行版，可以执行：

```
wsl --list --online
```

查看可用发行版，再选择 Ubuntu 安装。微软官方文档说明，`wsl --install` 可以完成 WSL 所需组件与 Linux 发行版的安装。

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn



### 2. 初始化 Ubuntu

安装完成后，在 Windows 开始菜单搜索 `Ubuntu 24.04` 并打开。

第一次启动时，系统会要求你创建 Linux 用户名和密码。这里的用户名、密码是 Ubuntu 内部的，不需要与你的 Windows 账号相同。

然后执行：

```
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  git \
  curl \
  wget \
  ca-certificates \
  build-essential \
  python3 \
  python3-venv \
  python3-pip
```

检查是否正常：

```
python3 --version
git --version
uname -a
```

再回到 Windows PowerShell，检查 WSL 版本：

```
wsl --list --verbose
```

正常情况下，Ubuntu 对应的 `VERSION` 应该是 `2`。

## 三、第二步：创建你的 Linux 学习项目

在 Ubuntu 终端中执行：

```
mkdir -p ~/projects/llm-infra-roadmap
cd ~/projects/llm-infra-roadmap
```

这里建议把项目保存在 Ubuntu 自己的 Linux 文件系统中，而不是直接放在 `/mnt/c/Users/...` 下。这种方式更适合 Linux 开发工具处理大量文件。

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn



接下来，我们要让 Mac 可以直接访问这个项目。

## 四、第三步：让 Ubuntu 提供远程隧道

为了避免你之前遇到的“启用此会话 / 作为服务安装”选项造成混淆，我们先用命令行建立隧道，验证成功后再考虑后台服务。

### 1. 在 Ubuntu 安装 VS Code CLI

在 Ubuntu 终端执行：

```
mkdir -p ~/.local/bin
cd /tmp

curl -L \
  'https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64' \
  -o vscode_cli.tar.gz

tar -xzf vscode_cli.tar.gz

install -m 755 code ~/.local/bin/code
```

这里下载的是 Linux x64 版 VS Code CLI，不是 macOS 版本。假如你的 Windows PC 是 ARM 架构，则应改为对应的 Linux ARM64 版本。官方支持使用独立 CLI 在没有图形界面的远程机器上创建隧道。

[image](https://www.google.com/s2/favicons?domain=https://code.visualstudio.com\&sz=32)

Visual Studio Code



### 2. 启动隧道

先回到项目目录：

```
cd ~/projects/llm-infra-roadmap
~/.local/bin/code tunnel --accept-server-license-terms
```

第一次执行时，按照终端提示使用 GitHub 账号完成身份验证。

启动成功后，终端会显示隧道相关信息。先不要关闭这个终端，也不要退出 Ubuntu。 我们需要先确认 Mac 能连接成功。

## 五、第四步：在 Mac 上连接 Ubuntu

打开 Mac 上的 VS Code。

### 1. 安装扩展

进入 Extensions，搜索并安装：

`Remote - Tunnels`

这是 VS Code 官方的远程隧道扩展。

### 2. 连接远程计算机

按 `Command + Shift + P` 打开命令面板，输入：

```
Remote Tunnels: Connect to Tunnel
```

然后选择刚才在 Ubuntu 中创建的隧道，并按提示登录相同的账号。

### 3. 打开项目文件夹

连接成功后，在远程 VS Code 窗口中打开：

```
/home/你的Linux用户名/projects/llm-infra-roadmap
```

此时，Mac 上的 VS Code 编辑的是 Ubuntu 中的文件。你打开集成终端时，运行的也是远程 Linux 环境，而不是 macOS。

官方连接流程与详细说明可参考：[VS Code Remote Tunnels](https://code.visualstudio.com/docs/remote/tunnels?utm_source=chatgpt.com)。

[image](https://www.google.com/s2/favicons?domain=https://code.visualstudio.com\&sz=32)

Visual Studio Code



## 六、第五步：验证代码运行位置

在 Mac 的 VS Code 远程窗口里，新建 `test_remote.py`：

```
import platformimport sysprint("Operating system:", platform.system())print("Python path:", sys.executable)print("Python version:", sys.version)
```

运行：

```
python3 test_remote.py
```

期望看到的结果类似：

```
Operating system: Linux
Python path: /usr/bin/python3
Python version: 3.12...
```

具体 Python 版本以实际安装结果为准。

出现 `Linux` 就说明你已经实现了：在 Mac 上写代码，在 Windows PC 的 Ubuntu 环境中执行代码。

## 七、第六步：为 PyTorch 和 GPU 做准备

建议在 Ubuntu 项目目录中创建独立的 Python 虚拟环境：

```
cd ~/projects/llm-infra-roadmap

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch
```

对于后续学习 Megatron 等训练基础设施，先使用 Linux 上的 Python 3.12 环境通常比急着统一到 Python 3.14 更稳妥。

如果 Windows PC 有 NVIDIA 显卡，先在 Ubuntu 里执行：

```
nvidia-smi
```

如果能够识别 GPU，再运行：

```
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

需要特别注意：WSL2 的 NVIDIA GPU 支持依赖 Windows 侧安装的兼容 NVIDIA 驱动。不要在 WSL Ubuntu 中再安装普通的 Linux NVIDIA 显卡驱动，否则可能破坏 WSL 提供的 GPU 驱动接口。

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

CUDA on WSL 13.4 documentation



## 八、最后两个注意事项

首先，刚开始建议保持运行 `code tunnel` 的终端打开，并确保 Windows PC 不进入睡眠。验证成功后，可以再考虑使用 `code tunnel service install` 将隧道作为服务运行；WSL 的服务配置可能还需要额外设置。

其次，WSL2 很适合学习 Linux、Python、PyTorch 和部分 CUDA 工作流，但它不完全等同于专用的 Linux 多 GPU 训练服务器。等你进入 Megatron、分布式训练和 NCCL 等内容时，还需要根据电脑硬件和具体实验需求决定是否使用真实 Linux 服务器或云 GPU。

现在最关键的是确认你的 Windows 11 PC 配置。

你可以在 Windows 上按 `Ctrl + Shift + Esc` 打开任务管理器，进入“性能”页面，查看 GPU 的名称、显存大小，以及电脑的内存容量。这些信息将决定你能在本地做多大规模的 PyTorch 和 LLM 实验。