
parser.add_argument("--times", ...)

nargs="+" 表示接收一个或多个参数。

type=float 表示尝试把输入转换为浮点数。

logging.basicConfig(...) 用于配置日志。

LOGGER.error(...) 用于记录错误。

LOGGER.info(...) 用于记录正常运行信息。




---
### 1. logging
`format="%(levelname)s:%(name)s:%(message)s"`

它是专门用来**给日志排版**的“模板”。
```python
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # 在这里配置了日志模板
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",  # <--- 就是这里！
    )
    
    # 后面打印的日志，都会严格按照这个模板来排版
    LOGGER.debug("Received step durations: %s", args.times)

```

### 2. 运行后（实际效果）


```text
DEBUG:week02_system.day06.app:Received step durations: [0.8, 1.2]

```

### 3. 模板里的每个占位符分别代表什么？

对照上面那行真实的输出，模板里的三个部分是这样一一对应拼起来的：

1. **`%(levelname)s`** = **`DEBUG`**（代表日志的严重级别）。
2. **`%(name)s`** = **`week02_system.day06.app`**（代表哪个模块/文件打出的这条日志，也就是 `__name__`）。
3. **`%(message)s`** = **`Received step durations: [0.8, 1.2]`**（代表你传进去的具体日志内容）。
4. 中间的冒号 **`:`** 就是模板里写死的隔断符号。

---

### 1. 核心计算指标
* **Step Count (`count`)**：训练步数总和。
* **Mean Step Time (`mean_seconds`)**：步数时间的算术平均值。
* **Max Step Time (`max_seconds`)**：单步耗时的最大值。

### 2. 防御性校验规则 (Validation)
为防止非法输入破坏后续流程，必须进行前置校验：
* **空输入校验**：若传入空列表，必须抛出 `ValueError`。
* **非负数校验**：耗时、步数等必须为非负数（`>= 0`），出现负数抛出 `ValueError`。
* **非有限数校验**：防止传入 `NaN` 或无穷大（`float('inf')`），触发异常拦截。

---


### 1. 参数设计的最佳实践
* **短选项与长选项结合**：将简写和全称绑定到同一个参数上：
```python
  parser.add_argument("-t", "--times", nargs="+", type=float, required=True, help="Training step durations in seconds.")

```

* `type=float`：自动将输入转换为浮点数。
* `nargs='+'`：强制要求传入 **1 个或多个值**，并自动打包成 Python 列表。
* `required=True`：将可选参数变为**必填项**，漏传时自动触发退出。
* **日志级别控制**：
```python
parser.add_argument("--log-level", default="INFO", help="Logging level.")

```



---

## 工业级日志规范

在多模块项目中，摒弃零散的 `print()`，使用结构化的 Logger：

```python
import logging

LOGGER = logging.getLogger(__name__)

```

* **妙用 `__name__`**：
* 直接运行脚本时，`__name__` 为 `"__main__"`。
* 作为模块被导入时，`__name__` 自动转化为包路径（如 `"week02_system.day06.app"`），实现精准的日志来源追踪。



---

## 高级单元测试与上下文管理器 (`unittest`)

为保证代码的鲁棒性，编写了全覆盖的自动化单元测试。其中核心运用了 Python 的上下文管理器（`with / as`）。

### 1. 流量拦截与标准流重定向

* **`StringIO`**：在内存中创建虚拟文本容器。
* **`redirect_stdout` & `redirect_stderr`**：临时劫持屏幕输出与错误提示，保持测试终端干净，同时便于捕获和断言。
```python
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

stdout = StringIO()
with redirect_stdout(stdout):
    main(["--times", "1.0", "2.0"])

```



### 2. 日志捕获断言 (`self.assertLogs`)

验证程序是否在正确的位置记录了特定级别的日志信息：

```python
with self.assertLogs("week02_system.day06.app", level="ERROR") as captured:
    main(["--times", "0.8", "-0.3"])

self.assertIn("non-negative", "\n".join(captured.output))

```

### 3. 异常与退出码验证 (`self.assertRaises`)

我断言（要求）接下来运行的代码必须抛出 SystemExit 异常, 捕获异常。
：

```python
with self.assertRaises(SystemExit) as context:
    build_parser().parse_args([])

self.assertEqual(context.exception.code, 2)

```
验证当传入非法参数时，`argparse` 是否触发 `SystemExit` 且退出码为标准错误码 `2`

### 4. 断言（Assertion）语句

```python
self.assertIn("non-negative", ...)

```
这是 unittest 提供的一个断言方法，“我断言（要求）"non-negative"必须包含在第二个参数（那个长字符串日志）里面。”

如果包含了，测试通过；如果不包含，测试就会直接报错失败（AssertionError）。

