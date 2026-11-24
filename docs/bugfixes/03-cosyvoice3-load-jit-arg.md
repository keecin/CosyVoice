# Bug 报告 3：CosyVoice3 不支持 `load_jit` 参数（接口不一致，小问题）

## 一句话结论

同一个入口 `AutoModel(...)`，CosyVoice2 可以传 `load_jit=...`，但 CosyVoice3 的构造函数
不接受这个参数；用一种通用的方式启动两个模型时，指定 CosyVoice3 会直接报：

```
TypeError: CosyVoice3.__init__() got an unexpected keyword argument 'load_jit'
```

这是一个**接口不一致的小问题**，我们在调用层做了兼容处理（不需要改模型代码）。

---

## 0. 背景

- `AutoModel(model_dir=...)` 是 CosyVoice 的统一入口，会根据模型目录里是
  `cosyvoice2.yaml` 还是 `cosyvoice3.yaml` 自动选择对应的类（`CosyVoice2` 或 `CosyVoice3`）。
- `load_jit` 是“是否加载 JIT 优化模型”的开关（本项目里我们统一关闭，不启用）。
- `CosyVoice3` 继承自 `CosyVoice2`，但两者构造函数参数并不完全一致。

---

## 1. 现象

用同一段启动代码跑两个模型：

```python
cosyvoice = AutoModel(model_dir=model_dir, load_trt=False, load_jit=False, load_vllm=False, fp16=False)
```

- 跑 CosyVoice2：正常；
- 跑 CosyVoice3：报 `TypeError: CosyVoice3.__init__() got an unexpected keyword argument 'load_jit'`。

---

## 2. 根因：两个类的构造参数不同

`cosyvoice/cli/cosyvoice.py`：

```python
class CosyVoice2:
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        ...

class CosyVoice3(CosyVoice2):
    def __init__(self, model_dir, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        # 注意：没有 load_jit
        ...
```

`AutoModel` 会把 kwargs 原样透传给对应类的构造函数，所以给 CosyVoice3 传 `load_jit` 就报“多余的参数”。

> 说明：这只是接口层面的不一致，不是模型算法的问题；`load_jit` 本身在本项目里也没启用。

---

## 3. 修复（调用层兼容）

在启动脚本 `run_npu.py` 里，按模型类型决定是否传 `load_jit`：

```python
kwargs = dict(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=False)
if not is_v3(model_dir):          # 仅 CosyVoice2 传 load_jit
    kwargs['load_jit'] = False
cosyvoice = AutoModel(**kwargs)
```

其中 `is_v3()` 通过是否存在 `cosyvoice3.yaml` 来判断模型类型。

另一种做法（不改调用方）是给 `CosyVoice3.__init__` 也加上 `load_jit=False` 参数（显式忽略），
但那样会改动上游模型代码；本项目选择“调用层兼容”，改动面更小。

---

## 4. 验证

- CosyVoice3 能正常加载并合成（零样本 10.7 s、双流 4 块共 10.4 s）；
- CosyVoice2 行为不变。

---

## 5. 影响范围

- 仅影响“向 CosyVoice3 传 `load_jit`”的调用方式；不传就不会触发。
- 与硬件、transformers 版本无关；纯 Python 参数匹配问题。

---

## 6. 关键文件与位置

- `cosyvoice/cli/cosyvoice.py`：`CosyVoice2.__init__`（约第 141 行）、`CosyVoice3.__init__`（约第 191 行）
- `run_npu.py`：根据模型类型构造 `kwargs` 的兼容逻辑
