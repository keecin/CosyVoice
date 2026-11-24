# Bug 报告 2：双流（Bi-Streaming）合成报 dtype 不一致，无法生成

## 一句话结论

在“边输入文本、边输出语音”的双流模式（bistream）里，有一段文本会被单独喂给模型，
它携带的是 **bf16**（16 位低精度）数据类型，而前面已经存进缓存的中间结果却是
**fp32**（32 位）数据类型。注意力计算（SDPA）要求两者类型一致，于是直接报错：

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float instead.
```

修复位置：`cosyvoice/llm/llm.py` 的 `Qwen2Encoder.forward_one_step()`（把增量输入
统一转换成与缓存一致的数据类型）。

> 这是一个**通用**的代码问题（与硬件无关），只是在我们的运行配置下更容易暴露出来。

---

## 0. 先补几个概念（不懂代码也能看）

- **dtype（数据类型/精度）**：模型里的数字有不同“精度”。常见的有
  `float32`（fp32，单精度，占 4 字节）、`bfloat16`（bf16，半精度，占 2 字节）。
  运算时，**参与同一运算的张量 dtype 必须一致**，否则会报错或结果错误。
- **SDPA（Scaled Dot-Product Attention）**：注意力计算的核心函数。它强制要求
  query / key / value（查询/键/值）三者 dtype 完全相同。
- **embedding（嵌入）**：把“token 编号”查表变成“向量”的模块。CosyVoice 里有两类：
  - Qwen 文本模型的词嵌入（`embed_tokens`）；
  - CosyVoice 自己新增的语音/文本嵌入（`speech_embedding`、`llm_embedding`）。
- **Bi-Streaming（双流）**：一种低延迟模式——文本不是一次性给全，而是“来一块文本
  就生成一点语音”，同时语音也分块输出。代码在 `Qwen2LM.inference_bistream()`。

---

## 1. 现象（怎么发现的）

普通（非双流）合成正常；一旦用双流模式（`--bistream --stream`）就崩溃：

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float instead.
错误发生在 transformers 的 sdpa_attention_forward（注意力计算）里。
```

含义：这一步的 **query 是 bf16**，但缓存里的 **key/value 是 fp32**，类型对不上。

---

## 2. 根因：模型里本来就“混用了两种数据类型”

### 2.1 两类嵌入的 dtype 不一样

- Qwen 文本权重来自 checkpoint `CosyVoice-BlankEN`，它存的是 **bf16**
  （`config.json` 里 `"torch_dtype": "bfloat16"`，`model.safetensors` 里 290 个张量全是 BF16）。
  所以文本词嵌入 `embed_tokens.weight` 是 **bf16**。
- 而 CosyVoice 自己新建的 `speech_embedding` / `llm_embedding` 是 Python 默认的 **fp32**。

两者相加/拼接时，PyTorch 会把结果“提升”成精度更高的 fp32：

```python
torch.cat([tensor_fp32, tensor_bf16])  # 结果 dtype = float32
```

这本身不报错，问题出在双流的某一步**绕过了这个“提升”**。

### 2.2 双流里有一段“纯文本块”被直接使用

`cosyvoice/llm/llm.py` 的 `inference_bistream()` 中：

```python
text_cache = self.llm.model.model.embed_tokens(prompt_text)   # bf16
...
for this_text in text:
    text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
    ...
    lm_input_text = text_cache[:, :self.mix_ratio[0]]          # 纯文本块（bf16）
    ...
    if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
        lm_input = lm_input_text                               # ← 直接使用，没有和 fp32 拼接！
    else:
        lm_input = torch.concat([lm_input, lm_input_text], dim=1)
```

- 大多数情况下，`lm_input` 会和 fp32 的语音嵌入拼接，被提升成 fp32。
- 但“遇到 fill token 需要补文本”这一分支里，`lm_input = lm_input_text` **直接把它当成
  下一步输入**，于是它保持 **bf16**。

### 2.3 缓存却是 fp32

- 最开始的那一步（prefill）里，输入里混合了 fp32 的 sos/语音嵌入，拼接后是 fp32，
  模型算出的 Key/Value 也就存成 **fp32** 缓存。
- 下一步突然来了一个 **bf16** 的 query，去和 **fp32** 的缓存做注意力 → dtype 冲突 → 报错。

实测打点（临时日志）清楚显示了这一点：

```
fwd_one_step xs=float32 cache_k=float32 past=118 cur=1
fwd_one_step xs=float32 cache_k=float32 past=119 cur=1
fwd_one_step xs=float32 cache_k=float32 past=120 cur=1
fwd_one_step xs=bfloat16 cache_k=float32 past=121 cur=5   ← 纯 bf16 文本块，报错
```

---

## 3. 为什么说是“通用问题”，不是某硬件特有

- **dtype 混用来自模型本身**：checkpoint 是 bf16 + 新建嵌入是 fp32，这在任何设备上都一样。
- **报错的校验是框架层的**：我用普通张量在本机 **CPU** 上直接复现了同一条错误：

  ```python
  q = torch.randn(..., dtype=torch.bfloat16)
  k = torch.randn(..., dtype=torch.float32)
  v = torch.randn(..., dtype=torch.float32)
  F.scaled_dot_product_attention(q, k, v)
  # RuntimeError: Expected query, key, and value to have the same dtype ... query bf16 ...
  ```

  `scaled_dot_product_attention` 要求 q/k/v 同 dtype，是 PyTorch 的通用约束，CUDA/CPU/NPU 都一样。
- 是否“暴露”取决于运行配置：如果开启半精度自动混合（autocast/fp16），框架会自动把参与运算的
  张量统一成同一类型，这个 bug 就被掩盖了；而 CosyVoice 默认 `fp16=False`（关闭 autocast），
  朴素推理时就会暴露。

---

## 4. 修复

### 4.1 思路

既然“缓存里的 Key/Value 是什么类型”，就让“下一步的输入”也转成同样的类型，
保证进入注意力计算时三者一致。

### 4.2 改动（`cosyvoice/llm/llm.py` 的 `forward_one_step`）

```python
def forward_one_step(self, xs, masks=None, cache=None):
    past_len = self._cache_len(cache)

    # 若已有缓存，把当前输入 xs 转成与缓存 Key/Value 相同的 dtype
    if cache is not None:
        try:
            cache_dtype = cache.layers[0].keys.dtype if hasattr(cache, 'layers') else cache[0][0].dtype
        except (IndexError, TypeError, AttributeError):
            cache_dtype = None
        if cache_dtype is not None and xs.dtype != cache_dtype:
            xs = xs.to(cache_dtype)

    total_len = past_len + xs.size(1)
    attention_mask = torch.ones((1, total_len), device=xs.device, dtype=torch.bool)
    outs = self.model(inputs_embeds=xs, attention_mask=attention_mask,
                      use_cache=True, past_key_values=cache, ...)
    return outs.hidden_states[-1], outs.past_key_values
```

要点：

- 兼容两种缓存：新版 `transformers` 的 `DynamicCache`（有 `.layers[0].keys`）和旧版的
  tuple 缓存（`cache[0][0]`）。
- **只在“已有缓存”时转换**：第一步（prefill）没有缓存，保持原样，不影响普通路径。
- 转换方向跟随缓存：缓存是 fp32 就把输入转 fp32；缓存是 bf16 就转 bf16，始终一致。

---

## 5. 验证

双流 + 流式输出（`--bistream --stream`）：

| 模型 | 结果 | 分块数 | 总时长 |
| :--- | :--- | :--- | :--- |
| CosyVoice2-0.5B | 正常生成 | 4 块 | 11.3 s |
| Fun-CosyVoice3-0.5B | 正常生成 | 4 块 | 10.4 s |

普通（非双流）合成回归正常：CosyVoice2 输出 11.8 s，未受影响。

---

## 6. 影响范围 / 兼容性

- 只影响“增量解码时输入与缓存 dtype 不一致”的场景（最典型就是双流）；
- 修复对普通路径、训练路径无影响（无缓存时不做转换）；
- 与硬件无关，对 CUDA / CPU / NPU 都是更稳健的做法；
- 顺带也降低了“某些后端不容忍混合 dtype 运算”导致的潜在崩溃风险。

---

## 7. 关键文件与位置

- `cosyvoice/llm/llm.py`
  - `Qwen2Encoder.forward_one_step()`：dtype 对齐逻辑（约第 255~276 行）
  - `Qwen2LM.inference_bistream()`：纯文本块直接作为输入的代码（约第 646 行）
- 模型 dtype 来源：`pretrained_models/*/CosyVoice-BlankEN/config.json`（`torch_dtype: bfloat16`）
