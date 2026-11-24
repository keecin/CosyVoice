# Bug 报告 1：transformers 5.x 下语音被“拖长、乱读”（KV-cache 增量解码错误）

## 一句话结论

CosyVoice 的文本大模型（LLM）在“一步一步往前生成语音 token”时，给模型传的
`attention_mask`（注意力掩码）长度算错了；在旧版 `transformers`（<5）下这个错误
恰好被“容忍”，在新版 `transformers 5.x` 下就变成真错误，导致模型生成出又长又乱的
语音（一句约 40 字的文本生成了 40 秒音频）。

修复位置：`cosyvoice/llm/llm.py` 的 `Qwen2Encoder._cache_len()` / `forward_one_step()`。

---

## 0. 先补几个概念（不懂代码也能看）

CosyVoice 合成一句话大致分三步：

1. **文本 → 语音 token（LLM 阶段）**：一个类似 Qwen2 的语言模型，一个 token 一个
   token 地“吐”出语音 token（类似 ChatGPT 一个字一个字生成）。这一步叫**自回归解码**。
2. **语音 token → 梅尔频谱（Flow 阶段）**：把 token 还原成声学特征。
3. **梅尔频谱 → 波形（HiFi-GAN 阶段）**：还原成能听的 wav。

本 bug 出在**第 1 步**。要理解它，需要知道两个词：

- **KV Cache（键值缓存）**：自回归生成时，前面已经算过的 token 的中间结果（Key/Value）
  会被缓存下来，生成下一个 token 时不必重算，只需算“新的一个 token”，再和缓存拼起来。
  这让生成变快，但要求每一步的“注意力掩码”和缓存**对齐**。
- **attention_mask（注意力掩码）**：告诉模型“哪些位置是真实 token、可以看；哪些是
  填充/不该看的”。它必须覆盖**所有**参与注意力的位置，也就是
  **历史缓存长度 + 当前新 token 长度**。

---

## 1. 现象（怎么发现的）

在昇腾容器里用系统自带 `transformers 5.14.1` 跑 CosyVoice2 的零样本合成：

- 输入一句约 40 字的中文；
- 生成的 wav **长达 40.0 秒**，而且整段都是有声音的（不是尾部静音），听起来在乱读/重复；
- 正常情况下这句应约 **11~12 秒**。

这不是“模型质量差”，而是模型算错了。

---

## 2. 根因：掩码长度只算了“当前”，没算“历史缓存”

### 2.1 出问题的代码（修复前）

`cosyvoice/llm/llm.py` 里，`Qwen2Encoder.forward_one_step` 负责“走一步”：

```python
def forward_one_step(self, xs, masks, cache=None):
    input_masks = masks[:, -1, :]          # 只取掩码的最后一行
    outs = self.model(
        inputs_embeds=xs,
        attention_mask=input_masks,         # ← 问题在这里
        use_cache=True,
        past_key_values=cache,             # ← 历史缓存
        ...
    )
```

而调用方（`inference_wrapper`）每一步都**重新按当前输入长度**造掩码：

```python
for i in range(max_len):
    y_pred, cache = self.llm.forward_one_step(
        lm_input,
        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]))).to(torch.bool),
        cache=cache,
    )
    ...
    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)  # 之后每步只有 1 个 token
```

### 2.2 错在哪

- 第 1 步（prefill）一次性喂入很长的输入（比如 118 个 token），这时掩码长度 = 118，正确。
- 从第 2 步开始，`lm_input` 只有 **1 个** token，所以掩码长度变成 **1**。
- 但此刻 KV-cache 里已经有 **118、119、120…** 个历史 token。
- 于是模型收到的是“长度=1 的掩码 + 长度=118 的缓存”，**掩码和缓存对不上**。

### 2.3 为什么以前没报错

- `transformers < 5`（如官方推荐的 4.51.3）用的是“老的 tuple 缓存”接口，它内部会把
  这个长度=1 的掩码“凑合”处理，凑巧能跑出正确结果。
- `transformers >= 5` 改成了 `DynamicCache` 等新接口，对掩码/缓存的对齐更严格，
  于是按错误的方式理解掩码 → 注意力看错了历史信息 → 生成的 token 全是错的 →
  模型迟迟不输出“结束符”，一直生成到上限（`max_len`），表现就是“音频特别长”。

> 关键：这是**代码逻辑问题**（掩码长度没算历史），不是 NPU 特有的，也不是模型权重坏了。
> 只是 `transformers` 版本变化把它从“隐藏”变成了“显性”。

---

## 3. 怎么定位的（可复现的最小实验）

不看 CosyVoice 的大模型，只拿一个极小的 Qwen2（随机权重）做对照实验：

- 用“整段一次性前向”得到每个位置的真实输出（标准答案）；
- 再用“一步一步 + KV-cache”的方式增量前向，比较两者差异。

结果（最大误差，越小越好）：

| 环境 | `full vs incremental` 最大误差 | 结论 |
| :--- | :--- | :--- |
| transformers 5.14.1 | **1.83** | 完全不一致（错误） |
| transformers 4.51.3 | 2.38e-07 | 一致（正确） |

这直接证明：问题出在“增量解码的注意力掩码”，且与 transformers 版本有关。

---

## 4. 修复

### 4.1 核心思路

让 `forward_one_step` **自己**按“历史缓存长度 + 当前输入长度”构造正确长度的掩码，
不再依赖调用方传进来的（长度可能不对的）掩码。

### 4.2 改动内容（`cosyvoice/llm/llm.py`）

新增一个兼容不同版本的“取缓存长度”函数（旧版是 tuple，新版是 DynamicCache）：

```python
@staticmethod
def _cache_len(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):      # transformers >= 5 (DynamicCache)
        return cache.get_seq_length()
    try:
        return cache[0][0].shape[-2]          # transformers < 5 (legacy tuple cache)
    except (IndexError, TypeError, AttributeError):
        return 0
```

`forward_one_step` 改为：

```python
def forward_one_step(self, xs, masks=None, cache=None):
    past_len = self._cache_len(cache)          # 历史长度
    total_len = past_len + xs.size(1)          # 历史 + 当前
    attention_mask = torch.ones((1, total_len), device=xs.device, dtype=torch.bool)
    outs = self.model(
        inputs_embeds=xs,
        attention_mask=attention_mask,         # ← 长度对了
        use_cache=True,
        past_key_values=cache,
        ...
    )
    return outs.hidden_states[-1], outs.past_key_values
```

同时，`inference_bistream` 里原本有两处用 `cache[0][0].size(2)` 计算长度的写法
（在 `DynamicCache` 上会直接报错），也一并去掉，统一交给 `forward_one_step` 处理。

> 说明：这里用“全 1 的掩码”是安全的，因为 CosyVoice 推理是单条、无 padding 的序列，
> 而因果（不能看未来）的约束由 Qwen2 模型内部自己保证，不需要我们再用下三角矩阵。

---

## 5. 验证

- 最小 Qwen2：`full vs incremental` 最大误差从 **1.83 → 2.38e-07**（恢复一致）。
- 端到端（`transformers 5.14.1`，不降级、不用隔离环境）：

| 模型 | 修复前 | 修复后 |
| :--- | :--- | :--- |
| CosyVoice2-0.5B | 40.0 s（乱读） | **11.8 s（正常）** |
| Fun-CosyVoice3-0.5B | （同类问题） | **10.7 s（正常）** |

---

## 6. 影响范围 / 兼容性

- 只影响“带 KV-cache 的自回归解码”（LLM 阶段）；Flow、HiFi-GAN、训练用的 `forward()`
  不受影响。
- 修复对 `transformers < 5` 和 `>= 5` 都兼容（`_cache_len` 两种缓存都认）。
- 与硬件无关：CUDA / CPU / NPU 都适用；也正因为如此，我们才**不需要降级 transformers、
  不需要虚拟环境**。

---

## 7. 关键文件与位置

- `cosyvoice/llm/llm.py`
  - `Qwen2Encoder._cache_len()`（约第 243 行）
  - `Qwen2Encoder.forward_one_step()`（约第 255 行）
  - `Qwen2LM.inference_wrapper()` 增量解码循环（约第 538 行）
  - `Qwen2LM.inference_bistream()` 中两处旧的长度计算（约第 654 / 677 行调用处）
