# Zero Shot By ID 接口使用说明

## 功能概述

`zero_shot_by_id`接口允许客户端通过传入说话人ID（string类型），在服务端固定目录读取预定义的音频文件，进行zero_shot风格的语音合成。这样可以避免每次都传输音频数据，提高效率。

## 接口设计

### 1. Proto定义

```protobuf
message zeroshotByIdRequest{
  string tts_text = 1;      // 要合成的文本
  string spk_id = 2;        // 说话人ID
  bool stream = 3;          // 是否流式推理
}
```

### 2. 说话人信息文件格式

说话人信息使用一个统一的JSON文件`speaker_info.json`来描述所有说话人：

#### speaker_info.json 文件格式

```json
{
  "speaker1": {
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_wav": "speaker1.wav"
  },
  "speaker2": {
    "prompt_text": "今天天气真不错。",
    "prompt_wav": "speaker2.wav"
  },
  "speaker3": {
    "prompt_text": "欢迎使用语音合成服务。",
    "prompt_wav": "speaker3.wav"
  }
}
```

- 顶层key: 说话人ID（spk_id）
- `prompt_text`: 提示文本
- `prompt_wav`: 音频文件名（相对于speaker_info_dir）

### 3. 目录结构示例

```
model_dir/
├── cosyvoice3.yaml
├── llm.pt
├── flow.pt
├── hift.pt
└── speaker_info/
    ├── speaker_info.json    # 统一的配置文件
    ├── speaker1.wav
    ├── speaker2.wav
    └── speaker3.wav
```

## 使用方法

### 1. 服务端启动

```bash
python runtime/python/grpc/server.py \
    --model_dir /path/to/cosyvoice3/model \
    --port 50000 \
    --speaker_info_dir /path/to/speaker_info
```

参数说明：
- `--speaker_info_dir`: 说话人信息文件目录，默认为`model_dir/speaker_info`

### 2. 客户端调用

#### 方式1: 使用gRPC客户端

```bash
python runtime/python/grpc/client.py \
    --mode zero_shot_by_id \
    --host 0.0.0.0 \
    --port 50000 \
    --tts_text "你好，欢迎使用语音合成服务。" \
    --spk_id speaker1 \
    --stream \
    --tts_wav output.wav
```

#### 方式2: 使用Python代码

```python
import grpc
import cosyvoice_pb2
import cosyvoice_pb2_grpc

# 创建gRPC通道
channel = grpc.insecure_channel('0.0.0.0:50000')
stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

# 创建请求
request = cosyvoice_pb2.Request()
zero_shot_by_id_request = cosyvoice_pb2.zeroshotByIdRequest()
zero_shot_by_id_request.tts_text = "你好，欢迎使用语音合成服务。"
zero_shot_by_id_request.spk_id = "speaker1"
zero_shot_by_id_request.stream = True
request.zero_shot_by_id_request.CopyFrom(zero_shot_by_id_request)

# 发送请求并接收响应
response = stub.Inference(request)

# 处理响应
import numpy as np
import torch
import torchaudio

all_audio_chunks = []
for r in response:
    chunk = np.frombuffer(r.tts_audio, dtype=np.int16)
    all_audio_chunks.append(chunk)

# 保存音频
full_audio = np.concatenate(all_audio_chunks)
tts_speech = torch.from_numpy(full_audio).unsqueeze(dim=0).float() / (2**15)
torchaudio.save('output.wav', tts_speech, 24000)
```

#### 方式3: 直接使用CosyVoice3类

```python
from cosyvoice.cli.cosyvoice import CosyVoice3

# 初始化模型
cosyvoice = CosyVoice3(
    model_dir='/path/to/model',
    speaker_info_dir='/path/to/speaker_info'
)

# 执行推理
for result in cosyvoice.inference_zero_shot_by_id(
    tts_text="你好，欢迎使用语音合成服务。",
    spk_id="speaker1",
    stream=True
):
    tts_speech = result['tts_speech']
    # 处理音频
```

## 创建说话人信息文件

### 准备speaker_info.json文件

1. 准备所有说话人的音频文件（建议16kHz或24kHz采样率）
2. 创建统一的`speaker_info.json`文件：

```json
{
  "中文女": {
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_wav": "chinese_female.wav"
  },
  "中文男": {
    "prompt_text": "今天天气真不错。",
    "prompt_wav": "chinese_male.wav"
  },
  "英文女": {
    "prompt_text": "Welcome to our voice synthesis service.",
    "prompt_wav": "english_female.wav"
  }
}
```

3. 将`speaker_info.json`和所有音频文件放入`speaker_info_dir`目录

### 目录结构示例

```
speaker_info/
├── speaker_info.json
├── chinese_female.wav
├── chinese_male.wav
└── english_female.wav
```

## 性能优化

### 1. 服务启动时预加载

所有说话人信息在服务启动时一次性加载到内存，推理时直接从缓存读取，无需重复加载文件。

### 2. 流式推理

使用流式推理可以降低首包延迟：

```bash
python client.py --mode zero_shot_by_id --spk_id speaker1 --stream
```

## 错误处理

### 错误1: Speaker ID not found

**现象**: `ValueError: Speaker ID xxx not found`

**解决**:
- 检查`speaker_info_dir`目录是否正确
- 确认`{spk_id}.json`或`{spk_id}.pt`文件是否存在

### 错误2: Audio file not found

**现象**: 音频文件加载失败

**解决**:
- 检查JSON文件中的`prompt_wav`路径是否正确
- 确认音频文件存在于`speaker_info_dir`目录

### 错误3: Sample rate mismatch

**现象**: 音频质量差或速度异常

**解决**:
- 确保音频文件采样率为16kHz或24kHz
- 使用torchaudio进行重采样

## 最佳实践

1. **音频质量**: 使用高质量的参考音频，建议时长3-10秒
2. **文本匹配**: prompt_text应与prompt_wav内容匹配
3. **文件命名**: 使用有意义的spk_id，如"中文女"、"英文男"等
4. **目录管理**: 定期清理不使用的说话人信息文件
5. **备份**: 保存原始音频文件的备份

## 示例场景

### 场景1: 多说话人语音合成

```bash
# 说话人1
python client.py --mode zero_shot_by_id --spk_id speaker1 --tts_text "你好"

# 说话人2
python client.py --mode zero_shot_by_id --spk_id speaker2 --tts_text "Hello"
```

### 场景2: 批量合成

```python
import grpc
import cosyvoice_pb2
import cosyvoice_pb2_grpc

channel = grpc.insecure_channel('0.0.0.0:50000')
stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

texts = ["第一句话", "第二句话", "第三句话"]
spk_ids = ["speaker1", "speaker2", "speaker1"]

for text, spk_id in zip(texts, spk_ids):
    request = cosyvoice_pb2.Request()
    zero_shot_by_id_request = cosyvoice_pb2.zeroshotByIdRequest()
    zero_shot_by_id_request.tts_text = text
    zero_shot_by_id_request.spk_id = spk_id
    request.zero_shot_by_id_request.CopyFrom(zero_shot_by_id_request)
    
    response = stub.Inference(request)
    # 处理响应...
```

## 更新日志

### v1.0.0 (2026-03-05)
- 初始版本
- 支持JSON和PT格式的说话人信息文件
- 支持流式推理
- 支持gRPC和Python API调用
