# CosyVoice2/3 Ascend 部署手册

本发布包提供离线服务和流式服务；离线服务同时支持 CosyVoice2（v2）与 CosyVoice3（v3），
流式服务分别由 v3 原生 gRPC 和 v2 官方 FastAPI 提供：

| 服务形态 | v2 | v3 | 服务入口 | 默认端口 |
| :--- | :---: | :---: | :--- | :--- |
| **离线服务**（一请求一完整 WAV） | ✅ | ✅ | `offline/service/run_service.sh` | 9500（示例） |
| **流式服务：v3 原生 gRPC** | ❌ | ✅ | `streaming/grpc/run_server.sh` | 50000 |
| **流式服务：v2 官方 FastAPI** | ✅ | ❌ | `streaming/v2_fastapi/run_server_v2.sh` | 50000 |

> v2 没有可用的 gRPC 服务；v3 的流式使用原生 gRPC。离线的 v2/v3 使用同一个
> `offline/service/`（通过 `--model-type` 或 `--model-dir` 选择模型）。

---

## 1. 发布包结构

```
cosyvoice/
├── cosyvoice_deploy.md                 # 部署手册（本文件）
├── cosyvoice_api.md                    # 服务接口使用手册（给调用方）
├── download_weights.sh                 # 下载权重 + 建相对软链（v2/v3）
├── create_container.sh                 # 创建/进入容器（默认容器名 cosyvoice）
├── create_speaker_info.sh              # 生成/更新 v3 speaker_info（zero_shot_by_id）
├── cosyvoice_models/                   # 运行 download_weights.sh 后生成（不含在发布包）
├── offline/
│   ├── cosyvoice_models            # download_weights.sh 生成（指向 --dest）
│   └── service/
│       ├── run_service.sh              # 离线服务启动（v2/v3）
│       ├── server.py                   # FastAPI /health + /synthesize
│       ├── client.py                   # 离线服务测试客户端
│       └── README.md
└── streaming/
    ├── cosyvoice_models             # download_weights.sh 生成（指向 --dest）
    ├── test_streaming.py               # 统一流式测试客户端（v2 FastAPI / v3 gRPC）
    ├── grpc/
    │   └── run_server.sh            # 直接调用 /opt 源码（pb2/客户端用镜像自带）
    └── v2_fastapi/                     # v2 官方 FastAPI（HTTP 流式）
        ├── run_server_v2.sh
        ├── serve_v2.py
        ├── client_v2.py
        └── README.md
```

说明：
- 发布包**不包含模型权重**，用 `download_weights.sh` 下载到 `cosyvoice_models/` 或 `--dest` 指定目录。
- 发布包**不包含 CosyVoice 源码**；服务端代码来自镜像内已打补丁的 `/opt/CosyVoice`。
- `streaming/cosyvoice_models`、`offline/cosyvoice_models` 由 `download_weights.sh` 生成，指向 `--dest`（默认发布包根目录）。

---

## 2. 环境与容器创建

### 2.1 前置条件

- 昇腾 Atlas 800T A2（910B3）或 A3（910C）服务器，驱动/固件正常（`npu-smi info` 可查看）。
- 已加载 Ascend-SACT 的 `tritonserver-ascend` 镜像：
  - A2：`tritonserver-ascend:910b-v3`
  - A3：`tritonserver-ascend:910c-v3`
  - 如果环境里没有对应镜像，按 2.2 从官方仓库下载分片、合并并 `docker load`。
- 镜像内 `/opt/CosyVoice` 已包含本发布对应的已打补丁源码与 `runtime/python/{grpc,fastapi}/`。
- 数据盘（如 `/data`）用于放权重，**不要放在 `/opt/CosyVoice`**。

### 2.2 获取/导入镜像（本地无镜像时）

先检查本地是否已有对应镜像：

```sh
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'tritonserver-ascend:(910b|910c)-v3'
```

**如果没有输出（镜像不存在）**，从 Ascend-SACT 官方仓库获取：

- 仓库：<https://gitcode.com/Ascend-SACT/tritonserver-ascend>
- 镜像基于 Triton + 昇腾 CANN；A2 用 `910b-v3`，A3 用 `910c-v3`。
- 仓库提供的是**分片 tar** + `merge_*.sh` 合并脚本；需要先合并、校验，再 `docker load`。
- 预留足够磁盘空间（分片 + 合并后的 tar + Docker 镜像）。

```sh
# 1) 克隆镜像仓库（在有网络的机器上）
git clone https://gitcode.com/Ascend-SACT/tritonserver-ascend.git
cd tritonserver-ascend

# 2A) Atlas A2 / 910B：合并 4 个分片并导入
bash merge_910b_v3.sh
docker load -i tritonserver-ascend-910b-v3.tar.gz

# 2B) Atlas A3 / 910C：合并 4 个分片并导入
bash merge_910c_v3.sh
docker load -i tritonserver-ascend-910c-v3.tar.gz
```

说明：
- `merge_910b_v3.sh` / `merge_910c_v3.sh` 会合并对应版本的 4 个分片，并自动校验
  合并后文件的 **sha256**；校验失败会中止，不要跳过校验直接 `docker load`。
- 只需要导入与服务器平台匹配的一个镜像（A2 选 910b-v3，A3 选 910c-v3）。
- 如果现场已经拿到**合并后的完整 tar 包**，可直接执行 `docker load -i <tar.gz>`。
- 如果仓库访问受限，可由有网络的机器克隆后，把 `merge_*.sh` 与分片/完整 tar 拷到现场再导入；
  也可以把已加载的镜像用 `docker save` 导出后转移到现场。

导入完成后确认：

```sh
docker images | grep tritonserver-ascend
```

应能看到：

```
tritonserver-ascend   910b-v3    ...
# 或
tritonserver-ascend   910c-v3    ...
```

### 2.3 创建容器

发布包提供 `create_container.sh`，容器创建命令已固化到脚本，无需手工复制粘贴。

推荐把发布包解压到宿主机挂载给容器的 `/data` 下（例如 `/data/cosyvoice`），
容器内的工作目录就是该目录：

```sh
# 宿主机：把发布包放到 /data 下（/data 可按需修改，见下文）
mkdir -p /data/cosyvoice
tar -xzf /data/cosyvoice.tar.gz \
  -C /data/cosyvoice --strip-components=1

cd /data/cosyvoice
bash create_container.sh
```

脚本内部实际执行的核心容器创建命令（默认参数）如下：

```sh
docker run -itd --name cosyvoice \
  --network host --ipc=host \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data:/data \
  -w /data/cosyvoice \
  tritonserver-ascend:910b-v3 bash
```

以上内容已经写到发布包根目录的 `create_container.sh` 中，**无需复制粘贴**；按需修改容器名/镜像/数据盘后，
直接执行 `bash create_container.sh` 即可。

`create_container.sh` 默认：
- 容器名 **`cosyvoice`**
- 镜像 `tritonserver-ascend:910b-v3`
- 将宿主机 `/data` 挂载到容器 `/data`
- 工作目录 `/data/cosyvoice`（由脚本按发布包在 `DATA_DIR` 下的相对路径自动推导）

脚本会：
1. 检查镜像是否存在（不存在会提示按 2.2 获取）；
2. 创建容器（`--network host --ipc=host`，挂载 `/data`、Ascend driver）；
3. 进入容器 shell；非交互执行时打印 `docker exec` 命令。

常用变体：

```sh
# 容器重名/想换名字：改脚本顶部的 CONTAINER_NAME，或用 --name
bash create_container.sh --name cosyvoice-test

# Atlas A3 / 910C
bash create_container.sh --image tritonserver-ascend:910c-v3

# 宿主机数据盘不是 /data：修改脚本顶部 DATA_DIR，或执行时覆盖
DATA_DIR=/your/data bash create_container.sh
```

说明：
- **`/data` 可以自行修改**：编辑 `create_container.sh` 顶部的 `DATA_DIR`，或用
  `DATA_DIR=/your/data` 执行；发布包必须解压在该目录下，容器内才能看到。
- **容器名可以按需修改**：改脚本顶部的 `CONTAINER_NAME` 默认值，或执行时传 `--name`；
  重名时脚本会报错，除非加 `--force` 删除旧容器。
- 多卡时请编辑脚本里的 `--device /dev/davinciN` 列表；服务侧仍建议**每卡一个实例**。
- 如果镜像尚未导入，请先按 **2.2** 合并分片并 `docker load`，再执行本脚本。
- 发布包目录需要可写（下载脚本会在 `streaming/`、`offline/` 下建软链）；只读时请先把
  发布包拷贝到可写的数据盘目录。

---

## 3. 下载权重

### 3.1 下载 v2 + v3 默认权重

#### 3.1.1 有网络环境时

部署机器可以访问 ModelScope 时，直接执行：

```sh
cd /data/cosyvoice
bash download_weights.sh --dest /data/cosyvoice_models
```

默认下载：

- `CosyVoice2-0.5B`（ModelScope `iic/CosyVoice2-0.5B`）
- `Fun-CosyVoice3-0.5B`（ModelScope `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`）

脚本会下载权重，并自动在 `streaming/`、`offline/` 下创建指向 `--dest` 的相对软链。

#### 3.1.2 无网络环境时

部署机器不能访问 ModelScope 时，先在另一台有网络的机器下载，再通过 U 盘/移动介质导入：

1. **有网机器下载到 U 盘**

   ```sh
   mkdir -p /media/usb/cosyvoice_models
   bash download_weights.sh --dest /media/usb/cosyvoice_models
   ```

2. **拷贝到现场数据盘**

   把整个 `cosyvoice_models/` 目录拷到现场机器，保持目录结构：

   ```
   /data/cosyvoice_models/
   ├── CosyVoice2-0.5B/
   └── Fun-CosyVoice3-0.5B/
   ```

   建议使用 `cp -a` 或 `rsync -a` 保留目录结构和文件权限；v3 的 `speaker_info/`
   （如已生成）也一并拷贝。

3. **生成软链**

   ```sh
   cd /data/cosyvoice
   bash download_weights.sh --dest /data/cosyvoice_models
   ```

   模型目录已存在时脚本会跳过下载，只补建 `streaming/`、`offline/` 下的相对软链：

   ```sh
   ls -l streaming/cosyvoice_models/Fun-CosyVoice3-0.5B
   ls -l offline/cosyvoice_models/Fun-CosyVoice3-0.5B
   ```


两种方式下，脚本生成的软链都指向实际 `--dest`；具体相对路径随 `--dest` 变化
（例如 `--dest /data/cosyvoice_models` 时，`streaming/cosyvoice_models` 会指向它）。
脚本同时支持传入额外软链目录：

```sh
bash download_weights.sh --dest /data/cosyvoice_models ../my-deploy-dir
```

更多参数见：

```sh
bash download_weights.sh --help
```

### 3.2 预置音色（v2 / v3）

CosyVoice 是**零样本 TTS**：合成时需要一段参考音频（以及它的逐字转写）来提供音色、
韵律和发音风格。所谓“预置音色”，就是把这段参考音频/转写提前准备好，避免每次请求临时上传
或临时填写。

- **v3**：可以把参考音频 + 转写固化到 `speaker_info/`，请求时只传 `spk_id`（`zero_shot_by_id`），
  更适合固定音色业务，也减少每次上传参考音频的开销。
- **v2**：没有 `speaker_info`/`zero_shot_by_id` 的注册机制；所谓“预置”是提前把参考音频和
  逐字转写准备好，在每次 `zero_shot` 请求里通过 `prompt_wav`/`prompt_text` 传入。

参考音频要求：3–10 秒、单人、干净、无明显噪声；`prompt_text` 必须是参考音频的逐字转写。
v3 的 `prompt_text` 还必须包含 `<|endofprompt|>`。

#### 3.2.1 v3 预置音色（zero_shot_by_id）

`download_weights.sh` **只负责下载权重**，不会生成 `speaker_info`。v3 预置音色由独立脚本
`create_speaker_info.sh` 生成（v3 专属）。

```sh
bash create_speaker_info.sh \
  --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
  --spk-id spk1 \
  --prompt-wav /data/prompt.wav \
  --prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
```

生成：
```
/data/cosyvoice_models/Fun-CosyVoice3-0.5B/speaker_info/prompt.wav
/data/cosyvoice_models/Fun-CosyVoice3-0.5B/speaker_info/speaker_info.json
```

参数：
| 参数 | 说明 |
| :--- | :--- |
| `--prompt-wav PATH` | 参考音频（必填，3–10s、单人、干净） |
| `--prompt-text TEXT` | 逐字转写，必须含 `<|endofprompt|>`（省略时使用默认示例文本） |
| `--spk-id ID` | `speaker_info.json` 中的音色 ID（默认 `spk1`） |
| `--model-dir DIR` | v3 模型目录；省略时自动查找 `cosyvoice_models/Fun-CosyVoice3-0.5B` |
| `--speaker-info-dir DIR` | 自定义输出目录（默认 `<model-dir>/speaker_info`） |
| `--force` | 覆盖已存在的同名音色/音频 |

说明：
- 多个音色可重复执行脚本并用不同 `--spk-id`；脚本会合并 JSON，不覆盖其他音色。
- `spk1` 的参考音频保存为 `prompt.wav`；其他 ID 保存为 `prompt_<spk-id>.wav`。
- 服务调用时使用 `--mode zero_shot_by_id --spk-id spk1`（离线/流式接口具体参数见 `cosyvoice_api.md`）。

#### 3.2.2 v2 预置音色（参考音频）

v2 **没有** `zero_shot_by_id`/`speaker_info`，也没有对应的“注册脚本”；v2 的预置就是
准备好参考音频文件并记住其逐字转写，调用时随请求传入。

```sh
# 准备一份参考音频（示例；后续示例统一用 /data/prompt.wav）
cp /path/to/v2_prompt.wav /data/prompt.wav
```

调用 `zero_shot` 时同时传：

```sh
python3 offline/service/client.py --url http://127.0.0.1:9501 \
  --mode zero_shot \
  --prompt_wav /data/prompt.wav \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --text "你好，这是 v2 预置音色测试。" --out /tmp/v2.wav
```

不同模式的参考音频要求：

| v2 模式 | 需要的参考输入 |
| :--- | :--- |
| `zero_shot` | `prompt_wav` + `prompt_text`（逐字转写） |
| `cross_lingual` | `prompt_wav`（仅借音色） |
| `instruct2` | `prompt_wav` + `instruct_text` |
| `sft` / `instruct` | `spk_id`（依赖模型自带 `spk2info.pt`，普通下载模型通常没有） |

> v2 不能使用 `zero_shot_by_id`；不要把 v3 的 `speaker_info` 配置用于 v2 服务。

---

## 4. 部署一：离线服务（v2 + v3）

离线服务是常驻 HTTP 服务：加载一次模型，`POST /synthesize` **一次性返回完整 WAV**（无 SSE/chunk），
单实例用全局锁串行处理。v2/v3 使用同一个入口，按模型目录里的 `cosyvoice2.yaml` / `cosyvoice3.yaml` 分流。

### 4.1 启动 v3

```sh
cd /data/cosyvoice
bash offline/service/run_service.sh --model-type v3 --device 0 --port 9500
```

### 4.2 启动 v2

```sh
bash offline/service/run_service.sh --model-type v2 --device 1 --port 9501
```

也可以显式给模型目录（与 `--model-type` 二选一）：

```sh
bash offline/service/run_service.sh \
  --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
  --device 0 --port 9500
```

参数：

| 参数 | 说明 |
| :--- | :--- |
| `--model-type v2\|v3` | 自动选 `cosyvoice_models/{CosyVoice2-0.5B,Fun-CosyVoice3-0.5B}`；与 `--model-dir` 互斥 |
| `--model-dir DIR` | 显式模型目录；类型按 yaml 自动识别 |
| `--device N` | 单张 NPU 卡号；多卡请起多个实例 |
| `--port PORT` | HTTP 端口（默认 9500） |
| `--speaker-info-dir DIR` | v3 speaker_info（默认 `<model-dir>/speaker_info`） |
| `--host HOST` | 绑定地址（默认 0.0.0.0） |
| `--graph-mode` | v3 torchair 图模式（启动更慢） |
| `--vllm` / `--no-vllm` | 强制开/关 vLLM（默认 v3 开、v2 关） |

模型加载完成后 `/health` 才可用，客户端需等待或重试。

### 4.3 接口与验证

接口字段、模式矩阵、返回头、错误码和 curl/Python 示例见 `cosyvoice_api.md` 的
“离线服务接口”。部署后先等 `/health` 返回 `status=ok`，再用客户端生成 WAV 试听：

```sh
# v3
python3 offline/service/client.py --url http://127.0.0.1:9500 \
  --mode zero_shot_by_id --spk_id spk1 \
  --text "你好，这是离线 v3 测试。" --out /tmp/offline_v3.wav

# v2
python3 offline/service/client.py --url http://127.0.0.1:9501 \
  --mode zero_shot --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /data/prompt.wav \
  --text "你好，这是离线 v2 测试。" --out /tmp/offline_v2.wav
```

> 并发：单实例串行（全局锁）。需要并发请起多个实例（每卡一个、不同端口），或只对 LLM 开 vLLM。

## 5. 部署二：v3 流式原生 gRPC

当前 `streaming/grpc` 是 **v3 专用**。`run_server.sh` 默认**直接调用**
`/opt/CosyVoice/runtime/python/grpc/server.py`，使用镜像内已打补丁源码，
**不需要把发布包脚本覆盖到 `/opt`**。只要 `/opt/CosyVoice` 更新，启动的就是更新后的源码。

### 5.1 直接调用 /opt 源码（无需覆盖）

脚本会自动：
- 把 `/opt/CosyVoice` 及 `third_party/Matcha-TTS` 加入 `PYTHONPATH`；
- 执行 `/opt/CosyVoice/runtime/python/grpc/server.py`（Python 会自动把该目录加入
  `sys.path`，server.py 也会加入 `/opt/CosyVoice`）。

如果镜像里的 `grpcio` 与 pb2 版本不一致，在源码树内重新生成即可：

```sh
cd /opt/CosyVoice/runtime/python/grpc
bash update_grpc_file.sh
```

### 5.2 启动

```sh
cd /data/cosyvoice
bash streaming/grpc/run_server.sh \
  --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
  --speaker-info-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B/speaker_info \
  --device 0 --max-conc 8 --port 50000 --graph-mode
```

如果使用自行重建的源码树（不是 `/opt/CosyVoice`），加 `--cosy-repo /path/to/cosyvoice-src`。

| 参数 | 说明 |
| :--- | :--- |
| `--cosy-repo DIR` | CosyVoice 源码树（默认 `/opt/CosyVoice`；重建源码树时传入） |
| `--model-dir DIR` | 显式指定 v3 模型目录（与 `--model-type` 二选一） |
| `--model-type v3` | 自动使用 `Fun-CosyVoice3-0.5B` 目录；v2 不支持 gRPC |
| `--speaker-info-dir DIR` | 预置音色目录（默认 `<model-dir>/speaker_info`） |
| `--device N` | 单卡卡号（默认 0） |
| `--max-conc N` | 服务调度槽位/最大并发（默认 8） |
| `--port N` | gRPC 端口（默认 50000） |
| `--graph-mode` | 开启 torchair 图 + gRPC 启动后 warmup；省略则为非图（启动更快） |
| `--prewarm-timeout N` / `--no-prewarm` | 非图预热超时 / 跳过预热 |
| `--extra-args "ARGS"` | 追加 server.py 参数，例如 `--extra-args "--enable_vllm_profile"` |

模型选择与 `offline/service/run_service.sh` 一致：`--model-dir` 与 `--model-type v3` 二选一；
都不给时默认查找 `$HERE/../cosyvoice_models/Fun-CosyVoice3-0.5B`、
`$HERE/../../../cosyvoice_models/Fun-CosyVoice3-0.5B`，再回退
`/data/cosyvoice_models/Fun-CosyVoice3-0.5B`。例如：

```sh
bash run_server.sh --model-type v3 \
  --device 0 --max-conc 8 --port 50000 --graph-mode
```

启动完成日志：`server listening on 0.0.0.0:50000`。

### 5.3 接口与验证

RPC 定义、请求字段、当前支持的模式、分块格式和错误处理见 `cosyvoice_api.md` 的
“v3 流式 gRPC 接口”。启动后验证：

```sh
cd /data/cosyvoice
python3 streaming/test_streaming.py --model-type v3 --api grpc \
  --url 127.0.0.1:50000 --mode zero_shot_by_id --spk-id spk1 \
  --text "你好，这是 v3 流式测试。" --out /tmp/stream_v3.wav
```

输出包含 `chunks / ttfb / total / rtf`，并保存 wav。
发布包不再携带 gRPC 的 pb2/客户端脚本；如需批量压测，可直接使用镜像
`/opt/CosyVoice/runtime/python/grpc/` 下自带的测试脚本（当前镜像中是
`test_client_npu.sh` / `batch_test_client.py`，以镜像实际文件名为准；重建源码树中对应
`test_client.sh`）。

## 6. 部署三：v2 流式官方 FastAPI

v2 流式服务直接运行官方 `runtime/python/fastapi/server.py`（不改动），发布包只做 Ascend 适配包装。
**v2 没有 gRPC 服务；v2 流式请用本 FastAPI。**

### 6.1 启动

```sh
cd /data/cosyvoice
ASCEND_RT_VISIBLE_DEVICES=0 PORT=50000 \
MODEL_DIR=/data/cosyvoice_models/CosyVoice2-0.5B \
  bash streaming/v2_fastapi/run_server_v2.sh
```

启动日志出现 `Uvicorn running on http://0.0.0.0:50000` 即就绪（模型加载在监听前完成）。

### 6.2 接口与验证

端点、字段、模式限制、裸 PCM 返回格式和示例见 `cosyvoice_api.md` 的
“v2 流式 FastAPI 接口”。启动后验证：

```sh
cd /data/cosyvoice
python3 streaming/v2_fastapi/client_v2.py --url http://127.0.0.1:50000 \
  --mode zero_shot --tts_text "你好，这是 v2 流式测试。" \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /data/prompt.wav --tts_wav /tmp/stream_v2.wav
```

> 官方 FastAPI 不支持 vLLM；首个请求通常包含 eager 首次算子编译，TTFB 会高于后续请求。

## 7. 验证矩阵与验收建议

| 服务 | v2 | v3 | 验证命令 |
| :--- | :---: | :---: | :--- |
| 离线服务 | ✅ | ✅ | `offline/service/client.py` 或 curl `/synthesize` |
| v3 流式 gRPC | ❌ | ✅ | `streaming/test_streaming.py --model-type v3 --api grpc`（或镜像自带 `test_client_npu.sh`） |
| v2 流式 FastAPI | ✅ | ❌ | `streaming/v2_fastapi/client_v2.py` 或 `streaming/test_streaming.py --model-type v2 --api fastapi` |

建议验收步骤：

1. `npu-smi info` 确认卡健康、显存空闲；
2. 下载权重并准备 speaker_info / v2 prompt wav；
3. 起对应服务，等待 `/health` 或 `server listening`；
4. 用客户端生成 WAV 并**试听**；
5. 检查指标：离线看 RTF/总耗时；流式看 `chunks>1`、TTFB、RTF；
6. 多卡时每卡一个实例，不要单实例跨卡。

---

## 8. 常见问题

1. **`grpcio`/pb2 版本不匹配**
   `grpc package installed is at version 1.85.3, but generated code depends on grpcio>=1.85.4`：
   在 `/opt/CosyVoice/runtime/python/grpc` 执行 `bash update_grpc_file.sh`，或升级 `grpcio`。

2. **`speaker 'spk1' not found`（v3）**
   检查 `SPEAKER_INFO_DIR/speaker_info.json` 是否存在、`spk_id` 是否正确、`prompt_text` 是否含
   `<|endofprompt|>`。

3. **v3 `prompt_text must contain '<|endofprompt|>'`**
   v3 `zero_shot` 的 `prompt_text` 必须是：
   `"You are a helpful assistant.<|endofprompt|>" + 参考音频逐字转写`。

4. **v2 zero_shot 音色/内容异常**
   必须同时提供 `prompt_wav` 和**逐字转写** `prompt_text`；只给音频或转写不一致会影响音色/内容。

5. **服务退出后显存未释放 / 下次 OOM**
   异常退出可能遗留 `VLLM::EngineCore` 进程（占用大量 HBM）。停服时确认进程树退出：
   `ps -ef | grep -E 'EngineCore|server.py'`，必要时清理后再启动。

6. **启动很慢**
   非 graph 加载约 2–3 分钟；graph 模式（torchair 图 + warmup）约 5–7 分钟；
   离线服务首次请求还包含首次算子编译。请用 `/health`/日志判断是否仍在加载。

7. **v2/v3 混用接口**
   v3 gRPC 只支持 `zero_shot_by_id`/`sft`；v2 官方 FastAPI 端点只用于 v2。不要把
   `CosyVoice2-0.5B` 指向 v3 gRPC 服务，也不要把 `Fun-CosyVoice3-0.5B` 指向 v2 FastAPI。

8. **权重位置**
   权重放数据盘（如 `/data/cosyvoice_models`），不要放 `/opt/CosyVoice`；下载脚本生成的相对软链
   让服务默认能找到它。需要显式覆盖时：
   - 离线服务：`run_service.sh --model-dir ... --speaker-info-dir ...`
   - v3 gRPC：`run_server.sh --model-dir ... --speaker-info-dir ...`
   - v2 FastAPI：`MODEL_DIR=...` 环境变量（该脚本仍使用环境变量）。
