# 在“当前昇腾容器镜像”里安装 CosyVoice（方案 A：源码目录 + PYTHONPATH）

本文说明如何把**本仓库（已打补丁的 CosyVoice 源码）**安装进你当前正在使用的昇腾镜像，
做成一个可直接运行 CosyVoice2/3（含 vLLM / gRPC 服务）的镜像。

> 核心原则：CosyVoice **没有 `setup.py`/`pyproject.toml`**，它按“**源码目录 + `PYTHONPATH`**”
> 运行（官方 Dockerfile 也是这么做的）。**不要把 `cosyvoice/` 拷进 `site-packages`**：
> 还需要 `third_party/Matcha-TTS`（`matcha` 包）在 `sys.path` 上，且代码依赖仓库根下的
> `pretrained_models/`、`asset/`、`runtime/python/grpc` 等相对结构。

---

## 0. 前提

- 基础镜像：**你当前使用的 Ascend 910B 镜像**（已含 `torch_npu`、`vllm_ascend`、CANN、
  `onnxruntime-cann`、Python 3.12），例如你现在 `docker exec` 进去的这个容器所用镜像。
- **`third_party/Matcha-TTS` 子模块必须已初始化**（否则目录为空，`import matcha` 失败）：
  ```sh
  cd /path/to/CosyVoice
  git submodule update --init --recursive
  ls third_party/Matcha-TTS/    # 应为非空（含 matcha/ setup.py 等）
  ```
- 已提供现成文件：`docker/Dockerfile.npu`、`.dockerignore`（仓库根）。

---

## 1. 构建镜像

在**仓库根目录**执行（build context = 仓库根）：

```sh
cd /path/to/CosyVoice

# 1) 把 Dockerfile.npu 里的 FROM 改成你实际的基础镜像 tag
#    FROM <YOUR_ASCEND_IMAGE:TAG>
vi docker/Dockerfile.npu

# 2) 构建
docker build -f docker/Dockerfile.npu -t cosyvoice-npu:latest .
```

`docker/Dockerfile.npu` 内容（方案 A）如下，关键是三件事：**拷贝源码（含 Matcha 子模块）**、
**追加 `PYTHONPATH`**、**只装缺失依赖**：

```dockerfile
FROM <YOUR_ASCEND_IMAGE:TAG>

COPY . /opt/CosyVoice
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}
WORKDIR /opt/CosyVoice

RUN pip install --no-cache-dir -r requirements-npu.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
 && pip install --no-cache-dir --only-binary=:all: pyarrow==18.1.0 \
        -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

RUN cd runtime/python/grpc && bash update_grpc_file.sh

CMD ["bash"]
```

说明：
- `ENV PYTHONPATH=...:${PYTHONPATH}` 是**追加**，不会覆盖基础镜像已有的 Ascend/DALI 路径。
  ⚠️ 若你的基础镜像**没有把 `PYTHONPATH` 定义成 ENV**（`echo $PYTHONPATH` 为空），请改用
  文末“**方案 B：`.pth` 文件**”，更稳妥。
- 严禁使用仓库里的**官方 `requirements.txt`**（它锁 CUDA 版 `torch/onnxruntime-gpu/tensorrt-cu12`，
  会破坏 NPU 栈）；只装 `requirements-npu.txt` + `pyarrow==18.1.0`。
- `.dockerignore` 已排除 `.git/`、`pretrained_models/`、`outputs/`、`*.wav`、`kernel_meta/` 等，
  构建更快、镜像更小。

---

## 2. 运行容器（沿用你原来启动昇腾镜像的参数）

设备/驱动挂载方式**与你现在启动该昇腾镜像时一致**（davinci 设备、driver 挂载、`ASCEND_*`
环境变量等），只把镜像换成 `cosyvoice-npu:latest`。示例（按你们环境的实际参数调整）：

```sh
# 模型建议放在宿主机，运行时挂载进容器（不要打进镜像）
# 宿主机目录示例：/data/cosyvoice/pretrained_models/{CosyVoice2-0.5B,Fun-CosyVoice3-0.5B}

docker run -it --rm \
  <你原来使用的设备/驱动挂载参数> \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -v /data/cosyvoice/pretrained_models:/opt/CosyVoice/pretrained_models \
  cosyvoice-npu:latest bash
```

进入容器后（工作目录已是 `/opt/CosyVoice`，`PYTHONPATH` 生效）：

```sh
# 若宿主机没有模型，也可在容器内下载（需网络）
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
PY
```

---

## 3. 验证安装

```sh
# 1) 包能导入（cosyvoice + matcha 都应来自 /opt/CosyVoice）
python3 -c "import cosyvoice, matcha; print(cosyvoice.__file__); print(matcha.__file__)"
#   期望：/opt/CosyVoice/cosyvoice/__init__.py 与 /opt/CosyVoice/third_party/Matcha-TTS/matcha/__init__.py

# 2) 设备垫片可用（run_npu.py 内部已 import transfer_to_npu）
python3 -c "import torch, torch_npu, torch_npu.contrib.transfer_to_npu; print('cuda->npu', torch.cuda.is_available())"

# 3) 跑一句普通推理
python3 run_npu.py pretrained_models/CosyVoice2-0.5B outputs/v2.wav
python3 run_npu.py pretrained_models/Fun-CosyVoice3-0.5B outputs/v3.wav

# 4) 双流 / bf16 / vLLM（可选）
python3 run_npu.py pretrained_models/CosyVoice2-0.5B outputs/v2_bs.wav --bistream --stream
python3 run_npu.py pretrained_models/Fun-CosyVoice3-0.5B outputs/v3_vllm.wav --vllm

# 5) gRPC 批量服务（可选；pb2 已在镜像构建时生成）
cd runtime/python/grpc
python3 server.py --port 50099 --model_dir /opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    --max_conc 8 --speaker_info_dir /opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/speaker_info
```

---

## 4. 以后“更新源码”怎么做

方案 A 的好处：**源码就是 `/opt/CosyVoice` 这个目录**，`PYTHONPATH` 固定指过去。

- **重新构建镜像（推荐用于发布）**：
  ```sh
  git pull                # 或替换成新的源码
  git submodule update --init --recursive
  docker build -f docker/Dockerfile.npu -t cosyvoice-npu:latest .
  ```
- **不重建镜像、临时验证新源码**：把新源码目录挂载覆盖进去即可（进程重启生效）：
  ```sh
  docker run ... -v /data/cosyvoice/new_source:/opt/CosyVoice  cosyvoice-npu:latest bash
  ```
  （因为 `PYTHONPATH=/opt/CosyVoice`，挂载覆盖后 import 的就是新代码。）

---

## 5. 关键注意事项

1. **不要** `pip install -r requirements.txt`；只用 `requirements-npu.txt`（+ `pyarrow==18.1.0`）。
2. **不要**把 `cosyvoice/` 拷进 `site-packages`：还需要 `third_party/Matcha-TTS` 在路径上，
   且依赖仓库根相对结构（`pretrained_models/`、`asset/`、gRPC 的 pb2/相对导入）。
3. `third_party/Matcha-TTS` 必须是**有内容**的子模块（构建前 `git submodule update --init`）。
   其 Cython 扩展 `monotonic_align` 仅**训练**用，推理不需要；靠 `PYTHONPATH` 即可。
4. **模型/音频资产不进镜像**（体积大）：运行时挂载或 modelscope 下载到 `pretrained_models/`。
5. **设备垫片**：`run_npu.py` 已内置 `import torch_npu.contrib.transfer_to_npu`。若用
   `webui.py` / gRPC `server.py` 等入口，它们各自也需要 `transfer_to_npu`（本项目 server.py 已在
   文件顶部处理 vLLM/NPU 相关初始化；如改用其他入口请确认已 import）。
6. **gRPC pb2**：`*_pb2.py` 被 `.gitignore` 忽略，构建时已通过
   `runtime/python/grpc/update_grpc_file.sh` 生成；如改动 proto 需重新生成。
7. `ENV PYTHONPATH=...:${PYTHONPATH}` 依赖基础镜像已定义 `PYTHONPATH` ENV。若不确定，用方案 B。

---

## 6. 方案 B（可选）：用 `.pth` 固化路径（不依赖 PYTHONPATH 环境变量）

如果基础镜像没有 `PYTHONPATH` ENV，或你不想动环境变量，可在 Dockerfile 里改用 `.pth`
（Python 启动时自动把每行路径加入 `sys.path`）：

```dockerfile
RUN printf '/opt/CosyVoice\n/opt/CosyVoice/third_party/Matcha-TTS\n' \
    > /usr/local/lib/python3.12/dist-packages/cosyvoice_paths.pth
```
其余（COPY 源码、装依赖、pb2）与方案 A 相同。效果等价于 `PYTHONPATH`，且不会覆盖任何现有路径。

---

## 7. 可选：想用 `pip install`（不推荐，需自行补打包）

上游没有打包文件；且 `cosyvoice/` 下多个子目录没有 `__init__.py`（`flow/、tokenizer/、hifigan/、
llm/、vllm/` 等，依赖 PEP 420 namespace package），所以要用 `find_namespace_packages` 并带上
`tokenizer/assets/*.tiktoken` 资源。若必须 pip 安装，新建 `pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "cosyvoice"
version = "0.0.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["cosyvoice*"]
namespaces = true

[tool.setuptools.package-data]
cosyvoice = ["tokenizer/assets/*.tiktoken"]
```

```sh
cd /opt/CosyVoice
pip install -e .                       # 可编辑安装，改源码即时生效
pip install -e third_party/Matcha-TTS  # matcha 也可单独装（会编译 Cython，仅训练用；或继续用 PYTHONPATH）
python3 -c "import cosyvoice.flow, cosyvoice.tokenizer, cosyvoice.cli.cosyvoice; print('ok')"
```
> 仍建议方案 A/B；pip 安装只是可选，且不适合频繁更新源码的场景。

---

## 8. 常见问题

| 现象 | 原因 / 解决 |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'cosyvoice'` | `PYTHONPATH` 没生效或路径写错；检查 `python3 -c "import sys;print(sys.path)"` 是否含 `/opt/CosyVoice` |
| `ModuleNotFoundError: No module named 'matcha'` | `third_party/Matcha-TTS` 为空（子模块没初始化）或未在路径上 |
| `pyarrow requires NumPy 2.0 or newer` | 没降级 pyarrow；`pip install --only-binary=:all: pyarrow==18.1.0` |
| `torch.cuda.is_available()` 为 False | 没打 `transfer_to_npu` 垫片（非 `run_npu.py` 入口需自行 import） |
| gRPC `No module named 'cosyvoice_pb2'` | 未生成 pb2；在 `runtime/python/grpc` 执行 `bash update_grpc_file.sh` |
| 启动报 “Free memory ... less than desired gpu_memory_utilization” | 该 NPU 被其他进程占用；换空闲卡（`ASCEND_RT_VISIBLE_DEVICES`）或调低 vLLM `gpu_memory_utilization` |

---

## 9. 最小命令清单（TL;DR）

```sh
# 构建
cd /path/to/CosyVoice && git submodule update --init --recursive
vi docker/Dockerfile.npu      # 改 FROM 为你的昇腾镜像
docker build -f docker/Dockerfile.npu -t cosyvoice-npu:latest .

# 运行（设备参数沿用你现有昇腾容器的启动方式）
docker run -it --rm <你的设备/驱动挂载参数> -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -v /data/cosyvoice/pretrained_models:/opt/CosyVoice/pretrained_models \
  cosyvoice-npu:latest bash

# 容器内
python3 -c "import cosyvoice, matcha; print('ok')"
python3 run_npu.py pretrained_models/CosyVoice2-0.5B outputs/v2.wav
```
