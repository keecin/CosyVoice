# Bug 报告 4：MindIE gRPC 批处理服务“崩溃”——CANN TBE 在多线程进程里 fork 失败

## 一句话结论

MindIE 的 gRPC 批量服务第一次做 **Flow 算子编译**时，CANN 的“算子知识库（TBE repository_manager）”
会**懒加载** `multiprocessing.Manager`（fork）和 `forkserver` 的 `task_distribute` 工作进程。
这次 fork 恰好发生在**多线程进程**（gRPC 线程池 + vLLM 线程 + t2w 线程）里，gRPC 的 `atfork`
处理被跳过，导致 fork 出来的子进程与父进程的 IPC 队列断开（`EOFError`），子进程打印
`main process disappeared` 后 `os._exit(1)`；于是算子编译永远不返回，`flow.inference` 挂起、
请求失败。

修复：**在启动 gRPC 线程之前，先单线程跑一遍完整流水线预热**（方案 1），让这次 fork 在
“没有 gRPC 线程”的上下文里安全完成。

---

## 1. 现象

- gRPC 服务能正常启动、加载 vLLM 引擎、监听端口、收到请求（日志有 `add_request`）。
- 一进入首帧推理（`Token2Wave.step()` → `self.flow.inference(...)`）就失败：
  - 日志 `t2w: flow.inference START` 之后再没有 `OK`，也没有 Python `Traceback`；
  - 约 5 秒后出现：
    ```
    WARNING multiprocessing/popen_fork.py: This process is multi-threaded, use of fork() may lead to deadlocks
    I... fork_posix.cc: Other threads are currently calling into gRPC, skipping fork() handlers
    [ERROR] TBE Subprocess[task_distribute] raised error[], main process disappeared!   (×8)
    ```
  - 客户端 `All requests failed`。
- 与并发数无关：`max_conc=1` 单请求也复现；主进程本身没死（`kill -0` 仍存活）。

> 注意 `main process disappeared` 是**误导性文案**：它其实是子进程 `task_q.get()` 收到
> `EOFError` 的分支，并非主进程真的退出。

---

## 2. 复现方法

环境：`ASCEND_RT_VISIBLE_DEVICES` 指向一个空闲 910B3（示例用 4）；工作目录必须在
`runtime/python/grpc`（pb2 需先 `bash update_grpc_file.sh` 重新生成）。

```sh
cd <repo>/runtime/python/grpc
export ASCEND_RT_VISIBLE_DEVICES=4

python3 server.py --port 50140 \
  --model_dir /path/Fun-CosyVoice3-0.5B \
  --max_conc 1 \
  --speaker_info_dir /path/Fun-CosyVoice3-0.5B/speaker_info \
  > /tmp/grpc.log 2>&1 &
# 等待 /tmp/grpc.log 出现 "server listening"

python3 batch_test_client.py --host 127.0.0.1 --port 50140 --stream \
  --mode zero_shot_by_id --spk_id spk1 --total_req 1 --concurrency 1
# 结果：All requests failed；/tmp/grpc.log 出现上面的 fork/TBE 序列
```

`specific speaker_info/speaker_info.json` 示例（`prompt_wav` 用绝对路径最稳）：
```json
{ "spk1": { "prompt_wav": "/abs/path/asset/zero_shot_prompt.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" } }
```

排查工具：`faulthandler.enable(all_threads=True)`；或
```sh
gdb -q --args python3 server.py ...
(gdb) set follow-fork-mode child
(gdb) catch fork
(gdb) catch signal SIGABRT
(gdb) run            # 触发请求后停在 fork
(gdb) thread apply all bt
```

---

## 3. 调用堆栈（关键！）

### 3.1 主流程 → 触发首次算子编译（t2w 线程内）
```
[grpc handler] CosyVoiceServiceImpl.Inference
  → self.process_request(...)
    → self.add_request('zero_shot_by_id', ...)          # 往 vLLM 加请求
  ...
[infer thread] CosyVoiceServiceImpl.infer  (server.py)
  → self.step()
    → self.maybe_t2w_infer(now)                         # server.py:598 附近
      → self.t2w.step()                                 # server.py:600
        → Token2Wave.step()                             # server.py:270
          → self.flow.inference(...)  <-- 第一次，需要编译算子（本次卡死点）
```
日志对应：`t2w_infer: sections` → `t2w: flow.inference START`（之后无 `OK`）。

### 3.2 fork 的诞生：CANN TBE 知识库懒加载（发生在上面 flow.inference 内部）
```
flow.inference(...)                          # 自定义算子/推理触发 CANN 算子查找
  → tbe.common.repository_manager ...        # 首次使用 -> RouteServer.initialize()
    → tbe/common/repository_manager/route.py: RouteServer.initialize()
      → GlobalMgr(threading.Thread)          # multiprocess_util.py
        → self.mgr = mp.Manager()           # ← fork 出 Manager 服务进程（py<3.14 默认 fork）
        → self.task_q = self.mgr.Queue(); ...
      → ctx = multiprocessing.get_context("forkserver")     # route.py:179
      → ctx.set_forkserver_preload([])
      → ctx.Process(target=cls.task_distribute, ...).start() # route.py:185  (×8)
        → forkserver 首次使用：从当前进程 fork 出 forkserver
      → ctx.Process(target=daemon_process, ...).start()      # route.py:189
```
此处的 fork 发生在**多线程进程**中（此时同时存在：gRPC `ThreadPoolExecutor` 线程、vLLM 线程、
`infer` 线程），于是：
```
CPython: multiprocessing/popen_fork.py: "This process is multi-threaded, use of fork() may lead to deadlocks"
gRPC:    fork_posix.cc:71: "Other threads are currently calling into gRPC, skipping fork() handlers"
```
gRPC 的 `pthread_atfork` 处理被跳过 → 子进程继承了不一致的锁/文件描述符状态。

### 3.3 子进程失败：task_distribute 的队列 EOF → “main process disappeared”
```
[forkserver 子进程] RouteServer.task_distribute(...)         # route.py:~256
  → 读取 GlobalMgr 队列
    → tbe/common/repository_manager/utils/multiprocess_util.py:73
      item = self.task_q.get()   -> conn.recv(...) -> EOFError
  → 被 subprocess_troubleshoot 捕获（route.py:67-77）
      except (EOFError, KeyboardInterrupt) as e:
          logger.error("[ERROR] TBE Subprocess[%s] raised error[%s], main process disappeared!")
          os._exit(1)
```
结果：8 个 `task_distribute` worker 全部 `os._exit(1)`，排队的“算子编译任务”没人处理 →
`flow.inference` 内部等待编译结果永远不返回 → 请求挂起/失败。

### 3.4 附带发现并修复的另一个服务端 bug（首请求后的 ZeroDivisionError）
预热修好 fork 后，又暴露一个纯 Python 逻辑 bug（会杀死 `infer` 线程），真实 traceback：
```
Exception in thread Thread-2 (infer):
  File ".../server.py", line 746, in infer
    t2w_step_outputs = self.maybe_t2w_infer(now)
  File ".../server.py", line 565, in maybe_t2w_infer
    requests_t2w_sched = self.schedule_nonfirst_requests_t2w(now)
  File ".../server.py", line 473, in schedule_nonfirst_requests_t2w
    t2w_min_token_len = max(self.t2w_min_token_len,
        available_token_len // len(requests_remain_token_len) - (...))
ZeroDivisionError: integer division or modulo by zero
```
原因：没有“非首帧剩余 token”的请求时 `requests_remain_token_len` 为空，仍做了除法。
修复：为空时直接 `return {}`。

---

## 4. 根因总结

1. CANN TBE 的“知识库/编译调度”子系统是**懒加载**的；第一次算子编译时用
   `multiprocessing.Manager()`（fork）和 `forkserver`（首次也 fork）创建后台进程。
2. 这次 fork 发生在一个**已经在多线程运行**的服务进程里，且其他线程正在 gRPC 调用中，
   导致 gRPC 跳过 `atfork` 处理器，fork 子进程的 IPC/锁状态不一致。
3. `task_distribute` 子进程读取 Manager 队列时 `EOFError`，走上 `subprocess_troubleshoot`
   的 `EOFError` 分支：打印 `main process disappeared` 并 `os._exit(1)`。
4. 编译任务无人处理 → `flow.inference` 挂起 → 请求失败。

**为什么单线程/预编译环境正常**：
- `run_npu.py`（非服务、主线程推理）：首次编译时没有 gRPC 线程，fork 安全 → 正常。
- MindIE 原始环境里 `runtime/python/grpc/` 下有大量 `kernel_meta/*.log`（算子已预编译/缓存），
  运行期不再触发这套“首次编译+懒加载 fork” → 稳定。

---

## 5. 修复（方案 1：单线程预编译/预热）

在 **gRPC 服务器与线程启动之前**，先在主线程跑一遍完整流水线，令 TBE 知识库初始化与
Flow/HiFi 算子编译在没有 gRPC 线程的环境下完成。

改动（`runtime/python/grpc/server.py`）：
- `CosyVoiceServiceImpl.__init__` 里，在 `self.inference_work_thread.start()` 之后、返回之前，
  调用一次 `self.warmup()`（跑 `add_request` + `step`/`t2w` 全链路）：
  ```python
  if not getattr(args, 'no_prewarm', False):
      thc_print("prewarm (pre-grpc): start")
      self.warmup()
      thc_print("prewarm (pre-grpc): done")
  ```
- 删除 `main()` 里原先“grpcServer.start() 之后才 warmup”的逻辑（那已太晚，fork 已在多线程里发生）。
- `schedule_nonfirst_requests_t2w`：`requests_remain_token_len` 为空时 `return {}`（修 ZeroDivision）。
- 让 server.py 自包含（不再需要外部 wrapper）：`import vllm` 之前设置
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`、把 `nvidia` triton backend 置为非活跃、中和被注入的
  `debugpy.wait_for_client()`；`import vllm` 之后注册 `CosyVoice2ForCausalLM`。

关键点：**必须让那次 fork/知识库初始化发生在 gRPC 线程启动之前**；预热用哪种文本/形状都行，
只要能触发一次算子编译，把 `repository_manager`/`forkserver` 建好，后续编译只入队、不再 fork。

---

## 6. 验证

修复后在 910B3（device 4）上（vLLM 0.30 后端）：

| 场景 | 结果 |
| :--- | :--- |
| 预热日志 | `prewarm (pre-grpc): start` → 之后才 `server listening`（无 fork 报错） |
| 单请求 / 2 请求（`max_conc=1` 直连 `server.py`） | 成功 2/2，首包 TTFB ~1.46s |
| 8 请求，`concurrency=4` | **8/8 成功**，wall 34.77s，TTFB ~1.6s，RTF 中位 ~1.49 |
| 8 请求，`concurrency=1` | 8/8 成功，wall 91.25s，TTFB ~1.45s |
| 日志 | 不再出现 `main process disappeared` / `popen_fork` 警告 / ZeroDivision |

吞吐对比（同模型同文本，8 请求）：并发 4 相对并发 1 的 **wall 时间约 2.6× 缩短**
（91.25s → 34.77s），说明 vLLM 批处理 + NPU 流水线确实带来吞吐提升。

---

## 7. 相关源码位置

- 业务侧（本仓库补丁）：`runtime/python/grpc/server.py`
  - `CosyVoiceServiceImpl.__init__`（预热调用）
  - `Token2Wave.step()`（`flow.inference`/`hift.inference`）
  - `schedule_nonfirst_requests_t2w()`（ZeroDivision 修复）
- CANN 侧（只读排查用）：
  - `/usr/local/Ascend/cann-9.2.1/python/site-packages/tbe/common/repository_manager/route.py`
    - `:50 validate_cannkb_entry()`（`CANN_KNOWLEDGE_BANK_PROCESS_NUM=0` 可整体关闭该子系统）
    - `:67-77 subprocess_troubleshoot`（`EOFError` → “main process disappeared” → `os._exit(1)`）
    - `:179/185/189`（`get_context("forkserver")` + `task_distribute`/`daemon_process`）
  - `.../utils/multiprocess_util.py:73`（`task_q.get()` → EOFError）；`GlobalMgr` 用 `mp.Manager()`
  - `.../interface.py:34+`（`validate_cannkb_entry` 为 False 时 `cann_kb_*` 全部 no-op）
