# FreeToken Docker 安装说明书（中文）

> 适用对象：在 **NVIDIA 显卡的 Linux 机器**上，用 Docker 部署 FreeToken 推理服务。
> 本说明书基于 `hsly1975/FreeToken` 的 **`docker` 分支**（commit `394e3bc`，已含上游 #438 修复）。
> 目标场景：在 **RTX 4070 Super（12GB 显存 + 62GB 内存）** 上跑 **Ornith-1.5-35B-A3B-FP8**（MoE 专家权重放内存、GPU 做 LRU 缓存）。

---

## 0. 这套东西是什么（30 秒理解）

- **FreeToken** 是一个"边缘 MoE 推理引擎"：大 MoE 模型的**专家权重放在主机内存（RAM）**，GPU 上只放一个 **LRU 专家缓存**，没命中的专家走 PCIe 拉取或在 CPU 上算。这样 12GB 显存的卡也能跑 35B 级 MoE。
- 它对外提供 **OpenAI 兼容 API**（`/v1/chat/completions`、`/v1/responses`、`/v1/models`）和 **Anthropic 兼容 API**（`/v1/messages`），所以任何支持"自定义 base URL"的客户端都能直接接。
- 本分支额外提供了 **Docker 部署**（官方 main 没有 Dockerfile，这是从社区 PR #295 改造而来，并修掉了它装错版本导致 Ornith FP8 加载失败的问题）。

---

## 1. 硬件 / 系统要求

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | **Linux x86_64** | macOS / Windows 原生不支持（WSL2 理论可行但未验证） |
| GPU | NVIDIA，**驱动 r580+（CUDA 13）** | 支持 RTX 30/40/50 系；**V100（Volta）未验证** |
| 显存 | 12GB 起（跑 35B MoE 需 offload） | 4070S 的 12GB 够用，靠内存 offload |
| 内存 | **≥ 40GB 空闲**（跑 35B FP8） | 4070S 机器有 62GB，够；专家权重常驻 RAM |
| 磁盘 | ≥ 60GB 空闲 | 镜像 ~10GB + 模型权重 ~32GB + HF 缓存 |
| Docker | 已安装 Docker Engine + Compose v2 | `docker compose version` 能出版本号 |
| GPU 运行时 | **NVIDIA Container Toolkit** | 让容器能 `--gpus all` 看到显卡 |

> ⚠️ **4070S 特别注意**：这台机器平时跑 ComfyUI（音视频工坊）。FreeToken 和 ComfyUI 会**抢同一块显存**。部署 FreeToken 时请先**停掉 ComfyUI**，或确认显存足够两者共存（一般不够，建议二选一）。

---

## 2. 前置准备（一次性）

### 2.1 确认驱动版本 ≥ r580

```bash
nvidia-smi
```

看右上角 `Driver Version`，例如 `580.xx`。如果低于 580，先升级驱动（CUDA 13 需要 r580+）。

### 2.2 安装 Docker（若未装）

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # 把当前用户加进 docker 组，然后重新登录
```

### 2.3 安装 NVIDIA Container Toolkit（让容器用 GPU）

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.4 验证容器能看到 GPU

```bash
docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu26.04 nvidia-smi
```

能打印出显卡信息即成功。

---

## 3. 获取代码

```bash
git clone -b docker https://github.com/hsly1975/FreeToken.git
cd FreeToken
```

> 用 `-b docker` 拉的是带 Docker 支持的分支。

---

## 4. 构建镜像

```bash
docker build -t freetoken:latest .
```

- 首次构建会**编译 C++ 扩展**（`pinned_tensor` / `cpu_moe` / `ple_store`）并装依赖，**耗时较长（约 10–30 分钟）**，属正常。
- 镜像基于 `nvidia/cuda:13.3.1-devel-ubuntu26.04`，内含 `nvcc`，用于首次运行时 JIT 编译 CUDA kernel。
- 本 Dockerfile **从源码安装 FreeToken**（`uv pip install ".[accel]"`），所以镜像始终跟随 git main，自动带上 #438 等修复；torch 锁定在 `>=2.11,<2.12`（cu130）。

构建成功会看到 `Successfully tagged freetoken:latest`。

---

## 5. 配置（docker-compose.yml 字段说明）

打开 `docker-compose.yml`，逐字段含义如下：

```yaml
services:
  freetoken:
    image: freetoken:latest
    container_name: freetoken-server
    ipc: host                 # 【必须】共享内存，给 KV cache / 多 worker 通信用。
                              # 漏掉会导致 Bus error 或 CUDA OOM 崩溃。
    ports:
      - "1919:1919"           # 宿主机端口:容器端口。1919 是 FreeToken 默认端口。
                              # 若 1919 被占用，改成 "1920:1919" 之类（左边改即可）。
    environment:
      - HF_TOKEN=${HF_TOKEN:-}   # 仅当模型是 gated/私有仓库才需要。
                                 # 公开模型留空即可。
      - HF_HOME=/root/.cache/huggingface   # 容器内 HF 缓存目录。
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
        # 把宿主机 HF 缓存挂进容器 → 模型权重跨重启保留，不用每次重下。
    command:
      - "--model"
      - "${MODEL_NAME:-ornith-ai/Ornith-1.5-35B-A3B-FP8}"
        # 要跑的模型。默认 Ornith-1.5-35B-A3B-FP8（4070S 推荐）。
        # 可换成 HF repo id 或本地目录，见第 7 节。
      - "--host"
      - "0.0.0.0"             # 监听所有网卡（允许局域网其它机器访问）。
                              # 只想本机访问可改 127.0.0.1。
      - "--port"
      - "1919"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all      # 把全部 GPU 交给容器。
              capabilities: [gpu]
    restart: unless-stopped   # 异常退出自动拉起（手动 stop 除外）。
```

### 5.1 用环境变量覆盖（不改文件）

`MODEL_NAME` 和 `HF_TOKEN` 支持从环境注入，方便换模型：

```bash
# 例：换模型 + 提供 HF token
MODEL_NAME=Qwen/Qwen3.6-35B-A3B-FP8 HF_TOKEN=hf_xxx docker compose up -d
```

---

## 6. 启动

```bash
docker compose up -d
```

实时看日志，等出现这行才算就绪：

```bash
docker compose logs -f freetoken
# 看到：API server is ready to serve on 127.0.0.1:1919
```

> 首次启动会**下载模型权重（~32GB）**，取决于网速可能几十分钟。日志里能看到下载进度。

---

## 7. 验证

### 7.1 看服务了哪个模型

```bash
curl http://127.0.0.1:1919/v1/models
```

### 7.2 发一条对话（流式）

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ornith-1.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "什么是 MoE 模型？"}],
    "max_tokens": 256,
    "stream": true
  }'
```

> `model` 字段用 `/v1/models` 返回的 id（默认是模型目录名）。

### 7.3 终端里直接聊（可选）

```bash
docker compose exec freetoken ft shell
```

进入后 `/help` 看命令（`/think`、`/cache`、`/reset`）。

### 7.4 接 coding agent（可选）

```bash
docker compose exec freetoken ft launch hermes
# 支持 claude / codex / dsh / hermes / openclaw / opencode
```

它会自动写好该 agent 的 provider 配置并启动，指向本服务。

---

## 8. 模型选择（4070S 推荐）

| 模型 | 是否适合 4070S(12GB) | 说明 |
|---|---|---|
| **ornith-ai/Ornith-1.5-35B-A3B-FP8** | ✅ 推荐（默认） | 35B MoE，FP8，靠 offload 跑；需 ~40GB 内存 |
| Qwen/Qwen3.6-35B-A3B-FP8 | ✅ 可 | 同级别 MoE，FP8 |
| Qwen/Qwen3.5-35B-A3B-FP8 | ✅ 可 | 同上 |
| 27B dense（如 Qwen3.8-27B-FP8） | ⚠️ 勉强 | dense 模型专家全驻 GPU，12GB 偏紧 |
| 12B 级（gemma-4-12B 等） | ✅ 轻松 | 显存富余，速度快 |

- MoE 模型默认走 **offload** 策略（专家在内存、GPU 做 LRU 缓存），无需手动指定。
- 想调优可加 `--moe-strategy auto`（默认）或先跑 `ft bench bw` 校准 CPU/PCIe 带宽，让引擎自动在 offload/hybrid 间选。

---

## 9. 常用运维命令

```bash
docker compose logs -f freetoken     # 看日志
docker compose restart freetoken     # 重启
docker compose down                  # 停止并删除容器（权重在挂载里，不丢）
docker compose up -d                 # 再启动

# 健康检查 / 统计（在容器内）
docker compose exec freetoken ft ctl health
docker compose exec freetoken ft ctl stats
```

---

## 10. 更新到最新

```bash
cd FreeToken
git pull
docker build -t freetoken:latest .   # 重新构建（源码安装，自动带上上游修复）
docker compose up -d                 # 重建容器
```

---

## 11. 常见问题（排错）

| 现象 | 原因 / 解决 |
|---|---|
| 构建时 C++ 扩展编译报错 | 确认基础镜像是 CUDA 13 devel（含 nvcc）；看具体报错，多为依赖拉取失败，重试 `docker build` |
| 启动报 `Bus error` / `CUDA OOM` | 多半漏了 `ipc: host`；确认 compose 里有这一行 |
| 容器看不到 GPU | NVIDIA Container Toolkit 没装好；跑第 2.4 步验证 |
| 驱动版本不够 | `nvidia-smi` 看 Driver Version，需 ≥ 580 |
| 模型下载慢 / 失败 | 国内可设 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`（加到 compose 的 environment） |
| 1919 端口被占 | 改 compose 的 `ports` 左边，如 `1920:1919`，客户端也用 1920 |
| 显存不够 / 和 ComfyUI 冲突 | 先停 ComfyUI；或换更小模型 |
| Ornith FP8 加载失败 | 确认用的是本 `docker` 分支（源码安装，含 #438）；别用 PyPI 0.1.2 旧包 |
| 局域网其它机器访问不了 | `--host` 要 `0.0.0.0`，且防火墙放行 1919 |

---

## 12. 安全提示

- `--host 0.0.0.0` 会让**局域网内任何机器**都能调用（且**无鉴权**）。若机器在不可信网络，建议改回 `127.0.0.1`，或加一层反代 + token。
- 公开模型不需要 `HF_TOKEN`；填了也只会用于 gated 模型。

---

*本说明书随 `docker` 分支维护；命令与 `Dockerfile` / `docker-compose.yml` / `docs/cli.md` 保持一致。*
