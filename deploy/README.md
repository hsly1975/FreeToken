# FreeToken Docker 部署使用说明

> 本目录包含**已验证的生产部署配置**和一键脚本。
> 验证环境：X99 双路服务器 + RTX 4070S 12GB（device=1）+ 62GB 内存
> 模型：Ornith-1.5-35B-A3B-NVFP4（MoE 35B-A3B，NVFP4 量化，磁盘约 24GB）
> 实测：稳态显存 ~11.5G/12G，实际可用上下文 ~8K tokens

## 目录内容

| 文件 | 用途 |
|------|------|
| `docker-compose.prod.yml` | 生产 compose 配置（已验证参数） |
| `install.sh` | 一键安装脚本（检查环境→启动→验证） |
| `export-image.sh` | 导出镜像为离线 tar（有网机器执行） |
| `import-image.sh` | 导入离线镜像 tar（目标机器执行） |

## 一、前置条件

1. **Linux + NVIDIA GPU**（12GB 显存起步，推荐 16GB+）
2. **Docker** + **nvidia-container-toolkit**
   ```bash
   # 安装 nvidia-container-toolkit（以 Ubuntu 为例）
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. **内存 ≥ 32GB**（MoE 权重 offload 到内存，62GB 实测最佳）
4. **磁盘 ≥ 30GB 空闲**（模型 24GB + 镜像 15GB）

## 二、获取镜像（二选一）

### 方式 A：在线拉取
```bash
docker pull freetoken:latest
```

### 方式 B：离线导入（内网/无外网机器）
在**有网的机器**上导出：
```bash
./export-image.sh            # 生成 freetoken-image.tar（约 15GB）
```
把 tar 包传到目标机器后导入：
```bash
./import-image.sh freetoken-image.tar
```

## 三、准备模型

把模型文件放到模型目录（默认 `/data/models`），需要两个文件：

```
/data/models/
├── Ornith-1.5-35B-A3B-NVFP4.safetensors   # 权重（约 24GB）
└── config.json                            # 模型配置
```

> 模型来源：HuggingFace `FlashML/Ornith-1.5-35B-A3B-NVFP4`
> ```bash
> huggingface-cli download FlashML/Ornith-1.5-35B-A3B-NVFP4 \
>   --include "*.safetensors" "config.json" --local-dir /data/models
> ```

## 四、一键部署

```bash
cd deploy/
./install.sh
```

脚本会自动：环境检查 → 确认镜像 → 生成配置 → 启动容器 → 等待健康检查。

**自定义参数**（环境变量覆盖）：
```bash
MODEL_DIR=/data/models PORT=1919 GPU=1 MEMORY_RATIO=0.85 KV_RESERVE=81920 ./install.sh
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_DIR` | `/data/models` | 模型文件目录 |
| `PORT` | `1919` | 宿主机 API 端口 |
| `GPU` | `0` | 使用的 GPU 编号（nvidia-smi 里的编号） |
| `MEMORY_RATIO` | `0.85` | 显存占比，12GB 卡建议 0.85，16GB+ 可试 0.9 |
| `KV_RESERVE` | `81920` | KV cache 预留 token 数（决定上下文长度） |

## 五、手动部署（不用脚本）

```bash
cd deploy/
# 按需修改 docker-compose.prod.yml 里的 MODEL_DIR 和 device_ids
docker compose -f docker-compose.prod.yml up -d
docker logs -f freetoken-ornith        # 看启动日志
```

## 六、验证

```bash
# 健康检查
curl http://127.0.0.1:1919/health

# 查看模型列表
curl http://127.0.0.1:1919/v1/models

# 测试对话
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Ornith-1.5-35B-A3B-NVFP4","messages":[{"role":"user","content":"你好"}]}'
```

## 七、日常运维

```bash
# 查看状态
docker compose -f deploy/docker-compose.prod.yml ps

# 查看日志
docker logs -f freetoken-ornith

# 停止 / 启动 / 重启
docker compose -f deploy/docker-compose.prod.yml down
docker compose -f deploy/docker-compose.prod.yml start
docker compose -f deploy/docker-compose.prod.yml restart
```

## 八、性能与调优（实测数据）

| 配置 | 显存占用 | 可用上下文 | 说明 |
|------|---------|-----------|------|
| `memory-ratio 0.85` + `kv-reserve 81920` | ~11.5G/12G | **~8K** | 12GB 卡推荐配置 |
| `memory-ratio 0.85` + `kv-reserve 163840` | ~11.5G/12G | ~16K（理论） | KV 超出部分 offload 到内存，速度下降 |
| `memory-ratio 0.9` | ~11.8G/12G | 同上 | 更激进，有 OOM 风险 |

**关键结论**：
- 12GB 卡上**实际可用上下文约 8K tokens**（KV cache 仅驻 GPU 的部分）
- 超过 8K 的请求 KV 会 offload 到内存，能跑但速度明显下降
- 长上下文需求请上 16GB+ 显卡，或降低 `kv-reserve-tokens`
- 首次启动加载权重约 1-3 分钟，属正常现象

### 80K 长上下文实测（2026-09-16，冷启动、无缓存命中）

| 指标 | 8K 基线 | 80K 上下文 | 变化 |
|------|---------|-----------|------|
| 解码速度（逐字输出） | 70.6 tok/s | **61.1 tok/s** | 慢 ~13% |
| TTFT（首字延迟） | 1.6s | **31.8s** | 慢 ~20 倍 |
| Prefill 吞吐 | ~4900 tok/s | ~2200 tok/s | — |
| 总耗时（70K 输入 + 300 字输出） | 5.8s | **36.7s** | — |

**结论**：
- **解码速度基本没掉**（80K 下仍 ~61 tok/s，仅慢 13%），KV offload 到内存的机制工作良好，逐字生成体验几乎无感
- **真正的代价在首字延迟**：80K 的 prefill 需 ~32s 才吐出第一个字（70K token prefill 吞吐约 2200 tok/s），这是长上下文的固有成本
- 80K 请求完整处理成功（prompt_tokens=70028），**不 OOM**
- 相同前缀的重复请求会命中前缀缓存，TTFT 可降到 ~2s

## 九、故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | 显存不足 | 降低 `MEMORY_RATIO`（如 0.8）或 `KV_RESERVE` |
| 启动后一直不 ready | 权重加载中 | 等 1-3 分钟，`docker logs -f` 看进度 |
| `nvidia-container` 报错 | toolkit 未装好 | 重装 nvidia-container-toolkit 并 `sudo systemctl restart docker` |
| 端口被占用 | 1919 已被使用 | 改 `PORT` 环境变量 |
| 推理很慢 | KV offload 到内存 | 缩短上下文，或换更大显存显卡 |

## 十、OpenAI 兼容 API

服务提供标准 OpenAI 兼容接口，可直接接入任何 OpenAI SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://<服务器IP>:1919/v1", api_key="none")
resp = client.chat.completions.create(
    model="Ornith-1.5-35B-A3B-NVFP4",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```
