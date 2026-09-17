# FreeToken 35B (Ornith-1.5-35B-A3B-NVFP4) 性能基准

测试环境：RTX 4070S 12G（device=1），196K 上下文 + D2D 优化（`--moe-prefill-hit-d2d`）
测试日期：2026-09-17
测试方法：`deploy/bench_prefill.py`，唯一 prompt（避开 radix cache，测真实 prefill）

## Prefill（首字延迟 TTFT）

| Prompt | TTFT | prefill 速度 |
|---|---|---|
| 4K (4096) | 1.60s | ~2560 tok/s |
| 8K (8192) | 1.62s | ~5057 tok/s |
| 16K (16384) | 3.28s | ~5000 tok/s |
| 32K (32768) | 7.03s | ~4661 tok/s |
| 64K (65536) | 16.09s | ~4073 tok/s |
| 192K (196608) | 46.9s | ~4192 tok/s |

## Decode（生成速度）

| 输出长度 | decode 速度 |
|---|---|
| 64 tokens | 33–37 tok/s |
| 256 tokens | 38.7 tok/s |
| 192K 上下文下 | 35.7 tok/s |

## 对比：Qwen3.8-27B-MTP（V100 32G，8081 端口）

| 指标 | 1919 (MoE 35B, 4070S) | 8081 (27B dense, V100) |
|---|---|---|
| prefill 4K | **1.60s** | 6.00s |
| prefill 速度 | **~5000 tok/s** | ~683 tok/s |
| decode | 33–39 tok/s | **40.9 tok/s** |

## 结论

- **Prefill 快 7 倍**：MoE 只激活 3B 参数 + D2D 优化，长 prompt 处理碾压 27B dense
- **D2D 优化生效**：8K→192K prefill 速度稳定在 4000–5000 tok/s，无性能悬崖
- **196K 满上下文可用**：192K prompt 实测通过（TTFT 46.9s）
- **decode 不受长上下文影响**：192K 上下文下仍 35.7 tok/s
- **选型**：长文档/长上下文用 1919；纯短对话追求生成速度用 8081

## 配置说明

- 256K 上下文（kv-reserve-tokens=262144）会导致专家并行构建（21.8G）阶段 OOM
- 196K（200704）KV 占 3.83 GiB，专家构建顺利通过，剩余 1.37 GiB
- 稳定配置：memory-ratio 0.85、max-prefill 8192、mem-fraction 0.93
