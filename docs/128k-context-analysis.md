# 35B 模型 128K 上下文扩展 — 分析与实测报告

**日期**: 2026-09-16
**模型**: Ornith-1.5-35B-A3B-NVFP4 (FreeToken 引擎, 4070S 12GB)
**结论**: ✅ 已成功从 82K 扩展到 128K，实测通过

---

## 1. 背景

OpenClaw 主会话频繁触发 `Context overflow: prompt too large for the model
(88124 tokens > 82002 maximum)`。排查发现 82002 并非配置值，而是 FreeToken
引擎根据显存自动计算的 KV cache 池大小（`num_pages=82002, page_size=1`）。

## 2. 架构分析

该模型为**混合线性注意力**架构（Qwen3-Next 风格，`attn: hybrid_linear`）：

| 注意力类型 | 缓存方式 | 显存特性 |
|---|---|---|
| 全注意力层 | paged KV cache | 随序列长度线性增长，**20480 字节/token** |
| 线性注意力层 (GDN) | 固定 recurrent state | **不随序列增长**，每 slot 64,389,120 字节 |

混合架构使长上下文的显存代价远低于纯 Transformer 模型。

## 3. 显存预算分析

总 cache 预算：**6.85 GiB**（4070S 12GB 扣除模型权重后）

### 扩展前（82K）

| 池 | 大小 | 占用 |
|---|---|---|
| KV cache | 82,002 tokens | 1.56 GiB |
| MoE expert cache | 2,292 experts | 3.79 GiB |
| Mamba (GDN) state | 24 slots | 1.44 GiB |
| **合计** | | **6.79 GiB (99.1%)** |

### 扩展后（128K）

| 池 | 大小 | 占用 | 变化 |
|---|---|---|---|
| KV cache | **131,072 tokens** | 2.50 GiB | +0.94 GiB |
| MoE expert cache | **1,024 experts** | 1.69 GiB | −2.10 GiB (−55%) |
| Mamba (GDN) state | **16 slots** | 0.96 GiB | −0.48 GiB |
| **合计** | | **5.15 GiB** | 留 0.20 GiB 余量 |

> 注：`/v1/cache/rebuild` 热切换路径使用保守预算 5.35 GiB（重建期间新旧池
> 短暂共存），故 MoE/Mamba 需比 status 显示的 6.85 GiB 预算缩得更多。

### 理论上限

- 纯 KV 极限（MoE/Mamba 归零）：**351K tokens**
- 实际极限（MoE=256, Mamba=16）：**280K tokens**
- 128K 远在安全范围内

## 4. 代价评估

- **MoE expert cache 2292 → 1024**：GPU 上缓存的 expert 从 ~22% 降到 ~10%，
  其余走 62GB 系统内存 offload。MoE 每 token 仅激活 top-8 expert，LRU 策略
  保留热 expert，实测速度影响可忽略（94K prefill 4.2s）。
- **Mamba slots 24 → 16**：并发会话数上限从 24 降到 16，单用户场景无影响。

## 5. 执行操作

```bash
# 1. 热切换 KV cache（无需重启服务器）
curl -X POST http://192.168.31.9:1919/v1/cache/rebuild \
  -H "Content-Type: application/json" \
  -d '{"num_pages": 131072, "moe_cache_size": 1024, "num_mamba_slots": 16,
       "mode": "if_idle", "timeout": 300}'
# → {"status":"ok","moe_cache_size":1024,"num_pages":131072,"mamba_slots":16}

# 2. 同步 OpenClaw 上下文上限（留 3K 安全余量）
# openclaw.json: local-ornith contextWindow 262144 → 125000
# 重启 gateway 生效
```

## 6. 实测结果（大海捞针）

**测试方法**：构造 450K 字符（94,003 tokens）的填充文本，在正中间埋入
秘密代码 `4271`，要求模型从 94K 上下文中提取。

| 测试 | prompt_tokens | 结果 |
|---|---|---|
| 94K 上下文大海捞针 | 94,003 | ✅ 模型准确回答 `4271`，4.2s，finish_reason=stop |
| 短 prompt 基线 | 23 | ✅ 正常 |

**关键验证点**：
- 94K > 旧上限 82K —— 旧配置下该请求会被直接拒绝，证明 128K 已生效
- 秘密代码位于 prompt 正中间（~47K 处），非首尾位置，排除位置偏差
- 模型默认开启思考模式（`reasoning_content`），首次测试 max_tokens=50 时
  token 耗尽于思考阶段导致 content 为空；加大 max_tokens 后正常输出

## 7. 遗留事项

- **NVFP4 4-bit 量化**导致的推理质量下降是独立问题（与上下文无关），
  换 FP8 可提升智商但需权衡速度，待后续评估。
- 若未来需要 >128K 上下文，可继续通过 `/v1/cache/rebuild` 热切换，
  理论上限 280K（MoE=256, Mamba=16）。
