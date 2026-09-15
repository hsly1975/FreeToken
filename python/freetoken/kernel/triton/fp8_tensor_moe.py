"""Per-row (per-channel) FP8 (W8A16 decode / W8A8 prefill) grouped-MoE kernels.

Reads the routed experts' fp8-e4m3 weights directly and dequantizes inside the
K-loop using one bf16 scale per output row (``[E, N]``), so the grouped MoE GEMM
never materializes a gather/copy or a separate bf16 dequant of the experts.
Structure mirrors ``fp8_blockscale_moe`` with the 128x128 block scale replaced
by a per-row vector.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


# ==============================================================================
# Decode: W8A16 inline-dequant grouped GEMV (one route per program row).
# ==============================================================================
@triton.jit
def _decode_fp8_tensor_moe_kernel(
    a_ptr,        # [M, K] activation (bf16)
    w_ptr,        # [E, N, K] float8_e4m3fn
    s_ptr,        # [E, N] bf16 (per-row weight scale)
    c_ptr,        # [M, TOP_K, N] output
    topk_weights_ptr, topk_ids_ptr,
    total_routes, N, K,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_se, stride_sn,
    stride_cm, stride_ck, stride_cn,
    stride_twm, stride_twk, stride_tidm, stride_tidk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tidm + route_k * stride_tidk).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am
    w_slot = w_ptr + slot * stride_we
    s_slot = s_ptr + slot * stride_se

    # per-row scale: one bf16 value per output row, same for all K-blocks
    sc = tl.load(s_slot + offs_n * stride_sn, mask=n_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kb in range(tl.cdiv(K, BLOCK_K)):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        if e4m3_native_cx():
            w = tl.load(
                w_slot + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(
                w_slot + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :], other=0,
            ))
        a = tl.load(a_base + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1)

    acc *= sc
    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(topk_weights_ptr + token_id * stride_twm + route_k * stride_twk)
    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=(route_id < total_routes) & n_mask)


def _decode_gemm(a, w, s, c, topk_weights, topk_ids, mul_routed_weight, a_row_is_route):
    M, top_k = topk_ids.shape
    N = w.shape[1]
    total_routes = M * top_k
    w = e4m3_kernel_view(w)
    BLOCK_N = 16
    grid = (total_routes, triton.cdiv(N, BLOCK_N))
    _decode_fp8_tensor_moe_kernel[grid](
        a, w, s, c, topk_weights, topk_ids, total_routes, N, w.shape[2],
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        s.stride(0), s.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1), topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=128, TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route, MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_TL.get(c.dtype, tl.bfloat16), num_warps=4,
    )


def fused_experts_decode_fp8_tensor(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, activation="silu", act_alpha=1.0, act_limit=float("inf"),
) -> torch.Tensor:
    """Decode (bs-1) inline-dequant per-row-fp8 MoE."""
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _decode_gemm(hidden_states, gate_up, gate_up_scale, ic1, topk_weights, topk_ids, False, False)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _decode_gemm(ic2, down, down_scale, ic3, topk_weights, topk_ids, True, True)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


# ==============================================================================
# Prefill: M-tiled W8A8 grouped GEMM (moe_align_block_size).
# ==============================================================================
@triton.jit
def _prefill_fp8_tensor_moe_kernel(
    a_ptr, w_ptr, s_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    a_scale_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_se, stride_sn,
    stride_asm, stride_ask,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """W8A8: ``a`` is fp8 (per-token act scale ``a_scale``), ``w`` fp8 (per-row scale ``s``);
    ``tl.dot`` runs on fp8 tensor cores, both scales applied per tile."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_rows = offs_token // top_k
    a_ptrs = a_ptr + (a_rows[:, None] * stride_am + offs_k[None, :] * stride_ak)
    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    w_base = w_ptr + slot * stride_we + offs_bn[None, :] * stride_wn
    # per-row scale: one bf16 per output row
    s_base = s_ptr + slot * stride_se
    wsc = tl.load(s_base + offs_bn * stride_sn, mask=offs_bn < N, other=0.0).to(tl.float32)
    as_base = a_scale_ptr + a_rows * stride_asm

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kmask = offs_k[None, :] < K - kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & kmask, other=0.0)
        w_mask = offs_k[:, None] < K - kb * BLOCK_SIZE_K
        if e4m3_native_cx():
            w = tl.load(w_base + offs_k[:, None] * stride_wk, mask=w_mask, other=0.0)
        else:
            w = e4m3_u8_to_f32(tl.load(w_base + offs_k[:, None] * stride_wk, mask=w_mask, other=0.0)).to(tl.bfloat16)
        asc = tl.load(as_base + kb * stride_ask, mask=token_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, w) * asc[:, None] * wsc[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_base += BLOCK_SIZE_K * stride_wk

    if MUL_ROUTED_WEIGHT:
        mw = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0.0)
        acc = acc * mw[:, None]
    acc = acc.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc, mask=token_mask[:, None] & (offs_cn[None, :] < N))


def _prefill_gemm(a_fp8, a_scale, w, s, c, tw, sorted_ids, expert_ids, ntpp, num_valid,
                  kernel_top_k, mul_routed_weight, cfg):
    N, K = w.shape[1], w.shape[2]
    EM = sorted_ids.shape[0]
    w = e4m3_kernel_view(w)
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)  # noqa: E731
    _prefill_fp8_tensor_moe_kernel[grid](
        a_fp8, w, s, c, tw, sorted_ids, expert_ids, ntpp, a_scale, N, K, EM, num_valid,
        a_fp8.stride(0), a_fp8.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        s.stride(0), s.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        c.stride(1), c.stride(2), tw.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=kernel_top_k,
        compute_type=_TL.get(c.dtype, tl.bfloat16), **cfg,
    )


def fused_experts_fp8_tensor(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, num_experts, activation="silu", act_alpha=1.0, act_limit=float("inf"),
) -> torch.Tensor:
    """Prefill inline-dequant per-row-fp8 MoE."""
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul
    from freetoken.kernel.triton.fp8_block_linear import per_token_group_quant_fp8
    from freetoken.moe.fused import moe_align_block_size

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = dict(BLOCK_SIZE_M=64 if M > 64 else 16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
               GROUP_SIZE_M=8, num_warps=8 if M > 64 else 4, num_stages=3)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    a1_fp8, a1_scale = per_token_group_quant_fp8(hidden_states, 128)
    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(a1_fp8, a1_scale, gate_up, gate_up_scale, ic1, tw, sorted_ids, expert_ids, ntpp,
                  num_valid, top_k, False, cfg)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    a2_fp8, a2_scale = per_token_group_quant_fp8(ic2, 128)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(a2_fp8, a2_scale, down, down_scale, ic3, tw, sorted_ids, expert_ids, ntpp,
                  num_valid, 1, True, cfg)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_fp8_tensor",
    "fused_experts_fp8_tensor",
    "_decode_fp8_tensor_moe_kernel",
    "_prefill_fp8_tensor_moe_kernel",
]
