"""Per-row (per-channel) FP8 routed-expert MoE dispatch (offload ``_expert_gemm``
"fp8_tensor" branch).

The experts live in the offload cache as per-row fp8 (fp8-e4m3 weight + bf16
per-output-row ``weight_scale``) -- the compressed-tensors ``float-quantized``
per-channel layout (e.g. Ornith-1.5-35B-A3B-FP8). The grouped GEMMs read the
routed experts' fp8 rows directly and dequantize inside the K-loop
(``kernel/triton/fp8_tensor_moe``), so no gather/copy or separate bf16 dequant
of the experts is ever materialized. Same entry points as ``fused_fp8_block``.
"""

from __future__ import annotations


def fused_experts_fp8_tensor(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, num_experts, activation="silu",
    apply_router_weight_on_input=False, act_alpha=1.0, act_limit=float("inf"),
):
    """Prefill: W8A8 fused grouped GEMM over the materialized-layer banks
    (``[num_experts, ...]``, position == expert id)."""
    assert not apply_router_weight_on_input
    from freetoken.kernel.triton.fp8_tensor_moe import fused_experts_fp8_tensor as _impl

    return _impl(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, num_experts, activation, act_alpha, act_limit,
    )


def fused_experts_decode_fp8_tensor(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, activation="silu", apply_router_weight_on_input=False,
    act_alpha=1.0, act_limit=float("inf"),
):
    """Decode: W8A16 fused inline-dequant grouped GEMV -- reads the routed experts' fp8 rows
    directly (``topk_ids`` index the banks) and dequantizes in the K-loop. CUDA-graph safe."""
    assert not apply_router_weight_on_input
    from freetoken.kernel.triton.fp8_tensor_moe import fused_experts_decode_fp8_tensor as _impl

    return _impl(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, activation, act_alpha, act_limit,
    )


__all__ = ["fused_experts_fp8_tensor", "fused_experts_decode_fp8_tensor"]
