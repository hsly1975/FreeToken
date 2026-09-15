"""fp8 e4m3 experts with one bf16 scale per output row (per-channel / per-row).

The compressed-tensors ``float-quantized`` per-channel layout (e.g.
Ornith-1.5-35B-A3B-FP8): each expert projection stores an fp8-e4m3 weight plus a
``[N, 1]`` bf16 ``weight_scale`` (one scale per output row). The grouped GEMMs
dequantize inside the K-loop (``kernel/triton/fp8_tensor_moe``).
"""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, ExpertView, fused_piece, gated_epilogue_reason, limit_or_inf, MoEConfig, MoEKernel, MoEMethod

FP8 = torch.float8_e4m3fn


class TritonFp8TensorMoEKernel(MoEKernel):
    name = "triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=True, tp_ok=False, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"fp8 tensor MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "fp8 tensor MoE kernel cannot apply the router weight on the input"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h), FP8),
            "gate_up_scale": BankSpec((2 * i,), torch.bfloat16),
            "down": BankSpec((h, i), FP8),
            "down_scale": BankSpec((h,), torch.bfloat16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["down"].copy_(pieces["down"])
        # the per-row scale pieces arrive as [1, N, 1] (checkpoint [N, 1] unsqueezed);
        # the banks are [E, N], so squeeze the trailing singleton before the copy
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale").reshape(out["gate_up_scale"].shape))
        out["down_scale"].copy_(pieces["down_scale"].reshape(out["down_scale"].shape))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_fp8_tensor import fused_experts_decode_fp8_tensor, fused_experts_fp8_tensor

        t = view.tensors
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            n = view.n if view.n is not None else layer.num_experts
            return fused_experts_fp8_tensor(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, n, layer.activation, layer.apply_router_weight_on_input, alpha, limit)
        return fused_experts_decode_fp8_tensor(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input, alpha, limit)


@register_method(QuantKind.FP8_TENSOR, LayerKind.MOE)
class Fp8TensorMoEMethod(MoEMethod):
    candidates = (TritonFp8TensorMoEKernel,)

    def create_weights(self, layer) -> None:
        g = self.cfg
        e, i, h = g.num_experts, g.intermediate, g.hidden
        layer.gate_up_proj = torch.empty(e, 2 * i, h, dtype=FP8)
        layer.gate_up_scale = torch.empty(e, 2 * i, dtype=torch.bfloat16)
        layer.down_proj = torch.empty(e, h, i, dtype=FP8)
        layer.down_scale = torch.empty(e, h, dtype=torch.bfloat16)

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({
            "gate_up": layer.gate_up_proj, "gate_up_scale": layer.gate_up_scale,
            "down": layer.down_proj, "down_scale": layer.down_scale,
        })
