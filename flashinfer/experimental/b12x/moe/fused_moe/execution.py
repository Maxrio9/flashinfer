"""Declarative canonical fused-MoE execution preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import FrozenMapping


@dataclass(frozen=True, kw_only=True)
class ExecutionCapacity:
    """Prefill token capacity and exact planned decode specializations."""

    max_tokens: int
    top_k: int
    warmup_token_counts: tuple[int, ...] = ()
    route_num_experts: int | None = None

    def __post_init__(self):
        max_tokens, top_k = int(self.max_tokens), int(self.top_k)
        if max_tokens <= 0 or top_k <= 0:
            raise ValueError("max_tokens and top_k must be positive")
        counts = tuple(sorted({int(value) for value in self.warmup_token_counts}))
        if any(value <= 0 or value > max_tokens for value in counts):
            raise ValueError("warmup token counts must be positive and within capacity")
        if self.route_num_experts is not None and int(self.route_num_experts) < 0:
            raise ValueError("route_num_experts cannot be negative")
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "warmup_token_counts", counts)
        if self.route_num_experts is not None:
            object.__setattr__(self, "route_num_experts", int(self.route_num_experts))


@dataclass(frozen=True, kw_only=True)
class RoutingSpec:
    """Immutable routing, numerical controls, and retained output layout.

    ``token_sum`` retains the ordinary reduced token output. ``token_and_routes``
    also retains weighted BF16 rows in original token/top-k slot order through
    ``Binding.route_output``. Retained routes require deterministic reduction;
    omitted determinism becomes true and an explicit false is rejected.
    """

    apply_router_weight_on_input: bool = False
    logits_dtype: torch.dtype | None = None
    deterministic_output: bool | None = None
    collect_activation_amax: bool = False
    score_func: str = "softmax"
    renormalize: bool = True
    has_correction_bias: bool = False
    has_image_correction_bias: bool = False
    has_image_mask: bool = False
    routed_scaling_factor: float = 1.0
    output_layout: str = "token_sum"

    def __post_init__(self):
        if self.output_layout not in ("token_sum", "token_and_routes"):
            raise ValueError("MoE output_layout must be token_sum or token_and_routes")
        if self.output_layout == "token_and_routes":
            if (
                self.deterministic_output is not None
                and self.deterministic_output is not True
            ):
                raise ValueError("token_and_routes requires deterministic_output=True")
            object.__setattr__(self, "deterministic_output", True)


def plan_execution(
    *,
    experts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
    scale_expansion=None,
):
    """Return a composite declaration, never an executable plan or warmup handle."""
    from ._preparation import plan

    return plan(
        experts,
        capacity=capacity,
        routing=routing,
        invocation=invocation,
        override=override,
        scale_expansion=scale_expansion,
    )


__all__ = ["ExecutionCapacity", "RoutingSpec", "plan_execution"]
