"""Explicit numerical recipes shared by prepared MoE implementations.

``BF16_FC1_WEIGHTED_MXFP8``: FC1 outputs rounded to BF16, SiLU in FP32,
multiplication by the FP32 route weight before MXFP8 quantization of the
intermediate with Cake MegaMoE's exponent rule (scale byte floor 105), and an
FC2 that does not apply the route weight again.
"""

BF16_FC1_WEIGHTED_MXFP8 = "bf16_fc1_silu_fp32_route_weight_mxfp8_floor105"


def validate_numerical_recipe(recipe):
    if recipe not in (None, BF16_FC1_WEIGHTED_MXFP8):
        raise ValueError(f"unsupported MoE numerical_recipe: {recipe!r}")
    return recipe
