#!/usr/bin/env python3
"""GPU numerics test for the fused Triton kernels in fused_kernels.py.

Run this on the CUDA box before trusting any sweep with the fused path on:

    python src/test_kernels.py

It compares ln_fwd / add_ln_fwd against the eager op sequence they replace
(across the appendix feature widths plus non-power-of-two sizes, for fp32 and
fp16 outputs), then checks the fused and eager model paths end to end on a
small config with identical weights. Exits non-zero on any failure.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import fused_kernels  # noqa: E402


def eager_ln(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """The op sequence ln_fwd replaces: fp32 LayerNorm, then the cast."""
    normalized = F.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
    )
    return normalized.to(out_dtype)


def check(name: str, got: torch.Tensor, want: torch.Tensor, atol: float) -> bool:
    diff = (got.float() - want.float()).abs().max().item()
    ok = diff <= atol
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={diff:.3e} (atol={atol:g})")
    return ok


def unit_tests(device: torch.device) -> bool:
    ok = True
    eps = 1e-5
    for n_cols in (32, 128, 1024, 33, 100, 768, 2048):
        rows = 257
        x = torch.randn(rows, n_cols, device=device) * 3
        y = torch.randn(rows, n_cols, device=device, dtype=torch.float16)
        weight = torch.randn(n_cols, device=device)
        bias = torch.randn(n_cols, device=device)

        # fp16 outputs may differ by one ulp where the slightly different
        # fp32 accumulation order lands on the far side of a rounding
        # boundary; 1e-3 is ~2 fp16 ulps at unit magnitude.
        for out_dtype, atol in ((torch.float32, 1e-5), (torch.float16, 1e-3)):
            got = fused_kernels.ln_fwd(x, weight, bias, eps, out_dtype)
            want = eager_ln(x, weight, bias, eps, out_dtype)
            ok &= check(f"ln_fwd n={n_cols} out={out_dtype}", got, want, atol)

            sum_got, norm_got = fused_kernels.add_ln_fwd(
                x, y, weight, bias, eps, out_dtype
            )
            sum_want = x + y.to(x.dtype)
            norm_want = eager_ln(sum_want, weight, bias, eps, out_dtype)
            ok &= check(
                f"add_ln_fwd.sum n={n_cols} out={out_dtype}",
                sum_got,
                sum_want,
                1e-6,
            )
            ok &= check(
                f"add_ln_fwd.norm n={n_cols} out={out_dtype}",
                norm_got,
                norm_want,
                atol,
            )

    # 3-D and non-contiguous inputs go through the same wrappers.
    x3 = torch.randn(4, 16, 128, device=device).transpose(0, 1)
    weight = torch.randn(128, device=device)
    bias = torch.randn(128, device=device)
    got = fused_kernels.ln_fwd(x3, weight, bias, eps, torch.float16)
    want = eager_ln(x3, weight, bias, eps, torch.float16)
    ok &= check("ln_fwd non-contiguous 3d", got, want, 1e-3)
    return ok


def end_to_end(device: torch.device) -> bool:
    import optimized
    from torch_transformer_benchmark import (
        BaselineTransformer,
        TransformerConfig,
        copy_model_weights,
    )

    config = TransformerConfig(
        batch_size=4,
        seq_len=16,
        d_model=64,
        num_heads=4,
        ffn_dim=128,
        num_layers=3,
        causal=True,
    )
    base = BaselineTransformer(config)
    fused = optimized.OptimizedTransformer(
        config, replace(optimized.OptimizerSettings(), fused_norm=True)
    )
    eager = optimized.OptimizedTransformer(
        config, replace(optimized.OptimizerSettings(), fused_norm=False)
    )
    copy_model_weights(base, fused)
    copy_model_weights(base, eager)
    fused = fused.to(device).eval()
    eager = eager.to(device).eval()

    torch.manual_seed(1)
    x = torch.randn(
        config.batch_size, config.seq_len, config.d_model, device=device
    )
    mask = torch.ones(
        config.batch_size, config.seq_len, device=device, dtype=torch.bool
    )
    with torch.inference_mode():
        out_fused = fused(x, mask)
        out_eager = eager(x, mask)

    # The matmuls are identical between the two paths; only the LayerNorm
    # accumulation order differs, which can flip single fp16 ulps in the
    # intermediates. Anything past ~1e-3 here means a real kernel bug.
    ok = check("model fused vs eager", out_fused, out_eager, 1e-3)

    # A padded mask must route around the fused path and still work.
    mask_padded = mask.clone()
    mask_padded[:, -4:] = False
    x_padded = x.masked_fill(~mask_padded[..., None], 0)
    with torch.inference_mode():
        out_fused_padded = fused(x_padded, mask_padded)
        out_eager_padded = eager(x_padded, mask_padded)
    ok &= check(
        "model padded (eager fallback)",
        out_fused_padded,
        out_eager_padded,
        1e-6,
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return 1
    if not fused_kernels.HAVE_TRITON:
        print("SKIP: triton is not importable")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(0)

    ok = unit_tests(device)
    ok &= end_to_end(device)

    print("ALL PASS" if ok else "FAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
