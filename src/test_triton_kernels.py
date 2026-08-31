#!/usr/bin/env python3
"""
Pytest suite for the Triton kernels — runnable with or without a GPU.

Covers both kernel modules:
  * ``fused_kernels.py`` — the shipping fused LayerNorm / residual+LayerNorm
    kernels (also covered on-GPU by the ``src/test_kernels.py`` script);
  * ``kernels.py`` — the experimental pass-3 flash-attention forward.

On a CUDA machine:

    pytest src/test_triton_kernels.py -v

Without a GPU, the same tests run through Triton's numpy-backed interpreter —
the environment variable must be set *before* Python imports triton:

    TRITON_INTERPRET=1 pytest src/test_triton_kernels.py -v

Tolerances mirror the competition budget where fp16 storage is involved
(abs 2e-3 / rel 2%) and are effectively-exact (1e-5) for pure-fp32 paths.
Attention references are computed with the same operation order the
benchmark's baseline uses (fp32 scores -> scale -> mask -> fp32 softmax -> PV).
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
import torch

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import fused_kernels  # noqa: E402
import kernels  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
HAVE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAVE_CUDA else "cpu"

pytestmark = pytest.mark.skipif(
    not kernels.triton_available()
    or (not HAVE_CUDA and not INTERPRET),
    reason="needs triton and either a CUDA device or TRITON_INTERPRET=1",
)


def _seeded(*shape, dtype=torch.float32, seed=0, scale=1.0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    return t.to(device=DEVICE, dtype=dtype)


# --------------------------------------------------------------------------- #
# fused_kernels: ln_fwd / add_ln_fwd
# --------------------------------------------------------------------------- #

LN_DIMS = [32, 64, 100, 128, 1024]


@pytest.mark.parametrize("d", LN_DIMS)
@pytest.mark.parametrize("residual_dtype", [None, torch.float16, torch.float32])
def test_fused_layernorm_matches_torch(d, residual_dtype):
    rows = 48
    x = _seeded(rows, d, seed=d)
    weight = _seeded(d, seed=d + 1, scale=0.5) + 1.0
    bias = _seeded(d, seed=d + 2, scale=0.5)
    eps = 1e-5

    if residual_dtype is None:
        y = fused_kernels.ln_fwd(x, weight, bias, eps, out_dtype=torch.float16)
        expected_sum = x
    else:
        residual = _seeded(rows, d, dtype=residual_dtype, seed=d + 3)
        expected_sum = x + residual.to(torch.float32)
        s, y = fused_kernels.add_ln_fwd(
            x, residual, weight, bias, eps, out_dtype=torch.float16
        )
        # The fp32 add is the same elementwise IEEE op the eager path performs.
        torch.testing.assert_close(s, expected_sum, atol=0.0, rtol=0.0)

    ref = torch.nn.functional.layer_norm(
        expected_sum, (d,), weight, bias, eps
    ).to(torch.float16)
    assert y.dtype == torch.float16 and y.shape == x.shape
    torch.testing.assert_close(y, ref, atol=2e-3, rtol=2e-2)


def test_fused_layernorm_fp32_out_is_tight():
    d, rows = 128, 32
    x = _seeded(rows, d, seed=7)
    residual = _seeded(rows, d, seed=8)
    weight = _seeded(d, seed=9) + 1.0
    bias = _seeded(d, seed=10)

    s, y = fused_kernels.add_ln_fwd(x, residual, weight, bias, 1e-5)
    ref = torch.nn.functional.layer_norm(x + residual, (d,), weight, bias, 1e-5)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_fused_layernorm_three_dim_input():
    b, sq, d = 3, 5, 128
    x = _seeded(b, sq, d, seed=21)
    residual = _seeded(b, sq, d, dtype=torch.float16, seed=22)
    weight = _seeded(d, seed=23) + 1.0
    bias = _seeded(d, seed=24)

    s, y = fused_kernels.add_ln_fwd(x, residual, weight, bias, 1e-5)
    ref = torch.nn.functional.layer_norm(
        x + residual.to(torch.float32), (d,), weight, bias, 1e-5
    )
    assert y.shape == (b, sq, d) and s.shape == (b, sq, d)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# kernels.flash_attention
# --------------------------------------------------------------------------- #


def _attention_reference(q, k, v, scale, causal):
    """The baseline's op order: fp32 scores, post-scale, -inf mask, fp32
    softmax, PV in fp32. Returns [B, S, H*hd]."""
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        s = q.shape[-2]
        blocked = torch.ones(
            (s, s), device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(blocked, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, vf)
    b, h, s, hd = context.shape
    return context.transpose(1, 2).reshape(b, s, h * hd)


# (B, H, S, hd) — covers every appendix head_dim (8 pads to 16; 256 uses the
# small-tile config), sequence lengths off the tile grid, and single-head.
ATTN_SHAPES = [
    (2, 4, 128, 32),
    (2, 16, 64, 8),
    (1, 1, 64, 128),
    (2, 4, 96, 32),
    (1, 2, 128, 64),
    (1, 2, 64, 256),
    (1, 4, 32, 16),
]


@pytest.mark.parametrize("shape", ATTN_SHAPES, ids=lambda s: "b{}h{}s{}d{}".format(*s))
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_flash_attention_matches_reference(shape, causal):
    b, h, s, hd = shape
    scale = hd ** -0.5
    q = _seeded(b, h, s, hd, dtype=torch.float16, seed=1)
    k = _seeded(b, h, s, hd, dtype=torch.float16, seed=2)
    v = _seeded(b, h, s, hd, dtype=torch.float16, seed=3)

    out = kernels.flash_attention(q, k, v, scale, causal)
    ref = _attention_reference(q, k, v, scale, causal)

    assert out.shape == (b, s, h * hd)
    torch.testing.assert_close(
        out.float(), ref, atol=2e-3, rtol=2e-2
    )


def test_flash_attention_strided_qkv_views():
    """Feed the kernel the exact views optimized.py produces: permuted slices
    of a fused [B, S, 3, H, hd] projection, no .contiguous() anywhere."""
    b, h, s, hd = 2, 4, 128, 32
    scale = hd ** -0.5
    qkv = _seeded(b, s, 3, h, hd, dtype=torch.float16, seed=42)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    assert not q.is_contiguous()

    out = kernels.flash_attention(q, k, v, scale, causal=True)
    ref = _attention_reference(
        q.contiguous(), k.contiguous(), v.contiguous(), scale, causal=True
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-2)


def test_flash_attention_output_layout_feeds_linear():
    """The [B, S, H*hd] output must equal transpose+reshape of [B, H, S, hd]."""
    b, h, s, hd = 1, 2, 64, 64
    scale = hd ** -0.5
    q = _seeded(b, h, s, hd, dtype=torch.float16, seed=5)
    k = _seeded(b, h, s, hd, dtype=torch.float16, seed=6)
    v = _seeded(b, h, s, hd, dtype=torch.float16, seed=7)

    out = kernels.flash_attention(q, k, v, scale, causal=True)
    sdpa_like = _attention_reference(q, k, v, scale, causal=True)
    # Same tolerance budget; the point is the layout, checked by shape and by
    # element order agreeing with the reference reshape.
    assert out.stride() == (s * h * hd, h * hd, 1)
    torch.testing.assert_close(out.float(), sdpa_like, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("s", [32, 33, 64, 100])
def test_flash_attention_ragged_sequence_lengths(s):
    """S off the tile grid exercises the boundary masks in both loop phases."""
    b, h, hd = 1, 2, 32
    scale = hd ** -0.5
    q = _seeded(b, h, s, hd, dtype=torch.float16, seed=s)
    k = _seeded(b, h, s, hd, dtype=torch.float16, seed=s + 1)
    v = _seeded(b, h, s, hd, dtype=torch.float16, seed=s + 2)

    out = kernels.flash_attention(q, k, v, scale, causal=True)
    ref = _attention_reference(q, k, v, scale, causal=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-2)


def test_supported_probes():
    dev = torch.device(DEVICE)
    assert kernels.attention_supported(8, torch.float16, dev)
    assert kernels.attention_supported(256, torch.float16, dev)
    assert not kernels.attention_supported(512, torch.float16, dev)
    if not INTERPRET:
        assert not kernels.attention_supported(64, torch.float32, dev)
