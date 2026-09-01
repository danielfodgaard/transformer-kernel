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
import gemm_kernels  # noqa: E402
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


# --------------------------------------------------------------------------- #
# fused_kernels: add_ln write_sum variant + causal softmax (pass 4)
# --------------------------------------------------------------------------- #


def test_add_ln_write_sum_false_matches():
    d, rows = 128, 32
    x = _seeded(rows, d, seed=31)
    residual = _seeded(rows, d, seed=32)
    weight = _seeded(d, seed=33) + 1.0
    bias = _seeded(d, seed=34)

    s_ref, h_ref = fused_kernels.add_ln_fwd(x, residual, weight, bias, 1e-5)
    s_none, h = fused_kernels.add_ln_fwd(
        x, residual, weight, bias, 1e-5, write_sum=False
    )
    assert s_none is None and s_ref is not None
    torch.testing.assert_close(h, h_ref, atol=0.0, rtol=0.0)


def _softmax_reference(scores, scale, causal):
    """The baseline's op order: upcast (exact), fp32 scale, -inf mask, fp32
    softmax, cast back to the score dtype."""
    x = scores.float() * scale
    if causal:
        s = scores.shape[-1]
        blocked = torch.ones(
            (s, s), device=scores.device, dtype=torch.bool
        ).triu(diagonal=1)
        x = x.masked_fill(blocked, float("-inf"))
    return torch.softmax(x, dim=-1).to(scores.dtype)


@pytest.mark.parametrize("s", [16, 32, 33, 100, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_causal_softmax_matches_reference(s, dtype, causal):
    bh = 6
    scores = _seeded(bh, s, s, dtype=dtype, seed=s, scale=3.0)
    scale = 0.17677669529663687  # 32 ** -0.5

    out = fused_kernels.causal_softmax_fwd(scores, scale, causal=causal)
    ref = _softmax_reference(scores, scale, causal)

    assert out.shape == scores.shape and out.dtype == scores.dtype
    tol = dict(atol=2e-3, rtol=2e-2) if dtype == torch.float16 else dict(
        atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(out, ref, **tol)
    if causal:
        # Above-diagonal probabilities must be exactly zero.
        upper = torch.ones((s, s), dtype=torch.bool).triu(diagonal=1)
        assert float(out[:, upper].abs().max()) == 0.0


def test_causal_softmax_four_dim_input():
    b, h, s = 2, 3, 64
    scores = _seeded(b, h, s, s, dtype=torch.float16, seed=9)
    out = fused_kernels.causal_softmax_fwd(scores, 0.25)
    ref = _softmax_reference(scores, 0.25, causal=True)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)


# --------------------------------------------------------------------------- #
# gemm_kernels: fused GEMM + bias + residual + LayerNorm / + GELU (pass 4)
# --------------------------------------------------------------------------- #

GEMM_SITES = [
    # (tokens, k_in, n_out) -- out_proj-like (k == n) and ffn_out-like
    # (k != n) shapes, tile-aligned and ragged, incl. a non-pow2 row width.
    (256, 128, 128),
    (100, 128, 128),
    (64, 64, 64),
    (256, 512, 128),
    (48, 128, 100),
]


def _gemm_reference(a, w, b, x, ln_w, ln_b, eps):
    """The eager pair this kernel replaces, minus the intermediate fp16
    rounding of the GEMM output (the kernel feeds LN the fp32 accumulator,
    which is the tighter order)."""
    c = torch.nn.functional.linear(a.float(), w.float(), b.float())
    s = x.float() + c
    h = torch.nn.functional.layer_norm(s, (w.shape[0],), ln_w, ln_b, eps)
    return s, h


@pytest.mark.parametrize("site", GEMM_SITES, ids=lambda s: "t{}k{}n{}".format(*s))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gemm_add_ln_matches_reference(site, dtype):
    if dtype == torch.float32 and not INTERPRET:
        pytest.skip("fp32 tl.dot has no tensor-core path on sm_75; gated off")
    tokens, k_in, n_out = site
    a = _seeded(tokens, k_in, dtype=dtype, seed=1)
    w = _seeded(n_out, k_in, dtype=dtype, seed=2, scale=0.3)
    b = _seeded(n_out, dtype=dtype, seed=3)
    x = _seeded(tokens, n_out, seed=4)
    ln_w = _seeded(n_out, seed=5, scale=0.5) + 1.0
    ln_b = _seeded(n_out, seed=6, scale=0.5)

    # The fp16 case exercises the production configuration (fp16 in, fp16
    # out); the fp32 interpreter case pins the epilogue math tightly with an
    # fp32 output, so no store rounding blurs the comparison.
    out_dtype = torch.float16 if dtype == torch.float16 else torch.float32
    s, h = gemm_kernels.gemm_add_ln_fwd(
        a, w, b, x, ln_w, ln_b, 1e-5, out_dtype=out_dtype
    )
    s_ref, h_ref = _gemm_reference(a, w, b, x, ln_w, ln_b, 1e-5)

    assert s.dtype == x.dtype and h.dtype == out_dtype
    tol = dict(atol=2e-3, rtol=2e-2) if dtype == torch.float16 else dict(
        atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(s, s_ref.to(s.dtype), **tol)
    torch.testing.assert_close(h.float(), h_ref, **tol)


def test_gemm_add_ln_write_sum_false():
    a = _seeded(64, 128, dtype=torch.float16, seed=11)
    w = _seeded(128, 128, dtype=torch.float16, seed=12, scale=0.3)
    b = _seeded(128, dtype=torch.float16, seed=13)
    x = _seeded(64, 128, seed=14)
    ln_w = _seeded(128, seed=15) + 1.0
    ln_b = _seeded(128, seed=16)

    s_ref, h_ref = gemm_kernels.gemm_add_ln_fwd(a, w, b, x, ln_w, ln_b, 1e-5)
    s, h = gemm_kernels.gemm_add_ln_fwd(
        a, w, b, x, ln_w, ln_b, 1e-5, write_sum=False
    )
    assert s is None and s_ref is not None
    torch.testing.assert_close(h, h_ref, atol=0.0, rtol=0.0)


def test_gemm_add_ln_real_sdpa_layout():
    """Feed the wrapper the exact tensor optimized.py produces: the
    mem-efficient SDPA output is [B, S, H, hd]-contiguous returned as a
    transpose(1, 2) view, and the model reshapes it back -- a free view, not
    a copy (pass-4 erratum). The kernel must consume it unchanged."""
    b, h, s, hd = 2, 4, 32, 32
    d = h * hd
    sdpa_native = _seeded(b, s, h, hd, dtype=torch.float16, seed=21)
    context = sdpa_native.transpose(1, 2)  # what SDPA hands back: [B, H, S, hd]
    a = context.transpose(1, 2).reshape(b, s, d)  # what the model builds
    assert a.data_ptr() == sdpa_native.data_ptr()  # genuinely zero-copy

    w = _seeded(d, d, dtype=torch.float16, seed=22, scale=0.2)
    bias = _seeded(d, dtype=torch.float16, seed=23)
    x = _seeded(b, s, d, seed=24)
    ln_w = _seeded(d, seed=25) + 1.0
    ln_b = _seeded(d, seed=26)

    s_out, h_out = gemm_kernels.gemm_add_ln_fwd(
        a, w, bias, x, ln_w, ln_b, 1e-5, out_dtype=torch.float16
    )
    s_ref, h_ref = _gemm_reference(
        a.reshape(-1, d), w, bias, x.reshape(-1, d), ln_w, ln_b, 1e-5
    )
    torch.testing.assert_close(
        s_out.reshape(-1, d), s_ref.to(s_out.dtype), atol=2e-3, rtol=2e-2
    )
    torch.testing.assert_close(
        h_out.reshape(-1, d).float(), h_ref, atol=2e-3, rtol=2e-2
    )


@pytest.mark.parametrize("site", [(256, 128, 128), (100, 64, 128), (64, 128, 100)],
                         ids=lambda s: "t{}k{}n{}".format(*s))
def test_gemm_bias_gelu_matches_erf_reference(site):
    tokens, k_in, n_out = site
    a = _seeded(tokens, k_in, dtype=torch.float16, seed=41)
    w = _seeded(n_out, k_in, dtype=torch.float16, seed=42, scale=0.3)
    b = _seeded(n_out, dtype=torch.float16, seed=43)

    out = gemm_kernels.gemm_bias_gelu_fwd(a, w, b)
    ref = torch.nn.functional.gelu(
        torch.nn.functional.linear(a.float(), w.float(), b.float()),
        approximate="none",
    )
    assert out.dtype == a.dtype and out.shape == (tokens, n_out)
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-2)


def test_gemm_envelope_guards():
    cpu = torch.zeros(4, 4)
    if INTERPRET:
        assert gemm_kernels.can_fuse_gemm(cpu, 128, 128)
    else:
        assert not gemm_kernels.can_fuse_gemm(cpu, 128, 128)
    dummy = cpu if INTERPRET else cpu.cuda() if HAVE_CUDA else cpu
    if HAVE_CUDA or INTERPRET:
        assert not gemm_kernels.can_fuse_gemm(dummy, 1024, 128)  # case 8
        assert not gemm_kernels.can_fuse_gemm(dummy, 128, 8)  # tl.dot K >= 16
        assert not gemm_kernels.can_fuse_gemm(
            dummy, 128, 128, compute_dtype=None
        ) or INTERPRET  # fp32 dispatch stays eager on hardware
