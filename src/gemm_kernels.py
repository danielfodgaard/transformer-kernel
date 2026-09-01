#!/usr/bin/env python3
"""Triton fused-GEMM kernels for the d_model<=128 shapes (pass four, opt-in).

Two kernels extend the pass-2 fusion story from the elementwise chain into the
GEMM epilogues themselves:

  gemm_add_ln_fwd(a, w, b, x, ln_w, ln_b, eps, ...)  -> (s, h)
      One kernel computing  c = a @ w^T + b  (fp16 operands, fp32 accumulator),
      then  s = x + c  in fp32 (the residual add) and  h = LayerNorm(s)  cast
      to ``out_dtype`` -- statistics in fp32, exactly the op order of the eager
      ``F.linear -> add_ln_fwd`` pair it replaces. Used at the two GEMM sites
      that feed a residual add + LayerNorm: ``out_proj`` (--fused-out-proj) and
      ``ffn_out`` (--fused-ffn).

  gemm_bias_gelu_fwd(a, w, b) -> y
      c = a @ w^T + b, then erf-GELU computed on the fp32 accumulator, stored
      fp16. Replaces ``F.linear -> F.gelu(approximate="none")`` for ``ffn_in``
      (--fused-ffn). ``tl.erf`` is exact-erf, matching the baseline; the tanh
      approximation is never used.

Why this is not the pass-3 attention-kernel mistake again (E2 measured that
tl.dot kernel losing on every shape): that kernel is a long-inner-loop,
compute-heavy pipeline that sm_75 cannot feed without ``cp.async``. These
GEMMs have K = d_model or ffn_dim <= a few hundred -- the K loop is 2-4
iterations, arithmetic intensity ~22 flops/byte against the T4's ~200
flops/byte ridge, so the kernel only needs ~10% of MMA peak to be DRAM-bound,
and it hides latency through occupancy exactly like the row-wise LN kernels
that won in pass 2. What the fusion buys is bytes and launches: each fused
site skips one fp16 GEMM-output round trip (~4 bytes/element/site) and one
kernel launch; ``--fused-ffn`` additionally deletes the separate GELU pass.

Numerics: the epilogue feeds the residual add and the LayerNorm the UNROUNDED
fp32 GEMM accumulator, where the eager chain first rounds the GEMM output to
fp16. Per-site this is statistically tighter than the eager chain; it is not
element-wise identical (different fp32 summation order than cuBLAS), which is
the same reduction-order noise class the shipped LN kernels carry. Case 6's
extreme-value margin must be re-stressed before any default flip (notebook).

Layout note (pass-4 erratum to docs/pass3-research.md §2.3): the mem-efficient
SDPA kernel already returns its output as a transposed VIEW of a
[B, S, H, hd]-contiguous tensor, so ``context.transpose(1, 2).reshape(B, S, D)``
is free and ``a`` arrives here as a plain contiguous 2-D view -- there is no
transpose copy to delete, and these kernels use ordinary row-major addressing.

One program computes a [BLOCK_M, BLOCK_N] tile where BLOCK_N covers the FULL
output row (next_pow2(N), N <= 128), so the LayerNorm mean/var are
program-local reductions -- the same mechanism that makes add_ln_fwd fast.
Config table is fixed (no autotune: capture-safe, and matches the repo's
fixed-table convention).

Everything is opt-in behind run_case.py flags until measured on the T4;
``can_fuse_gemm()`` gates the envelope and callers fall back to the eager pair.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

import os

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

# One program owns a full output row, so N is capped by the fp32 accumulator's
# register budget (BLOCK_M x BLOCK_N x 4B). 128 covers every appendix shape
# that reaches the fp16 path (d=32 dispatches to fp32, d=1024 is excluded).
MAX_FUSED_N = 128
# K is looped in BLOCK_K steps; bounded only to keep the loop count sane.
MAX_FUSED_K = 4096


def can_fuse_gemm(
    x: torch.Tensor,
    n_out: int,
    k_in: int,
    compute_dtype: Optional[torch.dtype] = torch.float16,
) -> bool:
    """Whether the fused GEMM kernels apply to this site.

    ``x`` locates the activations (CUDA, or CPU under the interpreter);
    ``compute_dtype`` is the dtype the GEMM operands will be in -- it must be
    fp16 (fp32 tl.dot has no tensor-core path on sm_75, so the fp32 dispatches
    keep the eager pair). The output row must be narrow enough for one
    program to own it whole.
    """
    if not HAVE_TRITON:
        return False
    if not (x.is_cuda or _INTERPRET):
        return False
    if compute_dtype != torch.float16 and not _INTERPRET:
        return False
    return n_out <= MAX_FUSED_N and 16 <= k_in <= MAX_FUSED_K


def _config(tokens: int, block_n: int) -> Tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_K, num_warps, num_stages) for the T4 (64 KB smem,
    no cp.async so num_stages stays at 2). Small-token shapes take a narrower
    tile for parallelism; the fp32 accumulator (BLOCK_M x BLOCK_N x 4B) sets
    the warp count."""
    block_m = 64 if tokens >= 4096 else 32
    num_warps = 8 if block_m * block_n >= 8192 else 4
    return block_m, 32, num_warps, 2


if HAVE_TRITON:

    @triton.jit
    def _gemm_add_ln_kernel(
        a_ptr, w_ptr, b_ptr,          # GEMM: a [T, K] fp16, w [N, K] fp16, b [N]
        x_ptr,                        # residual [T, N]
        ln_w_ptr, ln_b_ptr,           # LayerNorm affine [N] (fp32)
        sum_ptr, out_ptr,             # outputs: s [T, N], h [T, N]
        T, N, K,
        eps,
        stride_am, stride_ak,
        stride_xm, stride_sm, stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WRITE_SUM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < T
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            # w is [N, K] row-major; load the [BLOCK_K, BLOCK_N] tile of w^T.
            wt = tl.load(
                w_ptr + offs_n[None, :] * K + offs_k[:, None],
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, wt)

        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

        # Residual add in fp32 on the unrounded accumulator (the eager chain
        # rounds the GEMM output to fp16 first -- this is the tighter order).
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        s = tl.where(n_mask[None, :], x + acc, 0.0)
        if WRITE_SUM:
            tl.store(
                sum_ptr + offs_m[:, None] * stride_sm + offs_n[None, :],
                s.to(sum_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )

        mean = tl.sum(s, axis=1) / N
        centered = tl.where(n_mask[None, :], s - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        ln_w = tl.load(ln_w_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        h = centered * rstd[:, None] * ln_w[None, :] + ln_b[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            h.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _gemm_bias_gelu_kernel(
        a_ptr, w_ptr, b_ptr, out_ptr,
        T, N, K,
        stride_am, stride_ak, stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < T
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            wt = tl.load(
                w_ptr + offs_n[None, :] * K + offs_k[:, None],
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, wt)

        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

        # Exact erf-GELU on the fp32 accumulator; the baseline uses
        # approximate="none", so the tanh form is deliberately absent.
        y = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def gemm_add_ln_fwd(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
    write_sum: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Fused ``F.linear(a, weight, bias)`` + residual add + LayerNorm.

    ``a`` is [..., K] in the compute dtype, ``weight`` [N, K], ``x`` the
    residual stream [..., N]. Returns ``(s, h)`` like
    ``fused_kernels.add_ln_fwd``: ``s = x + (a @ w^T + b)`` in ``x``'s dtype
    (None when ``write_sum=False`` -- the final-norm site where it is dead)
    and ``h = LayerNorm(s)`` in ``out_dtype``.
    """
    assert HAVE_TRITON, "gemm_add_ln_fwd called without triton; guard with can_fuse_gemm()"
    out_dtype = x.dtype if out_dtype is None else out_dtype

    n_out, k_in = weight.shape
    shape = x.shape
    a2 = a.reshape(-1, k_in)
    if not (a2.stride(-1) == 1 and a2.stride(0) >= k_in):
        a2 = a2.contiguous()
    x2 = x.contiguous().view(-1, n_out)
    tokens = x2.shape[0]

    s = torch.empty_like(x2) if write_sum else x2  # dummy ptr when unused
    h = torch.empty_like(x2, dtype=out_dtype)

    block_n = triton.next_power_of_2(n_out)
    block_m, block_k, num_warps, num_stages = _config(tokens, block_n)
    _gemm_add_ln_kernel[(triton.cdiv(tokens, block_m),)](
        a2, weight.contiguous(), bias, x2, ln_weight, ln_bias,
        s, h,
        tokens, n_out, k_in,
        eps,
        a2.stride(0), a2.stride(1),
        x2.stride(0), s.stride(0), h.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        WRITE_SUM=write_sum,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (s.view(shape) if write_sum else None), h.view(shape)


def gemm_bias_gelu_fwd(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Fused ``F.linear(a, weight, bias)`` + exact erf-GELU, output in
    ``a``'s dtype. ``a`` is [..., K], ``weight`` [N, K]."""
    assert HAVE_TRITON, "gemm_bias_gelu_fwd called without triton; guard with can_fuse_gemm()"

    n_out, k_in = weight.shape
    a2 = a.reshape(-1, k_in)
    if not (a2.stride(-1) == 1 and a2.stride(0) >= k_in):
        a2 = a2.contiguous()
    tokens = a2.shape[0]

    out = torch.empty((tokens, n_out), dtype=a.dtype, device=a.device)

    block_n = triton.next_power_of_2(n_out)
    block_m, block_k, num_warps, num_stages = _config(tokens, block_n)
    _gemm_bias_gelu_kernel[(triton.cdiv(tokens, block_m),)](
        a2, weight.contiguous(), bias, out,
        tokens, n_out, k_in,
        a2.stride(0), a2.stride(1), out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.view(*a.shape[:-1], n_out)
