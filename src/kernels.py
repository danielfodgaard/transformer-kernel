#!/usr/bin/env python3
"""
Experimental FlashAttention-2-style Triton forward for sm_75 (pass three).

Design (docs/pass3-research.md): fp16 operands, fp32 accumulation, exact
online softmax with scores scaled in fp32 *after* the QK^T dot (the
baseline's ``matmul -> * scale`` order). Two things the CUTLASS
memory-efficient SDPA kernel cannot do here:

  * the output is written directly in ``[B, S, H*hd]`` layout, deleting the
    ``transpose(1, 2).reshape`` pass before the output projection, and
  * Q/K/V are consumed through arbitrary strides straight out of the fused
    ``[B, S, 3, H, hd]`` QKV projection, no ``.contiguous()`` copies.

**Measured result (Tesla T4, results/pass3-e2-attn.json): it loses to the
CUTLASS SDPA kernel on every appendix shape** — e.g. case 11 (head_dim 8)
7.47 ms vs 3.47 ms eager, case 13 (S=1024) 187 ms vs 28.8 ms. Likely causes,
untested: no ``cp.async`` on sm_75 leaves Triton's pipeliner without an
async-copy stage while the CUTLASS kernel hand-pipelines; ``tl.exp`` rather
than the folded ``exp2`` trick; conservative tile/warp choices. Kept as a
documented negative result and a base for future tuning — dispatched only
when ``--attention triton`` is set explicitly, never by default. The fused
residual+LayerNorm kernels that DID win live in ``fused_kernels.py``.

Turing sizing rules baked into the config table:
  * 64 KB usable shared memory per SM, no ``cp.async`` -> ``num_stages <= 2``.
  * fp16 tensor cores via ``mma.sync.m16n8k8``; ``tl.dot`` needs K >= 16, so
    head_dim 8 (appendix case 11) is zero-padded to 16.
  * fp32 ``tl.dot`` would fall back to FMA loops on sm_75, so the kernel is
    only dispatched for fp16 inputs (the d_model<64 fp32 dispatch keeps
    using SDPA).

If Triton is missing, or a shape/dtype/mask falls outside the envelope,
``attention_supported()`` returns False and callers use SDPA instead.

Testable without a GPU through Triton's interpreter (``TRITON_INTERPRET=1``,
numpy-backed) — see src/test_triton_kernels.py.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:  # Triton ships with CUDA builds of torch; guard anyway.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    triton = None
    tl = None
    _HAVE_TRITON = False

__all__ = [
    "triton_available",
    "attention_supported",
    "flash_attention",
]

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

# Attention tile table, keyed by the padded head dim. Values chosen for the
# T4's 64 KB shared-memory ceiling with num_stages<=2 (see the module
# docstring); every entry keeps BLOCK_M % BLOCK_N == 0 so the causal loop can
# split into full off-diagonal tiles plus masked diagonal tiles.
#   padded_hd: (BLOCK_M, BLOCK_N, num_warps, num_stages)
_ATTN_CONFIGS = {
    16: (128, 64, 4, 2),
    32: (128, 64, 4, 2),
    64: (64, 64, 4, 2),
    128: (64, 32, 8, 2),
    256: (32, 32, 8, 1),
}
_ATTN_MAX_HEAD_DIM = 256


def triton_available() -> bool:
    return _HAVE_TRITON


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _on_supported_device(device: torch.device) -> bool:
    # The interpreter runs kernels on CPU tensors; real launches need CUDA.
    return device.type == "cuda" or _INTERPRET


def attention_supported(
    head_dim: int, dtype: torch.dtype, device: torch.device
) -> bool:
    if not (_HAVE_TRITON and _on_supported_device(device)):
        return False
    if dtype != torch.float16 and not _INTERPRET:
        return False  # fp32 tl.dot has no tensor-core path on sm_75
    return _next_pow2(max(head_dim, 16)) <= _ATTN_MAX_HEAD_DIM


# --------------------------------------------------------------------------- #
# FlashAttention-2-style forward for sm_75
# --------------------------------------------------------------------------- #

if _HAVE_TRITON:

    @triton.jit
    def _attn_tile_update(
        acc, m_i, l_i, q, k, v, scale, offs_m, offs_n, S,
        APPLY_CAUSAL: tl.constexpr,
        APPLY_SEQ_MASK: tl.constexpr,
    ):
        # scores: [BLOCK_M, BLOCK_N] fp32 accumulator from fp16 operands.
        scores = tl.dot(q, tl.trans(k)) * scale
        if APPLY_SEQ_MASK:
            scores = tl.where(offs_n[None, :] < S, scores, float("-inf"))
        if APPLY_CAUSAL:
            scores = tl.where(
                offs_m[:, None] >= offs_n[None, :], scores, float("-inf")
            )

        # Online softmax (Milakov & Gimelshein 2018 / FlashAttention-2).
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v)
        return acc, m_new, l_i

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, O,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        H, S,
        scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # Flattened 1-D grid: program -> (batch, head, query tile). A 1-D
        # grid sidesteps the 65535 limit on higher grid dimensions (case 6
        # runs B*H = 40000).
        pid = tl.program_id(0)
        num_m = tl.cdiv(S, BLOCK_M)
        m_block = pid % num_m
        bh = pid // num_m
        b = bh // H
        h = bh % H

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < S
        d_mask = offs_d < HEAD_DIM

        q_ptrs = (
            Q
            + b * stride_qb
            + h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh

        if IS_CAUSAL:
            # Phase 1: KV tiles strictly below the diagonal — no causal mask
            # needed. BLOCK_M % BLOCK_N == 0 guarantees tile alignment, and
            # diag_start <= S so no sequence-boundary mask either.
            diag_start = m_block * BLOCK_M
            end_n = tl.minimum(diag_start + BLOCK_M, S)
        else:
            diag_start = S  # phase 1 covers everything
            end_n = S

        for start_n in range(0, diag_start, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_tile = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                mask=(offs_n[:, None] < S) & d_mask[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                mask=(offs_n[:, None] < S) & d_mask[None, :],
                other=0.0,
            )
            acc, m_i, l_i = _attn_tile_update(
                acc, m_i, l_i, q, k_tile, v_tile, scale, offs_m, offs_n, S,
                APPLY_CAUSAL=False,
                APPLY_SEQ_MASK=not IS_CAUSAL,
            )

        if IS_CAUSAL:
            # Phase 2: diagonal tiles, masked elementwise. Every query row
            # keeps at least its own key, so m_i is finite after this loop.
            for start_n in range(diag_start, end_n, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_tile = tl.load(
                    k_base
                    + offs_n[:, None] * stride_kn
                    + offs_d[None, :] * stride_kk,
                    mask=(offs_n[:, None] < S) & d_mask[None, :],
                    other=0.0,
                )
                v_tile = tl.load(
                    v_base
                    + offs_n[:, None] * stride_vn
                    + offs_d[None, :] * stride_vk,
                    mask=(offs_n[:, None] < S) & d_mask[None, :],
                    other=0.0,
                )
                acc, m_i, l_i = _attn_tile_update(
                    acc, m_i, l_i, q, k_tile, v_tile, scale, offs_m, offs_n, S,
                    APPLY_CAUSAL=True,
                    APPLY_SEQ_MASK=True,
                )

        out = acc / l_i[:, None]
        o_ptrs = (
            O
            + b * stride_ob
            + h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_ok
        )
        tl.store(
            o_ptrs,
            out.to(O.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Exact causal attention; returns the context in ``[B, S, H*hd]``.

    ``q``/``k``/``v`` are logically ``[B, H, S, hd]`` and may be arbitrarily
    strided in the first three dims (e.g. views straight out of the fused
    ``[B, S, 3, H, hd]`` QKV projection); the last dim must be contiguous.
    No padding-mask support — callers fall back to SDPA when one is present.
    """
    B, Hh, S, hd = q.shape
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1, (
        "flash_attention needs contiguous head_dim"
    )

    block_d = _next_pow2(max(hd, 16))
    block_m, block_n, num_warps, num_stages = _ATTN_CONFIGS[block_d]

    out = torch.empty((B, S, Hh * hd), dtype=q.dtype, device=q.device)
    # (b, h, s, d) strides into the flattened [B, S, H*hd] output.
    stride_ob, stride_om = out.stride(0), out.stride(1)
    stride_oh, stride_ok = hd, 1

    grid = (B * Hh * triton.cdiv(S, block_m),)
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        stride_ob, stride_oh, stride_om, stride_ok,
        Hh, S,
        scale,
        HEAD_DIM=hd,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
