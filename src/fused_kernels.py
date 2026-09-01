#!/usr/bin/env python3
"""Triton fused kernels for the fp32 residual stream (pass two).

Between the GEMMs, one transformer block spends roughly nine eager kernels on
the elementwise chain: two LayerNorms, two casts down to the compute dtype,
two casts back up, two residual adds, and a GELU -- each one a full read and
write of the activation through DRAM. On the D=128 appendix shapes that chain,
not the matmuls, is where the optimized forward spends its time (the pass-one
sweep runs at ~4% of the T4's fp16 peak on those cases).

Two fused kernels replace most of it:

  ln_fwd(x, w, b, eps, out_dtype)        -> h
      LayerNorm over the last dim, statistics in fp32, output cast to
      ``out_dtype`` in the same kernel.

  add_ln_fwd(x, y, w, b, eps, out_dtype) -> (s, h)
      s = x + y computed in fp32 (the residual add, with y typically the
      fp16 branch output), then h = LayerNorm(s) cast to ``out_dtype`` --
      one read of each operand, s never round-trips DRAM between the ops.

Because every residual add in the block structure is immediately followed by
a LayerNorm (norm2 after the attention add; the next layer's norm1 or the
final norm after the FFN add), these two kernels carry the entire residual
stream: per forward, ~9L+1 elementwise launches become 2L+1 fused launches
plus L GELUs.

Numerics match the eager path where it matters: the add and every LayerNorm
statistic are computed in fp32, and the cast to the compute dtype happens
last, exactly as the eager sequence add -> F.layer_norm -> .to() does for the
default fp32 activations. (For an fp16 activation stream the fused add is
*more* precise than eager -- fp32 add before the store instead of an fp16
add -- which only shrinks the diff against the fp32 baseline.)

Callers must fall back to eager when ``can_fuse()`` is False: no Triton (CPU
wheels), non-CUDA tensors, or rows wider than MAX_FUSED_SIZE (one block must
hold a full row; every appendix shape is far below the limit).

For the final submission paste, this module's contents can be prepended to
the ``UserOptimizedTransformer`` stub -- nothing here depends on the runtime
patching in run_case.py.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

# Triton's numpy interpreter runs these kernels on CPU tensors for testing.
_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

# One program normalizes one row, so the row must fit in a single block.
MAX_FUSED_SIZE = 8192


def can_fuse(x: torch.Tensor) -> bool:
    """Whether the fused kernels apply to activations shaped like ``x``."""
    return (
        HAVE_TRITON
        and (x.is_cuda or _INTERPRET)
        and x.shape[-1] <= MAX_FUSED_SIZE
    )


def _num_warps(block: int) -> int:
    if block <= 256:
        return 1
    if block <= 512:
        return 2
    if block <= 2048:
        return 4
    return 8


if HAVE_TRITON:

    @triton.jit
    def _ln_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        n_cols,
        eps,
        x_row_stride,
        out_row_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(
            x_ptr + row * x_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)

        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * rstd * w + b

        tl.store(
            out_ptr + row * out_row_stride + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _add_ln_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        sum_ptr,
        out_ptr,
        n_cols,
        eps,
        x_row_stride,
        y_row_stride,
        sum_row_stride,
        out_row_stride,
        BLOCK: tl.constexpr,
        WRITE_SUM: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(
            x_ptr + row * x_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        y = tl.load(
            y_ptr + row * y_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)

        s = x + y
        if WRITE_SUM:
            tl.store(
                sum_ptr + row * sum_row_stride + cols,
                s.to(sum_ptr.dtype.element_ty),
                mask=mask,
            )

        mean = tl.sum(s, axis=0) / n_cols
        centered = tl.where(mask, s - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        h = centered * rstd * w + b

        tl.store(
            out_ptr + row * out_row_stride + cols,
            h.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def ln_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """LayerNorm over the last dim, fp32 statistics, output in ``out_dtype``."""
    assert HAVE_TRITON, "ln_fwd called without triton; guard with can_fuse()"
    out_dtype = x.dtype if out_dtype is None else out_dtype

    shape = x.shape
    n_cols = shape[-1]
    x2 = x.contiguous().view(-1, n_cols)
    out = torch.empty_like(x2, dtype=out_dtype)

    block = triton.next_power_of_2(n_cols)
    _ln_kernel[(x2.shape[0],)](
        x2,
        weight,
        bias,
        out,
        n_cols,
        eps,
        x2.stride(0),
        out.stride(0),
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return out.view(shape)


def add_ln_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
    write_sum: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Fused residual add + LayerNorm.

    Returns ``(s, h)`` where ``s = x + y`` in ``x``'s dtype (the residual
    stream) and ``h = LayerNorm(s)`` in ``out_dtype`` (the next block input).
    The add and the statistics are computed in fp32 regardless of dtypes.

    ``write_sum=False`` skips the store of ``s`` and returns ``(None, h)``:
    the final-norm site consumes only ``h``, so the fp32 residual write there
    is dead -- pass four stopped paying for it (~4 bytes/element). ``h`` is
    bitwise identical either way; the sum is still computed for the
    statistics, just never stored.
    """
    assert HAVE_TRITON, "add_ln_fwd called without triton; guard with can_fuse()"
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    out_dtype = x.dtype if out_dtype is None else out_dtype

    shape = x.shape
    n_cols = shape[-1]
    x2 = x.contiguous().view(-1, n_cols)
    y2 = y.contiguous().view(-1, n_cols)
    s = torch.empty_like(x2) if write_sum else x2  # dummy ptr when unused
    h = torch.empty_like(x2, dtype=out_dtype)

    block = triton.next_power_of_2(n_cols)
    _add_ln_kernel[(x2.shape[0],)](
        x2,
        y2,
        weight,
        bias,
        s,
        h,
        n_cols,
        eps,
        x2.stride(0),
        y2.stride(0),
        s.stride(0),
        h.stride(0),
        BLOCK=block,
        WRITE_SUM=write_sum,
        num_warps=_num_warps(block),
    )
    return (s.view(shape) if write_sum else None), h.view(shape)


# --------------------------------------------------------------------------- #
# Fused causal softmax (pass four, backs the opt-in --attention bmm path)
# --------------------------------------------------------------------------- #

# One program normalizes one score row; S must fit a single block.
MAX_SOFTMAX_SIZE = 8192

if HAVE_TRITON:

    @triton.jit
    def _causal_softmax_kernel(
        scores_ptr,
        out_ptr,
        seq_len,
        scale,
        in_row_stride,
        out_row_stride,
        BLOCK: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        # Row r of the flattened [B*H*S, S] score matrix attends over keys;
        # its query position within the sequence is r % seq_len.
        row = tl.program_id(0)
        q_pos = row % seq_len
        cols = tl.arange(0, BLOCK)
        mask = cols < seq_len

        x = tl.load(
            scores_ptr + row * in_row_stride + cols, mask=mask, other=float("-inf")
        ).to(tl.float32)
        # Baseline op order: matmul -> * scale (fp32) -> mask -> fp32 softmax.
        # fp16 -> fp32 is exact, so scaling after the upcast matches the
        # baseline's post-matmul fp32 scale bit for bit.
        x = x * scale
        if CAUSAL:
            x = tl.where(cols <= q_pos, x, float("-inf"))

        m = tl.max(x, axis=0)
        e = tl.exp(x - m)  # -inf rows -> 0; col <= q_pos keeps >= 1 finite entry
        p = e / tl.sum(e, axis=0)

        tl.store(
            out_ptr + row * out_row_stride + cols,
            p.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def can_softmax(scores: torch.Tensor) -> bool:
    return (
        HAVE_TRITON
        and (scores.is_cuda or _INTERPRET)
        and scores.shape[-1] <= MAX_SOFTMAX_SIZE
        and scores.stride(-1) == 1
    )


def causal_softmax_fwd(
    scores: torch.Tensor, scale: float, causal: bool = True
) -> torch.Tensor:
    """Fused scale + causal mask + fp32 softmax over the last dim.

    ``scores`` is [..., S, S] (typically [B*H, S, S] fp16 straight out of a
    batched QK^T matmul). One kernel replaces the eager upcast / scale /
    masked_fill / softmax / downcast chain; the mask is an index compare, so
    no mask tensor is ever materialized. Statistics and the exp/sum run in
    fp32; output is in the input dtype.
    """
    assert HAVE_TRITON, "causal_softmax_fwd called without triton; guard with can_softmax()"
    seq_len = scores.shape[-1]
    assert scores.shape[-2] == seq_len, "causal softmax expects square [S, S] scores"

    s2 = scores.reshape(-1, seq_len)
    if s2.stride(-1) != 1:
        s2 = s2.contiguous()
    out = torch.empty_like(s2)

    block = triton.next_power_of_2(seq_len)
    _causal_softmax_kernel[(s2.shape[0],)](
        s2,
        out,
        seq_len,
        scale,
        s2.stride(0),
        out.stride(0),
        BLOCK=block,
        CAUSAL=causal,
        num_warps=_num_warps(block),
    )
    return out.view(scores.shape)
