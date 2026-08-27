#!/usr/bin/env python3
"""
Optimized Transformer for the "Implement a GPU Kernel for a Transformer Layer"
challenge (TikTok Jam Session 2026).

This module deliberately lives *outside* the organizers' benchmark file. The
benchmark (``torch_transformer_benchmark.py``) is kept byte-identical to what
the organizers shipped; ``run_case.py`` swaps this class in at runtime. For the
final submission the body of ``OptimizedTransformer`` can be pasted into the
``UserOptimizedTransformer`` stub without any other change.

Pass one is PyTorch-level only -- no custom Triton or CUDA kernels:

  1. The three Q/K/V projections are fused into a single matmul and split.
  2. Attention goes through ``F.scaled_dot_product_attention`` instead of a
     materialized ``[B, H, S, S]`` score tensor.
  3. Matmuls run in fp16 (Turing tensor cores) while the residual stream,
     LayerNorm statistics and the softmax accumulation stay in fp32.
  4. ``torch.compile`` is NOT applied here -- the benchmark already exposes it
     as ``--compile-user``, so its contribution can be measured separately.

Hard constraints imposed by the benchmark (see docs/pass1-decisions.md):

  * ``forward(self, x, valid_token_mask=None) -> Tensor`` is unchanged.
  * The output is ``[batch_size, seq_len, d_model]`` in the input dtype.
  * ``state_dict()`` keys are identical to the baseline's, because
    ``copy_model_weights()`` does a *strict* ``load_state_dict``. Nothing here
    registers a new parameter or a persistent buffer; every fused/cast tensor
    is a plain attribute, invisible to ``state_dict()``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

__all__ = ["OptimizerSettings", "OptimizedTransformer", "configure", "active_settings"]


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimizerSettings:
    """Knobs exposed on the command line by run_case.py.

    Every knob defaults to the configuration we expect to be fastest on a
    Tesla T4 (sm_75), and every knob can be turned off individually so the
    contribution of each optimization can be measured in isolation.
    """

    # fp16   -- weights pre-cast once, matmuls in fp16, residual stream fp32.
    # autocast -- torch.autocast(fp16) around the forward; re-casts weights per
    #             call, which is the cost we want to measure against "fp16".
    # fp32   -- no reduced precision at all (structural wins only).
    precision: str = "fp16"

    # sdpa -- F.scaled_dot_product_attention.
    # math -- the baseline's explicit attention, kept for bisecting numerics.
    attention: str = "sdpa"

    # Force a specific SDPA backend. "auto" lets PyTorch choose; on sm_75 that
    # is the memory-efficient (cutlass) backend, since flash requires sm_80+.
    sdpa_backend: str = "auto"

    # Fuse the three Q/K/V projections into one matmul.
    fuse_qkv: bool = True

    # Skip the one device->host sync per forward that checks whether the
    # padding mask is all-True. Only safe when --padding-ratio is 0.
    assume_dense_mask: bool = False


_ACTIVE = OptimizerSettings()


def configure(**kwargs) -> OptimizerSettings:
    """Set the settings used by ``OptimizedTransformer(config)``.

    The benchmark constructs the model as ``UserOptimizedTransformer(config)``
    with no room for extra arguments, so the configuration is module-level.
    """
    global _ACTIVE
    _ACTIVE = replace(_ACTIVE, **kwargs)
    return _ACTIVE


def active_settings() -> OptimizerSettings:
    return _ACTIVE


# --------------------------------------------------------------------------- #
# Cached per-layer tensors
# --------------------------------------------------------------------------- #


@dataclass
class _LayerWeights:
    """Weights for one block, pre-fused and pre-cast to the compute dtype.

    LayerNorm weights are intentionally *not* cast: LayerNorm runs in fp32 so
    that the variance accumulation keeps full precision.
    """

    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    norm1_eps: float
    qkv_weight: Optional[torch.Tensor]      # [3*d, d]; None when fuse_qkv=False
    qkv_bias: Optional[torch.Tensor]        # [3*d]
    q_weight: Optional[torch.Tensor]        # unfused fallback
    q_bias: Optional[torch.Tensor]
    k_weight: Optional[torch.Tensor]
    k_bias: Optional[torch.Tensor]
    v_weight: Optional[torch.Tensor]
    v_bias: Optional[torch.Tensor]
    out_weight: torch.Tensor
    out_bias: torch.Tensor
    norm2_weight: torch.Tensor
    norm2_bias: torch.Tensor
    norm2_eps: float
    ffn_in_weight: torch.Tensor
    ffn_in_bias: torch.Tensor
    ffn_out_weight: torch.Tensor
    ffn_out_bias: torch.Tensor


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class OptimizedTransformer(BaselineTransformer):
    """Drop-in replacement for ``UserOptimizedTransformer``.

    Subclassing the baseline keeps every submodule -- and therefore every
    ``state_dict`` key -- exactly as the baseline has it, which is what the
    strict ``load_state_dict`` in ``copy_model_weights()`` requires.
    """

    def __init__(
        self,
        config: TransformerConfig,
        settings: Optional[OptimizerSettings] = None,
    ) -> None:
        super().__init__(config)
        self.settings = settings if settings is not None else _ACTIVE

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        # The baseline multiplies the scores by head_dim ** -0.5 after the
        # matmul. SDPA applies the same factor internally, but we pass it
        # explicitly so the two can never drift apart.
        self.attn_scale = self.head_dim**-0.5

        # Lazily-built caches. Plain attributes, so state_dict() never sees
        # them and the strict weight copy keeps working.
        self._weight_cache: Optional[List[_LayerWeights]] = None
        self._weight_cache_key: Optional[tuple] = None
        self._causal_cache: Optional[torch.Tensor] = None
        self._causal_cache_key: Optional[tuple] = None

    # -- cache management --------------------------------------------------- #

    def _apply(self, *args, **kwargs):
        # .to(device=..., dtype=...) funnels through _apply. Any such move
        # invalidates the cast/fused copies.
        result = super()._apply(*args, **kwargs)
        self._weight_cache = None
        self._weight_cache_key = None
        self._causal_cache = None
        self._causal_cache_key = None
        return result

    def _compute_dtype(self, x: torch.Tensor) -> Optional[torch.dtype]:
        """Dtype for the matmuls, or None to leave tensors as they are.

        fp16 is only selected on CUDA: on CPU the fp16 kernels are either
        missing or far slower than fp32. bfloat16 is never selected -- Turing
        (sm_75) has no bf16 tensor cores.
        """
        if self.settings.precision != "fp16":
            return None
        if not x.is_cuda:
            return None
        if x.dtype in (torch.float16, torch.bfloat16):
            return None  # already reduced precision; leave it alone
        return torch.float16

    def _weights(self, compute_dtype: Optional[torch.dtype]) -> List[_LayerWeights]:
        """Fused + pre-cast weights, built once per (device, dtype) combination.

        Safe to cache: ``copy_model_weights()`` loads the state dict and
        ``.to(device, dtype)`` runs before the first forward, and nothing
        mutates the parameters afterwards. ``_apply`` invalidates the cache if
        the model is moved again.
        """
        reference = self.layers[0].attention.q_proj.weight
        key = (reference.device, reference.dtype, compute_dtype, self.settings.fuse_qkv)
        if self._weight_cache is not None and self._weight_cache_key == key:
            return self._weight_cache

        def cast(t: torch.Tensor) -> torch.Tensor:
            return t if compute_dtype is None else t.to(compute_dtype)

        built: List[_LayerWeights] = []
        # no_grad keeps the cache out of any autograd graph. The benchmark runs
        # everything under torch.inference_mode(), so these are inference
        # tensors; they are only ever read back inside the same regime.
        with torch.no_grad():
            for layer in self.layers:
                attn = layer.attention
                if self.settings.fuse_qkv:
                    # Rows of a Linear weight are output features, so cat along
                    # dim 0 produces [q_out | k_out | v_out]. The result is
                    # mathematically identical to three separate matmuls.
                    qkv_weight = cast(
                        torch.cat(
                            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                            dim=0,
                        ).contiguous()
                    )
                    qkv_bias = cast(
                        torch.cat(
                            [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias],
                            dim=0,
                        ).contiguous()
                    )
                    q_w = k_w = v_w = q_b = k_b = v_b = None
                else:
                    qkv_weight = qkv_bias = None
                    q_w, q_b = cast(attn.q_proj.weight), cast(attn.q_proj.bias)
                    k_w, k_b = cast(attn.k_proj.weight), cast(attn.k_proj.bias)
                    v_w, v_b = cast(attn.v_proj.weight), cast(attn.v_proj.bias)

                built.append(
                    _LayerWeights(
                        norm1_weight=layer.norm1.weight,
                        norm1_bias=layer.norm1.bias,
                        norm1_eps=layer.norm1.eps,
                        qkv_weight=qkv_weight,
                        qkv_bias=qkv_bias,
                        q_weight=q_w,
                        q_bias=q_b,
                        k_weight=k_w,
                        k_bias=k_b,
                        v_weight=v_w,
                        v_bias=v_b,
                        out_weight=cast(attn.out_proj.weight),
                        out_bias=cast(attn.out_proj.bias),
                        norm2_weight=layer.norm2.weight,
                        norm2_bias=layer.norm2.bias,
                        norm2_eps=layer.norm2.eps,
                        ffn_in_weight=cast(layer.ffn_in.weight),
                        ffn_in_bias=cast(layer.ffn_in.bias),
                        ffn_out_weight=cast(layer.ffn_out.weight),
                        ffn_out_bias=cast(layer.ffn_out.bias),
                    )
                )

        self._weight_cache = built
        self._weight_cache_key = key
        return built

    def _causal_allow_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Lower-triangular True-means-attend mask, built once per forward.

        The baseline rebuilds its ``[S, S]`` mask inside every layer; this one
        is shared across all layers and cached across calls.

        Note the polarity flip: the baseline builds a triu "block this"
        mask for ``masked_fill``, whereas SDPA's bool ``attn_mask`` marks
        positions to *keep*.
        """
        key = (seq_len, device)
        if self._causal_cache is not None and self._causal_cache_key == key:
            return self._causal_cache
        mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
        self._causal_cache = mask
        self._causal_cache_key = key
        return mask

    def _mask_is_dense(self, valid_token_mask: Optional[torch.Tensor]) -> bool:
        """True when no token is padded, so all masking can be skipped.

        ``valid_token_mask.all()`` costs one device->host sync per forward
        (~20us). That is noise at 14ms but measurable on the small shapes,
        hence the --assume-dense-mask escape hatch.
        """
        if valid_token_mask is None:
            return True
        if self.settings.assume_dense_mask:
            return True
        return bool(valid_token_mask.all())

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        mask = None if self._mask_is_dense(valid_token_mask) else valid_token_mask

        if self.settings.precision == "autocast" and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self._run_layers(x, mask, compute_dtype=None)
        else:
            out = self._run_layers(x, mask, self._compute_dtype(x))

        # Match the baseline's dtype so compare_outputs() does not warn.
        return out.to(input_dtype)

    def _run_layers(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        compute_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        causal = self.config.causal
        weights = self._weights(compute_dtype)
        normalized_shape = (self.d_model,)

        # Precompute the three mask views the baseline needs, once instead of
        # once per layer.
        invalid_tokens = None if mask is None else ~mask[..., None]
        attn_allow = None
        use_is_causal = False
        if mask is None:
            # No padding: SDPA's built-in causal fast path, no mask tensor.
            use_is_causal = causal
        else:
            allow = mask[:, None, None, :]                       # [B, 1, 1, S]
            if causal:
                allow = allow & self._causal_allow_mask(seq_len, x.device)
            attn_allow = allow

        # The residual stream stays in the input dtype (fp32 for the default
        # benchmark run). Only the matmul operands are cast down.
        for layer_weights in weights:
            # ---- attention ----
            h = F.layer_norm(
                x,
                normalized_shape,
                layer_weights.norm1_weight,
                layer_weights.norm1_bias,
                layer_weights.norm1_eps,
            )
            if compute_dtype is not None:
                h = h.to(compute_dtype)

            if layer_weights.qkv_weight is not None:
                qkv = F.linear(h, layer_weights.qkv_weight, layer_weights.qkv_bias)
                # qkv is contiguous, so this view is free. Layout is
                # [B, S, {q,k,v}, H, head_dim] -> three [B, H, S, head_dim].
                qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            else:
                q = self._split_heads(
                    F.linear(h, layer_weights.q_weight, layer_weights.q_bias),
                    batch,
                    seq_len,
                )
                k = self._split_heads(
                    F.linear(h, layer_weights.k_weight, layer_weights.k_bias),
                    batch,
                    seq_len,
                )
                v = self._split_heads(
                    F.linear(h, layer_weights.v_weight, layer_weights.v_bias),
                    batch,
                    seq_len,
                )

            if self.settings.attention == "sdpa":
                with _sdpa_backend(self.settings.sdpa_backend):
                    context = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_allow,
                        dropout_p=0.0,
                        is_causal=use_is_causal,
                        scale=self.attn_scale,
                    )
            else:
                context = self._math_attention(q, k, v, attn_allow, use_is_causal)

            context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
            attn_out = F.linear(
                context, layer_weights.out_weight, layer_weights.out_bias
            )
            if invalid_tokens is not None:
                attn_out = attn_out.masked_fill(invalid_tokens, 0)
            x = x + attn_out.to(x.dtype)

            # ---- feed-forward ----
            h = F.layer_norm(
                x,
                normalized_shape,
                layer_weights.norm2_weight,
                layer_weights.norm2_bias,
                layer_weights.norm2_eps,
            )
            if compute_dtype is not None:
                h = h.to(compute_dtype)
            h = F.linear(h, layer_weights.ffn_in_weight, layer_weights.ffn_in_bias)
            # approximate="none" (erf) matches the baseline exactly; the tanh
            # approximation would introduce a needless ~1e-3 difference.
            h = F.gelu(h, approximate="none")
            h = F.linear(h, layer_weights.ffn_out_weight, layer_weights.ffn_out_bias)
            x = x + h.to(x.dtype)

            if invalid_tokens is not None:
                x = x.masked_fill(invalid_tokens, 0)

        x = F.layer_norm(
            x,
            normalized_shape,
            self.final_norm.weight,
            self.final_norm.bias,
            self.final_norm.eps,
        )
        if invalid_tokens is not None:
            x = x.masked_fill(invalid_tokens, 0)
        return x

    def _split_heads(
        self, projected: torch.Tensor, batch: int, seq_len: int
    ) -> torch.Tensor:
        return projected.view(batch, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _math_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_allow: Optional[torch.Tensor],
        use_is_causal: bool,
    ) -> torch.Tensor:
        """The baseline's attention, for bisecting SDPA numerics.

        Kept element-for-element identical to BaselineSelfAttention: fp32
        softmax, -inf masking, then cast back to the operand dtype.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        if use_is_causal:
            seq_len = q.shape[-2]
            blocked = ~self._causal_allow_mask(seq_len, q.device)
            scores = scores.masked_fill(blocked, float("-inf"))
        if attn_allow is not None:
            scores = scores.masked_fill(~attn_allow, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        return torch.matmul(probs, v)


# --------------------------------------------------------------------------- #
# SDPA backend selection
# --------------------------------------------------------------------------- #


def _sdpa_backend(name: str):
    """Context manager pinning an SDPA backend, or a no-op for "auto".

    Guarded because torch.nn.attention only exists in newer PyTorch builds. On
    sm_75 the interesting choices are "efficient" (cutlass memory-efficient,
    the expected winner) and "math" (the unfused reference path). "flash" needs
    sm_80+ and will raise if forced on a T4.
    """
    if name == "auto":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - depends on the torch build
        return contextlib.nullcontext()

    backends = {
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
    }
    if name not in backends:
        return contextlib.nullcontext()
    return sdpa_kernel(backends[name])
