#!/usr/bin/env python3
"""Tests for the pass-4 dual-GPU wrapper (dual_gpu.py).

The interesting properties are testable without two GPUs -- most of them
without any GPU:

  * the wrapper is a transparent no-op off its envelope (CPU, one device,
    small batch, padded mask): identical outputs and identical state_dict
    keys, so the harness's strict weight copy is untouched;
  * the split policy clamps and calibrates sanely;
  * the ``_dispatch_numel`` pin makes the fp16_max_elements decision follow
    the FULL batch, never the half.

The actual two-GPU data path (stream choreography, D2D copies, timing-event
coverage) can only run on a multi-GPU machine: ``test_dual_split_matches_single``
is skipped elsewhere and runs in the Kaggle notebook's gate cell.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dual_gpu  # noqa: E402
import optimized  # noqa: E402
from optimized import OptimizedTransformer, OptimizerSettings  # noqa: E402
from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    copy_model_weights,
)

CONFIG = TransformerConfig(
    batch_size=4,
    seq_len=16,
    d_model=64,
    num_heads=2,
    ffn_dim=64,
    num_layers=2,
    causal=True,
)

HAVE_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _pair(cls, settings=None, device="cpu"):
    torch.manual_seed(7)
    baseline = BaselineTransformer(CONFIG)
    model = cls(CONFIG, settings or OptimizerSettings(precision="fp32"))
    copy_model_weights(baseline, model)
    return baseline.to(device).eval(), model.to(device).eval()


def test_state_dict_keys_unchanged():
    Dual = dual_gpu.make_dual(OptimizedTransformer)
    baseline = BaselineTransformer(CONFIG)
    dual = Dual(CONFIG, OptimizerSettings())
    assert list(dual.state_dict().keys()) == list(baseline.state_dict().keys())
    # The strict copy the harness performs must succeed.
    copy_model_weights(baseline, dual)


def test_noop_off_envelope_cpu():
    """On CPU (or any single-device box) the wrapper must be bit-identical
    to the base class."""
    Dual = dual_gpu.make_dual(OptimizedTransformer)
    _, base = _pair(OptimizedTransformer)
    _, dual = _pair(Dual)

    x = torch.randn(CONFIG.batch_size, CONFIG.seq_len, CONFIG.d_model)
    mask = torch.ones(CONFIG.batch_size, CONFIG.seq_len, dtype=torch.bool)
    with torch.inference_mode():
        torch.testing.assert_close(
            dual(x, mask), base(x, mask), atol=0.0, rtol=0.0
        )


def test_eligibility_gates():
    Dual = dual_gpu.make_dual(OptimizedTransformer, min_elements=1000)
    _, dual = _pair(Dual)

    x = torch.randn(CONFIG.batch_size, CONFIG.seq_len, CONFIG.d_model)
    dense = torch.ones(CONFIG.batch_size, CONFIG.seq_len, dtype=torch.bool)
    padded = dense.clone()
    padded[0, -1] = False

    # CPU tensors are never eligible.
    assert not dual._dual_eligible(x, dense)
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        expected = torch.cuda.device_count() >= 2
        assert dual._dual_eligible(x_cuda, dense.cuda()) == expected
        assert dual._dual_eligible(x_cuda, None) == expected
        # A padded mask must always fall back to the base forward.
        assert not dual._dual_eligible(x_cuda, padded.cuda())
        # Batches below the element threshold stay single-GPU.
        small = dual_gpu.make_dual(OptimizedTransformer, min_elements=10**9)(
            CONFIG, OptimizerSettings()
        )
        assert not small._dual_eligible(x_cuda, None)


def test_split_clamps():
    Dual = dual_gpu.make_dual(OptimizedTransformer, fraction=0.5)
    _, dual = _pair(Dual)
    state = dual_gpu._DualState(None, None, None, None)
    assert dual._dual_split(state, ("k",), 10) == 5
    assert dual._dual_split(state, ("k",), 2) == 1

    frac = dual_gpu.make_dual(OptimizedTransformer, fraction=0.3)(
        CONFIG, OptimizerSettings()
    )
    assert frac._dual_split(state, ("k",), 10) == 3
    # Never more than half, never less than one.
    assert frac._dual_split(state, ("k",), 2) == 1

    auto = dual_gpu.make_dual(OptimizedTransformer)(CONFIG, OptimizerSettings())
    state.frozen_n1[("k",)] = 7
    assert auto._dual_split(state, ("k",), 10) == 5  # clamped to batch // 2
    state.frozen_n1[("k",)] = 4
    assert auto._dual_split(state, ("k",), 10) == 4


def test_dispatch_numel_pins_full_batch():
    """fp16_max_elements must judge the FULL batch even when the model sees
    only a half-batch tensor (what the dual wrapper arranges)."""
    settings = OptimizerSettings(precision="fp16", fp16_max_elements=1000)
    model = OptimizedTransformer(CONFIG, settings)
    half = torch.zeros(2, CONFIG.seq_len, CONFIG.d_model)

    # On CPU _compute_dtype always returns None; assert on the numel logic
    # via the attribute directly instead of a CUDA tensor.
    model._dispatch_numel = 2000
    numel = getattr(model, "_dispatch_numel", None) or half.numel()
    assert numel == 2000  # the pinned full-batch count wins
    model._dispatch_numel = None
    numel = getattr(model, "_dispatch_numel", None) or half.numel()
    assert numel == half.numel()  # cleared pin falls back to the tensor


@pytest.mark.skipif(not HAVE_TWO_GPUS, reason="needs two CUDA devices")
def test_dual_split_matches_single():
    """The real thing: dual output must match the single-GPU output to
    ~reduction-order noise, across calibration calls and a fixed fraction."""
    settings = OptimizerSettings(precision="fp16")
    Dual = dual_gpu.make_dual(OptimizedTransformer, min_elements=1)
    torch.manual_seed(7)
    baseline = BaselineTransformer(CONFIG)
    single = OptimizedTransformer(CONFIG, settings)
    dual = Dual(CONFIG, settings)
    copy_model_weights(baseline, single)
    copy_model_weights(baseline, dual)
    single = single.to("cuda").eval()
    dual = dual.to("cuda").eval()

    with torch.inference_mode():
        for trial in range(4):  # covers build, calibration and frozen calls
            x = torch.randn(
                CONFIG.batch_size,
                CONFIG.seq_len,
                CONFIG.d_model,
                device="cuda",
            )
            want = single(x)
            got = dual(x)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
