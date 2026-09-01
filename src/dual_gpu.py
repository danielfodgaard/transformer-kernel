#!/usr/bin/env python3
"""Opt-in batch-split data parallelism across two GPUs (pass four).

The user's measurement environment is Kaggle's 2x Tesla T4. The benchmark
constructs the input on ``cuda:0`` and times ``model(x, mask)`` with CUDA
events on cuda:0's current stream, so a second GPU is usable only if (a) the
end event transitively covers all cross-device work with zero host syncs in
the timed region, and (b) numerics stay element-identical to the single-GPU
path. Batch splitting satisfies (b) by construction: the model has no
cross-batch operation (LayerNorm is per-row, attention per (batch, head),
GEMM rows independent), so each element sees the identical arithmetic; the
only deviation channel is cuBLAS/CUTLASS algorithm choice at a different M --
the same chunked-vs-unchunked equivalence ``run_case14.validate_proxy``
already measures at ~1e-6.

Choreography (the (a) part), leaning on PyTorch's documented cross-device
``copy_`` semantics -- the copy is enqueued on the SOURCE device's current
stream, first waits on an event recorded on the DEST device's current stream,
then blocks the DEST device's current stream on the copy:

  S0  = the harness's current stream on cuda:0 (looked up fresh every call)
  S0c = a dedicated cuda:0 side stream (both PCIe transfers live here)
  S1  = a dedicated cuda:1 compute stream

  1. record e_in on S0; S0c waits e_in            (input is ready)
  2. [S0c ctx + S1 ctx]  x1 = x[n0:].to(cuda:1)   -> runs on S0c, blocks S1
  3. [device(1) + S1 ctx] y1 = replica forward     -> ~all kernels on S1
  4. [S0 default ctx]     out = empty; y0 = self forward; out[:n0] = y0
  5. [S0c ctx + S1 ctx]  out[n0:].copy_(y1)       -> runs on S1 (source side,
       ordered after the replica forward by stream order), pre-waits S0c
       (nearly empty, so it overlaps GPU0's tail), post-blocks S0c
  6. record e_out on S0c; S0 waits e_out; return out

Step 5's double stream context is load-bearing: with only the S0c context
entered, cuda:1's current stream would be its DEFAULT stream, which has no
ordering against S1 (streams are non-blocking), and the copy would race the
replica forward. Step 3's explicit device context is belt-and-braces for
Triton, whose launcher binds kernels to ``torch.cuda.current_device()``.

The GPU1 work is enqueued BEFORE the GPU0 half so both GPUs crunch while the
host is still dispatching. Transfers cross the wire in the input's dtype
(fp32 for the benchmark) precisely so the replica's LayerNorm statistics see
bit-identical inputs.

The wrapper is a mixin over either ``OptimizedTransformer`` or
``GraphedTransformer``. Both half-forwards call ``OptimizedTransformer``'s
forward UNBOUND, so a graphed base never counts the half-batch shapes toward
CUDA-graph capture (dual-eligible shapes bypass graphs entirely; smaller
shapes still capture and replay as before). Ineligible calls -- fewer than
two GPUs, small batches, padded masks, autocast -- fall through to the base
class unchanged, so ``--dual-gpu`` is always safe to pass.

Split calibration: eligible call 1 (accuracy trial 1, untimed) runs an even
split and absorbs the one-time replica build; call 2 times both halves with
events and freezes the per-sample-balanced split for the rest of the process
(``--dual-fraction`` pins it instead). The competition judges "a given GPU
model", so this path is a documented environment extension -- never the
submission default.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import torch

import optimized
from optimized import OptimizedTransformer

__all__ = ["make_dual", "DEFAULT_MIN_ELEMENTS"]

# Below ~4M elements the doubled host launch work plus two PCIe hops cost
# more than the halved compute saves (cases 1-5, 7, 9-12); cases 6, 8 and 13
# clear it (163.8M / 8.39M / 8.39M elements).
DEFAULT_MIN_ELEMENTS = 4_000_000


class _DualState:
    """Replica model, streams, and per-shape split calibration."""

    __slots__ = ("replica", "dev1", "s0c", "s1", "calls", "frozen_n1")

    def __init__(self, replica, dev1, s0c, s1) -> None:
        self.replica = replica
        self.dev1 = dev1
        self.s0c = s0c
        self.s1 = s1
        self.calls: Dict[Tuple, int] = {}
        self.frozen_n1: Dict[Tuple, int] = {}


def make_dual(
    base_cls: Type[OptimizedTransformer],
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    fraction: Optional[float] = None,
    verify: bool = False,
):
    """Class factory: ``base_cls`` extended with the dual-GPU forward.

    ``fraction`` (0 < f <= 0.5) pins the GPU1 share of the batch and skips
    calibration; ``verify`` prints max|dual - single| on the first eligible
    calls (host-syncing, so only ever active in the untimed accuracy phase).
    """

    class DualGpuTransformer(base_cls):
        _dual_min_elements = min_elements
        _dual_fraction = fraction
        _dual_verify = verify

        # -- lifecycle ------------------------------------------------------ #

        def _apply(self, *args, **kwargs):
            # A device/dtype move invalidates the replica along with every
            # other cache; it rebuilds lazily on the next eligible forward.
            result = super()._apply(*args, **kwargs)
            self.__dict__.pop("_dual", None)
            return result

        def _dual_eligible(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
        ) -> bool:
            if not x.is_cuda or torch.cuda.device_count() < 2:
                return False
            if x.shape[0] < 2 or x.numel() < self._dual_min_elements:
                return False
            if self.settings.precision == "autocast":
                return False
            if not self._mask_is_dense(valid_token_mask):
                return False
            compiling = getattr(
                getattr(torch, "compiler", None), "is_compiling", None
            )
            if compiling is not None and compiling():
                return False
            return True

        @optimized._outside_compile
        def _dual_build(self, x: torch.Tensor) -> _DualState:
            dev0_index = x.device.index if x.device.index is not None else 0
            dev1 = torch.device("cuda", 1 if dev0_index == 0 else 0)
            # Meta construction skips the CPU random init; assign=True swaps
            # the meta parameters for real cuda:1 copies of our weights.
            # state_dict() keys are byte-identical to the baseline's, so this
            # replica is exactly the model the harness validated.
            with torch.device("meta"):
                replica = OptimizedTransformer(self.config, self.settings)
            state = {
                key: value.detach().to(dev1)
                for key, value in self.state_dict().items()
            }
            replica.load_state_dict(state, assign=True)
            replica.eval()
            dual = _DualState(
                replica=replica,
                dev1=dev1,
                s0c=torch.cuda.Stream(device=x.device),
                s1=torch.cuda.Stream(device=dev1),
            )
            # Plain-object attribute: bypasses nn.Module registration, so
            # state_dict() and the strict weight copy never see the replica.
            self.__dict__["_dual"] = dual
            return dual

        def _dual_split(self, state: _DualState, key: Tuple, batch: int) -> int:
            if self._dual_fraction is not None:
                n1 = int(round(batch * self._dual_fraction))
            else:
                n1 = state.frozen_n1.get(key, batch // 2)
            return max(1, min(batch // 2, n1))

        # -- forward -------------------------------------------------------- #

        def forward(
            self,
            x: torch.Tensor,
            valid_token_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if not self._dual_eligible(x, valid_token_mask):
                return super().forward(x, valid_token_mask)
            return self._dual_forward(x)

        @optimized._outside_compile
        def _dual_forward(self, x: torch.Tensor) -> torch.Tensor:
            state = self.__dict__.get("_dual") or self._dual_build(x)
            key = (tuple(x.shape), x.dtype)
            call = state.calls.get(key, 0) + 1
            state.calls[key] = call

            batch = x.shape[0]
            n1 = self._dual_split(state, key, batch)
            n0 = batch - n1
            calibrate = (
                self._dual_fraction is None
                and call == 2
                and key not in state.frozen_n1
            )

            s0 = torch.cuda.current_stream(x.device)
            e_in = torch.cuda.Event()
            e_in.record(s0)
            state.s0c.wait_event(e_in)

            if calibrate:
                ev1_start = torch.cuda.Event(enable_timing=True)
                ev1_start.record(state.s0c)

            # GPU1 half first, so cuda:1 crunches while the host is still
            # dispatching cuda:0's kernels. The double stream context makes
            # the D2D copy run on S0c (source side) and post-block S1.
            with torch.cuda.stream(state.s0c), torch.cuda.stream(state.s1):
                x1 = x[n0:].to(state.dev1, non_blocking=True)

            self._dispatch_numel = x.numel()
            state.replica._dispatch_numel = x.numel()
            try:
                with torch.cuda.device(state.dev1), torch.cuda.stream(state.s1):
                    y1 = OptimizedTransformer.forward(state.replica, x1, None)

                out = torch.empty_like(x)
                if calibrate:
                    ev0_start = torch.cuda.Event(enable_timing=True)
                    ev0_start.record(s0)
                y0 = OptimizedTransformer.forward(self, x[:n0], None)
            finally:
                self._dispatch_numel = None
                state.replica._dispatch_numel = None
            out[:n0].copy_(y0)
            if calibrate:
                ev0_end = torch.cuda.Event(enable_timing=True)
                ev0_end.record(s0)

            # Copy-back: BOTH stream contexts, or cuda:1's current stream
            # would be its default stream and the copy would race the replica
            # forward (see the module docstring). Runs on S1, ordered after
            # the forward; pre-waits S0c so it overlaps GPU0's tail;
            # post-blocks S0c.
            with torch.cuda.stream(state.s0c), torch.cuda.stream(state.s1):
                out[n0:].copy_(y1, non_blocking=True)

            e_out = torch.cuda.Event()
            e_out.record(state.s0c)
            s0.wait_event(e_out)
            out.record_stream(state.s0c)
            x.record_stream(state.s0c)

            if calibrate:
                ev1_end = torch.cuda.Event(enable_timing=True)
                ev1_end.record(state.s0c)
                # Untimed accuracy phase: a host sync here is free.
                torch.cuda.synchronize(x.device)
                torch.cuda.synchronize(state.dev1)
                t0 = ev0_start.elapsed_time(ev0_end)  # n0 samples on GPU0
                t1 = ev1_start.elapsed_time(ev1_end)  # n1 + both transfers
                rate0, rate1 = t0 / max(n0, 1), t1 / max(n1, 1)
                balanced = batch * rate0 / max(rate0 + rate1, 1e-9)
                frozen = max(
                    max(1, int(round(0.2 * batch))),
                    min(batch // 2, int(round(balanced))),
                )
                state.frozen_n1[key] = frozen
                print(
                    f"[dual-gpu] calibrated split for batch {batch}: "
                    f"n1={frozen} (t0={t0:.2f} ms / {n0}, t1={t1:.2f} ms / {n1})"
                )

            if self._dual_verify and call <= 3:
                reference = OptimizedTransformer.forward(self, x, None)
                diff = float((out.float() - reference.float()).abs().max().item())
                print(f"[dual-gpu] verify max_abs(dual - single) = {diff:.3e}")

            return out

    DualGpuTransformer.__name__ = f"Dual{base_cls.__name__}"
    return DualGpuTransformer
