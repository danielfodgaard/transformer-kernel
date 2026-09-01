# Pass four — design record: dual-GPU split, GEMM-epilogue fusion, case-8 attention dispatch

Passes 1–3 ended at a measured geomean of **7.11x** (per-case table in the
README). This document is the design record for pass four: where the
remaining time is, which levers were built (all opt-in until measured), the
predictions each one must beat, and what was evaluated and rejected. It was
written in a CPU-only container; every performance number below is a
**prediction** until `notebooks/pass4-t4.ipynb` runs on the user's Kaggle
**2× Tesla T4**. Numerics, by contrast, are already validated here: the new
kernels pass the interpreter suite (`TRITON_INTERPRET=1 pytest
src/test_triton_kernels.py src/test_dual_gpu.py`) and every new path passes
the full harness accuracy check end to end on CPU.

Each design in this pass was drafted and then independently adversarially
reviewed before implementation; two reviews found real errors (§1.1's stream
bug, §2's phantom-transpose premise) that are fixed in what shipped. The
corrections are part of the record.

---

## 0. Erratum to pass 3: the transpose after SDPA was already free

`docs/pass3-research.md` §2.3 claims the
`context.transpose(1, 2).reshape(B, S, D)` after SDPA is "a full extra
read+write of the context tensor per layer". **It is not.** The only
tensor-core SDPA backend on sm_75 is the CUTLASS memory-efficient kernel,
and torch allocates its output as a `[B, S, H, hd]`-**contiguous** tensor
returned as a `transpose(1, 2)` view (verified in
`torch._meta_registrations.meta__scaled_dot_product_efficient_attention`,
whose strides must mirror the CUDA op). Transposing it back and reshaping
is a zero-copy view. Consequences:

* the pass-3 attention kernel's "writes `[B, S, H·hd]` directly, deleting
  the transpose pass" advantage never existed on the SDPA path — one more
  reason it lost;
* the pass-4 fused out_proj GEMM (§2) reads the context through a plain
  contiguous 2-D view — no exotic addressing, and no transpose-deletion
  savings to claim. A real copy exists only on the `math` and new `bmm`
  paths, whose own bmms produce `[B, H, S, hd]`.

The G0 gate cell in the notebook re-verifies the zero-copy claim on the
Kaggle stack (`view.data_ptr() == ctx.data_ptr()`).

## 1. Dual-GPU batch split — `--dual-gpu` (src/dual_gpu.py)

The measurement environment has two T4s; the repo was single-GPU. The batch
dimension is fully independent in this model (LayerNorm is per-row,
attention per `(batch, head)`, GEMM rows independent), so splitting the
batch across GPUs changes **no** per-element arithmetic — the only
deviation channel is cuBLAS/CUTLASS algorithm selection at a different M,
the same chunked-vs-unchunked equivalence `run_case14.validate_proxy`
already measures at ~1e-6.

The hard part is the harness contract: it times `model(x, mask)` with CUDA
events on cuda:0's current stream, so the wrapper must make the end event
transitively cover the cuda:1 half and both PCIe copies with **zero host
syncs** in the timed region. The choreography (module docstring of
`dual_gpu.py` has the full sequence) hangs on PyTorch's cross-device
`copy_` semantics: the copy runs on the *source* device's current stream,
pre-waits the *dest* device's current stream, then post-blocks it. Both the
copy-in and the copy-back are therefore enqueued under **both** stream
contexts (`S0c`, the cuda:0 side stream, and `S1`, the cuda:1 compute
stream); the harness's stream only waits one event (`e_out` on `S0c`) that
is ordered after everything.

> Review note: the first draft enqueued the copy-back under only the `S0c`
> context. Exiting the `S1` context restores cuda:1's *default* stream,
> which — streams being non-blocking — has no ordering against `S1`, so the
> copy would have raced the replica forward and produced garbage in
> `out[n0:]` while the timing looked great. Both reviewers caught it; the
> shipped version holds both contexts, and the two-GPU pytest case plus
> `--dual-verify` exist to catch any regression of this class.

Design points:

* **Replica**: a second `OptimizedTransformer` built lazily on the first
  eligible forward (accuracy trial 1, untimed) via meta-device construction
  + `load_state_dict(assign=True)` with cuda:1 copies of our own state dict.
  Stored in `self.__dict__`, so `state_dict()` keys stay byte-identical for
  the harness's strict weight copy. `_apply` drops it like every other cache.
* **Both half-forwards call `OptimizedTransformer.forward` unbound**: with a
  `GraphedTransformer` base this keeps the half-batch shapes out of the
  capture counters (dual-eligible shapes bypass graphs; smaller shapes still
  capture as before).
* **Split calibration**: eligible call 1 runs an even split (absorbs the
  replica build); call 2 times both halves with events and freezes the
  per-sample-balanced split, clamped to [0.2·B, B/2]; `--dual-fraction`
  pins it instead. Analytically f1 = T/(2T+τ) with τ = 2·bytes(x)/BW —
  ~0.35–0.43 for the eligible cases at 8 GB/s.
* **fp32 on the wire**: quantizing x for transfer would perturb norm1
  statistics vs the single-GPU path; rejected.
* **Dispatch consistency**: the wrapper pins `_dispatch_numel` to the FULL
  batch so `--fp16-max-elements` can never decide differently between the
  split and unsplit forwards of the same call.
* **Eligibility** (all else falls through to the base class unchanged):
  ≥2 devices, CUDA input, batch ≥ 2, `numel ≥ --dual-min-elements`
  (default 4M: admits cases 6/8/13 at 163.8M/8.39M/8.39M elements, excludes
  the rest), dense mask, not autocast, not compiling. `--dual-gpu` is
  therefore always safe to pass; with one visible GPU it is a no-op.
* **run_case14 --dual-gpu**: simpler scheme — timing chunks alternate
  between per-GPU model instances, outputs discarded as today (zero
  inter-GPU traffic), wall-clock timed between dual syncs. Accuracy stays
  single-GPU against the fp32 proxy.

Predictions (balance model, 8 GB/s PCIe mid-case; the G1 gate cell measures
the real bandwidth first): case 6 198.6 → ~128 ms (rel 1.55x → **~11.4x**
total), case 8 25.2 → ~14.4 ms (~8.9x), case 13 18.75 → ~11.1 ms (~28.7x),
case 14 ~155 → ~80 s. Geomean 7.11 → **~8.0x**. Case 5 sits under the
threshold (predicted ≤1.3x; priced anyway with `--dual-min-elements
1000000`).

**Scope caveat**: the hackathon judges "a given GPU model". The dual path is
an environment extension for the 2×T4 measurement box — opt-in, never the
submission configuration, reported separately.

## 2. Fused Triton GEMM epilogues — `--fused-out-proj`, `--fused-ffn` (src/gemm_kernels.py)

Every out_proj and ffn_out feeds residual-add + LayerNorm; ffn_in feeds an
erf-GELU. Pass 4 folds those epilogues into the GEMMs: one kernel computes
`c = a @ Wᵀ + b`, `s = x + c` (fp32), `h = LN(s)` (statistics fp32, output
fp16) with the full output row per program (`BLOCK_N = next_pow2(d_model)`,
d ≤ 128), so the LN reduction is program-local — the same mechanism that
made `add_ln_fwd` win in pass 2. A second kernel does `GELU_erf(a @ Wᵀ + b)`
on the fp32 accumulator (`tl.erf`; the tanh form is deliberately absent).

Why this is not the pass-3 E2 mistake again: the attention kernel lost
because it is a long-inner-loop compute pipeline that sm_75 cannot feed
without `cp.async`. These GEMMs have K = 128 (2–4 `BLOCK_K` iterations —
nothing to pipeline), arithmetic intensity ~22 flops/byte against the T4's
~200 ridge (≈10% of MMA peak saturates DRAM), and they hide latency by
occupancy like the row-wise LN kernels. And they don't have to beat cuBLAS
at GEMM: each fused site deletes the fp16 GEMM-output round trip
(~4 bytes/element/site — the accumulator flows straight into the residual
add, which is also *statistically tighter* numerically) plus a launch;
`--fused-ffn` additionally deletes the separate GELU pass.

> Review note: the first draft targeted the SDPA transpose with a special
> head-split addressing mode and booked ~8·T·D bytes/layer for deleting it.
> Both reviewers proved the transpose is a free view (§0) and the layout
> assumed by that addressing mode was wrong for every H>1 shape. Shipped:
> plain 2-D strides for both sites, and the corrected ~4·T·D/site model.

Predictions (corrected): case 6 −15–31 ms → ~168–186 ms (**8.0–8.8x**);
case 1 ~1.42 → ~1.2–1.3 ms; case 5 ~2.66 → ~2.2–2.4 ms; case 13 ~18.75 →
~17.2–17.9 ms; graphed cases lose ~12 of ~37 in-graph kernels (~35–60 µs
off 0.30–0.43 ms). Envelope: dense path, fp16 compute, d_model ≤ 128 (and
ffn_dim ≤ 128 for the GELU kernel's full-row variant) — case 7 (fp32
dispatch) and case 8 (d=1024) are automatically excluded. Risks that only
the T4 can price: Triton's sm_75 fp16 `tl.dot` efficiency, register
pressure of the 64×128 fp32 accumulator (watch for spills), and case 6's
extreme-value margin under a different fp32 summation order — the 25-trial
× 2-seed stress gates any dispatch adoption.

## 3. Materialized-score bmm attention — `--attention bmm`

Case 8 (d=1024, hd=256) runs at ~26% of fp16 roofline. The arithmetic
budget (profiling cell G3 confirms or kills it): ~11 ms GEMMs + ~4 ms LN
traffic + ~1 ms misc leaves **~5–9 ms in the CUTLASS mem-efficient kernel**,
which head_dim > 128 pushes off its single-value-iteration fast path into
tiny tiles under the 64 KB smem ceiling. At S=128 the score tensor is only
8.4 MB in fp16, so materializing it is cheap: cuBLAS batched QKᵀ → one
fused Triton row-wise scale+causal-mask+fp32-softmax kernel
(`fused_kernels.causal_softmax_fwd` — the no-`tl.dot` kernel class that
wins on sm_75) → cuBLAS PV. Envelope: dense + **causal** + fp16 + score
tensor ≤ 64 MB (auto-excludes case 13's 536 MB) — everything else falls
back to SDPA, exactly like the `triton` option.

Numerics: this path rounds the pre-softmax scores to fp16 once — a
**genuinely new error source** (SDPA keeps scores in fp32 registers through
its softmax; only the probs are rounded). Case 8's current margin is 1.5e-3
of the 2e-3 budget, so the 25-trial stress in G3 is the ship/kill gate, and
the documented fallback if it fails is fp32 QKᵀ (float the q/k operands,
~2× the QK bmm cost, everything else unchanged). Predicted: case 8
25.2 → ~20–22 ms (**~5.9–6.3x**); cases 9/11/12 expected neutral-to-worse
(measured to close the question; the dispatch table adopts only measured,
accuracy-passing wins).

## 4. Small riders

* **`write_sum=False`** on the final fused `add_ln_fwd`/`gemm_add_ln_fwd`
  call: the fp32 residual sum written at the final-norm site was dead
  (~4 bytes/element). On by default — the normalized output is bitwise
  identical (the sum is still computed for the statistics, never stored);
  covered by tests. Worth ~0.13 ms on case 8, noise elsewhere; the F1
  regression prices it implicitly.
* **`--gelu-epilogue`**: `aten._addmm_activation` routes ffn_in through
  cuBLASLt's fused bias+GELU epilogue — but that epilogue computes the
  **tanh approximation** (~1e-3-class deviation vs the baseline's erf).
  Measurement-only; the G5 stress cells decide if any case can absorb it
  (cases 6 and 7 almost certainly cannot). Ignored when `--fused-ffn`
  already covers the site with exact erf.
* **`--graph-streams 2`**: inside CUDA-graph capture, fork the batch into
  two half chains on two streams (event fork/join, capture-legal), so the
  graphed cases' serial per-kernel latencies overlap. Per-element math
  unchanged. Cheap to measure on cases 3/4/12; fully shadowed if the fused
  GEMMs land there.
* **Tools**: `src/profile_case.py` (per-op CUDA table + launches/forward —
  replaces every inferred kernel inventory with a measurement) and
  `src/bench_micro.py` (`gemm` / `sdpa` / `bmmattn` / `p2p` microbenches
  with the harness's event methodology).

## 5. Evaluated and rejected (with the arithmetic)

* **INT8 tensor cores** (130 TOPS): per-token absmax quantization of LN
  outputs gives a rounding σ ≈ 0.011 through the GEMM — 5× the *entire*
  0.002 absolute budget at one σ, 20–50× the current fp16 error. No
  headroom exists; rejected without measurement.
* **Kahan-compensated residual stream** (for case 6's 0.00208 stress
  failure): the residual adds are already fp32 (~1e-6 accumulated); the
  failing 2e-3 comes from fp16 quantization of the branch tensors and
  extreme-value sampling over 4.1B elements. Kahan buys ~1e-6 of a 2e-3
  problem. The honest mitigations stay `--fp16-max-elements` (F3 prices
  it) or a last-layer-fp32 variant (unbuilt, pass-5 candidate).
* **Whole-block Triton megakernel** for the launch-bound cases: plausible
  (the launch-bound regime forgives arithmetic inefficiency) but heavy
  `tl.dot` engineering directly against the E2 lesson, and its adversarial
  review never completed. Deferred to pass 5 pending the G4 probes — if
  `--attention math` ties SDPA inside a graph, the floor is latency, not
  work, and the megakernel's case strengthens.
* **torch._scaled_mm / fp8** (sm_89+), **cublasLt algo enumeration** (needs
  a C++ extension; only the 5-minute TunableOp probe survives in G5),
  **preallocated SDPA output** (no `out=` param; allocator cost is ~µs and
  hidden), **skipping the final `.to(input_dtype)`** (already a no-op —
  `Tensor.to` returns `self` on matching dtype), **CUDA graphs for case 6**
  (the harness keeps the baseline's ~10 GB of transients resident; and a
  198 ms forward has <1% launch overhead to save).

## 6. Measurement matrix

Everything lands in `notebooks/pass4-t4.ipynb`, ordered G0 (gate: kernel
tests on the T4, P2P bandwidth, SDPA-layout probe) → G1 (dual-GPU + same-
session control) → G2 (fused GEMMs + case-6 stress) → G3 (case-8
attribution + bmm + stress) → G4 (launch-bound probes) → G5 (GELU-epilogue
pricing) → F1–F4 (the pass-3 follow-up backlog, still open: best-config
regression, case-7 compile head-to-head, case-6 fp32-dispatch pricing,
attention-triton negative-result repro). Every run writes `results/*.json`;
the summary cell diffs against the pass-2 table and `dispatch.py` distills
accuracy-passing wins into `configs/dispatch.json` per shape.

Decision rules, pre-registered: a kernel wins only if it beats the
same-session control outside the (measured 1.000x) noise floor with
accuracy PASS; case 6 changes additionally need the 25-trial × 2-seed
stress; `--attention bmm` dies (or falls back to fp32 QKᵀ) if its stress
fails; dual-GPU numbers are reported as an environment extension, never as
the submission headline.
