# transformer-kernel

My entry for **Task 3 — Implement a GPU Kernel for a Transformer Layer**, part of
the TikTok Jam Session 2026 hackathon.

## The challenge

The task is to speed up a Transformer layer on a specific GPU while keeping the
output numerically equivalent to a reference PyTorch implementation.

Organisers provide a benchmark script (a PyTorch and a TensorFlow variant — one
is enough) containing a baseline Transformer and an empty
`UserOptimizedTransformer` class. Participants fill in that class with a faster
implementation, then run the script, which:

1. checks accuracy against the baseline, element-wise, and
2. measures inference latency for both implementations.

**Correctness rule** — every output element must satisfy
`abs(user - ref) <= atol` **or** `abs(user - ref) <= rtol * abs(ref)`, with
relative error < 0.02 and absolute error < 0.002.

Test cases sweep a range of input shapes (large/small batch size, sequence
length, and hidden dimension), so different shapes may warrant different kernels
selected by a shape check inside the layer.

Suggested optimisation directions from the problem statement: operator fusion,
memory layout optimisation, reduced-precision compute, tensor core usage,
softmax optimisation, and custom CUDA, Triton, TensorFlow, or PyTorch
implementations.

In scope: AI-assisted code generation, GPU kernel fusion, profiling tools.
Out of scope: production-ready deployment.

## This repository

I picked the **PyTorch** track.

| Path | Description |
| --- | --- |
| `src/torch_transformer_benchmark.py` | The organisers' benchmark harness, **unmodified**. Contains `BaselineTransformer`, the `UserOptimizedTransformer` stub, the accuracy comparison, and the latency benchmark. |
| `src/optimized.py` | The optimized implementation. Subclasses `BaselineTransformer`; fused QKV projection, `scaled_dot_product_attention`, fp16 matmuls with an fp32 residual stream, fused Triton residual+LayerNorm kernels on the dense path. |
| `src/fused_kernels.py` | Custom Triton kernels: fused LayerNorm+cast, fused residual-add+LayerNorm+cast (statistics in fp32; the dead final residual store is skipped since pass 4), and the pass-4 fused causal softmax backing `--attention bmm`. |
| `src/gemm_kernels.py` | Pass-4 fused Triton GEMM epilogues (opt-in): GEMM+bias+residual+LayerNorm for out_proj/ffn_out (`--fused-out-proj`, `--fused-ffn`) and GEMM+bias+erf-GELU for ffn_in. |
| `src/dual_gpu.py` | Pass-4 opt-in batch-split data parallelism for 2x-T4 environments (`--dual-gpu`): stream/event choreography that keeps the harness's cuda:0 event timing sound with zero host syncs; numerics element-identical to single-GPU. An environment extension, never the submission path. |
| `src/kernels.py` | Experimental FlashAttention-2-style Triton forward for sm_75 (`--attention triton`). Measured slower than SDPA on the T4 — kept as a documented negative result; see the pass-3 section. |
| `src/profile_case.py` | Per-op CUDA time attribution for one shape (steady-state, past graph capture) — replaces inferred kernel inventories with measurements. |
| `src/bench_micro.py` | Isolated microbenchmarks: cuBLAS GEMM TFLOPS, the SDPA head_dim cliff, the bmm-attention chain, and cuda:0<->cuda:1 P2P bandwidth. |
| `src/test_kernels.py` | GPU numerics test for the Triton kernels (unit level and fused-vs-eager end to end), pass-4 kernels included. Run it before sweeping with the fused paths on. |
| `src/test_triton_kernels.py` | Pytest twin of the above covering `fused_kernels.py`, `gemm_kernels.py` and `kernels.py`; runs on a GPU or CPU-only via `TRITON_INTERPRET=1` (Triton's numpy interpreter). |
| `src/test_dual_gpu.py` | Dual-GPU wrapper tests: no-op transparency off the envelope, eligibility gates, split clamping (CPU-runnable), plus the real two-GPU equivalence test (runs in the notebook gate). |
| `src/run_case.py` | Runs the harness with `UserOptimizedTransformer` swapped for ours, so the organisers' file stays untouched. All of its flags pass through. |
| `src/run_case14.py` | Out-of-core runner for appendix case 14, which the official harness cannot grade on any hardware. Chunked fp16 candidate vs a chunked fp32 proxy reference, validated against the true baseline at a feasible sequence length. |
| `src/sweep.py` | Runs a set of shapes (one subprocess each) and writes the numbers to `results/*.json`. |
| `src/dispatch.py` | Data-driven shape dispatch: scans `results/*.json`, keeps the fastest accuracy-passing settings per shape, and writes `configs/dispatch.json`; `run_case.py --dispatch` applies it. |
| `configs/shapes.json` | The 14 appendix test shapes. |
| `configs/best.json` | The appendix shapes annotated with the best-known harness-level flags per case (compile everywhere except case 6). |
| `docs/pass1-decisions.md` | Why each optimisation was chosen, what stays in fp32, and how to bisect an accuracy failure. |
| `docs/pass3-research.md` | Research survey (what current attention-kernel work does and doesn't transfer to sm_75), per-regime bottleneck analysis, attention-kernel design record, and the measured addendum. Carries a pass-4 erratum on the SDPA output layout. |
| `docs/pass4-plan.md` | Pass-4 design record: dual-GPU split, GEMM-epilogue fusion, the case-8 bmm attention dispatch, riders, the rejected-with-arithmetic list, and the pre-registered decision rules for the measurement session. |
| `notebooks/pass3-t4.ipynb` | Kaggle notebook: kernel tests on the T4, best-config regression, and the pass-3 follow-up measurements. **Superseded by `pass4-t4.ipynb`** — its F1–F4 cells never ran (the branch it checks out was deleted when pass 3 merged) and are carried forward there. |
| `notebooks/pass4-t4.ipynb` | Kaggle 2x-T4 notebook: gate (kernel tests, P2P bandwidth, SDPA-layout probe), the pass-4 experiment matrix G1–G5, and the inherited F1–F4 backlog. |

The upstream script was last updated 27 August 2026. It is kept byte-identical
to what the organisers shipped; for submission the body of `OptimizedTransformer`
is pasted into the `UserOptimizedTransformer` stub.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch          # CUDA build matching your GPU — see pytorch.org
```

## Reproducing the results

Sweep every appendix shape and write the numbers to `results/`:

```bash
python src/sweep.py --skip 14 --out results/pass1-fp16.json
```

Case 14 is skipped because the *baseline* cannot run it anywhere: its
materialized `[32, 16, 100000, 100000]` score tensor is ~20 TB, and the fp32
input activation alone is 13.1 GB on a 15 GB T4 — see
`docs/pass1-decisions.md`. The optimized path *can* run it; `run_case14.py`
does so out of core and grades it against a validated fp32 proxy reference:

```bash
python src/run_case14.py --max-samples 4   # quick read (~a few minutes)
python src/run_case14.py                   # full accuracy pass + timing
```

Anything after `--` is forwarded to the run, so
precision modes and `torch.compile` are reachable from the sweep:

```bash
python src/sweep.py --cases 1,7,8,13 --out results/fp32.json -- --precision fp32
python src/sweep.py --cases 1 --out results/noise-floor.json -- --reference-check
```

A single shape, with the harness's own flags:

```bash
python src/run_case.py --causal --batch-size 64 --d-model 128 \
  --heads 4 --seq-len 128 --layers 4 --ffn-dim 128

python src/run_case.py --precision autocast --compile-user
```

Kernel numerics tests — the script needs a GPU, the pytest suite also runs
CPU-only through Triton's interpreter:

```bash
python src/test_kernels.py                            # GPU
pytest src/test_triton_kernels.py                     # GPU
TRITON_INTERPRET=1 pytest src/test_triton_kernels.py  # CPU-only machine
```

The stock baseline-vs-baseline harness still runs on its own:

```bash
python src/torch_transformer_benchmark.py --help
```

Accuracy thresholds default to the competition values (`--rtol 0.02`,
`--atol 0.002`). Timing defaults are 20 warmup iterations and 100 repeats over
3 rounds, alternating baseline and optimized to cancel clock drift.

## Test shapes

All 14 shape combinations the test cases sweep (from the problem statement
appendix). Every case is causal.

| # | Batch | QKV dim | Heads | Seq len | Layers | Causal | FFN dim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

Case 1 is the reference point; cases 2-6 vary batch size, 7-8 vary the model
dimension, 9-11 vary head count, and 12-14 vary sequence length. Case 14 is the
extreme one: 100k tokens at `d_model` 1024, which is where a memory-efficient
attention kernel matters most.

Mapped onto the benchmark's flags, case 1 is:

```bash
python src/torch_transformer_benchmark.py --causal \
  --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
```

## Deliverables

- **Devpost project description** — how the solution addresses the problem, plus
  dev tools, APIs, libraries, and datasets used.
- **This public repository** — commented code and a README covering the project
  overview, setup, reproduction steps, limitations, and contributions.
- **Demo video** — a short public YouTube walkthrough linked from Devpost.
  A walkthrough of API usage, inference examples, and result analysis is
  acceptable for a backend track like this one.
- **Tech report** — the environment (CPU, GPU, disk), the optimisations applied,
  the AI skills and tools used, and the final test results.

## Judging criteria

| Criterion | Weight |
| --- | --- |
| Technical Execution | 35% |
| Innovation & Problem Insight | 20% |
| Impact & Relevance | 20% |
| Feasibility & Practicality | 15% |
| Presentation & Communication (final event only) | 10% |

## Results (measured)

Tesla T4, torch 2.10.0+cu128, fp32 baseline, fp16 optimized path, harness
defaults (5 accuracy trials, 20 warmup, 3x100 alternating repeats, median of
300 CUDA-event samples). Raw data in `results/pass1-fp16.json`; the
baseline-vs-baseline noise floor measured 1.000x (`results/noise-floor.json`).
All 13 runnable cases pass accuracy with zero failed elements.

| # | Shape (b/s/d/h/l/ffn) | Baseline ms | Optimized ms | Speedup | max_abs |
| --- | --- | --- | --- | --- | --- |
| 1 | 64/128/128/4/4/128 | 9.44 | 2.44 | **3.87x** | 1.4e-03 |
| 2 | 1/128/128/4/4/128 | 3.25 | 1.48 | 2.19x | 1.4e-03 |
| 3 | 4/128/128/4/4/128 | 3.55 | 1.53 | 2.32x | 1.6e-03 |
| 4 | 16/128/128/4/4/128 | 3.31 | 1.43 | 2.32x | 1.5e-03 |
| 5 | 128/128/128/4/4/128 | 18.41 | 4.96 | 3.71x | 1.7e-03 |
| 6 | 10000/128/128/4/4/128 | 1486.38 | 407.55 | 3.65x | 1.9e-03 |
| 7 | 64/128/32/4/4/32 | 6.31 | 1.43 | 4.41x | 2.0e-03 |
| 8 | 64/128/1024/4/4/1024 | 140.57 | 34.09 | 4.12x | 1.5e-03 |
| 9 | 64/128/128/1/4/128 | 6.51 | 2.59 | 2.51x | 1.4e-03 |
| 10 | 64/128/128/2/4/128 | 8.00 | 2.60 | 3.08x | 1.6e-03 |
| 11 | 64/128/128/16/4/128 | 22.45 | 3.47 | **6.46x** | 1.4e-03 |
| 12 | 64/32/128/4/4/128 | 3.17 | 1.38 | 2.30x | 1.5e-03 |
| 13 | 64/1024/128/4/4/128 | 324.52 | 28.83 | **11.26x** | 1.5e-03 |

Geometric mean speedup: **3.56x** eager.

### With torch.compile (CUDA graphs)

`--compile-user --compile-mode reduce-overhead` on top of the same fp16 path
(`results/pass2-fp16-compiled.json`) lifts every case, most of all the
launch-overhead-bound ones, whose ~1.4 ms floor drops to ~0.7 ms:

| # | Eager | Compiled | Optimized ms |
| --- | --- | --- | --- |
| 1 | 3.87x | **6.00x** | 1.63 |
| 2 | 2.19x | **4.20x** | 0.69 |
| 3 | 2.32x | **4.11x** | 0.70 |
| 4 | 2.32x | **3.92x** | 0.73 |
| 5 | 3.71x | **6.54x** | 2.94 |
| 6 | 3.65x | OOM (see below) | 407.55 (eager) |
| 7 | 4.41x | **7.39x** | 0.85 |
| 8 | 4.12x | **5.10x** | 28.59 |
| 9 | 2.51x | **4.05x** | 1.66 |
| 10 | 3.08x | **4.97x** | 1.66 |
| 11 | 6.46x | **8.91x** | 2.57 |
| 12 | 2.30x | **4.11x** | 0.75 |
| 13 | 11.26x | **16.32x** | 19.92 |

Geometric mean with the best configuration per case (compiled everywhere,
eager for case 6): **5.52x**. Case 6 compiled hits CUDA out of memory - not in
our code, but in the *baseline's* forward: CUDA graphs pin ~2.8 GB of private
pools for the compiled model, and at batch 10000 the baseline's ~10 GB of
transient fp32 score tensors no longer fit beside them on 14.6 GB. The
per-case configuration lives in `configs/best.json`
(`python src/sweep.py --config configs/best.json --skip 14`).

### Accuracy stress and the d32 dispatch

A 25-trial stress test of case 7 (`results/case7-stress-a.json`) **failed**:
one element out of 6.5M reached abs error 0.00208 against the 0.002 budget.
The 5-trial official run passes, but a margin that a seed sweep can break is
not shippable. `d_model < 64` therefore dispatches to fp32
(`fp16_min_d_model=64`, tunable via `--fp16-min-d-model`) - the problem
statement explicitly allows per-shape implementations. fp32 keeps the
structural speedups (2.93x eager on case 7) at ~2e-6 error; the compiled fp32
number for case 7 is pending. Caveat noted for the report: worst-element
error is an extreme-value statistic, so other cases' margins (case 6 sits at
0.00187 over 5 trials) also shrink as trial count grows - a wider stress
sweep is on the list.

A separate ablation session (same GPU model) decomposed the win: with the fp16
path disabled (`--precision fp32`, structural changes only) case 1 gives
1.96x, case 8 gives 1.09x, and case 13 gives 4.48x - so on GEMM-heavy shapes
the tensor-core path is nearly the whole speedup, while at long sequence
length avoiding the materialised score tensor dominates. `--precision
autocast` matched the cached-fp16 default within noise except on the smallest
model (case 7: 3.51x vs 4.49x), where re-casting every weight each call is
proportionally expensive.

The cases fall into three regimes: launch-overhead-bound (2, 3, 4, 12: ~2.3x,
optimized latency pinned at ~1.4 ms regardless of batch size),
compute/bandwidth-bound (1, 5, 6, 8: 3.6-4.1x), and score-memory-bound
(11, 13: 6.5-11.3x, where the baseline's `[B, H, S, S]` fp32 score tensor
scales with heads and sequence length).

`--compile-user --compile-mode reduce-overhead` initially failed on every
case: the lazily built weight cache allocated its tensors inside the
CUDA-graph region, and graph replays overwrote them
(`results/pass1-fp16-compiled.json` records the failure). The cache builders
are now wrapped in `torch.compiler.disable`; the compiled table above is the
post-fix re-measurement.

### Pass two, measured (fused Triton kernels + CUDA graphs)

Pass two turns on the fused Triton residual+LayerNorm kernels by default and
adds manual CUDA-graph capture (`--cuda-graphs`). Measured best configuration
per case (`results/pass2-default.json`, `results/pass2-cg.json`; CUDA graphs
on the launch-bound cases 1–4 and 12, plain defaults elsewhere):

| # | Pass-1 best | Pass-2 best | Optimized ms | Config |
| --- | --- | --- | --- | --- |
| 1 | 6.00x | **7.05x** | 1.42 | cuda-graphs |
| 2 | 4.20x | **9.55x** | 0.30 | cuda-graphs |
| 3 | 4.11x | **9.19x** | 0.31 | cuda-graphs |
| 4 | 3.92x | **6.64x** | 0.43 | cuda-graphs |
| 5 | 6.54x | **6.98x** | 2.66 | default |
| 6 | 3.65x | **7.35x** | 198.57 | default |
| 7 | 7.39x | 3.93x | 1.60 | default (fp32 dispatch: max_abs 1.2e-06) |
| 8 | 5.10x | 5.09x | 25.23 | default |
| 9 | 4.05x | **4.43x** | 1.43 | default |
| 10 | 4.97x | **5.24x** | 1.50 | default |
| 11 | 8.91x | **10.19x** | 2.18 | default |
| 12 | 4.11x | **6.95x** | 0.42 | cuda-graphs |
| 13 | 16.32x | **16.98x** | 18.75 | default |

Geometric mean: **7.11x** (was 5.52x). Case 7's drop is the deliberate
accuracy trade — its fp16 margin failed a 25-trial stress, so it runs fp32.
All 13 cases pass accuracy; case 14 additionally passes out-of-core (below).

### Pass three, measured (independent cross-check + attention-kernel experiment)

A separate session built the same fused residual+LayerNorm design
independently, plus a FlashAttention-2-style Triton attention kernel for
sm_75, and measured both on another Kaggle T4 **before** this branch and the
pass-2 integration merged (raw data: `results/pass3-e*.json`; kernel design:
`docs/pass3-research.md`). Two independent implementations of the LN fusion
converging on the same win is the strongest evidence in this repo that the
win is real; the merged tree ships the pass-2 `fused_kernels.py`
implementation and keeps the attention kernel as an opt-in experiment.

What that session measured:

- **Fused-LN eager confirmation (E1)** — case 1 6.06x, case 5 6.10x,
  case 6 5.30x, case 13 14.5x, agreeing in direction and rough magnitude
  with the pass-2 table (6.79x / 6.98x / 7.35x / 16.98x). The residual gap
  is cross-session spread plus one real difference since fixed: that branch
  still paid an uncached per-forward mask sync, which stalls kernel-launch
  overlap exactly as the `_mask_is_dense` docstring describes.
- **The Triton attention kernel loses everywhere (E2)** — `--attention
  triton` eager: case 1 8.21 ms, case 5 16.3 ms, case 9 4.52 ms, case 11
  7.47 ms, case 13 187 ms — versus 2.4–3.5 ms / 28.8 ms class numbers on the
  SDPA path. The CUTLASS memory-efficient kernel keeps its crown on sm_75;
  hypotheses (no `cp.async` for Triton's pipeliner, `exp` vs `exp2`,
  conservative tiles) are recorded in `src/kernels.py`. The kernel stays
  in-tree as a documented negative result answering "a fused attention
  kernel is the natural next kernel" — with data.
- **Case 6 via `--compile-mode default` (E3)** — 6.25x on that branch;
  independently confirms the pass-2 conclusion that compile adds nothing
  over the fused defaults on case 6.
- **Triton kernels inside `reduce-overhead` compile (E4)** — mixed to
  negative (case 12 regressed to 0.85x); the idea is dropped from the
  merged flag surface.
- **Case 7 dispatch verification (E5)** — 25-trial stress now passes at
  max_abs 1.2e-06 with the shipped `fp16_min_d_model=64` dispatch (the
  pre-dispatch failure was 0.00208). Compiled `reduce-overhead` measured
  1.46 ms for case 7 vs 1.60 ms plain-default in the pass-2 session — a
  cross-session hint worth one head-to-head, queued in the notebook.
- **25-trial × 2-seed stress, all cases (E6)** — every case except 6 passes
  both seeds with margin (worst max_abs ≈ 1.7e-3); **case 6 fails seed 1234
  outright at 0.002074** and reaches 0.0022 on seed 9999 (that element was
  saved by the relative criterion). This independently reproduces the
  fragility documented below: case 6's fp16 margin does not survive trial
  scaling.

### Pass four, implemented — measurement pending (Kaggle 2x T4)

Pass four is implemented and locally validated (interpreter kernel tests +
end-to-end CPU harness runs) but **not yet measured on the T4** — the
predictions and decision rules are pre-registered in `docs/pass4-plan.md`,
and `notebooks/pass4-t4.ipynb` is the measurement session. Everything is
opt-in until the numbers exist:

- **`--dual-gpu`** — batch-split data parallelism across the measurement
  box's two T4s, with stream/event choreography that keeps the harness's
  cuda:0 event timing sound (zero host syncs) and numerics
  element-identical to single-GPU. Predicted: case 6 198.6 -> ~128 ms
  (~11.4x), case 8 ~1.75x, case 13 ~1.69x, case 14 ~2x; an environment
  extension, never the submission configuration.
- **`--fused-out-proj` / `--fused-ffn`** — Triton GEMM epilogues folding
  bias+residual+LayerNorm into out_proj/ffn_out and bias+erf-GELU into
  ffn_in (d_model <= 128, fp16). Predicted: case 6 -15-31 ms, the graphed
  cases lose ~12 in-graph kernels. The residual add consumes the unrounded
  fp32 accumulator — statistically tighter than the eager order.
- **`--attention bmm`** — materialized-score attention (cuBLAS bmm + fused
  Triton causal softmax) for head_dim > 128, targeting case 8's ~5-9 ms
  CUTLASS tiny-tile overhead; stress-gated (the fp16 score materialization
  is a new rounding source).
- Riders: the dead final residual store is skipped (`write_sum=False`,
  default on — bitwise-identical output), `--gelu-epilogue` (cuBLASLt
  tanh-GELU probe, accuracy-risky, measurement only), `--graph-streams 2`
  (two half-batch chains inside the graph), and the attribution tools
  `profile_case.py` / `bench_micro.py`.
- A **pass-4 erratum** corrects pass 3's §2.3: the transpose+reshape after
  mem-efficient SDPA was always a zero-copy view, so no kernel should be
  (or now is) justified by deleting it.

## Status

Passes one to three are implemented and measured (tables above): fused Q/K/V
projection, `scaled_dot_product_attention` in place of a materialised
`[B, H, S, S]` score tensor, fp16 matmuls on the T4's tensor cores with an
fp32 residual stream, fused Triton residual+LayerNorm kernels on by default,
manual CUDA-graph capture for the launch-bound shapes, the d<64 fp32
dispatch, out-of-core case 14 — plus, from pass three, an experimental
Triton attention kernel kept as a measured negative result, an
interpreter-runnable kernel test suite, and an independent cross-check of
the pass-2 numbers from a second T4 session. Pass four (above) is
implemented and awaiting its 2x-T4 measurement session; the pass-3 F1-F4
follow-ups ride along in the same notebook.

## Limitations and next steps

- The Triton kernels earn their place: `--no-fused-norm` costs case 1
  1.40→2.62 ms, case 5 2.66→5.27 ms, case 13 18.8→28.2 ms.
- Manual `--cuda-graphs` beats `torch.compile reduce-overhead` on every
  launch-bound shape (0.30–0.43 ms vs 0.54–0.57 ms replay latency) and
  compile beats the plain defaults nowhere now that the Triton kernels cover
  the elementwise fusion — the best configuration no longer uses
  `torch.compile` at all.
- `--fp32-reductions` measured **no effect** on either accuracy (identical
  max_abs on case 6) or speed (case 8 within noise) — cuBLAS was evidently
  not using fp16 reductions on this stack to begin with. Kept only as a
  probe.
- **Case 6 fails a 25-trial stress** (max_abs 0.00208, 1 element of 4.1B)
  while passing the official 5-trial run at 0.00187 — now reproduced on a
  second seed and session (pass-3 E6: 0.002074 fail on seed 1234, 0.0022 on
  seed 9999 saved only by the relative criterion). Same extreme-value
  mechanism as case 7, but the fp32 hammer would cost most of case 6's 7.3x,
  so the default stays fp16 with the fragility documented;
  `--fp16-max-elements 100000000` dispatches oversized forwards to fp32 for
  seed-robustness at that price (cost measurement queued in the notebook).
- Case 6 and compile: `reduce-overhead` OOMs beside the baseline at batch
  10000, and `--compile-mode default` measured no gain over the fused
  defaults (7.34x vs 7.35x). It runs plain defaults.
- **Case 14 runs and passes**: chunked fp16 vs the validated fp32 proxy
  reference over all 3.28B output elements, max_abs 0.00102, zero failures,
  155.4 s per forward (`results/case14-full.json`). The official harness
  still cannot grade it anywhere (the baseline reference is uncomputable);
  methodology question for the organisers stands.
- The fused attention kernel for the degenerate `head_dim=8` shape was
  built and measured in pass three: it **loses to the CUTLASS SDPA kernel
  on every appendix shape** (case 11: 7.47 ms vs 3.47 ms eager). It stays
  in-tree behind `--attention triton` as a documented negative result;
  future tuning directions (exp2 softmax, tile/warp retuning, single-phase
  causal loop) are listed in `src/kernels.py`. Case 7 compiled
  `reduce-overhead` (1.46 ms vs 1.60 ms default, cross-session) is the one
  open head-to-head; both are in `notebooks/pass3-t4.ipynb`.
- Shape-specialised dispatch is data-driven rather than hand-written: after
  sweeping settings variants, `python src/dispatch.py` distills the fastest
  accuracy-passing configuration per appendix shape into
  `configs/dispatch.json`, and `--dispatch` applies it (explicit flags still
  win; the in-model d32→fp32 dispatch applies regardless). `configs/best.json`
  is the hand-written equivalent for the harness-level flags.
