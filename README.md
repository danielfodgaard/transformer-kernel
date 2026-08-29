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
| `src/optimized.py` | The optimized implementation. Subclasses `BaselineTransformer`; fused QKV projection, `scaled_dot_product_attention`, fp16 matmuls with an fp32 residual stream. |
| `src/run_case.py` | Runs the harness with `UserOptimizedTransformer` swapped for ours, so the organisers' file stays untouched. All of its flags pass through. |
| `src/sweep.py` | Runs a set of shapes (one subprocess each) and writes the numbers to `results/*.json`. |
| `configs/shapes.json` | The 14 appendix test shapes. |
| `docs/pass1-decisions.md` | Why each optimisation was chosen, what stays in fp32, and how to bisect an accuracy failure. |

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

Case 14 is skipped because it does not fit on a 16 GB GPU — see
`docs/pass1-decisions.md`. Anything after `--` is forwarded to the run, so
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

Geometric mean speedup: **3.56x**.

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
are now wrapped in `torch.compiler.disable`; the compiled configuration is
pending re-measurement.

## Status

Pass one is implemented and measured (table above). The optimizations in
`src/optimized.py` are PyTorch-level only: fused Q/K/V projection,
`scaled_dot_product_attention` in place of a materialised `[B, H, S, S]` score
tensor, fp16 matmuls on the T4's tensor cores with the residual stream,
LayerNorm statistics and softmax accumulation left in fp32, and the per-layer
causal mask hoisted out of the layer loop.

## Limitations and next steps

- No custom Triton or CUDA kernel yet — pass one is deliberately PyTorch-level.
  A fused LayerNorm+residual Triton kernel is the next planned step.
- The launch-overhead-bound shapes (2, 3, 4, 12) are pinned at ~1.4 ms by
  kernel launch cost; CUDA graphs via `--compile-user --compile-mode
  reduce-overhead` target this and need re-measurement after the cache fix.
- Case 7's accuracy margin is thin (max_abs 0.00197 of the 0.002 budget);
  shape-dispatching `d_model <= 32` to fp32 is under consideration - the
  problem statement explicitly allows per-shape implementations.
- Appendix case 14 (batch 32 × seq 100 000 × `d_model` 1024) does not fit on a
  T4 in either implementation: the fp32 input activation alone is 13.1 GB and
  the baseline's per-sample score tensor would be 640 GB, so the reference is
  uncomputable on any hardware. Our path needs sequential batch chunking;
  validation methodology is an open question for the organisers.
