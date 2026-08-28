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
| `src/dispatch.py` | Data-driven shape dispatch: scans `results/*.json`, keeps the fastest accuracy-passing settings per shape, and writes `configs/dispatch.json`; `run_case.py --dispatch` applies it. |
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

## Status

Pass one is written but **not yet measured**. The optimizations in
`src/optimized.py` are PyTorch-level only: fused Q/K/V projection,
`scaled_dot_product_attention` in place of a materialised `[B, H, S, S]` score
tensor, fp16 matmuls on the T4's tensor cores with the residual stream,
LayerNorm statistics and softmax accumulation left in fp32, and the per-layer
causal mask hoisted out of the layer loop.

Reference point on a Tesla T4, stock baseline vs. the unmodified stub
(batch 8, seq 128, `d_model` 512, 8 heads, FFN 2048, 6 layers, non-causal,
fp32, torch 2.10.0+cu128): baseline median 13.857 ms, 73 900 token/s. That is
40.3 GFLOP per forward, or ~2.9 TFLOPS — about 36% of the T4's 8.1 TFLOPS fp32
peak, against ~65 TFLOPS available on its fp16 tensor cores.

## Limitations and next steps

- **No results yet.** Nothing in this repo has been run on a GPU; the numbers
  above are the organisers' baseline, not a measurement of the optimized path.
- No custom Triton or CUDA kernel yet — pass one is deliberately PyTorch-level.
- Appendix case 14 (batch 32 × seq 100 000 × `d_model` 1024) does not fit on a
  T4 in either implementation and needs sequential batch chunking.
- Shape-specialised dispatch is data-driven rather than hand-written: after
  sweeping settings variants, `python src/dispatch.py` distills the fastest
  accuracy-passing configuration per appendix shape into
  `configs/dispatch.json`, and `--dispatch` applies it (explicit flags still
  win). The table is only as good as the sweeps behind it — re-run the
  generator after measuring new variants.
