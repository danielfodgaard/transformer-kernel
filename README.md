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
| `src/torch_transformer_benchmark.py` | The organisers' benchmark harness (Python 3 + PyTorch). Contains `BaselineTransformer`, the `UserOptimizedTransformer` class to fill in, the accuracy comparison, and the latency benchmark. |

The upstream script was last updated 27 August 2026.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch          # CUDA build matching your GPU — see pytorch.org
```

## Reproducing the results

Run the benchmark with its defaults (batch 8, seq len 128, `d_model` 512,
8 heads, FFN 2048, 6 layers, float32, device auto-detected):

```bash
python src/torch_transformer_benchmark.py
```

Useful flags:

```bash
# sweep a different shape
python src/torch_transformer_benchmark.py --batch-size 32 --seq-len 1024 --d-model 1024

# half precision on a specific GPU, causal masking
python src/torch_transformer_benchmark.py --device cuda:0 --dtype float16 --causal

# torch.compile the optimized side only
python src/torch_transformer_benchmark.py --compile-user --compile-mode max-autotune
```

Accuracy thresholds default to the competition values (`--rtol 0.02`,
`--atol 0.002`). Timing defaults are 20 warmup iterations and 100 repeats over
3 rounds; `--help` lists every flag.

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

The optimized path is still the stock baseline — `UserOptimizedTransformer.forward`
currently delegates to `super().forward()`, so the script runs end to end and
reports a ~1x speedup. Kernel work is next.

## Limitations and next steps

- No custom kernel yet; the first targets are fused attention via
  `scaled_dot_product_attention`, then fused LayerNorm + residual and FFN.
- Shape-specialised dispatch (small vs. large sequence length) is not written.
- Numbers have not been collected on the target GPU, so no performance figures
  are quoted here yet.
