# Devpost project description

Ready to paste into the Devpost submission form. Fill in the two links at the
bottom before submitting.

---

## transformer-kernel — a 7.1x Transformer layer for the Tesla T4

**Task 3 — Implement a GPU Kernel for a Transformer Layer.** The organisers
provide a benchmark with a baseline PyTorch Transformer and an empty
`UserOptimizedTransformer`; the job is to make it faster on a specific GPU
while every output element stays within `abs <= 0.002 OR rel <= 2%` of the
fp32 baseline, across 14 test shapes spanning batch 1–10 000, sequence
32–100 000, width 32–1024, and 1–16 heads.

### How the solution addresses the problem statement

Target GPU: **Tesla T4** (Turing sm_75 — fp16 tensor cores, 320 GB/s, 64 KB
shared memory per SM, no bf16/fp8, no `cp.async`). The solution is three
measured optimisation passes, each attacking what profiling of the previous
one showed:

1. **Restructure the graph (PyTorch-level).** Fuse the three Q/K/V
   projections into one GEMM; replace the baseline's materialised
   `[B, H, S, S]` score tensor with `scaled_dot_product_attention`
   (CUTLASS memory-efficient kernel — the score matrix never touches DRAM);
   pre-cast weights to fp16 once and run every GEMM on the tensor cores
   while the residual stream, LayerNorm statistics, and softmax stay in
   fp32; hoist the per-layer causal mask out of the loop. → **3.56x**
   geometric mean.

2. **Custom Triton kernels + launch-overhead elimination.** At d_model=128
   even the GEMMs are bandwidth-bound, and the elementwise chain between
   them (add → LayerNorm → cast, nine kernels per block) dominates. A
   custom fused Triton kernel runs each residual-add + LayerNorm + downcast
   as one launch with fp32 statistics — measured worth 1.9x on its own on
   the reference shape (2.62 ms → 1.40 ms). For the small shapes whose
   ~1.4 ms floor is pure host-side dispatch, the whole forward is captured
   into a **CUDA graph** and replayed as a single launch (0.30 ms). A
   data-driven dispatch table picks the fastest *accuracy-passing* settings
   per shape from the recorded sweeps. → **7.11x** geometric mean across
   the 13 harness-runnable shapes, all passing accuracy.

3. **Research-driven experiments, measured honestly.** A survey of current
   attention-kernel work (FlashAttention 1–4, FlashInfer, FlexAttention,
   SageAttention, ByteTransformer, Liger) filtered by what actually
   transfers to a 2018 sm_75 part, then an experiment matrix run on a
   second T4 session: an independently built duplicate of the LN fusion
   **confirmed the pass-2 win** (two implementations, two sessions, same
   result); a hand-written FlashAttention-2-style Triton attention kernel
   for sm_75 **lost to CUTLASS SDPA on every shape** and ships as a
   documented negative result; `torch.compile` was **retired** — measured
   slower than the custom kernels + CUDA graphs everywhere once pass 2
   landed.

Accuracy is treated as an engineering budget, not an afterthought:
worst-element error is an extreme-value statistic, so official 5-trial
passes are re-verified with 25-trial × 2-seed stress runs. Two shapes
whose fp16 margins break under stress get shape dispatch — the problem
statement explicitly invites per-shape implementations — d_model < 64 runs
fp32 (error 2e-6, still 3.9x), and the same mechanism is available for
oversized outputs (`--fp16-max-elements`). The extreme case 14 (batch 32 ×
100 000 tokens, whose baseline would need a ~20 TB score tensor — ungradable
on any hardware) runs **out of core** in chunks and passes against a
validated fp32 proxy reference over all 3.28 B output elements.

Headline numbers (Tesla T4, organisers' harness and thresholds, median of
300 CUDA-event samples; full per-case tables in the README):

| | |
| --- | --- |
| Geometric-mean speedup, 13 runnable shapes | **7.11x** |
| Best case (batch 1, CUDA graph replay) | **9.5x** (0.30 ms) |
| Long-sequence case (S=1024) | **17.0x** |
| Accuracy | all cases pass; stress-tested beyond the official trials |

### Development tools used

- **Claude Code** (Anthropic) — agentic AI pair programmer; used for
  implementation, kernel design, measurement analysis, and docs across
  multiple sessions, with every session linked from its commits. All GPU
  numbers were measured by (human-run) Kaggle sessions, never estimated.
- **Kaggle Notebooks** — free Tesla T4 runtime; the repo's
  `notebooks/*.ipynb` are the exact measurement runners used.
- **Jupyter, git + GitHub** (PR-based workflow between AI sessions),
  **pytest** for the kernel test suites.

### APIs used

- **PyTorch APIs**: `scaled_dot_product_attention` (CUTLASS
  memory-efficient backend), `torch.cuda.CUDAGraph` capture/replay,
  `torch.compile` (evaluated, then retired by measurement), cuBLAS fp16
  GEMMs via `F.linear`, CUDA events for timing.
- **Triton** JIT API for the custom kernels (`tl.dot` on tensor cores,
  online-softmax reductions), plus `TRITON_INTERPRET` for CPU-side
  numerical validation.
- **Anthropic Claude API** indirectly, as the engine behind Claude Code.
  No external web/data APIs are used at runtime — the deliverable is a
  self-contained GPU kernel.

### Libraries and frameworks used

- **PyTorch 2.10 (cu128)** — model, harness, and GPU runtime.
- **OpenAI Triton 3.x** (bundled with PyTorch) — custom fused
  residual+LayerNorm kernels (shipping) and the experimental
  flash-attention forward (documented negative result).
- **CUTLASS** (via PyTorch's memory-efficient SDPA backend) — the
  attention kernel that our measurements say to keep on sm_75.
- **NumPy** — backs Triton's interpreter so the kernel test suite runs on
  CPU-only machines; **pytest** for the suites.

### Datasets and assets used

- **No external datasets.** All inputs are synthetic tensors generated by
  the organisers' **unmodified** benchmark harness (`torch.randn` with
  fixed seeds; the harness file is kept byte-identical and its methodology
  — 5 accuracy trials, 20 warmup, 3×100 alternating timed repeats — is
  never reimplemented).
- The **14 appendix test shapes** from the problem statement
  (`configs/shapes.json`).
- Every measurement in the write-up is backed by a committed JSON in
  `results/` (raw harness output, parsed), including the failed and
  negative-result runs.

### Links

- **Repository:** https://github.com/danielfodgaard/transformer-kernel
- **Demo video:** `[YouTube link — see docs/video-script.md for the
  recording plan]`
