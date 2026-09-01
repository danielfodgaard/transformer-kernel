# Pass three — research survey, bottleneck analysis, and kernel design

Pass one restructured the PyTorch graph (fused QKV, SDPA, cached fp16 weights,
fp32 residual stream) for 3.56x geomean; pass two added `torch.compile
--compile-mode reduce-overhead` for 5.52x with the best per-case configuration.
This document is the design record for pass three: what the current research
frontier offers a **Tesla T4 (sm_75)**, where the remaining time actually goes,
and why the pass-3 kernels look the way they do.

Everything below was written in a CPU-only environment. Performance numbers
from earlier passes are measured (Kaggle T4, torch 2.10.0+cu128); pass-3
expectations are clearly labeled as predictions until the next T4 session runs
`configs/pass3-experiments.json`.

---

## 1. What the newest work offers — filtered through sm_75

The T4 is a 2018 part: 40 SMs, 65 TFLOPS fp16 tensor-core peak, 320 GB/s HBM,
64 KB usable shared memory per SM, **no** `cp.async` (async global→shared
copies are sm_80+), **no** bf16, **no** fp8. Most 2024–2026 kernel research
targets Hopper/Blackwell features (TMA, warp specialization, wgmma, fp8/fp4),
so the first job is separating what transfers from what does not.

### Transfers to the T4

- **FlashAttention's algorithm** (Dao et al. 2022; FlashAttention-2, 2023) —
  tiling + online softmax (Milakov & Gimelshein 2018) so the `[S, S]` score
  matrix never touches HBM. The *official* FA2 CUDA kernels require sm_80+,
  but the algorithm is architecture-independent. PyTorch's SDPA already gives
  us the same asymptotics on sm_75 through the xformers/CUTLASS
  memory-efficient kernel (supports back to sm_60), which is what passes 1–2
  use. A community FA2 port to Turing
  ([ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing))
  reports the mem-efficient kernel leaves up to ~2x on the table on a T4 for
  attention in isolation — evidence that a *shape-tuned* attention kernel can
  still beat the general CUTLASS one on this GPU. We get the same opportunity
  without a custom CUDA build via a Triton kernel (§3.2).
- **Fused residual + LayerNorm kernels** — the standard trick of inference
  engines: ByteTransformer (IPDPS 2023, notably a ByteDance paper) fuses
  add-bias + LayerNorm and reports 3.2% end-to-end from that fusion alone on
  BERT; LightSeq and FasterTransformer do the equivalent; Liger Kernel (2024)
  ships the same fusions as Triton for training stacks. On our d=128 shapes
  the LN sites are pure bandwidth, so the win is proportionally much larger
  than on BERT-base — §2.3 quantifies it.
- **Triton on Turing** — Triton lowers `tl.dot` to `mma.sync.m16n8k8` on
  sm_75 (fp16 in, fp32 accumulate), the same instruction class the T4's
  tensor cores expose. Constraints that shaped the kernels: 64 KB shared
  memory ceiling, no `cp.async` so `num_stages` buys little (keep it ≤ 2),
  fp16 only, and `tl.dot` needs K ≥ 16 (head_dim 8 → padded to 16; current
  `min_dot_size` on CUDA targets is (1, 1, 16), older Tritons wanted
  16×16×16, our blocks satisfy both). Empirically Triton definitely runs on
  the T4: every Inductor-generated kernel in the pass-2 results is a Triton
  kernel.
- **CUDA graphs** (already banked in pass 2 via `reduce-overhead`).

### Does not transfer (and why it's documented anyway)

- **FlashAttention-3/4, FlashInfer, ThunderKittens, CUTLASS 3.x pipelines** —
  built on TMA/warp-specialization/wgmma (sm_90+) or `cp.async` (sm_80+).
  Their *scheduling ideas* (persistent kernels, software pipelining) partially
  transfer, but Triton implements what sm_75 can use of them.
- **FlexAttention** (PyTorch 2.5+) — compiles custom attention variants; its
  value is mask/score-mod flexibility, which this workload doesn't need
  (plain causal), and current support targets newer GPUs than Turing anyway.
- **SageAttention 1/2 (2024–2025)** — INT8/INT4 QK^T with smoothing.
  The T4 *does* have INT8 tensor cores (130 TOPS), so this is the one
  quantized-attention line that could in principle run on Turing, but Triton's
  int8 `tl.dot` path on sm_75 is unproven and the competition's absolute
  tolerance (0.002) leaves no appetite for a second precision cliff after the
  d=32 fp16 failure (see pass-1 stress results). Rejected for risk, noted as
  future work.
- **fp8/fp4 attention, MXFP** — no hardware support before sm_89.

## 2. Where the remaining time goes

The pass-2 numbers split the 13 runnable cases into three regimes. Pass 3
targets each with a different tool.

### 2.1 Launch-overhead-bound: cases 2, 3, 4, 12 (~0.7 ms compiled)

Compiled with CUDA graphs, a forward is a single graph replay; the ~0.7 ms
floor is now the *serial sum of many small kernels*, not launch overhead.
The only lever left is fewer/wider kernels — which is what the fused LN
kernel does (3 pointwise/reduction kernels → 1 per LN site; 10 sites at
4 layers). Expectation: modest gains; these cases are already 4x. Eager +
custom kernels will not beat CUDA graphs here: ~50 Python-dispatched launches
cost more than a graph replay. Prediction: compiled stays the best config for
this regime, possibly with Inductor fusing around our kernels if
`triton_in_compile` measures well (§3.3).

### 2.2 Bandwidth-bound: cases 1, 5, 6, 8 (3.9–6.5x compiled)

At d_model=128 every GEMM has arithmetic intensity ≈ `K·N/(K+N)` ≈ 96
flops/byte against the T4's ~200 flops/byte ridge — even the matmuls are
bandwidth-bound, and everything between them doubly so. Per token and layer,
the eager LN sites move ~32 B/element in separate add/LN/cast kernels; the
fused kernel moves ~12 B/element (§3.1). Roughly 20–25% of non-GEMM traffic
disappears.

**Case 6 (batch 10000) is the marquee target**: it cannot use CUDA graphs on
a 16 GB card (the graph pools + the *baseline's* ~10 GB of transient fp32
score tensors OOM), so it is stuck at eager 3.65x — the largest case with the
weakest optimization. The Triton LN fusion applies in eager mode exactly
where compile cannot go. Its forward moves ~2.0 GB of LN-site traffic today
(4 layers × 10 sites-worth × 1.28 M tokens × 128 features); cutting that to
~0.75 GB saves ~4 ms of the 407 ms at 320 GB/s, plus kernel-count effects.
The bigger case-6 lever is `--compile-mode default` (Inductor fusion without
graph pools) — untested in pass 2, first item in the pass-3 matrix.

### 2.3 Score-memory-bound: cases 11, 13 (8.9x / 16.3x compiled)

SDPA already removed the `[B, H, S, S]` materialization; what remains is the
attention kernel's own efficiency. Two T4-specific soft spots in the CUTLASS
mem-efficient kernel:

1. **head_dim 8** (case 11, 16 heads × d=128): the kernel's minimum tile in
   the head dimension is far wider than 8; most of each tensor-core MAC is
   padding. A Triton kernel specialized for hd ∈ {8, 16, 32} pads only to 16.
2. **Output layout**: SDPA returns `[B, H, S, hd]`, forcing
   `transpose(1, 2).reshape(B, S, D)` — a full extra read+write of the
   context tensor per layer — before `out_proj`. A custom kernel can write
   `[B, S, H·hd]` directly, deleting that pass (the same trick
   ByteTransformer applies to its unpadded MHA epilogue).

   > **Pass-4 erratum:** this point is wrong. The mem-efficient kernel's
   > output is physically `[B, S, H, hd]`-contiguous returned as a
   > transposed *view* (torch's meta registration mirrors the CUDA op), so
   > the `transpose(1, 2).reshape` is zero-copy and there was never a pass
   > to delete on the SDPA path — one more reason §5's E2 kernel lost.
   > See `docs/pass4-plan.md` §0.

Case 13 (S=1024) additionally likes larger KV tiles than the general kernel
picks for hd=32. Prediction: the Triton attention kernel is worth trying on
cases 11, 13, 9 (hd=128 single head), 1/5 (hd=32); it should lose on case 8
(hd=256 needs tiny tiles under the 64 KB smem ceiling — dispatcher keeps SDPA
there by default).

### 2.4 Case 7 (d=32) and case 14 (100k tokens)

Case 7 now dispatches to fp32 (`fp16_min_d_model=64`) after the 25-trial
stress failure at 0.00208 abs; its compiled-fp32 number is still unmeasured —
in the matrix. Case 14 remains infeasible on a 16 GB T4 in any
implementation: the fp32 input activation is 12.8 GiB and the harness keeps
`x` alive across both models' calls, so even an fp16-I/O chunked forward
(6.4 GiB output) cannot coexist with the input, and no hardware anywhere can
materialize the baseline's 20.5 TB of score tensors for a reference. Still
documented as organizer-question, not engineering backlog.

## 3. The pass-3 kernels (src/kernels.py)

> **Post-merge note:** a parallel session landed the same §3.1 design on
> `main` first as `src/fused_kernels.py` (`ln_fwd`/`add_ln_fwd`, on by
> default); that is the shipping implementation, and §3.1 stands as this
> session's independent design record — two implementations converging on
> the same kernel is part of the evidence. §3.2's attention kernel shipped
> from this branch as `src/kernels.py`, opt-in. §5 has the measurements.

### 3.1 `fused_add_layernorm` — residual add + LayerNorm + downcast, one kernel

Every LayerNorm in the network except block 0's first sits immediately after
a residual add, and every LN output except the final one is immediately cast
to fp16 for the next GEMM. Eager PyTorch runs this as 3–4 kernels and ~32
bytes/element of traffic:

    attn_out(fp16) --cast--> fp32 --add--> x --F.layer_norm--> y --cast--> fp16

The Triton kernel does `s = x + r; y = (s − μ)/σ·w + b` with one read of
`x` (fp32) and `r` (fp16), one write of `s` (fp32, the new residual stream)
and one write of `y` (fp16) — ~12 bytes/element, one launch. Flags cover the
three sites: `HAS_RESIDUAL=False` for block 0's first LN, `WRITE_SUM=False`
+ fp32 output for the final norm. All statistics accumulate in fp32
(`tl.sum` tree reduction); the add is performed in fp32 exactly as the eager
`x + r.to(fp32)` sequence, so the only numerical difference from
`F.layer_norm` is sub-1e-6 reduction-order noise — the same class of
difference cuDNN/Inductor LN reimplementations carry. One program per token
row handles d ≤ 4096 (covers d=1024) with `BLOCK = next_pow2(d)`.

Under `torch.compile`, Inductor already fuses this chain (differently but
comparably), so the kernel's value is concentrated in **eager mode — which is
exactly where case 6 lives**, and as a kernel-count reduction inside CUDA
graphs if `triton_in_compile` measures well.

### 3.2 `flash_attention_forward` — FA-2-style causal attention for sm_75

A Triton implementation of the FlashAttention-2 forward pass (no backward —
inference only), specialized to this benchmark:

- fp16 Q/K/V, fp32 accumulators, **exact** softmax — scores scaled in fp32
  *after* the QK^T dot (matching the baseline's `matmul → · scale` order,
  not the prescaled-fp16-q shortcut), online max/sum in fp32, `exp` (not the
  `exp2` trick — one fewer rounding difference vs the fp32 reference, and
  SFU throughput is not the bottleneck at these sizes), probabilities cast
  to fp16 only for the PV tensor-core dot. Error class identical to the
  mem-efficient kernel that already passes at 1.4–1.9e-3.
- **Writes `[B, S, H·hd]` directly** (strided store into the flattened
  layout), so the `transpose+reshape` pass disappears and `out_proj` reads
  the kernel's output in place. Reads Q/K/V through strides straight out of
  the fused-QKV projection's `[B, S, 3, H, hd]` view — no `.contiguous()`
  anywhere.
- Turing-sized tiles chosen for the 64 KB smem ceiling with `num_stages=2`
  and no `cp.async`: (BLOCK_M, BLOCK_N, warps) = (128, 64, 4) for hd ≤ 16,
  (64, 64, 4) for hd ≤ 64, (64, 32, 8) for hd = 128, (32, 32, 8, 1 stage)
  for hd = 256. head_dim 8 is zero-padded to 16 inside the kernel (`tl.dot`
  K ≥ 16).
- Causal handled by iterating KV tiles only up to the diagonal and masking
  the diagonal tile elementwise; every query row keeps ≥ 1 valid key, so the
  running max never stays −inf. 1-D launch grid (flattened `B·H·M-tiles`),
  immune to the 65535 limit on higher grid dimensions (case 6 has
  B·H = 40 000 programs per M-tile).
- Scope guard: dispatched only when CUDA + fp16 + **no padding mask**
  (the harness's accuracy and timing runs use `padding_ratio 0`; any real
  mask falls back to SDPA) + head_dim ≤ 256 + Triton importable. Everything
  else silently uses the pass-1 SDPA path.

### 3.3 Dispatch and interaction with torch.compile

As designed on this branch, both kernels sat behind opt-in settings
(`--ln triton`, `--attention triton`, `--triton-in-compile`) so unmeasured
code could not become a default. After measurement and the merge with the
parallel pass-2 integration, the surviving surface is: LN fusion on by
default via `fused_kernels.py` (`--no-fused-norm` to ablate), the attention
kernel opt-in via `--attention triton`, and `--triton-in-compile` dropped —
E4 measured the kernels inside `reduce-overhead` compile as mixed-to-negative
(case 12 regressed to 0.85x), and the merged tree's manual `--cuda-graphs`
capture makes compile unnecessary anyway.

## 4. Measurement matrix — RAN 2026-08-31, results in results/pass3-e*.json

All six experiments executed on a Kaggle T4 (torch 2.10.0+cu128), pre-merge;
§5 interprets them. `notebooks/pass3-t4.ipynb` now carries the *follow-up*
matrix (F1–F4) for the merged tree. The original experiments:

| # | Question | Command core |
| --- | --- | --- |
| E1 | Does Triton LN move eager big-batch cases? | cases 1,5,6,13 `-- --ln triton` |
| E2 | Does Triton attention beat CUTLASS SDPA where predicted? | cases 1,5,9,11,13 `-- --attention triton` (then + `--ln triton`) |
| E3 | Case 6 without graph pools | case 6 `--compile-user --compile-mode default`, and eager `--ln triton --attention triton` |
| E4 | Custom kernels inside compile | winners of E1/E2 + `--compile-user --compile-mode reduce-overhead --triton-in-compile` |
| E5 | Case 7 compiled-fp32 (dispatch verification) | case 7 `--compile-user --compile-mode reduce-overhead` (defaults now dispatch d<64 to fp32) |
| E6 | Accuracy stress with final config | all cases, `--accuracy-trials 25`, two seeds |

Notes for whoever runs it: the Kaggle notebook must `git checkout` the
working branch if pass 3 isn't merged to `main` yet (the current notebook
clones `main`); and `max-autotune` is a known dead end on the T4 — Inductor
hard-gates its GEMM templates on ≥ 68 SMs (`torch/_inductor/utils.py`,
"Not enough SMs", exactly the warning visible in the pass-2 Kaggle log) and
the T4 has 40, so `max-autotune` degenerates to pointwise autotuning.

## 5. Measured addendum (2026-08-31, Kaggle T4, pre-merge branch)

Predictions vs reality, experiment by experiment:

- **E1, fused LN (`--ln triton` eager)** — predicted "modest, targets
  bandwidth"; measured **much bigger than the traffic model said**: case 1
  3.87→6.06x, case 5 3.71→6.10x, case 6 3.65→5.30x (407→294 ms), case 13
  11.26→14.5x. §2.2's ~1% estimate for case 6 was off by an order of
  magnitude — the fused kernel removes whole kernel launches and their
  latency tails, not just DRAM bytes; a lesson recorded for the next cost
  model. The parallel session's equivalent measurement (pass-2 table:
  6.79 / 6.98 / 7.35 / 16.98x) agrees in direction; its extra margin is
  cross-session spread plus its weakref-cached mask check (this branch
  still paid a per-forward sync that stalls launch overlap).
- **E2, Triton attention (`--attention triton`)** — predicted "worth trying
  on 11, 13, 9, 1/5"; measured **a loss on every shape**: case 11 7.47 ms
  vs 3.47 ms eager SDPA-path, case 13 187 ms vs 28.8 ms, case 1 8.21 ms vs
  2.44 ms. The §2.3 head_dim-8 padding argument was correct about CUTLASS's
  waste but wrong about who wastes more: without `cp.async`, Triton's
  pipeliner cannot hide global-load latency behind the MMA pipeline on
  sm_75, and the hand-pipelined CUTLASS kernel keeps its crown. Kept
  opt-in as a documented negative result (`src/kernels.py` lists tuning
  directions: exp2 softmax, tile/warp retune, single-phase causal loop).
- **E3, case 6 `--compile-mode default`** — 6.25x on this branch;
  the parallel session measured 7.34x vs 7.35x for its fused default. Both
  sessions conclude compile buys nothing over fused eager on case 6.
- **E4, kernels inside `reduce-overhead`** — mixed to negative (case 2
  3.22x vs 4.20x compiled-torch-path; case 12 **0.85x**); the
  `--triton-in-compile` flag did not survive the merge.
- **E5, case 7 dispatch verification** — 25-trial stress passes at
  max_abs 1.19e-06 (was 0.00208 pre-dispatch): the pass-2 fix works.
  Compiled `reduce-overhead` measured 1.46 ms vs the parallel session's
  1.60 ms plain-default — a cross-session hint, queued as notebook F2.
- **E6, 25-trial × 2-seed stress, all cases** — all pass on both seeds
  except case 6: seed 1234 **fails** at 0.002074, seed 9999 reaches 0.0022
  with the worst element saved by the relative criterion. Reproduces the
  parallel session's finding; mitigation pricing (`--fp16-max-elements`)
  is notebook F3.

Net effect on the shipped configuration: none of the pass-3 experiments
displaced a pass-2 winner (`configs/best.json` unchanged; geomean stays
7.11x), which is itself the finding — the pass-2 integration's choices
survive an independent measurement session, the LN-fusion win is
double-confirmed, and the attention-kernel question is answered with data
instead of a hunch.

## 6. Sources

- Dao et al., *FlashAttention* (2022) / *FlashAttention-2* (2023) —
  [github.com/Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention)
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018)
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) —
  FA2 CUDA port for sm_75, T4 benchmarks vs PyTorch SDPA
- Zhai et al., *ByteTransformer: A High-Performance Transformer Boosted for
  Variable-Length Inputs* (IPDPS 2023) —
  [arxiv.org/abs/2210.03052](https://arxiv.org/pdf/2210.03052),
  [github.com/bytedance/ByteTransformer](https://github.com/bytedance/ByteTransformer)
- xformers memory-efficient attention (CUTLASS, sm_60+) —
  [facebookresearch.github.io/xformers](https://facebookresearch.github.io/xformers/components/ops.html)
- Liger Kernel (2024) — Triton fused LayerNorm/RMSNorm for LLM stacks
- PyTorch blog, *Towards Free Normalization: Fusing Normalization into GEMM
  and Attention Kernels* —
  [pytorch.org/blog](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)
- SageAttention / SageAttention2 (2024–2025) — INT8 attention, evaluated and
  rejected for sm_75 risk (§1)
- Jia et al., *Dissecting the NVidia Turing T4 GPU via Microbenchmarking*
  (2019) — [arxiv.org/abs/1903.07486](https://arxiv.org/pdf/1903.07486) —
  smem/latency/tensor-core numbers used in §2–3
- Triton compiler, NVIDIA backend (`min_dot_size`, capability gates) — read
  from the installed 3.7.1 source; PyTorch `_inductor/utils.py` SM gate ditto
