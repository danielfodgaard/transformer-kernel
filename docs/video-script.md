# Demo video — recording script (~4 minutes)

Backend track: the accepted format is a walkthrough of API usage, inference
examples, and result analysis — this script does exactly that. One screen
recording (terminal + browser), no camera needed.

## Compliance checklist (from the submission rules)

- [ ] Uploaded to YouTube, visibility **Public** (not unlisted).
- [ ] Link pasted into the Devpost description (`docs/devpost.md` bottom).
- [ ] No third-party trademarks: no TikTok/Kaggle/NVIDIA logos on screen
      beyond what incidentally appears in the notebook UI; name the event
      in plain text only; no music, no stock footage, no copied figures.
      Everything shown is this repo, its terminal output, and its README.
- [ ] Suggested title: `transformer-kernel — 7.1x Transformer layer on a
      Tesla T4 (Jam Session 2026, Task 3)`; description links the repo.

## Preparation (before recording)

On a Kaggle T4 notebook (or any T4 box), have one cell/terminal ready with
the repo cloned and this pre-warmed so nothing stalls on camera:

```bash
cd /kaggle/working && rm -rf transformer-kernel
git clone -q https://github.com/danielfodgaard/transformer-kernel.git
cd transformer-kernel
nvidia-smi --query-gpu=name --format=csv,noheader   # expect: Tesla T4
```

Open in browser tabs: the repo README at the pass-2 results table, and
`src/fused_kernels.py` around the `_add_ln_kernel` definition.

## Shot list

### Scene 1 — the problem (0:00–0:35, README on screen)

> "This is my entry for Task 3: implement a GPU kernel for a Transformer
> layer. The organisers ship a benchmark with a baseline PyTorch
> Transformer and an empty `UserOptimizedTransformer`. The rules: same
> outputs — every element within an absolute error of 0.002 or a relative
> error of 2 percent — measured across 14 shapes, from batch 1 up to batch
> ten thousand and sequences up to one hundred thousand tokens. My target
> GPU is the Tesla T4: fp16 tensor cores, no bf16, 64 kilobytes of shared
> memory. The result, up front: a **7.1x geometric-mean speedup** across
> all 13 shapes the harness can run, with every case passing accuracy."

Scroll slowly over the README's pass-2 table while saying the last line.

### Scene 2 — live inference, launch-bound shape (0:35–1:20, terminal)

Run the organisers' own harness end to end — accuracy phase, then timing —
on the batch-1 shape with CUDA-graph capture:

```bash
python src/run_case.py --causal --batch-size 1 --d-model 128 --heads 4 \
  --seq-len 128 --layers 4 --ffn-dim 128 --cuda-graphs
```

> "Everything is measured by the organisers' unmodified script — I never
> touch their file; my runner just swaps the class in. First it checks
> accuracy: five trials, every element, PASS. Then latency: the baseline
> takes about three milliseconds; my version replays the whole forward as
> a single CUDA graph in 0.3 milliseconds — about 9.5x — because at this
> size the GPU work is microseconds and the real enemy is launch overhead."

(While it runs, point at the accuracy PASS lines, then the speedup line.)

### Scene 3 — what the kernel does (1:20–2:20, editor + terminal)

Show `src/optimized.py` docstring bullets, then `src/fused_kernels.py`'s
`_add_ln_kernel`:

> "Three layers of optimisation. Pass one restructures the graph: one fused
> Q-K-V matmul, memory-efficient attention so the S-by-S score matrix never
> touches DRAM, and fp16 tensor-core GEMMs with the residual stream,
> LayerNorm statistics, and softmax kept in fp32 — that's what keeps the
> error inside the budget. Pass two is a custom Triton kernel: at model
> width 128 this workload is bandwidth-bound, and the chain between the
> matmuls — add, LayerNorm, cast, nine kernels per block — costs more than
> the matmuls. This kernel does the residual add, the normalisation
> statistics in fp32, and the downcast in a single pass."

Then run the kernel test suite live:

```bash
python src/test_kernels.py
```

> "Every kernel is tested against the eager ops it replaces before any
> benchmark is trusted — and there's a pytest twin that runs the same
> numerics through Triton's interpreter on machines with no GPU at all."

### Scene 4 — inference at the other extreme (2:20–3:00, terminal)

Start the long-sequence case; jump-cut the wait in the edit:

```bash
python src/run_case.py --causal --batch-size 64 --d-model 128 --heads 4 \
  --seq-len 1024 --layers 4 --ffn-dim 128
```

> "Other regimes, other physics. At sequence length 1024 the baseline
> materialises a gigabyte-scale score tensor per forward; avoiding it plus
> the fused kernels gives seventeen-x. And appendix case 14 — one hundred
> thousand tokens, where the baseline would need a twenty-terabyte score
> tensor and can't run on any hardware in existence — my implementation
> streams out of core and passes against a validated fp32 proxy reference
> over all 3.3 billion output elements." *(Optionally flash
> `python src/run_case14.py --max-samples 4` starting, then cut.)*

### Scene 5 — result analysis and the honest bits (3:00–3:45, README)

Scroll the pass-2 and pass-3 sections:

> "Every number in the README has its raw harness output committed in the
> results folder — including the experiments that failed. A
> FlashAttention-style Triton kernel I wrote for this GPU lost to the
> built-in CUTLASS kernel on every shape; it ships as a documented negative
> result. torch.compile got retired the same way — measured, not assumed.
> And because worst-element error is an extreme-value statistic, I
> stress-test accuracy at 25 trials across two seeds: the two shapes whose
> fp16 margins break under stress are dispatched to fp32 by a shape check,
> which the problem statement explicitly allows."

### Scene 6 — wrap (3:45–4:00)

> "7.1x geometric mean on the T4, organisers' harness, organisers'
> thresholds, fully reproducible: one notebook re-runs every number you've
> seen. Repo link below. Thanks for watching."

## After recording

1. Upload to YouTube, set **Public**, title/description per the checklist.
2. Paste the link into `docs/devpost.md` (bottom) and the README's
   Deliverables section, commit, push.
