# Pass one — proposed adjustments, for review

Everything below is implemented and can be flipped from the command line, so
nothing here is locked in. Each item says what I chose, why, what the risk is,
and what I want you to decide. **None of it has been executed** — there is no
GPU (and no `torch`) in this environment, so treat every performance claim as
an estimate and every numerical claim as an argument, not a measurement.

---

## A. Explicit cached fp16 weights instead of autocast — **default: `--precision fp16`**

You asked for the forward to run under `torch.autocast(fp16)`. I implemented
that as `--precision autocast`, but made the default a mode that casts the
weights **once** and reuses them.

**Why.** `torch.autocast` caches its fp16 weight copies only *inside* the
autocast region. Entering and exiting the context once per forward throws that
cache away every call, so every weight is re-cast on every call. For your
current config that's ~18.9M params ≈ 113 MB of extra traffic per forward,
≈ 0.35 ms at the T4's 320 GB/s. That's 2.5% of the 13.86 ms baseline — but it
would be ~12% of a 3 ms optimized forward, i.e. it grows into real money
exactly as we get faster.

**Numerically the two are the same** for the matmuls: both feed fp16 operands to
cuBLAS with fp32 accumulation. The difference is only *when* the cast happens.

**Risk.** The cache assumes the weights never change after the first forward.
That holds here — `copy_model_weights()` runs at line 692 and `.to(device,
dtype)` at line 698, both before any forward, and nothing writes to the
parameters afterwards. `_apply` is overridden to invalidate the cache if the
model is ever moved again. If you'd rather not rely on that, `--precision
autocast` has no cache at all.

**Decide:** keep `fp16` as default, or make `autocast` the default and treat
`fp16` as the experiment. Either way please run both once — the delta is the
one number that tells us whether this whole item was worth it.

---

## B. What stays in fp32 — residual stream, LayerNorm, softmax

Concretely, per block:

| Operation | dtype | Why |
| --- | --- | --- |
| `F.layer_norm` | fp32 | Variance accumulation; also what autocast does on its own |
| Q/K/V + out + FFN matmuls | fp16 | Tensor cores; cuBLAS accumulates in fp32 internally |
| SDPA softmax | fp32 | The memory-efficient kernel accumulates in fp32 for fp16 operands |
| GELU | fp16 | Elementwise, cheap; `approximate="none"` to match the baseline's erf exactly |
| Residual add `x + attn_out` | fp32 | **The important one** — see below |

The residual stream is the one I'd defend hardest. Keeping `x` in fp32 across
all layers means each block's fp16 rounding is injected once and then
renormalized by the next LayerNorm, instead of compounding through 4–6 layers
of fp16 accumulation. It costs one cast per residual add and roughly doubles
the residual tensor's traffic versus a fully-fp16 stream.

**Why it matters for the tolerance.** The check is
`|err| <= 0.002 OR |err| <= 0.02*|ref|`, and the model's last op is a LayerNorm
so outputs are ≈N(0,1). Elements with `|ref| >= 0.1` are covered by the relative
term. The exposed band is `|ref| ≈ 0.05`, where relative gives you only 0.001
and you must hold absolute error under 0.002. An fp32 residual stream is what
keeps us comfortably inside that band; a fully-fp16 stream is where I'd expect
the first failures to show up.

**This is the item most likely to be wrong.** I cannot test it. If accuracy
fails, bisect in this order: `--precision fp32` (if that passes, it's a
precision problem, not a restructuring bug) → `--attention math` (if *that*
passes, it's SDPA specifically) → `--no-fuse-qkv`.

---

## C. `torch.compile` — not called anywhere in my code

You asked for it behind a flag rather than always-on. The benchmark already has
`--compile-user` / `--compile-baseline` / `--compile-mode`, applied at line 703
after construction, weight copy, `.to()` and `.eval()` — which is the correct
place. So the right move was to write **zero** compile code and let the
harness's flag do it. Compiling inside `forward()` would also have fought with
the harness's own wrapper.

One wrinkle: my weight cache is built lazily on the first forward, so a
compiled run takes a graph break on call one. That happens during the accuracy
phase, ~25 calls before timing starts, so it should not touch the measurement.

**Note on `--compile-mode max-autotune`:** worth trying on cases 1–13, but
expect long compile times, and I would not point it at case 14.

---

## D. Mask handling — one sync per forward, overridable

The harness never passes `None`; with `--padding-ratio 0.0` it passes an
all-True mask, so the baseline pays for `masked_fill` and the masked attention
path even when nothing is padded. Detecting "all True" lets me use SDPA's
`is_causal=True` fast path and skip three `masked_fill`s per layer.

Detecting it means `bool(valid_token_mask.all())`, which forces a device→host
sync: ~20 µs per forward. Negligible at 13.9 ms, but 1–2% on the small cases
(2, 3, 12). It's **one** sync per forward, not per layer.

`--assume-dense-mask` skips the check. It is only correct when
`--padding-ratio 0`, which is the default, and I've deliberately not made it
the default because it's silently wrong otherwise.

**Decide:** whether to measure with it on for the small cases.

---

## E. Where the fused causal mask is built

The baseline allocates a fresh `[S, S]` bool mask **inside every layer, on
every call** (line 97). Mine is built once per forward and cached across calls,
and in the no-padding case isn't built at all (`is_causal=True` instead). At
S=1024 that's 1 MB × 4 layers × every call that simply stops happening.

Note the polarity flip, which is an easy place to introduce a silent bug: the
baseline builds a `triu` "block these" mask for `masked_fill`; SDPA's bool
`attn_mask` marks positions to **keep**, so mine is `tril`. If attention output
ever looks anti-causal, this is the line.

---

## F. Not editing the organizers' file — runtime patching

`src/torch_transformer_benchmark.py` is untouched. `run_case.py` rebinds
`torch_transformer_benchmark.UserOptimizedTransformer` to our class before
calling the harness's own `main()`, which resolves that name as a module global
at call time. All original flags pass through.

For submission, paste the body of `OptimizedTransformer` into the stub —
nothing in the design depends on the patching.

**Structural constraint this had to respect:** `copy_model_weights()` does a
*strict* `load_state_dict`, so the optimized model's state-dict keys must match
the baseline's exactly. `OptimizedTransformer` therefore subclasses
`BaselineTransformer` and registers **no** new parameters or buffers — the
fused QKV and fp16 copies are plain Python attributes, invisible to
`state_dict()`. Do not reach for `--non-strict-weight-copy` to work around
this: a genuinely new parameter would silently keep its random init instead of
the baseline's weights, and you'd get an accuracy failure that looks numerical
but isn't.

---

## G. Benchmark the causal path — your 13.86 ms number is not a competition case

`--causal` defaults to False, but **all 14 appendix cases are causal**. The
sweep passes `--causal` for every case, so the numbers won't be directly
comparable to the baseline you posted. Worth re-running your reference number
with `--causal` so there's a like-for-like pair.

Also: `--dtype float16` casts *both* models, making the reference itself fp16
and the accuracy check trivial. The honest configuration is fp32 baseline with
fp16 only inside the optimized forward — that's what the sweep does by default.
I'd report only those numbers.

---

## H. Case 14 is out of scope for pass one

Batch 32 × seq 100000 × d_model 1024 in fp32 is **13.1 GB for the input
activation alone**, on a 15 GB card, before any attention exists. In fp16 it's
6.6 GB and Q/K/V triples it. Separately, the baseline tries to materialize a
`[32, 16, 100000, 100000]` score tensor, so **there is no fp32 reference to
compare against on a T4 at all**.

It needs sequential batch chunking (per sample it's ~205 MB per tensor, which
fits). That's pass two. The sweep records it as `status: "error"` with the
stderr tail and keeps going — use `--skip 14` to not wait for it.

---

## What I want back from you

1. A/B on item A — is the cached-fp16 default acceptable, or do you want
   autocast as the headline?
2. Item B — comfortable with the fp32 residual stream, or should I also build a
   full-fp16-stream variant to measure?
3. Item D — run the small cases with `--assume-dense-mask` too?
4. Anything you want dropped before the first Kaggle run.

## First run I'd suggest

```bash
python src/sweep.py --skip 14 --out results/pass1-fp16.json
python src/sweep.py --cases 1,7,8,13 --out results/pass1-fp32.json    -- --precision fp32
python src/sweep.py --cases 1,7,8,13 --out results/pass1-autocast.json -- --precision autocast
python src/sweep.py --cases 1 --out results/noise-floor.json -- --reference-check
```

The last one runs the baseline against itself; it should print ~1.00x, and
however far it deviates is the harness's measurement noise — the floor below
which no speedup difference means anything.
