#!/usr/bin/env python3
"""Run appendix case 14 (batch 32, seq 100000, d_model 1024) out of core.

The organizers' harness cannot grade this case on any hardware: the baseline
materializes a ``[32, 16, 100000, 100000]`` attention score tensor (~20 TB in
fp32), and the fp32 input activation alone is 13.1 GB on a 15 GB T4. There is
therefore no runnable reference to diff against -- for anyone. This script is
the closest achievable substitute, built as a chain of trust:

  1. **Proxy validation** (feasible sequence length, default 1024): the true
     ``BaselineTransformer`` is compared against this script's *reference
     path* -- the optimized model restructured but with fp32 matmuls and
     memory-efficient SDPA, run in batch chunks. Expected agreement ~1e-6,
     far below the competition tolerance. Chunked-vs-unchunked consistency
     is checked for both precisions at the same time (the forward has no
     cross-batch ops, so chunking must be exact up to GEMM algorithm choice).
  2. **Case-14 accuracy**: the fp16 candidate is compared chunk by chunk
     against that fp32 reference path at the full 100k sequence length,
     using the harness's exact element rule (abs <= atol OR rel <= rtol).
     Inputs are generated per chunk and streamed, so nothing close to the
     13 GB full activation ever lives on the GPU.
  3. **Timing**: the fp16 candidate only, full batch as a chunked pass with
     inputs streamed from host memory (the honest out-of-core setting; the
     H2D copies are a few percent of a forward at this size).

Runtime expectations on a T4: the candidate forward is roughly a minute
(~1.3e15 causal FLOPs); the full 32-sample fp32 reference pass is on the
order of 10-20 minutes. Use ``--max-samples 4`` for a quick accuracy read,
``--skip-accuracy`` for timing only, ``--seq-len 4096`` for a dry run.

Examples:

    python src/run_case14.py --max-samples 4          # quick end-to-end
    python src/run_case14.py                          # the full thing
    python src/run_case14.py --skip-accuracy --repeats 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import List, Optional

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import OptimizedTransformer, OptimizerSettings  # noqa: E402

# The candidate mirrors the sweep defaults; the reference keeps the same
# restructuring but full fp32 matmuls, with the SDPA backend pinned to the
# memory-efficient kernel so an unsupported-config fallback to the math
# backend (which would materialize the score tensor and OOM) raises instead.
CANDIDATE_SETTINGS = OptimizerSettings(precision="fp16")
REFERENCE_SETTINGS = OptimizerSettings(precision="fp32", sdpa_backend="efficient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=1024)

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2,
        help="samples per candidate (fp16) forward chunk",
    )
    parser.add_argument(
        "--reference-chunk-size",
        type=int,
        default=1,
        help="samples per reference (fp32) forward chunk",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="limit the accuracy comparison to the first N samples (0 = all)",
    )
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="skip the feasible-length validation of the fp32 proxy reference",
    )
    parser.add_argument(
        "--proxy-seq-len",
        type=int,
        default=1024,
        help="sequence length for the proxy validation (must fit the baseline)",
    )
    parser.add_argument("--proxy-batch-size", type=int, default=8)

    parser.add_argument(
        "--dual-gpu",
        action="store_true",
        help="pass 4: alternate timing chunks across two GPUs (needs 2 "
        "visible CUDA devices; accuracy still runs single-GPU against the "
        "fp32 reference). Zero inter-GPU traffic: outputs are discarded in "
        "the timing phase exactly as in the single-GPU pass.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None, help="optional JSON summary path")
    return parser.parse_args()


def build_pair(
    config: bench.TransformerConfig, device: torch.device, seed: int
):
    """Candidate and reference models sharing one set of baseline weights."""
    torch.manual_seed(seed)
    source = bench.BaselineTransformer(config)
    candidate = OptimizedTransformer(config, CANDIDATE_SETTINGS)
    reference = OptimizedTransformer(config, REFERENCE_SETTINGS)
    bench.copy_model_weights(source, candidate)
    bench.copy_model_weights(source, reference)
    del source
    return (
        candidate.to(device).eval(),
        reference.to(device).eval(),
    )


def generate_chunk(
    config: bench.TransformerConfig,
    device: torch.device,
    seed: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Deterministic fp32 input rows [start, stop); independent of chunking."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 100003 + start)
    return torch.randn(
        stop - start,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
    )


def chunked(model, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outputs = [
        model(x[start : start + chunk_size])
        for start in range(0, x.shape[0], chunk_size)
    ]
    return torch.cat(outputs, dim=0)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def validate_proxy(args: argparse.Namespace, device: torch.device) -> bool:
    """Feasible-length check that the fp32 chunked path matches the baseline."""
    config = bench.TransformerConfig(
        batch_size=args.proxy_batch_size,
        seq_len=args.proxy_seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=True,
    )
    print(f"\n=== Proxy validation @ seq_len={config.seq_len} ===")
    torch.manual_seed(args.seed)
    baseline = bench.BaselineTransformer(config)
    candidate = OptimizedTransformer(config, CANDIDATE_SETTINGS)
    reference = OptimizedTransformer(config, REFERENCE_SETTINGS)
    bench.copy_model_weights(baseline, candidate)
    bench.copy_model_weights(baseline, reference)
    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    reference = reference.to(device).eval()

    x = generate_chunk(config, device, args.seed, 0, config.batch_size)
    # An uneven chunk size exercises the last-partial-chunk path.
    chunk = max(1, config.batch_size // 2 - 1)

    with torch.inference_mode():
        want = baseline(x)
        ref_full = reference(x)
        ref_chunked = chunked(reference, x, chunk)
        cand_full = candidate(x)
        cand_chunked = chunked(candidate, x, chunk)

    proxy_error = max_abs(want, ref_chunked)
    ref_consistency = max_abs(ref_full, ref_chunked)
    cand_consistency = max_abs(cand_full, cand_chunked)
    harness_view = bench.compare_outputs(want, cand_chunked, args.rtol, args.atol)

    print(f"baseline vs fp32-chunked reference : max_abs={proxy_error:.3e}")
    print(f"fp32 chunked vs unchunked          : max_abs={ref_consistency:.3e}")
    print(f"fp16 chunked vs unchunked          : max_abs={cand_consistency:.3e}")
    print(
        "fp16 chunked vs baseline (harness rule): "
        f"{'PASS' if harness_view.passed else 'FAIL'} | "
        f"max_abs={harness_view.max_abs_error:.6g} | "
        f"failed={harness_view.failed_elements}/{harness_view.total_elements}"
    )

    # The proxy stands in for the baseline, so it must sit far below the
    # tolerance it will later arbitrate: two orders of margin.
    ok = proxy_error <= args.atol / 20 and harness_view.passed
    print(f"proxy validation: {'PASS' if ok else 'FAIL'}")

    del baseline, candidate, reference, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ok


def accuracy_and_inputs(
    candidate,
    reference,
    config: bench.TransformerConfig,
    args: argparse.Namespace,
    device: torch.device,
):
    """Stream chunks: compare candidate to the fp32 reference, keep inputs on CPU."""
    limit = args.max_samples if args.max_samples > 0 else config.batch_size
    stats = {"max_abs": 0.0, "max_rel": 0.0, "failed": 0, "total": 0, "passed": True}
    cpu_chunks: List[torch.Tensor] = []

    print(
        f"\n=== Case accuracy (fp16 candidate vs fp32 reference, "
        f"first {limit}/{config.batch_size} samples) ==="
    )
    with torch.inference_mode():
        for start in range(0, config.batch_size, args.chunk_size):
            stop = min(start + args.chunk_size, config.batch_size)
            x = generate_chunk(config, device, args.seed, start, stop)
            if not args.skip_accuracy and start < limit:
                began = time.time()
                ref_out = chunked(reference, x, args.reference_chunk_size)
                cand_out = candidate(x)
                result = bench.compare_outputs(
                    ref_out, cand_out, args.rtol, args.atol
                )
                stats["max_abs"] = max(stats["max_abs"], result.max_abs_error)
                stats["max_rel"] = max(stats["max_rel"], result.max_relative_error)
                stats["failed"] += result.failed_elements
                stats["total"] += result.total_elements
                stats["passed"] &= result.passed
                print(
                    f"samples {start:>2}-{stop - 1:>2}: "
                    f"{'PASS' if result.passed else 'FAIL'} | "
                    f"max_abs={result.max_abs_error:.6g} | "
                    f"failed={result.failed_elements}/{result.total_elements} | "
                    f"{time.time() - began:.1f}s",
                    flush=True,
                )
                del ref_out, cand_out
            cpu_chunks.append(x.to("cpu"))
            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not args.skip_accuracy:
        print(
            f"summary: {'PASS' if stats['passed'] else 'FAIL'} | "
            f"max_abs={stats['max_abs']:.6g} | max_rel={stats['max_rel']:.6g} | "
            f"failed={stats['failed']}/{stats['total']}"
        )
    return stats, cpu_chunks


def benchmark(
    candidate,
    cpu_chunks: List[torch.Tensor],
    config: bench.TransformerConfig,
    args: argparse.Namespace,
    device: torch.device,
    candidate2=None,
) -> Optional[dict]:
    if device.type != "cuda":
        print("\nbenchmark skipped: CUDA device required")
        return None

    dual = candidate2 is not None
    device2 = torch.device("cuda", 1) if dual else None
    print(
        f"\n=== Timing (fp16 candidate, chunk={args.chunk_size}, inputs "
        f"streamed from host{', chunks alternating across 2 GPUs' if dual else ''}) ==="
    )

    def one_pass() -> None:
        # With two GPUs, even chunks go to cuda:0 and odd chunks to cuda:1.
        # The pageable H2D copy blocks the host briefly, but each forward is
        # enqueued asynchronously, so while one GPU crunches its ~seconds of
        # kernels the host is already feeding the other -- compute on the two
        # devices overlaps almost fully at this chunk size. Outputs are
        # discarded, so no inter-GPU traffic exists.
        #
        # The device guard around the cuda:1 chunks is load-bearing: Triton
        # binds kernel launches to torch.cuda.current_device()'s stream, so
        # without it the fused-LN kernels would launch on cuda:0 against
        # cuda:1 pointers (illegal address) -- the same rule dual_gpu.py
        # documents and guards for.
        for index, x_cpu in enumerate(cpu_chunks):
            if dual and index % 2 == 1:
                with torch.cuda.device(device2):
                    candidate2(x_cpu.to(device2))
            else:
                candidate(x_cpu.to(device))

    def sync_all() -> None:
        torch.cuda.synchronize(device)
        if dual:
            torch.cuda.synchronize(device2)

    samples_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            one_pass()
        sync_all()
        for _ in range(args.repeats):
            if dual:
                # CUDA events on one device cannot cover the other's work, so
                # the dual pass is timed by wall clock between full syncs --
                # honest at these multi-second pass times.
                began = time.perf_counter()
                one_pass()
                sync_all()
                samples_ms.append((time.perf_counter() - began) * 1000.0)
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                one_pass()
                end.record()
                torch.cuda.synchronize(device)
                samples_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(samples_ms)
    tokens = config.batch_size * config.seq_len
    throughput = tokens * 1000.0 / median_ms
    print(
        f"candidate: median={median_ms:.1f} ms | mean="
        f"{statistics.fmean(samples_ms):.1f} ms | min={min(samples_ms):.1f} ms "
        f"| throughput={throughput:.0f} token/s | repeats={args.repeats}"
    )
    print("(no baseline column: the reference implementation cannot run this case)")
    return {"median_ms": median_ms, "samples_ms": samples_ms, "tokens_per_second": throughput}


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA device; this will be impractically slow")

    config = bench.TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=True,
    )
    config.validate()
    print("=== Configuration ===")
    print(config)
    print(f"device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    proxy_ok = True
    if not args.skip_proxy:
        proxy_ok = validate_proxy(args, device)
        if not proxy_ok:
            print("aborting: the fp32 proxy reference failed validation")
            return 2

    candidate, reference = build_pair(config, device, args.seed)
    stats, cpu_chunks = accuracy_and_inputs(
        candidate, reference, config, args, device
    )
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate2 = None
    if args.dual_gpu:
        if device.type == "cuda" and torch.cuda.device_count() >= 2:
            # Same weights, second device. Numerics are irrelevant for the
            # timing pass (outputs discarded) but keeping the weights
            # identical means a chunk computes the same result either way.
            candidate2 = OptimizedTransformer(config, CANDIDATE_SETTINGS)
            candidate2.load_state_dict(candidate.state_dict())
            candidate2 = candidate2.to(torch.device("cuda", 1)).eval()
        else:
            print("[dual-gpu] fewer than two visible GPUs; timing single-GPU")

    timing = benchmark(candidate, cpu_chunks, config, args, device, candidate2)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "config": vars(args),
                    "gpu": torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu",
                    "proxy_ok": proxy_ok,
                    "accuracy": stats if not args.skip_accuracy else None,
                    "timing": timing,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {out_path}")

    accuracy_ok = args.skip_accuracy or stats["passed"]
    return 0 if (proxy_ok and accuracy_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
