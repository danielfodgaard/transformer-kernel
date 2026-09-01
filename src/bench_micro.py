#!/usr/bin/env python3
"""Isolated microbenchmarks backing the pass-4 attribution work.

Four subcommands, all median-of-N with CUDA-event timing on the current
stream (the harness's own methodology):

  gemm     -- fp16 torch.matmul TFLOPS for given MxKxN shapes; answers how
              much of case 8's 25 ms is simply cuBLAS on 40 SMs.
  sdpa     -- F.scaled_dot_product_attention across a head_dim sweep;
              measures the mem-efficient kernel's hd cliff on sm_75.
  bmmattn  -- the pass-4 materialized-score path (bmm + fused Triton causal
              softmax + bmm) and its eager fp32-softmax equivalent, head to
              head with sdpa at the same shape.
  p2p      -- cuda:0 <-> cuda:1 copy bandwidth and peer-access capability;
              the go/no-go number for the dual-GPU split's balance model.

Examples:
    python src/bench_micro.py gemm --shapes 8192x1024x3072,8192x1024x1024
    python src/bench_micro.py sdpa --b 64 --heads 4 --seq 128 --hd-sweep 32,64,128,256 --causal
    python src/bench_micro.py bmmattn --b 64 --heads 4 --seq 128 --hd 256 --causal
    python src/bench_micro.py p2p
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from typing import Callable, List

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def time_ms(fn: Callable[[], None], warmup: int = 20, repeats: int = 100) -> float:
    """Median CUDA-event latency of ``fn`` on the current stream."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for index in range(repeats):
            starts[index].record()
            fn()
            ends[index].record()
        torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def cmd_gemm(args: argparse.Namespace) -> None:
    for spec in args.shapes.split(","):
        m, k, n = (int(value) for value in spec.lower().split("x"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        median = time_ms(lambda: torch.matmul(a, b), args.warmup, args.repeats)
        flops = 2.0 * m * k * n
        print(
            f"gemm {m}x{k}x{n} fp16: {median:.3f} ms | "
            f"{flops / median / 1e9:.1f} TFLOPS"
        )


def _qkv(args: argparse.Namespace, hd: int):
    shape = (args.b, args.heads, args.seq, hd)
    generator = torch.Generator(device="cuda").manual_seed(0)
    make = lambda: torch.randn(  # noqa: E731
        shape, generator=generator, device="cuda", dtype=torch.float16
    )
    return make(), make(), make()


def cmd_sdpa(args: argparse.Namespace) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for hd in (int(value) for value in args.hd_sweep.split(",")):
        q, k, v = _qkv(args, hd)

        def run() -> None:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                F.scaled_dot_product_attention(
                    q, k, v, is_causal=args.causal, scale=hd**-0.5
                )

        median = time_ms(run, args.warmup, args.repeats)
        print(
            f"sdpa[efficient] b{args.b} h{args.heads} s{args.seq} hd{hd} "
            f"causal={args.causal}: {median:.3f} ms"
        )


def cmd_bmmattn(args: argparse.Namespace) -> None:
    import fused_kernels

    hd = args.hd
    scale = hd**-0.5
    q, k, v = _qkv(args, hd)
    bh = args.b * args.heads
    q2 = q.view(bh, args.seq, hd)
    k2 = k.view(bh, args.seq, hd)
    v2 = v.view(bh, args.seq, hd)
    causal_block = torch.ones(
        (args.seq, args.seq), device="cuda", dtype=torch.bool
    ).triu(diagonal=1)

    def run_triton() -> None:
        scores = torch.bmm(q2, k2.transpose(1, 2))
        probs = fused_kernels.causal_softmax_fwd(scores, scale, causal=args.causal)
        torch.bmm(probs, v2)

    def run_eager() -> None:
        scores = torch.bmm(q2, k2.transpose(1, 2)).float().mul_(scale)
        if args.causal:
            scores.masked_fill_(causal_block, float("-inf"))
        probs = scores.softmax(-1).to(torch.float16)
        torch.bmm(probs, v2)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    def run_sdpa() -> None:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            F.scaled_dot_product_attention(
                q, k, v, is_causal=args.causal, scale=scale
            )

    shape = f"b{args.b} h{args.heads} s{args.seq} hd{hd} causal={args.causal}"
    if fused_kernels.HAVE_TRITON:
        print(f"bmm+triton-softmax {shape}: {time_ms(run_triton, args.warmup, args.repeats):.3f} ms")
    else:
        print("bmm+triton-softmax skipped: no triton")
    print(f"bmm+eager-fp32-softmax {shape}: {time_ms(run_eager, args.warmup, args.repeats):.3f} ms")
    print(f"sdpa[efficient] {shape}: {time_ms(run_sdpa, args.warmup, args.repeats):.3f} ms")


def cmd_p2p(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() < 2:
        print("p2p: fewer than two visible CUDA devices")
        return
    peer01 = torch.cuda.can_device_access_peer(0, 1)
    peer10 = torch.cuda.can_device_access_peer(1, 0)
    print(f"can_device_access_peer: 0->1 {peer01} | 1->0 {peer10}")
    mb = args.mb
    a = torch.empty(mb * 1024 * 1024 // 4, device="cuda:0", dtype=torch.float32)
    b = torch.empty_like(a, device="cuda:1")
    for name, fn in (
        ("cuda:0 -> cuda:1", lambda: b.copy_(a, non_blocking=True)),
        ("cuda:1 -> cuda:0", lambda: a.copy_(b, non_blocking=True)),
    ):
        median = time_ms(fn, warmup=5, repeats=20)
        print(f"D2D {name}: {median:.2f} ms for {mb} MB = {mb / 1024 / (median / 1000):.2f} GB/s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gemm = sub.add_parser("gemm")
    p_gemm.add_argument("--shapes", default="8192x1024x3072,8192x1024x1024")

    for name in ("sdpa", "bmmattn"):
        p = sub.add_parser(name)
        p.add_argument("--b", type=int, default=64)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--seq", type=int, default=128)
        p.add_argument("--causal", action="store_true")
        if name == "sdpa":
            p.add_argument("--hd-sweep", default="32,64,128,256")
        else:
            p.add_argument("--hd", type=int, default=256)

    p_p2p = sub.add_parser("p2p")
    p_p2p.add_argument("--mb", type=int, default=256)

    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("bench_micro needs a CUDA device")
        return 1
    {"gemm": cmd_gemm, "sdpa": cmd_sdpa, "bmmattn": cmd_bmmattn, "p2p": cmd_p2p}[
        args.command
    ](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
