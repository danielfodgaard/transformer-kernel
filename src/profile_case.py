#!/usr/bin/env python3
"""Per-kernel CUDA time attribution for one shape of the optimized model.

The pass-4 designs lean on inferred kernel inventories (which ops dominate
case 8, how many kernels a graphed forward replays); this tool replaces the
inference with a measurement. It builds ``OptimizedTransformer`` (or the
CUDA-graph wrapper) with the requested settings, warms up past any lazy
caches and graph capture, then profiles steady-state forwards and prints the
per-op CUDA time table plus the distinct-kernel count.

    python src/profile_case.py --causal --batch-size 64 --seq-len 128 \
        --d-model 1024 --heads 4 --ffn-dim 1024 --layers 4
    python src/profile_case.py --causal --batch-size 1 --seq-len 128 \
        --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --cuda-graphs

Weights are random (no baseline is built): profiling needs representative
shapes and dtypes, not validated numerics.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

import optimized  # noqa: E402
from torch_transformer_benchmark import TransformerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument("--precision", choices=("fp16", "autocast", "fp32"), default="fp16")
    parser.add_argument("--attention", choices=("sdpa", "triton", "bmm", "math"), default="sdpa")
    parser.add_argument("--sdpa-backend", choices=("auto", "efficient", "math", "flash"), default="auto")
    parser.add_argument("--no-fused-norm", action="store_true")
    parser.add_argument("--fused-out-proj", action="store_true")
    parser.add_argument("--fused-ffn", action="store_true")
    parser.add_argument("--gelu-epilogue", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--graph-streams", type=int, default=1)

    parser.add_argument("--warmup", type=int, default=30, help=">= 9 so graph capture completes first")
    parser.add_argument("--profiled-iters", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("profile_case needs a CUDA device")
        return 1
    device = torch.device("cuda")

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    settings = optimized.configure(
        precision=args.precision,
        attention=args.attention,
        sdpa_backend=args.sdpa_backend,
        fused_norm=not args.no_fused_norm,
        fused_out_proj=args.fused_out_proj,
        fused_ffn=args.fused_ffn,
        gelu_epilogue=args.gelu_epilogue,
    )

    if args.cuda_graphs:
        import cuda_graphs

        if args.graph_streams > 1:
            cuda_graphs.GraphedTransformer.graph_streams = args.graph_streams
        model = cuda_graphs.GraphedTransformer(config, settings)
    else:
        model = optimized.OptimizedTransformer(config, settings)
    model = model.to(device).eval()

    x = torch.randn(
        config.batch_size, config.seq_len, config.d_model, device=device
    )
    mask = torch.ones(
        config.batch_size, config.seq_len, device=device, dtype=torch.bool
    )

    print(f"config: {config}")
    print(f"settings: {settings}")
    print(f"gpu: {torch.cuda.get_device_name(device)} | torch {torch.__version__}")

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, mask)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(args.profiled_iters):
                model(x, mask)
            torch.cuda.synchronize()

    averages = prof.key_averages()
    print(
        averages.table(
            sort_by="self_cuda_time_total", row_limit=args.row_limit
        )
    )
    kernels = [
        event
        for event in averages
        if event.self_device_time_total > 0 and event.device_type is not None
    ]
    launches = sum(
        event.count
        for event in averages
        if event.key in ("cudaLaunchKernel", "cudaGraphLaunch")
    )
    print(
        f"distinct device-time entries: {len(kernels)} | "
        f"kernel/graph launches over {args.profiled_iters} iters: {launches} "
        f"(~{launches / max(args.profiled_iters, 1):.1f}/forward)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
