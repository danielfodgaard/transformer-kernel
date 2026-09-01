#!/usr/bin/env python3
"""
Run the organizers' benchmark with our optimized implementation swapped in.

The benchmark file itself is never edited. ``main()`` looks up
``UserOptimizedTransformer`` as a module global at call time, so rebinding that
attribute before calling ``main()`` is enough to substitute our class.

Every flag the benchmark defines still works and is passed straight through;
the flags added here all control the optimized implementation:

    --precision {fp16,autocast,fp32}   how the matmuls are run
    --attention {sdpa,triton,bmm,math} SDPA; the experimental pass-3 Triton
                                       flash kernel (measured slower on T4);
                                       the pass-4 materialized-score bmm path
                                       for head_dim > 128 (case 8); or the
                                       baseline's attention
    --sdpa-backend {auto,efficient,math,flash}
    --no-fuse-qkv                      keep three separate Q/K/V matmuls
    --no-fused-norm                    eager residual+LayerNorm chain instead
                                       of the fused Triton kernels
    --fused-out-proj                   pass-4: fold bias+residual+LayerNorm
                                       into the out_proj GEMM (Triton)
    --fused-ffn                        pass-4: same for ffn_out, plus fused
                                       bias+erf-GELU into ffn_in (Triton)
    --gelu-epilogue                    pass-4 probe: cuBLASLt fused bias+GELU
                                       for ffn_in -- TANH approximation, may
                                       cost accuracy margin; measurement only
    --assume-dense-mask                skip the all-True mask check (sync)
    --fp32-reductions                  forbid fp16 partial-sum accumulation
                                       inside cuBLAS fp16 matmuls (accuracy-
                                       margin knob; baseline is unaffected)
    --cuda-graphs                      capture the dense forward into a CUDA
                                       graph and replay it (single launch per
                                       call; do not combine with
                                       --compile-user)
    --graph-streams N                  pass-4 probe with --cuda-graphs:
                                       capture the forward as N=2 half-batch
                                       chains on two streams so per-kernel
                                       latencies overlap inside the graph
    --dual-gpu                         pass-4, 2x-T4 environments: split the
                                       batch across two GPUs (opt-in, an
                                       environment extension -- never the
                                       submission configuration). No-op with
                                       a single visible GPU; small shapes
                                       stay single-GPU via --dual-min-elements
    --dual-min-elements N              minimum input elements before --dual-gpu
                                       splits (default 4000000)
    --dual-fraction F                  pin the GPU1 batch share (0 < F <= 0.5)
                                       instead of auto-calibrating
    --dual-verify                      print max|dual - single| on the first
                                       eligible calls (untimed phase only)
    --dispatch                         apply the measured-best settings for
                                       this shape from configs/dispatch.json
                                       (generate with src/dispatch.py);
                                       explicit flags still win
    --reference-check                  run the baseline against itself, which
                                       measures the harness's own noise floor

Examples:
    python src/run_case.py --causal --batch-size 64 --d-model 128 \
        --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
    python src/run_case.py --precision autocast --compile-user
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
import optimized  # noqa: E402

EXTRA_FLAGS = (
    "--precision",
    "--attention",
    "--sdpa-backend",
    "--no-fuse-qkv",
    "--no-fused-norm",
    "--fused-out-proj",
    "--fused-ffn",
    "--gelu-epilogue",
    "--assume-dense-mask",
    "--fp16-min-d-model",
    "--fp16-max-elements",
    "--fp32-reductions",
    "--cuda-graphs",
    "--graph-streams",
    "--dual-gpu",
    "--dual-min-elements",
    "--dual-fraction",
    "--dual-verify",
    "--dispatch",
    "--reference-check",
)


def build_parser() -> argparse.ArgumentParser:
    # add_help=False so that --help reaches the benchmark's own parser; we
    # print our extra flags ourselves just before that happens.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--precision", choices=("fp16", "autocast", "fp32"), default="fp16"
    )
    parser.add_argument(
        "--attention", choices=("sdpa", "triton", "bmm", "math"), default="sdpa"
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "efficient", "math", "flash"),
        default="auto",
    )
    parser.add_argument("--no-fuse-qkv", action="store_true")
    parser.add_argument("--no-fused-norm", action="store_true")
    parser.add_argument("--fused-out-proj", action="store_true")
    parser.add_argument("--fused-ffn", action="store_true")
    parser.add_argument("--gelu-epilogue", action="store_true")
    parser.add_argument("--fp16-min-d-model", type=int, default=64)
    parser.add_argument("--fp16-max-elements", type=int, default=0)
    parser.add_argument("--assume-dense-mask", action="store_true")
    parser.add_argument("--fp32-reductions", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--graph-streams", type=int, default=1)
    parser.add_argument("--dual-gpu", action="store_true")
    parser.add_argument("--dual-min-elements", type=int, default=4_000_000)
    parser.add_argument("--dual-fraction", type=float, default=None)
    parser.add_argument("--dual-verify", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--reference-check", action="store_true")
    return parser


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Flags below are the benchmark's own:\n")

    raw_argv = list(sys.argv[1:])
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    # Hand the remaining argv to the benchmark's parse_args() untouched.
    sys.argv = [sys.argv[0]] + passthrough

    if args.fp32_reductions:
        # cuBLAS may split an fp16 GEMM along K and accumulate the partial
        # sums in fp16; this forbids that, keeping every reduction in fp32.
        # Only fp16 matmuls are affected, so the fp32 baseline is untouched.
        # Costs a little GEMM speed; buys absolute-error margin on the shapes
        # that sit near the 2e-3 tolerance (appendix case 7 measured 1.97e-3).
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    dispatch_overrides = {}
    dispatch_note = None
    if args.dispatch and not args.reference_check:
        import dispatch  # deferred: only needed when the table is in play

        table = dispatch.load_table()
        if table is None:
            dispatch_note = (
                "no table at configs/dispatch.json (generate with "
                "src/dispatch.py); using command-line settings"
            )
        else:
            # The benchmark re-parses the same argv inside bench.main(); this
            # early parse only reads the shape, so parsing twice is harmless.
            shape = bench.parse_args()
            entry = dispatch.lookup(
                table,
                batch_size=shape.batch_size,
                seq_len=shape.seq_len,
                d_model=shape.d_model,
                heads=shape.heads,
                layers=shape.layers,
                ffn_dim=shape.ffn_dim,
                causal=shape.causal,
            )
            if entry is None:
                dispatch_note = "shape not in table; using command-line settings"
            else:
                explicit = dispatch.explicit_setting_fields(raw_argv)
                dispatch_overrides = dispatch.applicable_settings(
                    entry["settings"], explicit
                )
                dispatch_note = (
                    f"applied {dispatch_overrides or 'nothing new'} "
                    f"from {entry['source']} "
                    f"(measured {entry['median_ms']} ms)"
                )

    if args.reference_check:
        # Baseline vs baseline: any deviation from 1.00x is harness noise.
        bench.UserOptimizedTransformer = bench.BaselineTransformer
        print("optimizer: reference-check (baseline vs baseline)")
    else:
        settings = optimized.configure(
            precision=args.precision,
            attention=args.attention,
            sdpa_backend=args.sdpa_backend,
            fuse_qkv=not args.no_fuse_qkv,
            fused_norm=not args.no_fused_norm,
            fused_out_proj=args.fused_out_proj,
            fused_ffn=args.fused_ffn,
            gelu_epilogue=args.gelu_epilogue,
            assume_dense_mask=args.assume_dense_mask,
            fp16_min_d_model=args.fp16_min_d_model,
            fp16_max_elements=args.fp16_max_elements,
        )
        if dispatch_overrides:
            settings = optimized.configure(**dispatch_overrides)
        if args.cuda_graphs:
            if "--compile-user" in passthrough:
                print(
                    "[warning] --cuda-graphs with --compile-user is untested; "
                    "prefer one or the other"
                )
            import cuda_graphs  # deferred: pulls in the CUDA graph wrapper

            if args.graph_streams > 1:
                cuda_graphs.GraphedTransformer.graph_streams = args.graph_streams
            model_cls = cuda_graphs.GraphedTransformer
        else:
            model_cls = optimized.OptimizedTransformer
        if args.dual_gpu:
            if "--compile-user" in passthrough:
                print(
                    "[warning] --dual-gpu with --compile-user is unsupported; "
                    "the dual region always runs eager"
                )
            import dual_gpu  # deferred: pulls in the dual-GPU wrapper

            model_cls = dual_gpu.make_dual(
                model_cls,
                min_elements=args.dual_min_elements,
                fraction=args.dual_fraction,
                verify=args.dual_verify,
            )
        bench.UserOptimizedTransformer = model_cls
        print(
            "optimizer: "
            f"precision={settings.precision} "
            f"attention={settings.attention} "
            f"sdpa_backend={settings.sdpa_backend} "
            f"fuse_qkv={settings.fuse_qkv} "
            f"fused_norm={settings.fused_norm} "
            f"fused_out_proj={settings.fused_out_proj} "
            f"fused_ffn={settings.fused_ffn} "
            f"gelu_epilogue={settings.gelu_epilogue} "
            f"assume_dense_mask={settings.assume_dense_mask} "
            f"fp16_min_d_model={settings.fp16_min_d_model} "
            f"fp16_max_elements={settings.fp16_max_elements} "
            f"fp32_reductions={args.fp32_reductions} "
            f"cuda_graphs={args.cuda_graphs} "
            f"graph_streams={args.graph_streams} "
            f"dual_gpu={args.dual_gpu}"
        )
        if args.dual_gpu and torch.cuda.is_available() and torch.cuda.device_count() < 2:
            print("[dual-gpu] fewer than two visible GPUs; running single-GPU")
        if dispatch_note is not None:
            print(f"dispatch: {dispatch_note}")

    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
