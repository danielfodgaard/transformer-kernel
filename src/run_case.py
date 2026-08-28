#!/usr/bin/env python3
"""
Run the organizers' benchmark with our optimized implementation swapped in.

The benchmark file itself is never edited. ``main()`` looks up
``UserOptimizedTransformer`` as a module global at call time, so rebinding that
attribute before calling ``main()`` is enough to substitute our class.

Every flag the benchmark defines still works and is passed straight through;
the flags added here all control the optimized implementation:

    --precision {fp16,autocast,fp32}   how the matmuls are run
    --attention {sdpa,math}            fused SDPA or the baseline's attention
    --sdpa-backend {auto,efficient,math,flash}
    --no-fuse-qkv                      keep three separate Q/K/V matmuls
    --assume-dense-mask                skip the all-True mask check (sync)
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

import torch_transformer_benchmark as bench  # noqa: E402
import optimized  # noqa: E402

EXTRA_FLAGS = (
    "--precision",
    "--attention",
    "--sdpa-backend",
    "--no-fuse-qkv",
    "--assume-dense-mask",
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
    parser.add_argument("--attention", choices=("sdpa", "math"), default="sdpa")
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "efficient", "math", "flash"),
        default="auto",
    )
    parser.add_argument("--no-fuse-qkv", action="store_true")
    parser.add_argument("--assume-dense-mask", action="store_true")
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
            assume_dense_mask=args.assume_dense_mask,
        )
        if dispatch_overrides:
            settings = optimized.configure(**dispatch_overrides)
        bench.UserOptimizedTransformer = optimized.OptimizedTransformer
        print(
            "optimizer: "
            f"precision={settings.precision} "
            f"attention={settings.attention} "
            f"sdpa_backend={settings.sdpa_backend} "
            f"fuse_qkv={settings.fuse_qkv} "
            f"assume_dense_mask={settings.assume_dense_mask}"
        )
        if dispatch_note is not None:
            print(f"dispatch: {dispatch_note}")

    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
