#!/usr/bin/env python3
"""Shape-specialized settings dispatch, generated from measured sweeps.

The problem statement explicitly invites per-shape implementations
("participants can choose different implementations for different shapes by
adding shape checks"). Rather than hand-writing those choices, this module
derives them from data: every sweep already records, per shape, which
optimizer settings ran and how fast. The generator scans ``results/*.json``,
keeps the fastest *accuracy-passing* run per shape, and writes the winning
settings to ``configs/dispatch.json``; ``run_case.py --dispatch`` then applies
that table for whatever shape it is given.

Generate (or refresh) the table after any new sweep:

    python src/dispatch.py                          # scan results/*.json
    python src/dispatch.py results/a.json results/b.json --out configs/dispatch.json

Use it:

    python src/run_case.py --dispatch --causal --batch-size 64 --d-model 128 \
        --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
    python src/sweep.py --skip 14 -- --dispatch

Rules of engagement:

  * Only accuracy-passing, status-ok records compete; reference-check runs
    are skipped.
  * Explicit command-line optimizer flags always beat the table -- dispatch
    fills in the fields you did not set.
  * Only ``OptimizerSettings`` fields are applied. Keys recorded from
    run-level flags of other branches (e.g. a future cuda_graphs) are kept
    in the table for provenance but ignored at apply time.
  * Medians from different sweep files were measured in different sessions
    (different clocks, temperatures, GPUs), so treat cross-file wins within
    a few percent as ties; the table records its provenance so you can
    re-measure head to head.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Dict, Iterable, Optional

SRC_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DEFAULT_TABLE_PATH = REPO_ROOT / "configs" / "dispatch.json"

SHAPE_FIELDS = (
    "batch_size",
    "seq_len",
    "d_model",
    "heads",
    "layers",
    "ffn_dim",
    "causal",
)

def _field_types() -> Dict[str, type]:
    """Field name -> declared type of OptimizerSettings, resolved lazily.

    Imported inside the function so that table *generation* stays runnable
    without torch installed; only applying a table needs the real dataclass.
    """
    from optimized import OptimizerSettings

    types: Dict[str, type] = {}
    for field in dataclasses.fields(OptimizerSettings):
        if field.type in ("int", int):
            types[field.name] = int
        elif field.type in ("bool", bool):
            types[field.name] = bool
        else:
            types[field.name] = str
    return types


# Command-line spellings of the OptimizerSettings fields in run_case.py; used
# to detect which fields the user set explicitly (those beat the table).
FLAG_TO_FIELD = {
    "--precision": "precision",
    "--attention": "attention",
    "--sdpa-backend": "sdpa_backend",
    "--no-fuse-qkv": "fuse_qkv",
    "--no-fused-norm": "fused_norm",
    "--assume-dense-mask": "assume_dense_mask",
    "--fp16-min-d-model": "fp16_min_d_model",
    "--fp16-max-elements": "fp16_max_elements",
}


def shape_key(
    batch_size: int,
    seq_len: int,
    d_model: int,
    heads: int,
    layers: int,
    ffn_dim: int,
    causal: bool,
) -> str:
    suffix = "causal" if causal else "bidir"
    return (
        f"b{batch_size}-s{seq_len}-d{d_model}-h{heads}"
        f"-l{layers}-f{ffn_dim}-{suffix}"
    )


def parse_optimizer_line(text: str) -> Dict[str, object]:
    """'precision=fp16 fuse_qkv=True ...' -> dict with real booleans."""
    settings: Dict[str, object] = {}
    for token in text.split():
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if value == "True":
            settings[key] = True
        elif value == "False":
            settings[key] = False
        else:
            try:
                settings[key] = int(value)
            except ValueError:
                settings[key] = value
    return settings


def load_table(path: Optional[pathlib.Path] = None) -> Optional[dict]:
    path = DEFAULT_TABLE_PATH if path is None else path
    if not path.exists():
        return None
    return json.loads(path.read_text())


def lookup(table: dict, **shape) -> Optional[dict]:
    """Table entry for a shape given as SHAPE_FIELDS keyword arguments."""
    return (table.get("entries") or {}).get(shape_key(**shape))


def explicit_setting_fields(argv: Iterable[str]) -> set:
    """OptimizerSettings fields the user pinned on the command line."""
    explicit = set()
    for token in argv:
        field = FLAG_TO_FIELD.get(token.split("=", 1)[0])
        if field is not None:
            explicit.add(field)
    return explicit


def applicable_settings(
    entry_settings: Dict[str, object], explicit: set
) -> Dict[str, object]:
    """The table settings that are real OptimizerSettings fields and were
    not explicitly set on the command line."""
    valid = _field_types()
    applied = {}
    for key, value in entry_settings.items():
        if key not in valid or key in explicit:
            continue
        # Tables written before values were typed carry strings; coerce
        # against the real OptimizerSettings field type so an int field is
        # never handed a str (that broke --dispatch with a TypeError in
        # _compute_dtype).
        field_type = valid[key]
        if field_type is int and not isinstance(value, bool):
            value = int(value)
        elif field_type is bool and isinstance(value, str):
            value = value == "True"
        applied[key] = value
    return applied


# --------------------------------------------------------------------------- #
# Table generation
# --------------------------------------------------------------------------- #


def build_entries(paths: Iterable[pathlib.Path]) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for record in data.get("cases", []):
            if record.get("status") != "ok":
                continue
            accuracy = record.get("accuracy") or {}
            if not accuracy.get("passed"):
                continue
            optimized_timing = record.get("optimized") or {}
            median = optimized_timing.get("median_ms")
            optimizer_line = (
                record.get("optimizer")
                or (data.get("environment") or {}).get("optimizer")
                or ""
            )
            settings = parse_optimizer_line(optimizer_line)
            if median is None or not settings:  # e.g. reference-check runs
                continue
            case = record.get("case") or {}
            try:
                key = shape_key(**{name: case[name] for name in SHAPE_FIELDS})
            except KeyError:
                continue
            best = entries.get(key)
            if best is None or median < best["median_ms"]:
                entries[key] = {
                    "settings": settings,
                    "median_ms": median,
                    "speedup": record.get("speedup"),
                    "max_abs_error": accuracy.get("max_abs_error"),
                    "source": path.name,
                    "case_id": case.get("id"),
                }
    return dict(sorted(entries.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results",
        nargs="*",
        help="result JSON files (default: results/*.json)",
    )
    parser.add_argument("--out", default=str(DEFAULT_TABLE_PATH))
    args = parser.parse_args()

    paths = [pathlib.Path(p) for p in args.results] or sorted(
        (REPO_ROOT / "results").glob("*.json")
    )
    if not paths:
        print("no result files found; run a sweep first")
        return 1

    entries = build_entries(paths)
    if not entries:
        print("no accuracy-passing records found in the given files")
        return 1

    payload = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [path.name for path in paths],
        "entries": entries,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    width = max(len(key) for key in entries)
    for key, entry in entries.items():
        print(
            f"{key:<{width}}  {entry['median_ms']:>9.3f} ms  "
            f"{entry['settings']}  <- {entry['source']}"
        )
    print(f"\nwrote {out_path} ({len(entries)} shapes from {len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
