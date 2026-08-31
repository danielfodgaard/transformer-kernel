#!/usr/bin/env python3
"""
Run a set of shapes through the benchmark and record the numbers as JSON.

The organizers' script benchmarks exactly one shape per invocation, so a sweep
means running it once per case. Each case runs as its own subprocess, which
keeps an out-of-memory case (notably appendix case 14) from taking the rest of
the sweep down with it, and stops torch.compile state and CUDA fragmentation
from leaking between shapes.

    # all 14 appendix shapes, default fp16 settings
    python src/sweep.py

    # a few cases, and compare precision modes
    python src/sweep.py --cases 1,7,8 --out fp16.json
    python src/sweep.py --cases 1,7,8 --out autocast.json -- --precision autocast

    # what the harness's own noise floor looks like
    python src/sweep.py --cases 1 --out noise.json -- --reference-check

Anything after ``--`` is forwarded verbatim to run_case.py, so every flag of
both the benchmark and the optimized implementation is reachable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

SRC_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
RUN_CASE = SRC_DIR / "run_case.py"

# Exit codes used by the benchmark's main().
EXIT_OK = 0
EXIT_ACCURACY_FAILED = 2

_ENV_RE = re.compile(
    r"^device=(?P<device>[^,]+), dtype=(?P<dtype>[^,]+), torch=(?P<torch>\S+)"
)
_GPU_RE = re.compile(r"^gpu=(?P<gpu>.+)$")
_OPTIMIZER_RE = re.compile(r"^optimizer: (?P<settings>.+)$")
_SUMMARY_RE = re.compile(
    r"^summary: (?P<status>PASS|FAIL) \| max_abs=(?P<max_abs>\S+) \| "
    r"max_rel=(?P<max_rel>\S+) \| failed=(?P<failed>\d+)/(?P<total>\d+)"
)
_TIMING_RE = re.compile(
    r"^(?P<which>baseline|optimized)\s*: median=(?P<median>\S+) ms \| "
    r"mean=(?P<mean>\S+) ms \| p90=(?P<p90>\S+) ms \| min=(?P<min>\S+) ms \| "
    r"throughput=(?P<throughput>\S+) token/s"
)
_SPEEDUP_RE = re.compile(r"^speedup\s*: (?P<speedup>\S+)x")


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def parse_benchmark_output(stdout: str) -> Dict:
    """Pull the structured numbers out of the benchmark's printed report.

    Parsing stdout rather than reimplementing the measurement keeps the
    official methodology (5 accuracy trials, 20 warmup iterations, 3 rounds of
    100 alternating repeats, median of 300 CUDA-event samples) exactly intact.
    """
    parsed: Dict = {
        "device": None,
        "dtype": None,
        "torch": None,
        "gpu": None,
        "optimizer": None,
        "accuracy": None,
        "baseline": None,
        "optimized": None,
        "speedup": None,
    }

    for line in stdout.splitlines():
        line = line.rstrip()

        match = _ENV_RE.match(line)
        if match:
            parsed["device"] = match.group("device")
            parsed["dtype"] = match.group("dtype")
            parsed["torch"] = match.group("torch")
            continue

        match = _GPU_RE.match(line)
        if match:
            parsed["gpu"] = match.group("gpu")
            continue

        match = _OPTIMIZER_RE.match(line)
        if match:
            parsed["optimizer"] = match.group("settings")
            continue

        match = _SUMMARY_RE.match(line)
        if match:
            parsed["accuracy"] = {
                "passed": match.group("status") == "PASS",
                "max_abs_error": _to_float(match.group("max_abs")),
                "max_relative_error": _to_float(match.group("max_rel")),
                "failed_elements": int(match.group("failed")),
                "total_elements": int(match.group("total")),
            }
            continue

        match = _TIMING_RE.match(line)
        if match:
            parsed[match.group("which")] = {
                "median_ms": _to_float(match.group("median")),
                "mean_ms": _to_float(match.group("mean")),
                "p90_ms": _to_float(match.group("p90")),
                "min_ms": _to_float(match.group("min")),
                "tokens_per_second": _to_float(match.group("throughput")),
            }
            continue

        match = _SPEEDUP_RE.match(line)
        if match:
            parsed["speedup"] = _to_float(match.group("speedup"))

    return parsed


def case_command(case: Dict, passthrough: List[str]) -> List[str]:
    command = [
        sys.executable,
        str(RUN_CASE),
        "--batch-size", str(case["batch_size"]),
        "--seq-len", str(case["seq_len"]),
        "--d-model", str(case["d_model"]),
        "--heads", str(case["heads"]),
        "--ffn-dim", str(case["ffn_dim"]),
        "--layers", str(case["layers"]),
    ]
    if case.get("causal"):
        command.append("--causal")
    # Global passthrough first, then the case's own extra_args so a per-shape
    # config (e.g. configs/best.json) can specialize flags per case.
    return command + passthrough + list(case.get("extra_args", []))


def classify(returncode: int, parsed: Dict, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == EXIT_OK and parsed.get("speedup") is not None:
        return "ok"
    if returncode == EXIT_ACCURACY_FAILED:
        return "accuracy_failed"
    return "error"


def run_case(
    case: Dict, passthrough: List[str], timeout: float, verbose: bool
) -> Dict:
    command = case_command(case, passthrough)
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        stdout, stderr, returncode = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        stdout = expired.stdout or ""
        stderr = expired.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = -1

    if verbose:
        print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)

    parsed = parse_benchmark_output(stdout)
    record = {
        "case": case,
        "command": command,
        "status": classify(returncode, parsed, timed_out),
        "returncode": returncode,
        "wall_seconds": round(time.time() - started, 2),
        **parsed,
    }
    if record["status"] != "ok":
        # Keep enough of the failure to tell an OOM from a real bug, without
        # dragging a full traceback into every result file.
        record["stderr_tail"] = "\n".join(stderr.strip().splitlines()[-12:])
    return record


def format_table(records: List[Dict]) -> str:
    header = (
        f"{'#':>3}  {'shape':<34} {'status':<16} "
        f"{'baseline':>10} {'optimized':>10} {'speedup':>8}  {'max_abs':>9}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        case = record["case"]
        shape = (
            f"b{case['batch_size']} s{case['seq_len']} d{case['d_model']} "
            f"h{case['heads']} l{case['layers']} f{case['ffn_dim']}"
        )
        baseline = record.get("baseline") or {}
        optimized = record.get("optimized") or {}
        accuracy = record.get("accuracy") or {}
        speedup = record.get("speedup")
        max_abs = accuracy.get("max_abs_error")
        lines.append(
            f"{case['id']:>3}  {shape:<34} {record['status']:<16} "
            f"{_fmt(baseline.get('median_ms')):>10} "
            f"{_fmt(optimized.get('median_ms')):>10} "
            f"{_fmt(speedup, 'x'):>8}  {_fmt(max_abs, '', '.2e'):>9}"
        )
    return "\n".join(lines)


def _fmt(value: Optional[float], suffix: str = "", spec: str = ".3f") -> str:
    if value is None:
        return "-"
    return f"{value:{spec}}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "shapes.json"),
        help="JSON file with a 'cases' list (default: the appendix shapes)",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case ids to run (default: all)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated case ids to skip, e.g. 14 on a small GPU",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path (default: results/<config name>-<timestamp>.json)",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream each case's output instead of only the summary table",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run and exit",
    )
    return parser.parse_args()


def main() -> int:
    # Everything after "--" goes to run_case.py untouched.
    passthrough: List[str] = []
    if "--" in sys.argv:
        separator = sys.argv.index("--")
        passthrough = sys.argv[separator + 1 :]
        sys.argv = sys.argv[:separator]

    args = parse_args()

    config = json.loads(pathlib.Path(args.config).read_text())
    cases = config["cases"]

    if args.cases:
        wanted = {int(value) for value in args.cases.split(",") if value.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.skip:
        unwanted = {int(value) for value in args.skip.split(",") if value.strip()}
        cases = [case for case in cases if case["id"] not in unwanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 1

    if args.dry_run:
        for case in cases:
            print(" ".join(case_command(case, passthrough)))
        return 0

    records: List[Dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] case {case['id']} ...", flush=True)
        record = run_case(case, passthrough, args.timeout, args.verbose)
        records.append(record)
        print(
            f"    {record['status']}"
            + (
                f" | speedup {record['speedup']:.3f}x"
                if record.get("speedup") is not None
                else ""
            ),
            flush=True,
        )

    first_ok = next((r for r in records if r.get("gpu")), None)
    payload = {
        "schema": 1,
        "config": config.get("name", pathlib.Path(args.config).stem),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "passthrough_args": passthrough,
        "environment": {
            "gpu": first_ok.get("gpu") if first_ok else None,
            "torch": first_ok.get("torch") if first_ok else None,
            "device": first_ok.get("device") if first_ok else None,
            "dtype": first_ok.get("dtype") if first_ok else None,
            "optimizer": first_ok.get("optimizer") if first_ok else None,
        },
        "cases": records,
    }

    out_path = pathlib.Path(
        args.out
        or REPO_ROOT
        / "results"
        / f"{payload['config']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(format_table(records))
    print()
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")

    return 0 if all(record["status"] == "ok" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
