#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import subprocess
import time


LATENCY_RE = re.compile(
    r"latency_ms\s+full=(?P<full>[0-9.]+)\s+"
    r"(?P<label>tp[48])_compute=(?P<compute>[0-9.]+)\s+"
    r"(?P=label)_reduce=(?P<reduce>[0-9.]+)\s+"
    r"(?P=label)_total=(?P<total>[0-9.]+)\s+"
    r"compute_speedup=(?P<compute_speedup>[0-9.]+)\s+"
    r"total_speedup=(?P<total_speedup>[0-9.]+)\s+"
    r"input_mib=(?P<input_mib>[0-9.]+)\s+"
    r"output_mib=(?P<output_mib>[0-9.]+)"
)
CORRECTNESS_RE = re.compile(
    r"correctness(?:_host_sum)?\s+routes=(?P<routes>[0-9]+)\s+"
    r"values=(?P<values>[0-9]+)\s+"
    r"max_abs=(?P<max_abs>[0-9.eE+-]+)\s+"
    r"rel=(?P<rel>[0-9.eE+-]+)\s+"
    r"bad=(?P<bad>[0-9]+)\s+"
    r"(?:bad_frac=[0-9.eE+-]+\s+)?"
    r"nan=(?P<nan>[0-9]+)\s+"
    r"(?P<result>PASS|FAIL|ok)"
)


def run_command(cmd, cwd, out_dir, timeout):
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.time() - started
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_profile_json(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def parse_workbench(stdout):
    out = {}
    for line in stdout.splitlines():
        m = CORRECTNESS_RE.search(line)
        if m:
            out.update(
                {
                    "routes": int(m.group("routes")),
                    "values": int(m.group("values")),
                    "max_abs": float(m.group("max_abs")),
                    "rel": float(m.group("rel")),
                    "bad": int(m.group("bad")),
                    "nan": int(m.group("nan")),
                    "correctness": "PASS" if m.group("result") == "ok" else m.group("result"),
                }
            )
        m = LATENCY_RE.search(line)
        if m:
            out.update(
                {
                    "full_ms": float(m.group("full")),
                    "tp_label": m.group("label"),
                    "tp_compute_ms": float(m.group("compute")),
                    "tp_reduce_ms": float(m.group("reduce")),
                    "tp_total_ms": float(m.group("total")),
                    "compute_speedup": float(m.group("compute_speedup")),
                    "total_speedup": float(m.group("total_speedup")),
                    "input_mib": float(m.group("input_mib")),
                    "output_mib": float(m.group("output_mib")),
                }
            )
    return out


def run_ep8(args):
    out_dir = args.artifact_dir / "ep8-direct"
    cmd = [
        str(args.python),
        "tools/ds4-v100-tp-ep-profile.py",
        "--run-mode",
        "direct-token-major",
        "--tool",
        "none",
        "--artifact-dir",
        str(out_dir),
        "--pack-dir",
        str(args.pack_dir),
        "--contract",
        str(args.contract),
        "--turbomind-lib",
        str(args.turbomind_lib),
        "--tokens",
        str(args.tokens),
        "--position",
        str(args.position),
        "--slots",
        str(args.slots),
        "--model-router-routes",
        "--compact-moe-decode",
        "--request-timeout-seconds",
        str(args.timeout_seconds),
    ]
    result = run_command(cmd, args.repo_dir, out_dir, args.timeout_seconds + 60)
    summary = parse_profile_json(result["stdout"])
    return {
        "kind": "ep8-direct-serving",
        "returncode": result["returncode"],
        "elapsed_s": result["elapsed_s"],
        "tokens": args.tokens,
        "slots": args.slots,
        "position": args.position,
        "first_token": summary.get("output_head_first_token"),
        "direct_generated_tok_s": summary.get("serving_aggregate_generated_tok_s_decode"),
        "direct_wall_tok_s": summary.get("serving_aggregate_generated_tok_s_wall"),
        "decode_ms": summary.get("serving_total_decode_ms"),
        "wall_ms": summary.get("serving_total_wall_ms"),
        "ep_ms": summary.get("scaffold_sum_ep_ms"),
        "compose_ms": summary.get("scaffold_sum_compose_ms"),
        "compact_moe_routes": summary.get("compact_moe_routes"),
        "compact_moe_all_dest_bytes": summary.get("compact_moe_all_dest_bytes"),
        "compact_moe_compact_bytes": summary.get("compact_moe_compact_bytes"),
    }


def run_tp8_case(args, tokens_per_active):
    out_dir = args.artifact_dir / "tp8-turbomind" / f"tokens-per-active-{tokens_per_active}"
    cmd = [
        "tools/ds4-v100-tp8-turbomind-ffn-smoke",
        "--lib",
        str(args.turbomind_lib),
        "--tokens-per-active",
        str(tokens_per_active),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    result = run_command(cmd, args.repo_dir, out_dir, args.timeout_seconds)
    parsed = parse_workbench(result["stdout"])
    parsed.update(
        {
            "kind": "tp8-turbomind-workbench",
            "returncode": result["returncode"],
            "elapsed_s": result["elapsed_s"],
            "tokens_per_active": tokens_per_active,
        }
    )
    return parsed


def run_tp4_case(args, tokens_per_active):
    out_dir = args.artifact_dir / "tp4-turbomind" / f"tokens-per-active-{tokens_per_active}"
    cmd = [
        "tools/ds4-v100-tp4-turbomind-layer-smoke",
        "--lib",
        str(args.turbomind_lib),
        "--tokens-per-active",
        str(tokens_per_active),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    result = run_command(cmd, args.repo_dir, out_dir, args.timeout_seconds)
    parsed = parse_workbench(result["stdout"])
    parsed.update(
        {
            "kind": "tp4-turbomind-workbench",
            "returncode": result["returncode"],
            "elapsed_s": result["elapsed_s"],
            "tokens_per_active": tokens_per_active,
        }
    )
    return parsed


def write_outputs(args, rows):
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "slots": args.slots,
        "position": args.position,
        "tokens": args.tokens,
        "route_tiers": args.tokens_per_active,
        "rows": rows,
        "notes": [
            "EP8 direct serving includes attention, dense/control, EP, compose, and output head.",
            "TP4/TP8 TurboMind workbenches are expert-only synthetic MXFP4 timing.",
            "Do not compare EP8 tok/s and TP expert-only ms as an apples-to-apples serving result.",
        ],
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    keys = [
        "kind",
        "returncode",
        "tokens_per_active",
        "routes",
        "correctness",
        "bad",
        "nan",
        "full_ms",
        "tp_label",
        "tp_compute_ms",
        "tp_reduce_ms",
        "tp_total_ms",
        "compute_speedup",
        "total_speedup",
        "direct_generated_tok_s",
        "ep_ms",
        "compose_ms",
        "first_token",
    ]
    with (args.artifact_dir / "summary.tsv").open("w", encoding="utf-8") as out:
        out.write("\t".join(keys) + "\n")
        for row in rows:
            out.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")


def parse_tiers(text):
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise argparse.ArgumentTypeError("tokens-per-active tiers must be positive")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("at least one tier is required")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=pathlib.Path, default=pathlib.Path("/workspace/ds4-sprint181"))
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path("python3"))
    parser.add_argument("--pack-dir", type=pathlib.Path, default=pathlib.Path("/workspace/packs/ds4-appliance-full-tm-gated-s181"))
    parser.add_argument(
        "--contract",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv"),
    )
    parser.add_argument(
        "--turbomind-lib",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so"),
    )
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--tokens-per-active", type=parse_tiers, default=parse_tiers("16,32,64"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--skip-ep8", action="store_true")
    parser.add_argument("--skip-tp4", action="store_true")
    parser.add_argument("--skip-tp8", action="store_true")
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.skip_ep8:
        rows.append(run_ep8(args))
    if not args.skip_tp4:
        for tier in args.tokens_per_active:
            rows.append(run_tp4_case(args, tier))
    if not args.skip_tp8:
        for tier in args.tokens_per_active:
            rows.append(run_tp8_case(args, tier))
    write_outputs(args, rows)
    print(json.dumps({"artifact_dir": str(args.artifact_dir), "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
