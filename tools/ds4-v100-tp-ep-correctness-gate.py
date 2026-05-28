#!/usr/bin/env python3
"""Run a repeatable deterministic correctness gate for DS4 V100 TP/EP serving.

The gate launches two same-shape selected-token HTTP profiles, compares the
response artifacts, and checks the serving invariants that must hold before
performance work is trusted:

- all requests return HTTP 200
- generated token sequences are present and have the expected length
- VRAM admission has no failures
- NCCL graph dump has zero SYS edges when present
- optional direct peer-copy accounting has zero SYS ops when requested
- control/candidate responses match deterministically
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


COMMON_PROFILE_FLAGS = [
    "--tool", "none",
    "--http-endpoint", "selected-token",
    "--startup-warmup", "auto",
    "--kill-stale-server",
    "--hc-current-nccl-allgather",
    "--hc-current-stream-sync",
    "--post-attention-ffn-input",
    "--defer-nccl-init",
    "--model-router-routes",
    "--gpu-route-plan",
    "--compact-moe-decode",
    "--parallel-expert-load",
    "--routed-ffn-norm-input",
    "--routed-ffn-rank-major-input",
    "--model-router-rank-major-logits",
    "--post-attention-fixed-capacity-route-plan",
    "--lazy-output-head",
    "--vram-report",
]


def default_global_lock_file() -> pathlib.Path:
    localpool = pathlib.Path("/localpool/ds4/workspace")
    if localpool.exists():
        return localpool / "ds4-tp-ep-http-ab.lock"
    return pathlib.Path("/tmp/ds4-tp-ep-http-ab.lock")


def acquire_global_lock(path: pathlib.Path, wait: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = path.open("a+", encoding="utf-8")
    flags = fcntl.LOCK_EX
    if not wait:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(lock_fh.fileno(), flags)
    except BlockingIOError as exc:
        lock_fh.seek(0)
        owner = lock_fh.read().strip()
        detail = f"; current owner: {owner}" if owner else ""
        lock_fh.close()
        raise RuntimeError(f"could not acquire global TP/EP lock {path}{detail}") from exc
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid={os.getpid()} argv={' '.join(sys.argv)}\n")
    lock_fh.flush()
    return lock_fh


def run(cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path,
        check: bool = False) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def find_case_dir(root: pathlib.Path) -> pathlib.Path:
    matches = sorted(root.rglob("summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one summary.json under {root}, found {len(matches)}")
    return matches[0].parent


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value


def add_check(checks: list[dict[str, Any]], name: str, ok: bool,
              detail: str, actual: Any = None, expected: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "ok": bool(ok), "detail": detail}
    if actual is not None:
        item["actual"] = actual
    if expected is not None:
        item["expected"] = expected
    checks.append(item)


def profile_cmd(args: argparse.Namespace, case: str, port: int,
                extra: list[str], requests: int | None = None) -> list[str]:
    artifact_dir = args.artifact_dir / case
    request_count = requests if requests is not None else args.requests
    max_requests = max(args.max_requests, request_count)
    cmd = [
        sys.executable,
        "tools/ds4-v100-tp-ep-profile.py",
        "--repo-dir", str(args.repo_dir),
        "--artifact-dir", str(artifact_dir),
        "--ctx", str(args.ctx),
        "--slots", str(args.slots),
        "--position", str(args.position),
        "--tokens", str(args.tokens),
        "--requests", str(request_count),
        "--max-requests", str(max_requests),
        "--request-concurrency", str(args.request_concurrency or request_count),
        "--port", str(port),
        "--readiness-seconds", str(args.readiness_seconds),
        "--request-timeout-seconds", str(args.request_timeout_seconds),
        "--gpu-sample-interval-ms", str(args.gpu_sample_interval_ms),
        "--gpu-sampler", args.gpu_sampler,
        "--dcgmi-fields", args.dcgmi_fields,
        "--tp-runtime-scratch-mib", str(args.tp_runtime_scratch_mib),
        "--nccl-min-free-mib", str(args.nccl_min_free_mib),
        "--vram-min-free-mib", str(args.vram_min_free_mib),
    ]
    if args.experimental_ctx_slot_cap is not None:
        cmd.extend(["--experimental-ctx-slot-cap", str(args.experimental_ctx_slot_cap)])
    cmd.extend(COMMON_PROFILE_FLAGS)
    if args.tp_peer_accounting:
        cmd.append("--tp-peer-accounting")
    if args.tp_peer_reject_sys:
        cmd.append("--tp-peer-reject-sys")
    cmd.extend(extra)
    return cmd


def check_summary(name: str, summary: dict[str, Any], args: argparse.Namespace,
                  expected_requests: int,
                  checks: list[dict[str, Any]]) -> None:
    prefix = f"{name}_"
    add_check(
        checks,
        prefix + "http_200",
        summary.get("http_200") == expected_requests,
        "all selected-token requests completed",
        summary.get("http_200"),
        expected_requests,
    )
    add_check(
        checks,
        prefix + "tokens",
        summary.get("tokens") == args.tokens,
        "summary token count matches gate shape",
        summary.get("tokens"),
        args.tokens,
    )
    add_check(
        checks,
        prefix + "vram_failures",
        int(summary.get("vram_failures", 0) or 0) <= args.max_vram_failures,
        "VRAM admission failures are within threshold",
        summary.get("vram_failures", 0),
        args.max_vram_failures,
    )
    min_free = summary.get("vram_min_free_mib")
    add_check(
        checks,
        prefix + "min_free_mib",
        isinstance(min_free, (int, float)) and float(min_free) >= args.vram_min_free_mib,
        "minimum free VRAM meets threshold",
        min_free,
        args.vram_min_free_mib,
    )
    sys_edges = summary.get("nccl_graph_sys_edge_count")
    add_check(
        checks,
        prefix + "nccl_no_sys",
        sys_edges in (0, None),
        "NCCL graph has no SYS edges when graph dump is present",
        sys_edges,
        0,
    )
    if args.tp_peer_accounting or args.tp_peer_reject_sys:
        sys_ops = summary.get("peer_copy_sys_ops")
        add_check(
            checks,
            prefix + "peer_copy_no_sys",
            sys_ops == 0,
            "direct peer-copy accounting reports zero SYS ops",
            sys_ops,
            0,
        )


def response_json(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.split("\n\nHTTP_STATUS:", 1)[0].strip()
    value = json.loads(first)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: response root is not an object")
    return value


def response_file(case_dir: pathlib.Path, index: int) -> pathlib.Path:
    return case_dir / f"response-{index:02d}.txt"


def run_self_mode(args: argparse.Namespace) -> int:
    requests = args.requests * 2
    self_cmd = profile_cmd(args, "self", args.port_base, args.candidate_extra_arg, requests=requests)
    (args.artifact_dir / "self-command.txt").write_text(
        " ".join(self_cmd) + "\n", encoding="utf-8"
    )
    proc = run(self_cmd, args.repo_dir, args.artifact_dir / "self.log")
    checks: list[dict[str, Any]] = []
    add_check(checks, "self_profile_exit", proc.returncode == 0,
              "self profile exits cleanly", proc.returncode, 0)
    if proc.returncode != 0:
        passed = write_summary(args, checks)
        print(json.dumps({
            "passed": passed,
            "artifact_dir": str(args.artifact_dir),
            "self_returncode": proc.returncode,
        }, sort_keys=True))
        return 0 if passed else 1

    self_case = find_case_dir(args.artifact_dir / "self")
    self_summary = load_json(self_case / "summary.json")
    check_summary("self", self_summary, args, requests, checks)

    matched = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    fields = ["generated_token_sequence", "selected_token", "checksum", "decode_step_checksums"]
    for i in range(args.requests):
        left_path = response_file(self_case, i)
        right_path = response_file(self_case, i + args.requests)
        if not left_path.exists() or not right_path.exists():
            failed += 1
            failures.append({
                "pair": i,
                "reason": "missing_response",
                "left": str(left_path),
                "right": str(right_path),
            })
            continue
        left = response_json(left_path)
        right = response_json(right_path)
        mismatches = [
            field for field in fields
            if left.get(field) != right.get(field)
        ]
        if mismatches:
            failed += 1
            failures.append({
                "pair": i,
                "reason": "field_mismatch",
                "fields": mismatches,
                "left_token": left.get("selected_token"),
                "right_token": right.get("selected_token"),
            })
        else:
            matched += 1

    parity_json = {
        "mode": "self",
        "paired_count": args.requests,
        "matched_pairs": matched,
        "failed_pairs": failed,
        "match": failed == 0 and matched == args.requests,
        "failures": failures[:16],
    }
    (args.artifact_dir / "self-parity.json").write_text(
        json.dumps(parity_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    add_check(checks, "self_response_parity", bool(parity_json["match"]),
              "first half and second half selected-token responses match",
              parity_json["matched_pairs"], args.requests)

    passed = write_summary(
        args,
        checks,
        control_summary=self_summary,
        control_case=self_case,
        parity_json=parity_json,
    )
    print(json.dumps({
        "passed": passed,
        "artifact_dir": str(args.artifact_dir),
        "self_case_dir": str(self_case),
        "matched_pairs": matched,
        "failed_pairs": failed,
    }, sort_keys=True))
    return 0 if passed else 1


def write_summary(args: argparse.Namespace, checks: list[dict[str, Any]],
                  control_summary: dict[str, Any] | None = None,
                  candidate_summary: dict[str, Any] | None = None,
                  control_case: pathlib.Path | None = None,
                  candidate_case: pathlib.Path | None = None,
                  parity_json: dict[str, Any] | None = None) -> bool:
    passed = all(bool(item["ok"]) for item in checks)
    summary = {
        "schema": "ds4_tp_ep_correctness_gate.v1",
        "passed": passed,
        "mode": args.mode,
        "artifact_dir": str(args.artifact_dir),
        "control_case_dir": str(control_case) if control_case else None,
        "candidate_case_dir": str(candidate_case) if candidate_case else None,
        "shape": {
            "ctx": args.ctx,
            "slots": args.slots,
            "position": args.position,
            "requests": args.requests,
            "tokens": args.tokens,
        },
        "control": control_summary or {},
        "candidate": candidate_summary or {},
        "parity": parity_json or {},
        "checks": checks,
    }
    (args.artifact_dir / "correctness-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DS4 V100 TP/EP Correctness Gate",
        "",
        f"Passed: `{passed}`",
        "",
        "| Check | OK | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        lines.append(f"| `{item['name']}` | `{item['ok']}` | {item['detail']} |")
    lines.append("")
    if control_case:
        lines.append(f"Control case: `{control_case}`")
    if candidate_case:
        lines.append(f"Candidate case: `{candidate_case}`")
    lines.append("")
    (args.artifact_dir / "correctness-summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--experimental-ctx-slot-cap", type=int, default=32)
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--request-concurrency", type=int, default=0)
    parser.add_argument("--mode", choices=["two-run", "self"], default="two-run",
                        help="two-run starts isolated control/candidate servers; self uses one server and compares first half vs second half")
    parser.add_argument("--port-base", type=int, default=19100)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=0)
    parser.add_argument("--gpu-sampler", choices=["dmon", "dcgmi", "query"], default="dmon")
    parser.add_argument("--dcgmi-fields", default="203,252,155,150,1002,1003,1005,1009,1010,1001,1011,1012")
    parser.add_argument("--tp-runtime-scratch-mib", type=int, default=1024)
    parser.add_argument("--nccl-min-free-mib", type=int, default=64)
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--max-vram-failures", type=int, default=0)
    parser.add_argument("--tp-peer-accounting", action="store_true")
    parser.add_argument("--tp-peer-reject-sys", action="store_true")
    parser.add_argument("--control-extra-arg", action="append", default=[])
    parser.add_argument("--candidate-extra-arg", action="append", default=[])
    parser.add_argument("--global-lock-file", type=pathlib.Path)
    parser.add_argument("--no-global-lock", action="store_true")
    parser.add_argument("--wait-global-lock", action="store_true")
    args = parser.parse_args()

    args.repo_dir = args.repo_dir.resolve()
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = None
    try:
        if not args.no_global_lock:
            lock_fh = acquire_global_lock(
                args.global_lock_file or default_global_lock_file(),
                args.wait_global_lock,
            )
        if args.mode == "self":
            return run_self_mode(args)

        control_cmd = profile_cmd(args, "control", args.port_base, args.control_extra_arg)
        candidate_cmd = profile_cmd(args, "candidate", args.port_base + 1, args.candidate_extra_arg)
        (args.artifact_dir / "control-command.txt").write_text(
            " ".join(control_cmd) + "\n", encoding="utf-8"
        )
        (args.artifact_dir / "candidate-command.txt").write_text(
            " ".join(candidate_cmd) + "\n", encoding="utf-8"
        )

        control = run(control_cmd, args.repo_dir, args.artifact_dir / "control.log")
        candidate = run(candidate_cmd, args.repo_dir, args.artifact_dir / "candidate.log")
        checks: list[dict[str, Any]] = []
        add_check(checks, "control_profile_exit", control.returncode == 0,
                  "control profile exits cleanly", control.returncode, 0)
        add_check(checks, "candidate_profile_exit", candidate.returncode == 0,
                  "candidate profile exits cleanly", candidate.returncode, 0)
        if control.returncode != 0 or candidate.returncode != 0:
            passed = write_summary(args, checks)
            print(json.dumps({
                "passed": passed,
                "artifact_dir": str(args.artifact_dir),
                "control_returncode": control.returncode,
                "candidate_returncode": candidate.returncode,
            }, sort_keys=True))
            return 0 if passed else 1

        control_case = find_case_dir(args.artifact_dir / "control")
        candidate_case = find_case_dir(args.artifact_dir / "candidate")
        control_summary = load_json(control_case / "summary.json")
        candidate_summary = load_json(candidate_case / "summary.json")
        check_summary("control", control_summary, args, args.requests, checks)
        check_summary("candidate", candidate_summary, args, args.requests, checks)

        parity_cmd = [
            sys.executable,
            "tools/ds4-v100-http-response-parity.py",
            "--control-dir", str(control_case),
            "--candidate-dir", str(candidate_case),
            "--out", str(args.artifact_dir / "response-parity.json"),
            "--ignore-text",
        ]
        parity = run(parity_cmd, args.repo_dir, args.artifact_dir / "response-parity.log")
        parity_json = load_json(args.artifact_dir / "response-parity.json")
        add_check(checks, "response_parity", parity.returncode == 0 and bool(parity_json.get("match")),
                  "selected-token response artifacts match", parity_json.get("match"), True)
        add_check(checks, "response_pairs", parity_json.get("paired_count") == args.requests,
                  "parity compared expected request count", parity_json.get("paired_count"), args.requests)

        passed = write_summary(
            args,
            checks,
            control_summary=control_summary,
            candidate_summary=candidate_summary,
            control_case=control_case,
            candidate_case=candidate_case,
            parity_json={
                "match": parity_json.get("match"),
                "paired_count": parity_json.get("paired_count"),
                "matched_pairs": parity_json.get("matched_pairs"),
                "failed_pairs": parity_json.get("failed_pairs"),
            },
        )
        print(json.dumps({
            "passed": passed,
            "artifact_dir": str(args.artifact_dir),
            "control_case_dir": str(control_case),
            "candidate_case_dir": str(candidate_case),
            "matched_pairs": parity_json.get("matched_pairs"),
            "failed_pairs": parity_json.get("failed_pairs"),
        }, sort_keys=True))
        return 0 if passed else 1
    finally:
        if lock_fh is not None:
            try:
                lock_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
