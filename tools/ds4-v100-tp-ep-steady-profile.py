#!/usr/bin/env python3
"""Run a long TP/EP HTTP decode and collect steady-state V100 counter windows.

This is intentionally different from the normal profile harness:

- `nvidia-smi dmon` runs for the full process as cheap NVML health telemetry.
- DCGMI profiling counters are collected only after the request window starts.
- Conflicting V100 A-subgroups are measured in separate windows so the samples
  are not blurred by DCGM multiplexing.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any


DS4_PROCESS_PATTERNS = (
    "ds4-v100-tp-ep-appliance",
    "ds4-v100-tp-ep-profile.py",
    "ds4-v100-tp-ep-nccl-http-ab.py",
    "dcgmi dmon",
    "nvidia-smi dmon",
    "nsys",
    "ncu",
)

MONITOR_ONLY_PATTERNS = (
    "ps -eo",
    "nvidia-smi --query-gpu",
    "tail -n",
    "find /localpool",
    "patterns=(",
    "active_ds4_processes",
)

DCGMI_WINDOWS = [
    ("a1_sm_occupancy", "1002,1003"),
    ("a2_tensor", "1004"),
    ("a3_fp64", "1006"),
    ("a4_fp32", "1007"),
    ("a5_fp16", "1008"),
]

HEALTH_FIELDS = "203,252,155,150"
COMMON_PROFILE_FIELDS = "1005,1009,1010,1001,1011,1012"


def log(path: pathlib.Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(line + "\n")


def default_global_lock_file() -> pathlib.Path:
    localpool = pathlib.Path("/localpool/ds4/workspace")
    if localpool.exists():
        return localpool / "ds4-tp-ep-http-ab.lock"
    return pathlib.Path("/tmp/ds4-tp-ep-http-ab.lock")


def acquire_global_lock(path: pathlib.Path, log_path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = path.open("a+", encoding="utf-8")
    log(log_path, f"waiting_for_global_lock path={path}")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid={os.getpid()} argv={' '.join(sys.argv)}\n")
    lock_fh.flush()
    log(log_path, f"global_lock_acquired path={path}")
    return lock_fh


def terminate_process(proc: subprocess.Popen[Any] | None, timeout_s: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def active_ds4_processes() -> list[str]:
    try:
        text = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    except Exception:
        return []
    self_pid = os.getpid()
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if "ds4-v100-tp-ep-steady-profile.py" in args:
            continue
        if any(pattern in args for pattern in MONITOR_ONLY_PATTERNS):
            continue
        if any(pattern in args for pattern in DS4_PROCESS_PATTERNS):
            out.append(line)
    return out


def wait_for_idle(timeout_s: float, poll_s: float, log_path: pathlib.Path) -> None:
    deadline = time.time() + timeout_s
    last_report = 0.0
    while True:
        active = active_ds4_processes()
        if not active:
            return
        now = time.time()
        if now >= deadline:
            detail = "\n".join(active[:8])
            raise TimeoutError(f"timed out waiting for idle DS4 node; active:\n{detail}")
        if now - last_report >= 30.0:
            log(log_path, f"waiting_for_idle active_count={len(active)} first={active[0]}")
            last_report = now
        time.sleep(poll_s)


def load_events(lifecycle_path: pathlib.Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if not lifecycle_path.exists():
        return events
    with lifecycle_path.open("r", encoding="utf-8", errors="replace") as src:
        next(src, None)
        for line in src:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) < 3:
                continue
            try:
                events[parts[0]] = {
                    "unix_s": float(parts[1]),
                    "elapsed_s": float(parts[2]),
                    "detail": parts[3] if len(parts) > 3 else "",
                }
            except ValueError:
                continue
    return events


def find_case_dir(profile_dir: pathlib.Path) -> pathlib.Path | None:
    if not profile_dir.exists():
        return None
    matches = [path for path in profile_dir.iterdir() if path.is_dir()]
    if not matches:
        return None
    return sorted(matches)[0]


def wait_for_event(
    profile_dir: pathlib.Path,
    event: str,
    timeout_s: float,
    poll_s: float,
    harness: subprocess.Popen[Any],
    log_path: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, dict[str, Any]]]:
    deadline = time.time() + timeout_s
    last_report = 0.0
    while time.time() < deadline:
        case_dir = find_case_dir(profile_dir)
        if case_dir is not None:
            events = load_events(case_dir / "lifecycle.csv")
            if event in events:
                return case_dir, events
        if harness.poll() is not None:
            raise RuntimeError(f"profile harness exited before lifecycle event {event}")
        now = time.time()
        if now - last_report >= 30.0:
            log(log_path, f"waiting_for_event event={event}")
            last_report = now
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for lifecycle event {event}")


def dcgmi_command(fields: str, interval_ms: int, samples: int, gpus: str) -> list[str]:
    return [
        "dcgmi",
        "dmon",
        "-i",
        gpus,
        "-e",
        fields,
        "-d",
        str(interval_ms),
        "-c",
        str(samples),
    ]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return float(value.split()[0])
    except ValueError:
        return None


def summarize_dcgmi(path: pathlib.Path) -> dict[str, Any]:
    aliases = {
        "GPUTL": "gpu_util",
        "FBUSD": "fb_used_mib",
        "FBFRE": "fb_free_mib",
        "POWER": "power_w",
        "TMPTR": "gpu_temp_c",
        "SMACT": "sm_active",
        "SMOCC": "sm_occupancy",
        "TENSO": "tensor_active",
        "FP64A": "fp64_active",
        "FP32A": "fp32_active",
        "FP16A": "fp16_active",
        "DRAMA": "dram_active",
        "PCITX": "pcie_tx_bytes",
        "PCIRX": "pcie_rx_bytes",
        "GRACT": "gr_engine_active",
        "NVLTX": "nvlink_tx_bytes",
        "NVLRX": "nvlink_rx_bytes",
    }
    headers: list[str] | None = None
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as src:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#Entity":
                headers = [aliases.get(item, item.lower()) for item in parts[1:]]
                continue
            if headers is None or parts[0] in ("ID", "Error"):
                continue
            if len(parts) < 2 + len(headers):
                continue
            parsed: dict[str, float] = {}
            for key, value in zip(headers, parts[2:]):
                number = parse_float(value)
                if number is not None:
                    parsed[key] = number
            if parsed:
                rows.append(parsed)
    out: dict[str, Any] = {"sample_rows": len(rows)}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            out[f"{key}_avg"] = sum(values) / len(values)
            out[f"{key}_max"] = max(values)
    return out


def summarize_nvidia_dmon(path: pathlib.Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as src:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            try:
                parts = [part.strip() for part in next(csv.reader([line]))]
            except csv.Error:
                continue
            if parts and parts[0].startswith("#"):
                parts[0] = parts[0].lstrip("#").strip()
                header = [part.lower().replace(" ", "_") for part in parts]
                continue
            if header is None or len(parts) < len(header):
                continue
            rec = dict(zip(header, parts))
            row = {}
            for src_key, dst_key in (
                ("sm", "sm_util"),
                ("mem", "mem_util"),
                ("fb", "fb_used_mib"),
                ("pwr", "power_w"),
                ("gtemp", "gpu_temp_c"),
                ("mclk", "mem_clock_mhz"),
                ("pclk", "sm_clock_mhz"),
                ("rxpci", "pcie_rx_kib_s"),
                ("txpci", "pcie_tx_kib_s"),
            ):
                number = parse_float(rec.get(src_key))
                if number is not None:
                    row[dst_key] = number
            if row:
                rows.append(row)
    out: dict[str, Any] = {"sample_rows": len(rows)}
    for key in sorted({key for row in rows for key in row}):
        values = [row[key] for row in rows if key in row]
        if values:
            out[f"{key}_avg"] = sum(values) / len(values)
            out[f"{key}_max"] = max(values)
    return out


def profile_command(args: argparse.Namespace, artifact_dir: pathlib.Path) -> list[str]:
    cmd = [
        sys.executable,
        "tools/ds4-v100-tp-ep-profile.py",
        "--run-mode",
        "http",
        "--tool",
        "none",
        "--artifact-dir",
        str(artifact_dir),
        "--ctx",
        str(args.ctx),
        "--slots",
        str(args.slots),
        "--position",
        str(args.position),
        "--tokens",
        str(args.tokens),
        "--requests",
        str(args.requests),
        "--max-requests",
        str(args.max_requests),
        "--port",
        str(args.port),
        "--readiness-seconds",
        str(args.readiness_seconds),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--request-concurrency",
        str(args.request_concurrency or args.slots),
        "--gpu-sample-interval-ms",
        "0",
        "--gpu-sampler",
        "query",
        "--kill-stale-server",
        "--model-router-routes",
        "--compact-moe-decode",
        "--parallel-expert-load",
        "--lazy-output-head",
        "--vram-report",
        "--vram-min-free-mib",
        str(args.vram_min_free_mib),
        "--nccl-min-free-mib",
        str(args.nccl_min_free_mib),
        "--tp-runtime-scratch-mib",
        str(args.tp_runtime_scratch_mib),
        "--http-endpoint",
        "chat",
        "--defer-nccl-init",
        "--routed-ffn-rank-major-input",
        "--model-router-rank-major-logits",
        "--hc-current-stream-sync",
        "--hc-current-nccl-allgather",
    ]
    if args.extra_profile_arg:
        for item in args.extra_profile_arg:
            cmd.extend(item.split())
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--max-requests", type=int, default=256)
    parser.add_argument("--position", type=int, default=262000)
    parser.add_argument("--port", type=int, default=18840)
    parser.add_argument("--readiness-seconds", type=int, default=900)
    parser.add_argument("--request-timeout-seconds", type=int, default=3600)
    parser.add_argument("--request-concurrency", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--dcgmi-window-seconds", type=int, default=60)
    parser.add_argument("--dcgmi-sample-ms", type=int, default=500)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--nccl-min-free-mib", type=int, default=1536)
    parser.add_argument("--tp-runtime-scratch-mib", type=int, default=1280)
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--idle-timeout-seconds", type=int, default=7200)
    parser.add_argument("--global-lock-file", type=pathlib.Path)
    parser.add_argument("--no-global-lock", action="store_true")
    parser.add_argument("--extra-profile-arg", action="append")
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.artifact_dir / "steady-profile.log"
    log_path.write_text("", encoding="utf-8")
    lock_fh = None

    try:
        if not args.no_global_lock:
            lock_fh = acquire_global_lock(args.global_lock_file or default_global_lock_file(), log_path)

        if args.wait_for_idle:
            wait_for_idle(args.idle_timeout_seconds, 5.0, log_path)
        else:
            active = active_ds4_processes()
            if active:
                raise RuntimeError("DS4 processes already active; use --wait-for-idle or stop them first")
    except Exception:
        if lock_fh is not None:
            lock_fh.close()
        raise

    profile_dir = args.artifact_dir / "profile"
    nvidia_dmon_path = args.artifact_dir / "nvidia-smi-dmon.csv"
    profile_stdout = (args.artifact_dir / "profile.stdout").open("w", encoding="utf-8")
    profile_stderr = (args.artifact_dir / "profile.stderr").open("w", encoding="utf-8")
    dmon_stderr = (args.artifact_dir / "nvidia-smi-dmon.stderr").open("w", encoding="utf-8")
    dmon_proc: subprocess.Popen[Any] | None = None
    profile_proc: subprocess.Popen[Any] | None = None

    try:
        dmon_proc = subprocess.Popen(
            [
                "nvidia-smi",
                "dmon",
                "-s",
                "pucmt",
                "-d",
                "1",
                "-o",
                "DT",
                "--format",
                "csv,nounit",
                "-f",
                str(nvidia_dmon_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=dmon_stderr,
        )
        log(log_path, f"nvidia_smi_dmon_pid={dmon_proc.pid}")

        cmd = profile_command(args, profile_dir)
        (args.artifact_dir / "profile-command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        profile_proc = subprocess.Popen(cmd, stdout=profile_stdout, stderr=profile_stderr)
        log(log_path, f"profile_pid={profile_proc.pid}")

        case_dir, events = wait_for_event(
            profile_dir,
            "requests_start",
            args.readiness_seconds + 120,
            1.0,
            profile_proc,
            log_path,
        )
        (args.artifact_dir / "case-dir.txt").write_text(str(case_dir) + "\n", encoding="utf-8")
        log(log_path, f"requests_start case_dir={case_dir}")
        if args.warmup_seconds > 0:
            log(log_path, f"warmup_seconds={args.warmup_seconds}")
            time.sleep(args.warmup_seconds)

        dcgmi_results: dict[str, Any] = {}
        samples = max(1, int((args.dcgmi_window_seconds * 1000) / args.dcgmi_sample_ms))
        for name, a_fields in DCGMI_WINDOWS:
            if profile_proc.poll() is not None:
                log(log_path, f"profile_ended_before_window={name}")
                break
            fields = ",".join([HEALTH_FIELDS, a_fields, COMMON_PROFILE_FIELDS])
            raw_path = args.artifact_dir / f"dcgmi-{name}.txt"
            err_path = args.artifact_dir / f"dcgmi-{name}.stderr"
            log(log_path, f"dcgmi_start name={name} fields={fields} samples={samples}")
            with raw_path.open("w", encoding="utf-8") as raw, err_path.open("w", encoding="utf-8") as err:
                proc = subprocess.run(
                    dcgmi_command(fields, args.dcgmi_sample_ms, samples, args.gpus),
                    stdout=raw,
                    stderr=err,
                    text=True,
                )
            dcgmi_results[name] = {
                "returncode": proc.returncode,
                "fields": fields,
                **summarize_dcgmi(raw_path),
            }
            log(log_path, f"dcgmi_done name={name} rc={proc.returncode}")

        log(log_path, "waiting_for_profile_completion")
        profile_rc = profile_proc.wait()
        log(log_path, f"profile_returncode={profile_rc}")
        terminate_process(dmon_proc)

        final_events = load_events(case_dir / "lifecycle.csv")
        summary_path = case_dir / "summary.json"
        profile_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        out = {
            "schema": "ds4_v100_tp_ep_steady_profile.v1",
            "artifact_dir": str(args.artifact_dir),
            "case_dir": str(case_dir),
            "profile_returncode": profile_rc,
            "lifecycle": final_events,
            "profile_summary": profile_summary,
            "nvidia_smi_dmon": summarize_nvidia_dmon(nvidia_dmon_path) if nvidia_dmon_path.exists() else {},
            "dcgmi_windows": dcgmi_results,
        }
        (args.artifact_dir / "steady-summary.json").write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if profile_rc != 0:
            return profile_rc
        if "responses_complete" not in final_events:
            return 2
        return 0
    finally:
        terminate_process(dmon_proc)
        if profile_proc is not None and profile_proc.poll() is None:
            terminate_process(profile_proc, timeout_s=15.0)
        profile_stdout.close()
        profile_stderr.close()
        dmon_stderr.close()
        if lock_fh is not None:
            lock_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
