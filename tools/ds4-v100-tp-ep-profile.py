#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pathlib
import re
import signal
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


DCGMI_HEALTH_FIELDS = "203,252,155,150"
DCGMI_PROFILE_FIELDS = "1002,1003,1005,1009,1010,1001,1011,1012"
DCGMI_TENSOR_FIELDS = "1004"
DCGMI_DEFAULT_FIELDS = ",".join([DCGMI_HEALTH_FIELDS, DCGMI_PROFILE_FIELDS])

NCCL_DEFAULT_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
NCCL_NO_SYS_VISIBLE_DEVICES = "0,3,2,1,5,7,6,4"
NCCL_NO_SYS_RING = "0 3 2 1 5 7 6 4"
NCCL_ENV_KEYS = [
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_RINGS",
    "NCCL_P2P_LEVEL",
    "NCCL_SHM_DISABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_TOPO_DUMP_FILE",
    "NCCL_GRAPH_DUMP_FILE",
]

V100_NVLINK_COUNTS = {
    frozenset((0, 1)): 1,
    frozenset((0, 2)): 1,
    frozenset((0, 3)): 2,
    frozenset((0, 4)): 2,
    frozenset((1, 2)): 2,
    frozenset((1, 3)): 1,
    frozenset((1, 5)): 2,
    frozenset((2, 3)): 2,
    frozenset((2, 6)): 1,
    frozenset((3, 7)): 1,
    frozenset((4, 5)): 1,
    frozenset((4, 6)): 1,
    frozenset((4, 7)): 2,
    frozenset((5, 6)): 2,
    frozenset((5, 7)): 1,
    frozenset((6, 7)): 2,
}


DCGMI_FIELD_ALIASES = {
    "GPUTL": "gpu_util",
    "MCUTL": "mem_copy_util",
    "FBUSD": "fb_used_mib",
    "FBFRE": "fb_free_mib",
    "POWER": "power_w",
    "TMPTR": "gpu_temp_c",
    "SMCLK": "sm_clock_mhz",
    "MMCLK": "mem_clock_mhz",
    "GRACT": "gr_engine_active",
    "SMACT": "sm_active",
    "SMOCC": "sm_occupancy",
    "TENSO": "tensor_active",
    "DRAMA": "dram_active",
    "FP64A": "fp64_active",
    "FP32A": "fp32_active",
    "FP16A": "fp16_active",
    "PCITX": "pcie_tx_bytes",
    "PCIRX": "pcie_rx_bytes",
    "NVLTX": "nvlink_tx_bytes",
    "NVLRX": "nvlink_rx_bytes",
    "TIMMA": "tensor_imma_active",
    "THMMA": "tensor_hmma_active",
    "TDFMA": "tensor_dfma_active",
    "INTAC": "integer_active",
}


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def parse_int(value):
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def profiler_log_pattern(case_dir, stem):
    return str(case_dir / f"{stem}.%p.csv")


def http_get(base, path, timeout=10):
    req = urllib.request.Request(base + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_post(base, path, payload, timeout=1200):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def port_is_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def ds4_server_pids_for_port(port):
    try:
        text = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
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
        if pid == os.getpid():
            continue
        if "ds4-v100-tp-ep-full-layer-smoke" not in args or "--serve-http" not in args:
            continue
        try:
            argv = shlex.split(args)
        except ValueError:
            argv = args.split()
        matched = False
        for idx, value in enumerate(argv):
            if value == "--port" and idx + 1 < len(argv) and argv[idx + 1] == str(port):
                matched = True
                break
            if value.startswith("--port=") and value.split("=", 1)[1] == str(port):
                matched = True
                break
        if matched:
            out.append(pid)
    return sorted(set(out))


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_pids(pids, timeout_s=10):
    pids = sorted(set(int(pid) for pid in pids if int(pid) > 1))
    if not pids:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            alive = [pid for pid in pids if pid_alive(pid)]
            if not alive:
                return
            time.sleep(0.2)


def cleanup_managed_server_port(port, case_dir=None, reason="cleanup"):
    pids = ds4_server_pids_for_port(port)
    if case_dir is not None and pids:
        with open(case_dir / "harness-cleanup.log", "a", encoding="utf-8") as out:
            out.write(f"{reason}: terminating port={port} pids={','.join(map(str, pids))}\n")
    terminate_pids(pids)
    if port_is_open(port):
        pids = ds4_server_pids_for_port(port)
        raise RuntimeError(f"port {port} still open after {reason}; ds4_pids={pids}")


def load_prompt_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as src:
        for lineno, raw in enumerate(src, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: prompt record must be an object")
            prompt_id = record.get("id", f"prompt-{len(records):03d}")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{path}:{lineno}: id must be a non-empty string")
            messages = record.get("messages")
            if messages is None and "prompt" in record:
                prompt = record["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(f"{path}:{lineno}: prompt must be a string")
                messages = [{"role": "user", "content": prompt}]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{lineno}: messages must be a non-empty list")
            normalized = []
            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValueError(f"{path}:{lineno}: each message must be an object")
                role = msg.get("role")
                content = msg.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    raise ValueError(f"{path}:{lineno}: messages need string role/content")
                normalized.append({"role": role, "content": content})
            records.append({"id": prompt_id, "messages": normalized})
    if not records:
        raise ValueError(f"{path}: no prompt records found")
    return records


def prompt_digest(records):
    h = hashlib.sha256()
    for record in records:
        h.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


class GpuSampler:
    def __init__(self, path, interval_s, mode="dmon", dcgmi_fields=DCGMI_DEFAULT_FIELDS):
        self.path = pathlib.Path(path)
        self.interval_s = interval_s
        self.mode = mode
        self.dcgmi_fields = dcgmi_fields
        self.stop = threading.Event()
        self.thread = None
        self.proc = None
        self.err_fh = None
        self.out_fh = None

    def __enter__(self):
        if self.interval_s <= 0:
            return self
        if self.mode == "dcgmi" and shutil.which("dcgmi"):
            self.thread = threading.Thread(target=self._run_dcgmi, daemon=True)
            self.thread.start()
            return self
        if self.mode == "dmon" and shutil.which("nvidia-smi"):
            self._start_dmon()
            return self
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            self.stop.set()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3.0)
            if self.thread is not None:
                self.thread.join(timeout=2.0)
            if self.err_fh is not None and not self.err_fh.closed:
                self.err_fh.close()
            if self.out_fh is not None and not self.out_fh.closed:
                self.out_fh.close()
            return False
        if self.thread is None:
            return False
        self.stop.set()
        self.thread.join(timeout=2.0)
        return False

    def _start_dmon(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        delay_s = max(1, int(round(self.interval_s)))
        self.err_fh = open(self.path.with_suffix(".err"), "wb")
        gpm_metrics = os.environ.get("DS4_V100_DMON_GPM_METRICS", "").strip()
        command = [
            "nvidia-smi",
            "dmon",
            "-s",
            "pucmt",
        ]
        if gpm_metrics and gpm_metrics.lower() not in ("0", "false", "off", "none"):
            command.extend(["--gpm-metrics", gpm_metrics, "--gpm-options", "d"])
        command.extend(
            [
                "-d",
                str(delay_s),
                "-o",
                "DT",
                "--format",
                "csv,nounit",
                "-f",
                str(self.path),
            ]
        )
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=self.err_fh,
        )

    def _run_dcgmi(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = os.environ.get("DS4_V100_DCGMI_FIELDS", self.dcgmi_fields).strip()
        if not fields:
            fields = DCGMI_DEFAULT_FIELDS
        delay_ms = max(1, int(round(self.interval_s * 1000.0)))
        command = [
            "dcgmi",
            "dmon",
            "-e",
            fields,
            "-d",
            str(delay_ms),
            "-c",
            "0",
        ]
        entity_ids = os.environ.get("DS4_V100_DCGMI_ENTITY_IDS", "").strip()
        if entity_ids:
            command.extend(["-i", entity_ids])
        host = os.environ.get("DS4_V100_DCGMI_HOST", "").strip()
        if host:
            command.extend(["--host", host])
        with open(self.path, "w", encoding="utf-8", newline="") as out:
            writer = None
            headers = None
            self.err_fh = open(self.path.with_suffix(".err"), "wb")
            try:
                self.proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=self.err_fh,
                    text=True,
                    bufsize=1,
                )
                for raw in self.proc.stdout:
                    if self.stop.is_set():
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts and parts[0] == "#Entity":
                        headers = [DCGMI_FIELD_ALIASES.get(part, part.lower()) for part in parts[1:]]
                        fieldnames = ["sample_unix_s", "entity_type", "gpu", *headers]
                        writer = csv.DictWriter(out, fieldnames=fieldnames)
                        writer.writeheader()
                        out.flush()
                        continue
                    if not writer or not headers or parts[0] in ("ID", "Error"):
                        continue
                    if len(parts) < 2 + len(headers):
                        continue
                    rec = {
                        "sample_unix_s": f"{time.time():.6f}",
                        "entity_type": parts[0],
                        "gpu": parts[1],
                    }
                    for key, value in zip(headers, parts[2:]):
                        rec[key] = value
                    writer.writerow(rec)
                    out.flush()
            except Exception as exc:
                out.write(f"# dcgmi_sampler_error {exc}\n")
                out.flush()
            finally:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=3.0)
                if self.err_fh is not None:
                    self.err_fh.close()

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as out:
            out.write("sample_unix_s,index,utilization.gpu,memory.used,memory.total\n")
            if not shutil.which("nvidia-smi"):
                out.write("sample_error,-1,0,0,0 # nvidia-smi not found\n")
                return
            while not self.stop.is_set():
                try:
                    text = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    now = time.time()
                    for line in text.splitlines():
                        if line.strip():
                            out.write(f"{now:.6f},{line}\n")
                    out.flush()
                except Exception as exc:
                    out.write(f"sample_error,-1,0,0,0 # {exc}\n")
                    out.flush()
                self.stop.wait(self.interval_s)


class LifecycleEvents:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = []
        with open(self.path, "w", encoding="utf-8") as out:
            out.write("event,unix_s,elapsed_s,detail\n")
        self.start_s = time.time()

    def mark(self, event, detail=""):
        now = time.time()
        elapsed = now - self.start_s
        safe_detail = str(detail).replace("\n", " ").replace(",", ";")
        with open(self.path, "a", encoding="utf-8") as out:
            out.write(f"{event},{now:.6f},{elapsed:.6f},{safe_detail}\n")
        self._events.append((event, now, elapsed, safe_detail))


def load_lifecycle_events(path):
    path = pathlib.Path(path)
    events = {}
    if not path.exists():
        return events
    with open(path, "r", encoding="utf-8", errors="replace") as src:
        next(src, None)
        for line in src:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) < 3:
                continue
            try:
                unix_s = float(parts[1])
                elapsed_s = float(parts[2])
            except ValueError:
                continue
            events[parts[0]] = {
                "unix_s": unix_s,
                "elapsed_s": elapsed_s,
                "detail": parts[3] if len(parts) > 3 else "",
            }
    return events


def summarize_gpu_rows(rows):
    if not rows:
        return {"sample_count": 0}
    utils = [row["util"] for row in rows]
    mem_used = [row["mem_used_mib"] for row in rows]
    by_gpu = {}
    for row in rows:
        by_gpu.setdefault(row["gpu"], []).append(row)
    out = {
        "sample_count": len(rows),
        "util_avg": sum(utils) / len(utils),
        "util_max": max(utils),
        "mem_used_max_mib": max(mem_used),
    }
    summary_metrics = [
        "mem_util",
        "power_w",
        "gpu_temp_c",
        "mem_temp_c",
        "mclk_mhz",
        "pclk_mhz",
        "bar1_mib",
        "pcie_rx",
        "pcie_tx",
        "nvlink_rx",
        "nvlink_tx",
        "gpu_util",
        "mem_copy_util",
        "fb_free_mib",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "gr_engine_active",
        "sm_active",
        "sm_occupancy",
        "tensor_active",
        "dram_active",
        "fp64_active",
        "fp32_active",
        "fp16_active",
        "pcie_tx_bytes",
        "pcie_rx_bytes",
        "nvlink_tx_bytes",
        "nvlink_rx_bytes",
        "integer_active",
    ]
    summary_metrics = sorted(
        set(summary_metrics)
        | {
            key
            for row in rows
            for key, value in row.items()
            if key not in ("sample_unix_s", "gpu", "util", "mem_used_mib", "mem_total_mib")
            and isinstance(value, (int, float))
        }
    )
    gpm_metrics = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("gpm_") and isinstance(value, (int, float))
        }
    )
    for metric in [*summary_metrics, *gpm_metrics]:
        values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
        if values:
            out[f"{metric}_avg"] = sum(values) / len(values)
            out[f"{metric}_max"] = max(values)
    per_gpu = {}
    for gpu, gpu_rows in sorted(by_gpu.items()):
        gpu_utils = [row["util"] for row in gpu_rows]
        gpu_mem = [row["mem_used_mib"] for row in gpu_rows]
        gpu_summary = {
            "samples": len(gpu_rows),
            "util_avg": sum(gpu_utils) / len(gpu_utils),
            "util_max": max(gpu_utils),
            "mem_used_max_mib": max(gpu_mem),
        }
        for metric in [*summary_metrics, *gpm_metrics]:
            values = [row[metric] for row in gpu_rows if isinstance(row.get(metric), (int, float))]
            if values:
                gpu_summary[f"{metric}_avg"] = sum(values) / len(values)
                gpu_summary[f"{metric}_max"] = max(values)
        per_gpu[str(gpu)] = gpu_summary
    out["per_gpu"] = per_gpu
    return out


def grouped_gpu_timeline(rows, events, window=5):
    by_time = {}
    for row in rows:
        sample_unix_s = row.get("sample_unix_s")
        if sample_unix_s is None:
            continue
        by_time.setdefault(sample_unix_s, []).append(row)
    if not by_time:
        return []
    process_start = events.get("process_start", {}).get("unix_s")
    request_start = events.get("requests_start", {}).get("unix_s")
    response_done = events.get("responses_complete", {}).get("unix_s")
    ready = events.get("server_ready", {}).get("unix_s")
    timeline = []
    for sample_unix_s, sample_rows in sorted(by_time.items()):
        aggregate = summarize_gpu_rows(sample_rows)
        util_values = [row["util"] for row in sample_rows]
        mem_values = [row["mem_used_mib"] for row in sample_rows]
        power_values = [row.get("power_w") for row in sample_rows if isinstance(row.get("power_w"), (int, float))]
        if request_start is not None and sample_unix_s >= request_start:
            phase = "request"
        elif ready is not None and sample_unix_s >= ready:
            phase = "ready_idle"
        else:
            phase = "startup"
        if response_done is not None and sample_unix_s > response_done:
            phase = "post_request"
        timeline.append(
            {
                "sample_unix_s": sample_unix_s,
                "elapsed_s": sample_unix_s - process_start if process_start is not None else None,
                "phase": phase,
                "gpu_count": len(sample_rows),
                "sm_util_avg": sum(util_values) / len(util_values),
                "sm_util_max": max(util_values),
                "mem_used_max_mib": max(mem_values),
                "mem_used_sum_mib": sum(mem_values),
                "power_w_avg": sum(power_values) / len(power_values) if power_values else None,
                "sample_count": aggregate["sample_count"],
            }
        )
    moving = []
    for item in timeline:
        moving.append(item["sm_util_avg"])
        if len(moving) > window:
            moving.pop(0)
        item["sm_util_avg_ma"] = sum(moving) / len(moving)
    return timeline


def write_gpu_timeline(path, rows, events):
    timeline = grouped_gpu_timeline(rows, events)
    if not timeline:
        return {}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_unix_s",
        "elapsed_s",
        "phase",
        "gpu_count",
        "sm_util_avg",
        "sm_util_avg_ma",
        "sm_util_max",
        "mem_used_max_mib",
        "mem_used_sum_mib",
        "power_w_avg",
    ]
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for item in timeline:
            writer.writerow({field: item.get(field) for field in fields})
    peak = max(timeline, key=lambda item: item["sm_util_avg_ma"])
    request_rows = [item for item in timeline if item["phase"] == "request"]
    steady = request_rows[len(request_rows) // 3:] if len(request_rows) >= 3 else request_rows
    summary = {
        "gpu_timeline_sample_count": len(timeline),
        "gpu_timeline_peak_sm_util_avg_ma": peak["sm_util_avg_ma"],
        "gpu_timeline_peak_phase": peak["phase"],
    }
    if peak.get("elapsed_s") is not None:
        summary["gpu_timeline_peak_elapsed_s"] = peak["elapsed_s"]
    if steady:
        steady_values = [item["sm_util_avg"] for item in steady]
        steady_ma_values = [item["sm_util_avg_ma"] for item in steady]
        summary["gpu_timeline_request_steady_sm_util_avg"] = sum(steady_values) / len(steady_values)
        summary["gpu_timeline_request_steady_sm_util_avg_ma"] = sum(steady_ma_values) / len(steady_ma_values)
        summary["gpu_timeline_request_steady_sample_count"] = len(steady)
    return summary


def parse_dmon_gpu_samples(path):
    rows = []
    header = None
    with open(path, "r", encoding="utf-8", errors="replace") as src:
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
            if not header or len(parts) < len(header):
                continue
            rec = dict(zip(header, parts))
            gpu = parse_int(rec.get("gpu"))
            if gpu is None or gpu < 0:
                continue
            sample_unix_s = None
            if rec.get("date") and rec.get("time"):
                try:
                    sample_unix_s = time.mktime(
                        time.strptime(f"{rec['date']} {rec['time']}", "%Y%m%d %H:%M:%S")
                    )
                except ValueError:
                    sample_unix_s = None
            row = {
                "sample_unix_s": sample_unix_s,
                "gpu": gpu,
                "util": parse_float(rec.get("sm")) or 0.0,
                "mem_util": parse_float(rec.get("mem")),
                "mem_used_mib": parse_float(rec.get("fb")) or 0.0,
                "mem_total_mib": 0.0,
                "power_w": parse_float(rec.get("pwr")),
                "gpu_temp_c": parse_float(rec.get("gtemp")),
                "mem_temp_c": parse_float(rec.get("mtemp")),
                "mclk_mhz": parse_float(rec.get("mclk")),
                "pclk_mhz": parse_float(rec.get("pclk")),
                "bar1_mib": parse_float(rec.get("bar1")),
                "pcie_rx": parse_float(rec.get("rxpci")),
                "pcie_tx": parse_float(rec.get("txpci")),
                "nvlink_rx": parse_float(rec.get("nvlrx")),
                "nvlink_tx": parse_float(rec.get("nvltx")),
            }
            consumed = {
                "date",
                "time",
                "gpu",
                "pwr",
                "gtemp",
                "mtemp",
                "sm",
                "mem",
                "enc",
                "dec",
                "jpg",
                "ofa",
                "mclk",
                "pclk",
                "fb",
                "bar1",
                "ccpm",
                "rxpci",
                "txpci",
                "nvlrx",
                "nvltx",
            }
            for key, value in rec.items():
                if key in consumed:
                    continue
                number = parse_float(value)
                if number is not None:
                    row[f"gpm_{key}"] = number
            rows.append(row)
    return rows


def parse_dcgmi_gpu_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as src:
        reader = csv.DictReader(line for line in src if not line.lstrip().startswith("#"))
        for rec in reader:
            gpu = parse_int(rec.get("gpu"))
            if gpu is None or gpu < 0:
                continue
            sample_unix_s = parse_float(rec.get("sample_unix_s"))
            row = {
                "sample_unix_s": sample_unix_s,
                "gpu": gpu,
                "gpu_util": parse_float(rec.get("gpu_util")),
                "mem_copy_util": parse_float(rec.get("mem_copy_util")),
                "mem_used_mib": parse_float(rec.get("fb_used_mib")) or 0.0,
                "mem_total_mib": 0.0,
                "fb_free_mib": parse_float(rec.get("fb_free_mib")),
                "power_w": parse_float(rec.get("power_w")),
                "gpu_temp_c": parse_float(rec.get("gpu_temp_c")),
                "sm_clock_mhz": parse_float(rec.get("sm_clock_mhz")),
                "mem_clock_mhz": parse_float(rec.get("mem_clock_mhz")),
                "gr_engine_active": parse_float(rec.get("gr_engine_active")),
                "sm_active": parse_float(rec.get("sm_active")),
                "sm_occupancy": parse_float(rec.get("sm_occupancy")),
                "tensor_active": parse_float(rec.get("tensor_active")),
                "dram_active": parse_float(rec.get("dram_active")),
                "fp64_active": parse_float(rec.get("fp64_active")),
                "fp32_active": parse_float(rec.get("fp32_active")),
                "fp16_active": parse_float(rec.get("fp16_active")),
                "pcie_tx_bytes": parse_float(rec.get("pcie_tx_bytes")),
                "pcie_rx_bytes": parse_float(rec.get("pcie_rx_bytes")),
                "nvlink_tx_bytes": parse_float(rec.get("nvlink_tx_bytes")),
                "nvlink_rx_bytes": parse_float(rec.get("nvlink_rx_bytes")),
                "integer_active": parse_float(rec.get("integer_active")),
            }
            if row["gpu_util"] is not None:
                row["util"] = row["gpu_util"]
            elif row["sm_active"] is not None:
                row["util"] = row["sm_active"] * 100.0
            else:
                row["util"] = 0.0
            for key, value in rec.items():
                if key in row or key in ("sample_unix_s", "entity_type", "gpu"):
                    continue
                number = parse_float(value)
                if number is not None:
                    row[key] = number
            rows.append(row)
    return rows


def parse_query_gpu_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as src:
        next(src, None)
        for line in src:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            sample_unix_s = None
            try:
                sample_unix_s = float(parts[0])
                gpu = int(parts[1])
                util = float(parts[2])
                mem_used = float(parts[3])
                mem_total = float(parts[4].split()[0])
            except ValueError:
                try:
                    # Backward compatibility with old nvidia-smi CSV rows:
                    # timestamp,index,utilization.gpu,memory.used,memory.total.
                    gpu = int(parts[1])
                    util = float(parts[2])
                    mem_used = float(parts[3])
                    mem_total = float(parts[4].split()[0])
                except ValueError:
                    continue
            if gpu < 0:
                continue
            rows.append(
                {
                    "sample_unix_s": sample_unix_s,
                    "gpu": gpu,
                    "util": util,
                    "mem_used_mib": mem_used,
                    "mem_total_mib": mem_total,
                }
            )
    return rows


def copy_gpu_aggregate(summary, prefix, aggregate):
    summary[f"{prefix}_sample_count"] = aggregate["sample_count"]
    if not aggregate["sample_count"]:
        return
    mapping = {
        "util_avg": "util_avg",
        "util_max": "util_max",
        "mem_used_max_mib": "mem_used_max_mib",
    }
    for src, dst in mapping.items():
        if src in aggregate:
            summary[f"{prefix}_{dst}"] = aggregate[src]
    for key, value in aggregate.items():
        if key in ("sample_count", "per_gpu", *mapping.keys()):
            continue
        summary[f"{prefix}_{key}"] = value
    summary[f"{prefix}_per_gpu"] = aggregate["per_gpu"]


def summarize_gpu_samples(path, lifecycle_path=None, timeline_path=None):
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as src:
        first = src.read(256)
    if first.lstrip().startswith("sample_unix_s,entity_type,gpu"):
        rows = parse_dcgmi_gpu_samples(path)
        source = "dcgmi"
    elif first.lstrip().startswith("#"):
        rows = parse_dmon_gpu_samples(path)
        source = "dmon"
    else:
        rows = parse_query_gpu_samples(path)
        source = "query"
    if not rows:
        return {"gpu_sample_count": 0}
    aggregate = summarize_gpu_rows(rows)
    summary = {"gpu_sample_source": source}
    copy_gpu_aggregate(summary, "gpu", aggregate)
    events = load_lifecycle_events(lifecycle_path) if lifecycle_path else {}
    if timeline_path:
        summary.update(write_gpu_timeline(timeline_path, rows, events))
    for name, event in events.items():
        summary[f"lifecycle_{name}_elapsed_s"] = event["elapsed_s"]
    ready = events.get("server_ready", {}).get("unix_s")
    request_start = events.get("requests_start", {}).get("unix_s")
    response_done = events.get("responses_complete", {}).get("unix_s")
    if ready is not None:
        startup_rows = [
            row for row in rows
            if row["sample_unix_s"] is not None and row["sample_unix_s"] <= ready
        ]
        startup = summarize_gpu_rows(startup_rows)
        copy_gpu_aggregate(summary, "gpu_startup", startup)
    if request_start is not None and response_done is not None:
        steady_rows = [
            row for row in rows
            if row["sample_unix_s"] is not None
            and request_start <= row["sample_unix_s"] <= response_done
        ]
        steady = summarize_gpu_rows(steady_rows)
        copy_gpu_aggregate(summary, "gpu_steady", steady)
    return summary


def profiler_prefix(args, case_dir):
    if args.tool == "none":
        return []
    if args.tool == "nvprof-gpu-trace":
        return [
            args.nvprof,
            "--profile-child-processes",
            "--csv",
            "--print-gpu-trace",
            "--log-file",
            profiler_log_pattern(case_dir, "nvprof-gpu-trace"),
        ]
    if args.tool == "nvprof-window-gpu-trace":
        return [
            args.nvprof,
            "--profile-from-start",
            "off",
            "--csv",
            "--print-gpu-trace",
            "--log-file",
            profiler_log_pattern(case_dir, "nvprof-window-gpu-trace"),
        ]
    if args.tool == "nvprof-api-trace":
        return [
            args.nvprof,
            "--profile-child-processes",
            "--csv",
            "--print-api-trace",
            "--log-file",
            profiler_log_pattern(case_dir, "nvprof-api-trace"),
        ]
    if args.tool == "nvprof-window-api-trace":
        return [
            args.nvprof,
            "--profile-from-start",
            "off",
            "--csv",
            "--print-api-trace",
            "--log-file",
            profiler_log_pattern(case_dir, "nvprof-window-api-trace"),
        ]
    if args.tool == "ncu-basic":
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--set",
            "basic",
            "--launch-count",
            str(args.ncu_launch_count),
            "--csv",
            "--log-file",
            str(case_dir / "ncu-basic.csv"),
        ]
        if args.ncu_launch_skip:
            cmd.extend(["--launch-skip", str(args.ncu_launch_skip)])
        if args.ncu_kernel_name:
            cmd.extend(["--kernel-name", args.ncu_kernel_name])
        return cmd
    if args.tool == "ncu-window-basic":
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--set",
            "basic",
            "--launch-count",
            str(args.ncu_launch_count),
            "--csv",
            "--log-file",
            str(case_dir / "ncu-window-basic.csv"),
        ]
        if args.ncu_launch_skip:
            cmd.extend(["--launch-skip", str(args.ncu_launch_skip)])
        if args.ncu_kernel_name:
            cmd.extend(["--kernel-name", args.ncu_kernel_name])
        return cmd
    if args.tool == "ncu-nvlink":
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--set",
            "nvlink",
            "--launch-count",
            str(args.ncu_launch_count),
            "--csv",
            "--log-file",
            str(case_dir / "ncu-nvlink.csv"),
        ]
        if args.ncu_launch_skip:
            cmd.extend(["--launch-skip", str(args.ncu_launch_skip)])
        if args.ncu_kernel_name:
            cmd.extend(["--kernel-name", args.ncu_kernel_name])
        return cmd
    if args.tool == "ncu-window-nvlink":
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--set",
            "nvlink",
            "--launch-count",
            str(args.ncu_launch_count),
            "--csv",
            "--log-file",
            str(case_dir / "ncu-window-nvlink.csv"),
        ]
        if args.ncu_launch_skip:
            cmd.extend(["--launch-skip", str(args.ncu_launch_skip)])
        if args.ncu_kernel_name:
            cmd.extend(["--kernel-name", args.ncu_kernel_name])
        return cmd
    raise ValueError(f"unsupported tool {args.tool}")


def apply_nccl_env(args, env, case_dir=None):
    if not (
        args.nccl_no_sys_ring
        or args.nccl_algo
        or args.nccl_proto
        or args.nccl_rings
        or args.nccl_p2p_level
        or args.nccl_debug
        or args.nccl_debug_subsys
        or args.nccl_shm_disable
        or args.nccl_topo_dump_file
        or args.nccl_graph_dump_file
    ):
        return

    if args.nccl_no_sys_ring:
        env.setdefault("NCCL_RINGS", NCCL_NO_SYS_RING)
        env.setdefault("NCCL_P2P_LEVEL", "NVL")
        env.setdefault("NCCL_DEBUG", "INFO")
        env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,COLL")

    if args.nccl_algo:
        env["NCCL_ALGO"] = args.nccl_algo
    if args.nccl_proto:
        env["NCCL_PROTO"] = args.nccl_proto
    if args.nccl_rings:
        env["NCCL_RINGS"] = args.nccl_rings
    if args.nccl_p2p_level:
        env["NCCL_P2P_LEVEL"] = args.nccl_p2p_level
    if args.nccl_debug:
        env["NCCL_DEBUG"] = args.nccl_debug
    if args.nccl_debug_subsys:
        env["NCCL_DEBUG_SUBSYS"] = args.nccl_debug_subsys
    if args.nccl_shm_disable:
        env["NCCL_SHM_DISABLE"] = args.nccl_shm_disable
    if args.nccl_topo_dump_file:
        env["NCCL_TOPO_DUMP_FILE"] = args.nccl_topo_dump_file
    if args.nccl_graph_dump_file:
        env["NCCL_GRAPH_DUMP_FILE"] = args.nccl_graph_dump_file
    if case_dir is not None and ("NCCL_DEBUG" in env or "NCCL_RINGS" in env):
        env.setdefault("NCCL_TOPO_DUMP_FILE", str(case_dir / "nccl-topology.xml"))
        env.setdefault("NCCL_GRAPH_DUMP_FILE", str(case_dir / "nccl-graph.xml"))


def nccl_env_summary(env):
    return {key: env[key] for key in NCCL_ENV_KEYS if key in env}


def write_nccl_env(case_dir, env):
    values = nccl_env_summary(env)
    if not values:
        return
    with open(case_dir / "nccl-env.txt", "w", encoding="utf-8") as out:
        for key in NCCL_ENV_KEYS:
            if key in values:
                out.write(f"{key}={values[key]}\n")


def summarize_nccl_artifacts(case_dir):
    server_err = case_dir / "server.err"
    text = server_err.read_text(errors="replace") if server_err.exists() else ""
    summary = {
        "nccl_log_sys_mentions": len(re.findall(r"\bSYS\b", text)),
        "nccl_log_net_mentions": len(re.findall(r"\bNET/", text)),
        "nccl_log_p2p_mentions": len(re.findall(r"\bP2P\b", text)),
        "nccl_log_ring_mentions": len(re.findall(r"\bRing\b|\bring\b", text)),
        "nccl_log_channel_mentions": len(re.findall(r"\bChannel\b|\bchannel\b", text)),
    }
    for name in ("nccl-topology.xml", "nccl-graph.xml"):
        path = case_dir / name
        if not path.exists():
            continue
        xml = path.read_text(errors="replace")
        stem = name.replace(".xml", "").replace("-", "_")
        summary[f"{stem}_bytes"] = path.stat().st_size
        summary[f"{stem}_sys_mentions"] = len(re.findall(r"\bSYS\b", xml))
        summary[f"{stem}_net_mentions"] = len(re.findall(r"\bNET\b", xml))
        if name == "nccl-graph.xml":
            summary.update(summarize_nccl_graph_edges(xml))
    return summary


def v100_link_class(src, dst):
    if src == dst:
        return "SELF"
    count = V100_NVLINK_COUNTS.get(frozenset((src, dst)))
    if count:
        return f"NV{count}"
    return "SYS"


def summarize_nccl_graph_edges(xml):
    channels = []
    for channel_text in re.findall(r"<channel>(.*?)</channel>", xml, flags=re.S):
        devices = [int(value) for value in re.findall(r'<gpu dev="(\d+)"', channel_text)]
        if devices:
            channels.append(devices)
    counts = {"NV1": 0, "NV2": 0, "SYS": 0, "SELF": 0}
    bad_edges = []
    edge_count = 0
    for devices in channels:
        for idx, src in enumerate(devices):
            dst = devices[(idx + 1) % len(devices)]
            link = v100_link_class(src, dst)
            counts[link] = counts.get(link, 0) + 1
            edge_count += 1
            if link == "SYS":
                bad_edges.append(f"{src}->{dst}")
    return {
        "nccl_graph_channel_count": len(channels),
        "nccl_graph_edge_count": edge_count,
        "nccl_graph_nv1_edge_count": counts.get("NV1", 0),
        "nccl_graph_nv2_edge_count": counts.get("NV2", 0),
        "nccl_graph_sys_edge_count": counts.get("SYS", 0),
        "nccl_graph_sys_edges": ",".join(bad_edges),
    }


def build_env(args, port, case_dir=None):
    env = os.environ.copy()
    env.update(
        {
            "DS4_V100_SERVE_MODE": "tp-ep",
            "DS4_V100_CTX": str(args.ctx),
            "DS4_V100_SLOTS": str(args.slots),
            "DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP": str(args.experimental_ctx_slot_cap)
            if args.experimental_ctx_slot_cap is not None
            else "",
            "DS4_V100_ACTIVE_MICROBATCH": str(args.slots),
            "DS4_V100_CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "DS4_V100_APPLIANCE_DIR": args.pack_dir,
            "DS4_V100_TP_EP_CONTRACT": args.contract,
            "DS4_V100_TURBOMIND_LIB": args.turbomind_lib,
            "DS4_V100_TP_EP_TOKENIZER_MODEL": args.tokenizer_model,
            "DS4_V100_TOKENS": str(args.tokens),
            "DS4_V100_STARTUP_WARMUP": args.startup_warmup,
            "DS4_V100_TP_EP_POSITION": str(args.position),
            # The HTTP server counts harness control requests too: one readiness
            # /health probe plus final /status and /metrics reads.
            "DS4_V100_MAX_REQUESTS": str(max(args.max_requests, args.requests + 3)),
            "DS4_V100_TP_EP_HC_PERSIST_STATE": "1",
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER": "1" if args.hc_current_peer_gather else "0",
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER": "1"
            if args.hc_current_nccl_allgather
            else "0",
            "DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE": "1"
            if args.hc_current_allreduce
            else "0",
            "DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY": "1"
            if args.hc_current_full_parity
            else "0",
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC": "1" if args.hc_current_stream_sync else "0",
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK": "1"
            if args.hc_current_fused_fill_pack
            else "0",
            "DS4_V100_TP_EP_PEER_ACCOUNTING": "1" if args.tp_peer_accounting else "0",
            "DS4_V100_TP_EP_PEER_REJECT_SYS": "1" if args.tp_peer_reject_sys else "0",
            "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD": "1",
            "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY": "1"
            if args.lazy_output_head
            else "0",
            "DS4_V100_TP_EP_ASYNC_OUTPUT": "1" if args.async_output else "0",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH": "1" if args.decode_cudagraph else "0",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC": "1"
            if args.decode_cudagraph_output_sync
            else "0",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC": "1"
            if args.decode_cudagraph_hc_current_sync
            else "0",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC": args.decode_cudagraph_stage_sync
            or "",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE": args.decode_cudagraph_suffix_stage
            or "",
            "DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT": "1"
            if args.persistent_decode_cudagraph
            else "0",
            "DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM": "1"
            if args.decode_stage_checksum
            else "0",
            "DS4_V100_TP_EP_BATCHED_PAGED_ATTN": "1" if args.batched_paged_attn else "0",
            "DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE": "0"
            if args.disable_compact_route_compose
            else "1",
            "DS4_V100_TP_EP_MODEL_ROUTER_ROUTES": "1" if args.model_router_routes else "0",
            "DS4_V100_TP_EP_ROUTER_CUBLAS": "1" if args.router_cublas else "0",
            "DS4_V100_TP_EP_ROUTER_HASH_FAST": "1" if args.router_hash_fast else "0",
            "DS4_V100_TP_EP_GPU_ROUTE_PLAN": "1" if args.gpu_route_plan else "0",
            "DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD": "0"
            if args.disable_route_plan_async_upload
            else "1",
            "DS4_V100_TP_EP_COMPACT_MOE_DECODE": "1" if args.compact_moe_decode else "0",
            "DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD": "1" if args.parallel_expert_load else "0",
            "DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE": "1"
            if args.nccl_reduce_scatter_compose
            else "0",
            "DS4_V100_TP_EP_FUSED_GATED_SILU": "1" if args.fused_gated_silu else "0",
            "DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT": "1" if args.routed_ffn_norm_input else "0",
            "DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT": "1"
            if args.routed_ffn_rank_major_input
            else "0",
            "DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS": "1"
            if args.model_router_rank_major_logits
            else "0",
            "DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS": "1"
            if args.model_router_allreduce_logits
            else "0",
            "DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN": "1"
            if args.post_attention_fixed_capacity_route_plan
            else "0",
            "DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC": "1"
            if args.post_attention_device_actual_route_sync
            else "0",
            "DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM": "1"
            if args.post_attention_slot_major_ffn_norm
            else "0",
            "DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM": "1"
            if args.post_attention_skip_slot_major_ffn_norm
            else "0",
            "DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY": "1"
            if args.post_attention_masked_compact_copy
            else "0",
            "DS4_V100_TP_EP_FP8_E5M2_KV": "1" if args.fp8_e5m2_kv else "0",
            "DS4_V100_TP_EP_VRAM_REPORT": "1" if args.vram_report else "0",
            "DS4_V100_TP_EP_VRAM_MIN_FREE_MIB": str(args.vram_min_free_mib),
            "DS4_V100_TP_EP_NCCL_MIN_FREE_MIB": str(args.nccl_min_free_mib),
            "DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE": "1"
            if args.skip_tp_runtime_comp_state
            else "0",
            "DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB": str(args.tp_runtime_scratch_mib),
            "DS4_V100_TP_EP_DEFER_NCCL_INIT": "1" if args.defer_nccl_init else "0",
            "DS4_V100_RESERVE_MIB": "0",
            "DS4_V100_PORT": str(port),
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT": "1"
            if args.attention_output
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER": "1"
            if args.attention_output_nccl_allgather
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT": "1"
            if args.post_attention_ffn_input
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS": "1"
            if args.semantic_skip_stats
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL": "1"
            if args.fused_compressed_input_fill
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND": "1"
            if args.fused_compressed_rope_round
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND": "1"
            if args.fused_compressed_pool_norm_rope_round
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL": "1"
            if args.direct_compressed_input_fill
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL": "1"
            if args.fused_compressed_attn_input_fill
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT": "1"
            if args.attention_projection_rank_local_input
            else "0",
            "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS": "1"
            if not args.disable_skip_compressed_dense_stats
            else "0",
            "DS4_V100_CUDA_PROFILER_WINDOW": "1" if "window" in args.tool else "0",
        }
    )
    if args.fused_compressed_pool_norm:
        env["DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM"] = "1"
    elif getattr(args, "disable_fused_compressed_pool_norm", False):
        env["DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM"] = "0"
    if args.compressed_dense_event_wait:
        env["DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT"] = "1"
    elif getattr(args, "disable_compressed_dense_event_wait", False):
        env["DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT"] = "0"
    apply_nccl_env(args, env, case_dir)
    return env


def variant_suffix(args):
    suffix = ""
    if args.hc_current_peer_gather:
        suffix += "-hc-peer-gather"
    if getattr(args, "hc_current_nccl_allgather", False):
        suffix += "-hc-nccl-allgather"
    if args.hc_current_stream_sync:
        suffix += "-hc-stream-sync"
    if getattr(args, "hc_current_fused_fill_pack", False):
        suffix += "-hc-fused-fill-pack"
    if getattr(args, "tp_peer_accounting", False):
        suffix += "-peeracct"
    if getattr(args, "tp_peer_reject_sys", False):
        suffix += "-rejectsys"
    if getattr(args, "skip_compressed_store", False):
        suffix += "-skip-compressed-store"
    if getattr(args, "skip_indexer_store", False):
        suffix += "-skip-indexer-store"
    if getattr(args, "fused_compressed_input_fill", False):
        suffix += "-fused-compressed-input-fill"
    if getattr(args, "fused_compressed_rope_round", False):
        suffix += "-fused-compressed-rope-round"
    if getattr(args, "fused_compressed_pool_norm", False):
        suffix += "-fused-compressed-pool-norm"
    if getattr(args, "disable_fused_compressed_pool_norm", False):
        suffix += "-no-fused-compressed-pool-norm"
    if getattr(args, "fused_compressed_pool_norm_rope_round", False):
        suffix += "-fused-compressed-pool-norm-rope-round"
    if getattr(args, "direct_compressed_input_fill", False):
        suffix += "-direct-compressed-input-fill"
    if getattr(args, "attention_output", False):
        suffix += "-attention-output"
    if getattr(args, "attention_output_nccl_allgather", False):
        suffix += "-attention-output-nccl-allgather"
    if getattr(args, "post_attention_ffn_input", False):
        suffix += "-post-attention-ffn-input"
    if getattr(args, "compressed_dense_event_wait", False):
        suffix += "-compressed-dense-event-wait"
    if getattr(args, "disable_compressed_dense_event_wait", False):
        suffix += "-no-compressed-dense-event-wait"
    if getattr(args, "skip_compressed_dense_stats", False):
        suffix += "-skip-compressed-dense-stats"
    if getattr(args, "disable_skip_compressed_dense_stats", False):
        suffix += "-no-skip-compressed-dense-stats"
    if getattr(args, "fused_compressed_attn_input_fill", False):
        suffix += "-fused-compressed-attn-input-fill"
    if getattr(args, "attention_projection_rank_local_input", False):
        suffix += "-attn-proj-rank-local"
    if getattr(args, "async_output", False):
        suffix += "-async-output"
    if getattr(args, "decode_cudagraph", False):
        suffix += "-decode-cudagraph"
    if getattr(args, "persistent_decode_cudagraph", False):
        suffix += "-persistent-decode-cudagraph"
    if getattr(args, "decode_cudagraph_suffix_stage", ""):
        suffix += f"-suffix-{args.decode_cudagraph_suffix_stage}"
    if getattr(args, "tp_runtime_scratch_mib", 1024) != 1024:
        suffix += f"-scratch{args.tp_runtime_scratch_mib}"
    if getattr(args, "cuda_visible_devices", "0,1,2,3,4,5,6,7") != "0,1,2,3,4,5,6,7":
        digest = hashlib.sha1(args.cuda_visible_devices.encode("utf-8")).hexdigest()[:8]
        suffix += f"-cuda-visible-h{digest}"
    if getattr(args, "nccl_no_sys_ring", False):
        suffix += "-nccl-no-sys"
    if getattr(args, "defer_nccl_init", False):
        suffix += "-defer-nccl"
    if getattr(args, "cuda_profiler_device", None) is not None:
        suffix += f"-profdev{args.cuda_profiler_device}"
    if getattr(args, "cuda_profiler_all_devices", False):
        suffix += "-prof-all-devices"
    if getattr(args, "batched_paged_attn", False):
        suffix += "-batched-paged-attn"
    if getattr(args, "model_router_routes", False):
        suffix += "-model-router"
    if getattr(args, "router_cublas", False):
        suffix += "-router-cublas"
    if getattr(args, "router_hash_fast", False):
        suffix += "-router-hash-fast"
    if getattr(args, "gpu_route_plan", False):
        suffix += "-gpu-route-plan"
    if getattr(args, "route_plan_async_upload", False):
        suffix += "-route-plan-async-upload"
    if getattr(args, "disable_route_plan_async_upload", False):
        suffix += "-no-route-plan-async-upload"
    if getattr(args, "disable_compact_route_compose", False):
        suffix += "-no-compact-route"
    if getattr(args, "compact_moe_decode", False):
        suffix += "-compact-moe"
    if getattr(args, "parallel_expert_load", True):
        suffix += "-parallel-expert-load"
    else:
        suffix += "-serial-expert-load"
    if getattr(args, "fused_gated_silu", False):
        suffix += "-fused-gated-silu"
    if getattr(args, "routed_ffn_norm_input", False):
        suffix += "-routed-norm"
    if getattr(args, "routed_ffn_rank_major_input", False):
        suffix += "-routed-rank-major"
    if getattr(args, "model_router_rank_major_logits", False):
        suffix += "-router-rank-major"
    if getattr(args, "model_router_allreduce_logits", False):
        suffix += "-router-allreduce"
    if getattr(args, "post_attention_fixed_capacity_route_plan", False):
        suffix += "-post-attn-fixed-route"
    if getattr(args, "post_attention_device_actual_route_sync", False):
        suffix += "-post-attn-actual-route"
    if getattr(args, "post_attention_slot_major_ffn_norm", False):
        suffix += "-post-attn-slot-major-ffn-norm"
    if getattr(args, "post_attention_skip_slot_major_ffn_norm", False):
        suffix += "-post-attn-skip-slot-major-ffn-norm"
    if getattr(args, "post_attention_masked_compact_copy", False):
        suffix += "-post-attn-masked-copy"
    if getattr(args, "fp8_e5m2_kv", False):
        suffix += "-fp8-e5m2-kv"
    if len(suffix) > 180:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:16]
        suffix = f"{suffix[:150]}-h{digest}"
    return suffix


def direct_command(args):
    kv_slot = min(7, max(0, args.slots - 1))
    cmd = [
        "./tools/ds4-v100-tp-ep-full-layer-smoke",
        "--pack-dir", args.pack_dir,
        "--contract", args.contract,
        "--tm-index", str(pathlib.Path(args.pack_dir) / "turbomind-pack-index.tsv"),
        "--lib", args.turbomind_lib,
        "--slots", str(args.slots),
        "--top-k", "6",
        "--kv-slot", str(kv_slot),
        "--position", str(args.position),
        "--warmup", "0",
        "--iters", "1",
        "--decode-steps", str(args.tokens),
        "--fuse-compose-sum",
        "--dense-f16-cublas-compose",
        "--dense-f16-cache-compose",
        "--skip-descriptor-checks",
        "--skip-predecode-probes",
        "--shared-expert-bindings",
        "--shared-dense-ops",
        "--overlap-ep-dense",
        "--source-copy-schedule",
        "--skip-self-compose-copy",
        "--multi-copy-streams",
        "--token-major-all-layers",
        "--all-layers",
        "--serving-bench",
        "--copy-event-compose",
        "--tp-hc-final-expand-gate",
        "--tp-hc-current-input-gate",
        "--tp-hc-persist-state-gate",
        "--true-ds4-attention-residency-gate",
        "--true-ds4-attention-projection-gate",
        "--true-ds4-attention-state-gate",
        "--true-ds4-attention-rope-gate",
        "--true-ds4-attention-raw-read-gate",
        "--true-ds4-attention-raw-window-gate",
        "--true-ds4-attention-typed-kv-raw-gate",
        "--true-ds4-attention-typed-kv-compressed-gate",
        "--true-ds4-attention-typed-kv-indexer-gate",
        "--true-ds4-attention-typed-kv-history-gate",
        "--true-ds4-attention-typed-kv-skip-current-load-gate",
        "--true-ds4-attention-typed-kv-quiet-gate",
        "--true-ds4-attention-typed-kv-batch-rows-gate",
        "--true-ds4-attention-typed-kv-stream-sync-gate",
        "--diagnostic-output-head",
    ]
    if args.resident_profile_layer is not None:
        cmd.extend(["--resident-profile-layer", str(args.resident_profile_layer)])
    if args.async_output:
        cmd.append("--async-output-gate")
    if args.decode_cudagraph:
        cmd.append("--decode-cudagraph-gate")
    if args.decode_cudagraph_output_sync:
        cmd.append("--decode-cudagraph-output-sync-gate")
    if args.decode_cudagraph_hc_current_sync:
        cmd.append("--decode-cudagraph-hc-current-sync-gate")
    if args.decode_cudagraph_stage_sync:
        cmd.extend(["--decode-cudagraph-stage-sync-gate", args.decode_cudagraph_stage_sync])
    if args.persistent_decode_cudagraph:
        cmd.append("--decode-cudagraph-persistent-replay-gate")
    if args.decode_cudagraph_suffix_stage:
        cmd.extend([
            "--decode-cudagraph-suffix-stage-gate",
            args.decode_cudagraph_suffix_stage,
        ])
    if args.decode_stage_checksum:
        cmd.append("--decode-stage-checksum-gate")
    if args.cuda_profiler_device is not None:
        cmd.extend(["--cuda-profiler-device", str(args.cuda_profiler_device)])
    if args.cuda_profiler_all_devices:
        cmd.append("--cuda-profiler-all-devices")
    if args.batched_paged_attn:
        cmd.append("--batched-paged-attn-gate")
    if not args.disable_compact_route_compose:
        cmd.append("--compact-route-compose")
    if args.model_router_routes:
        cmd.append("--model-router-routes")
    if args.router_cublas:
        cmd.append("--router-cublas-gate")
    if args.router_hash_fast:
        cmd.append("--router-hash-fast-gate")
    if args.gpu_route_plan:
        cmd.append("--gpu-route-plan-gate")
    if not args.disable_route_plan_async_upload:
        cmd.append("--route-plan-async-upload-gate")
    if args.compact_moe_decode:
        cmd.append("--compact-moe-decode-gate")
    if args.parallel_expert_load:
        cmd.append("--parallel-expert-load-gate")
    if args.nccl_reduce_scatter_compose:
        cmd.append("--nccl-reduce-scatter-compose-gate")
    if args.fused_gated_silu:
        cmd.append("--fused-gated-silu-gate")
    if args.routed_ffn_norm_input:
        cmd.append("--routed-ffn-norm-input-gate")
    if args.routed_ffn_rank_major_input:
        cmd.append("--routed-ffn-rank-major-input-gate")
    if args.model_router_rank_major_logits:
        cmd.append("--model-router-rank-major-logits-gate")
    if args.model_router_allreduce_logits:
        cmd.append("--model-router-allreduce-logits-gate")
    if args.post_attention_fixed_capacity_route_plan:
        cmd.append("--post-attention-fixed-capacity-route-plan-gate")
    if args.post_attention_route_reuse_audit:
        cmd.append("--post-attention-route-reuse-audit-gate")
    if args.post_attention_device_actual_route_sync:
        cmd.append("--post-attention-device-actual-route-sync-gate")
    if args.post_attention_slot_major_ffn_norm:
        cmd.append("--post-attention-slot-major-ffn-norm-gate")
    if args.post_attention_skip_slot_major_ffn_norm:
        cmd.append("--post-attention-skip-slot-major-ffn-norm-gate")
    if args.post_attention_masked_compact_copy:
        cmd.append("--post-attention-masked-compact-copy-gate")
    if args.vram_report:
        cmd.append("--vram-report")
    if args.vram_min_free_mib > 0:
        cmd.extend(["--vram-min-free-mib", str(args.vram_min_free_mib)])
    if args.nccl_min_free_mib > 0:
        cmd.extend(["--nccl-min-free-mib", str(args.nccl_min_free_mib)])
    if args.skip_tp_runtime_comp_state:
        cmd.append("--tp-runtime-skip-unused-comp-state-gate")
    cmd.extend(["--tp-runtime-scratch-mib", str(args.tp_runtime_scratch_mib)])
    if args.defer_nccl_init:
        cmd.append("--defer-nccl-init-gate")
    if args.fp8_e5m2_kv:
        cmd.append("--fp8-e5m2-kv-gate")
    if args.lazy_output_head:
        cmd.append("--diagnostic-output-head-lazy-gate")
    if "window" in args.tool:
        cmd.append("--cuda-profiler-window")
    if args.hc_current_peer_gather:
        cmd.append("--tp-hc-current-input-peer-gather-gate")
    if args.hc_current_nccl_allgather:
        cmd.append("--tp-hc-current-input-nccl-allgather-gate")
    if args.hc_current_allreduce:
        cmd.append("--tp-hc-current-allreduce-gate")
    if args.hc_current_full_parity:
        cmd.append("--tp-hc-current-full-parity-gate")
    if args.hc_current_stream_sync:
        cmd.append("--tp-hc-current-input-stream-sync-gate")
    if args.hc_current_fused_fill_pack:
        cmd.append("--tp-hc-current-input-fused-fill-pack-gate")
    if args.tp_peer_accounting:
        cmd.append("--tp-peer-accounting-gate")
    if args.tp_peer_reject_sys:
        cmd.append("--tp-peer-reject-sys-gate")
    if args.attention_projection_rank_local_input:
        cmd.append("--true-ds4-attention-projection-rank-local-input-gate")
    if args.attention_output:
        cmd.append("--true-ds4-attention-output-gate")
    if args.attention_output_nccl_allgather:
        cmd.append("--true-ds4-attention-output-nccl-allgather-gate")
    if args.post_attention_ffn_input:
        cmd.append("--true-ds4-post-attention-ffn-input-gate")
    if args.semantic_skip_stats and (
        args.attention_output
        or args.attention_output_nccl_allgather
        or args.post_attention_ffn_input
    ):
        cmd.append("--true-ds4-semantic-skip-stats-gate")
    if args.skip_compressed_store:
        cmd.append("--true-ds4-attention-typed-kv-skip-compressed-store-gate")
    if args.skip_indexer_store:
        cmd.append("--true-ds4-attention-typed-kv-skip-indexer-store-gate")
    if args.fused_compressed_input_fill:
        cmd.append("--true-ds4-compressed-kv-fused-input-fill-gate")
    if args.fused_compressed_rope_round:
        cmd.append("--true-ds4-compressed-kv-fused-rope-round-gate")
    if args.fused_compressed_pool_norm:
        cmd.append("--true-ds4-compressed-kv-fused-pool-norm-gate")
    if args.fused_compressed_pool_norm_rope_round:
        cmd.append("--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate")
    if args.direct_compressed_input_fill:
        cmd.append("--true-ds4-compressed-kv-direct-input-fill-gate")
    if args.compressed_dense_event_wait or not args.disable_compressed_dense_event_wait:
        cmd.append("--true-ds4-compressed-kv-dense-event-wait-gate")
    if args.skip_compressed_dense_stats or not args.disable_skip_compressed_dense_stats:
        cmd.append("--true-ds4-compressed-kv-skip-dense-stats-gate")
    if args.fused_compressed_attn_input_fill:
        cmd.append("--true-ds4-compressed-kv-fused-attn-input-fill-gate")
    return cmd


def parse_tab_line(line):
    parts = line.strip().split("\t")
    if not parts:
        return "", {}
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1]
    return parts[0], out


def maybe_number(value):
    if value is None:
        return None
    try:
        if re.search(r"[.eE]", value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value


def add_tp_ep_line_summaries(summary, stdout):
    compressed_sum_keys = [
        "attn_input_fill_ms",
        "attn_dense_ms",
        "attn_gather_ms",
        "attn_state_emit_ms",
        "attn_typed_ms",
        "indexer_input_fill_ms",
        "indexer_dense_ms",
        "indexer_gather_rope_ms",
        "indexer_state_emit_ms",
        "indexer_typed_score_ms",
        "reference_diff_ms",
        "ratio_shift_ms",
        "direct_input_fill",
        "dense_event_wait",
        "fused_attn_input_fill",
        "fused_input_fill",
        "fused_rope_round",
        "fused_pool_norm",
        "fused_pool_norm_rope_round",
        "ms",
    ]
    compressed_counts = {
        "layers": 0,
        "emitted_layers": 0,
        "ratio4_layers": 0,
        "ratio128_layers": 0,
        "direct_input_fill_layers": 0,
        "dense_event_wait_layers": 0,
        "skip_dense_stats_layers": 0,
        "fused_attn_input_fill_layers": 0,
        "fused_input_fill_layers": 0,
        "fused_rope_round_layers": 0,
        "fused_pool_norm_layers": 0,
        "fused_pool_norm_rope_round_layers": 0,
    }
    for line in stdout.splitlines():
        tag, fields = parse_tab_line(line)
        if tag == "tp_ep_serving_bench":
            for key in [
                "generated_tokens",
                "continuation_tokens",
                "total_decode_ms",
                "total_wall_ms",
                "aggregate_generated_tok_s_decode",
                "aggregate_generated_tok_s_wall",
                "aggregate_continuation_tok_s_decode",
                "aggregate_continuation_tok_s_wall",
            ]:
                summary[f"serving_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_token_major_scaffold":
            for key in [
                "pass_invocations",
                "sum_decode_ms",
                "ms_per_token",
                "projected_slot_step_tok_s",
                "sum_ep_ms",
                "sum_dense_ms",
                "sum_compose_ms",
                "sum_hc_current_input_ms",
                "sum_hc_current_seed_ms",
                "sum_hc_current_attn_mix_ms",
                "sum_hc_current_split_ms",
                "sum_hc_current_gather_ms",
                "sum_hc_current_ffn_router_ms",
                "sum_hc_current_ffn_norm_ms",
                "sum_hc_current_router_select_ms",
                "sum_hc_current_router_d2h_ms",
                "sum_hc_current_route_upload_ms",
                "sum_hc_current_fill_pack_ms",
                "sum_pre_ep_hc_current_ms",
                "sum_pre_ep_attention_projection_ms",
                "sum_pre_ep_compressed_kv_ms",
                "sum_pre_ep_attention_state_ms",
                "sum_pre_ep_typed_history_ms",
                "sum_pre_ep_raw_read_ms",
                "sum_pre_ep_attention_output_ms",
                "sum_pre_ep_post_attention_ffn_input_ms",
                "tp_hc_current_input_peer_gather",
                "tp_hc_current_input_nccl_allgather",
                "tp_hc_current_allreduce",
                "tp_hc_current_input_stream_sync",
                "compact_moe_decode_gate",
                "router_cublas_gate",
                "router_hash_fast_gate",
                "gpu_route_plan_gate",
                "route_plan_async_upload_gate",
                "decode_cudagraph_capture_attempted",
                "decode_cudagraph_capture_succeeded",
                "decode_cudagraph_capture_error",
                "decode_cudagraph_capture_nodes",
                "decode_cudagraph_replay_attempted",
                "decode_cudagraph_replay_succeeded",
                "decode_cudagraph_replay_error",
                "decode_cudagraph_persistent_cache_hits",
                "decode_cudagraph_persistent_cache_misses",
                "decode_cudagraph_persistent_invalidations",
                "decode_cudagraph_persistent_invalidate_layer",
                "decode_cudagraph_persistent_invalidate_slots",
                "decode_cudagraph_persistent_invalidate_position",
                "decode_cudagraph_persistent_invalidate_root_device",
                "decode_cudagraph_persistent_invalidate_root_stream",
                "decode_cudagraph_instantiate_ms",
                "decode_cudagraph_replay_ms",
                "fused_gated_silu_gate",
                "routed_ffn_norm_input_gate",
                "attention_projection_rank_local_input_gate",
                "routed_ffn_rank_major_input_gate",
                "model_router_rank_major_logits_gate",
                "model_router_allreduce_logits_gate",
                "post_attention_fixed_capacity_route_plan_gate",
                "post_attention_device_actual_route_sync_gate",
                "post_attention_static_rank_route_cap",
                "post_attention_static_executor_route_cap",
                "post_attention_static_compose_route_cap",
                "post_attention_slot_major_ffn_norm_gate",
                "post_attention_skip_slot_major_ffn_norm_gate",
                "post_attention_masked_compact_copy_gate",
                "routed_gate_standalone_swiglu",
                "sum_final_hc_ms",
                "wall_ms",
            ]:
                if key in fields:
                    summary[f"scaffold_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_decode_cudagraph_audit":
            for key in [
                "steps",
                "sync_all_calls",
                "event_barrier_calls",
                "stream_sync_count",
                "rank_stream_sync_count",
                "dense_stream_sync_count",
                "copy_stream_sync_count",
                "output_head_outside_step",
                "host_selected_token_dependency",
                "helper_host_sync_blocker_classes",
                "capture_attempted",
                "capture_succeeded",
                "capture_error_code",
                "capture_error_name",
                "capture_nodes",
                "replay_attempted",
                "replay_succeeded",
                "replay_error_code",
                "replay_error_name",
                "persistent_cache_hits",
                "persistent_cache_misses",
                "persistent_invalidations",
                "persistent_invalidate_layer",
                "persistent_invalidate_slots",
                "persistent_invalidate_position",
                "persistent_invalidate_root_device",
                "persistent_invalidate_root_stream",
                "sum_instantiate_ms",
                "sum_replay_ms",
                "capture_eligible",
                "blocker",
            ]:
                if key in fields:
                    summary[f"graph_audit_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_peer_copy_summary":
            for key in [
                "accounting",
                "reject_sys",
                "ops",
                "bytes",
                "self_ops",
                "self_bytes",
                "nv1_ops",
                "nv1_bytes",
                "nv2_ops",
                "nv2_bytes",
                "sys_ops",
                "sys_bytes",
                "unknown_ops",
                "unknown_bytes",
                "first_sys_src",
                "first_sys_dst",
                "first_sys_bytes",
                "first_sys_site",
                "first_sys_line",
            ]:
                if key in fields:
                    summary[f"peer_copy_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_peer_copy_site":
            site = fields.get("site", "-")
            sys_bytes = maybe_number(fields.get("sys_bytes"))
            sys_ops = maybe_number(fields.get("sys_ops"))
            current = summary.get("peer_copy_top_sys_site_bytes")
            if isinstance(sys_bytes, int) and (
                not isinstance(current, int) or sys_bytes > current
            ):
                summary["peer_copy_top_sys_site"] = site
                summary["peer_copy_top_sys_site_line"] = maybe_number(fields.get("line"))
                summary["peer_copy_top_sys_site_ops"] = sys_ops
                summary["peer_copy_top_sys_site_bytes"] = sys_bytes
                summary["peer_copy_top_sys_site_total_ops"] = maybe_number(fields.get("ops"))
                summary["peer_copy_top_sys_site_total_bytes"] = maybe_number(fields.get("bytes"))
        elif tag == "tp_ep_vram_summary":
            label = fields.get("label", "unknown")
            min_free = maybe_number(fields.get("min_free_mib"))
            max_used = maybe_number(fields.get("max_used_mib"))
            threshold = maybe_number(fields.get("threshold_mib"))
            failures = maybe_number(fields.get("failures"))
            summary[f"vram_{label}_min_free_mib"] = min_free
            summary[f"vram_{label}_max_used_mib"] = max_used
            summary[f"vram_{label}_threshold_mib"] = threshold
            summary[f"vram_{label}_failures"] = failures
            if isinstance(min_free, (int, float)):
                previous = summary.get("vram_min_free_mib")
                summary["vram_min_free_mib"] = min_free if previous is None else min(previous, min_free)
            if isinstance(max_used, (int, float)):
                previous = summary.get("vram_max_used_mib")
                summary["vram_max_used_mib"] = max_used if previous is None else max(previous, max_used)
            if isinstance(threshold, (int, float)):
                summary["vram_threshold_mib"] = threshold
            if isinstance(failures, int):
                summary["vram_failures"] = summary.get("vram_failures", 0) + failures
        elif tag == "tp_ep_compressed_kv_projection":
            compressed_counts["layers"] += 1
            if maybe_number(fields.get("emitted_compressed_rows")):
                compressed_counts["emitted_layers"] += 1
            ratio = maybe_number(fields.get("ratio"))
            if ratio == 4:
                compressed_counts["ratio4_layers"] += 1
            elif ratio == 128:
                compressed_counts["ratio128_layers"] += 1
            if maybe_number(fields.get("fused_input_fill")):
                compressed_counts["fused_input_fill_layers"] += 1
            if maybe_number(fields.get("direct_input_fill")):
                compressed_counts["direct_input_fill_layers"] += 1
            if maybe_number(fields.get("dense_event_wait")):
                compressed_counts["dense_event_wait_layers"] += 1
            if maybe_number(fields.get("skip_dense_stats")):
                compressed_counts["skip_dense_stats_layers"] += 1
            if maybe_number(fields.get("fused_attn_input_fill")):
                compressed_counts["fused_attn_input_fill_layers"] += 1
            if maybe_number(fields.get("fused_rope_round")):
                compressed_counts["fused_rope_round_layers"] += 1
            if maybe_number(fields.get("fused_pool_norm")):
                compressed_counts["fused_pool_norm_layers"] += 1
            if maybe_number(fields.get("fused_pool_norm_rope_round")):
                compressed_counts["fused_pool_norm_rope_round_layers"] += 1
            for key in compressed_sum_keys:
                value = maybe_number(fields.get(key))
                if isinstance(value, (int, float)):
                    summary[f"compressed_kv_sum_{key}"] = (
                        summary.get(f"compressed_kv_sum_{key}", 0.0) + float(value)
                    )
        elif tag == "tp_ep_diagnostic_output_head":
            for key in ["total_ms", "projection_ms", "top1_ms", "first_token", "finite_bad"]:
                summary[f"output_head_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_compact_moe_route_stats":
            for key in [
                "layer",
                "routes",
                "duplicate_slots",
                "max_same_rank_routes",
                "all_dest_bytes",
                "compact_bytes",
            ]:
                summary[f"compact_moe_{key}"] = maybe_number(fields.get(key))
    for key, value in compressed_counts.items():
        summary[f"compressed_kv_{key}"] = value
    return summary


def add_decode_domain_summary(summary):
    total = summary.get("scaffold_sum_decode_ms")
    if not isinstance(total, (int, float)) or total <= 0:
        return summary
    coarse_keys = [
        ("hc_current_input", "scaffold_sum_hc_current_input_ms"),
        ("ep", "scaffold_sum_ep_ms"),
        ("dense", "scaffold_sum_dense_ms"),
        ("compose", "scaffold_sum_compose_ms"),
        ("final_hc", "scaffold_sum_final_hc_ms"),
    ]
    coarse = []
    accounted = 0.0
    for name, key in coarse_keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            ms = float(value)
            accounted += ms
            coarse.append({"name": name, "ms": ms, "pct": 100.0 * ms / float(total)})
    remainder = max(0.0, float(total) - accounted)
    coarse.append({"name": "other_or_overlap", "ms": remainder, "pct": 100.0 * remainder / float(total)})
    summary["decode_domain_total_ms"] = float(total)
    summary["decode_domain_coarse_ranked"] = sorted(coarse, key=lambda item: item["ms"], reverse=True)

    fine_keys = [
        ("hc_current_seed", "scaffold_sum_hc_current_seed_ms"),
        ("hc_current_attn_mix", "scaffold_sum_hc_current_attn_mix_ms"),
        ("hc_current_split", "scaffold_sum_hc_current_split_ms"),
        ("hc_current_gather", "scaffold_sum_hc_current_gather_ms"),
        ("hc_current_ffn_router", "scaffold_sum_hc_current_ffn_router_ms"),
        ("hc_current_ffn_norm", "scaffold_sum_hc_current_ffn_norm_ms"),
        ("hc_current_router_select", "scaffold_sum_hc_current_router_select_ms"),
        ("hc_current_router_d2h", "scaffold_sum_hc_current_router_d2h_ms"),
        ("hc_current_route_upload", "scaffold_sum_hc_current_route_upload_ms"),
        ("hc_current_fill_pack", "scaffold_sum_hc_current_fill_pack_ms"),
        ("pre_ep_hc_current", "scaffold_sum_pre_ep_hc_current_ms"),
        ("pre_ep_attention_projection", "scaffold_sum_pre_ep_attention_projection_ms"),
        ("pre_ep_compressed_kv", "scaffold_sum_pre_ep_compressed_kv_ms"),
        ("pre_ep_attention_state", "scaffold_sum_pre_ep_attention_state_ms"),
        ("pre_ep_typed_history", "scaffold_sum_pre_ep_typed_history_ms"),
        ("pre_ep_raw_read", "scaffold_sum_pre_ep_raw_read_ms"),
        ("pre_ep_attention_output", "scaffold_sum_pre_ep_attention_output_ms"),
        ("pre_ep_post_attention_ffn_input", "scaffold_sum_pre_ep_post_attention_ffn_input_ms"),
    ]
    fine = []
    for name, key in fine_keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            ms = float(value)
            fine.append({"name": name, "ms": ms, "pct": 100.0 * ms / float(total)})
    if fine:
        summary["decode_domain_fine_ranked"] = sorted(fine, key=lambda item: item["ms"], reverse=True)
    return summary


def summarize_direct(case_dir, tool, rc, elapsed_s, args=None, env=None):
    stdout = (case_dir / "stdout.txt").read_text(errors="replace")
    stderr = (case_dir / "stderr.txt").read_text(errors="replace")
    summary = {
        "tool": tool,
        "run_mode": "direct-token-major",
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "cuda_visible_devices": args.cuda_visible_devices,
        "nccl_no_sys_ring": args.nccl_no_sys_ring if args is not None else False,
        "nccl_env": nccl_env_summary(env or {}),
        "profiler_marker_lines": len(re.findall(r"tp_ep_cuda_profiler_window", stderr)),
    }
    add_tp_ep_line_summaries(summary, stdout)
    summary.update(summarize_nccl_artifacts(case_dir))
    return add_decode_domain_summary(summary)


def run_direct_case(args):
    case_name = f"{args.tool}-direct{variant_suffix(args)}"
    case_dir = args.artifact_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cmd = profiler_prefix(args, case_dir) + direct_command(args)
    (case_dir / "profile-command.txt").write_text(" ".join(cmd) + "\n")
    (case_dir / "command.txt").write_text(" ".join(direct_command(args)) + "\n")
    apply_nccl_env(args, env, case_dir)
    write_nccl_env(case_dir, env)
    lifecycle = LifecycleEvents(case_dir / "lifecycle.csv")
    lifecycle.mark("process_start", f"run_mode=direct-token-major tool={args.tool}")
    started = time.time()
    with GpuSampler(
        case_dir / "gpu_util.csv",
        args.gpu_sample_interval_ms / 1000.0,
        mode=args.gpu_sampler,
        dcgmi_fields=args.dcgmi_fields,
    ):
        lifecycle.mark("requests_start", f"decode_steps={args.tokens} slots={args.slots}")
        with open(case_dir / "stdout.txt", "wb") as stdout, open(case_dir / "stderr.txt", "wb") as stderr:
            proc = subprocess.run(
                cmd,
                cwd=args.repo_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=args.request_timeout_seconds,
                check=False,
            )
        lifecycle.mark("responses_complete", f"returncode={proc.returncode}")
    elapsed_s = time.time() - started
    summary = summarize_direct(case_dir, args.tool, proc.returncode, elapsed_s, args, env)
    summary.update(
        summarize_gpu_samples(
            case_dir / "gpu_util.csv",
            case_dir / "lifecycle.csv",
            case_dir / "gpu_timeline.csv",
        )
    )
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lifecycle.mark("summary_written")
    write_top_kernels(case_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return proc.returncode


def parse_nvprof_gpu_trace(path):
    if not path.exists():
        return []
    rows = []
    header = None
    for line in path.read_text(errors="replace").splitlines():
        if not line or line.startswith("=="):
            continue
        try:
            parts = next(csv.reader([line]))
        except csv.Error:
            continue
        parts = [part.strip() for part in parts]
        if parts and parts[0] == "Start":
            header = {name: i for i, name in enumerate(parts)}
            continue
        if parts and parts[0] == "s":
            continue
        if len(parts) < 2:
            continue
        name_idx = header.get("Name") if header else None
        name = parts[name_idx] if name_idx is not None and name_idx < len(parts) else parts[-2]
        if "(" not in name and "kernel" not in name.lower():
            continue
        duration_s = None
        duration_idx = header.get("Duration") if header else 1
        unit_idx = duration_idx
        while unit_idx < len(parts) and parts[unit_idx] not in ("s", "ms", "us", "ns"):
            unit_idx += 1
        try:
            value = float(parts[duration_idx])
        except (TypeError, ValueError):
            continue
        if unit_idx < len(parts):
            scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[parts[unit_idx]]
        else:
            scale = 1e-3
        duration_s = value * scale
        if duration_s is not None:
            rows.append((name, duration_s))
    by_name = {}
    for name, duration_s in rows:
        total, count = by_name.get(name, (0.0, 0))
        by_name[name] = (total + duration_s, count + 1)
    return sorted(
        (
            {
                "kernel": name,
                "calls": count,
                "total_ms": total * 1000.0,
                "avg_us": total * 1_000_000.0 / count if count else 0.0,
            }
            for name, (total, count) in by_name.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )


def write_top_kernels(case_dir):
    kernels = []
    for path in sorted(case_dir.glob("nvprof-gpu-trace.*.csv")):
        kernels.extend(parse_nvprof_gpu_trace(path))
    for path in sorted(case_dir.glob("nvprof-window-gpu-trace.*.csv")):
        kernels.extend(parse_nvprof_gpu_trace(path))
    if not kernels:
        return
    by_name = {}
    for item in kernels:
        total_ms, calls = by_name.get(item["kernel"], (0.0, 0))
        by_name[item["kernel"]] = (total_ms + item["total_ms"], calls + item["calls"])
    merged = sorted(
        (
            {
                "kernel": name,
                "calls": calls,
                "total_ms": total_ms,
                "avg_us": total_ms * 1000.0 / calls if calls else 0.0,
            }
            for name, (total_ms, calls) in by_name.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )
    with open(case_dir / "top-kernels.tsv", "w") as out:
        out.write("rank\tcalls\ttotal_ms\tavg_us\tkernel\n")
        for i, item in enumerate(merged[:80], 1):
            out.write(
                f"{i}\t{item['calls']}\t{item['total_ms']:.6f}\t"
                f"{item['avg_us']:.6f}\t{item['kernel']}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=pathlib.Path, default=pathlib.Path("/workspace/ds4-sprint181"))
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--pack-dir", default="/workspace/packs/ds4-appliance-full-tm-gated-s181")
    parser.add_argument(
        "--contract",
        default="/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv",
    )
    parser.add_argument("--turbomind-lib", default="/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so")
    parser.add_argument("--tokenizer-model", default="/models/DSv4-Flash-256e-fixed.gguf")
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--experimental-ctx-slot-cap", type=int)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument(
        "--cuda-visible-devices",
        default=NCCL_DEFAULT_VISIBLE_DEVICES,
        help=(
            "physical CUDA device order exposed to the process. The TP/EP "
            "default keeps this natural and forces topology through NCCL_RINGS."
        ),
    )
    parser.add_argument(
        "--nccl-no-sys-ring",
        action="store_true",
        default=True,
        help=(
            "enable the V100 no-SYS NCCL policy. This is the default: natural "
            "CUDA device order, physical rank ring "
            f"{NCCL_NO_SYS_RING}, NVLink-only P2P, and NCCL debug/dump files. "
            "NCCL_ALGO/NCCL_PROTO are left unset unless explicitly provided."
        ),
    )
    parser.add_argument(
        "--disable-nccl-no-sys-ring",
        dest="nccl_no_sys_ring",
        action="store_false",
        help="diagnostic opt-out; allows NCCL to choose topology without the no-SYS guardrail",
    )
    parser.add_argument("--nccl-algo", default="")
    parser.add_argument("--nccl-proto", default="")
    parser.add_argument("--nccl-rings", default="")
    parser.add_argument("--nccl-p2p-level", default="")
    parser.add_argument("--nccl-debug", default="")
    parser.add_argument("--nccl-debug-subsys", default="")
    parser.add_argument("--nccl-shm-disable", default="")
    parser.add_argument("--nccl-topo-dump-file", default="")
    parser.add_argument("--nccl-graph-dump-file", default="")
    parser.add_argument("--startup-warmup", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--position", type=int, default=100000)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument(
        "--prompt-file",
        type=pathlib.Path,
        help="JSONL chat prompt records; each line has id plus messages or prompt",
    )
    parser.add_argument(
        "--http-endpoint",
        choices=["chat", "selected-token"],
        default="chat",
    )
    parser.add_argument("--hc-current-peer-gather", action="store_true")
    parser.add_argument("--hc-current-nccl-allgather", action="store_true")
    parser.add_argument("--hc-current-allreduce", action="store_true")
    parser.add_argument("--hc-current-full-parity", action="store_true")
    parser.add_argument("--hc-current-stream-sync", action="store_true")
    parser.add_argument("--hc-current-fused-fill-pack", action="store_true")
    parser.add_argument("--tp-peer-accounting", action="store_true")
    parser.add_argument("--tp-peer-reject-sys", action="store_true")
    parser.add_argument("--attention-projection-rank-local-input", action="store_true")
    parser.add_argument("--resident-profile-layer", type=int)
    parser.add_argument("--attention-output", action="store_true")
    parser.add_argument("--attention-output-nccl-allgather", action="store_true")
    parser.add_argument("--post-attention-ffn-input", action="store_true")
    parser.add_argument(
        "--semantic-skip-stats",
        dest="semantic_skip_stats",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable-semantic-skip-stats",
        dest="semantic_skip_stats",
        action="store_false",
    )
    parser.add_argument("--skip-compressed-store", action="store_true")
    parser.add_argument("--skip-indexer-store", action="store_true")
    parser.add_argument("--fused-compressed-input-fill", action="store_true")
    parser.add_argument("--fused-compressed-rope-round", action="store_true")
    parser.add_argument("--fused-compressed-pool-norm", action="store_true")
    parser.add_argument("--fused-compressed-pool-norm-rope-round", action="store_true")
    parser.add_argument("--direct-compressed-input-fill", action="store_true")
    parser.add_argument("--compressed-dense-event-wait", action="store_true")
    parser.add_argument("--disable-compressed-dense-event-wait", action="store_true")
    parser.add_argument("--skip-compressed-dense-stats", action="store_true")
    parser.add_argument("--disable-skip-compressed-dense-stats", action="store_true")
    parser.add_argument("--fused-compressed-attn-input-fill", action="store_true")
    parser.add_argument("--disable-fused-compressed-pool-norm", action="store_true")
    parser.add_argument("--async-output", action="store_true")
    parser.add_argument("--decode-cudagraph", action="store_true")
    parser.add_argument("--decode-cudagraph-output-sync", action="store_true")
    parser.add_argument("--decode-cudagraph-hc-current-sync", action="store_true")
    parser.add_argument("--decode-cudagraph-stage-sync", default="")
    parser.add_argument("--decode-cudagraph-suffix-stage", default="")
    parser.add_argument("--persistent-decode-cudagraph", action="store_true")
    parser.add_argument("--decode-stage-checksum", action="store_true")
    parser.add_argument("--tp-runtime-scratch-mib", type=int, default=1024)
    parser.add_argument("--defer-nccl-init", action="store_true")
    parser.add_argument("--batched-paged-attn", action="store_true")
    parser.add_argument("--model-router-routes", action="store_true")
    parser.add_argument("--router-cublas", action="store_true")
    parser.add_argument("--router-hash-fast", action="store_true")
    parser.add_argument("--gpu-route-plan", action="store_true")
    parser.add_argument("--route-plan-async-upload", action="store_true")
    parser.add_argument("--disable-route-plan-async-upload", action="store_true")
    parser.add_argument("--disable-compact-route-compose", action="store_true")
    parser.add_argument("--compact-moe-decode", action="store_true")
    parser.add_argument(
        "--parallel-expert-load",
        dest="parallel_expert_load",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable-parallel-expert-load",
        dest="parallel_expert_load",
        action="store_false",
    )
    parser.add_argument("--nccl-reduce-scatter-compose", action="store_true")
    parser.add_argument("--fused-gated-silu", action="store_true")
    parser.add_argument("--routed-ffn-norm-input", action="store_true")
    parser.add_argument("--routed-ffn-rank-major-input", action="store_true")
    parser.add_argument("--model-router-rank-major-logits", action="store_true")
    parser.add_argument("--model-router-allreduce-logits", action="store_true")
    parser.add_argument("--post-attention-fixed-capacity-route-plan", action="store_true")
    parser.add_argument("--post-attention-route-reuse-audit", action="store_true")
    parser.add_argument("--post-attention-device-actual-route-sync", action="store_true")
    parser.add_argument("--post-attention-slot-major-ffn-norm", action="store_true")
    parser.add_argument("--post-attention-skip-slot-major-ffn-norm", action="store_true")
    parser.add_argument("--post-attention-masked-compact-copy", action="store_true")
    parser.add_argument("--fp8-e5m2-kv", action="store_true")
    parser.add_argument(
        "--skip-tp-runtime-comp-state",
        dest="skip_tp_runtime_comp_state",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable-skip-tp-runtime-comp-state",
        dest="skip_tp_runtime_comp_state",
        action="store_false",
    )
    parser.add_argument("--lazy-output-head", action="store_true")
    parser.add_argument("--vram-report", action="store_true")
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--nccl-min-free-mib", type=int, default=1536)
    parser.add_argument("--port", type=int, default=18357)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=0,
        help=(
            "maximum simultaneous HTTP requests; default 0 uses --requests. "
            "Use this for long steady-state runs that should keep a fixed slot "
            "count busy without opening one connection per queued request."
        ),
    )
    parser.add_argument(
        "--run-mode",
        choices=["http", "direct-token-major"],
        default="http",
    )
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--nvprof", default="/usr/local/cuda/bin/nvprof")
    parser.add_argument("--ncu-launch-count", type=int, default=160)
    parser.add_argument("--ncu-launch-skip", type=int, default=0)
    parser.add_argument("--cuda-profiler-device", type=int)
    parser.add_argument("--cuda-profiler-all-devices", action="store_true")
    parser.add_argument(
        "--gpu-sample-interval-ms",
        type=int,
        default=0,
        help="sample nvidia-smi telemetry into gpu_util.csv; 0 disables sampling",
    )
    parser.add_argument(
        "--gpu-sampler",
        choices=["dmon", "dcgmi", "query"],
        default="dmon",
        help="GPU sampler backend; dcgmi captures V100 PROF counters but contends with Nsight/ncu",
    )
    parser.add_argument(
        "--dcgmi-fields",
        default=DCGMI_DEFAULT_FIELDS,
        help=(
            "comma-separated dcgmi dmon field IDs. Default avoids V100 A-subgroup "
            "multiplexing by excluding tensor_active; use 1004 in a separate pass."
        ),
    )
    parser.add_argument(
        "--kill-stale-server",
        action="store_true",
        help="terminate stale managed DS4 serve-http processes on the selected port before and after the run",
    )
    parser.add_argument(
        "--ncu-kernel-name",
        default="",
        help="optional Nsight Compute --kernel-name filter, e.g. regex:.*cutlass_70_wmma.*",
    )
    parser.add_argument(
        "--tool",
        choices=[
            "none",
            "nvprof-gpu-trace",
            "nvprof-window-gpu-trace",
            "nvprof-api-trace",
            "nvprof-window-api-trace",
            "ncu-basic",
            "ncu-window-basic",
            "ncu-nvlink",
            "ncu-window-nvlink",
        ],
        default="nvprof-gpu-trace",
    )
    args = parser.parse_args()
    if args.nccl_no_sys_ring and args.cuda_visible_devices != NCCL_DEFAULT_VISIBLE_DEVICES:
        parser.error(
            "--nccl-no-sys-ring uses physical rank order and requires "
            f"--cuda-visible-devices {NCCL_DEFAULT_VISIBLE_DEVICES}; use "
            "--disable-nccl-no-sys-ring for visible-order diagnostics"
        )
    if args.fused_compressed_pool_norm and args.disable_fused_compressed_pool_norm:
        parser.error("--fused-compressed-pool-norm and --disable-fused-compressed-pool-norm are mutually exclusive")
    if args.skip_compressed_dense_stats and args.disable_skip_compressed_dense_stats:
        parser.error("--skip-compressed-dense-stats and --disable-skip-compressed-dense-stats are mutually exclusive")
    if args.compressed_dense_event_wait and args.disable_compressed_dense_event_wait:
        parser.error("--compressed-dense-event-wait and --disable-compressed-dense-event-wait are mutually exclusive")
    prompt_records = None
    if args.prompt_file:
        prompt_records = load_prompt_records(args.prompt_file)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "direct-token-major":
        raise SystemExit(run_direct_case(args))

    case_dir = args.artifact_dir / f"{args.tool}{variant_suffix(args)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args, args.port, case_dir)
    env["DS4_V100_LOG_DIR"] = str(case_dir / "launcher")
    env["DS4_LOCK_FILE"] = str(case_dir / "ds4.lock")
    write_nccl_env(case_dir, env)
    base = f"http://127.0.0.1:{args.port}"
    if args.kill_stale_server:
        cleanup_managed_server_port(args.port, case_dir, "preflight")
    elif port_is_open(args.port):
        pids = ds4_server_pids_for_port(args.port)
        raise RuntimeError(
            f"port {args.port} is already open before profile run; "
            f"rerun with --kill-stale-server if it is a stale DS4 harness process; "
            f"ds4_pids={pids}"
        )

    with open(case_dir / "command.txt", "wb") as out:
        subprocess.run(
            ["./tools/ds4-v100-run-appliance.sh", "--print-command"],
            cwd=args.repo_dir,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )

    cmd = profiler_prefix(args, case_dir) + ["./tools/ds4-v100-run-appliance.sh"]
    (case_dir / "profile-command.txt").write_text(" ".join(cmd) + "\n")
    lifecycle = LifecycleEvents(case_dir / "lifecycle.csv")
    lifecycle.mark(
        "process_start",
        f"run_mode=http tool={args.tool} requests={args.requests} slots={args.slots} tokens={args.tokens}",
    )
    server_out = open(case_dir / "server.out", "wb")
    server_err = open(case_dir / "server.err", "wb")
    gpu_sampler = GpuSampler(
        case_dir / "gpu_util.csv",
        args.gpu_sample_interval_ms / 1000.0,
        mode=args.gpu_sampler,
        dcgmi_fields=args.dcgmi_fields,
    )
    gpu_sampler.__enter__()
    proc = subprocess.Popen(
        cmd,
        cwd=args.repo_dir,
        env=env,
        stdout=server_out,
        stderr=server_err,
        preexec_fn=os.setsid,
    )
    (case_dir / "server.pid").write_text(f"{proc.pid}\n")
    lifecycle.mark("server_spawned", f"pid={proc.pid} port={args.port}")

    try:
        for _ in range(args.readiness_seconds):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                _, body = http_get(base, "/health", timeout=2)
                (case_dir / "health.json").write_bytes(body)
                lifecycle.mark("server_ready", f"port={args.port}")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("readiness timeout")

        if args.http_endpoint == "selected-token":
            post_path = "/v100/selected-token"
            payloads = [
                {
                    "max_tokens": args.tokens,
                    "session_id": f"s357-selected-{args.tool}-{i:02d}",
                }
                for i in range(args.requests)
            ]
        else:
            post_path = "/v1/chat/completions"
            payloads = []
            for i in range(args.requests):
                if prompt_records:
                    record = prompt_records[i % len(prompt_records)]
                    prompt_id = record["id"]
                    messages = record["messages"]
                else:
                    prompt_id = f"default-{i:02d}"
                    messages = [
                        {
                            "role": "user",
                            "content": f"Say one short sentence about TP Nsight profile {i}.",
                        }
                    ]
                payloads.append(
                    {
                        "model": "ds4-v100-tp-ep-diagnostic",
                        "messages": messages,
                        "max_tokens": args.tokens,
                        "session_id": f"s345-profile-{args.tool}-{prompt_id}-{i:02d}",
                    }
                )

        lifecycle.mark("requests_start", f"endpoint={post_path} count={len(payloads)}")
        started = time.time()
        results = []

        def one(item):
            i, payload = item
            status, body = http_post(
                base,
                post_path,
                payload,
                timeout=args.request_timeout_seconds,
            )
            (case_dir / f"response-{i:02d}.txt").write_text(
                body.decode(errors="replace") + f"\nHTTP_STATUS:{status}\n"
            )
            return status, body

        request_concurrency = args.request_concurrency or args.requests
        request_concurrency = max(1, min(request_concurrency, args.requests))
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_concurrency) as executor:
            for result in executor.map(one, enumerate(payloads)):
                results.append(result)

        elapsed_s = time.time() - started
        ok = sum(1 for status, _ in results if status == 200)
        lifecycle.mark("responses_complete", f"http_200={ok}/{len(results)} elapsed_s={elapsed_s:.6f}")
        _, body = http_get(base, "/status", timeout=30)
        (case_dir / "status.json").write_bytes(body)
        status_json = json.loads(body)
        _, body = http_get(base, "/metrics", timeout=30)
        (case_dir / "metrics.txt").write_bytes(body)
        lifecycle.mark("status_metrics_complete")

        metas = [
            json.loads(body.decode("utf-8", errors="replace")).get("ds4_v100", {})
            for status, body in results
            if status == 200
        ]
        first = metas[0] if metas else {}
        timing = first.get("timing_ms", {})
        server_text = (case_dir / "server.out").read_text(errors="replace")
        summary = {
            "tool": args.tool,
            "http_endpoint": args.http_endpoint,
            "prompt_file": str(args.prompt_file) if args.prompt_file else "",
            "prompt_count": len(prompt_records) if prompt_records else 0,
            "prompt_digest": prompt_digest(prompt_records) if prompt_records else "",
            "http_200": ok,
            "requests": len(results),
            "tokens": args.tokens,
            "cuda_visible_devices": args.cuda_visible_devices,
            "nccl_no_sys_ring": args.nccl_no_sys_ring,
            "nccl_env": nccl_env_summary(env),
            "elapsed_s": elapsed_s,
            "client_generated_tok_s": (ok * args.tokens) / elapsed_s if elapsed_s > 0 else 0.0,
            "server_generated_tok_s": timing.get("generated_tokens_per_second", 0.0),
            "server_generated_tok_s_decode": timing.get("generated_tokens_per_second_decode", 0.0),
            "server_continuation_tok_s": timing.get("continuation_tokens_per_second", 0.0),
            "server_continuation_tok_s_decode": timing.get("continuation_tokens_per_second_decode", 0.0),
            "cache_hits": status_json.get("cache_hits"),
            "cache_misses": status_json.get("cache_misses"),
            "coalesced_batch_size": first.get("coalesced_batch_size"),
            "generated_tokens_meta": first.get("batch_generated_tokens"),
            "fp8_e5m2_kv_meta": first.get("fp8_e5m2_kv_gate"),
            "status_fp8_e5m2_kv": status_json.get("fp8_e5m2_kv_gate"),
            "status_router_hash_fast": status_json.get("router_hash_fast_gate"),
            "status_route_plan_async_upload": status_json.get("route_plan_async_upload_gate"),
            "peer_copy_accounting": status_json.get("peer_copy_accounting"),
            "peer_copy_reject_sys": status_json.get("peer_copy_reject_sys"),
            "peer_copy_ops": status_json.get("peer_copy_ops"),
            "peer_copy_bytes": status_json.get("peer_copy_bytes"),
            "peer_copy_nv1_ops": status_json.get("peer_copy_nv1_ops"),
            "peer_copy_nv1_bytes": status_json.get("peer_copy_nv1_bytes"),
            "peer_copy_nv2_ops": status_json.get("peer_copy_nv2_ops"),
            "peer_copy_nv2_bytes": status_json.get("peer_copy_nv2_bytes"),
            "peer_copy_sys_ops": status_json.get("peer_copy_sys_ops"),
            "peer_copy_sys_bytes": status_json.get("peer_copy_sys_bytes"),
            "peer_copy_unknown_ops": status_json.get("peer_copy_unknown_ops"),
            "peer_copy_unknown_bytes": status_json.get("peer_copy_unknown_bytes"),
            "peer_copy_first_sys_src": status_json.get("peer_copy_first_sys_src"),
            "peer_copy_first_sys_dst": status_json.get("peer_copy_first_sys_dst"),
            "peer_copy_first_sys_bytes": status_json.get("peer_copy_first_sys_bytes"),
            "peer_copy_first_sys_site": status_json.get("peer_copy_first_sys_site"),
            "peer_copy_first_sys_line": status_json.get("peer_copy_first_sys_line"),
            "peer_copy_top_sys_site": status_json.get("peer_copy_top_sys_site"),
            "peer_copy_top_sys_site_line": status_json.get("peer_copy_top_sys_site_line"),
            "peer_copy_top_sys_site_ops": status_json.get("peer_copy_top_sys_site_ops"),
            "peer_copy_top_sys_site_bytes": status_json.get("peer_copy_top_sys_site_bytes"),
            "peer_copy_top_sys_site_total_ops": status_json.get("peer_copy_top_sys_site_total_ops"),
            "peer_copy_top_sys_site_total_bytes": status_json.get("peer_copy_top_sys_site_total_bytes"),
            "typed_raw_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_raw", server_text)),
            "typed_compressed_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_compressed", server_text)),
            "typed_indexer_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_indexer", server_text)),
            "typed_history_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_history", server_text)),
        }
        add_tp_ep_line_summaries(summary, server_text)
        add_decode_domain_summary(summary)
        summary.update(
            summarize_gpu_samples(
                case_dir / "gpu_util.csv",
                case_dir / "lifecycle.csv",
                case_dir / "gpu_timeline.csv",
            )
        )
        summary.update(summarize_nccl_artifacts(case_dir))
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        lifecycle.mark("summary_written")
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        gpu_sampler.__exit__(None, None, None)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        if args.kill_stale_server:
            cleanup_managed_server_port(args.port, case_dir, "finally")
        server_out.close()
        server_err.close()

    write_top_kernels(case_dir)


if __name__ == "__main__":
    main()
