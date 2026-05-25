#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


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
    if args.tool == "nvprof-api-trace":
        return [
            args.nvprof,
            "--profile-child-processes",
            "--csv",
            "--print-api-trace",
            "--log-file",
            profiler_log_pattern(case_dir, "nvprof-api-trace"),
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
    raise ValueError(f"unsupported tool {args.tool}")


def build_env(args, port):
    env = os.environ.copy()
    env.update(
        {
            "DS4_V100_SERVE_MODE": "tp-ep",
            "DS4_V100_CTX": str(args.ctx),
            "DS4_V100_SLOTS": str(args.slots),
            "DS4_V100_ACTIVE_MICROBATCH": str(args.slots),
            "DS4_V100_APPLIANCE_DIR": args.pack_dir,
            "DS4_V100_TP_EP_CONTRACT": args.contract,
            "DS4_V100_TURBOMIND_LIB": args.turbomind_lib,
            "DS4_V100_TP_EP_TOKENIZER_MODEL": args.tokenizer_model,
            "DS4_V100_TOKENS": str(args.tokens),
            "DS4_V100_MAX_REQUESTS": str(max(args.max_requests, args.requests)),
            "DS4_V100_TP_EP_HC_PERSIST_STATE": "1",
            "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD": "1",
            "DS4_V100_RESERVE_MIB": "0",
            "DS4_V100_PORT": str(port),
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC": "1",
        }
    )
    return env


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
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--port", type=int, default=18357)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--nvprof", default="/usr/local/cuda/bin/nvprof")
    parser.add_argument("--ncu-launch-count", type=int, default=160)
    parser.add_argument("--ncu-launch-skip", type=int, default=0)
    parser.add_argument(
        "--ncu-kernel-name",
        default="",
        help="optional Nsight Compute --kernel-name filter, e.g. regex:.*cutlass_70_wmma.*",
    )
    parser.add_argument(
        "--tool",
        choices=["none", "nvprof-gpu-trace", "nvprof-api-trace", "ncu-basic", "ncu-nvlink"],
        default="nvprof-gpu-trace",
    )
    args = parser.parse_args()

    case_dir = args.artifact_dir / args.tool
    case_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args, args.port)
    base = f"http://127.0.0.1:{args.port}"

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
    server_out = open(case_dir / "server.out", "wb")
    server_err = open(case_dir / "server.err", "wb")
    proc = subprocess.Popen(
        cmd,
        cwd=args.repo_dir,
        env=env,
        stdout=server_out,
        stderr=server_err,
        preexec_fn=os.setsid,
    )
    (case_dir / "server.pid").write_text(f"{proc.pid}\n")

    try:
        for _ in range(args.readiness_seconds):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                _, body = http_get(base, "/health", timeout=2)
                (case_dir / "health.json").write_bytes(body)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("readiness timeout")

        payloads = [
            {
                "model": "ds4-v100-tp-ep-diagnostic",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Say one short sentence about TP Nsight profile {i}.",
                    }
                ],
                "max_tokens": args.tokens,
                "session_id": f"s345-profile-{args.tool}-{i:02d}",
            }
            for i in range(args.requests)
        ]

        started = time.time()
        results = []

        def one(item):
            i, payload = item
            status, body = http_post(
                base,
                "/v1/chat/completions",
                payload,
                timeout=args.request_timeout_seconds,
            )
            (case_dir / f"response-{i:02d}.txt").write_text(
                body.decode(errors="replace") + f"\nHTTP_STATUS:{status}\n"
            )
            return status, body

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as executor:
            for result in executor.map(one, enumerate(payloads)):
                results.append(result)

        elapsed_s = time.time() - started
        _, body = http_get(base, "/status", timeout=30)
        (case_dir / "status.json").write_bytes(body)
        status_json = json.loads(body)
        _, body = http_get(base, "/metrics", timeout=30)
        (case_dir / "metrics.txt").write_bytes(body)

        ok = sum(1 for status, _ in results if status == 200)
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
            "http_200": ok,
            "requests": len(results),
            "tokens": args.tokens,
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
            "typed_raw_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_raw", server_text)),
            "typed_compressed_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_compressed", server_text)),
            "typed_indexer_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_indexer", server_text)),
            "typed_history_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_history", server_text)),
        }
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        server_out.close()
        server_err.close()

    write_top_kernels(case_dir)


if __name__ == "__main__":
    main()
