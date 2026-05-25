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
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER": "1" if args.hc_current_peer_gather else "0",
            "DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC": "1" if args.hc_current_stream_sync else "0",
            "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD": "1",
            "DS4_V100_RESERVE_MIB": "0",
            "DS4_V100_PORT": str(port),
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS": "1",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC": "1",
            "DS4_V100_CUDA_PROFILER_WINDOW": "1" if "window" in args.tool else "0",
        }
    )
    return env


def variant_suffix(args):
    suffix = ""
    if args.hc_current_peer_gather:
        suffix += "-hc-peer-gather"
    if args.hc_current_stream_sync:
        suffix += "-hc-stream-sync"
    return suffix


def direct_command(args):
    cmd = [
        "./tools/ds4-v100-tp-ep-full-layer-smoke",
        "--pack-dir", args.pack_dir,
        "--contract", args.contract,
        "--tm-index", str(pathlib.Path(args.pack_dir) / "turbomind-pack-index.tsv"),
        "--lib", args.turbomind_lib,
        "--slots", str(args.slots),
        "--top-k", "6",
        "--kv-slot", "7",
        "--position", "100000",
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
        "--compact-route-compose",
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
    if "window" in args.tool:
        cmd.append("--cuda-profiler-window")
    if args.hc_current_peer_gather:
        cmd.append("--tp-hc-current-input-peer-gather-gate")
    if args.hc_current_stream_sync:
        cmd.append("--tp-hc-current-input-stream-sync-gate")
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


def summarize_direct(case_dir, tool, rc, elapsed_s):
    stdout = (case_dir / "stdout.txt").read_text(errors="replace")
    stderr = (case_dir / "stderr.txt").read_text(errors="replace")
    summary = {
        "tool": tool,
        "run_mode": "direct-token-major",
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "profiler_marker_lines": len(re.findall(r"tp_ep_cuda_profiler_window", stderr)),
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
                "tp_hc_current_input_stream_sync",
                "sum_final_hc_ms",
                "wall_ms",
            ]:
                summary[f"scaffold_{key}"] = maybe_number(fields.get(key))
        elif tag == "tp_ep_diagnostic_output_head":
            for key in ["total_ms", "projection_ms", "top1_ms", "first_token", "finite_bad"]:
                summary[f"output_head_{key}"] = maybe_number(fields.get(key))
    return summary


def run_direct_case(args):
    case_name = f"{args.tool}-direct{variant_suffix(args)}"
    case_dir = args.artifact_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    cmd = profiler_prefix(args, case_dir) + direct_command(args)
    (case_dir / "profile-command.txt").write_text(" ".join(cmd) + "\n")
    (case_dir / "command.txt").write_text(" ".join(direct_command(args)) + "\n")
    started = time.time()
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
    elapsed_s = time.time() - started
    summary = summarize_direct(case_dir, args.tool, proc.returncode, elapsed_s)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
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
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--hc-current-peer-gather", action="store_true")
    parser.add_argument("--hc-current-stream-sync", action="store_true")
    parser.add_argument("--port", type=int, default=18357)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--run-mode",
        choices=["http", "direct-token-major"],
        default="http",
    )
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
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "direct-token-major":
        raise SystemExit(run_direct_case(args))

    case_dir = args.artifact_dir / f"{args.tool}{variant_suffix(args)}"
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
