#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


def http_get(base, path, timeout=10):
    req = urllib.request.Request(base + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_post(base, path, payload, timeout=900):
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


def run_case(args, name, port, typed, skip_stores=None, typed_quiet=False,
             batch_rows=False, stream_sync=False):
    skip_stores = set(skip_stores or [])
    case_dir = args.artifact_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

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
            "DS4_V100_MAX_REQUESTS": str(args.max_requests),
            "DS4_V100_TP_EP_HC_PERSIST_STATE": "1",
            "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD": "1",
            "DS4_V100_RESERVE_MIB": "0",
            "DS4_V100_PORT": str(port),
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY": "1" if typed else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD":
                "1" if typed and (args.typed_skip_current_load or skip_stores) else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE":
                "1" if "raw" in skip_stores else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE":
                "1" if "compressed" in skip_stores else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE":
                "1" if "indexer" in skip_stores else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET":
                "1" if typed_quiet else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS":
                "1" if batch_rows else "0",
            "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC":
                "1" if stream_sync else "0",
        }
    )

    with open(case_dir / "command.txt", "wb") as out:
        subprocess.run(
            ["./tools/ds4-v100-run-tp-ep-appliance.sh", "--print-command"],
            cwd=args.repo_dir,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )

    server_out = open(case_dir / "server.out", "wb")
    server_err = open(case_dir / "server.err", "wb")
    proc = subprocess.Popen(
        ["./tools/ds4-v100-run-tp-ep-appliance.sh"],
        cwd=args.repo_dir,
        env=env,
        stdout=server_out,
        stderr=server_err,
        preexec_fn=os.setsid,
    )
    (case_dir / "server.pid").write_text(f"{proc.pid}\n")
    base = f"http://127.0.0.1:{port}"

    try:
        for _ in range(args.readiness_seconds):
            if proc.poll() is not None:
                raise RuntimeError(f"{name} server exited rc={proc.returncode}")
            try:
                _, body = http_get(base, "/health", timeout=2)
                (case_dir / "health.json").write_bytes(body)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"{name} readiness timeout")

        payloads = []
        for i in range(args.requests):
            payloads.append(
                {
                    "model": "ds4-v100-tp-ep-diagnostic",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Say one short sentence about TP serving case {i}.",
                        }
                    ],
                    "max_tokens": args.tokens,
                    "session_id": f"s338-{name}-{i:02d}",
                }
            )

        started = time.time()
        results = []

        def one(item):
            i, payload = item
            status, body = http_post(base, "/v1/chat/completions", payload, timeout=900)
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
        try:
            _, body = http_get(base, "/v100/slots", timeout=30)
            (case_dir / "slots.json").write_bytes(body)
        except Exception as exc:
            (case_dir / "slots-error.txt").write_text(f"{exc}\n")

        ok = sum(1 for status, _ in results if status == 200)
        metas = []
        for status, body in results:
            if status == 200:
                metas.append(json.loads(body.decode("utf-8", errors="replace")).get("ds4_v100", {}))
        first = metas[0] if metas else {}
        timing = first.get("timing_ms", {})
        server_text = (case_dir / "server.out").read_text(errors="replace")
        summary = {
            "case": name,
            "typed_history": typed,
            "http_200": ok,
            "requests": len(results),
            "elapsed_s": elapsed_s,
            "client_generated_tok_s": (ok * args.tokens) / elapsed_s if elapsed_s > 0 else 0.0,
            "server_generated_tok_s": timing.get("generated_tokens_per_second", 0.0),
            "server_generated_tok_s_decode": timing.get("generated_tokens_per_second_decode", 0.0),
            "server_continuation_tok_s": timing.get("continuation_tokens_per_second", 0.0),
            "server_continuation_tok_s_decode": timing.get("continuation_tokens_per_second_decode", 0.0),
            "coalesced_batch_size": first.get("coalesced_batch_size"),
            "generated_tokens_meta": first.get("batch_generated_tokens"),
            "cache_hits": status_json.get("cache_hits"),
            "cache_misses": status_json.get("cache_misses"),
            "typed_gate_meta": first.get("true_ds4_attention_typed_kv_history_gate"),
            "typed_skip_current_load_meta": first.get(
                "true_ds4_attention_typed_kv_skip_current_load_gate"
            ),
            "typed_skip_raw_store_meta": first.get(
                "true_ds4_attention_typed_kv_skip_raw_store_gate"
            ),
            "typed_skip_compressed_store_meta": first.get(
                "true_ds4_attention_typed_kv_skip_compressed_store_gate"
            ),
            "typed_skip_indexer_store_meta": first.get(
                "true_ds4_attention_typed_kv_skip_indexer_store_gate"
            ),
            "typed_quiet_meta": first.get("true_ds4_attention_typed_kv_quiet_gate"),
            "typed_batch_rows_meta": first.get(
                "true_ds4_attention_typed_kv_batch_rows_gate"
            ),
            "typed_stream_sync_meta": first.get(
                "true_ds4_attention_typed_kv_stream_sync_gate"
            ),
            "typed_raw_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_raw", server_text)),
            "typed_compressed_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_compressed", server_text)),
            "typed_indexer_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_indexer", server_text)),
            "typed_history_lines": len(re.findall(r"tp_ep_true_attention_typed_kv_history", server_text)),
            "history_loaded_attn_rows_2": len(re.findall(r"loaded_attn_rows\t2", server_text)),
            "history_loaded_indexer_rows_2": len(re.findall(r"loaded_indexer_rows\t2", server_text)),
            "history_reloaded_attn_rows_nonzero": len(re.findall(r"reloaded_attn_rows\t[1-9]", server_text)),
            "history_reloaded_indexer_rows_nonzero": len(re.findall(r"reloaded_indexer_rows\t[1-9]", server_text)),
            "typed_current_store_0": len(re.findall(r"current_store\t0", server_text)),
            "typed_current_store_1": len(re.findall(r"current_store\t1", server_text)),
            "typed_current_load_0": len(re.findall(r"current_load\t0", server_text)),
            "typed_current_load_1": len(re.findall(r"current_load\t1", server_text)),
        }
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        server_out.close()
        server_err.close()
        wait_for_gpu_idle(args.case_cooldown_seconds)


def wait_for_gpu_idle(timeout_s):
    deadline = time.time() + max(0, timeout_s)
    while time.time() < deadline:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            used = [int(x.strip()) for x in proc.stdout.splitlines() if x.strip()]
            if used and max(used) == 0:
                return
        except Exception:
            pass
        time.sleep(1)


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
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--typed-skip-current-load", action="store_true")
    parser.add_argument(
        "--typed-store-variant",
        action="append",
        choices=["baseline", "no-raw", "no-compressed", "no-indexer", "no-stores"],
        default=[],
        help="additional typed candidates for store-family cost isolation",
    )
    parser.add_argument("--control-port", type=int, default=18338)
    parser.add_argument("--typed-port", type=int, default=18339)
    parser.add_argument("--readiness-seconds", type=int, default=240)
    parser.add_argument("--case-cooldown-seconds", type=int, default=20)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument(
        "--typed-quiet-variant",
        action="append",
        choices=["quiet", "no-stores-quiet"],
        default=[],
        help="additional typed candidates that suppress per-layer typed KV PASS logs",
    )
    parser.add_argument(
        "--typed-batch-rows-variant",
        action="append",
        choices=["batch-rows", "batch-rows-quiet"],
        default=[],
        help="additional typed candidates that batch typed KV row operations across slots",
    )
    parser.add_argument(
        "--typed-stream-sync-variant",
        action="append",
        choices=["stream-sync-quiet", "batch-rows-stream-sync-quiet"],
        default=[],
        help="additional typed candidates that narrow typed KV barriers to stream sync",
    )
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    if not args.skip_control:
        summaries.append(run_case(args, "control", args.control_port, False))
    variants = args.typed_store_variant or ["baseline"]
    port = args.typed_port
    for variant in variants:
        if variant == "baseline":
            summaries.append(run_case(args, "typed-history", port, True))
        elif variant == "no-raw":
            summaries.append(run_case(args, "typed-no-raw-store", port, True, {"raw"}))
        elif variant == "no-compressed":
            summaries.append(
                run_case(args, "typed-no-compressed-store", port, True, {"compressed"})
            )
        elif variant == "no-indexer":
            summaries.append(run_case(args, "typed-no-indexer-store", port, True, {"indexer"}))
        elif variant == "no-stores":
            summaries.append(
                run_case(
                    args,
                    "typed-no-stores",
                    port,
                    True,
                    {"raw", "compressed", "indexer"},
                )
            )
        port += 1
    for variant in args.typed_quiet_variant:
        if variant == "quiet":
            summaries.append(run_case(args, "typed-quiet", port, True, typed_quiet=True))
        elif variant == "no-stores-quiet":
            summaries.append(
                run_case(
                    args,
                    "typed-no-stores-quiet",
                    port,
                    True,
                    {"raw", "compressed", "indexer"},
                    typed_quiet=True,
                )
            )
        port += 1
    for variant in args.typed_batch_rows_variant:
        if variant == "batch-rows":
            summaries.append(run_case(args, "typed-batch-rows", port, True, batch_rows=True))
        elif variant == "batch-rows-quiet":
            summaries.append(
                run_case(
                    args,
                    "typed-batch-rows-quiet",
                    port,
                    True,
                    typed_quiet=True,
                    batch_rows=True,
                )
            )
        port += 1
    for variant in args.typed_stream_sync_variant:
        if variant == "stream-sync-quiet":
            summaries.append(
                run_case(
                    args,
                    "typed-stream-sync-quiet",
                    port,
                    True,
                    typed_quiet=True,
                    stream_sync=True,
                )
            )
        elif variant == "batch-rows-stream-sync-quiet":
            summaries.append(
                run_case(
                    args,
                    "typed-batch-rows-stream-sync-quiet",
                    port,
                    True,
                    typed_quiet=True,
                    batch_rows=True,
                    stream_sync=True,
                )
            )
        port += 1
    (args.artifact_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    keys = [
        "case",
        "typed_history",
        "http_200",
        "requests",
        "elapsed_s",
        "client_generated_tok_s",
        "server_generated_tok_s",
        "server_generated_tok_s_decode",
        "server_continuation_tok_s",
        "server_continuation_tok_s_decode",
        "coalesced_batch_size",
        "generated_tokens_meta",
        "cache_hits",
        "cache_misses",
        "typed_gate_meta",
        "typed_skip_current_load_meta",
        "typed_skip_raw_store_meta",
        "typed_skip_compressed_store_meta",
        "typed_skip_indexer_store_meta",
        "typed_quiet_meta",
        "typed_batch_rows_meta",
        "typed_stream_sync_meta",
        "typed_raw_lines",
        "typed_compressed_lines",
        "typed_indexer_lines",
        "typed_history_lines",
        "history_loaded_attn_rows_2",
        "history_loaded_indexer_rows_2",
        "history_reloaded_attn_rows_nonzero",
        "history_reloaded_indexer_rows_nonzero",
        "typed_current_store_0",
        "typed_current_store_1",
        "typed_current_load_0",
        "typed_current_load_1",
    ]
    with open(args.artifact_dir / "summary.tsv", "w") as out:
        out.write("\t".join(keys) + "\n")
        for summary in summaries:
            out.write("\t".join(str(summary.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main()
