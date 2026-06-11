#!/usr/bin/env python3
"""Sprint 597 Phase 0 analyzer.

Parses a ds4-v100-tp-ep-http-bench.sh case dir:
  - responses.json -> per-coalesced-batch decode/wall timing, aggregate decode
    tok/s including and excluding the first (capture/warmup) batch.
  - server.log tp_ep_token_major_item lines -> eager per-layer-step
    decode-domain attribution table (Sprint 581 buckets), steady-state window
    (first batch excluded).

Usage: s597-phase0-analyze.py CASE_DIR [--label NAME]
"""
import json
import os
import sys
from collections import defaultdict


def parse_responses(case_dir):
    with open(os.path.join(case_dir, "responses.json")) as f:
        responses = json.load(f)
    metas = [r.get("ds4_v100", r) for r in responses]
    batches = {}
    per_request = []
    for m in metas:
        bid = m.get("coalesced_batch_id")
        t = m["timing_ms"]
        batches[bid] = {
            "batch_id": bid,
            "batch_size": m.get("coalesced_batch_size"),
            "total_decode_ms": t["total_decode"],
            "total_wall_ms": t["total_wall"],
            "continuation_decode_ms": t["continuation_decode"],
            "continuation_wall_ms": t["continuation_wall"],
            "ep_ms": t.get("ep", 0.0),
            "dense_ms": t.get("dense", 0.0),
            "compose_ms": t.get("compose", 0.0),
            "generated_tokens_batch": m.get("batch_generated_tokens"),
        }
        per_request.append({
            "generated_tokens": m["generated_tokens"],
            "continuation_tokens": m["continuation_tokens"],
            "batch_id": bid,
            "gen_tok_s_decode": t.get("generated_tokens_per_second_decode", 0.0),
            "cont_tok_s_decode": t.get("continuation_tokens_per_second_decode", 0.0),
            "gen_tok_s_wall": t.get("generated_tokens_per_second", 0.0),
        })
    return batches, per_request, metas


def batch_aggregates(batches, per_request, skip_first=0):
    order = sorted(batches.keys())
    use = order[skip_first:]
    gen = sum(b["generated_tokens_batch"] or 0 for bid, b in batches.items()
              if bid in use)
    if not gen:
        gen = sum(r["generated_tokens"] for r in per_request
                  if r["batch_id"] in use)
    decode_ms = sum(batches[b]["total_decode_ms"] for b in use)
    wall_ms = sum(batches[b]["total_wall_ms"] for b in use)
    return {
        "batches": len(use),
        "generated_tokens": gen,
        "decode_ms": decode_ms,
        "wall_ms": wall_ms,
        "agg_tok_s_decode": gen * 1000.0 / decode_ms if decode_ms else 0.0,
        "agg_tok_s_wall": gen * 1000.0 / wall_ms if wall_ms else 0.0,
    }


STAGE_FIELDS = [
    "decode_ms_per_step",
    "decode_ep_ms_per_step",
    "decode_dense_ms_per_step",
    "decode_compose_ms_per_step",
    "decode_compose_reduce_ms_per_step",
    "decode_compose_copy_ms_per_step",
    "decode_compose_final_ms_per_step",
    "decode_hc_current_input_ms_per_step",
    "decode_hc_current_router_select_ms_per_step",
    "decode_hc_current_router_d2h_ms_per_step",
    "decode_hc_current_route_upload_ms_per_step",
    "decode_hc_current_fill_pack_ms_per_step",
    "decode_pre_ep_attention_projection_ms_per_step",
    "decode_pre_ep_compressed_kv_ms_per_step",
    "decode_pre_ep_attention_state_ms_per_step",
    "decode_pre_ep_typed_history_ms_per_step",
    "decode_pre_ep_raw_read_ms_per_step",
    "decode_pre_ep_attention_output_ms_per_step",
    "decode_pre_ep_post_attention_ffn_input_ms_per_step",
    "decode_final_hc_ms_per_step",
    "decode_cudagraph_replay_attempted",
    "decode_cudagraph_replay_succeeded",
    "decode_cudagraph_persistent_cache_hits",
    "decode_cudagraph_persistent_cache_misses",
]


def parse_token_major(case_dir):
    path = os.path.join(case_dir, "server.log")
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("tp_ep_token_major_item\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            kv = {}
            i = 1
            while i + 1 < len(parts):
                kv[parts[i]] = parts[i + 1]
                i += 2
            row = {"layer": int(kv.get("layer", -1)),
                   "position": int(kv.get("position", -1))}
            for fld in STAGE_FIELDS:
                if fld in kv:
                    row[fld] = float(kv[fld])
            rows.append(row)
    return rows


def split_batches(rows):
    """Group rows into serving batches: position resets to min mark new batch."""
    if not rows:
        return []
    base = min(r["position"] for r in rows)
    batches = []
    cur = None
    prev_pos = None
    for r in rows:
        if r["position"] == base and (prev_pos is None or prev_pos != base):
            cur = []
            batches.append(cur)
        if cur is None:
            cur = []
            batches.append(cur)
        cur.append(r)
        prev_pos = r["position"]
    return batches


def attribution(rows, layer_max=42):
    """Mean per-layer-step stage ms across rows (layers 0..layer_max)."""
    use = [r for r in rows if 0 <= r["layer"] <= layer_max]
    n = len(use)
    if not n:
        return None
    mean = {f: sum(r.get(f, 0.0) for r in use) / n for f in STAGE_FIELDS}
    total = mean["decode_ms_per_step"]
    attn = sum(mean[f] for f in [
        "decode_pre_ep_attention_projection_ms_per_step",
        "decode_pre_ep_compressed_kv_ms_per_step",
        "decode_pre_ep_attention_state_ms_per_step",
        "decode_pre_ep_typed_history_ms_per_step",
        "decode_pre_ep_raw_read_ms_per_step",
        "decode_pre_ep_attention_output_ms_per_step",
    ])
    host_sync = sum(mean[f] for f in [
        "decode_hc_current_route_upload_ms_per_step",
        "decode_hc_current_fill_pack_ms_per_step",
        "decode_hc_current_router_select_ms_per_step",
    ])
    table = [
        ("EP (MoE all-to-all)", mean["decode_ep_ms_per_step"]),
        ("attention (proj+kv+state+hist+raw+output)", attn),
        ("compose (+reduce/copy/final)", mean["decode_compose_ms_per_step"]),
        ("HC-current input", mean["decode_hc_current_input_ms_per_step"]),
        ("final_hc", mean["decode_final_hc_ms_per_step"]),
        ("host-sync (route_upload+fill_pack+router_select)", host_sync),
    ]
    return {"n_rows": n, "total_ms": total, "table": table, "mean": mean}


def main():
    case_dir = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else case_dir
    batches, per_request, _ = parse_responses(case_dir)
    print(f"== {label} ==")
    print("batch_id\tbatch_size\ttotal_decode_ms\ttotal_wall_ms\tep_ms\tdense_ms\tcompose_ms")
    for bid in sorted(batches):
        b = batches[bid]
        print(f"{bid}\t{b['batch_size']}\t{b['total_decode_ms']:.3f}\t"
              f"{b['total_wall_ms']:.3f}\t{b['ep_ms']:.3f}\t{b['dense_ms']:.3f}\t{b['compose_ms']:.3f}")
    allb = batch_aggregates(batches, per_request, 0)
    steady = batch_aggregates(batches, per_request, 1)
    print(f"all_batches\tn={allb['batches']}\tgen={allb['generated_tokens']}\t"
          f"agg_tok_s_decode={allb['agg_tok_s_decode']:.3f}\tagg_tok_s_wall={allb['agg_tok_s_wall']:.3f}")
    print(f"steady_state(excl batch 1)\tn={steady['batches']}\tgen={steady['generated_tokens']}\t"
          f"agg_tok_s_decode={steady['agg_tok_s_decode']:.3f}\tagg_tok_s_wall={steady['agg_tok_s_wall']:.3f}")
    pr = [r for r in per_request if r["batch_id"] != min(batches.keys())]
    if pr:
        mean_gen = sum(r["gen_tok_s_decode"] for r in pr) / len(pr)
        mean_wall = sum(r["gen_tok_s_wall"] for r in pr) / len(pr)
        print(f"per_request_decode_tok_s_mean(steady)\t{mean_gen:.3f}")
        print(f"per_request_wall_tok_s_mean(steady)\t{mean_wall:.3f}")

    rows = parse_token_major(case_dir)
    if rows:
        tb = split_batches(rows)
        print(f"token_major_rows\t{len(rows)}\tbatches_detected\t{len(tb)}")
        steady_rows = [r for grp in tb[1:] for r in grp] if len(tb) > 1 else rows
        att = attribution(steady_rows)
        if att and att["total_ms"] > 0:
            print(f"eager_attribution(steady, n={att['n_rows']} layer-steps, "
                  f"total {att['total_ms']:.3f} ms/layer-step):")
            for name, ms in att["table"]:
                pct = 100.0 * ms / att["total_ms"]
                print(f"  {name}\t{ms:.3f}\t{pct:.1f}%")
            m = att["mean"]
            print(f"  [graph counters] replay_attempted={m['decode_cudagraph_replay_attempted']:.2f} "
                  f"replay_succeeded={m['decode_cudagraph_replay_succeeded']:.2f} "
                  f"cache_hits={m['decode_cudagraph_persistent_cache_hits']:.2f}")
        first_att = attribution(tb[0]) if tb else None
        if first_att and first_att["total_ms"] > 0:
            print(f"first_batch_total_ms_per_layer_step\t{first_att['total_ms']:.3f}")
    else:
        print("token_major_rows\t0 (graph mode or per-stage timers inactive)")


if __name__ == "__main__":
    main()
