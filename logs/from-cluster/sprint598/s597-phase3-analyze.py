#!/usr/bin/env python3
"""Sprint 597 Phase 3 analyzer: assemble the EP sub-stage decomposition from
flag-on server logs (tp_ep_ep_stage_profile / tp_ep_ep_stage_routes lines).

Usage: s597-phase3-analyze.py SERVER_LOG [--mode replay_cache_hit] [--label X]

Outputs:
  - per-stage decomposition table (mean ms per layer-step, mean/max over
    ranks, per-step totals x43 layers), split by mode (capture vs replay).
  - residual closure: named-stage coverage of the per-rank ep_window
    (pct field of the synthetic ep_window stage), per layer-class and
    overall; residual = other/overlap.
  - route-skew distribution from tp_ep_ep_stage_routes (p50/p95/max
    per-rank routes, zero-route rank occurrences).
  - replay totals (tp_ep_decode_cudagraph_persistent replay_ms) for the
    rank-local vs critical-path comparison.
"""
import sys
from collections import defaultdict


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[i]


def main():
    path = sys.argv[1]
    label = path
    stage_ms = defaultdict(list)    # (mode, stage) -> [ms]
    stage_rank_ms = defaultdict(list)  # (mode, layer, rank, stage) -> [ms]
    window_cov = defaultdict(list)  # (mode) -> [coverage pct]
    window_ms_all = defaultdict(list)  # (mode, layer, rank) -> [window ms]
    routes_per_rank = []            # per (layer-step) list of 8 ints
    zero_route_events = 0
    routes_lines = 0
    replay_ms = defaultdict(list)   # layer -> [ms]
    capture_nodes = {}
    modes = set()

    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("tp_ep_ep_stage_profile\t"):
                p = line.rstrip("\n").split("\t")
                kv = {p[i]: p[i + 1] for i in range(1, len(p) - 1, 2)}
                mode = kv["mode"]
                stage = kv["stage"]
                layer = int(kv["layer"])
                rank = int(kv["rank"])
                ms = float(kv["ms_event"])
                modes.add(mode)
                if stage == "ep_window":
                    window_cov[mode].append(float(kv["pct"]))
                    window_ms_all[(mode, layer, rank)].append(ms)
                else:
                    stage_ms[(mode, stage)].append(ms)
                    stage_rank_ms[(mode, layer, rank, stage)].append(ms)
            elif line.startswith("tp_ep_ep_stage_routes\t"):
                p = line.rstrip("\n").split("\t")
                kv = {p[i]: p[i + 1] for i in range(1, len(p) - 1, 2)}
                vals = [int(x) for x in kv["routes"].split(",")]
                routes_per_rank.append(vals)
                zero_route_events += sum(1 for v in vals if v == 0)
                routes_lines += 1
            elif line.startswith("tp_ep_decode_cudagraph_persistent\t"):
                p = line.rstrip("\n").split("\t")
                kv = {p[i]: p[i + 1] for i in range(1, len(p) - 1, 2)}
                if "replay_ms" in kv:
                    replay_ms[int(kv["layer"])].append(float(kv["replay_ms"]))
                if "nodes" in kv:
                    capture_nodes[int(kv["layer"])] = int(kv["nodes"])

    print(f"== Phase 3 decomposition: {label} ==")
    for mode in sorted(modes):
        sts = sorted(set(s for (m, s) in stage_ms if m == mode))
        n_any = max((len(stage_ms[(mode, s)]) for s in sts), default=0)
        print(f"\n-- mode {mode} (samples per stage ~{n_any}) --")
        print("stage\tmean_ms\tmax_ms\trank_mean_sum_x43_ms")
        total_mean = 0.0
        for s in sts:
            v = stage_ms[(mode, s)]
            mean = sum(v) / len(v)
            total_mean += mean
            print(f"{s}\t{mean:.4f}\t{max(v):.4f}\t{mean*43:.2f}")
        print(f"sum_named_stages_mean\t{total_mean:.4f}\tper_step_x43\t"
              f"{total_mean*43:.2f} ms (rank-mean, rank-local elapsed)")
        if window_cov[mode]:
            cov = sorted(window_cov[mode])
            wins = [x for k, v in window_ms_all.items() if k[0] == mode
                    for x in v]
            print(f"ep_window mean {sum(wins)/len(wins):.4f} ms; named-stage "
                  f"coverage pct: mean {sum(cov)/len(cov):.1f} p5 "
                  f"{pct(cov,0.05):.1f} p50 {pct(cov,0.5):.1f} p95 "
                  f"{pct(cov,0.95):.1f} -> residual(other/overlap) mean "
                  f"{100-sum(cov)/len(cov):.1f}%")
        # per-rank window max (critical path proxy within EP region)
        per_rank_win = defaultdict(list)
        for (m, layer, rank), v in window_ms_all.items():
            if m == mode:
                per_rank_win[rank] += v
        if per_rank_win:
            print("per-rank ep_window mean ms: " + " ".join(
                f"r{r}:{sum(v)/len(v):.3f}" for r, v in
                sorted(per_rank_win.items())))

    if replay_ms:
        all_replays = [x for v in replay_ms.values() for x in v]
        print(f"\nlayer replay_ms (cache-hit): mean "
              f"{sum(all_replays)/len(all_replays):.4f} over "
              f"{len(all_replays)} layer-replays; nodes per layer graph: "
              f"{sorted(set(capture_nodes.values()))}")

    if routes_per_rank:
        flat = sorted(x for vals in routes_per_rank for x in vals)
        per_step_max = sorted(max(v) for v in routes_per_rank)
        print(f"\nroute-skew over {routes_lines} layer-steps "
              f"(per-rank actual routes, capacity 192):")
        print(f"  per-rank routes: p50 {pct(flat,0.5)} p95 {pct(flat,0.95)} "
              f"max {flat[-1]}; zero-route rank occurrences "
              f"{zero_route_events} ({100.0*zero_route_events/len(flat):.1f}%)")
        print(f"  per-layer-step max-rank routes: p50 {pct(per_step_max,0.5)} "
              f"p95 {pct(per_step_max,0.95)} max {per_step_max[-1]}")


if __name__ == "__main__":
    main()
