#!/usr/bin/env python3
"""Sprint 597 Phase 3a (authority leg): full-stage kernel-time mapping of the
unmodified promoted full-capture serving window (nsys sqlite from Phase 1).

Bins every kernel in the window into the sprint stage list by name (+ grid
for the EP-return copies), reports rank-summed and per-rank ms per layer-step
(window = 43 layers x 8 steps = 344 layer-steps), plus the window wall time.
"""
import sqlite3
import sys
from collections import defaultdict

LAYER_STEPS = 43 * 8

STAGES = [
    ("ep_return_copy_384", lambda n, g: n == "copy_f32_kernel" and g == 384),
    ("route_plan", lambda n, g: n in (
        "router_logits_allreduce_partial_kernel", "router_select_topk_rows_kernel",
        "gpu_route_count_all_kernel", "gpu_route_prefix_all_kernel",
        "gpu_route_init_compact_plan_kernel", "gpu_route_copy_own_offsets_kernel",
        "gpu_route_fill_all_kernel", "copy_i32_kernel")),
    ("routed_input_pack", lambda n, g: n in (
        "pack_rank_major_norm_current_to_routes_kernel",
        "pack_current_full_to_routes_kernel",
        "fill_two_hidden_inputs_half_from_rank_major_norm_kernel",
        "fill_dense_input_half_from_current_kernel",
        "fill_dense_input_from_current_kernel")),
    ("expert_gemm_turbomind", lambda n, g: n in ("gemm_kernel", "Kernel")
        or "turbomind" in n.lower() or "cutlass" in n.lower()),
    ("gate_up_swiglu_epilogue", lambda n, g:
        n == "routed_fused_gate_up_swiglu_clamp_kernel"),
    ("contrib_pack", lambda n, g: n == "ep_pack_route_dest_shards_kernel"),
    ("compose", lambda n, g: n.startswith("compose_next_hidden")),
    ("dense_f8_bf16", lambda n, g: n in (
        "f8_b128_dense_kernel", "bf16_dense_kernel", "f32_dense_colmajor_kernel",
        "splitKreduce_kernel") or "dense_kernel" in n),
    ("nccl", lambda n, g: n.startswith("ncclDevKernel")),
    ("hc_current", lambda n, g: n.startswith("hc_") or "current" in n
        or n.startswith("gather_hc") or n.startswith("rms_norm")),
    ("attention_kv", lambda n, g: "attention" in n or n.startswith("kv_")
        or "rope" in n or "head" in n or n.startswith("store_f32")
        or "swa" in n or "indexer" in n),
    ("other_copies", lambda n, g: n == "copy_f32_kernel"),
]


def classify(name, grid):
    for stage, fn in STAGES:
        if fn(name, grid):
            return stage
    return "other"


def main():
    db = sqlite3.connect(sys.argv[1])
    cur = db.cursor()
    cur.execute("""
        SELECT k.deviceId, k.gridX, k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
    """)
    per_stage = defaultdict(float)       # stage -> total ms (all ranks)
    per_stage_rank = defaultdict(float)  # (stage, dev) -> ms
    t0, t1 = None, None
    n = 0
    for dev, grid, start, end, name in cur:
        n += 1
        ms = (end - start) / 1e6
        st = classify(name, grid)
        per_stage[st] += ms
        per_stage_rank[(st, dev)] += ms
        t0 = start if t0 is None else min(t0, start)
        t1 = end if t1 is None else max(t1, end)

    wall_ms = (t1 - t0) / 1e6
    print(f"kernels {n}; window wall {wall_ms:.1f} ms over {LAYER_STEPS} "
          f"layer-steps -> {wall_ms/LAYER_STEPS:.3f} ms wall per layer-step "
          f"(x8 ranks busy budget {8*wall_ms/LAYER_STEPS:.2f} ms)")
    print("stage\tsum_ms_all_ranks\tms_per_layer_step_all_ranks\t"
          "ms_per_layer_step_per_rank_mean")
    tot = 0.0
    for st in sorted(per_stage, key=lambda s: -per_stage[s]):
        v = per_stage[st]
        tot += v
        print(f"{st}\t{v:.1f}\t{v/LAYER_STEPS:.4f}\t{v/LAYER_STEPS/8:.4f}")
    print(f"total_busy\t{tot:.1f}\t{tot/LAYER_STEPS:.4f}\t"
          f"{tot/LAYER_STEPS/8:.4f}")
    print(f"idle_share_of_window {100*(1-tot/(8*wall_ms)):.1f}% "
          f"(per-rank stream idle/wait incl. barriers)")
    print("\nper-rank ep_return_copy ms/layer-step: " + " ".join(
        f"r{d}:{per_stage_rank[('ep_return_copy_384', d)]/LAYER_STEPS:.3f}"
        for d in range(8)))


if __name__ == "__main__":
    main()
