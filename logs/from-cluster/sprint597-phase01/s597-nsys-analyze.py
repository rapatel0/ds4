#!/usr/bin/env python3
"""Sprint 597 Phase 1.3: attribute per-(dst,src) EP-return copy kernels from
the nsys sqlite export of one steady-state full-capture serving window.

The promoted EP return launches, per layer, on each dst rank's stream, one
copy_f32_kernel per src in ascending src order skipping src==dst
(engine/decode_loop.cu:1176-1195). At the fixed-capacity route plan the
payload is 192x512 f32 -> grid 384 blocks of 256 threads, which uniquely
identifies these kernels in the trace (other copy_f32 uses have different
grids). Stream order serializes the 7 copies, so within each consecutive
group of 7 on a device, position k maps to the k-th src in [0..7]\\{dst}.

Usage: s597-nsys-analyze.py nsys-insitu.sqlite [topo.txt]
"""
import re
import sqlite3
import sys
from collections import defaultdict


def parse_topo(path):
    cls = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^GPU(\d)\s", line)
            if not m:
                continue
            i = int(m.group(1))
            cols = line.split()
            for j, c in enumerate(cols[1:9]):
                cls[(i, j)] = c.strip()
    return cls


def main():
    db = sqlite3.connect(sys.argv[1])
    cls = parse_topo(sys.argv[2]) if len(sys.argv) > 2 else None
    cur = db.cursor()
    cur.execute("""
        SELECT k.deviceId, k.start, k.end, k.gridX, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE '%copy_f32%' AND k.gridX = 384
        ORDER BY k.deviceId, k.start
    """)
    rows = cur.fetchall()
    print(f"ep_return_copy_kernels\t{len(rows)}")
    per_dev = defaultdict(list)
    for dev, start, end, gridx, name in rows:
        per_dev[dev].append((start, end))

    pair_durs = defaultdict(list)
    for dst, lst in per_dev.items():
        srcs = [s for s in range(8) if s != dst]
        n_groups = len(lst) // 7
        for g in range(n_groups):
            grp = lst[g * 7:(g + 1) * 7]
            for k, (start, end) in enumerate(grp):
                pair_durs[(dst, srcs[k])].append((end - start) / 1000.0)  # us

    print("dst\tsrc\tclass\tn\tmean_us\tmin_us\tmax_us")
    cstats = defaultdict(list)
    for (dst, src) in sorted(pair_durs):
        d = pair_durs[(dst, src)]
        c = cls[(dst, src)] if cls else "?"
        mean = sum(d) / len(d)
        print(f"{dst}\t{src}\t{c}\t{len(d)}\t{mean:.2f}\t{min(d):.2f}\t{max(d):.2f}")
        cstats[c].append(mean)
    if cls:
        print("\nclass summary (mean per-pair us):")
        for c, v in sorted(cstats.items()):
            print(f"  {c}: n_pairs={len(v)} mean={sum(v)/len(v):.2f} "
                  f"min={min(v):.2f} max={max(v):.2f}")
        # per-layer cost: each dst runs its 7 copies serially on its stream;
        # critical path = worst dst serial sum
        per_dst_sum = defaultdict(float)
        for (dst, src), v in pair_durs.items():
            per_dst_sum[dst] += sum(v) / len(v)
        worst_dst = max(per_dst_sum, key=per_dst_sum.get)
        print("\nper-dst serial EP-return us/layer (mean):")
        for dst in sorted(per_dst_sum):
            print(f"  dst {dst}: {per_dst_sum[dst]:.1f}")
        w = per_dst_sum[worst_dst]
        print(f"worst dst {worst_dst}: {w:.1f} us/layer -> x43 layers = "
              f"{w*43/1000.0:.2f} ms/step on the EP-return critical path")


if __name__ == "__main__":
    main()
