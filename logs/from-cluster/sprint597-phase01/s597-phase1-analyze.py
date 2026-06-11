#!/usr/bin/env python3
"""Sprint 597 Phase 1 analyzer.

Inputs:
  - nvidia-smi-topo.txt (archived `nvidia-smi topo -m`)
  - peer-copy-microbench.tsv (s597-peer-copy-microbench output)

Outputs (stdout):
  - NVLink adjacency per GPU, the non-NVLink undirected pairs, and candidate
    one-hop NVLink relays per non-adjacent pair.
  - Per-pair measured bandwidth/latency table at each payload, with
    topo-derived class (NV1/NV2/SYS/self) and measured class agreement.
  - SYS-exposure cost model for the promoted EP return (56 copies/layer,
    fixed-capacity 192 routes x 512 f32 = 384 KiB per pair).
"""
import re
import sys
from collections import defaultdict

LAYERS = 43


def parse_topo(path):
    cls = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^GPU(\d)\s", line)
            if not m:
                continue
            i = int(m.group(1))
            cols = line.split()
            # cols[0] = GPUi, then 8 link columns
            for j, c in enumerate(cols[1:9]):
                cls[(i, j)] = c.strip()
    return cls


def main():
    topo_path, bench_path = sys.argv[1], sys.argv[2]
    cls = parse_topo(topo_path)

    adj = {i: sorted(j for j in range(8)
                     if j != i and cls[(i, j)].startswith("NV"))
           for i in range(8)}
    print("== NVLink adjacency (per GPU) ==")
    for i in range(8):
        links = ", ".join(f"{j}({cls[(i, j)]})" for j in adj[i])
        print(f"GPU{i}: {links}")

    non_nv = [(i, j) for i in range(8) for j in range(i + 1, 8)
              if not cls[(i, j)].startswith("NV")]
    print(f"\n== Non-NVLink undirected pairs ({len(non_nv)} of 28) ==")
    for (i, j) in non_nv:
        relays = sorted(set(adj[i]) & set(adj[j]))
        rl = ", ".join(
            f"{r}({cls[(i, r)]}+{cls[(r, j)]})" for r in relays)
        print(f"({i},{j}) [{cls[(i, j)]}]: one-hop NVLink relays via {rl}")

    rows = []
    with open(bench_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0].isdigit():
                continue
            rows.append({
                "bytes": int(parts[0]), "dst": int(parts[1]),
                "src": int(parts[2]), "same": int(parts[3]),
                "burst_us": float(parts[7]), "burst_gbps": float(parts[8]),
                "iso_us": float(parts[10]), "iso_gbps": float(parts[11]),
            })

    sizes = sorted(set(r["bytes"] for r in rows))
    print("\n== Per-pair microbench (dst-side UVA remote load) ==")
    for b in sizes:
        print(f"\n-- payload {b//1024} KiB --")
        print("dst\tsrc\tclass\tburst_us\tburst_GBps\tiso_us\tiso_GBps")
        cstats = defaultdict(list)
        for r in [r for r in rows if r["bytes"] == b]:
            c = "self" if r["same"] else cls[(r["dst"], r["src"])]
            print(f"{r['dst']}\t{r['src']}\t{c}\t{r['burst_us']:.2f}\t"
                  f"{r['burst_gbps']:.2f}\t{r['iso_us']:.2f}\t{r['iso_gbps']:.2f}")
            cstats[c].append(r)
        print("class summary (burst): " + "; ".join(
            f"{c}: n={len(v)} "
            f"us[{min(x['burst_us'] for x in v):.2f}-{max(x['burst_us'] for x in v):.2f}] "
            f"GBps[{min(x['burst_gbps'] for x in v):.2f}-{max(x['burst_gbps'] for x in v):.2f}]"
            for c, v in sorted(cstats.items())))

    # SYS exposure cost model at the promoted payload (384 KiB).
    promoted = [r for r in rows if r["bytes"] == 384 * 1024 and not r["same"]]
    if promoted:
        print("\n== Promoted EP-return cost model (384 KiB per pair, 56 copies/layer) ==")
        per_dst = defaultdict(lambda: {"nv_us": 0.0, "sys_us": 0.0, "n_sys": 0})
        for r in promoted:
            c = cls[(r["dst"], r["src"])]
            d = per_dst[r["dst"]]
            if c.startswith("NV"):
                d["nv_us"] += r["burst_us"]
            else:
                d["sys_us"] += r["burst_us"]
                d["n_sys"] += 1
        print("dst\tn_sys_srcs\tsum_nv_us\tsum_sys_us\tserial_total_us")
        worst = 0.0
        for dst in sorted(per_dst):
            d = per_dst[dst]
            tot = d["nv_us"] + d["sys_us"]
            worst = max(worst, tot)
            print(f"{dst}\t{d['n_sys']}\t{d['nv_us']:.1f}\t{d['sys_us']:.1f}\t{tot:.1f}")
        sys_pairs = sum(1 for r in promoted
                        if not cls[(r["dst"], r["src"])].startswith("NV"))
        sys_us = sum(r["burst_us"] for r in promoted
                     if not cls[(r["dst"], r["src"])].startswith("NV"))
        nv_us_mean = (sum(r["burst_us"] for r in promoted
                          if cls[(r["dst"], r["src"])].startswith("NV")) /
                      max(1, len(promoted) - sys_pairs))
        sys_us_mean = sys_us / max(1, sys_pairs)
        print(f"directed SYS pairs: {sys_pairs}/56; mean per-copy us: "
              f"NV={nv_us_mean:.2f} SYS={sys_us_mean:.2f}")
        # If each dst's 7 copies run serially on its stream, the per-layer
        # EP-return critical path is the worst dst's serial total.
        print(f"per-layer EP-return critical path (worst dst, serial on dst "
              f"stream): {worst:.1f} us; per step (x{LAYERS} layers): "
              f"{worst*LAYERS/1000.0:.2f} ms")
        excess = (sys_us_mean - nv_us_mean) * sys_pairs / 8.0
        print(f"naive SYS excess per layer per dst-avg: {excess:.1f} us -> "
              f"per step: {excess*LAYERS/1000.0:.2f} ms (avg dst)")


if __name__ == "__main__":
    main()
