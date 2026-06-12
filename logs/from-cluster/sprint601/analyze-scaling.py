#!/usr/bin/env python3
"""S601 Phase D: scaling-curve table from sustained_http.tsv rows."""
import sys

# stdin: lines "name<TAB>full tsv row" or use run-summaries format
rows = []
name = None
for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("== "):
        name = line[3:].strip()
        continue
    if not line or line.startswith("endpoint"):
        continue
    f = line.split("\t")
    if len(f) < 16:
        continue
    slots = int(f[4])
    reqs = int(f[5])
    wall = float(f[12])
    decode = float(f[14])
    step_ms = slots / decode * 1000.0
    per_slot = decode / slots
    rows.append((name, slots, reqs, decode, wall, per_slot, step_ms))

rows.sort(key=lambda r: r[1])
print(f"{'run':<10}{'S':>4}{'reqs':>6}{'decode':>10}{'wall':>9}{'tok/s/slot':>12}{'step_ms':>10}{'M_for_50':>10}{'M_for_50@.75':>13}")
for name, slots, reqs, decode, wall, per_slot, step_ms in rows:
    m50 = 50.0 * step_ms / 1000.0
    print(f"{name:<10}{slots:>4}{reqs:>6}{decode:>10.2f}{wall:>9.2f}{per_slot:>12.2f}{step_ms:>10.1f}{m50:>10.2f}{m50/0.75:>13.2f}")
