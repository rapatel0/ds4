# Sprint 214 Intent - Tile-Local Routed FFN Workbench

Build the first repo-owned workbench for a true tile-local/persistent six-route
routed-FFN executor. Sprints 200, 206, and 213 closed wrapper, graph, and
reducer-only explanations for the six-route production bottleneck. Sprint 214
should stop optimizing the existing sequence boundary and instead prototype a
kernel/workbench that keeps gate/up, activation, and down/reduce dataflow inside
a larger CUDA boundary.

Constraints:

- no TP runtime integration;
- no PP scheduler changes;
- no production default promotion from a standalone benchmark;
- use V100 `sm_70` and the existing TurboMind/CUTLASS-style low-bit code as the
  implementation substrate;
- compare against the current focused sequence baseline:
  `fused6_reduce + graph` served default and Sprint 213 focused numbers
  (`0.1391 ms` atomic, `0.1290 ms` materialized split-reduce).

Success requires V100 correctness, focused timing, and an explicit next
decision: continue toward appliance integration only if the workbench clears a
material focused speedup; otherwise pivot away from routed-FFN microkernel work.
