# GPU Profiling Guidance (homelab V100 / 4090 nodes)

**Audience:** any agent (or human) profiling GPU code on the homelab cluster —
primarily DSV4 / llama.cpp kernel work on **gpu-01** (8× Tesla V100-SXM2,
Volta sm_70), secondarily LLM serving on **gpu-02-4090rtx** (1× RTX 4090, Ada
sm_89).

**TL;DR:** Profile *on demand, targeted, and time-boxed* with the right tool
for the question. Do **not** rely on a persistent cluster metrics exporter for
kernel tuning — it both gives you the wrong kind of data and contends for the
GPU's profiling counters. There is effectively **one hardware-perf-counter
client at a time** per GPU; treat it as a mutex you must hold deliberately.

---

## Why this doc exists (the decision)

We deliberately **disabled the persistent `nvidia-dcgm-exporter` on gpu-01**
(2026-05-27). Rationale:

- Kernel tuning needs you to *choose what you measure per experiment* (this
  kernel, these counters, this window). That's a code/run-level decision, not
  a fleet-monitoring config.
- A persistent DCGM exporter holds the GPU's profiling counters continuously to
  emit `DCGM_FI_PROF_*` series. That **conflicts with targeted profilers**
  (Nsight Compute/Systems, CUPTI, `dcgmi dmon`) which need those same counters.
  Managing the metric set at the infra level (DaemonSet + ClusterPolicy) is the
  wrong altitude — it would mean editing cluster config every time you change
  what you profile.

So gpu-01's profiling counters are now **free for on-demand use**. gpu-02 still
runs a DCGM exporter (for the LLM-serving dashboards); pause it before
profiling on that node (see "Before you profile").

---

## The hardware constraint you must respect

GPU performance counters are a **fixed, small, shared physical resource**.

1. **One profiling session at a time (practically).** If DCGM is holding the
   profiling counters and you launch Nsight Compute, you get counter-busy
   errors or degraded/zeroed data — and vice versa. On Volta this is strict.
2. **Metric groups + time-multiplexing.** Profiling fields are organized into
   groups; fields needing the same counter slots cannot be measured
   simultaneously. A tool that requests many groups **round-robins** them, so
   each field is only sampled a fraction of the time → blurred, time-averaged
   numbers that miss short bursts. More fields ≠ more truth; it's often *less*
   truth per field. (Volta multiplexes harder than Ampere/Ada.)
3. **Two metric classes, very different cost:**
   - `DCGM_FI_DEV_*` / `nvidia-smi` fields (util, mem, power, temp, clocks,
     `NVLINK_BANDWIDTH_TOTAL`, XID, ECC) come from **NVML/the driver** — cheap,
     no counter contention, safe to read anytime, even while profiling.
   - `DCGM_FI_PROF_*` and all Nsight/CUPTI metrics (SM active/occupancy,
     pipe-active, achieved throughput, stall reasons) come from **hardware perf
     counters** — these are the contended, multiplexed, "heavy" ones.

**Driver is already permissive:** `NVreg_RestrictProfilingToAdminUsers=0` on
gpu-01, so non-root processes may profile (no `ERR_NVGPUCTRPERM`). You still
need the counters to be *free*.

---

## Pick the tool by the question

| Question | Tool | Counter contention | Overhead |
|---|---|---|---|
| Is it running? how hot / how much VRAM / power / clocks? | `nvidia-smi`, `nvidia-smi dmon` | none (NVML) | ~0 |
| Coarse "is the GPU busy" over a window, live | `nvidia-smi dmon -s pucm` | none | ~0 |
| Time-series PROF metrics for a short window, no other profiler running | `dcgmi dmon -e <fields>` | **yes** (holds counters) | low–med |
| Where is wall-time going? kernel timeline, gaps/bubbles, comms↔compute overlap, NVLink/PCIe transfers, CPU↔GPU sync | **Nsight Systems (`nsys`)** | light (tracing, not full counters) | low–med |
| Why is *this kernel* slow? occupancy, tensor/pipe utilization, memory throughput, stall reasons, roofline | **Nsight Compute (`ncu`)** | **yes** (exclusive; replays kernels) | **high** (serializes + replays) |

Rules of thumb:
- **Tuning code → Nsight, not DCGM.** DCGM time-series is fleet monitoring at a
  30 s cadence with multiplex-blurred PROF fields. It cannot tell you why a
  kernel is slow. `nsys` for "where's the time," `ncu` for "why is this kernel
  slow."
- **`nsys` first, `ncu` second.** Use `nsys` to find the expensive kernel /
  the bubble, then `ncu` on just that kernel. Don't `ncu` a whole run — it
  replays every kernel and is brutally slow.
- **`nvidia-smi`/NVML for health** — always safe, even mid-profile.

---

## Before you profile (pre-flight: claim the counter "mutex")

1. **gpu-01:** persistent DCGM exporter is already disabled. Confirm nothing
   else is profiling:
   ```bash
   # no dcgm-exporter pod should be on gpu-01:
   kubectl -n gpu-operator-resources get pods -o wide | grep gpu-01
   # no stray nsys/ncu/dcgmi holding counters:
   ssh ubuntu@192.168.102.5 'pgrep -a -f "nsys|ncu|dcgmi|cupti" || echo clear'
   ```
2. **gpu-02-4090rtx:** if profiling there, pause its exporter first, restore after:
   ```bash
   kubectl label node gpu-02-4090rtx nvidia.com/gpu.deploy.dcgm-exporter=paused --overwrite
   # ... profile ...
   kubectl label node gpu-02-4090rtx nvidia.com/gpu.deploy.dcgm-exporter=true --overwrite
   ```
   (Note: RTX 4090 is consumer silicon — DCGM/Nsight profiling-counter support
   is limited vs the V100s. Expect fewer PROF metrics there.)
3. **Pin to a specific GPU** so you don't perturb other GPUs on the node:
   `CUDA_VISIBLE_DEVICES=<idx>` (and remember tensor-parallel runs span several
   — profile the set you actually care about).

---

## Where the tools live

- `nvidia-smi`: **on the host** (`ssh ubuntu@192.168.102.5`), NVML — always safe.
- `dcgmi` + `nv-hostengine`: **installed on the gpu-01 host for debugging** (an
  `nv-hostengine` listens on `localhost:5555`, independent of the k8s GPU
  operator's exporter). So `dcgmi dmon` works on gpu-01 out of the box — see the
  dedicated section below. On a node *without* a running engine, `dcgmi` returns
  "Unable to connect to host engine" until you start one (`sudo nv-hostengine`).
- `nsys`, `ncu`: **not on the host** — they ship in the CUDA toolkit, i.e. in
  the workload's CUDA container or a CUDA-devel **dev pod**. Use the dev-pod
  pattern from the `homelab-k8s-dev` skill
  (`manifests/deepseek-v4-flash-lite/cuda-dev-pod.yaml`): a privileged pod with
  `SYS_ADMIN` (required for `ncu` perf counters), nvcc/nsys/ncu, /workspace on
  hostPath scratch. Profile the binary inside that pod, write reports to
  `/srv/dev/<workload>/`, copy out with `kubectl cp`.

---

## Recipes

**Health while a run is in flight (safe, NVML):**
```bash
ssh ubuntu@192.168.102.5 nvidia-smi dmon -s pucm -d 1   # 1s util/power/clk/mem
```

**Timeline of a run (`nsys`) — find the expensive region:**
```bash
nsys profile -o /workspace/prof/run1 \
  --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  ./your_binary --your-args
# then: nsys stats /workspace/prof/run1.nsys-rep   (or open the .nsys-rep)
```
Good for: kernel durations, gaps/bubbles, H2D/D2H copies, NVLink/NCCL collective
timing, whether comms overlap compute in tensor-parallel runs.

**Deep-dive one kernel (`ncu`) — target it, don't profile everything:**
```bash
ncu -o /workspace/prof/kern1 \
  --target-processes all \
  --kernel-name "regex_for_kernel" \
  --launch-skip 20 --launch-count 5 \
  --set full \
  ./your_binary --your-args
```
`--set full` is heavy; for occupancy-only use `--set basic` or a specific
`--metrics` list (e.g. `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__warps_active.avg.pct_of_peak_sustained_active` for occupancy,
`sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` for tensor cores).
Always `--launch-skip`/`--launch-count` to a handful of steady-state launches —
profiling every launch replays each kernel many times and can turn a 1-min run
into an hour.

---

## `dcgmi dmon` — live CLI monitoring (the simple middle tier)

`dcgmi dmon` streams a rolling per-GPU table in the terminal, like
`nvidia-smi dmon` but with the **DCGM field set** — including SM occupancy,
tensor activity, and directional NVLink, which `nvidia-smi` does not have. It's
the sweet spot when you want to *watch* a run live across all 8 GPUs without the
weight of Nsight reports.

**Where it fits:**

| | `nvidia-smi dmon` | **`dcgmi dmon`** | Nsight (`nsys`/`ncu`) |
|---|---|---|---|
| Source | NVML | NVML + **HW perf counters** | HW perf counters / trace |
| Has SM occupancy / tensor / per-pipe / NVLink dir? | no | **yes** | yes (deeper) |
| Granularity | per-GPU, 1s | per-GPU, sub-second–1s | per-kernel |
| Attributes to a kernel? | no | **no** (aggregate over time) | yes |
| Counter contention | none | **only if you select PROF fields** | yes |
| Setup | none | needs a host engine (have it on gpu-01) | dev-pod |

Use it for: "while this run executes, are the SMs occupied, are the tensor
cores lit, is NVLink saturated during the all-reduce, which GPUs are idle." Use
Nsight when you need *why a specific kernel* is slow — `dcgmi dmon` can't
attribute to kernels.

**Field IDs:** see the **"V100 supported metrics"** section below for the
verified field list, split into *free* (NVML) vs *profiling* (counter) fields,
plus the concurrency groups that determine what you can collect without
blurring. `dcgmi dmon -l` dumps the full ~300-field catalog; `dcgmi profile -l`
dumps just the profiling fields the silicon actually supports.

**Health watch — free, safe even alongside Nsight or a running bench:**
```bash
ssh ubuntu@192.168.102.5 'dcgmi dmon -e 203,252,155,150 -d 1000 -c 0'
#                                       util fbused power temp ; -c 0 = forever, ctrl-C to stop
```

**Inference-tuning watch — the zero-multiplex set (V100, no blurring):**
```bash
# A.1 sm_active+occupancy, B dram, C pcie, D gr_engine, E nvlink — all collect at once
ssh ubuntu@192.168.102.5 'dcgmi dmon -i 0,1,2,3 -e 1002,1003,1005,1009,1010,1001,1011,1012 -d 500 -c 60'
```
For tensor-pipe %, run a **separate** pass (1004 is in subgroup A.2; mixing it
with A.1 above forces round-robin and blurs both):
```bash
ssh ubuntu@192.168.102.5 'dcgmi dmon -i 0,1,2,3 -e 1004 -d 500 -c 60'
```
⚠ Both **grab the profiling counters** for the duration → do **not** run while
`ncu`/`nsys` profile the same GPUs (and vice-versa). Self-cleaning: counters
release when the command exits. `-i 0,1,2,3` pins to just the tensor-parallel
set. See "V100 supported metrics" below for the full concurrency-group rules.

**Tradeoffs vs the heavyweight tools:**
- Cheaper to run and reason about than Nsight; great for a live "is it
  saturated" read during a benchmark.
- But it's **aggregate-over-time per GPU** — if you select many PROF fields it
  still time-multiplexes (blur), and it can't tell you which kernel caused what.
  For kernel-level truth, still use `ncu` on the localized kernel.
- Because the host engine is always up on gpu-01, `dcgmi dmon` is the
  lowest-friction way to glance at PROF metrics — just remember the PROF subset
  is the counter mutex, so don't leave it running when you hand the GPU to
  `ncu`.

---

## V100 supported metrics (verified 2026-05-27 via `dcgmi profile -l` / `dcgmi dmon`)

`dcgmi dmon -l` lists ~300 fields, but most are N/A on this hardware (NvSwitch,
Grace CPU, ConnectX, vGPU, C2C — none present on the SYS-4029GP-TVRT). The lists
below are what the **Tesla V100-SXM2 actually reports**. Two classes, and the
distinction is the whole game: **free** fields cost nothing and never block a
profiler; **profiling** fields consume the shared hardware perf counters.

### Free metrics (NVML — no counter contention, safe anytime)

Read these as often as you like, even while `ncu`/`nsys` are running and during a
live benchmark. Sourced from the driver/NVML, not the perf counters.

| ID | Field | Notes |
|----|----|----|
| 100 / 101 / 102 | sm_clock / memory_clock / video_clock | throttling shows as a clock drop |
| 110 / 111 / 113 / 114 | sm_app / mem_app / sm_max / mem_max clock | |
| 112 | current_clocks_event_reasons | bitmask: why clocks are capped |
| 140 / 150 | memory_temp / gpu_temp | HBM + core temp |
| 158 / 159 | slowdown_temp / shutdown_temp | thermal thresholds |
| 155 / 157 / 156 | power_usage / power_usage_instant / total_energy | |
| 160–164 | power mgmt limits (min/max/default/enforced) | |
| 190 | pstate | P0 = max perf |
| 203 / 204 | gpu_utilization / mem_copy_utilization | coarse "busy %" |
| 206 / 207 | enc_utilization / dec_utilization | NVENC/NVDEC |
| 235 / 236 / 237 / 238 | pcie_max_link_gen/width, pcie_link_gen/width | confirm gen3 ×16 (max==current = healthy) |
| 250 / 251 / 252 / 253 | fb_total / fb_free / fb_used / fb_resv | VRAM pressure |
| 230 | xid_errors | N/A = none logged (good) |
| 240–247 | power/thermal/sync/board/reliability/clock violation time | nonzero = was throttled |
| 300 / 301 | ecc / ecc_pending | enabled flag |
| 310–345 | ecc SBE/DBE volatile+aggregate by unit | error counts |
| 390 / 391 / 392 | retired_pages sbe / dbe / pending | V100's bad-page mechanism; watch on aged cards |
| 202 | pcie_replay_counter | PCIe link health |

**Not available on V100** (return N/A): `191 fan_speed` (SXM2 is passively
cooled), `200/201 pcie_tx/rx_throughput` (NVML variant — use the profiling
`1009/1010` bytes instead).

### Profiling metrics (DCP — hardware counters, contended + multiplexed)

These are the only PROF fields the V100 supports. **`dcgmi profile -l` reports
them in concurrency groups**, and that grouping is the actual rule for what you
can measure cleanly at the same instant:

```
Group.Subgroup   Field
A.1   1002 sm_active        1003 sm_occupancy
A.2   1004 tensor_active
A.3   1006 fp64_active
A.4   1007 fp32_active
A.5   1008 fp16_active
B.0   1005 dram_active
C.0   1009 pcie_tx_bytes    1010 pcie_rx_bytes
D.0   1001 gr_engine_active
E.0   1011 nvlink_tx_bytes  1012 nvlink_rx_bytes
```

**The concurrency rule — read this before selecting fields:**
- Fields in **different top-level groups (A, B, C, D, E) collect simultaneously**
  with no penalty — they use separate counter hardware.
- Fields in **different subgroups of the same group (A.1 … A.5) are mutually
  exclusive** at a given instant. They share the SM counters, so DCGM
  **time-multiplexes** them: round-robin, each sampled only part of the time.

**Blurring (the cost of asking for too much):** if you request, say, A.1
(sm_active/occupancy) + A.2 (tensor) + A.4 (fp32) together, those three A-subgroups
round-robin. Each is then counted ~1/3 of the wall-clock, so the reported value is
a coarse time-average that **misses short bursts and smears spikes** — precisely
the opposite of what you want when judging whether decode occupies the SMs. More
A-subgroups = blurrier occupancy. (B/C/D/E ride along for free and don't blur the
A reading.)

**Overhead:** beyond blur, every multiplex switch reprograms the counters, which
has a small cost and can briefly serialize with the running kernel. At a 1 s dmon
cadence this is low, but on a node you're benchmarking it's non-zero — a reason to
keep the PROF set tight and time-boxed, and to never leave a PROF dmon running
when you hand the GPU to `ncu`.

**Zero-multiplex set (recommended default — everything below collects at once):**
```
1002,1003   (A.1 — SM active + occupancy : the decode-occupancy question)
1005        (B   — DRAM / memory-bandwidth pressure)
1009,1010   (C   — PCIe bytes)
1001        (D   — overall engine active)
1011,1012   (E   — NVLink GPU-GPU comms)
```
That's one field from each of A/B/C/D/E (A.1 chosen) → no round-robin, clean
numbers. **To read tensor-pipe % cleanly, run a *separate* pass** with `1004`
(A.2) alone — don't mix it with A.1, or you blur both.

**Not supported on V100** (Ampere+ only — would print N/A): `1013/1014/1015`
tensor imma/hmma/dfma breakdown (you get aggregate `1004 tensor_active` only),
`1016 integer_active`, and the per-NVLink-lane PROF byte counters `1040–1075`.

---

## What each metric actually tells you (V100 inference tuning)

- **SM Active** (`sm__...active` / DCGM SM_ACTIVE): fraction of time ≥1 warp ran
  on an SM. Low = launch/latency-bound or not enough work.
- **SM Occupancy** (SM_OCCUPANCY): resident warps ÷ max. Low occupancy +
  high SM-active = fine (latency-hidden); low + low = under-fed → bump batch /
  fuse / increase parallelism. **This is the metric that tells you if decode
  under-occupies the SMs** — the main reason it was wanted.
- **Tensor Active** (PIPE_TENSOR_ACTIVE): HMMA pipe utilization. If your matmul
  is supposed to be on tensor cores but this is near-zero, you've fallen back to
  FP32 FMA — check the FP32/FP16 pipe-active to confirm the fallback.
- **DRAM Active**: memory-bandwidth pressure. High DRAM + low tensor =
  memory-bound; you want tiling/reuse, not more FLOPs.
- **NVLink TX/RX** (directional, via `nsys`/`ncu` or DCGM PROF): tensor-parallel
  all-reduce/all-gather traffic. Watch for comms not overlapping compute.
- **PCIe TX/RX**: H2D/D2H — if high during steady decode, you have an unwanted
  host round-trip.

---

## If you ever want persistent monitoring back on gpu-01

This was a deliberate trade (free counters > always-on dashboards on the dev
node). To restore:

```bash
# full persistent exporter (re-introduces counter contention):
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=true --overwrite
```

Better middle ground if you want always-on **health** without blocking
profiling: run an exporter with a **DEV-only** metric set (no `DCGM_FI_PROF_*`).
NVML-sourced fields (util/mem/power/temp/clocks/nvlink-bandwidth-total) don't
touch profiling counters, so they coexist with `nsys`/`ncu`. That requires a
custom metrics CSV via `ClusterPolicy.spec.dcgmExporter.config` — ask before
doing it, since the ClusterPolicy edit re-rolls the exporter fleet-wide.

The Grafana **GPU Live Performance** dashboard (`/d/gpu-live-perf`) still works;
while gpu-01's exporter is disabled, its gpu-01 panels read empty — use
`nvidia-smi dmon` on the node for live health there instead.

---

## Golden rules

1. **One counter client at a time.** Pre-flight that nothing else holds the
   counters before `ncu`/`dcgmi`.
2. **`nsys` to localize, `ncu` to explain.** Never blanket-`ncu` a whole run.
3. **Target everything:** specific GPU (`CUDA_VISIBLE_DEVICES`), specific kernel
   (`--kernel-name`), specific launches (`--launch-skip/--launch-count`),
   specific metrics (`--metrics` / `--set basic`).
4. **NVML for health, perf counters for tuning.** Don't pay counter cost for
   numbers `nvidia-smi` gives free.
5. **Capture to `/srv/dev/<workload>/` (ZFS scratch), never commit `.nsys-rep` /
   `.ncu-rep`** — record path + sha256 in a `traces.md`, regenerate on demand
   (per the homelab-k8s-dev discipline).
6. **Clean up:** kill stray `nsys`/`ncu` daemons; restore any exporter you
   paused on gpu-02.
