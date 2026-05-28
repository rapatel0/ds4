# Sprint 516 - Remove Fixed Gate Telemetry From Appliance HTTP Surface

Date: 2026-05-28

## Goal

Continue the structural cleanup by removing old experiment gate vocabulary from
the production TP/EP appliance HTTP surface. These fields are not appliance CLI
flags anymore; reporting them made fixed internal defaults look like runtime
operator choices.

## Changes

- Removed fixed `*_gate` fields from TP/EP `/status` JSON.
- Removed matching fixed `ds4_tp_ep_*_gate` metrics from `/metrics`.
- Removed matching fixed gate fields from generation response JSON.
- Updated HTTP readiness/profile helpers so they no longer require or record
  the removed response/status fields.

## Validation

- `git diff --check`
- Local Python compile:
  `python3 -m py_compile tools/ds4-v100-http-readiness-check.py tools/ds4-v100-tp-ep-http-ab.py tools/ds4-v100-tp-ep-profile.py`
- Local source audit:
  `rg -n "kv_all_slots_gate|hc_persist_state_gate|true_ds4_attention_typed_kv_.*_gate|fp8_e5m2_kv_gate|router_hash_fast_gate|route_plan_async_upload_gate" appliance/http_server.cu`
  returned no matches.
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote source audit:
  `grep -En "kv_all_slots_gate|hc_persist_state_gate|true_ds4_attention_typed_kv_.*_gate|fp8_e5m2_kv_gate|router_hash_fast_gate|route_plan_async_upload_gate" appliance/http_server.cu`
  returned no matches.
- Remote Python compile:
  `python3 -m py_compile tools/ds4-v100-http-readiness-check.py tools/ds4-v100-tp-ep-http-ab.py tools/ds4-v100-tp-ep-profile.py`

## Notes

- This removes stale reporting only; it does not change decode execution.
- The local Darwin Makefile has no CUDA appliance target, so appliance build
  validation was performed on the V100 pod.
