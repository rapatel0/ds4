#!/bin/bash
# Sprint 606 Phase B/C driver: rendezvous-merge (RDZV_MERGE) amplifier gate
# + A/B perf on the promoted edges+fix base. Each run goes through s606-run.sh
# (idle/foreign preflight + bench + tolerance). Hard per-run timeout guards.
set -u
RUN=/workspace/ds4/tools/s606-run.sh

gate () { # name SLOTS extra_env...
  local name="$1"; local slots="$2"; shift 2
  echo "==== $name (slots=$slots) $* ===="
  env SLOTS="$slots" REQUESTS="$([ "$slots" = 32 ] && echo 128 || echo 32)" \
      "$@" timeout 1500 bash "$RUN" "$name"
  echo "==== $name rc=$? ===="
}

case "${1:-all}" in
  amp)
    # Amplifier gates: RDZV_MERGE on, the dense-hazard amplifier at the two
    # carrier sites. Must stay 1.0/1.0 (elided barrier did NOT reopen the
    # dense<->rank / compose-region hazard). S=32 reference shape (tolerance on).
    gate rdzv-amp20-aoa 32 DS4_V100_TP_EP_RDZV_MERGE=1 \
        DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
    gate rdzv-amp20-precompose 32 DS4_V100_TP_EP_RDZV_MERGE=1 \
        DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=pre_compose
    ;;
  ab)
    # A/B perf: incumbent (off) vs rdzv (on), paired, S=8 and S=32.
    SKIP_TOL=1 gate rdzv-off-s8 8 DS4_V100_TP_EP_RDZV_MERGE=0
    SKIP_TOL=1 gate rdzv-on-s8  8 DS4_V100_TP_EP_RDZV_MERGE=1
    gate rdzv-off-s32 32 DS4_V100_TP_EP_RDZV_MERGE=0
    gate rdzv-on-s32  32 DS4_V100_TP_EP_RDZV_MERGE=1
    # S=16 single point (no tolerance below 32)
    SKIP_TOL=1 gate rdzv-off-s16 16 DS4_V100_TP_EP_RDZV_MERGE=0
    SKIP_TOL=1 gate rdzv-on-s16  16 DS4_V100_TP_EP_RDZV_MERGE=1
    ;;
  *) echo "usage: s606-phaseBC.sh amp|ab"; exit 2;;
esac
