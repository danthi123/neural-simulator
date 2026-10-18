#!/usr/bin/env bash
# Run run_replay.sh over a label list in sequential chunks of N turns (one process per chunk): a single process that builds ~20 fresh-session
# brains grows past a 12 GB memcap and is OOM-killed, losing its whole batch (seen 2026-10-18, rc=137).  Output: <out_prefix>_<k>.json.
# usage (cwd = worktree): [RMODE=stub] bash .../run_replay_chunks.sh <s1|s3> <on|off> <out_prefix> <labels,csv> [chunk=7] [memcap_gb=16]
set -uo pipefail
STRATUM="${1:?}"; ARM="${2:?}"; PREFIX="${3:?}"; LABELS="${4:?}"; N="${5:-7}"; CAP="${6:-16}"
H=research/findings/raw/_moat_slug_norm_heldout
IFS=',' read -r -a ALL <<< "$LABELS"
i=0; k=1
while [ "$i" -lt "${#ALL[@]}" ]; do
  CH=$(IFS=,; echo "${ALL[*]:$i:$N}")
  bash $H/run_replay.sh "$STRATUM" "$ARM" "${PREFIX}_$k.json" "$CH" "$CAP" > "${PREFIX}_$k.log" 2>&1
  echo "[chunks] ${PREFIX}_$k rc=$? :: $CH"
  i=$((i + N)); k=$((k + 1))
done
echo "[chunks] done"
