#!/usr/bin/env bash
# The M2 evidence queue: every brain-loading step (pytest included) under the brain lock + mem_ok + memcap, in order,
# from a FROZEN export of the branch (git archive <sha>, research/findings + data symlinked; FROZEN_SHA inside):
#   bash docs/drafts/one-brain-m2/queue_runs.sh <frozen_export_dir> <step> [<step> ...]
# Steps: tests | probe7 | probe8 | off7 | off8 | on7 | on8 | pair7 | offpair7 | ltm7
# Logs: docs/drafts/one-brain-m2/runs/<step>.log (hash runs: runs/<name>_s<seed>_<script>.json/.log).
# The probe gets memcap 14 (its documented dev-run cap: the intact arm holds two per-session brains, ~11.9 GB at M0);
# test_webapp_server gets 18 (task rule); everything else 12.
set -uo pipefail
X="${1:?frozen export dir}"; shift
W="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCK=/home/dant123/Projects/sim-rev/brain.lock
PY=/home/dant123/Projects/sim/.venv/bin/python
R="$W/docs/drafts/one-brain-m2/runs"
mkdir -p "$R"

locked() {   # <log name> <GB> <cmd ...>: run cmd from the export under the lock + mem_ok + memcap
  local name="$1" gb="$2"; shift 2
  echo "[queue] $(date -Is) START $name"
  flock -w 7200 "$LOCK" bash -c 'cd "$1" && bash tools/mem_ok.sh "$2" && bash tools/memcap.sh "$2" -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "${@:3}"' \
    _ "$X" "$gb" "$@" > "$R/$name.log" 2>&1
  echo "[queue] $(date -Is) END $name rc=$?"
}

hashes() {   # <name> <seed> <single|pair> [ENV=VAL ...]
  echo "[queue] $(date -Is) START hashes $1 s$2 $3 ${*:4}"
  flock -w 7200 "$LOCK" bash "$W/docs/drafts/one-brain-m2/run_hashes.sh" "$1" "$X" "$2" "$3" "${@:4}"
  echo "[queue] $(date -Is) END hashes $1 s$2 $3 rc=$?"
}

probe() {    # <seed>
  locked "probe_s$1" 14 "$PY" -u -m research.runners.one_brain_probe --seed "$1" \
    --arms one_brain freeze_learning zero_learned_store intact restart_persist_on_one_brain \
    --out "$W/research/findings/raw/_one_brain_probe/m2_s$1.json"
}

for step in "$@"; do
  case "$step" in
    tests)
      locked pytest_csp 12 "$PY" -m pytest -q -p no:warnings tests/test_cross_session_persistence.py
      locked pytest_ws 18 "$PY" -m pytest -q -p no:warnings tests/test_webapp_server.py -k \
        "brain_chat_reset_idempotent or tiny_demo_answers_and_abstains or rich_false_escape or curiosity_d6_session_isolated or source_provenance_explicit_off"
      ;;
    probe7) probe 7 ;;
    probe8) probe 8 ;;
    off7) hashes off 7 single ;;
    off8) hashes off 8 single ;;
    on7) hashes on 7 single BRAIN_ONE_BRAIN=1 ;;
    on8) hashes on 8 single BRAIN_ONE_BRAIN=1 ;;
    pair7) hashes on 7 pair BRAIN_ONE_BRAIN=1 ;;
    offpair7) hashes off 7 pair ;;
    ltm7) hashes offltm 7 single BRAIN_SHARED_LTM=1 ;;
    *) echo "[queue] unknown step $step" ;;
  esac
done
echo "[queue] $(date -Is) DONE"
