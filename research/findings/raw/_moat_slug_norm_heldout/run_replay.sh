#!/usr/bin/env bash
# Composer-path replay: the production brain + rich path (webapp.server.brain_chat via the battery worker) with the Qwen renderer
# replaced by a replay of the recorded Qwen sentences of the stratum's corpus (drive_memo.py --renderer replay), flag
# BRAIN_MOAT_SLUG_NORM on or off, or (ARM=allowlist, 2026-10-20) BRAIN_MOAT_ALLOWLIST on with BRAIN_MOAT_SLUG_NORM unset.  Every verify/accept/abstain decision is the production code's; no model is loaded.
# RMODE=stub (env) runs the same turns with the GPU-free template StubRenderer instead of the replay (the production renderer of a
# GPU-less console; asserted SVO canonical, so the stock one-fact path is the trivial single-triple chat._verify).
# usage (cwd = worktree): [RMODE=stub] bash .../run_replay.sh <s1|s3> <on|off|allowlist> <out_json> <labels,csv> [memcap_gb=12]
set -uo pipefail
STRATUM="${1:?s1|s3}"; ARM="${2:?on|off|allowlist}"; OUT="${3:?out}"; LABELS="${4:?labels}"; CAP="${5:-12}"
H=research/findings/raw/_moat_slug_norm_heldout
unset BRAIN_MOAT_SLUG_NORM BRAIN_MOAT_ALLOWLIST
if [ "$STRATUM" = s1 ]; then
  CORPUS=$H/corpus.json
  EXTRA=$(.venv/bin/python -c "
import json
d=json.load(open('$H/entities.json'))
print(json.dumps([['h%02d'%i,'What is %s?'%e,'h%02d'%i,True] for i,e in enumerate(d['entities'],1)]))")
else
  CORPUS=$H/s3_corpus.json
  EXTRA=$(.venv/bin/python -c "
import json
d=json.load(open('$H/s3_entities.json'))
print(json.dumps([[l,'What is %s?'%e,l,True] for l,e,_f in d['turns']]))")
fi
if [ "$ARM" = on ]; then export DRIVE_ENV='{"BRAIN_MOAT_SLUG_NORM":"1"}'
elif [ "$ARM" = allowlist ]; then export DRIVE_ENV='{"BRAIN_MOAT_ALLOWLIST":"1"}'
elif [ "$ARM" = off ]; then export DRIVE_ENV='{}'
else echo "unknown arm $ARM" >&2; exit 2; fi
until bash tools/mem_ok.sh "$CAP"; do echo "[replay] waiting for RAM"; timeout 60 tail -f /dev/null; done
if [ "${RMODE:-replay}" = stub ]; then RARGS="--renderer stub"; else RARGS="--renderer replay --replay-corpus $CORPUS"; fi
bash tools/memcap.sh "$CAP" -- .venv/bin/python -u $H/drive_memo.py --out "$OUT" $RARGS --seed 1 \
  --labels "$LABELS" --extra "$EXTRA"
echo "rc=$?"
