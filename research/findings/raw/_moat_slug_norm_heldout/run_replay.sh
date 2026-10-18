#!/usr/bin/env bash
# Composer-path replay: the production brain + rich path (webapp.server.brain_chat via the battery worker) with the Qwen renderer
# replaced by a replay of the recorded Qwen sentences of the stratum's corpus (drive_memo.py --renderer replay), flag
# BRAIN_MOAT_SLUG_NORM on or off.  Every verify/accept/abstain decision is the production code's; no model is loaded.
# usage (cwd = worktree): bash .../run_replay.sh <s1|s3> <on|off> <out_json> <labels,csv> [memcap_gb=12]
set -uo pipefail
STRATUM="${1:?s1|s3}"; ARM="${2:?on|off}"; OUT="${3:?out}"; LABELS="${4:?labels}"; CAP="${5:-12}"
H=research/findings/raw/_moat_slug_norm_heldout
unset BRAIN_MOAT_SLUG_NORM
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
if [ "$ARM" = on ]; then export DRIVE_ENV='{"BRAIN_MOAT_SLUG_NORM":"1"}'; else export DRIVE_ENV='{}'; fi
until bash tools/mem_ok.sh "$CAP"; do echo "[replay] waiting for RAM"; timeout 60 tail -f /dev/null; done
bash tools/memcap.sh "$CAP" -- .venv/bin/python -u $H/drive_memo.py --out "$OUT" --renderer replay --replay-corpus "$CORPUS" --seed 1 \
  --labels "$LABELS" --extra "$EXTRA"
echo "rc=$?"
