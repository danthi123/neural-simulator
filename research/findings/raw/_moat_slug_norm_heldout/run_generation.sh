#!/usr/bin/env bash
# Generate the held-out real-Qwen corpus: one fresh-session RICH turn "What is <slug>?" per selected entity, through the
# production rich path (webapp.server.brain_chat via the battery worker), Qwen-0.5B on CPU, BRAIN_MOAT_SLUG_NORM UNSET.
# usage (cwd = the worktree):  bash research/findings/raw/_moat_slug_norm_heldout/run_generation.sh <out_json> [memcap_gb] [first_turn] [last_turn]
# One process per BATCH of turns: a single 29-turn process grew past a 12 GB cap and was OOM-killed at ~turn 25 (2026-10-18), losing
# everything (the worker writes its output only at the end). Batches of ~8 keep each process well inside the cap.
# RENDERER=qwen-cpu (default; turns 1-16 were made this way) or qwen-gpu (turns 17-29: Qwen fp16 on the GPU via tools/gpu_queue.sh).
# Needs data/corpus/tinystories.txt in the worktree (an ignored path; symlink it from the main checkout).
set -uo pipefail
OUT="${1:?out json}"; CAP="${2:-12}"; FIRST="${3:-1}"; LAST="${4:-0}"
H=research/findings/raw/_moat_slug_norm_heldout
unset BRAIN_MOAT_SLUG_NORM
EXTRA=$(.venv/bin/python - <<'PY'
import json
d = json.load(open("research/findings/raw/_moat_slug_norm_heldout/entities.json"))
print(json.dumps([["h%02d" % i, "What is %s?" % e, "h%02d" % i, True] for i, e in enumerate(d["entities"], 1)]))
PY
)
LABELS=$(.venv/bin/python -c "import json,sys; a=int(sys.argv[2]); b=int(sys.argv[3]); L=[x[0] for x in json.loads(sys.argv[1])]; print(','.join(L[a-1:(b or len(L))]))" "$EXTRA" "$FIRST" "$LAST")
bash tools/memcap.sh "$CAP" -- .venv/bin/python -u $H/drive_memo.py --out "$OUT" --renderer "${RENDERER:-qwen-cpu}" --seed 1 \
  --labels "$LABELS" --extra "$EXTRA"
