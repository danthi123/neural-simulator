#!/usr/bin/env bash
# Generate the held-out real-Qwen corpus: one fresh-session RICH turn "What is <slug>?" per selected entity, through the
# production rich path (webapp.server.brain_chat via the battery worker), Qwen-0.5B on CPU, BRAIN_MOAT_SLUG_NORM UNSET.
# usage (cwd = the worktree):  bash research/findings/raw/_moat_slug_norm_heldout/run_generation.sh <out_json> [memcap_gb]
set -uo pipefail
OUT="${1:?out json}"; CAP="${2:-12}"
H=research/findings/raw/_moat_slug_norm_heldout
unset BRAIN_MOAT_SLUG_NORM
EXTRA=$(.venv/bin/python - <<'PY'
import json
d = json.load(open("research/findings/raw/_moat_slug_norm_heldout/entities.json"))
print(json.dumps([["h%02d" % i, "What is %s?" % e, "h%02d" % i, True] for i, e in enumerate(d["entities"], 1)]))
PY
)
LABELS=$(.venv/bin/python -c "import json,sys; print(','.join(x[0] for x in json.loads(sys.argv[1])))" "$EXTRA")
bash tools/memcap.sh "$CAP" -- .venv/bin/python -u $H/drive_memo.py --out "$OUT" --renderer qwen-cpu --seed 1 \
  --labels "$LABELS" --extra "$EXTRA"
