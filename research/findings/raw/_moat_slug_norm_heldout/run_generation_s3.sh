#!/usr/bin/env bash
# Generate the S3 real-Qwen corpus (rule: select_s3.py docstring): one fresh-session RICH turn "What is <slug>?" per pick of
# s3_entities.json, through the production rich path (webapp.server.brain_chat via the battery worker), Qwen-0.5B fp16 on the GPU
# (renderer qwen-gpu), BRAIN_MOAT_SLUG_NORM UNSET.  Run as ONE gpu_queue job; turns go in sequential batches of BATCH inside it
# (one process per batch: a long single process outgrows its memcap; a lost batch costs only that batch).
# usage (cwd = the worktree):  bash research/findings/raw/_moat_slug_norm_heldout/run_generation_s3.sh <out_dir> [memcap_gb=16] [batch=12]
# Needs data/corpus/tinystories.txt in the worktree (an ignored path; symlinked from the main checkout).
set -uo pipefail
OUTDIR="${1:?out dir}"; CAP="${2:-16}"; BATCH="${3:-12}"
H=research/findings/raw/_moat_slug_norm_heldout
unset BRAIN_MOAT_SLUG_NORM
mkdir -p "$OUTDIR"
EXTRA=$(.venv/bin/python - <<'PY'
import json
d = json.load(open("research/findings/raw/_moat_slug_norm_heldout/s3_entities.json"))
print(json.dumps([[l, "What is %s?" % e, l, True] for l, e, _f in d["turns"]]))
PY
)
mapfile -t ALL < <(.venv/bin/python -c "import json,sys; [print(x[0]) for x in json.loads(sys.argv[1])]" "$EXTRA")
n=${#ALL[@]}; i=0; b=1
while [ "$i" -lt "$n" ]; do
  LABELS=$(IFS=,; echo "${ALL[*]:$i:$BATCH}")
  until bash tools/mem_ok.sh "$CAP"; do echo "[s3] waiting for RAM"; timeout 120 tail -f /dev/null; done
  echo "[s3] batch $b: $LABELS"
  bash tools/memcap.sh "$CAP" -- .venv/bin/python -u $H/drive_memo.py --out "$OUTDIR/s3_b$b.json" --renderer qwen-gpu --seed 1 \
    --labels "$LABELS" --extra "$EXTRA" > "$OUTDIR/s3_b$b.log" 2>&1
  echo "[s3] batch $b rc=$?"
  i=$((i + BATCH)); b=$((b + 1))
done
echo "[s3] done"
