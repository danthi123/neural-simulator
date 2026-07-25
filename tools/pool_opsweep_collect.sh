#!/usr/bin/env bash
# Collect the consolidation dendritic operating-point sweep results from the pool + summarize candidates.
# Run this ON RETURN (Tuesday). Rsyncs every node's JSON into the local repo, then ranks candidate operating points.
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=derisk-pool/sim
OUT=research/findings/raw/consol_opsweep
mkdir -p "$OUT"
for h in pool40 pool41 pool42; do
  rsync -az "$h:~/$REMOTE/$OUT/op*_seed*.json" "$OUT/" 2>/dev/null
  rsync -az "$h:~/$REMOTE/$OUT/QUEUE_DONE_*" "$OUT/" 2>/dev/null
  echo -n "$h: "; ssh "$h" "ls ~/$REMOTE/$OUT/op*.json 2>/dev/null | wc -l; ls ~/$REMOTE/$OUT/QUEUE_DONE_* 2>/dev/null" 2>/dev/null | tr '\n' ' '; echo
done
echo "--- local total: $(ls "$OUT"/op*_seed*.json 2>/dev/null | wc -l) cells ---"
.venv/bin/python - "$OUT" <<'PY'
import json, sys, glob, collections
out = sys.argv[1]
files = sorted(glob.glob(f"{out}/op*_seed*.json"))
cand = collections.defaultdict(list); best = []; errs = 0
for f in files:
    try: r = json.load(open(f))
    except Exception: continue
    if r.get("error"): errs += 1; continue
    v = r.get("VERDICT", {})
    key = tuple(sorted(r["op"].items()))
    row = (v.get("dend_selective"), v.get("dend_separated"), v.get("dend_ratio"), v.get("lin_selective"), r["seed"])
    best.append((v.get("dend_separated") or 0, v.get("dend_selective") or 0, v.get("dend_ratio") or 0, r["config_index"], r["seed"], r["op"]))
    if v.get("candidate"): cand[key].append(r["seed"])
print(f"cells={len(files)} errors={errs}")
print(f"\nCANDIDATE operating points (dendritic separates AND selective>=2 AND beats linear), by #seeds:")
for key, seeds in sorted(cand.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(seeds)}/6 seeds {sorted(seeds)}: {dict(key)}")
if not cand: print("  (none — no point-plateau operating point robustly separates -> the deeper dendritic LINE/BUMP attractor is the named next mechanism, per the design doc NEXT-(b))")
print(f"\nTOP 12 by (separated, selective, ratio):")
for sep, sel, ratio, ci, seed, op in sorted(best, reverse=True)[:12]:
    print(f"  op{ci:03d} seed{seed}: sep={sep}/3 sel={sel}/3 ratio={ratio}  {op}")
PY
