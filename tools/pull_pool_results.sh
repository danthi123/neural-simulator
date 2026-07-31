#!/usr/bin/env bash
# Pull pool artifacts AND stamp their provenance from the dispatch log.
#
# WHY. The automatic provenance door lives in research/runners/__init__.py and stamps files created under
# research/findings/raw/. Pool jobs write to ~/derisk-pool/sim/g5s_out/ and are rsync'd back, so they land in
# raw/ WITHOUT ever passing the door -- every pool artifact arrived unprovenanced and the gate blocked on it.
#
# The dispatch log already holds the exact command per artifact, including its #checked reason. So the
# provenance is not missing, it is merely unattached. This attaches it at pull time.
#
#   bash tools/pull_pool_results.sh '<glob>' <dest-subdir>
#   bash tools/pull_pool_results.sh 'g5w0_*.json' gap5_density
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
GLOB="${1:?usage: pull_pool_results.sh '<glob>' <dest-subdir>}"; DEST="research/findings/raw/${2:?}"
mkdir -p "$DEST"
for H in pool40 pool41 pool42; do
  timeout 30 rsync -q -e "ssh -o BatchMode=yes -o ConnectTimeout=6" "$H:derisk-pool/sim/g5s_out/$GLOB" "$DEST/" 2>/dev/null
done
SHA=$(git rev-parse --short HEAD)
N=0
for f in $DEST/$GLOB; do
  [ -f "$f" ] || continue
  [ -f "$f.prov.json" ] && continue
  BASE=$(basename "$f")
  CMD=$(grep -F "$BASE" research/queue/dispatch.log 2>/dev/null | tail -1 | sed 's/.*<- //')
  [ -z "$CMD" ] && { echo "  ⚠️  no dispatch record for $BASE — cannot attach provenance, NOT inventing one"; continue; }
  .venv/bin/python - "$f" "$CMD" "$SHA" <<'PY'
import json, sys
art, cmd, sha = sys.argv[1], sys.argv[2], sys.argv[3]
core, _, reason = cmd.partition("#checked:")
json.dump({"run_id": "pool", "runner": "research/runners/_gap5_btsp_place_field_derisk.py",
           "argv": core.strip().split(), "checked_reason": reason.strip(), "git_sha": sha,
           "started": "see research/queue/dispatch.log", "env": {"SIM_BACKEND": "numpy"},
           "artifact": art,
           "note": "attached at PULL time from dispatch.log: pool jobs write to g5s_out/ and never pass the "
                   "provenance door in research/runners/__init__.py"},
          open(art + ".prov.json", "w"), indent=1)
PY
  N=$((N+1))
done
echo "  pulled + stamped $N artifact(s) into $DEST"
