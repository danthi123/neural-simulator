#!/usr/bin/env bash
# AMENDMENT-3 harvest (idempotent): pull ONLY this lane's a3 session JSONs from the isolated pool revision on each
# node (not the whole raw/ tree of every lane), then score + aggregate with the amendment-3 scorer.
#   bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <FULL_SHA> [smoke]
# Writes research/findings/raw/_load_bearing/_oe_production_turn/a3/default/default_s<seed>_a3_verdict.json and
#        research/findings/raw/_load_bearing/_oe_production_turn/a3/default_a3_aggregate.json
# The aggregate reads UNDEFINED / ARM-FAILED for any seed whose 9 sessions are not all back -- never a pass.
set -uo pipefail
cd "$(dirname "$0")/../.."
SHA=${1:?full commit sha of the isolated pool revision}
SUB=a3; [ "${2:-}" = "smoke" ] && SUB=a3_smoke
REL=research/findings/raw/_load_bearing/_oe_production_turn/$SUB/
mkdir -p "$REL"
for N in pool41 pool42 pool40; do
  timeout 180 rsync -au --exclude='*.log' -e "ssh -o BatchMode=yes -o ConnectTimeout=6" \
    "$N:derisk-pool/revisions/$SHA/$REL" "$REL" 2>/dev/null && echo "  $N: synced" || echo "  $N: unreachable / nothing"
done
ls "$REL"default/ 2>/dev/null | grep -c '_n[0-9].json$' | sed 's/^/  session files: /'
[ "$SUB" = a3_smoke ] && exit 0
.venv/bin/python -m research.runners._lbf_open_ended_production_turn_probe --a3-score --mode default \
  --seeds 42,43,44,100,101,102 --out-dir "$REL"default --aggregate-out "$REL"default_a3_aggregate.json \
  --code-sha "$SHA" | tail -40
