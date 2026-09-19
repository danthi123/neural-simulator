#!/usr/bin/env bash
# mem_ok.sh — exit 0 IFF starting a job that needs ~<need_gb> would leave a safety margin of
# available RAM. Gate BEFORE launching a heavy-RAM job (a brain-build battery, an integrated
# pipeline, a large sweep) so it is not stacked onto the live training and the desktop into a
# global OOM. This is the "don't stack heavy RAM jobs" check the 2026-09-18 OOM needed.
#
# Usage: tools/mem_ok.sh <need_gb> [safety_margin_gb=8]
#   tools/mem_ok.sh 28   # the flip battery's real footprint -> refuses while training is up
# Pair with memcap.sh: mem_ok decides WHETHER to launch now; memcap bounds the job if it balloons.
set -uo pipefail
NEED="${1:?usage: tools/mem_ok.sh <need_gb> [margin_gb]}"
MARGIN="${2:-8}"
AVAIL=$(free -g | awk '/^Mem:/{print $7}')   # MemAvailable incl. reclaimable cache
LEFT=$(( AVAIL - NEED ))
if [ "$LEFT" -ge "$MARGIN" ]; then
  echo "[mem_ok] OK: avail=${AVAIL}G need=${NEED}G -> ${LEFT}G left (>= ${MARGIN}G margin)"
  exit 0
fi
echo "[mem_ok] ⛔ REFUSE: avail=${AVAIL}G need=${NEED}G -> only ${LEFT}G left (< ${MARGIN}G margin)." >&2
echo "[mem_ok]    Defer this job until the GPU/heavy work frees RAM, run it on the pool/AWS, or shrink it." >&2
exit 1
