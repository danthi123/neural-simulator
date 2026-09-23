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
# COMMITTED-BUT-UNUSED headroom of already-running memcap scopes (2026-09-23). Each job's cap is safe alone, but
# ~10 concurrent agents each passed this check at launch time and then GREW into their caps together: free RAM fell to
# 3 GB with 19 runners (caps summed past physical RAM). Count a fraction (MEM_OK_COMMIT_WEIGHT, default 0.5) of every
# running scope's (MemoryMax - MemoryCurrent) as already spoken for.
COMMIT_GB=0
if command -v systemctl >/dev/null 2>&1; then
  [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ] && export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  COMMIT_GB=$(systemctl --user list-units --type=scope --state=running --no-legend 'run-*.scope' 2>/dev/null \
    | awk '{print $1}' | while read -r u; do
        # query separately: `show -p A -p B` prints in systemd's own order, not the order asked
        printf '%s %s\n' "$(systemctl --user show "$u" -p MemoryMax --value 2>/dev/null)" \
                          "$(systemctl --user show "$u" -p MemoryCurrent --value 2>/dev/null)"
      done | awk -v w="${MEM_OK_COMMIT_WEIGHT:-0.5}" '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $1 > $2 {s += ($1 - $2)} END {printf "%d", w * s / 1073741824}')
  COMMIT_GB=${COMMIT_GB:-0}
fi
RAW_AVAIL=$AVAIL
AVAIL=$(( AVAIL - COMMIT_GB ))
LEFT=$(( AVAIL - NEED ))
if [ "$LEFT" -ge "$MARGIN" ]; then
  echo "[mem_ok] OK: avail=${AVAIL}G (raw ${RAW_AVAIL}G - ${COMMIT_GB}G committed-unused) need=${NEED}G -> ${LEFT}G left (>= ${MARGIN}G margin)"
  exit 0
fi
echo "[mem_ok] ⛔ REFUSE: avail=${AVAIL}G (raw ${RAW_AVAIL}G - ${COMMIT_GB}G committed-unused) need=${NEED}G -> only ${LEFT}G left (< ${MARGIN}G margin)." >&2
echo "[mem_ok]    Defer this job until the GPU/heavy work frees RAM, run it on the pool/AWS, or shrink it." >&2
exit 1
