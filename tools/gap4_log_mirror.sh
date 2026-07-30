#!/usr/bin/env bash
# Mirror the crux logs from /tmp (volatile, wiped on reboot) into the repo every 5 min.
# WHY: the runner writes its JSON only AFTER all arms finish, so for a multi-day run the ONLY record of a
# completed arm is its printed log line -- and that line lived solely in /tmp. A reboot (e.g. into Windows for
# gaming) would have destroyed 22.6 h of completed work that was never at risk from the compute itself.
cd /home/dant123/Projects/sim || exit 0
mkdir -p research/findings/raw/gap4/logs
while true; do
  for f in /tmp/claude-1000/gap4*.log; do
    [ -f "$f" ] && cp -u "$f" "research/findings/raw/gap4/logs/$(basename "$f")" 2>/dev/null
  done
  sleep 300
done
