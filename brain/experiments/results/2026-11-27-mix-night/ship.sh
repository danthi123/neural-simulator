#!/bin/bash
# W side: ship brain/ (HEAD, clean) to pool40/41/42 and drive the mix-night screen there. NOT run yet (prereg REGISTERED 11-27).
#   bash ship.sh ship       code + COMMIT + run_node.sh, and each node's foreign A5f files (3 small episodes.json from the
#                           next seed's node: pool40 gets s320's, pool41 s321's, pool42 s319's). Refuses while any
#                           brain.experiments process is live on a node (shipping would move COMMIT and code under it).
#   bash ship.sh strata     store-only P2 stratum sizes on every node (seconds)
#   bash ship.sh measure    step 0 on pool40 only (s319: A0r, A0 n13, A2 n13 under /usr/bin/time -v + determinism pair);
#                           NODES="pool41 pool42" MEMCAP_GB=10 for the 15 GB nodes (required there). THREADS=1 = fallback.
#                           MEMCAP_GB / THREADS unset are passed EMPTY; run_node.sh reads empty as unset (${VAR:-default},
#                           [ -z ]), tested in test_run_node_empty_memcap_and_threads_mean_default.
#   bash ship.sh run        the full screen on all three nodes (needs each node's measure file)
#   bash ship.sh status     done files
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) && cd "$W" || exit 1
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=; M=$1
D=brain-lane/route_repo/_scratch/route/kr
MAP="pool40:319 pool41:320 pool42:321"
node_of() { for x in $MAP; do [ "${x#*:}" = "$1" ] && echo "${x%:*}"; done; }
case $M in
ship)
  [ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
  H=$(git rev-parse HEAD)
  for x in $MAP; do N=${x%:*}
    # also mn_run_node: between arms (and while an arm waits on mem_ok) no python is live, but the driver is
    LIVE=$($E $N "pgrep -u \$USER -af 'brain.experiments|mn_run_node' | grep -v pgrep" </dev/null)
    [ -z "$LIVE" ] || { echo "$N has live brain.experiments / mn_run_node processes; ship later:"; echo "$LIVE"; exit 1; }
  done
  for x in $MAP; do N=${x%:*}; S=${x#*:}
    $E $N "test -d ~/$D/snaps_s$S" </dev/null || { echo "$N does not hold snaps_s$S"; exit 1; }
    rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
    echo $H | $E $N "cat > ~/brain-lane/route_repo/COMMIT" || exit 1
    rsync -a -e "$E" brain/experiments/results/2026-11-27-mix-night/run_node.sh $N:$D/mn_run_node.sh || exit 1
  done
  for x in $MAP; do N=${x%:*}; S=${x#*:}
    case $S in 319) F=320;; 320) F=321;; 321) F=319;; esac; FN=$(node_of $F)
    $E $FN "cd ~/$D/snaps_s$F && tar -c n13/episodes.json n14/episodes.json n15/episodes.json" </dev/null \
      | $E $N "mkdir -p ~/$D/foreign_s$F && tar -x -C ~/$D/foreign_s$F" || { echo "foreign s$F -> $N failed"; exit 1; }
    A=$($E $FN "cd ~/$D/snaps_s$F && sha256sum n1[345]/episodes.json" </dev/null)
    B=$($E $N "cd ~/$D/foreign_s$F && sha256sum n1[345]/episodes.json" </dev/null)
    [ -n "$A" ] && [ "$A" = "$B" ] || { echo "foreign s$F on $N differs from $FN"; exit 1; }
    echo "$N: code $H, foreign s$F ok"
  done ;;
strata|measure|run)
  L=${NODES:-pool40 pool41 pool42}; [ "$M" = measure ] && L=${NODES:-pool40}
  for N in $L; do
    if [ "$M" = strata ]; then $E $N "bash ~/$D/mn_run_node.sh strata" </dev/null
    else
      # refusals in run_node.sh are echo + exit 2: log them, and show the log, so a run cannot look launched when it is not
      LG="~/$D/mix_night/launch_$M.log"
      $E $N "mkdir -p ~/$D/mix_night; (MEMCAP_GB=$MEMCAP_GB THREADS=$THREADS setsid nohup bash ~/$D/mn_run_node.sh $M >> $LG 2>&1 &); sleep 5; echo '-- $N launch log:'; tail -5 $LG; tail -2 ~/$D/mix_night/done_s*.txt 2>/dev/null" </dev/null
    fi
  done ;;
status)
  for N in pool40 pool41 pool42; do echo "== $N"; $E $N "tail -4 ~/$D/mix_night/done_s*.txt; cat ~/$D/mix_night/measure_s*.txt 2>/dev/null" </dev/null; done ;;
*) echo "usage: ship.sh ship|strata|measure|run|status"; exit 2;;
esac
