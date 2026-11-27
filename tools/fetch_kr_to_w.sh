#!/usr/bin/env bash
# fetch_kr_to_w.sh -- put the KR DEV seeds' data on the workstation for the mix-night screen on the W 3090
# (brain/research/2026-11-27-mix-night-prereg.md, Amendment S2; spec tools/engine/specs/mix_night.yaml, node w3090).
#
#   bash tools/fetch_kr_to_w.sh            # copy, build foreign stores, link the model, verify (idempotent: rsync resumes)
#   bash tools/fetch_kr_to_w.sh --check    # verify only (what the engine's `requires` will look for), copy nothing
#
# From pool40 (holds all three KR seeds since 11-25, tools/engine/nodes.json kr_seeds), remote
#   /home/node/brain-lane/route_repo/_scratch/route/kr/{lifeKR_s<S>.json.json, snaps_s<S>/, trunk_s<S>/}, S = 319 320 321
# to $HOME/brain-lane/route_repo/_scratch/route/kr on W (the same layout the engine's `requires` and cmd use, under $HOME).
# Then, as ship.sh does node to node, each seed's A5f foreign store foreign_s<F>/n13..n15/episodes.json is built from the
# local snaps_s<F> (319 -> F 320, 320 -> 321, 321 -> 319), and $HOME/brain-lane/models/<1.5b> is a symlink to
# /home/dant123/models/rwkv7 (the engine checks brain-lane/models/<model> and runs with BRAIN_MODEL_DIR there).
# Disk: checks the remote size (du) against free space under $HOME first, and refuses below size + 5 GB.
# Reads only from pool40 (ssh with the agent socket off, as every pool tool here); writes only under ~/brain-lane.
set -euo pipefail
NODE=${NODE:-pool40}
RDIR=/home/node/brain-lane/route_repo/_scratch/route/kr
LDIR=$HOME/brain-lane/route_repo/_scratch/route/kr
MODEL=rwkv7-g1l-1.5b-20261031-ctx34304.pth
MSRC=/home/dant123/models/rwkv7/$MODEL
SEEDS="319 320 321"
export SSH_AUTH_SOCK=
SSH="ssh -o IdentitiesOnly=yes -o BatchMode=yes"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
foreign_of() { case $1 in 319) echo 320;; 320) echo 321;; 321) echo 319;; esac; }
W_LIFE_319=_scratch/brainv2/aws/20261124-1219/lifeKR_s319.json.json   # the driver-side copies (spec vars.lives)
W_LIFE_320=_scratch/brainv2/aws/20261124-1624/lifeKR_s320.json.json
W_LIFE_321=_scratch/brainv2/aws/20261124-1704/lifeKR_s321.json.json

check() {   # every path mix_night.yaml `requires` names for each seed, under $HOME; plus life JSON identity
  local bad=0 S F f w
  for S in $SEEDS; do F=$(foreign_of "$S")
    for f in "lifeKR_s$S.json.json" "snaps_s$S/n12/cortex_adapters.pt" "snaps_s$S/n15/episodes.json" \
             "foreign_s$F/n13/episodes.json" "foreign_s$F/n14/episodes.json" "foreign_s$F/n15/episodes.json"; do
      [ -e "$LDIR/$f" ] || { echo "MISSING $LDIR/$f"; bad=1; }
    done
    for k in 12 13 14 15; do for f in episodes.json keys.pt meta.json keep_bank.json; do
      [ -e "$LDIR/snaps_s$S/n$k/$f" ] || { echo "MISSING $LDIR/snaps_s$S/n$k/$f"; bad=1; }; done; done
    for k in 13 14 15; do
      if [ -e "$LDIR/foreign_s$F/n$k/episodes.json" ] && \
         ! cmp -s "$LDIR/foreign_s$F/n$k/episodes.json" "$LDIR/snaps_s$F/n$k/episodes.json"; then
        echo "DIFFERS foreign_s$F/n$k/episodes.json vs snaps_s$F"; bad=1
      fi
    done
    w="W_LIFE_$S"; w="$REPO/${!w}"
    if [ -e "$w" ] && [ -e "$LDIR/lifeKR_s$S.json.json" ] && ! cmp -s "$w" "$LDIR/lifeKR_s$S.json.json"; then
      echo "DIFFERS lifeKR_s$S.json.json (pool40 copy) vs the driver-side $w"; bad=1
    fi
  done
  [ -e "$HOME/brain-lane/models/$MODEL" ] || { echo "MISSING $HOME/brain-lane/models/$MODEL"; bad=1; }
  if [ "$bad" != 0 ]; then return 1; fi
  echo "fetch_kr_to_w: all KR data for $SEEDS present under $LDIR; model linked"
}

[ "${1:-}" = "--check" ] && { check; exit $?; }
[ -z "${1:-}" ] || { sed -n 2,6p "$0"; exit 2; }
[ -s "$MSRC" ] || { echo "refuse: model $MSRC missing"; exit 2; }

# 1. disk: remote size of what will be copied vs free space here (+ 5 GB headroom)
paths=""; for S in $SEEDS; do paths="$paths $RDIR/lifeKR_s$S.json.json $RDIR/snaps_s$S $RDIR/trunk_s$S"; done
need_kb=$($SSH "$NODE" "du -sck $paths | tail -1 | cut -f1" </dev/null) || { echo "refuse: cannot size the data on $NODE (missing path?)"; exit 3; }
mkdir -p "$LDIR"
free_kb=$(df -Pk "$LDIR" | awk 'NR==2{print $4}')
echo "fetch_kr_to_w: $NODE holds $((need_kb / 1048576)) GB for seeds $SEEDS; free here $((free_kb / 1048576)) GB"
[ "$free_kb" -gt $((need_kb + 5 * 1048576)) ] || { echo "refuse: not enough free disk (need size + 5 GB)"; exit 3; }

# 2. copy (rsync resumes; --partial keeps a broken transfer's progress)
for S in $SEEDS; do
  rsync -a --partial -e "$SSH" "$NODE:$RDIR/lifeKR_s$S.json.json" "$LDIR/"
  rsync -a --partial -e "$SSH" "$NODE:$RDIR/snaps_s$S" "$NODE:$RDIR/trunk_s$S" "$LDIR/"
  echo "  s$S copied"
done

# 3. foreign A5f stores from the local snapshots (what ship.sh ships node to node)
for S in $SEEDS; do F=$(foreign_of "$S")
  for k in 13 14 15; do
    mkdir -p "$LDIR/foreign_s$F/n$k"; cp -p "$LDIR/snaps_s$F/n$k/episodes.json" "$LDIR/foreign_s$F/n$k/episodes.json"
  done
done

# 4. the model where the engine looks for it on the w3090 node
mkdir -p "$HOME/brain-lane/models"
[ -e "$HOME/brain-lane/models/$MODEL" ] || ln -s "$MSRC" "$HOME/brain-lane/models/$MODEL"

check
