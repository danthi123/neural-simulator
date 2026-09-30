#!/usr/bin/env bash
# Integration wave 2: generate the pool job file for one step, pinned to one revision. Deterministic; no judgment.
# Run from the wave-2 WORKTREE (the registry, i.e. which rows exist, is read from the checkout). HEAD must be the pin,
# or a descendant that differs from it only under research/packets/integration-wave2/ and research/findings/raw/.
#
#   bash research/packets/integration-wave2/make_jobs.sh R        <P>                 # R_{fb,tr,si}_<P7>.txt, 12 lines each
#   bash research/packets/integration-wave2/make_jobs.sh flip     <F> <packet,packet> # production default at F
#   bash research/packets/integration-wave2/make_jobs.sh alloff   <F> <faculty> <packet,packet>  # X's 6 cells, all =0
#   bash research/packets/integration-wave2/make_jobs.sh bisect   <F> <faculty> <packet>  # X's 6 cells, one packet =0
#   bash research/packets/integration-wave2/make_jobs.sh identity <PIN> <arm> [<arm> ...]   # dev seed 7 identity arms
#   bash research/packets/integration-wave2/make_jobs.sh capwrap  <need_gb> <cap_gb>  < lines   # the memory wrapper only
#
# Writes research/packets/integration-wave2/jobs/<step>_<pin7>[_<extra>].txt (IW2_JOBS_OUTD overrides the directory,
# for tests) and prints its path and line count.
#
# MEMORY (wave-1 review lessons, built in; CLAUDE.md non-negotiable 10): every line runs its brain-building command
# under `bash tools/mem_ok.sh <need> <margin> && bash tools/memcap.sh <cap> -- env ...`, exactly B2b's shape. LB shards
# (load_bearing_fraction, measured peak ~6 GB in tools/pool_runner_mem.tsv): need 8, cap 12, declared mem_gb=8.
# Identity arms (ONE brain alive at a time: every segment of the script starts with reset): need/cap below, from the
# peak RSS measured on dev seed 7 (HANDOFF section 9); the declared mem_gb in the HANDOFF reasons is IDENT_NEED and
# must stay <= 11 (the largest size the dispatcher hands a 15 GB pool node). Every LB line also starts with the node
# preflight tools/assert_iw2_node.py (refuses, exit 3, without data/corpus/ or a resolvable LTM bundle); the identity
# runner carries the same refusal itself.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$W"
LB_NEED=8; LB_CAP=12
IDENT_NEED=11; IDENT_CAP=15   # peak RSS measured 2026-09-30 (dev seed 7): 7.8-11.0 GB over five arms, HANDOFF section 9
capwrap() {  # insert the memory gate + hard cap in front of the FIRST ` && env ` of each line
  sed "s# \&\& env # \&\& bash tools/mem_ok.sh $1 2 \&\& bash tools/memcap.sh $2 -- env #"
}
if [ "${1:-}" = capwrap ]; then capwrap "${2:?need_gb}" "${3:?cap_gb}"; exit 0; fi
STEP="${1:?step}"; PIN="${2:?pin}"
[ "${#PIN}" -eq 40 ] || { echo "make_jobs: pin must be a full 40-char sha" >&2; exit 2; }
HEAD="$(git rev-parse HEAD)"
# the checkout must BE the pin, or a descendant that differs from it only in this packet directory and raw results
# (the job-file / PINS.tsv / data commits the HANDOFF makes on top of a pin)
if [ "$HEAD" != "$PIN" ]; then
  git merge-base --is-ancestor "$PIN" HEAD || { echo "make_jobs: pin $PIN is not an ancestor of HEAD" >&2; exit 2; }
  git diff --quiet "$PIN" HEAD -- . ':!research/packets/integration-wave2' ':!research/findings/raw' || {
    echo "make_jobs: HEAD differs from pin $PIN outside the packet dir and raw results (code moved; re-pin)" >&2; exit 2; }
fi
P7="${PIN:0:7}"; ROOT="~/derisk-pool/revisions/$PIN"; OUTD="${IW2_JOBS_OUTD:-research/packets/integration-wave2/jobs}"
mkdir -p "$OUTD"
SEEDS="42 43 44 100 101 102"
PY=".venv/bin/python"
nodeguard() {  # insert the degraded-node preflight right after `cd <root> && ` (corpus + LTM bundle; exit 3)
  sed "s#^cd \([^ ]*\) && #cd \1 \&\& $PY tools/assert_iw2_node.py \&\& #"
}
guard() {  # insert the defaults guard right after `cd <root> && `
  sed "s#^cd \([^ ]*\) && #cd \1 \&\& $PY tools/assert_iw2_defaults.py --expect-on=$1 \&\& #"
}
off_env() {  # the explicit-off env for one packet
  case "$1" in
    false-belief-chat) printf ' BRAIN_FALSE_BELIEF_CHAT=0' ;;
    transitive-chat) printf ' BRAIN_TRANSITIVE_CHAT=0' ;;
    neural-selfid) printf ' BRAIN_NEURAL_SELFID=0' ;;
    *) echo "make_jobs: unknown packet $1" >&2; return 2 ;;
  esac
}
case "$STEP" in
  R)  # three files, one per packet (each is queued with its own prereg in --checked): the capability row AND its
      # permuted control row (research/runners/lbf_rows/iw2_controls.py, opt-in by LB_IW2_CONTROLS=1), 12 lines each
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw2-R-fb-$P7" --no-fixes \
      --faculties tom-false-belief tom-true-belief-ctl \
      --extra-env BRAIN_FALSE_BELIEF_CHAT=1 LB_IW2_CONTROLS=1 --root "$ROOT" --pin "$PIN" \
      | nodeguard | capwrap $LB_NEED $LB_CAP > "$OUTD/R_fb_$P7.txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw2-R-tr-$P7" --no-fixes \
      --faculties reasoning-transitive-chat transitive-reversed-ctl \
      --extra-env BRAIN_TRANSITIVE_CHAT=1 LB_IW2_CONTROLS=1 --root "$ROOT" --pin "$PIN" \
      | nodeguard | capwrap $LB_NEED $LB_CAP > "$OUTD/R_tr_$P7.txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw2-R-si-$P7" --no-fixes \
      --faculties neural-selfid selfid-nofact-ctl \
      --extra-env BRAIN_NEURAL_SELFID=1 LB_IW2_CONTROLS=1 --root "$ROOT" --pin "$PIN" \
      | nodeguard | capwrap $LB_NEED $LB_CAP > "$OUTD/R_si_$P7.txt"
    wc -l "$OUTD"/R_*_"$P7".txt; exit 0 ;;
  flip)
    FLIPPED="${3:?comma-separated flipped packets}"
    OUT="$OUTD/flip_$P7.txt"
    $PY tools/assert_iw2_defaults.py --expect-on="$FLIPPED" || { echo "make_jobs: anchors != $FLIPPED" >&2; exit 2; }
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw2-flip-$P7" --no-fixes --root "$ROOT" --pin "$PIN" \
      | guard "$FLIPPED" | nodeguard | capwrap $LB_NEED $LB_CAP > "$OUT" ;;
  alloff|bisect)
    FAC="${3:?faculty}"; PKTS="${4:?packet(s)}"
    [ "$STEP" = bisect ] && case "$PKTS" in *,*) echo "make_jobs: bisect takes ONE packet" >&2; exit 2 ;; esac
    OFF=""
    for PKT in ${PKTS//,/ }; do OFF="$OFF$(off_env "$PKT")"; done
    if [ "$STEP" = alloff ]; then TAG="iw2-alloff-$P7"; else TAG="iw2-bisect-$PKTS-$P7"; fi
    OUT="$OUTD/${STEP}_${P7}_${FAC}$( [ "$STEP" = bisect ] && printf '_%s' "$PKTS" ).txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "$TAG" --no-fixes --faculties "$FAC" \
      --extra-env $OFF --root "$ROOT" --pin "$PIN" | nodeguard | capwrap $LB_NEED $LB_CAP > "$OUT" ;;
  identity)
    shift 2; [ "$#" -ge 1 ] || { echo "make_jobs: identity needs >= 1 arm" >&2; exit 2; }
    OUT="$OUTD/identity_$P7.txt"; : > "$OUT"
    for ARM in "$@"; do
      SAFE="$(printf '%s' "$ARM" | tr ':+' '__')"
      D="research/findings/raw/_iw2/identity/$P7"
      echo "cd $ROOT && mkdir -p $D && bash tools/mem_ok.sh $IDENT_NEED 1 && bash tools/memcap.sh $IDENT_CAP -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -u -m research.runners.iw2_default_identity --arm $ARM --seed 7 --out $D/$SAFE.json" >> "$OUT"
    done ;;
  *) echo "make_jobs: unknown step $STEP" >&2; exit 2 ;;
esac
echo "$OUT $(wc -l < "$OUT") lines"
