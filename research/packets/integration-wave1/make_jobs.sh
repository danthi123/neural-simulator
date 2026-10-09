#!/usr/bin/env bash
# Integration wave 1: generate the pool job file for one step, pinned to one revision. Deterministic; no judgment.
# Run from the wave-1 WORKTREE (the registry, i.e. which rows exist, is read from the checkout). HEAD must be the pin,
# or a descendant that differs from it only under research/packets/integration-wave1/ and research/findings/raw/.
#
#   bash research/packets/integration-wave1/make_jobs.sh R        <P>                 # R_{d6,swm,wmf}_<P7>.txt, 6 lines each
#   bash research/packets/integration-wave1/make_jobs.sh flip     <F> <packet,packet> # production default at F
#   bash research/packets/integration-wave1/make_jobs.sh alloff   <F> <faculty> <packet,packet>  # X's 6 cells, all =0
#   bash research/packets/integration-wave1/make_jobs.sh bisect   <F> <faculty> <packet>  # X's 6 cells, one packet =0
#   bash research/packets/integration-wave1/make_jobs.sh identity <PIN> <arm> [<arm> ...]   # dev seed 7 identity arms
#   bash research/packets/integration-wave1/make_jobs.sh capwrap  <need_gb> <cap_gb>  < lines   # the memory wrapper only
#
# Writes research/packets/integration-wave1/jobs/<step>_<pin7>[_<extra>].txt (IW1_JOBS_OUTD overrides the directory,
# for tests) and prints its path and line count.
#
# MEMORY (adversarial review 2026-09-29; CLAUDE.md non-negotiable 10): every line runs its brain-building command
# under `bash tools/mem_ok.sh <need> <margin> && bash tools/memcap.sh <cap> -- env ...`, exactly the shape B2b's lines
# had. LB shards (load_bearing_fraction, measured peak ~6 GB in tools/pool_runner_mem.tsv): need 8, cap 12. Identity
# arms (two chat sessions in one process; peak measured locally, see the HANDOFF section 9): see IDENT_NEED/IDENT_CAP.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$W"
LB_NEED=8; LB_CAP=12
IDENT_NEED=11; IDENT_CAP=15   # peak RSS measured 2026-09-29 (main, seed 7): see the HANDOFF section 9
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
  git diff --quiet "$PIN" HEAD -- . ':!research/packets/integration-wave1' ':!research/findings/raw' || {
    echo "make_jobs: HEAD differs from pin $PIN outside the packet dir and raw results (code moved; re-pin)" >&2; exit 2; }
fi
P7="${PIN:0:7}"; ROOT="~/derisk-pool/revisions/$PIN"; OUTD="${IW1_JOBS_OUTD:-research/packets/integration-wave1/jobs}"
mkdir -p "$OUTD"
SEEDS="42 43 44 100 101 102"
PY=".venv/bin/python"
guard() {  # insert the defaults guard right after `cd <root> && `
  sed "s#^cd \([^ ]*\) && #cd \1 \&\& $PY tools/assert_iw1_defaults.py --expect-on=$1 \&\& #"
}
case "$STEP" in
  R)  # three files, one per packet (each is queued with its own prereg in --checked)
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw1-R-d6-$P7" --no-fixes --faculties d6-hebbian-write \
      --extra-env BRAIN_D6_HEBBIAN_STORE=1 BRAIN_D6_ENGRAM_VOCAB=1 BRAIN_D6_ENGRAM_READTIME=1 \
      --root "$ROOT" --pin "$PIN" | capwrap $LB_NEED $LB_CAP > "$OUTD/R_d6_$P7.txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw1-R-swm-$P7" --no-fixes --faculties xedge-surprise-worldmodel \
      --extra-env BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=1 --root "$ROOT" --pin "$PIN" | capwrap $LB_NEED $LB_CAP \
      > "$OUTD/R_swm_$P7.txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw1-R-wmf-$P7" --no-fixes --faculties wm-binding-advanced \
      --extra-env LB_WMB_FOCUS_PROBE=1 --root "$ROOT" --pin "$PIN" | capwrap $LB_NEED $LB_CAP > "$OUTD/R_wmf_$P7.txt"
    wc -l "$OUTD"/R_*_"$P7".txt; exit 0 ;;
  flip)
    FLIPPED="${3:?comma-separated flipped packets}"
    OUT="$OUTD/flip_$P7.txt"
    $PY tools/assert_iw1_defaults.py --expect-on="$FLIPPED" || { echo "make_jobs: anchors != $FLIPPED" >&2; exit 2; }
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "iw1-flip-$P7" --no-fixes --root "$ROOT" --pin "$PIN" \
      | guard "$FLIPPED" | capwrap $LB_NEED $LB_CAP > "$OUT" ;;
  alloff|bisect)
    FAC="${3:?faculty}"; PKTS="${4:?packet(s)}"
    [ "$STEP" = bisect ] && case "$PKTS" in *,*) echo "make_jobs: bisect takes ONE packet" >&2; exit 2 ;; esac
    OFF=""
    for PKT in ${PKTS//,/ }; do
      case "$PKT" in
        d6-hebbian-write) OFF="$OFF BRAIN_D6_HEBBIAN_STORE=0 BRAIN_D6_ENGRAM_VOCAB=0 BRAIN_D6_ENGRAM_READTIME=0" ;;
        wm-focus-bind) OFF="$OFF BRAIN_MULTIREF_FOCUS_BIND=0" ;;
        xedge-surprise-worldmodel) OFF="$OFF BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=0" ;;
        *) echo "make_jobs: unknown packet $PKT" >&2; exit 2 ;;
      esac
    done
    if [ "$STEP" = alloff ]; then TAG="iw1-alloff-$P7"; else TAG="iw1-bisect-$PKTS-$P7"; fi
    OUT="$OUTD/${STEP}_${P7}_${FAC}$( [ "$STEP" = bisect ] && printf '_%s' "$PKTS" || true ).txt"
    $PY tools/lb_shard.py jobs --seeds $SEEDS --tag "$TAG" --no-fixes --faculties "$FAC" \
      --extra-env $OFF --root "$ROOT" --pin "$PIN" | capwrap $LB_NEED $LB_CAP > "$OUT" ;;
  identity)
    shift 2; [ "$#" -ge 1 ] || { echo "make_jobs: identity needs >= 1 arm" >&2; exit 2; }
    OUT="$OUTD/identity_$P7.txt"; : > "$OUT"
    for ARM in "$@"; do
      SAFE="$(printf '%s' "$ARM" | tr ':+' '__')"
      D="research/findings/raw/_iw1/identity/$P7"
      echo "cd $ROOT && mkdir -p $D && bash tools/mem_ok.sh $IDENT_NEED 1 && bash tools/memcap.sh $IDENT_CAP -- env SIM_BACKEND=numpy PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -u -m research.runners.iw1_default_identity --arm $ARM --seed 7 --out $D/$SAFE.json" >> "$OUT"
    done ;;
  *) echo "make_jobs: unknown step $STEP" >&2; exit 2 ;;
esac
echo "$OUT $(wc -l < "$OUT") lines"
