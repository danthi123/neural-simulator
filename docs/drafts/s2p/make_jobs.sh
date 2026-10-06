#!/usr/bin/env bash
# Session-2 host-decision packets: generate the exact pool job lines for one packet and step, pinned to one revision.
# Deterministic; no judgment. Run from the packet's worktree (research/<packet>, cut from main after the owner landed
# claude/session2-packets). HEAD must be the pin, or a descendant that differs from it only under docs/drafts/ and
# research/findings/raw/.
#
#   bash docs/drafts/s2p/make_jobs.sh <packet> sweep <PIN>        # 8 lines: dev seeds 1-8, every GRID index
#   bash docs/drafts/s2p/make_jobs.sh <packet> O     <PIN> <OP>   # 6 lines: gate seeds, the selected index
#   bash docs/drafts/s2p/make_jobs.sh <packet> R     <PIN> <OP>   # 12 lines: LB shards, capability + control row
#
# <packet> is fb-spiking-witness | transitive-spiking-reach | word-role-parse | word-role-commit. Writes docs/drafts/<packet>/jobs/
# <step>_<pin7>.txt (S2P_JOBS_OUTD overrides, for tests) and prints its path and line count. Memory: organ lines need
# 2 GB (measured peak well under 1 GB on dev seeds), capped at 4; LB shards as integration wave 2 (need 8, cap 12).
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$W"
PKT="${1:?packet}"; STEP="${2:?step}"; PIN="${3:?pin}"; OP="${4:-}"
[ "${#PIN}" -eq 40 ] || { echo "make_jobs: pin must be a full 40-char sha" >&2; exit 2; }
case "$PKT" in
  fb-spiking-witness)
    MOD=research.runners.tom_spiking_witness_organ; T=fb
    ROWS="fb-spiking-witness fb-spiking-true-belief-ctl"
    ENV="BRAIN_FALSE_BELIEF_CHAT=1 BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1 BRAIN_FALSE_BELIEF_WITNESS_OP=$OP"
    PREREG=research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md ;;
  transitive-spiking-reach)
    MOD=research.runners.transitive_spiking_reach_organ; T=tr
    ROWS="transitive-spiking-reach transitive-spiking-reversed-ctl"
    ENV="BRAIN_TRANSITIVE_CHAT=1 BRAIN_TRANSITIVE_SPIKING_REACH=1 BRAIN_TRANSITIVE_REACH_OP=$OP"
    PREREG=research/findings/2026-10-01-transitive-spiking-reach-PREREGISTRATION.md ;;
  word-role-parse)
    MOD=research.runners.word_role_parser_organ; T=wr
    ROWS="word-role-selfid word-role-nofact-ctl"
    ENV="BRAIN_NEURAL_SELFID=1 BRAIN_WORD_ROLE_PARSE=1 BRAIN_WORD_ROLE_OP=$OP"
    PREREG=research/findings/2026-10-01-word-role-parse-PREREGISTRATION.md ;;
  word-role-commit)
    MOD=research.runners.word_role_commit_organ; T=wc
    ROWS="word-role-commit-selfid word-role-commit-nofact-ctl"
    ENV="BRAIN_NEURAL_SELFID=1 BRAIN_WORD_ROLE_PARSE=1 BRAIN_WORD_ROLE_COMMIT=1 BRAIN_WORD_ROLE_COMMIT_OP=$OP"
    PREREG=research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md ;;
  *) echo "make_jobs: unknown packet $PKT" >&2; exit 2 ;;
esac
HEAD="$(git rev-parse HEAD)"
if [ "$HEAD" != "$PIN" ]; then
  git merge-base --is-ancestor "$PIN" HEAD || { echo "make_jobs: pin $PIN is not an ancestor of HEAD" >&2; exit 2; }
  git diff --quiet "$PIN" HEAD -- . ':!docs/drafts' ':!research/findings/raw' || {
    echo "make_jobs: HEAD differs from pin $PIN outside docs/drafts and raw results (code moved; re-pin)" >&2; exit 2; }
fi
git cat-file -e "$PIN:$PREREG" || { echo "make_jobs: $PREREG is not committed at $PIN" >&2; exit 2; }
P7="${PIN:0:7}"; ROOT="~/derisk-pool/revisions/$PIN"; OUTD="${S2P_JOBS_OUTD:-docs/drafts/$PKT/jobs}"
RAW="research/findings/raw/_s2p/$PKT/$P7"
PY=".venv/bin/python"
ENVC="SIM_BACKEND=numpy PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
mkdir -p "$OUTD"
OUT="$OUTD/${STEP}_$P7.txt"; : > "$OUT"
case "$STEP" in
  sweep)
    for S in 1 2 3 4 5 6 7 8; do
      echo "cd $ROOT && mkdir -p $RAW/sweep && bash tools/mem_ok.sh 2 1 && bash tools/memcap.sh 4 -- env $ENVC $PY -u -m $MOD --sweep --seeds $S --out-dir $RAW/sweep" >> "$OUT"
    done ;;
  O)
    [ -n "$OP" ] || { echo "make_jobs: O needs the selected operating point" >&2; exit 2; }
    for S in 42 43 44 100 101 102; do
      echo "cd $ROOT && mkdir -p $RAW/O && bash tools/mem_ok.sh 2 1 && bash tools/memcap.sh 4 -- env $ENVC $PY -u -m $MOD --seed $S --op $OP --allow-gate-seed --prereg $PREREG --out $RAW/O/s$S.json" >> "$OUT"
    done ;;
  R)
    [ -n "$OP" ] || { echo "make_jobs: R needs the selected operating point" >&2; exit 2; }
    # the integration-wave-2 shape (research/packets/integration-wave2/make_jobs.sh R): node preflight + memory wrapper
    $PY tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag "s2p-R-$T-$P7" --no-fixes --faculties $ROWS \
      --extra-env $ENV --root "$ROOT" --pin "$PIN" \
      | sed "s#^cd \([^ ]*\) && #cd \1 \&\& $PY tools/assert_iw2_node.py \&\& #" \
      | sed "s# \&\& env # \&\& bash tools/mem_ok.sh 8 2 \&\& bash tools/memcap.sh 12 -- env #" > "$OUT" ;;
  *) echo "make_jobs: unknown step $STEP" >&2; exit 2 ;;
esac
echo "$OUT $(wc -l < "$OUT") lines"
