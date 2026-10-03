#!/usr/bin/env bash
# T22 stage1-host-share: write the exact pool job lines (TSV: command <TAB> --checked reason) for one step, pinned to one
# revision. Deterministic; no judgment. Run from the packet's worktree (research/stage1-host-share). HEAD must be the
# pin, or a descendant that differs from it only under docs/drafts/ and research/findings/raw/.
#
#   bash docs/drafts/stage1-host-share/make_jobs.sh dev  <PIN>   # 22 lines: dev seed 7, every (group, arm)
#   bash docs/drafts/stage1-host-share/make_jobs.sh gate <PIN>   # 132 lines: seeds 42 43 44 100 101 102
#
# Writes docs/drafts/stage1-host-share/jobs/<step>_<pin7>.tsv (S1HS_JOBS_OUTD overrides, for tests).
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$W"
STEP="${1:?step}"; PIN="${2:?pin}"
[ "${#PIN}" -eq 40 ] || { echo "make_jobs: pin must be a full 40-char sha" >&2; exit 2; }
PREREG=research/findings/2026-10-02-stage1-host-share-PREREGISTRATION.md
HEAD="$(git rev-parse HEAD)"
if [ "$HEAD" != "$PIN" ]; then
  git merge-base --is-ancestor "$PIN" HEAD || { echo "make_jobs: pin $PIN is not an ancestor of HEAD" >&2; exit 2; }
  git diff --quiet "$PIN" HEAD -- . ':!docs/drafts' ':!research/findings/raw' || {
    echo "make_jobs: HEAD differs from pin $PIN outside docs/drafts and raw results (code moved; re-pin)" >&2; exit 2; }
fi
git cat-file -e "$PIN:$PREREG" || { echo "make_jobs: $PREREG is not committed at $PIN" >&2; exit 2; }
git cat-file -e "$PIN:research/runners/stage1_host_share.py" || { echo "make_jobs: runner not at $PIN" >&2; exit 2; }
PY="${PY:-.venv/bin/python}"
OUTD="${S1HS_JOBS_OUTD:-docs/drafts/stage1-host-share/jobs}"; mkdir -p "$OUTD"
OUT="$OUTD/${STEP}_${PIN:0:7}.tsv"
case "$STEP" in
  dev)  SIM_NO_PROVENANCE=1 "$PY" -m research.runners.stage1_host_share --emit-jobs "$PIN" --seeds 7 \
          --out-dir research/findings/raw/_stage1_host_share/dev > "$OUT"; WANT=22 ;;
  gate) SIM_NO_PROVENANCE=1 "$PY" -m research.runners.stage1_host_share --emit-jobs "$PIN" \
          --seeds 42,43,44,100,101,102 --out-dir research/findings/raw/_stage1_host_share/gate > "$OUT"; WANT=132 ;;
  *) echo "make_jobs: unknown step $STEP" >&2; exit 2 ;;
esac
N=$(wc -l < "$OUT")
[ "$N" -eq "$WANT" ] || { echo "make_jobs: $OUT has $N lines, expected $WANT" >&2; exit 2; }
echo "$OUT $N lines"
