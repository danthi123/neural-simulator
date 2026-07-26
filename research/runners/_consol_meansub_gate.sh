#!/usr/bin/env bash
# 6-seed GO gate for Miller-MacKay subtractive normalization on the cortical store.
#
# ARM      : --mean-subtract 1.0   (the mechanism)
# LESION   : --mean-subtract absent (== 0.0, the shipped behaviour) — so the claim tested is the
#            MECHANISM, not the operating point. Both arms run at the identical stable op-point.
#
# GATE (all four, per .claude/skills/verify-go/SKILL.md — a ratio alone is NOT a result):
#   1. own-is-max on >=2/3 facts, own/other >= 2.5                     [selectivity]
#   2. PERMUTED-TARGET control collapses to ~1.0                        [not a winner-slot artifact]
#   3. per-slot mass reported RAW (unbalanced mass does not by itself   [mass triad]
#      invalidate, but a high ratio on the HEAVY slot only does)
#   4. substrate physiological THROUGHOUT the write                     [arm not VOID]
#   + (B) hippo-lesioned recall = the capability-relevant half, not just the weight read.
#
# Serial by design: parallel GPU arms silently died on VRAM contention (exit 0, empty output) three
# times on 2026-07-26. grep is --line-buffered and the timeout is generous so a mid-flight kill
# cannot discard buffered output and masquerade as a null.
set -u
cd /home/dant123/Projects/sim
for MS in "--mean-subtract 1.0" ""; do
  LABEL=$([ -n "$MS" ] && echo "ARM(mean-sub)" || echo "LESION(none)")
  for S in 42 43 44 100 101 102; do
    echo "### $LABEL seed $S"
    timeout 2400 env SIM_BACKEND=cupy .venv/bin/python \
      -m research.runners._consol_cortical_store_probe \
      --seed "$S" --teaching-clamp --elig-tau 30 --freeze-gap --no-stdp \
      --hebbian-max-w 2.5 $MS 2>&1 \
      | grep -E --line-buffered "permuted control|A2\)|per-slot mass=\[[0-9]|INVALID|physiological THROUGH|HIPPO-LESIONED|VERDICT"
    echo "   rc=$?"
  done
done
echo "MEANSUB-GATE DONE"
