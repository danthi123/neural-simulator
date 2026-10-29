---
type: result
status: live
lane: stage2-C3
verdict: PASS
date: 2026-10-29
packet: c3-topic-stop
prereg: research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md
scorer: research/runners/c3_topic_stop_score.py --mode gate
---
# C3 per-topic stop (L+P) at the six gate seeds: PASS (RESULT, 2026-10-29)

host-decision: scaffold
decides: unchanged from the prereg. Every ask is still decided by host arithmetic at its last step (prereg section 9 (f)); this
result does not retire that threshold.

Scored by Claude from checkout R (`/opt/data/cache/wt-c3F`) with the prereg's command, verbatim:

    .venv/bin/python -m research.runners.c3_topic_stop_score --mode gate \
        --root research/findings/raw/_c3_topic_stop/gate --seeds 42,43,44,100,101,102 \
        --pin 34a4e5f756024b4f5aa5d145b3b29661ad976d0b --out research/findings/raw/_c3_topic_stop/gate/verdict.json

## Verdict

<!--derived-->
DEC **PASS**: every bar G1 to G10 holds on 6/6 gate seeds (96/96 cells DEFINED, every sidecar at pin 34a4e5f75). G4's
attributable fraction is 1.0 (17 treatment toad asks against 0 control; reversed treatment 16, reported). By prereg section 1,
L+P (`BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_PHASIC_DA=1`) now qualifies for a default-ON flip. That flip is a separate,
later step with its own prereg (smoke-first records, flag-off battery arms, answer rate) before the 11-21 Stage-2 re-pin.
Nothing is flipped by this file.

The identity canary that released the gate (`research/findings/raw/_c3_topic_stop/canary/identity_canary.json`, seeds 7 and 8, 32/32 arms IDENTICAL against the dev
shards) now carries a `preconditions` block (tools.verdict, five checks, all measured and held). It is committed with these shards.
Two stray partial writes (the `ask__C_reversed` partial-write pair under gate seed 100) are excluded. The complete shard of
that cell is present and was scored.

## Informational reads (prereg section 7), at the gate seeds

<!--derived-->
- Caveat (a) holds at the gate seeds. Lpinned (L alone, the host gain pinned to 1.0) passes every C3 clause in both orders on
  6/6. So P is shown necessary only against the gain formula's floor clip (G5), not for the per-topic stop itself.
- Caveat (b) holds. pconst (salience replaced by a constant) also passes in both orders on 6/6. The phasic drive's
  topic-locking is not what carries C3 here.
- The flag-OFF arms fail their C3 clauses in both orders on 6/6. This is the failure L+P addresses.
- The two G3 conventions disagree as section 9 (d) predicted. With its own moat convention, G3 shows moat 0.0 and gap 1.0. With
  the Stage-2 aggregate convention (`aggconv`) it shows moat 0.5 and gap 1.0, and it discriminates and tracks on 6/6. The
  margin `gap - 2 x moat` is 0.0 on every seed under aggconv. That margin is zero, not positive. Before the re-pin, the Stage-2
  retake must decide which convention binds, because under aggconv the permuted control's margin has no headroom.

## What this does not show

The prereg's section 9 stands unchanged. The ask is still a host threshold, and its spiking replacement is prereg section 8, not
built. Caveats (a) and (b) above mean the gate validates L+P as a package, not each part's necessity. Default flips are a separate
step, and so is the verify-go panel that step requires.
