---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-graded-recall-strength-conversation-visible
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_graded_production_flip_verify.py (conversation-visibility + moat + byte-id-off through
  the REAL EpisodicRecallOrgan.recall + recall_disclosure + continuous_engine.consolidate_used_memory) and
  research/runners/_d5_graded_flip_soak.py (4-turn OFF-vs-ON no-regression + a simulated mid-consolidation crash-rollback).
  The production organ EpisodicDapMemory.recall now emits the graded apical reads BESIDE the binary UP-fraction gate
  (verified byte-identical to _apical_up_read); the surfaced read is depth_hold = mean-held max(cp_v_apical − v_hold, 0),
  the substrate's own BTSP instructive signal IS_post.
runner: research/runners/_d5_graded_production_flip_verify.py
external: NO-EXTERNAL-NEEDED — a production-integration verification of the in-repo step-6 graded-read GO; no literature
  question. The graded read realizes the redirect the step-5 finding named and step-6 validated at a weak encode; this
  tests it at the PRODUCTION encode (train_events=40) through the real recall + disclosure.
artifacts:
  - research/findings/raw/_d5_graded_prodflip/summary_6seed.json
  - research/findings/raw/_d5_graded_prodflip/soak_summary_6seed.json
  - research/findings/raw/_d5_graded_prodflip/seed42.json
  - research/findings/raw/_d5_graded_prodflip/seed102.json
---
# The graded apical read makes D5 learn-through-use conversation-visible AT THE PRODUCTION encode — wired default-OFF; the default-ON flip is BLOCKED on emergent-assembly crosstalk (a real no-regression violation on ~1/6 builds)

Artifacts: research/findings/raw/_d5_graded_prodflip/summary_6seed.json (conversation-visibility verify) +
research/findings/raw/_d5_graded_prodflip/soak_summary_6seed.json (no-regression + crash-rollback soak).

**One line.** The prior ledger note said the D5 between-turn consolidation strengthens a memory's robustness reserve but
the handler-visible recall was flat pre/post on 5/6 seeds (the binary UP-fraction saturates), so it was not yet
load-bearing on the conversation. This wires the step-6 GRADED apical read into the PRODUCTION recall and confirms the
flatness is RESOLVED — the surfaced recall STRENGTH rises with use through the real handler on every completing seed —
but a soak surfaces a genuine no-regression violation that BLOCKS the default-on flip: on builds where two emergent
assemblies OVERLAP, consolidating one memory visibly perturbs a neighbor's surfaced strength.

## What was wired (additive; NO sim/ edit; moat preserved by construction)
`EpisodicDapMemory.recall` now emits the three graded apical reads (`depth_rest`/`depth_hold`/`soft`) from the SAME
`cp_v_apical` the binary read thresholds, via a `_apical_dual_read` helper added beside `_apical_up_read` (a verbatim
copy of the step-6 GO instrument). The BINARY UP-fraction + specificity gates STILL decide `in_memory` — verified
byte-identical to a direct `_apical_up_read` (`BINARY_BYTE_ID` 6/6, |Δ|<1e-12). The surfaced number is `depth_hold`
(= IS_post). `recall_disclosure` surfaces the strength ONLY when `BRAIN_D5_CONSOLIDATE` is armed, so with consolidation
OFF (the default) the recall reply is byte-identical to HEAD; the between-turn schedule is set to `_D5_EPISODES=1` (the
graded read rises smoothly at 1 episode/tick, saturates at 3 — step-6: depth_hold GO 5/6 @1 vs 2/6 @3).

## Conversation-visibility at the PRODUCTION encode — the flatness is resolved (verify, cupy, 6 seeds)
<!--derived-->
At the production encode strength (`note_topic`→`store`, train_events=40) the BINARY apical_cue is SATURATED at ~1.0 on
5/6 builds (`binary_moves`=False — exactly the flat signal the old ledger noted), but the graded `depth_hold` MOVES:
after the real `consolidate_used_memory` runs, the surfaced strength rises through the real `org.recall`, e.g.
29.96→30.98 (s42-verify), 30.40→31.90 (s43), 29.09→30.03 (s44) mV. On the 5 seeds whose memory COMPLETES it is
load-bearing 5/5: first-use rise + faithful (perm=nocue=0.000; formation-lesion `depth_hold`→0.000) + moat (never-spoken
'cat' stays in_memory=False, graded ≪ dog) + byte-identical-OFF (store hash before==off, later read identical) + 100%
attributable (ON vs LESION-OFF). Strict monotone-across-3-reuses is 3/6 — the 2 misses are saturating-tail wobbles
≤0.09 mV at the near-saturated ceiling (NOT dead-steps; the first-use rise holds on all 5). The 6th seed (102) self-
ignites (nocue apical UP with no cue) so the binary MOAT correctly abstains (in_memory=False) — the honesty gate working,
not a learn-through-use failure. Verdict: the graded read escapes the binary saturation ceiling and makes the used
memory recall visibly STRONGER at the real op-point.

## The soak: crash-safe + clean on disjoint builds, but a real crosstalk regression on overlapping builds (BLOCKS the flip)
<!--derived-->
A 4-turn conversation (form dog+bird; recall dog → mark → idle-consolidate dog; recall dog/bird/cat) run OFF vs ON, 6
seeds:
- **Crash-rollback intact 6/6**: a simulated mid-consolidation failure (reactivate raises on episode 2, after episode 1
  mutated the persistent store) rolls the store back BYTE-IDENTICALLY (hash_pre==hash_post) and DRAINS the armed topic,
  then re-raises so the tick logs it. The step-4 safety holds under the graded read.
- **No-regression CLEAN on 4/5 completing seeds** (43/44/100/101): the un-consolidated neighbor 'bird' reads
  BYTE-IDENTICAL OFF vs ON (Δ depth_hold = 0.000), 'cat' abstains identically, only 'dog' differs. When the emergent
  dog/bird assemblies are DISJOINT, consolidating dog leaves everything else untouched.
- **A real regression on the 1 overlapping build (s42)**: consolidating dog shifts neighbor 'bird's surfaced strength
  30.77→30.64 mV (a reply-visible change on a topic the user did not touch). `in_memory` is PRESERVED (bird still
  recalls) so it is not a content/abstain break, but the reply's mV number changes. The 4-disjoint-seeds-byte-identical
  vs 1-overlap-seed-wobble contrast proves it is WEIGHT-mediated shared-cell overlap — the only path, since
  consolidation strengthens ONLY dog's within-assembly recurrence. s102 (both assemblies self-ignite) is moat-abstained
  (not a regression). Soak no-regression: 4/6.

## The flip DECISION: NOT flipped (owner bar: any regression blocks) — wired, load-bearing, honestly held OFF
The task bar is explicit: a flip that regresses normal conversation is worse than no flip. The s42 crosstalk is a
genuine reply-visible change on an untouched memory whenever two emergent assemblies happen to overlap, so
`BRAIN_D5_CONSOLIDATE` stays DEFAULT-OFF. The graded read is landed additive + verified byte-identical-off, so shipping
it changes NO default reply. What is now DE-RISKED (and was the prior blocker): the conversation-visibility flatness is
gone. What now BLOCKS the flip (the precise residual): (1) overlapping-emergent-assembly crosstalk — a used memory's
consolidation perturbs an overlapping neighbor's surfaced strength (~1/6 builds); the faithful fix is a separation /
sparsity-set-point on the emergent membership so assemblies stay disjoint (this is the same pattern-separation residual
tracked as board #73), or a crosstalk-robust surfaced read; (2) self-ignition build-reliability — ~1/6 builds produce a
self-igniting assembly the moat abstains, so the effect is demonstrable on ~5/6 builds. Neither is a wall: both are
emergent-assembly-quality residuals, not a defect of the graded read itself. Scope honesty: the surfaced strength is a
faithful spiking read (not a phenomenal claim); the snapshot/restore determinism guard + the single full-strength encode
remain host idealizations.
