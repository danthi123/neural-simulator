---
type: finding
status: measured
claim_check: measured
date: 2026-09-05
mechanism: rank-2 integrated one-brain loop (spiking cue-match -> composer recall) at V=320 / K=32 production scale, 6-seed answer-preservation vs the host reference
lane: H · Memory / integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partB.json
runner: research/runners/_rank2_integrated_loop_webapp_thread_derisk.py
verdict: >
  UNDEFINED (answer-preservation NOT met at production scale), but the divergence is in the SAFE direction and
  the diagnosis is sharp. On the integrated loop at V=320 facts / K=32 (6-seed), the spiking-sequence path matches
  the host's recalled PATIENT on the large majority of cues (24-31 of 35 per seed) but ABSTAINS ("unknown") where
  the host returns a definite yes/no on a substantial minority (6-14 of 35 per seed) — so exact
  answer-preservation fails (17-26 of 35 answers match exactly), the gate reads UNDEFINED, and overall_go is False. Crucially
  the failure is RECALL-INCOMPLETENESS, not confabulation: the patient is mostly right and the extra "unknown"s are
  honest abstains (the honesty-boundary-preserving direction), with only ONE moat false-alarm across all six seeds
  (seed 101). This is a METHOD verdict at scale, not a capability wall: the recall-completeness gap connects
  directly to the composer's retrieval-completeness, and the DG-CA3 sublinear spiking retrieval (de-risked GO with
  recall 404/404 at 404 facts) whose production wire-in is IN FLIGHT is the plausible fix — the right next step is
  to RE-VERIFY rank-2 at scale AFTER the sharded composer is wired to the live path. The single moat FA (seed 101)
  is the one genuine slip to watch. Not a production flip (the integrated loop plumbing stays default-off).
---

# Rank-2 integrated one-brain loop at production scale: answer-preservation UNDEFINED — the gap is honest recall-incompleteness, and it points at the composer

## What ran
`research/runners/_rank2_integrated_loop_webapp_thread_derisk.py --skip-mechanical` (Part B, production V=320 / K=32),
6-seed on the GPU queue, harvesting `research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partB.json`.
This is the at-scale re-verify of the integrated loop (spiking cue-match selection -> composer recall wired to the
production path) the earlier plumbing de-risk named as its next rung.

## Result (6 seeds, V=320, K=32; all counts are direct reads of the cited artifact)
Per seed, of 35 probe cues: **answers exactly matching host (the runner's answer_identical field) = 17-26**; **patient (recalled object) matches host =
24-31**; **spiking path returns "unknown" where host is definite = 6-14**; **moat false-alarms = 0 except seed 101
(=1)**. The gate is UNDEFINED because the `seedXX_answer_identical` preconditions are unmet on all six seeds (and
`seed101_moat_fa_zero` is unmet). `overall_go: False`.

## Reading it (no-defer)
The divergence is dominated by the spiking path ABSTAINING ("unknown") on cues the host answers — i.e. it recalls
the right structure (patient matches on the large majority) but does not always complete the yes/no read at
V=320. That is recall-INCOMPLETENESS, and it is the honesty-preserving direction (an honest "unknown" beats a
confident wrong answer); the near-total absence of moat false-alarms (1 in 210 cue-seeds) confirms the
no-confabulation guarantee essentially holds even where recall is incomplete. This is a METHOD verdict at scale,
not a capability wall (THE LAW): the recall-completeness gap is the SAME quantity the composer's retrieval owns,
and the DG-CA3 sublinear SPIKING retrieval (`2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md`,
recall 404/404 at 404 facts) whose production wire-in is in flight is the plausible closer. NEXT: re-verify rank-2
at V=320 AFTER the sharded composer is wired to the live recall path; separately track the single seed-101 moat FA.

## Honest scope
Not a production flip — the integrated-loop plumbing stays default-off. The at-scale answer divergence is real and
must close before any flip; this finding characterizes it (honest abstains, not confabulation) and names the fix
(sharded composer recall) rather than tuning the abstain threshold, which would trade the honesty margin for a
cosmetic identity match.
