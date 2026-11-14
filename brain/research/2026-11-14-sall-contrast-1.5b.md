# Pre-registration: speaker contrast with the S_all key, live 1.5B unslept day (six seeds, pool)

Written 2026-11-14 before the run. Record: speaker contrast is an oracle LEAD (36 fresh facts 84 vs 64,
`results/2026-11-14-speaker-contrast-2-1.5b.md`) but NO-GO live with S0 (told +0.00), where S0 fetches the right
episode on about half the told questions; given the right episode it answers more (37 -> 42 of 54,
`2026-11-14-speaker-contrast-live-1.5b.md`). S_all fetches the right episode far more often (11-03 routing 11/12) and
answered 5.83 vs S0 4.83 on the unslept day (`2026-11-14-key-default-1.5b.md`, part A). The two fixes act on
different steps (routing, perspective), so they should add.

**Prerequisites.** Met: S_all + size bar, gain 0.5, `Brain(speaker_contrast=)` now supports S_all: routing reads a kept
plain copy of the change (`route_delta`), so routing is unchanged by construction (test:
`brain/tests/test_episode_switch.py`, similarities equal, survives save/load). Host decision: none.

**Method.** `unslept_day_probe.py --arms c,s,r --speaker-contrast 3`, seeds 42/43/44/100/101/102, pool CPU, paired
with the recorded part A runs (`results/2026-11-14-key-default-1.5b/A/unslept_s*.json`: same code path for the
telling day and the questions, no contrast). Primary arm r (S_all); arm s reported. Table
`brain/experiments/sall_contrast_table.py` (tested).

**Criteria (arm r, six-seed means).** Q0: contrast recorded on every contrast run, none on part A, and r's recalled
episode is the same on >= 95% of paired rows (else VOID). Q1 told (of 12): contrast - tell >= 1 and contrast >= tell
on >= 5/6 seeds. Q2 false: contrast <= tell + 0.5. Q3 self: contrast >= tell - 0.5.
GO -> the mixed-day AWS run (sleep) with S_all + contrast before any default flip.

**Expected.** Part A r 5.83 told with about 9 of 12 told questions routed right; if the contrast lifts
answered-given-right like S0 (+9%), about +1: P1-level uncertain again, but routing should no longer cap it.
