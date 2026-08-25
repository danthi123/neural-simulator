---
type: finding
status: contributing
date: 2026-08-26
mechanism: gap2-spiking-deltarule-binder
lane: gap#2
seeds: [42, 43, 44, 100, 101, 102]
---

# The spiking delta-rule binder recalls role->filler above the permuted-role control (gap#2, board #132)

## Claim
A spiking delta-rule binder learns role->filler associations and recalls them correctly, and the recall is
load-bearing on the CORRECT role binding — a role-permuted probe does not recall. This closes the gap#2 binder
de-risk at 6 seeds.

## Result
`research/runners/_gap2_spiking_deltarule_binder_derisk` (256 codes, D=128), 6 seeds (42/43/44/100/101/102):
aggregate `research/findings/raw/_harvest_2026_08_26/gap2_binder_6seed_agg.json` (source `gap2_binder_6seed.json`) -> `go: true`.

For every bound-pair count P in {1,2,3,4}:
- delta-rule recall = 1.0 (perfect role->filler recall),
- the decorrelated-code control = 1.0 (recall is not an overlap artifact of correlated codes),
- the PERMUTED-role control = 0.0 (querying with the wrong role recalls nothing) — the binding, not mere
  co-activation, is what carries the answer,
- an additive-binding reference = 1.0.

## Instrument + control
- Instrument: role->filler recall accuracy after delta-rule learning.
- Controls: the permuted-role probe (0.0) is the discriminating control — it isolates that the delta rule bound
  the specific role, not the filler's general presence; the decorrelated-code arm rules out a correlated-code
  shortcut.

## Next (no-defer)
The binder de-risk is GO. The next rung toward one-brain is co-residency: run the delta-rule binder on the shared
spiking substrate (its own bridge merged with the composer's) rather than as a standalone runner, and drive it
from live conversation role slots so binding happens through use.
