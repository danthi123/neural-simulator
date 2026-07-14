# EMERGE graded-confidence console — SCALE de-risk (2026-07-13)

**Question.** Does the graded three-level read + moat (just wired into the talkable console, 12-seed
GO — `2026-07-13-EMERGE-graded-confidence-CONSOLE-wire-in-12seed-GO.md`) HOLD as the concept
inventory grows? The wire-in proved the mechanism on a handful of concepts (1 property). Production
open-domain conversation needs it to survive a larger inventory with more COMPETING properties —
more chances for a spurious CONFIDENT or a collapsed HEDGE.

**Scale tested.** 14 members across 4 emergent categories, 4 taught properties (each taught to ONE
member of its category), 2 category-ambiguous members, 1 never-observed concept:

| category (context) | members | property (taught via ONE member) |
|---|---|---|
| bird (nest) | robin, sparrow, eagle, wren | fly (via robin) |
| fish (river) | trout, pike, bass | swim (via trout) |
| mammal (den) | mole, fox, wolf | walk (via mole) |
| insect (hive) | bee, ant | sting (via bee) |
| ambiguous | bat (nest+den), penguin (nest+river) | — |

**The load-bearing test = HELD-OUT generalization at scale.** Each property is taught to exactly ONE
member of its category; the probe asks a DIFFERENT, held-out member of that category (eagle→fly,
wren→fly, bass→swim, wolf→walk, ant→sting). A CONFIDENT+correct answer means the property was
inherited via the LEARNED category grouping (not memorized), and the graded margin still separates
the inherited property from the now-3 competing properties.

## Result — GO

- **Seed 42 (foreground):** `held_confident=5/5` (all held-out members inherit CONFIDENTLY),
  `amb=[HEDGED, HEDGED]` (bat, penguin), `cross=[ABSTAIN, ABSTAIN]` (trout→fly, eagle→sting — wrong
  category), `moat=MOAT` (griffin), `perm_breaks=True`, `lesion_breaks=True` → GO.
- **12-seed (standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): 12/12 GO** (standard 6/6 +
  fresh 6/6), every seed UNANIMOUS on all four levels + both controls. The pattern held identically
  across all 12 seeds: `held_confident=5/5`, `amb=[HEDGED, HEDGED]`, `cross=[ABSTAIN, ABSTAIN]`,
  `moat=MOAT`, `perm_breaks=True`, `lesion_breaks=True`.

GO gate per seed: all 5 held-out members CONFIDENT+correct AND both ambiguous HEDGED AND both
cross-category ABSTAIN AND unknown=MOAT, with PERMUTED (every member co-occurs equally with all
contexts) degrading held-out confidence AND LESION (coincidence off) driving all-abstain.

## Interpretation

The graded read scales: with 4 competing properties the margin (asked-property drive minus the
strongest competitor) still cleanly separates the three levels — a held-out category member's
inherited property dominates (CONFIDENT), a category-ambiguous member's two candidate properties tie
(HEDGED), and a wrong-category query has no drive (ABSTAIN). The `CONF_MARGIN=30` threshold inherited
from the validated completion probe transfers unchanged. This is generalization-at-scale on the
point-neuron substrate: teach ONE bird can fly → every observed bird confidently inherits it; teach
nothing about a category-ambiguous bat → it honestly hedges.

## Scope / next

- `capacity=84` (66 columns used); the M=192 first pass was correctness-identical but ~5× slower on
  CPU (dense coincidence pool) — a speed choice, not a mechanism change.
- Next: real-corpus-stream threshold calibration (CONFIDENT-vs-HEDGED on a noisier natural
  co-occurrence stream) + larger inventories still.
- NO `sim/` edit (reuse-by-import of `GradedExperientialConsole` + the committed HTM pool-bridge).
  Moat preserved by construction.

## Files

- `research/runners/_emerge_graded_confidence_console_scale_derisk.py`
- Builds on `_emerge_graded_confidence_console_derisk.py` (`GradedExperientialConsole`).
