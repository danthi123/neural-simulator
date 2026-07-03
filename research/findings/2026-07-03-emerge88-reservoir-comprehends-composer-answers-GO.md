# EMERGE-88 — FUNCTIONAL INTEGRATION: the form→role reservoir COMPREHENDS → the composer ANSWERS — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge88_reservoir_comprehends_composer_answers_derisk.py`
**Test:** `tests/test_emerge88_reservoir_comprehends_composer_answers.py`
**Raw:** `research/findings/raw/_emerge88_reservoir_comprehends.json`

## Why (closing the loop — comprehension DRIVES composition)

The whole EMERGE-78..87 arc de-risked the fronto-striatal reservoir as a comprehension **mechanism** — learned
form→role (78), uncontingent non-local (79), spiking (80), memory-survives (81), on-substrate (82), recursion-capable
(83–86), one-brain co-resident (87) — but always scored it **in isolation** (role-labeling accuracy vs baselines).
EMERGE-88 closes the loop: the reservoir's role output DRIVES the production `RFPhasorComposer`. This is a genuine
CAPABILITY COMPOSITION (comprehension → composition), not another isolated score — the reservoir understands a
sentence, and the composer stores + answers about it, replacing the hand-labeler / BridgeParser for role assignment.

## The mechanism

`ReservoirComprehender.comprehend(tokens)` parses a sentence into a fact: it takes the reservoir's whole-sentence
final state (Dominey-Hinaut), and for each **content (OPEN) position** predicts its thematic role via the slot
read-out, then reads the surface content word back out. Roles map to the composer's fields
(`AGENT→agent, PREDICATE→action, THEME→patient`). The content is abstracted to a single OPEN marker in the encoder,
so the roles come from **structure**, not memorized lexemes. The parsed fact feeds `RFPhasorComposer.store`; the
who/what turn (`query_patient`) and the no-confab moat then run on the reservoir's OWN comprehension.

## The de-risk — **GO** (6 seeds; reuse EMERGE-62/78 + the production composer; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **parse** — reservoir maps each transitive to (agent, action, patient) | **1.000** (all seeds) | ≥ 0.90 |
| **who/what recall** — reservoir comprehends → composer stores → `query_patient` returns the true patient | **1.000** (all seeds) | ≥ 0.90 |
| **no-confab MOAT** — an (agent, action) never stored → the composer abstains (false-accept rate) | **0.000** (all seeds) | ≤ 0.05 |
| **comprehension-lesion** — collapse the reservoir's closed-class identity → recall collapses | **0.000** (all seeds) | ≤ 0.55 |

**The result:** the reservoir's learned form→role comprehension drives the composer's bind/store/answer end-to-end —
who/what recall **1.000** over the reservoir-parsed facts, with the no-confab moat **intact** (0 false-accepts) and the
reservoir **load-bearing** for the whole turn (lesioning its closed-class identity collapses role-labeling → the stored
facts are wrong → recall → 0.000). Two independently-validated mechanisms now interact on one pipeline: comprehension
(the reservoir) drives composition (the composer).

## Anti-cheats (all pass)

- **Held-out CONTENT** — the test transitives use fresh content draws the slot read-out never saw; the reservoir
  abstracts content to the OPEN marker, so it generalizes its role-labeling to new words (parse 1.000).
- **No-confab MOAT** — the composer abstains on any (agent, action) never stored (0 false-accepts) — the moat holds on
  the reservoir-parsed facts, not just on hand-built ones.
- **Comprehension-lesion** — replacing the reservoir's closed-class identity with a single generic token collapses
  role-labeling → the whole turn collapses (recall 0.000). The reservoir's comprehension is necessary; the composer
  cannot answer correctly without it.

## Honest scope

- **RUNG 1** — the RATE reservoir (EMERGE-78) comprehends → the SPIKING `RFPhasorComposer` (RF resonate-and-fire
  bind/unbind) stores + answers. The spiking-reservoir swap is the **mechanical follow-on**: EMERGE-82's
  `OnBridgeLSM.final_state` has the identical signature, so the comprehender drops in with the reservoir running on the
  `SimulationBridge` substrate. This rung de-risks the load-bearing new thing — the HANDOFF (does the reservoir's role
  output correctly drive the composer's bind+recall+moat?).
- Validated on the core transitive SVO (the construction that fills all three of the composer's (agent, action,
  patient) roles). Richer constructions (ditransitive/PP arguments the reservoir already labels — EMERGE-72/77) mapping
  onto the composer's extra roles is a bounded extension.
- Reuse-by-import (EMERGE-62 discovery + EMERGE-78 reservoir + the production composer); NO `sim/` edit.

## The reservoir arc, now closed into the conversational loop

EMERGE-78 (learned map) → 79 (uncontingent non-local) → 80 (spiking) → 81 (memory survives) → 82 (on-substrate) →
83–86 (recursion boundary + surpass) → 87 (composes on the one brain) → **88 (comprehension DRIVES composition — the
reservoir understands, the composer stores + answers, moat intact)**. The anti-whack-a-mole form→role mechanism is no
longer an isolated score — it is the comprehension front-end of the conversational turn.

## Files
- `research/runners/_emerge88_reservoir_comprehends_composer_answers_derisk.py` — `ReservoirComprehender` (reservoir →
  fact) + the integration de-risk (parse / recall / moat / lesion).
- `tests/test_emerge88_reservoir_comprehends_composer_answers.py` — 3 CPU tests.
- `research/findings/raw/_emerge88_reservoir_comprehends.json` — the 6-seed integration.
