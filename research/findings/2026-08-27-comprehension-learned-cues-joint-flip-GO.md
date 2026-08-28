---
type: finding
status: live
date: 2026-08-27
mechanism: comprehension-learned-cues-joint-flip
---

# Comprehension organ's two corpus-learned cues — JOINT 6-seed flip-soak GO; both flipped default-ON together (Vikunja #190)

**2026-08-27.** Closes Vikunja #190 (and its parent #175). The comprehension organ's two open-vocab cues —
`BRAIN_LEARNED_ANIMACY_CUE` (`research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md`)
and `BRAIN_LEARNED_VERB_SELECTS` (`research/findings/2026-08-27-comprehension-verb-selects-wired-GO.md`) — were
each individually wired and GO-verified, but both shipped **default-OFF**, and neither session tested them
**together**. This session runs the joint flip-soak #190 asked for (6 seeds, both switches on at once, "catch
any interaction"), gets a genuine non-hollow GO on every gate, and **stages the flip**: both flags now default
ON in this branch's `research/runners/comprehension_production_organ.py`.

## Verdict: **FLIP-GO** — both cues flipped default-ON together

All four criteria the task set (byte-identical-off, zero hand-covered regression, open-vocab coverage strictly
improves, no interaction) are met, backed by a **6-seed, fresh-process-per-condition organ-level soak** plus a
**single-seed full-production-turn re-verify through the real `webapp.server.brain_chat` handler** (the handler
itself is not seed-parameterized — `get_organ(seed=42)` is hardcoded at every call site in `webapp/server.py`,
so seed=42 is the only seed the production path ever actually builds with).

## Method

`research/runners/_comprehension_learned_cues_joint_arm.py` (one flag/seed condition, one fresh process) +
`research/runners/_comprehension_learned_cues_joint_flip_soak.py` (the controller: spawns one arm subprocess
per condition, aggregates, computes the gate verdicts). **Fresh process per condition, not a same-process
sequential flag toggle**: while building the CI regression test (`tests/test_comprehension_learned_cues_joint.py`)
a same-process sequential-call version of the byte-identical check FAILED on raw margin floats for the
identical `"the dog eats the apple"` read with no flag difference <!--derived: observed interactively during
this session's test authoring, not saved to a committed artifact--> — this is the project's own
already-documented chaotic inter-turn spiking jitter (`research/FAILURE_LOG.md`, 2026-08-25 gap5-store entry),
re-confirmed here, not a new defect. It does NOT affect the decision-level outputs (`competent`/`comprehended`
booleans, `svo`, `repair.kind`), which the organ's own per-read `_hard_reset` keeps stable — only the raw
spiking-margin float drifts run-to-run within one process. The 6-seed soak (fresh process per condition) sidesteps
this entirely (the byte-identical proof below is a REAL zero-diff, not float-tolerant); the CI test compares only
the decision-level fields for the same reason.

Battery (`_comprehension_learned_cues_joint_arm.py`, 25 items):

* **10 HAND_COVERED** — one sentence per hand `VERB_SELECTS` verb (agent="dog", a hand-`ANIMACY`-matching
  patient) plus the two established covered-but-ambiguous sentences (`"the wolf watches the owl"` 2-animate
  symmetric-verb, `"the book carries the cup"` 2-inanimate-leaning) — the no-regression + interaction battery.
* **6 HELD_NOUN** — a hand-covered verb+patient with a held-out agent noun (3 animate: monkey/rabbit/kitten; 3
  inanimate: box/table/key) — isolates the animacy cue.
* **6 HELD_VERB** — hand-covered agent+patient with a held-out verb (3 inanimate-patient: clean/wash/open; 3
  animate-patient: help/hug/feed) — isolates the verb-selects cue.
* **2 JOINT** — agent noun AND verb BOTH held-out simultaneously (`"the monkey clean the box"`,
  `"the rabbit help the kitten"`) — the two-cue interaction stress test.
* **1 MOAT** — `"the wug blickets the glorp"`, fully OOV under any flag state.

Every held-out word was pre-verified (a standalone probe, before the battery was built) to be genuinely absent
from the hand `ANIMACY`/`VERB_SELECTS` tables AND to classify definitively (non-abstain) via the deployment
(seed=42) learned lexicons — so a battery item abstaining would be a genuine finding, not word-choice noise. All
12 held-out words classified as expected (monkey/rabbit/kitten/box/table/key -> animate/animate/animate/
inanimate/inanimate/inanimate; clean/wash/open/help/hug/feed -> inanimate_patient x3/animate_patient x3).

8 conditions x 6 seeds = 48 fresh-process arm runs: `C0` both flags unset (the literal current-main default),
`C1` both explicit `"0"`, `C2` animacy-only ON, `C3` verb-only ON, `C4` both ON, `C5` both ON + animacy
lesioned, `C6` both ON + verb-selects lesioned, `C7` both ON + both lesioned.

**Measured cost** (numpy-CPU, from each arm's own `.prov.json` `started` timestamp / the production artifact's
mtime): the 48-arm organ-level soak spans 178s wall-clock (~3.7s/arm mean); the single-seed full-production-turn
re-verify (9 real `/api/brain-chat` turns through the 40192-neuron tiny-demo recall composer + its co-resident
organs) took ~653s (~10.9 min) — an order of magnitude more expensive per-turn than the organ-level battery,
because the production handler builds the FULL brain, not the isolated toy comprehension bridge.

## Gate results — `research/findings/raw/_comprehension_learned_cues_joint_flip_soak_6seed.json`, `GO: true`

| gate | check | result |
|---|---|---|
| **G1** byte-identical-off | `C0` (unset) == `C1` (explicit `"0"`) on the FULL 25-item battery, all 6 seeds (150 item-seed comparisons) | **0 diffs** |
| **G2** no-regression + interaction | every one of `C2..C7` == `C1` on all 10 HAND_COVERED items, all 6 seeds (360 comparisons) | **0 diffs** — the hand table is an unconditional fast path, so this IS the literal "the two cues don't interfere on any hand-covered case" check |
| **G3** coverage extends | every HELD_NOUN + HELD_VERB item's `competent()` flips `C1`=False -> `C4`=True | **72/72** flips (12 items x 6 seeds), **0 regressions** (monotonic superset — nothing flips the other way) |
| **G4** lesion load-bearing | lesioning ONLY animacy reverts HELD_NOUN items while sparing HELD_VERB (and the mirror for verb-selects); lesioning BOTH reverts the WHOLE battery to an exact match of `C1` | **0 fails** |
| **G5** joint coverage | on the 2 JOINT items, `C1`'s `repair_target` names all 3 open-vocab words OOV; `C4`'s names none | **0 fails** |
| moat | the OOV item abstains (`comprehended: False`) in EVERY one of the 8 conditions, all 6 seeds | **0 fails** |

`GO: true` — all six checks pass. The artifact carries the verdict through `tools.verdict.Verdict`
(`preconditions`: 8 entries — one `require` per gate above, plus 2 `control` checks below — all `ok: true`,
none unmeasured/failed, so the gate-enforced `verdict-preconditions` class treats this GO as earned, not
asserted beside an unchecked instrument).

**Attribution** (`tools.lab.attributable_to`, the mean per-seed `competent()`-rate over each cue's own isolated
held-out battery, live vs that SAME cue alone lesioned — the sibling cue stays live in both arms, isolating this
cue's own contribution): animacy-cue held-noun coverage rate 1.0 (live) vs 0.0 (animacy-lesioned) —
**100.0% attributable** to the live coupling, 0% present in the lesioned control; verb-selects-cue held-verb
coverage rate 1.0 (live) vs 0.0 (verb-lesioned) — **100.0% attributable**, identically. Neither cue's
coverage-extension effect survives its own lesion even partially — a clean control, not a partial one.

## The G5 nuance (an honest correction made during this session, not shipped)

`competent()`'s pre-existing fully-OOV branch (verb unknown AND both nouns unknown to the hand tables — this
predates both learned cues by months) means the D4 gate fires on a doubly-held-out sentence **regardless of
these two flags**: with both flags off, `"the rabbit help the kitten"` is `competent()=True` via the
already-existing "genuinely all-unknown" path, not because either learned cue is engaged. The FIRST draft of
the production-turn check (`_comprehension_learned_cues_joint_production_verify.py`) wrongly asserted
`comprehension is None` (out-of-scope) for this flags-off case and failed; the organ-level `G5` gate (which
checks `repair_target`'s OOV-token naming, not `competent()`/abstain-or-not) had the right measurement the
whole time. Fixed before this finding landed — the joint-specific coverage signal for a doubly-held-out sentence
is **which repair the substrate can give** (`repair.kind` `"oov"` -> `"role"`), not whether it abstains at all.

## Full-production-turn re-verify — `research/findings/raw/_comprehension_learned_cues_joint_production_verify.json`, `ALL_OK: true`

Through the real `webapp.server.brain_chat` handler (numpy-CPU, the tiny-demo brain: 40192 neurons for the
recall composer + 624-neuron co-resident organs), mirroring `_gateB_repair_production_verify.py`'s own
methodology (distinct session ids + `reset=True` per turn):

| check | result |
|---|---|
| `hand_covered_byte_identical` (`"the book carries the cup"`, flags off vs both on) | PASS |
| `joint_off_oov_named` (`"the rabbit help the kitten"`, flags off: `repair.kind=="oov"`, names all 3 words) | PASS |
| `joint_on_abstained_with_repair` (same sentence, both on: `repair.kind=="role"`, a targeted role clarification, none of the 3 words named OOV any more) | PASS |
| `held_out_clear_off_out_of_scope` (`"the monkey eats the apple"`, flags off: `comprehension is None`) | PASS |
| `held_out_clear_on_comprehended` (same sentence, both on: `comprehended: True`) | PASS |
| `moat_unaffected_by_flags` | PASS |
| `lesion_both_reverts_joint_to_flagoff` (both flags on + both lesioned: exact match to flags-off) | PASS |

A concrete before/after on the joint-ambiguous sentence (the genuinely VISIBLE, qualitatively different
response the flip produces):

* **flags OFF:** *"I followed the shape of that, but I don't know the words 'rabbit' or 'kitten' yet — what do
  they refer to?"*
* **flags BOTH ON:** *"I caught the verb 'help' with the rabbit and the kitten, but my role-binding didn't
  resolve who does what — is the rabbit doing the 'help' to the kitten, or the other way round?"*
* **flags BOTH ON + BOTH lesioned:** reverts to the exact flags-OFF text above.

## The flip

`research/runners/comprehension_production_organ.py`: both `learned_animacy_cue_enabled()` and
`learned_verb_selects_enabled()` gained a `_LEARNED_ANIMACY_CUE_DEFAULT_ON` / `_LEARNED_VERB_SELECTS_DEFAULT_ON`
anchor constant (mirrors `webapp/confidence_forthcoming_chat.py`'s `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` escape
pattern exactly), flipped `True`. `BRAIN_LEARNED_ANIMACY_CUE=0` / `BRAIN_LEARNED_VERB_SELECTS=0` are the
byte-identical escapes to the pre-flip hand-table-only scope (verified equal to "unset" of the PRE-flip code by
G1 above). The lesion flags (`BRAIN_LEARNED_ANIMACY_LESION`, `BRAIN_LEARNED_VERB_SELECTS_LESION`) are untouched
— they stay default-OFF debug/verification affordances, orthogonal to the cue-enable default.

**Migration note for the two pre-existing per-cue CI test files**: `tests/test_comprehension_learned_animacy_cue.py`
and `tests/test_comprehension_learned_verbselects_cue.py` each had a `test_flag_off_byte_identical_to_pre_existing_scope`
test whose "off" setup was `_clear_flags()` (unset) — correct pre-flip, now wrong (unset means ON post-flip).
Fixed to set the flag explicitly to `"0"`; a new `test_unset_now_defaults_to_on` in each file pins the flip
itself. `research/runners/_comprehension_learned_animacy_wire_verify.py` /
`_comprehension_learned_verbselects_wire_verify.py` (the individual wire-in findings' own verify scripts) had
the identical latent issue in their `"flag_off"`/`"flag_off_again"` condition rows — fixed the same way (explicit
`"0"`/`"1"` always, never bare `_clear_flags()` for an "off" condition).

## Status against `docs/TERMS.md`

Both rows now read **wired + on-by-default**, still **NOT scaffold-retired** (the hand tables remain the
unconditional fast path and the moat floor for genuinely off-graph words — an EXTENSION, not a replacement,
exactly as both individual wire-in findings declared). Per the "on-by-default ... Level-3 spiking on-by-default
credit additionally needs a LESION test" rule: earned — `G4` above is exactly that lesion test, at the organ
level and (for the joint case) at the full production-handler level.

## Verification run (no regression on existing gates)

* `tests/test_comprehension_learned_animacy_cue.py` (9/9, incl. the new flip-pin test) — PASS.
* `tests/test_comprehension_learned_verbselects_cue.py` (10/10, incl. the new flip-pin test) — PASS.
* `tests/test_comprehension_learned_cues_joint.py` (10/10, new this session) — PASS.
* `tests/test_gap3_spiking_feature_compat.py` (7/7, the shared F_anim/F_inanim mechanism) — PASS.
* `tests/test_multireferent_biased_competition.py` (5/5) — PASS.
* `research/runners/_gateB_repair_production_verify.py` (the pre-existing T1-6 production regression gate, 6/6
  checks, now running with both cues default-ON) — `ALL_OK=True`, unaffected.
* `.venv/bin/python tools/check_docs.py` — W1=0, W2=0.
* `tools.gates.production_integration.check()` — 0 problems (both new ledger rows' `default_anchor`s resolve
  against live source).

## docs/PRODUCTION_INTEGRATION_LEDGER.yaml

Two new rows: `comprehension-learned-animacy-cue` and `comprehension-learned-verb-selects`, both
`de_risked: YES / wired: YES / on_by_default: YES / scaffold_retired: NO`, each with a `default_anchor` pointing
at its `_LEARNED_*_DEFAULT_ON` constant. `total_faculties` 55->57. `default_on_spiking_faculties` **unchanged
at 26** — neither row adds a new spiking substrate (both reuse the already-counted gap#3-A1 F_anim/F_inanim
pools and the already-counted D4 `SpikingRoleCompetition` bridge), same no-new-substrate rule already applied to
B3/B4/T1-6/gnw-multistep/sleep-replay-consolidation. The pre-existing `comprehension-monitor` row's VOCAB
CEILING residual note updated from "unresolved" to "PARTIAL — extended by two corpus-learned cues, both
flipped default-ON 2026-08-27" (cross-referencing the two new rows), since the hand table's byte-identical-off
floor for genuinely off-graph words is unchanged.

## Residuals (inherited, unchanged from the two individual wire-in findings — not newly introduced)

* Both hand tables (`ANIMACY` ~19 nouns, `VERB_SELECTS` 8 verbs) remain in place as the fast path — an
  EXTENSION, not a replacement (`scaffold_retired: NO` on both new rows).
* The noun/verb canonical-proxy remap ("dog"/"ball" for animacy, "watch"/"eat" for verb-selects) means
  `SpikingRoleCompetition` never reads the ACTUAL open-vocab word's own spiking representation during the
  role-competition read — only its learned category, via a stand-in population.
* The animate-patient verb-proxy class ("watch") structurally contributes no verbfit vote (inherited from the
  hand table's own pre-existing `chase`/`watch` behavior — not a new gap; a held-out animate-patient verb's
  `judge()` margin is governed by animacy-cue-only dynamics on a two-animate-noun sentence, same as the hand
  table's own symmetric verbs).
* The corpus (`data/corpus/tinystories.txt`) is a gitignored, regenerable cache; not re-pinned here (no
  label-propagation changes this session — only the already-committed deployment lexicons were reused).
* CO-RESIDENT: the comprehension monitor still runs on its own `SpikingRoleCompetition` bridge, not merged onto
  the one recall bridge (burn-down #1) — unaffected by this flip.

## Files

* `research/runners/_comprehension_learned_cues_joint_arm.py` — new: single-condition battery evaluator
  (subprocess-invoked by the controller for fresh-process isolation).
* `research/runners/_comprehension_learned_cues_joint_flip_soak.py` — new: the 6-seed x 8-condition controller +
  the 6-gate verdict computation.
* `research/runners/_comprehension_learned_cues_joint_production_verify.py` — new: the full-production-turn
  re-verify through `webapp.server.brain_chat`.
* `research/findings/raw/_comprehension_learned_cues_joint_flip_soak_6seed.json` (+ 48 per-arm artifacts under
  `research/findings/raw/_comprehension_learned_cues_joint/`) — the organ-level soak, `GO: true`.
* `research/findings/raw/_comprehension_learned_cues_joint_production_verify.json` — the production-turn
  re-verify, `ALL_OK: true`.
* `research/runners/comprehension_production_organ.py` — `_LEARNED_ANIMACY_CUE_DEFAULT_ON` /
  `_LEARNED_VERB_SELECTS_DEFAULT_ON` anchors added; `learned_animacy_cue_enabled` /
  `learned_verb_selects_enabled` flipped to the default-on-unless-explicit-off reading.
* `tests/test_comprehension_learned_cues_joint.py` — new CI guard, 10/10 passing.
* `tests/test_comprehension_learned_animacy_cue.py` / `tests/test_comprehension_learned_verbselects_cue.py` —
  migrated `test_flag_off_...` to explicit `"0"`, added `test_unset_now_defaults_to_on`.
* `research/runners/_comprehension_learned_animacy_wire_verify.py` /
  `_comprehension_learned_verbselects_wire_verify.py` — the `"flag_off"` condition rows now set the flag
  explicitly (`"0"`/`"1"`) instead of relying on unset.
* `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` — two new rows, `total_faculties` 55->57, the `comprehension-monitor`
  row's VOCAB CEILING residual note updated.
