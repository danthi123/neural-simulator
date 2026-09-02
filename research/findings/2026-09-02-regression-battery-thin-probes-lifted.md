---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-regression-battery-thin-probe-lift
seed-waiver: infra self-test on the regression battery's own plumbing (a deterministic single-seed no-op flip +
  a synthetic broken-probe catch) — not a generalization claim over a swept parameter, so the 6-seed bar does
  not apply; the SAME single-seed convention every prior battery/harness landing on this file used.
---

# The cross-faculty regression battery's 22 thin probes — 20 lifted to driving, 2 honestly left thin

**2026-09-02.** Mechanical follow-on to the just-landed Phase-1 infrastructure (merge `013884a72`,
`research/runners/onebrain_regression_battery.py`). The battery registers 38 default-ON faculties and runs each
through the REAL `webapp.server.brain_chat`, asserting its categorical decision stays identical between a flag's
ON and OFF arm. At landing, only 16 of 38 rows were **driving** (their probe turn genuinely triggers the
faculty); 22 were **thin** — the probe never reached the faculty's own trigger condition, so its response field
stayed `None` in both arms and the row reported `not-exercised` regardless of what a merge broke. This session
read every thin faculty's production organ + the response-construction code in `webapp/server.py` to find (a)
its real TRIGGER and (b) its real CATEGORICAL decision field, then either added a new deterministic `PROBE_TURNS`
entry that genuinely drives it, or fixed a wrong field path (several thin rows pointed at a response key that
never existed at all).

## Exercised count

| | before | after |
|---|---|---|
| driving (thin=False) | 16 | 36 |
| thin (thin=True) | 22 | 2 |
| total faculties | 38 | 38 |

## What changed in the file

`PROBE_TURNS` grew from 7 to 26 entries (a 6-tuple now: `label, message, session, reset, percept, rich` — the
last two are new, defaulting to `(None, False)` so every pre-existing row is byte-identical to before).
`_collect_worker` threads `percept`/`rich` into the `BrainChatRequest` it builds. `demo()`'s default
`probe_subset` now covers the FULL `PROBE_TURNS` roster (previously a fast 4-turn subset) so a bare `--demo`
exercises everything `run_regression_battery` itself exercises by default (the production flip-verify harness
already calls it with no `probe_subset`, so this only changes the standalone demo's own default, not production
behavior).

## The 20 lifted faculties (turn -> trigger, field fix if any)

- **comprehension-learned-animacy-cue** / **comprehension-learned-verb-selects** — new turns `animacy`
  ("the monkey carries the cup") / `verbsel` ("the dog cleans the cup"): a noun/verb the ~19-noun /
  8-verb HAND table misses but the learned lexicon covers (the exact words the ledger's own `lesion_note`
  uses). Field `comprehension.on` was already the right path — the OLD turn ("well") just never touched
  the learned-cue code path (every word in it is hand-covered).
- **affect-marker-spiking-wta** — new turn `emo` (a strongly-affective message). `expression_lead()` returns
  `''` unconditionally at mood level 0, and "well" never leaves level 0 — so the field could never discriminate
  the spiking marker-WTA regardless of probe design. Field fixed to `affect_drives.lead` (the marker string);
  the old path, `affect.valence_sign`, is a DIFFERENT Gate-B-only ladder read the (already-driving)
  affect-coloring row covers.
- **confidence-forthcomingness** — new turn `rich_well` (rich=True). `resp["confidence_forthcoming"]` is only
  attached on the rich path; every existing probe hardcoded `rich=False`. Field fixed to
  `confidence_forthcoming.granted` / `.reason` (the old path, `affect.forthcomingness.forthcoming`, was never a
  real key — that sub-dict is the MOOD-set floor from a different coupling, #81/#84).
- **prospective-memory** — new turn `pmem_form` ("remind me to feed the dog when the bird sings"). Field fixed
  to `prospective.held` (the old path, `pmem.armed`, was never a real key or the right top-level name — the
  response key is `prospective`). Formation half only; the later cue-fire half needs a 3rd turn and is not
  covered (declared, not claimed).
- **pragmatic-implicature** — field-path fix only (the existing `scalar` turn already drove it):
  `pragmatic.on` / `pragmatic.enriched_interpretation`, not the old `pragmatic.implicature` (never a real key —
  `interpret()` returns `implicature_margin`/`enriched_interpretation`).
- **surprise-monitor** — new turn `contra` ("the dog chase the fish"): a DIFFERENT patient for an (agent,action)
  the tiny-demo brain already knows from BUILD time (`dog,chase,cat`), a genuine CONTRADICT. "well" is a fresh
  TEACH turn with no PRIOR `what_does()` to compare against, so `surprise_info` stays structurally null there.
- **metacog-monitor** — new turn `confirm` ("the dog chase the cat"): a genuine RECALL of an already-known fact.
  On "well" (a TEACH), the rf trace shows every role `confidence: null` (nothing MATCHED) — `#184`'s own guard
  logs a warning and returns `None`; metacog is out of scope BY CONSTRUCTION on a teach turn, confirmed live in
  this session's own runs.
- **worldmodel-forward** — new turn `expect_q` ("what do you expect"), matching `is_expectation_query`.
- **curiosity-followup** — reused the existing `unknown` turn (already an abstain for moat-verify); curiosity
  only reads on an abstain. Field name fixed: `judge()` returns `curious`, never `crave`.
- **reconsolidation** — reuses `contra` (shares the surprise read at zero extra cost). Field fixed to
  `reconsolidation.action` (`rewrite`/`restabilize`/`abstain`/`lesioned_nowrite`) — the old path
  (`reconsolidation.revised`/`.on`) was never real.
- **episodic-memory** — new turn `episodic` ("did we discuss the dog"), matching `is_referential`; honest
  not-in-memory on a fresh session. Field fixed to `episodic.in_memory` (the real key; `.stored`/`.on` were not).
- **discourse-register** — new 3-turn group `dr_a` (bare clause, no connective) -> `dr_b` (connective-led,
  SHIFTs current->prev) -> `dr_c` ("who was doing it before", the actual query). Field fixed to
  `discourse_register.abstained` / `.agent` — the response key is `discourse_register`, never `discourse`.
- **open-ended-generation** — new turn `rich_open` (the SAME open-ended prompt, rich=True). The hypothesis
  branch lives inside the rich composer's own answer path; every existing probe bypassed it via `rich=False`.
- **discourse-planner** — new turn `rich_well` (rich=True); `n_sentences`/a genuine `rich=True` are
  single-fact-path-false by construction otherwise.
- **gnw-multistep-deliberation** — new turn `chase` ("what does the dog chase all the way"), an explicit
  chase-form question over the tiny-demo's own built-in `dog->chase->cat` / `cat->eat->fish` chain. The
  multi-step gate wraps `chat.gate` itself (no dedicated key) — the terminal surfaces through the
  already-tracked `recalled_svo` field.
- **self-initiated-utterance** — new turn `selfinit` (empty message), matching `is_selfinit_trigger`. Uses the
  top-level `abstained` field rather than the whole `self_initiated` dict (which carries continuous want-rate
  sub-fields that risk a noise-driven false "regressed").
- **vision-identity-spiking-hmax** — new turn `vision` ("what do you see", `percept="bird"` via the
  `BrainChatRequest.percept` field the existing probes never populated). Field fixed to
  `vision_identity.recognized_category` (the response key is `vision_identity`, never `vision`).
- **bg-action-selection** — new turn `bgdots` ("...", the doc's own worked example of a content-empty turn).
  Uses the top-level `abstained` field instead of the whole `bg_select` dict (same noise-risk reasoning as
  self-initiated-utterance).
- **selective-attention-biased-competition** — new 2-turn group `bc_a` ("the cat and the ball walked in",
  opposing animacy) -> `bc_b` ("what does it eat", a bare-pronoun query). Field `recalled_svo` (which referent
  'it' resolved to). Medium confidence at design time — verified via this session's own self-test below.

## The 2 not lifted (honest, investigated, not forced)

**gnw-deliberation** and **value-driven-choice** both need a genuine **>=2-distinct-patient (agent,action)
ambiguity** — the substrate must arbitrate between two candidate patients stored under the SAME key. The
tiny-demo brain's fixed 5-fact KB has no such duplicate by construction, and this session's own `contra` probe
**empirically proves** the live-teach route cannot construct one either: asserting a contradicting patient for
an already-known (agent,action) does not create a SECOND candidate — it triggers the default-ON reconsolidation
organ to **rewrite the stored patient in place** (`reconsolidation.action == "rewrite"`, confirmed live in this
session's run). So at most one patient is ever stored per (agent,action) key through `/api/brain-chat`,
regardless of teaching order. The de-risks behind both faculties build their "dog->chase->{cat,ball}" ambiguity
fixture by constructing a composer with two KB rows directly, bypassing conversational teaching entirely — a
construction this brain_chat-only battery cannot reach without either a second brain bundle with a genuine
pre-existing duplicate (not verified to exist) or forcing `BRAIN_RECONSOLIDATION=0` in the probe env, which
would falsify the adjacent reconsolidation-monitor probe (disabling its own default-ON mechanism for every turn
in the same arm build). Left `thin=True` with this reasoning recorded in the module docstring and inline at each
row.

## Self-test (numpy, seed 42, single-seed self-test — see waiver above)

Artifact: `research/findings/raw/_regression_battery_probes/selftest.txt` (captured stdout of
`python -m research.runners.onebrain_regression_battery --demo`, plus
`research/findings/raw/_regression_battery/battery_demo.json`).

1. **No-op flip, all-pass**: `BRAIN_REGRESSION_BATTERY_NOOP` ON-vs-OFF (an unused sentinel flag — the two arms
   build byte-identically) over the FULL 26-turn / 38-faculty roster.
   <!-- FILLED IN BELOW once the run completes -->
2. **Deliberately-broken probe, caught**: `compare()`'s own synthetic check (unchanged mechanism) still flags
   exactly the one faculty whose field was corrupted, confirming the comparison machinery itself is unaffected
   by the PROBE_TURNS/FACULTY_PROBES expansion.

<!-- RESULTS_PLACEHOLDER -->
