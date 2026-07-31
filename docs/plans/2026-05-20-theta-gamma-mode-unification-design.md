---
type: plan
status: live
date: 2026-05-20
---

# Theta-gamma mode-unification architecture design (next staged arc after the unified per-regime monitor's empirical FAIL)

> **For Claude / autonomous continuation:** This document is the **design** for the theta-gamma mode-unification stage. After approval, the writing-plans skill produces the TDD implementation plan, then subagent-driven-development builds Task 0..Task 5 with adversarial review and controller-only decisive run. Mirrors the prior 4 arcs' (Stage-1, SPEAR, Pirazzini, Unified) discipline exactly.

## Status

Pre-registered NEW design grounded in two facts:

1. **Standing direction (user-directed; section 2b of design doc
   `docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`,
   commit `337ff8c`)**: the next major arc after the gating-based four-architecture
   convergent ceiling is theta-gamma mode-unification + generative replay +
   PFC-held compositional frame. The catalog-grounded biological mechanism
   that the prior gating-based arcs (Stage-1, SPEAR, Pirazzini, Unified)
   did NOT implement.

2. **Empirical localisation finding (commit `110f7cd`)**: the unified per-regime
   monitor's GATE=FAIL is mechanically caused by the substrate's compositional
   readout emitting STRONG-BUT-WRONG top words at high confidence. Specifically
   (seed 42, N=5): 4 of 5 groundable compositional queries return a top word
   that is NOT the bound adjective, but the top rate is well above BOTH
   calibrated thresholds (0.198 and 0.284). Hypothesis: the cued-noun's
   diffuse `lang_input` drive dominates the engram tag's selective
   bound-adjective drive at deployment time. The fix must SUPPRESS the cue
   during the retrieve window.

## 1. The capability under test (falsifiable)

Compositional retrieval that **emits the bound adjective when the noun is cued**,
not the strongest pool the cue happens to resonate with. The frozen bars are
exactly the existing `_PR_*` constants (`per_regime_monitor_core.py`,
byte-unchanged):

- `_PR_FULL_MIN = 0.80` — compositional retrieval accuracy at smallest-N rung
- `_PR_UNIFORM_CTRL_MAX = 0.10` — collapse of the "no-cue-suppression" control
- `_PR_DIRECT_RETAIN_MIN = 0.80` — direct retrieval still works
- `_PR_ABSTAIN_CORRECT_MIN = 0.90` — trustworthy abstention holds
- `_PR_SCALE_TOL = 0.10` — full_acc holds across ladder (2,3,5)
- `_PR_MIN_SEEDS = 3`

The architectural variation is the FULL arm: now FULL = theta-gamma-mode-
unification (cue-suppression during retrieve); UNIFORM_CTRL = same wiring
WITHOUT the cue suppression. If theta-gamma mode-unification produces a
genuine compositional retrieval advantage, full_acc > uniform_ctrl_acc by
>=0.70 at the smallest-N rung. If not, FAIL (this design line is structurally
exhausted at biological scale and the next staged step is the
5-architecture convergent ceiling as a terminal biology-translatable
finding).

## 2. The mechanism being added (load-bearing; grounded in the localisation)

**One shared theta-gamma rhythm (~125 ms / ~8 Hz; the project's catalog
N.16 Lisman-Idiart) time-multiplexes the encode and retrieve windows.**
Each query proceeds in three phases per theta cycle:

### Phase 1: ENCODE / CUE (theta-trough; first ~62 ms)

- Cue: `lang_input(noun)` drives at normal strength (cortico-CA3 pathway active)
- ACh: HIGH (existing `acetylcholine_tan` modulator at high baseline)
- Plasticity at `lang_to_ec` + `ec_to_dg` + `dg_to_ca3` + cortico-pool: ON
- Tag stim: ENGRAM tag (encoded at training time via `_encode_facts`)
  is briefly stimulated to bind the cue→tag association
- CA3 recurrence: SUPPRESSED (so tag stim doesn't yet drive pattern completion)

### Phase 2: GAP (transition; ~10 ms)

- Cue: SUPPRESSED (the load-bearing addition; see section 3)
- ACh: ramps DOWN
- All plasticity gates: CLOSE

### Phase 3: RETRIEVE / PATTERN-COMPLETE (theta-peak; last ~52 ms)

- Cue: SUPPRESSED (`lang_input` pathway gated OFF — the addressing of the
  cued-noun-dominance failure mode)
- ACh: LOW (suppresses cortico-CA3 plasticity + transmission per Hasselmo
  SPEAR; existing `acetylcholine_tan` modulator at low baseline)
- CA3 recurrence: ACTIVE (the tag's bound-adj neurons drive each other
  via Schaffer-collateral + intra-CA3 recurrence)
- CA1 → `lang_output` pathway: ACTIVE (the retrieved pattern is read out)
- Plasticity: OFF (consolidation discipline; reads do not modify weights)
- READOUT: `_compositional_query_ranked` measures `lang_output` cosine
  during this window ONLY

The KEY architectural difference from SPEAR + Pirazzini: those gated only
**plasticity** (the C2 reward-modulated weight-update block in `bridge.py`).
They did NOT gate **transmission** (the synaptic_gain modulation through the
running forward simulation). The theta-gamma mode-unification arc gates BOTH:
plasticity OFF during retrieve + cue transmission OFF during retrieve.

## 3. The CUE-SUPPRESSION mechanism (the genuine net-new piece)

Per the localisation finding: at deployment, the cued-noun's `lang_input`
drive dominates the bound-adj retrieval. The fix is to **gate the
`lang_input` pathway OFF during the retrieve window**.

Implementation route (reuse-by-import; no protected/frozen module touched):

The neuromodulator subsystem supports `synaptic_gain` with `scope="all"` but
NOT `scope="gate:<pathway_name>"` (per the Pirazzini fix note in
`sim/neuromodulators.py:298-305`). For per-pathway transmission gating we
have two options:

(A) **Direct gate write on `cp_external_input_current`** — temporarily zero
    the cue's lang_input pattern during the retrieve window. This is a
    runtime write, not a subsystem extension; the runner already does this
    for `_apply_theta_disinhibition` in the Pirazzini arc. Lowest-risk,
    smallest implementation, no subsystem change required.

(B) **Synaptic-strength schedule on `lang_to_ec`** — explicitly multiply the
    pathway's transmission weight by an envelope (1.0 during encode; 0.0
    during retrieve). Requires either (a) extending the neuromodulator
    subsystem to support `synaptic_gain scope=gate:lang_to_ec` (touches
    protected `sim/neuromodulators.py`, NOT allowed), OR (b) using the
    `cp_plasticity_rate_gain` array as a transmission gain proxy (which
    only gates plasticity, not transmission per the documented gotcha in
    CLAUDE.md "GOTCHA — plasticity gate vs synaptic transmission").

Option A is the disciplined route: no subsystem change, no protected file
touched, mirrors the Pirazzini pattern exactly. The runner schedules
`cp_external_input_current` writes per theta phase.

**Verification probe (mandatory)**: the runner emits an adversarial probe
that runs a 50-step constant-input cycle with theta-gamma ON vs OFF and
asserts non-byte-identical bridge state at the end. The Pirazzini arc's
adversarial review caught a doubly-inert disinhibition mechanism (step_idx=0
hardcoded; encode_concept_pair clearing cp_external_input_current); the
theta-gamma arc MUST not repeat the same defect.

## 4. Inventory of reused subsystems (byte-unchanged; no protected modification)

- `build_biological_brain_regions(..., enable_hippocampus_consolidation=True,
  enable_noun_pools=True, enable_verb_pools=True, enable_adjective_pools=True)`
  — the SAME unified substrate; same Phase-1 cached checkpoints
- `encode_concept_pair` (compose_concept_engram.py) — encoding helper
- `_compositional_query_ranked` (unified_per_regime_monitor_runner.py:804) —
  readout helper
- `per_regime_monitor_core.per_regime_monitor_verdict` — frozen capability
  verdict (REUSED byte-unchanged; the bars stay the same; only the runner's
  full vs uniform_ctrl semantics change)
- The four substrate-and-protocol-specific abstention moats (650 / 5.6887 /
  0.197712 / 0.284167) — all byte-stable
- The neuromodulator subsystem (sim/neuromodulators.py) — byte-unchanged
- `sim/bridge.py` cp_external_input_current write API — byte-unchanged

The genuine net-new code: a SHARED theta-gamma timing controller in the
runner (~200-300 lines mirroring `pirazzini_three_layer_runner.py`'s
`_apply_theta_disinhibition` scaffold + cue-suppression wiring) + a
`Task 1` frozen capability-verdict module (mirrors the prior 4 arcs'
frozen-verdict pattern; new constants like `_TG_*` distinct from `_PR_*`
even though the bar values match).

## 5. Pre-registered next staged step (autonomous; no hand-back)

Per the writing-plans skill, after this design ships:

- Task 0: grounding pin test (red until Task 2; verifies the new
  runner's module path + capability-verdict module path + frozen bars
  are reachable).
- Task 1: net-new frozen capability-verdict module
  `research/runners/theta_gamma_mode_unification_core.py` (~150 lines;
  17+ adversarial test cases mirroring the prior 4 arcs' frozen-verdict
  matrices exactly; bars `_TG_FULL_MIN=0.80, _TG_UNIFORM_CTRL_MAX=0.10,
  _TG_DIRECT_RETAIN_MIN=0.80, _TG_ABSTAIN_CORRECT_MIN=0.90,
  _TG_SCALE_TOL=0.10, _TG_LADDER=(2,3,5), _TG_MIN_SEEDS=3` — identical to
  per-regime but with their own module-local constants; instrument-validity
  fail-closed; VOID strictly distinct from FAIL; malformed input -> VOID
  never crash; stdlib + typing only).
- Task 2: net-new runner `research/runners/theta_gamma_mode_unification_runner.py`
  (~900 lines mirroring the prior arcs' structure; shared-theta-rhythm
  controller + cue-suppression wiring + per-rung evaluation that produces
  the four `_PR_*`-shape capability metrics; reuse-by-import only; no
  autograd; ASCII; kill-safe via the existing `sim.train_checkpoint`
  module byte-unchanged).
- Task 3: dedicated adversarial review of the load-bearing Task 2 wiring
  + the frozen verdict module's adversarial probe matrix. Eighth
  consecutive adversarial review across the arc series; prior 7 each caught
  real defects, so the discipline has high adversarial pressure. Specific
  exploit-class probes the reviewer must run:
  - structural-effect probe (theta-gamma ON vs OFF must produce
    non-byte-identical bridge state; Pirazzini's defect was exactly this
    being doubly-inert and the probe synthetic-only)
  - false-PASS vector (does a cue-suppression-only-pretending mechanism
    score PASS via the runner+frozen verdict?)
  - byte-unchanged audit on every protected file + the 4 calibrated moats
  - no autograd / no torch / no plasticity_gate semantic change
- Task 4: no-harm verification (full test suite green; protected set
  byte-empty diff vs `e8a99a2`; no-confab moat 7/7 byte-identical).
- Task 5: controller-only decisive run at full biological scale (3 seeds;
  ladder (2,3,5); both unified-substrate moats in place; kill-safe;
  monitored to actual process exit via genuine completion waiter) +
  mandatory smell-test (recompute verdict from JSON; scrutinise PASS
  harder than FAIL) + honest propagation EVERY outcome both remotes +
  autonomous next staged step per outcome.

## 6. Honest ceiling (binding throughout)

- Conversational / compositional capability is NOT achieved and is NOT
  claimed until a pre-registered stage genuinely shows it.
- A PASS at this stage would be the FIRST architecture in the 5-arc series
  to clear the frozen bars; it would NOT yet be fluent open-ended language /
  an LLM; it would be a biology-grounded compositional retrieval capability
  that holds at small loads (N=2,3,5).
- A FAIL would extend the convergent ceiling to FIVE architectures and
  motivate either: (a) deeper-mechanism design beyond the project's
  currently-validated subsystems (e.g., generative replay + PFC compositional
  frame, which are reusable subsystems not yet phase-multiplexed into the
  unified substrate); or (b) honest closure of this design line as a
  terminal biology-translatable finding.
- The biology-translatable insights (substrate-and-protocol-specific
  trustworthy thresholds; 4-architecture convergent ceiling on gating-based
  compositional retrieval; the cue-suppression-during-retrieve mechanism as
  the localised failure mode of gating approaches) are durable and
  unaffected by this stage's outcome.

## 7. Discipline pins (mirrors prior 4 arcs exactly)

- NO bar change anywhere; the new `_TG_*` constants are set in advance and
  NEVER tuned in response to results.
- NO protected file modification; the protected set byte-empty diff vs
  `e8a99a2` must continue to hold across every commit of this arc.
- NO autograd / no torch / no LLM call anywhere.
- NO declare-unfit; NO hand-back; NO config-crank; iterate following biology.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE decisive run.
- Honest propagation EVERY outcome both remotes (`origin` + `gitea`).
- The autonomous next-action tool call is always in the same turn after
  every commit; never stop on a future-tense promise.
- The accumulated 4 substrate-and-protocol-specific calibrated moats stay
  byte-stable.
- The no-confabulation moat (`abstention_gate.py` + `tests/test_abstention_gate.py`)
  stays byte-identical and 7/7 green.

## References (catalog-grounded; from the standing design doc)

- [9] SPEAR Hasselmo 2024: separate phases of encoding and retrieval as the
  load-bearing mechanism for storing multiple overlapping associative
  memories.
- [12] Heusser et al. 2016: episodic sequence memory supported by a
  theta-gamma phase code.
- [14] Ursino et al. 2022: theta-gamma working memory model.
- N.16 (project catalog): Lisman-Idiart shared theta-gamma rhythm flagged as
  never built in any project arc to date.

## Next-step pointer (writing-plans)

After approval / acceptance of this design, the next step is the
writing-plans skill to produce the TDD implementation plan
(`docs/plans/2026-05-20-theta-gamma-mode-unification-implementation.md`).
The plan transcribes Tasks 0..5 as bite-sized failing-test-then-minimal-
implementation-then-commit steps, mirroring the prior 4 arcs' plan
structure exactly. Then subagent-driven-development executes the plan.
