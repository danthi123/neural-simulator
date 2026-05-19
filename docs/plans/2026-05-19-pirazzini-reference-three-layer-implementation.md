# Pirazzini-reference three-layer theta-gamma conversational stage — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans, task by
> task. Owner standing instruction pre-selects same-session subagent-driven
> execution (one fresh subagent per task; failing-test → minimal-impl → run
> → commit; controller trust-but-verify every diff). Task 5 is
> CONTROLLER-ONLY. Mirrors the proven Stage-1 and SPEAR arc structures
> (which both caught real defects via dedicated adversarial reviews before
> any decisive run). Design:
> `docs/plans/2026-05-19-pirazzini-reference-three-layer-theta-gamma-conversational-design.md`.

**Goal:** Build the Pirazzini-2024 three-layer biology-grounded
sequential-memory architecture, adapted to the project's validated
dlpfc_verb / ca3 / ca1 substrate, with the correct Hasselmo ACh polarity
(encode = HIGH ACh: suppresses CA3→CA1 + strengthens cortical input +
facilitates LTP; retrieve = LOW ACh: promotes pattern completion) and the
biology-faithful disinhibition-based theta mechanism (theta generator
rhythmically disinhibits CA3 via glutamatergic excitatory synapses onto
inhibitory interneurons, NOT synaptic-gain-modulation). Test — against
a new pre-registered fixed-bar three-state verdict whose decisive
built-in control is the convergent Stage-1 + SPEAR ceiling
(theta_disabled must collapse to ≈ that ceiling) — whether the
Pirazzini-reference architecture lifts compositional readout above the
calibrated no-confabulation threshold the prior arc localised.

**Architecture:** Reuse byte-unchanged: the validated concept substrate
+ hippocampal theta-gamma store + trisynaptic pattern-completion +
dlpfc PFC working-memory frame + NMDA bistability + neuromodulator
subsystem (all primitives reused) + replay-consolidation phase functions
+ awake/sleep gates + engram-tagging API + no-confabulation moat at
output. Net-new is a small external theta-generator controller, a
multi-target ACh NeuromodulatorConfig combining REUSED targets, a
one-shot encoding routine (250 ms per episode-pair, simultaneous to
L1 + L2), and a within-theta-cycle decode through the moat (fed the
calibrated raw firing-rate quantity).

**Tech Stack:** Python; CuPy on RTX 3090 for decisive runs (NumPy only
for `--tiny-synth`); the verdict module imports standard library +
typing only; reuse-by-import for all subsystems; ASCII-only output;
kill-safe via reused checkpoint module.

**Protected set (MUST be byte-unchanged across `git diff` for every task
commit; controller verifies):** `research/runners/abstention_gate.py` +
`tests/test_abstention_gate.py` (no-confabulation moat, MUST stay 7/7);
every frozen `*_core.py` including the prior Stage-1 + SPEAR + integrated
verdict modules; `research/runners/text_minimal_isolation.py`;
`research/runners/consolidation_trainer.py`;
`research/runners/validate_trisynaptic_loop.py`;
`research/runners/compose_concept_chat.py`;
`research/runners/compose_concept_engram.py`;
`research/runners/compose_retrieval_runner.py` + `spear_conversational_runner.py`
(the prior stages' runners); `sim/bridge.py`; `sim/regions.py`;
`sim/neuromodulators.py`; `sim/train_checkpoint.py`; `sim/backend.py`;
`sim/kernels.py`.

---

## Task 0: Grounding pin (red until Task 2)

**Files:** Create `tests/test_pirazzini_three_layer_pin.py`.

```python
"""Grounding pin; intentionally RED until Task 2 lands the runner."""
import importlib

def test_pirazzini_runner_importable():
    m = importlib.import_module("research.runners.pirazzini_three_layer_runner")
    assert hasattr(m, "run_pirazzini_three_layer")

def test_pirazzini_core_importable():
    m = importlib.import_module("research.runners.pirazzini_three_layer_core")
    assert hasattr(m, "pirazzini_three_layer_verdict")
```

Run → FAIL (intentional; the Task-1/Task-2 gate). Commit (`test: grounding
pin for Pirazzini-reference three-layer stage (red until Task 2)`).
Controller verifies protected set byte-empty.

---

## Task 1: The frozen capability-verdict module (LOAD-BEARING; transcribe exactly)

Mirrors the Stage-1 + SPEAR frozen-verdict discipline EXACTLY (both
adversarially CLEARed): fixed numeric thresholds set now and NEVER
tuned; instrument-validity FIRST; malformed → VOID, never crash; VOID
strictly distinct from FAIL; standard library + typing only; does NOT
import or modify any existing verdict module or the moat.

**Files:** Create `research/runners/pirazzini_three_layer_core.py`;
Test `tests/test_pirazzini_three_layer_core.py`.

**Frozen constants (verbatim; NEVER tuned):**
`_PZ_FULL_MIN = 0.80` (Pirazzini reports >99% on episodes 2-3; 0.80 is
the meaningful retrieval bar for our adapted substrate),
`_PZ_CONVERGENT_CEILING_MAX = 0.10` (Stage-1 and SPEAR scored 0.00; the
theta_disabled control must collapse to ≤ 0.10, allowing small noise
margin — the DECISIVE built-in control: capability must significantly
exceed the convergent ceiling, so a rhythm-artifact false-PASS is
structurally detectable),
`_PZ_ABSTAIN_MIN = 0.90`,
`_PZ_SCALE_TOL = 0.10`,
`_PZ_LADDER = (2, 3, 5)` (episodes per sequence; Pirazzini reports up to
5 per theta cycle, 2-3 high-confidence, 5 at limit),
`_PZ_MIN_SEEDS = 3`.

**Rung required keys:** `N` (episodes per sequence), `n_seeds`,
`full_acc`, `theta_disabled_acc` (decisive built-in control —
theta generator turned off; must reduce to the convergent
Stage-1/SPEAR ceiling), `abstain_correct_theta_disabled`.

**`pirazzini_three_layer_verdict(rungs) -> Dict[str, Any]` contract**
(same shape and discipline as Stage-1 / SPEAR verdict modules):
1. Instrument-validity FIRST → VOID (never raise): `rungs` non-empty
   list; each a dict with all required keys; `N` int (not bool) in
   `_PZ_LADDER`; `n_seeds` int (not bool) ≥ `_PZ_MIN_SEEDS`; three
   accuracy fields finite (`isinstance (int,float)` not bool,
   `math.isfinite`) in [0,1]; rung N set has no duplicates and equals
   first `len(rungs)` entries of `_PZ_LADDER` (a prefix). Extra keys
   (caller `verdict`) ignored — recompute from raw only.
2. `ok(r) := r.full_acc >= _PZ_FULL_MIN and r.theta_disabled_acc <= _PZ_CONVERGENT_CEILING_MAX and r.abstain_correct_theta_disabled >= _PZ_ABSTAIN_MIN`.
3. `r0` = smallest-N rung. Not `ok(r0)` → `{"gate":"FAIL",...}`. Else
   if any larger rung not `ok` OR `full_acc < r0.full_acc - _PZ_SCALE_TOL`
   → `{"gate":"WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",...}`. Else
   `{"gate":"PASS",...}`. Every dict carries `gate`, `reason`,
   `frozen_bars` (echoing the six constants). `gate` ∈ exactly the
   four allowed values.

**Adversarial test matrix (≥ 12 cases, written FIRST, must fail before
impl):** frozen-constant pins; clean PASS; theta_disabled not
collapsing → FAIL; abstain below bar → FAIL; full below bar → FAIL;
small-load-only → WORKS-SMALL; below-min-seeds → VOID; ladder mismatch
→ VOID; non-finite → VOID; missing key → VOID; empty/non-list/None
→ VOID; bool not numeric → VOID; duplicate N → VOID; precomputed
verdict ignored → FAIL on bad raw; degenerate always-abstain → FAIL;
degenerate always-answer → FAIL; VOID ≠ FAIL distinct with metadata.

Imports ONLY `from __future__ import annotations`, `math`, `typing`.
No other imports. Commit (`feat: frozen fixed-bar three-state verdict
module for Pirazzini-reference three-layer stage`). Controller
verifies protected set byte-empty + constants verbatim.

---

## Task 2: The net-new theta-generator controller + multi-target ACh + one-shot encoding + within-theta-cycle decode + runner

**Files:** Create `research/runners/pirazzini_three_layer_runner.py`;
Test `tests/test_pirazzini_three_layer_runner.py`.

**Behavioral spec (genuine net-new wiring; reuse everything else
byte-unchanged; READ the real reused interfaces — do NOT guess):**

- Build substrate via REUSED `build_biological_brain_regions`
  (`enable_hippocampus_consolidation=True`, `enable_dlpfc_verb=True`)
  + the validated v16 concept-pool recipe (the Stage-1-and-SPEAR
  CLEAR construction path; do not override `num_traits`).
  `CoreSimConfig.enable_nmda=True` for dlpfc bistability.
- **Net-new EXTERNAL THETA GENERATOR CONTROLLER** (~50 lines): a small
  per-step class. Each `bridge.step_simulation(1)` call: compute
  current theta phase (period ~250 ms = 4 Hz at the bridge dt) and
  gamma sub-cycle index (~25 ms = 40 Hz). At THETA-TROUGH phase
  (within each theta cycle), write a depolarising current via
  REUSED `bridge.cp_external_input_current` onto the existing CA3-
  targeted inhibitory population (`dg_pv_basket` or the CA3 FS
  interneuron group — confirm exact target by reading the validated
  builder), but in a *disinhibitory* direction (drive the
  interneurons NEGATIVELY at trough so they release CA3 pyramidals;
  or drive CA3 pyramidals POSITIVELY at trough — implementer chooses
  the biologically more-faithful of these against the existing
  bridge topology). Hold the current OFF at theta-peak (CA3 inhibited
  again). This is the Pirazzini disinhibition mechanism, biology-
  faithful, NOT a synaptic_gain modulation.
- **Net-new MULTI-TARGET ACh NEUROMODULATOR CONFIG** combining REUSED
  primitives (`sim/neuromodulators.py` byte-unchanged):
  `NeuromodulatorConfig(name="ach_pirazzini", baseline=1.0,
  decay_tau_ms=..., concentration_min/max=...,
  production_rules=[ProductionRule(rule_type="manual")],
  targets=[
      ModulatorTarget(target_type="synaptic_gain",
                       scope="gate:ca3_to_ca1",
                       sensitivity= -K_suppress)   # HIGH ACh suppresses CA3->CA1
      ModulatorTarget(target_type="synaptic_gain",
                       scope="gate:lang_input_to_ca3",
                       sensitivity= +K_strengthen) # HIGH ACh strengthens cortical input
      ModulatorTarget(target_type="plasticity_rate",
                       scope="all",
                       sensitivity= +K_ltp_facilitation), # HIGH ACh facilitates LTP
  ])`.
  The K_* sensitivities are net-new tuning of REUSED primitives;
  implementer chooses biologically-reasonable values (document each
  in one ASCII comment). The controller calls
  `NeuromodulatorManager.set_concentration("ach_pirazzini", HIGH)`
  during encode windows and `set_concentration("ach_pirazzini", LOW)`
  during retrieve. (Standard Hasselmo polarity: encode HIGH, retrieve
  LOW — opposite to the SPEAR runner's plasticity_window_gate
  polarity, which was driven by the TAN-model convention.)
- **One-shot encoding routine**: present each episode pair
  `(Ep_i to L2 = ca1, Ep_i+1 to L1 = ca3)` SIMULTANEOUSLY for 250 ms
  per pair, ONCE. During encoding, ACh is HIGH (multi-target
  configuration active); the validated engram-tagging API
  (`bridge.start_engram_recording / commit_engram_tag /
  stimulate_tag`) records the bound ensemble. Use OPAQUE tag names
  (`f"ep_{i}"`); the answer must not be readable from any string —
  Stage-1 lesson.
- **Within-theta-cycle decode**: over ONE theta cycle (~250 ms),
  read the raw `lang_output` firing-rate confidence via the
  validated `compose_concept_engram.lang_output_pattern_during_*` +
  `cosine_to_word` path; rank concepts; pass top through REUSED
  `abstention_gate.gate(ranked, 650.0)`. The Pirazzini "70 % max-
  activity threshold" maps onto the moat's calibrated raw firing-rate
  threshold (650 was calibrated on encoded ~796 vs control ~584).
- **theta_disabled control arm**: identical to full minus only the
  external theta generator (no rhythmic disinhibition; CA3 remains
  inhibited throughout); SAME seed and SAME random draws.
- Emit per (seed, N) cell: `full_acc`, `theta_disabled_acc`,
  `abstain_correct_theta_disabled` (of ungroundable queries in the
  theta_disabled arm, fraction the moat correctly abstained on).
  Aggregate into rung dicts the verdict consumes; call
  `pirazzini_three_layer_verdict(rungs)`; include verdict + per-seed
  raw in `--out` JSON. Kill-safe via REUSED `sim.train_checkpoint`.
  `--tiny-synth` shrinks scale for fast smoke (toy NOT a result).
  CuPy real path; NumPy only for `--tiny-synth`. ASCII only. NO
  torch / autograd anywhere on any shipped path.

**TDD:** tests FIRST — `--tiny-synth` end-to-end produces well-formed
rungs the verdict accepts (one of four states, never raises, not VOID
for structural reason); the theta_disabled arm consumes the SAME seed
and SAME draws as full and differs ONLY by the external theta generator
disabled; the structural-effect pin (a 50-step constant-input probe
holding theta ON vs OFF must produce a NON-byte-identical bridge state,
proving the controller is mechanistically active — mirror the SPEAR
review's pin); no torch/autograd shipped; decode uses the validated
neural readout (no string parse on tag names); moat fed calibrated raw
firing-rate quantity. Run-fail → implement minimally and faithfully
against REUSED interfaces (read Stage-1's cleared substrate-build,
SPEAR's cleared moat-input pattern, the engram API, the validated
neural readout) → run-pass (pin green; core 17+; moat 7/7) →
commit (`feat: net-new Pirazzini three-layer runner -- external
theta-generator controller + multi-target ACh modulator + one-shot
encoding + within-theta-cycle decode (reuse-only; no autograd)`).
Controller verifies protected set byte-empty.

---

## Task 3: Dedicated adversarial review (BEFORE no-harm)

Fresh adversarial reviewer (mirror the Stage-1 and SPEAR reviews that
each found real defects). Primary mandate, PIRAZZINI-SPECIFIC:

1. **Is the theta-generator controller's disinhibition mechanism
   genuinely Pirazzini-faithful** — does it rhythmically release CA3
   pyramidals at theta-trough via the inhibitory population, NOT via
   a synaptic_gain modulation like the SPEAR runner's fix? Trace the
   external-current path; run the 50-step probe with theta ON vs OFF
   and confirm bridge-state divergence (mirror the SPEAR re-review's
   14.15 mV finding).
2. **Is the multi-target ACh modulator's three effects ALL active**
   (suppress CA3→CA1 + strengthen cortical input + facilitate LTP)?
   Identify each target's consumption site in sim/bridge.py; confirm
   each runs every step (or each step during the relevant phase),
   NOT inside the C2-reward-gated block. Measure each multiplier at
   HIGH-ACh vs LOW-ACh.
3. **Is the ACh polarity Hasselmo-correct** (HIGH ACh during
   encoding, LOW during retrieval)? This is OPPOSITE to the SPEAR
   runner's polarity choice; verify the implementer didn't
   accidentally inherit the SPEAR polarity.
4. **Is the decisive built-in control faithful**: theta_disabled
   genuinely "full minus only the theta generator" with same draws;
   theta_disabled_acc empirically collapses (rather than being a
   structural constant like the Stage-1 ablation defect).
5. **Can a degenerate / empty / single-mechanism solver score PASS
   via the runner+frozen-verdict end-to-end?** Re-run Stage-1/SPEAR-
   class exploits adapted (string-parse, additive, single-arm).
6. **Frozen bars `_PZ_*` immovable + no autograd shipped + subsystems
   genuine identity-imports byte-unchanged**.

STRENGTHEN-only fixes to non-protected files; commit prefix `review:`;
fix → re-review loop until CLEAR. Controller verifies protected set
byte-empty.

---

## Task 4: No-harm phase

Prove the full protected set is byte-unchanged from the pre-Task-0
base to HEAD (empty diff for every protected path);
`tests/test_abstention_gate.py` still 7/7; full Pirazzini + SPEAR +
Stage-1 + integrated suites green; assert no shipped path imports
`torch.autograd` / `.backward`. Commit no-harm evidence; controller
trust-but-verify; push both remotes.

---

## Task 5: CONTROLLER-ONLY decisive run (NOT a subagent task)

Controller, same turn, never stopping on a promise: (1) grounding-
first tiny-synth run (toy numbers explicitly NOT propagated);
(2) decisive kill-safe multi-seed run at the frozen ladder (2, 3, 5),
seeds 42 43 44, CuPy on RTX 3090, DURABLE capture to
`research/findings/raw/`, monitored to ACTUAL completion via a
genuine completion waiter; (3) mandatory smell-test scrutinising a
nominal PASS HARDER than a FAIL — recompute the verdict from the
single recorded output (no re-run, no bar change); confirm `full`
genuinely clears the bars AND `theta_disabled` genuinely collapses
to the convergent-ceiling level AND abstention holds; (4) honest
propagation of EVERY outcome (findings doc + `webapp/capability_status.json`
pillar, status PREDICTED until a clean scrutinised PASS, schema-
green + state file + commit + push BOTH remotes); (5) autonomous
next step per outcome: PASS → the Pirazzini-reference is the
biology-faithful mechanism that lifts compositional readout above
the trustworthy threshold — biology-translatable insight directly
about hippocampal-PFC integration; queue Stage-B
(imagination/dreaming modes); FAIL/VOID/WORKS-SMALL → follow the
biology to the next integration-fidelity refinement (the Orchard
2023/2024 spiking-phasor FHRR phase-coded-VSA Stage-C is the
already-pre-registered candidate; broader-search-first applies
again at that design-pass entry).

**Honest ceiling (unchanged from design):** a clean scrutinised
success = a biology-grounded Pirazzini-reference three-layer
architecture, on our validated substrate with correct Hasselmo ACh
polarity and biology-faithful disinhibition-based theta, shows
grounded sequence-retrieval above the trustworthy moat threshold
that the convergent Stage-1 + SPEAR ceiling could not — at
biological scale, multi-seed. NOT fluent open-ended language,
NOT an LLM. Under the reframed top-level goal (artificial life with
proper brain analogue, biology-translatable insights), either
outcome IS the deliverable: a PASS confirms the missing mechanism;
a FAIL identifies the next ceiling and the next refinement.
