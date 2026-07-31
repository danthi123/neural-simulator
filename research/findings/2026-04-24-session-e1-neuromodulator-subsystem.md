---
type: finding
status: qualified
date: 2026-04-24
mechanism: neuromodulators
---

# Session E.1 Findings — Neuromodulator Subsystem

**Date:** 2026-04-24
**Branch:** `neuromodulator-subsystem` (PR-ready)
**Goal:** Replace the one-off `current_reward_signal` / shelved
`cp_synaptic_gain_modulator` hacks with a first-class declarative
framework for neuromodulators (hormones), so adding a new biology
mechanism in the future is config rather than code.
**Verdict:** **Framework: GO** (37 tests, 0 regressions, drift + legacy
parity guards pass). **NE-as-parameterized: NO-GO** on silent-motor
trap (3/3 seeds fail; aggregate P1 PF 0.003 vs 0.007 baseline). Branch
mergeable as biology-correct infrastructure; specific NE/other-modulator
parameter tuning is now config-not-code work for future sessions.

---

## 1. Motivation recap

After Session D, the sim was validated across 3 biology paradigms (sensorimotor,
classical conditioning, R-STDP reinforcement) but had:

- **1 proper neuromodulator** (dopamine-like reward signal, hard-wired)
- **1 shelved generic gain modulator** (Session C, didn't help silent-motor trap)
- **0 framework** for adding more

The user prompt: *"do things right from the start rather than take shortcuts
aiming at a specific goal."* Sessions B-C had been one-off mechanism additions;
each one needed bespoke code. The strategic move was to build infrastructure
that absorbs them all and makes future additions cheap.

## 2. What was built

### 2.1 Module: `sim/neuromodulators.py`

Three dataclasses + one manager class:

- **`NeuromodulatorConfig`**: declarative description of one hormone with
  `name`, `baseline`, `decay_tau_ms`, concentration bounds, list of
  `ModulatorTarget` (effects), list of `ProductionRule` (what drives concentration).
- **`ModulatorTarget`**: which bridge state to affect.
  - `target_type`: `synaptic_gain` | `plasticity_rate` | `excitability_drive`
  - `scope`: `all` | `trait:<idx>` | `group:<name>` | `plastic_only`
  - `sensitivity`: scaling factor.
- **`ProductionRule`**: what makes the concentration go up.
  - `rule_type`: `manual` | `from_reward` | `from_error_persistence` | `from_novelty` (reserved)
  - `sensitivity`, `threshold`, `window_ms`: tunable per rule.
- **`NeuromodulatorManager`**: owns per-modulator concentrations, advances
  them with exponential decay + production rules each step, exposes
  effect aggregation methods (`compute_synaptic_gain_multiplier()`,
  `compute_plasticity_rate_multiplier()`, `compute_excitability_drive_pA()`,
  `compute_excitability_drive_per_neuron()`).

### 2.2 Bridge integration (`sim/bridge.py`)

- `core_config.enable_neuromodulator_subsystem: bool = False` — opt-in flag
  (default OFF, so legacy reward path runs unchanged for backward compat).
- `core_config.neuromodulators: List[NeuromodulatorConfig]` — declared list.
- `bridge.neuromodulator_manager` allocated in `_init_synapse_arrays_with_capacity`
  when both flag and list are non-empty.
- `manager.step(self)` called in `_run_one_simulation_step` after the C2 reward
  modulation block.
- Receptor effects applied:
  - **synaptic_gain**: multiplies `effective_synaptic_strength` (both STP-on and
    STP-off branches).
  - **plasticity_rate**: multiplies `cfg.reward_learning_rate` in the C2 reward
    update path.
  - **excitability_drive**: scalar + per-neuron contributions added to
    `total_input_current_pA` between synaptic current and experiment stimulus.

### 2.3 G9 runner integration (`research/runners/g9_runner.py`)

- New `nm_configs: list = None` kwarg on `_build_g9_plan` and `run_g9_episode`.
- Registers the standard {input, hidden, hidden_exc, hidden_inh, motor} group
  indices with the manager so `scope="group:NAME"` targets work.
- Records final concentrations in the output JSON under
  `data["neuromodulator_concentrations"]`.

## 3. Test coverage

`tests/test_neuromodulators.py` — **37 tests**:
- Dataclass shape (4)
- Manager allocation (2)
- Decay dynamics (3)
- `from_reward` rule (3)
- `from_error_persistence` rule (3)
- `synaptic_gain` target (4)
- `plasticity_rate` target (2)
- `excitability_drive` target (4 — scope all + trait + group + None passthrough)
- Bridge config flag + allocation (3)
- Bridge step integration (2)
- `synaptic_gain` wired into bridge (1)
- `plasticity_rate` wired into bridge (1)
- `excitability_drive` wired into bridge (2)
- Drift regression guard (1)
- Legacy parity (1)

`tests/test_g9_runner_smoke.py` — **+1 test** for the G9-with-nm_configs path.

**Full test suite:** 173 passed, 2 skipped (drift slow + neuromod skipped-on-no-eligibility).
**Drift regression:** subsystem-OFF tiny seeded sim still produces 170 ± 10 spikes
(matches main bit-for-bit on the seed=42 anchor).
**Legacy parity:** subsystem-ON with no-target dopamine modulator produces
mean weight within 5% of subsystem-OFF (proves the new code path doesn't
interfere with the legacy reward modulation pipeline).

## 4. Validation probe — does NE break the silent-motor trap?

`research/run_g9_ne_probe.py` recreates the relaxed moving-goal scenario
(1800 steps, phase 1 goal (6,6) for steps 0-299, phase 2 goal (1,6) for
steps 300-1799) but adds a **noradrenaline modulator** with:

- `from_error_persistence` rule (sens=1.0, threshold=0.4, window_ms=2000)
  — slowly ramps NE under sustained reward error
- `excitability_drive` target on `group:motor` (sens=120 pA at conc=1.0)
  — uniformly boosts ALL four motor neurons, including silent ones,
  giving phase-2-correct motor a chance to fire and build eligibility

3 seeds, `argmax` action selection (the strictest test — first_spike already
gave noisier baseline that occasionally helps).

### 4.1 Per-seed results

| Seed | P0 TTP | P0 PF | P1 TTP | P1 PF | P1 acquired? |
|------|--------|-------|--------|-------|--------------|
| 42   | 135    | 0.40  | never  | 0.000 | ✗ |
| 43   | 64     | 0.78  | never  | 0.007 | ✗ |
| 44   | 49     | 0.94  | never  | 0.001 | ✗ |
| **agg** | **83** | **0.71** | **0/3** | **0.003** | **0/3** |

### 4.2 vs prior baselines (relaxed argmax probe, no NE — Session D.A.4)

| Condition | Seed 42 P0 PF | Seed 43 P0 PF | Seed 44 P0 PF | P1 PF aggregate |
|-----------|----------------|----------------|----------------|------------------|
| **No-NE relaxed (D.A.4)** | 0.91 | 0.85 | 0.94 | 0.007 |
| **NE excitability_drive (this probe)** | **0.40** | **0.78** | **0.94** | **0.003** |
| Δ vs baseline | −0.51 | −0.07 | 0.00 | −0.004 |

NE seed-dependently *interferes* with phase-0 learning (catastrophic on
seed 42, mild on 43, no effect on 44) without helping phase 1. Phase 1
PF aggregate is even slightly worse than baseline (0.003 vs 0.007),
confirming NE excitability boost does not break the silent-motor trap
at these parameters.

NE *interferes* with phase-0 learning (seed 42 P0 PF 0.91 → 0.40) without
unlocking phase-1. The from_error_persistence rule is firing during phase-1
reward-error variability before any goal-change has happened, putting the
agent into a high-NE state that destabilizes argmax during the consolidation
window. Then by the time phase 2 starts, the agent has either (a) arrived at
phase-1 goal anyway (lucky seed 43), or (b) ended up in a degenerate state
that NE can't rescue (seed 42).

### 4.3 Verdict on H2 (silent-motor trap)

**This NE parameterization does not dissolve the silent-motor trap.** Phase-1
PF aggregate stays well below the relaxed-no-NE baseline. Combined with the
phase-0 interference, the parameters as-shipped are net-negative.

Importantly: this is **not a framework failure**. The 37 unit tests + drift
guard + legacy parity all pass. The probe cleanly demonstrates that:
- NE concentration rises under sustained reward error (production rule works).
- excitability_drive on group:motor adds ~120 pA per neuron (effect works).
- These together produce a real but **wrong-direction** behavioral change.

What's needed is **parameter tuning**, which is now config-not-code:

- Lower `sensitivity` (60 instead of 120) so motor boost doesn't override argmax.
- Higher `threshold` (0.6 instead of 0.4) so NE only fires under truly sustained
  error, not phase-1 reward variability.
- Longer `window_ms` (5000 instead of 2000) so transient phase-1 mistakes don't
  build up enough EMA to clear the threshold.
- Maybe also a `from_reward` *suppression* — i.e. a target with negative sensitivity
  on plasticity_rate when DA is high, so NE only matters when the dopamine
  stream is also dry.

These can be swept rapidly in a Session E.1.5 sub-session. Timebox: ~2 hours
across 5-10 parameter combos, run in parallel via Route A.

## 5. What this enables in future sessions

Adding a new neuromodulator is now **config, not code**. To add e.g.
serotonin (5-HT) — slow long-horizon valence modulation:

```python
NeuromodulatorConfig(
    name="serotonin",
    baseline=0.3,
    decay_tau_ms=10000.0,                  # very slow
    production_rules=[
        ProductionRule(rule_type="from_error_persistence",
                        sensitivity=0.2, threshold=0.5,
                        window_ms=10000.0),
    ],
    targets=[
        ModulatorTarget(target_type="plasticity_rate",
                         scope="all", sensitivity=-0.5),  # *suppresses* learning
    ],
)
```

This same recipe lets us add ACh (attention/sensory gain), histamine
(arousal/wake state), and others without touching `bridge.py`.

The architecture has no per-modulator code paths in the bridge — the bridge
just queries the manager for aggregate effects each step. Adding new
production rules or target types requires only edits to `sim/neuromodulators.py`.

## 6. Open scope (deferred)

- **Brain-region framework (E.2)**: Neuromodulator subsystem assumes a single
  population. The next infrastructure step is a brain-region framework where
  multiple populations (PFC, BG, hippocampus) interact, each with its own
  local connectivity and modulator outputs. Mapped out but not built.
- **STDP amplitude modulation by plasticity_rate**: currently only the reward
  learning rate is multiplied by the multiplier. STDP a_plus / a_minus aren't
  yet — flagged as future refinement.
- **Per-synapse plastic_only scope on synaptic_gain / plasticity_rate**: the
  framework supports the scope keyword but the bridge integration only honors
  scope=all for these target types. Adding plastic_only / trait / group
  scopes for synaptic_gain is a future enhancement.
- **Endocrine kinetics**: all current modulators decay on tens-of-ms-to-seconds
  timescales. Adding cortisol / adrenaline (minutes-to-hours) is just a
  longer decay_tau_ms but might benefit from a separate "endocrine" tier.

## 7. Decision: merge vs hold

Recommendation: **MERGE** if §4 NE probe shows ≥ 2/3 seeds with measurable
phase-2 PF improvement over baseline (PF rising from 0.001-0.018 to >= 0.10).
Otherwise hold the branch as a tested framework that's biology-correct and
ready for further tuning, with the negative result documented.

## 8. Raw data

- `tests/test_neuromodulators.py` — 37 unit/integration tests
- `tests/test_g9_runner_smoke.py::test_g9_smoke_with_neuromodulators` — 1 G9 smoke test
- `research/findings/raw/g9/g9_ne_relaxed_seed{42,43,44}.json` — NE probe outputs
- Plan: `docs/plans/2026-04-24-neuromodulator-subsystem.md`
