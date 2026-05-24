# Direction E theta-gamma substrate biologization — design

**Date:** 2026-05-24
**Status:** DESIGN (pending fresh-agent review of algebra probe, then implementation)
**Predecessor:** Direction E algebra probe (CONTROLS_DECISIVE; commits 1e14548 + 794f2f8)
**Mirror pattern:** FHRR-biologization arc (algebra -> resonate-and-fire -> attractor cleanup -> grounded symbols)

## Goal

Replace the Direction E numpy algebra reference with a spiking-substrate
implementation that uses real neurons, real spike timing, and the
project's existing oscillator + region infrastructure. Validate that the
multi-seed >= 0.80 algebra PASS survives biological-noise replacement at
each substitution step.

## What the algebra has that the substrate doesn't

The numpy reference assumes:
1. Discrete `GAMMA_PERIOD`-sample windows with sharp boundaries
2. Uniform pattern activation across the entire window
3. Cosine-match decoder reading exactly the right window
4. No spike-timing jitter beyond `PHASE_NOISE_STD`
5. No inter-region transmission delay
6. Perfect zero between gamma windows of different slots

The substrate has:
1. Continuous spike trains with no discrete window boundaries
2. Stochastic firing per neuron per simulation step
3. Per-neuron drive integration via membrane potential + threshold
4. Real synaptic transmission with axonal+synaptic delay
5. Bridge-wide simulation steps (dt=0.5ms or 1ms typically)
6. Overlapping firing across slot boundaries

## Substitution sequence (each its own pre-registered fixed-bar test)

### Substitution 1: theta clock generator

**Replace:** abstract slot indexing.
**With:** an explicit theta-clock signal that gates which gamma slot is
"active" at each simulation step.

**REUSE INSIGHT (2026-05-24):** the pirazzini_three_layer_runner.py
ALREADY has theta+gamma step counting from the bridge dt:

```python
theta_ms = 250.0   # Pirazzini 4Hz
theta_steps = max(2, int(round(theta_ms / cfg.dt_ms)))
gamma_ms = 25.0    # 40Hz
gamma_steps = max(1, int(round(gamma_ms / cfg.dt_ms)))

def _phase_is_trough(step_idx, theta_steps):
    return (int(step_idx) % int(theta_steps)) >= max(1, theta_steps // 2)
```

For Direction E theta-gamma (Lisman-Idiart 8Hz theta x 7 gamma slots):
```python
theta_ms = 125.0   # 8Hz theta (Lisman-Idiart)
gamma_ms = 17.86   # 7 cycles per theta = 56Hz gamma
theta_steps = ...
gamma_steps = ...

def _phase_to_gamma_slot(step_idx, theta_steps, n_gamma=7):
    phase = int(step_idx) % int(theta_steps)
    return min(n_gamma - 1, (phase * n_gamma) // theta_steps)
```

This is the **SIMPLEST substrate-faithful implementation**: NO new
oscillator region needed; NO new pacemaker entity. The slot index
is computed directly from the simulation step counter. The pirazzini
discipline of THREADING the absolute step index through every phase-
check call (FIX A from pirazzini commit history) is the discipline
to follow.

Implementation options (in order of build complexity):

**(1a)** Step-index phase function (CHEAPEST, recommended). Reuses
the pirazzini pattern byte-equivalent except `_phase_is_trough` ->
`_phase_to_gamma_slot`. No new region, no new neurons.

**(1b)** If (1a) is sufficient at controller-discipline tests but the
substrate response needs an actual oscillating signal: add a
`theta_pacemaker` region (~50 neurons; external sinusoidal drive)
that downstream regions can consume via the existing
NeuromodulatorConfig synaptic_gain pattern. Build only if (1a) is
insufficient.

**(1c)** Full intrinsic theta generator (e.g. MS-DBB analog with
NMDA bistability driving 8Hz oscillation). Deferred; full
substrate-internal autonomy is a much larger build.

**Pick (1a) for first substrate build**.

### Substitution 2: gamma-slot-gated encoding

**Replace:** "place pattern_c at slot i of ensemble".
**With:** drive the concept's lang_input drive ONLY during slot-i
steps within the theta cycle. The engram tag captures the spike-
counts at each simulation step.

Implementation: for each (concept, position) in the sequence,
drive lang_input(word) ONLY during simulation steps [theta_start +
i*GAMMA_PERIOD, theta_start + (i+1)*GAMMA_PERIOD]. Across multiple
theta cycles, the same sequence is presented repeatedly so STDP +
engram tagging capture the full temporal pattern.

### Substitution 3: phase-cued retrieval

**Replace:** "decode slot i by cosine-matching window i".
**With:** stimulate the engram tag ONLY during slot-i steps within
a recall theta cycle. Read which concepts fire most strongly during
that phase window. Use the existing `lang_output_pattern_during_stim`
primitive but parameterized to capture firing within a phase window
rather than across all stim steps.

Decoder = cosine-match the phase-windowed lang_output firing
pattern against each candidate concept's reference pattern.

### Substitution 4: substrate noise + biology validation

**Replace:** `PHASE_NOISE_STD=0.05` added to abstract ensemble.
**With:** the substrate's real spike-timing jitter + OU noise (already
present in the bridge's `cp_ou_process_state`).

Validation:
- Compare per-step jitter from substrate to algebra `PHASE_NOISE_STD`
- If substrate noise > algebra equivalent, verify the algebra was
  tested at that noise level (the noise-stress probe showed PASS
  through noise=5.0)
- If still passing, proceed to capacity sweep (slot_count 3..7)

## Pre-registered acceptance criteria (frozen, never tuned)

Same as Direction E algebra probe and Direction A main:
- **BAR**: 0.80 multi-seed mean slot-i accuracy at every load 2-7
- **MIN_SEEDS**: 3 (seeds 42, 43, 44)
- **N_TRIALS**: comparable to algebra (~100-300 per load per seed)
- **CHANCE**: 1/N_VOCAB (e.g., 1/16 = 0.0625)

If at any substitution the substrate result drops below BAR but the
algebra result passed at equivalent noise, the deliverable is the
precise biology-translatable bound (what biological constraint
broke the algebra-validated mechanism).

## Build order (subagent-driven-development)

1. **Task 0**: grounding pin (smoke; ensures harness wiring works)
2. **Task 1**: theta_pacemaker region + theta phase tracking in the
   bridge runner (no protected module modified; pacemaker region is
   plain RegionPathway addition)
3. **Task 2**: slot-gated encoding runner — uses validated
   start_engram_recording + commit_engram_tag + per-step
   gated lang_input drive
4. **Task 3**: phase-cued retrieval runner — wraps existing
   stimulate_tag + parameterized window readout
5. **Task 4**: end-to-end main probe (multi-seed; same task as
   Direction E algebra; 0.80 bar)
6. **Task 5**: adversarial review of Tasks 2 + 3 + 4 BEFORE no-harm
   (mirror the validated SPEAR/compose-bind pattern)
7. **Task 6**: no-harm phase (full protected set byte-empty diff;
   moat 7/7 green)
8. **Task 7**: CONTROLLER-ONLY decisive multi-seed run + smell test
   + dedicated fresh-agent adversarial review + pillar record (if
   PASS_CONTROLS_DECISIVE)

## Estimated cost

- Tasks 0-4: ~1-2 days of design + implement + smoke (CPU-bound for
  scaffolding; GPU for end-to-end runs)
- Task 5: ~1 hr (dedicated reviewer)
- Task 6: ~30 min (no-harm verification)
- Task 7: ~3-6 hr GPU + ~30 min smell test + 1 hr review

## Risk / what could go wrong

1. **Substrate noise might exceed algebra-noise budget**. The algebra
   noise-stress showed PASS through noise=5.0 (100x biological); if
   substrate noise is in that range it's fine. If it's beyond, the
   precise gap is the deliverable.
2. **Gamma window boundaries might bleed** in the substrate
   (spikes from slot i continue into slot i+1's window). The
   refractory period helps but isn't perfect. May need a hard reset
   between slots OR widen GAMMA_PERIOD.
3. **Theta clock implementation choice matters**. If the pacemaker
   region's gating is too weak, slot windowing leaks; too strong,
   regions don't fire at all. Mid-range gating is the safe default.
4. **Plasticity during encoding** may shift the concept->slot
   mapping. Should freeze the relevant gates (motor_to_*, *_to_motor)
   during encoding/retrieval, as the validated multitag does.

## Honest scope

This is a DESIGN doc; no claim has been made about substrate
biologization. The algebra PASS justifies the build investment; only
after the substrate implementation reaches CONTROLLER-VERIFIED Task 7
will the substrate-validated capability be claimed (and only with a
dedicated fresh-agent adversarial review CLEAR).

Direction E ALGEBRA validated (CONTROLS_DECISIVE) is a finding;
Direction E SUBSTRATE remains an open implementation question.

## Coordination with Direction A

Direction A (ec_context spatial positional binding) is currently in
flight on GPU. Outcomes:
- Direction A PASS_CONTROLS_DECISIVE: theta-gamma substrate becomes
  a COMPLEMENTARY biology-grounded mechanism; both available in the
  substrate; richer conversational primitives
- Direction A FAILs/COLLAPSES: theta-gamma substrate is the
  PRIMARY positional binding mechanism (the principled fallback)
- Direction A BOUNDARY: both are valuable; precise comparison
  characterizes which substrate-component combinations work where

In ANY case, theta-gamma substrate biologization is worthwhile work
because it adds a temporal-phase code primitive the substrate
doesn't currently have. The algebra clearance means the build is
not speculative.
