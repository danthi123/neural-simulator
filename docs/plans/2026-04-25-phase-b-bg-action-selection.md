# Phase B: BG-style Action Selection Module

**Status:** Plan only (Phase A complete; awaiting direction to proceed)
**Branch:** pfc-working-memory (current)
**Estimated effort:** 1-2 days for MVP, 3-5 days for full validation

## Why this is the next step

After Phase A, the silent-motor-trap arc is conclusively closed. The trap
is structural — argmax over a random-readout layer interacts with reservoir
state bias to lock in entrenched winners. No runner-side intervention fixes
this on the existing G9 architecture (V1-V7 all explored).

The principled architectural fix is real action selection in a basal-ganglia-
style circuit: separate dedicated populations per action, lateral inhibition
between them, action-attributed dopamine modulation. This dissolves the
trap by *construction* — there is no shared spike-count argmax for E to
dominate over W.

Phase A delivered the cellular building blocks: 8 new IZH2007 presets for
striatum, GPe, GPi, STN, thalamus, hippocampus, DA. All work at 37°C with
distinct biological characteristics matching their target cell types.

## Architecture

```
                       VTA/SNc DA neurons (5-10 cells)
                                │
                ┌───────────────┴───────────────┐
                │ targeted DA per striatal pool │
                ↓                               ↓
  cortex ─→ striatum_D1[N,E,S,W]         striatum_D2[N,E,S,W]
            (50 MSN per action)          (50 MSN per action)
                  │                              │
             direct path                   indirect path
                  │                              │
                  ↓                              ↓
            GPi[N,E,S,W]  ←─ STN ──── GPe[N,E,S,W]
            (10 cells/action)         (10 cells/action)
                  │
                  ↓ disinhibition
          thalamus_VL[N,E,S,W]
            (10 cells/action)
                  │
                  ↓
            motor_M1[N,E,S,W]
            (10 cells/action, lateral inhibition between actions)
```

Total: 4 actions × ~150 neurons/action = ~600 BG neurons, plus 100 cortex
input + 40 motor + 10 DA = ~750 neurons. Manageable.

## BrainRegion declarations

```python
# Per-action striatal populations
for a, name in enumerate(["N", "E", "S", "W"]):
    BrainRegion(name=f"striatum_D1_{name}", n_neurons=50, exc_fraction=0.0,
                # MSN are inhibitory output; D1 directly inhibits GPi
                internal_density=0.05,
                neuron_type=IZH2007_STRIATAL_MSN, ...)
    BrainRegion(name=f"striatum_D2_{name}", n_neurons=50, exc_fraction=0.0,
                neuron_type=IZH2007_STRIATAL_MSN, ...)
    BrainRegion(name=f"gpi_{name}", n_neurons=10, exc_fraction=0.0,
                neuron_type=IZH2007_GPI_OUTPUT, ...)
    BrainRegion(name=f"gpe_{name}", n_neurons=10, exc_fraction=0.0,
                neuron_type=IZH2007_GPE_PACEMAKER, ...)
    BrainRegion(name=f"thal_{name}", n_neurons=10, exc_fraction=1.0,
                neuron_type=IZH2007_THALAMIC_RELAY, ...)
    BrainRegion(name=f"motor_{name}", n_neurons=10, exc_fraction=1.0,
                neuron_type=IZH2007_RS_CORTICAL_PYRAMIDAL, ...)

BrainRegion(name="stn", n_neurons=20, exc_fraction=1.0,
            neuron_type=IZH2007_STN_BURST, ...)
BrainRegion(name="dopamine", n_neurons=10,
            neuron_type=IZH2007_DOPAMINE, ...)
```

## RegionPathway declarations

```python
# Direct pathway (per action)
for a in ["N", "E", "S", "W"]:
    RegionPathway(
        from_region="cortex_input", to_region=f"striatum_D1_{a}",
        density=0.5, weight_mean=0.5, plastic=True,
        # cortical-striatal plasticity is the LEARNING site
    )
    RegionPathway(
        from_region=f"striatum_D1_{a}", to_region=f"gpi_{a}",
        density=0.5, weight_mean=2.0,  # strong inhibitory
        plastic=False,
    )
    RegionPathway(
        from_region=f"gpi_{a}", to_region=f"thal_{a}",
        density=0.5, weight_mean=2.0,  # tonic inhibition; "open the gate"
        plastic=False,
    )
    RegionPathway(
        from_region=f"thal_{a}", to_region=f"motor_{a}",
        density=0.5, weight_mean=1.5, plastic=False,
    )

# Indirect pathway (per action)
for a in ["N", "E", "S", "W"]:
    RegionPathway(
        from_region="cortex_input", to_region=f"striatum_D2_{a}",
        density=0.5, weight_mean=0.5, plastic=True,
    )
    RegionPathway(
        from_region=f"striatum_D2_{a}", to_region=f"gpe_{a}",
        density=0.5, weight_mean=2.0,  # inhibitory
        plastic=False,
    )
    RegionPathway(
        from_region=f"gpe_{a}", to_region="stn",
        density=0.3, weight_mean=1.0,  # inhibitory
        plastic=False,
    )
    RegionPathway(
        from_region="stn", to_region=f"gpi_{a}",
        density=0.3, weight_mean=1.0,  # excitatory
        plastic=False,
    )

# Lateral inhibition between motor populations (real M1 architecture)
for a in ["N", "E", "S", "W"]:
    for b in ["N", "E", "S", "W"]:
        if a == b: continue
        RegionPathway(
            from_region=f"motor_{a}", to_region=f"motor_{b}",
            density=0.2, weight_mean=0.3,
            # mediated by FS interneurons in real cortex; here as direct
            # inhibitory projection for simplicity
            inhibitory=True,
        )
```

## DA modulation

```python
# DA targets D1 (positive sensitivity) and D2 (negative sensitivity)
NeuromodulatorConfig(
    name="dopamine",
    baseline=0.0, decay_tau_ms=200.0,
    production_rules=[
        ProductionRule(rule_type="from_reward", sensitivity=1.0),
    ],
    targets=[
        # D1 pathway: DA enhances direct path response
        ModulatorTarget(target_type="excitability_drive",
                        scope="group:striatum_D1", sensitivity=20.0),
        # D2 pathway: DA suppresses indirect path response
        ModulatorTarget(target_type="excitability_drive",
                        scope="group:striatum_D2", sensitivity=-15.0),
        # Plasticity gating: DA enables/disables corticostriatal LTP
        ModulatorTarget(target_type="plasticity_rate",
                        scope="all", sensitivity=2.0),
    ],
)
```

## Tasks

### B.T1: Wiring infrastructure

- Add `g11_bg_runner.py` patterned after `g9_runner.py` but using
  `BrainRegion` + `RegionPathway` declarations above.
- Validate that all regions instantiate, all pathways connect.
- Smoke test: 30-step episode runs without errors.

### B.T2: Resting behavior validation

Run network with no input. Verify:
- DA neurons fire at ~3-5 Hz (tonic)
- GPe at ~30-60 Hz (autonomous)
- GPi at ~60-80 Hz (high tonic — suppressing thalamus)
- Thalamus suppressed (~0-2 Hz)
- Motor cortex silent (no thalamic drive)
- Cortex/Striatum at 0 Hz unless stimulated

If these match, the resting state is correct.

### B.T3: Action selection probe

Drive cortex_input with goal-direction signal. Verify:
- Striatum_D1 of correct action activates first
- That action's GPi is silenced
- That action's thalamus releases from inhibition
- That action's motor pool fires
- Lateral inhibition silences the other 3 motor pools

Stimulus-response cleanness is the key validation.

### B.T4: Single-goal learning

Run 100-trial episode with fixed goal (6,6). Track:
- Cortico-striatal weights to correct action grow
- Motor selection becomes consistent with goal direction
- Reward signal (via DA) potentiates correct D1 pathway and depresses
  incorrect D2 pathway

### B.T5: Moving-goal probe (silent-motor-trap rerun)

The acid test: rerun the 1800-step `(6,6)→(1,6)` scenario from G9. With
proper BG architecture and DA modulation, the agent should:
- Learn (6,6) in phase 0 via direct pathway potentiation
- Detect goal change via DA signal collapse
- Re-learn westward action in phase 1
- Phase 1 finalQ should drop below 4 (vs G9 baseline ~6.74, V1 ~6.40)

### B.T6: Tests, docs, commit

## Success criteria

1. Resting state has correct firing rates per region (per Phase A literature targets)
2. Action selection produces clean single-action winners (no shared-argmax bias)
3. Single-goal learning succeeds in <100 trials
4. **Moving-goal scenario: phase 1 finalQ < 4 in ≥2/3 seeds**

If criterion 4 holds, the silent-motor trap is dissolved at the architectural
level — which would be the headline result of the project so far.

## Estimated effort

- B.T1 (wiring): 4-6 hours
- B.T2 (resting validation): 2-3 hours
- B.T3 (action selection): 4-6 hours
- B.T4 (single-goal learning): 3-4 hours
- B.T5 (moving-goal probe): 3-4 hours wall + tuning
- B.T6 (tests/docs): 2 hours

Total: ~20-25 hours of focused work, spread over 1-2 days.

## Risks and unknowns

1. **MSN → GPi inhibition strength**: getting the disinhibition gate to
   open cleanly when D1 fires is parameter-sensitive. May need tuning.
2. **STN feedback timing**: indirect pathway has 3 hops (Striatum → GPe →
   STN → GPi). With Izh ~1ms spike timing, total path delay is ~10-15ms
   per cycle. Should oscillate at ~50-100 Hz beta range — consistent with
   biology.
3. **DA target scope**: `scope="group:striatum_D1"` would match all 4
   striatal_D1_{N,E,S,W} populations equally. For per-action DA delivery
   (the principled credit assignment), need `scope="group:striatum_D1_W"`
   selective by action — requires runner-side computation of action-specific
   reward error.
4. **Hyperparameter tuning**: this many regions and pathways = many degrees
   of freedom. Plan to start with biology-grounded defaults and only tune
   when behavior is clearly off.

## Alternative if this doesn't pan out

If the BG MVP shows the expected behaviors at rest but fails on moving-goal
readaptation, the next layer to add would be:
- PFC working memory (Session F revisit, but using IZH2007 instead of HH)
- Hippocampal context detection (DG pattern separation + CA3 replay)

These together address goal context + episodic recall, the other major
missing pieces from the brain mechanisms list.
