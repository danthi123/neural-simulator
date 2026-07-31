---
type: plan
status: live
date: 2026-05-25
---

# Direction Q design: dlpfc_wm scale-up to test Wang 2002 NMDA persistent activity at biological scale

**Date:** 2026-05-25
**Status:** Design pass (brainstorming output); next step writing-plans

## Goal

Test the central convergent hypothesis from the substrate-bound arc:
the project's 10+ convergent NEGATIVE / BOUNDARY findings across both
the dynamics-gating class (closed) and the phase-coded representation
class (substrate-bounded) point to **substrate scale** as the
load-bearing bottleneck. Direction I (PFC NMDA bistability) failed
at the 60-neuron dlpfc_wm substrate across 4 cheap probes (basic
smoke / parameter stress / direct injection / HH biophysics variant).
The convergent diagnosis: 60 neurons is below the biological scale
needed for cortical recurrent-NMDA attractor dynamics.

Direction Q **directly tests this hypothesis** by scaling dlpfc_wm
to ~1000+ neurons with dense recurrent connectivity and a stimulus
protocol matching the Wang 2002 published delayed-response paradigm.
The result is biology-translatable either way:

- **PASS** (>= 2 sec persistent activity above baseline, multi-seed):
  substantiates a scale-threshold for cortical NMDA attractor
  dynamics in our substrate; opens scale-up across other regions
- **FAIL** at 1000 neurons: deeper structural insight (scale alone
  insufficient; identifies what microcircuit element / cell type /
  parameter is the actual bottleneck)

Both outcomes are biology-translatable per the project goal
(artificial life with proper brain analogue; honest negatives under
strict biology ARE the scientific deliverable).

## Biology reference (load-bearing)

**Wang 2002** ("Probabilistic decision making by slow reverberation
in cortical circuits") established the canonical cortical recurrent-
NMDA attractor model:
- 1600 pyramidal + 400 interneurons (~2000 total)
- Dense recurrent excitatory connections (high probability)
- NMDA dominates recurrent excitation (slow ~100ms decay)
- AMPA and GABA fast (~2ms / ~10ms)
- Persistent activity > 2 sec after transient stimulus

**Brunel-Wang 1999** ("Effects of synaptic noise and filtering on
the frequency response of spiking neurons") established the
mean-field analysis predicting the bistability requires:
- Sufficient recurrent excitation (positive feedback gain > 1)
- Sufficient inhibition for stability (prevents runaway)
- NMDA's slow decay maintains depolarization across spike intervals

Scale-dependent prediction: persistent activity requires the
recurrent loop to sustain ~10-30 Hz population rate above
spontaneous; below a critical population size, the network cannot
maintain enough simultaneous spikers to keep the NMDA depolarization
above threshold.

## Existing infrastructure

Already present in the project (per grep of g11_bg_runner.py +
sim/bridge.py):

- `dlpfc_wm` BrainRegion (g11_bg_runner.py:412): exc_fraction=0.8,
  internal_density=pfc_internal_density, exc_weight_mean=2.0,
  inh_weight_mean=4.0, plastic_internal=True, izh_neuron_type=
  IZH2007_HIPPO_PYRAMIDAL, enable_nmda=pfc_enable_nmda
- NMDA dynamics (sim/kernels.py): fused_nmda_update_and_current
  with voltage-dependent Mg2+ block; nmda_tau_decay default 100ms
  (matches Wang 2002)
- Per-region enable_nmda flag (regions.py): bistability only on
  regions with enable_nmda=True
- `--enable-pfc-nmda` CLI flag in g11_bg_runner.py (Cluster G v2.5):
  validated for 4-action navigation task with PFC + cortex_X NMDA
- 4 cheap-probe runners in Direction I that all failed at n_pfc=60:
  - `direction_I_stage1_pfc_smoke.py`
  - `direction_I_stage1_pfc_stress.py`
  - `direction_I_stage1_direct_inject.py`
  - `direction_I_stage1_hh_inject.py`

## Architectural approaches considered

### Approach A: In-place dlpfc_wm scale-up via g11_bg_runner override

- Modify g11_bg_runner.py CLI to accept `--n-pfc 1000` (currently
  default is 60)
- Modify `--pfc-internal-density 0.10` (currently low)
- Run the same 4-probe pattern from Direction I at n=1000

Pros: Minimal architectural change; reuses validated runner
Cons: g11_bg_runner.py is the cheat-5 navigation runner; testing
PFC bistability there entangles with the BG cascade + curriculum.
The Direction I probes already bypass most of g11_bg_runner.py's
machinery to isolate PFC.

### Approach B (RECOMMENDED): Standalone dlpfc_wm test bridge

Build a fresh, isolated test bridge with ONLY:
- dlpfc_wm region at n=1000 neurons (1600 if HH-quality replication)
- Stimulus input region (small; for triggering the persistent activity)
- All NMDA + recurrent dense connectivity
- Direct activity recording (no readout via lang_output etc.)

Test protocol:
1. Baseline period (e.g., 500 ms): record spontaneous activity
2. Cue period (e.g., 500 ms): inject transient stimulus current
3. Delay period (e.g., 3000 ms): no stimulus, record dlpfc_wm rate
4. Measure persistence: mean rate during delay > baseline rate?

Pros: Isolates Wang 2002 mechanism cleanly; no integration confounds;
easy to debug; pure biology test
Cons: Doesn't test integration with broader substrate (but that's a
separate downstream step)

### Approach C: Direct Wang 2002 published-parameter replication

Build a bridge that matches Wang 2002's parameters exactly:
- 1600 pyramidal + 400 interneurons (= 2000 total)
- Connection probability ~0.2 recurrent
- NMDA-AMPA conductance ratio ~0.05-0.10 (Wang 2002 fig 2)
- Use HH biophysics (Wang 2002 used Hodgkin-Huxley)
- Replicate the delayed-response task: 2-choice stimulus, 6 sec delay

Pros: Highest biology-translatability; direct comparison to published
Wang 2002 results
Cons: Substantial implementation; HH at 2000 neurons is slow (~30
sec/sim-second on our GPU per prior HH timing); needs careful parameter
mapping from Wang 2002's GENESIS model to our Izh/HH framework

## Recommendation: Approach B (standalone) -> Approach C (replication) if B PASSes

Approach B is cheapest-first per the autonomous-runs principle.
Build standalone test bridge; isolate Wang 2002 mechanism; test
persistent activity bistability. If PASS, then Approach C provides
the Wang 2002 published replication for biology-translatable
comparison.

If Approach B FAILS at n=1000 (despite NMDA + dense recurrent),
that's a sharper biology finding than another "60-neuron fail"
- localizes the actual gap to something other than scale.

## Pre-registered test + bar (frozen; never tuned)

**Test (Wang 2002 delayed-response protocol simplified)**:
1. Run bridge for 500 ms baseline (no stimulus); record dlpfc_wm
   per-neuron spike counts; compute baseline population rate
2. Apply cue stimulus to a sub-population of dlpfc_wm (e.g., 50% of
   excitatory neurons) for 500 ms at amplitude sufficient to drive
   above-baseline firing
3. Remove cue; run 3000 ms delay period
4. Compute mean dlpfc_wm excitatory-population rate during the delay
5. Compare to baseline + control (NMDA disabled, otherwise identical)

**Bar (pre-registered fixed, multi-seed)**:
- `Q_BISTABILITY_PASS`: delay-period rate >= 2x baseline rate AND
  delay-period rate is sustained (variance < 50% of mean) for
  >= 3 sec (entire delay), on >= 3 of 3 seeds [42, 43, 44]
- `Q_BISTABILITY_PARTIAL`: 1 or 2 of 3 seeds PASS, or delay-period
  rate elevation present but <2x or <3 sec; honest BOUNDARY
- `Q_BISTABILITY_NEGATIVE`: 0 of 3 seeds PASS AND delay-period rate
  not distinguishable from baseline; closes the hypothesis that scale
  alone fixes Direction I

**Control (mandatory)**:
- Identical bridge with enable_nmda=False everywhere (AMPA-only)
- Same stimulus + delay protocol
- Should FAIL (no persistence) at any scale per Wang 2002 theory
- If control PASSes too, the persistence is not NMDA-driven and
  result is VOID not PASS

## Cost estimate

- Design + writing-plans: 1 turn (this + next)
- Implementation: 2-4 days (Approach B standalone test bridge)
- GPU testing: ~1-2 hr per seed at n=1000 Izhikevich (assuming
  similar to existing bridge wall clocks); 3 seeds = ~3-6 hr
- Multi-seed evaluation + smell test: ~30 min
- Findings doc + adversarial review: 1-2 turns

Total: ~3-5 days subagent-driven build + testing, well under the
1-2 week pre-registered estimate

## Discipline (binding)

- Bar UNCHANGED throughout (Q_BISTABILITY_* tags frozen above)
- No protected/frozen/moat modification
- No autograd
- GPU/CuPy for real runs; numpy only for cheap-first algebra
- Honest propagation EVERY outcome both remotes
- Pre-launch grep confirmed: no prior Direction Q dlpfc_wm n=1000
  work exists; this is genuinely net-new (Direction I tested n=60)
- Standalone test bridge approach isolates Wang 2002 mechanism;
  does NOT modify build_biological_brain_regions or any validated
  bridge builder

## Pre-staged post-Q chain (per verdict)

- **PASS**: pillar n=105 candidate; commit findings doc; adversarial
  reviewer dispatch; if reviewer CLEAR record pillar; integrate
  dlpfc_wm n=1000 into bio_brain_regions substrate (separate
  integration step); revisit Direction I bound (now closed by PASS
  at scale)
- **PARTIAL**: characterize the scaling envelope (test at n=200,
  500, 2000 to find the threshold); biology-translatable scale
  threshold finding
- **NEGATIVE**: deeper structural diagnosis required; localize what
  microcircuit element is missing (cell type ratios, AMPA/NMDA ratio,
  inhibitory connectivity pattern); informs next direction

## Files to create (writing-plans output expected)

- `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py`
  - Standalone test bridge with n=1000 dlpfc_wm
  - Stimulus + delay protocol
  - NMDA-on / NMDA-off control
  - Multi-seed [42, 43, 44]
  - Pre-registered verdict tags

- `tests/test_direction_Q_grounding.py`
  - Smoke tests for the test bridge construction
  - NMDA enabled verification
  - Stimulus protocol verification

## References

- Wang 2002: "Probabilistic decision making by slow reverberation
  in cortical circuits", Neuron 36:955-968
- Brunel & Wang 1999: "Effects of synaptic noise and filtering on
  the frequency response of spiking neurons", J Comput Neurosci 8:183
- Direction I findings (n=60 fails): `research/findings/2026-05-24-DIRECTION-I-Stage1-CLOSED-PFC-bistability-genuinely-fails-substrate-scale.md`
- Substrate-scale convergent finding: `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
- Cluster G v2.5 Wang 2002 calibration history: CLAUDE.md "Cluster G v2 (2026-05-01)" section
