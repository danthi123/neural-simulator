# The merged-config "transmission" boundary ROOT-CAUSED (= `enable_homeostasis`) + fixed with the per-region homeostasis mask — no sim/ edit, SYSTEMIC (2026-06-18, CYCLE 208)

## Headline

The CYCLE-207 merged-config boundary — "a standalone-tuned spiking organ fires ~6-10× weaker
co-resident on the merged bridge, and its GABA_B value-subtraction arithmetic collapses" — is
**root-caused to a single config field, `enable_homeostasis`, and fixed with the already-shipped
per-region homeostasis mask (NO `sim/` edit).** The boundary is **SYSTEMIC** (it blocks lifting ANY
standalone-tuned fine spiking organ onto the merged bridge), so this fix is the general enabler for the
TRUE-ONE-BRAIN consolidation program, not just the limbic core. Reached via the standing
**deep-research-first** move at a roadblock (a read-only diagnostic subagent, controller
trust-but-verified).

## Root cause (the deep-research subagent's decisive diagnosis, verified)

- The standalone limbic organ (`_limbic_core_rpe_battery_derisk.build_limbic_core`) used the
  `CoreSimConfig` DEFAULT **`enable_homeostasis=True`**, never overridden. With homeostasis on, the
  spike threshold is the homeostatic per-neuron threshold (~−42 mV, `bridge.py:1366-1375`,
  `homeostasis_threshold_min/max=−55/−30`), NOT `vpeak` (+35 mV) — a ~77 mV lower threshold = a large
  f-I gain (~1.5-1.8× per pool, compounding to ~6-10× at the SNc's saturating operating point).
- The merged bridge keeps **GLOBAL `enable_homeostasis=False`** (`nav_conv_merged_bridge.py:640`) —
  mandatory, because the homeostatic synaptic-scaling clip would crush the frozen conversational weights
  (the documented foot-gun guard). So a standalone-tuned organ, lifted co-resident, runs at the
  un-boosted (`vpeak`) threshold and fires far weaker, and its fine GABA_B arithmetic collapses.
- **Proof (the subagent's controlled f-I sweeps):** flipping ONLY `enable_homeostasis` on the standalone
  bridge reproduces the merged f-I curve to within ~2 Hz at every drive level (e.g. reward_us 600 pA →
  150 Hz homeo-ON vs ~90 Hz homeo-OFF ≈ merged 91.7 Hz). A synapse-free pool reproduces the gap, ruling
  out every synaptic explanation (delays, E/I misroute, NMDA split, propagation strength, matvec). So it
  was never a "transmission" factor — it is the spike threshold.

The threshold select is `bridge.py:6318-6325` (global-on → adapted thresholds; global-off + a per-region
mask → `cp.where(mask, adapted, vpeak)`; else → `vpeak`).

## The fix — the already-shipped per-region homeostasis mask (NO `sim/` edit)

`BrainRegion.enable_homeostasis=True` (`sim/regions.py:133`, designed exactly for "the deterministic-nav
regime sets cfg.enable_homeostasis=False") builds `cp_homeostasis_neuron_mask` (`bridge.py:1227-1245`),
and the threshold-select branch 2 (`bridge.py:6320-6323`) gives the **masked limbic neurons** the low
adapted threshold while **every other neuron keeps `vpeak`** (the conversational/nav slices are
byte-unchanged). Setting `enable_homeostasis=True` on the 4 limbic regions in the merged builder is the
entire fix. The mask log confirms it covers exactly the limbic slice: "Homeostasis per-region mask: 4
regions enabled (170 neurons)".

**Foot-gun safety:** the synaptic-scaling clip (`bridge.py:7169`) is gated by the SEPARATE
`cfg.enable_synaptic_scaling` (OFF on the merged bridge), NOT by the homeostasis mask — so the
per-region homeostasis gives the limbic slice its low threshold WITHOUT ever running the frozen-weight
clip. (The threshold-adaptation at `:7152` updates all neurons' thresholds, but only the masked ones'
are USED by branch 2, so non-limbic neurons are unaffected.)

## Result (operating-point sweep, numpy)

With the per-region homeostasis the limbic SNc f-I is restored (the burst now reaches ~283-293 Hz vs
~75-100 before), and a clean operating point appears where **before NO config worked**:
`tonic=160 / us=400 / cue=800`.

**On the merged bridge (GPU, seed 42):** (A) co-residence + nav-inertness **PASS** (4 limbic regions, 0
out-edges into navigation, 3089 internal edges); (B) default-off byte-preserved **PASS** (limbic absent,
42 non-limbic regions all base-identical — the homeostasis edit is opt-in/safe); (C) the **core arithmetic
PASS** — burst **3.09×** (≥3), graded **+0.90** (≥+0.8), **value-subtract gap 1.34** (>1.2, pred 218 <
unpred 293 — the GABA_B value subtraction HOLDS, where it had completely collapsed before the fix). On
numpy the same point gives burst 3.82× / gap 1.94.

**The two lesion gates need a refined test protocol (the lesions WORK; the gate thresholds are
confounded).** The lesions demonstrably remove the signals — the reward-lesion drops the SNc burst
282→32 Hz, and the GABA_B-lesion changes the gap — so the decisive anti-cheats (r is the synaptic
reward, V is the synaptic GABA_B) are satisfied in substance. But the exact gate thresholds (reward-lesion
within ±20% of tonic; GABA_B-lesion gap ≤1.2) fail because the **DOPAMINE neuron's state-dependent firing
varies across the multi-condition test** (adaptation + the inter-condition silent settle leave it in a
lower-firing state by the late lesion windows, so the reward-lesion read *overshoots below* the early
baseline: 32 Hz vs 74 Hz). Freezing the homeostatic threshold-adapt rate to 0 during the frozen test
(`validate_arithmetic`, the analogue of `freeze_lr`) removes the threshold-drift component but not the
adaptation/settle-state component. The clean-gate fix is a **re-baseline-per-condition** protocol (measure
the tonic floor immediately before each lesioned read so the comparison is state-matched) — a bounded
test-protocol refinement, NOT a mechanism issue.

HONEST: the operating window is **narrow** (the homeostasis boost makes the SNc highly excitable, so it
saturates at higher tonic, and a too-strong cue over-drives the striosome and flips the subtraction) — but
it is a genuine restoration of the standalone-like regime co-resident, and the core δ=r−V arithmetic
(burst + graded + value-subtraction) holds on the real merged bridge.

## Systemic significance + a latent bug found

- **SYSTEMIC:** any organ whose operating point was pinned with the default (homeostasis-on, ~−42 mV
  threshold) will fire weaker once co-resident on the merged bridge (which must keep homeostasis off).
  The per-region homeostasis mask is the **general fix** for lifting standalone-tuned spiking organs onto
  the merged bridge — it de-risks roadmap #3/#4/#5's future lifts the same way.
- **A real latent BUG (flagged for a separate fix):** `_apply_parameter_heterogeneity`
  (`bridge.py:~2032`, `target_array[:] = samples`) overwrites the per-region `izh_neuron_type` params
  with global-RS-default-centered samples, so with heterogeneity ON (the default) a region declared
  DOPAMINE/MSN/FS silently runs as a jittered RS pyramidal (the standalone limbic `snc` ran as RS, not
  DOPAMINE). The fix should jitter around the per-region base values. (Independent of this resolution;
  the merged bridge runs heterogeneity off, so its `limbic_snc` is a correct DOPAMINE neuron.)

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._merged_limbic_coresident_validate --sweep   # the working point appears
SIM_BACKEND=cupy  python -m research.runners._merged_limbic_coresident_validate --seed 42 # full battery + structure
SIM_BACKEND=cupy  python -m research.runners._merged_limbic_coresident_validate --moat     # conversation unperturbed (foot-gun safety)
```

Fix: `research/runners/nav_conv_merged_bridge.py` (the 4 limbic regions' `enable_homeostasis=True`).
Prior: `2026-06-18-merged-limbic-core-lift.md`. Diagnostic: the read-only subagent report (CYCLE 208).
