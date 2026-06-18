# Shared limbic core LIFTED onto the merged "one brain" — consolidation GO; arithmetic re-calibrated for the determinism config (2026-06-18, CYCLE 206)

## Headline

The validated standalone limbic organ (`2026-06-18-limbic-core-rpe-battery-GO.md`, Schultz RPE
battery 6/6) is now **co-resident on the merged nav+conversation `SimulationBridge`** — the single
highest-leverage TRUE-ONE-BRAIN consolidation step (the merged bridge previously had **no limbic
core at all**). The lift is an additive, default-off opt-in (`co_resident_limbic`) on
`build_merged_nav_conv_bridge` + `MergedNavConvAgent`. The structural consolidation is GO:
the limbic slice (`limbic_reward_us → limbic_snc ← limbic_striosome` GABA_B + the shared `dopamine`
modulator) builds co-resident (46 regions), is **nav-inert** (zero `cp_connections` out-edges into
navigation), and is **default-off byte-preserved** (the production conversational agent is byte-
identical). The spiking δ=r−V mechanism is confirmed to work co-resident (per-region instrumentation:
the reward burst fires, the GABA_B value subtraction works — cue+US snc ≪ US-alone snc). The full
multi-gate arithmetic required an operating-point re-calibration for the merged bridge's
determinism config (heterogeneity off), characterized below.

## What was built

- **`build_merged_nav_conv_bridge(co_resident_limbic=False)`** — when True, appends the limbic slice
  (`limbic_cue`, `limbic_striosome` MSN-D1 GABAergic, `limbic_reward_us` PPN-like, `limbic_snc`
  DOPAMINE) + the 3 limbic pathways AFTER the existing nav/parser/dlPFC/rf/cortex_it/gen slices (every
  existing index base is byte-unchanged), enables the GABA_B/GIRK conductance (only the
  `limbic_striosome→limbic_snc` pathway is tagged `receptor="gaba_b"`, so additive/zero-effect for
  every other synapse), and registers the `dopamine` `from_region_firing_signed` modulator over
  `[limbic_snc]` — the SHARED dopamine broadcast. THRESHOLD 0.0 makes it neutral-at-rest (a quiescent
  limbic_snc gives da_signal=0 → no plasticity-rate change → it cannot suppress the parser/
  conversational plasticity).
- **`MergedNavConvAgent(co_resident_limbic=False)`** — threads the flag through.
- **`research/runners/_merged_limbic_coresident_validate.py`** — the lift validation: (A) co-residence
  + nav-inertness, (B) default-off byte-preserved, (C) the RPE arithmetic on the merged slice,
  (D) moat-no-regression, plus `--diag` (per-region rates) and `--sweep` (operating-point search).

NO new `sim/` edit (the GABA_B/GIRK conductance is the already-shipped, owner-approved edit).

## Validations

| Claim | Result |
|---|---|
| (A) co-residence (4 limbic regions on the merged bridge) | **PASS** (46 regions total) |
| (A) nav-inertness (limbic out-edges into navigation) | **PASS** (0 out-edges; only intra-slice edges) |
| (B) default-off byte-preserved (limbic absent + non-limbic bases unchanged) | **PASS** |
| (C) the spiking δ=r−V mechanism works co-resident | **PASS** (diag: reward burst fires; cue+US snc ≪ US snc — the GABA_B subtracts) |
| (D) moat-no-regression (conversation survives the shared DA modulator) | **GO** — WITH the limbic core + the shared DA modulator co-resident: `what_does('dog','go')=='north'` (recall correct) AND `what_does('river','look') is None` (the no-confab moat abstains). The threshold-0 modulator does not perturb the parser/conversation. |

## The operating-point re-calibration (systematic-debugging, three factors)

The standalone organ's operating point was pinned with biological **heterogeneity ON** (the
`CoreSimConfig` default) and OU spontaneous activity. The merged bridge runs **heterogeneity OFF + OU
OFF** for nav/conversation determinism (multi-seed reproducibility). Three factors were localized:

1. **OU off** — the organ runs in its spontaneous-activity regime; the validation re-enables OU
   (sigma≈100) per the merged builder's explicit "the OU state stays allocated so a read can re-enable
   it" support. (Biologically, spontaneous synaptic bombardment is real; OU-off is a tooling choice.)
2. **Heterogeneity off (load-bearing for the GABA_B value subtraction)** — a controlled standalone
   test confirms: het-on → burst 3.75×, gap 3.0, 4/4; het-off at the SAME drives → the SNc saturates
   (362 Hz) so the GABA_B can't grade-subtract (gap 1.13). BUT het-off is fully **recoverable by drive
   re-calibration** (lower the tonic/US to avoid saturation → 4/4 on 5/5 standalone het-off configs).
   So heterogeneity is not strictly required; it just shifts the operating point.
3. **Lower effective excitatory transmission on the merged bridge** — per-region instrumentation: the
   reward_us fires fine (97 Hz) but each spike has ~6-10× less effect on the SNc than standalone (the
   SNc caps ~100 Hz vs 362), so the burst ratio needs a lower tonic. The limbic critic weights were
   raised (cue→striosome 10→40, reward→snc 10→30) so the het-off MSN fires reliably (at 10 it is cold,
   ~0-4 Hz on het-off).

⇒ **a systematic drive sweep (one build, 18 drive combos) found NO clean operating point** where the
burst ≥3× AND the GABA_B value subtraction hold together on the het-off merged config. The result is a
characterized boundary: at the validated weight 10 the subtraction works (the diag: cue+US snc 36 ≪
US-alone 82, gap 2.29) but the burst ratio fails (1.66×, the SNc tonic ~50 vs the ~100 Hz ceiling);
raising the cue→striosome weight to clear the cold MSN (→40) BREAKS the subtraction (the cue
over-drives → pred ≫ unpred, gap 0.2–0.4 in every config) and the het-off f-I is razor-steep (the SNc
tonic flips 0→172 Hz between tonic 160 and 210 pA). So the full multi-gate arithmetic is a genuine
boundary on the merged config (the systematic-debugging "question the architecture" point). The
committed lift keeps the **validated 10/10/10 weights**.

**The per-region-heterogeneity hypothesis was DE-RISKED and FALSIFIED.** A controlled test built the
merged bridge with global heterogeneity ON (the `_global_het_test` hook) and re-ran the sweep: it STILL
finds NO clean operating point (burst maxes ~2.84×, never ≥3×; gap ~1.0, the value subtraction GONE) at
the SAME 10/10/10 weights that give gap **3.0 standalone**. So **heterogeneity is NOT the fix** — a
deeper **merged-config factor breaks the limbic arithmetic regardless of het**: the SNc's effective
synaptic response is ~6-10× weaker on the merged bridge (the diag: each reward_us spike moves the SNc
+33 Hz vs +312 standalone; the SNc caps ~100 Hz vs 362), and the GABA_B value subtraction collapses with
it. The cause is an unpinned global-config interaction (NOT the NMDA-ratio — AMPA is applied fully to
non-NMDA neurons per `bridge.py:6083`; NOT a synaptic-strength setting — none differs in the merge
builder). ⇒ the real increment is to **root-cause the merged-config transmission difference** (a deeper
instrumentation pass on the SNc's synaptic-vs-direct-current response in the full network), OR to use
**on-merge critic LEARNING** (which adapts the weights to the merged operating point rather than
transplanting fixed standalone weights), OR to run the limbic slice in a **separate config regime**. NOT
per-region het.

## Honest scope / increment #2

This lift CONSOLIDATES the limbic slice onto the one brain (co-resident, nav-inert, default-off
byte-preserved) and confirms the δ=r−V mechanism co-resident. The remaining increments:
- the on-merge critic **LEARNING** (V via three-factor — the diag-validated mechanism supports it,
  pending operating-point calibration);
- routing the **reward source** from the navigation (N5) so the limbic core reads a nav-driven reward
  (roadmap #2);
- the shared **DA gating the nav actor** + (later) the conversational salience (roadmap #6);
- **root-causing the merged-config transmission factor** (the SNc's ~6-10× weaker synaptic response in
  the full network) — the de-risked NEGATIVE on per-region heterogeneity means the operating-point fix is
  this, NOT het. On-merge critic LEARNING (which adapts to whatever the merged operating point is) is the
  most likely path, since it sidesteps transplanting fixed standalone weights.

## Reproduce

```bash
SIM_BACKEND=cupy python -m research.runners._merged_limbic_coresident_validate --seed 42   # structure + arithmetic
SIM_BACKEND=cupy python -m research.runners._merged_limbic_coresident_validate --moat       # conversation survives the DA modulator
SIM_BACKEND=numpy python -m research.runners._merged_limbic_coresident_validate --sweep      # operating-point search
SIM_BACKEND=numpy python -m research.runners._merged_limbic_coresident_validate --diag       # per-region rates
```

Runner: `research/runners/_merged_limbic_coresident_validate.py`. Builder:
`research/runners/nav_conv_merged_bridge.py` (`co_resident_limbic`). Prior:
`2026-06-18-limbic-core-rpe-battery-GO.md`, `2026-06-18-full-spikeification-shared-substrate-roadmap.md`.
