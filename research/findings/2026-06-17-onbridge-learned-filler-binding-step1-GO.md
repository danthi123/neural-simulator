# On-bridge learned-filler binding — STEP 1 (does it survive real spiking?) = 6-seed GO at full parity

**Date:** 2026-06-17 (CYCLE 149 — the owner-greenlit on-bridge binding build, first step)
**Status:** **GO, 6 seeds** (42, 43, 44, 100, 101, 102). The fixed-role + learned-filler **bundled** bind survives
real LIF spiking on the `SimulationBridge` at the parity dimension D_h=256 — on-bridge held-out **0.969** = **98%
of the numpy reference (0.990)**, ≫ memorization floor 0.000. NO protected `sim/` edit; reuse-by-import.
**Runner:** `research/runners/_phaseB_onbridge_frlf_bundle_derisk.py`
**Raw:** `research/findings/raw/_phaseB_onbridge_frlf_bundle.json`

## Context — what this step gates

The dendritic-build gate (the 6-seed A/B `2026-06-17-fixed-role-learned-filler-bundling-derisk.md` + the D_h
capacity sweep) established that a **fixed self-inverse role** (±1) + **learned filler codes** (+ a learned
read-out) bundles multi-attribute facts to ~1.000 at D_h=256 (numpy) — full parity with the fixed FHRR algebra,
where a learned *linear* inverse and additive could not. The owner greenlit the on-bridge build to replace the
composer's exact-inverse vector-algebra **idealization** with this learned binder realized on real spiking
neurons. Cheap-first discipline: before any protected edit, prove the load-bearing claim — **does the learned
bundling survive real LIF spiking dynamics** (threshold, refractory, finite spike count)?

## Method (reuse-by-import; no `sim/` edit)

Reuses the validated on-bridge ON/OFF substrate (`_phaseB_onbridge_bind_nonlinearity_derisk`): two LIF
populations `bind_pos`/`bind_neg` (population code, 16 neurons per bind dimension for signal-to-noise), driven by
the positive and negative halves of the bound vector as external current, read back as per-neuron spike **rates**
= the spiking ON/OFF bound. The only change from the prior on-bridge harness is the binder: the **fixed ±1 role +
learned filler** (`FixedRoleLearnedFillerBinder`, the arm that reached parity) and the **bundled** (3-way
superposition) eval at D_h=256. Per split: train the filler/read-out weights bundle-aware on the train
role-filler combos; for each held-out bundled fact, drive the analog bundle onto the LIF ON/OFF populations, read
the spiking rates, reconstruct the bundle from the rate difference, unbind each role, clean up to the nearest
concept. Compare on-bridge (spiking) vs the numpy reference vs the memorization floor.

## Result — 6 seeds, D_h=256

| seed | on-bridge bundled held-out | numpy reference |
|---|---|---|
| 42 | 1.000 | 1.000 |
| 43 | 0.938 | 1.000 |
| 44 | 1.000 | 1.000 |
| 100 | 0.875 | 0.938 |
| 101 | 1.000 | 1.000 |
| 102 | 1.000 | 1.000 |
| **mean** | **0.969** | **0.990** |

memorization floor 0.000 · chance 0.062 · on-bridge = **98%** of numpy.

## Reading it (the load-bearing reframe)

- **The learned binding survives real spiking at full parity.** The population rate-code carries the bundled
  superposition through real threshold/refractory/finite-count dynamics with only a ~2% drop from the numpy
  ceiling — no collapse, multi-seed.
- **No new dendritic mechanism is needed for this binding.** Because the role is a *fixed* ±1 self-inverse
  pattern, the bind reduces to a fixed per-dimension sign / ON-OFF-channel swap (linear routing), not a product
  of two *variable* operands — so the `fused_coincidence_plateau` dendritic-multiplication primitive (scoped at
  CYCLE 144 as the likely requirement) is **not required** for the fixed-role learned-filler scheme. The genuine
  on-substrate question was whether the spiking superposition survives, and it does. This **shrinks the build**
  from "realize a dendritic multiplication" to "wire the existing ON/OFF population substrate as a production
  composer path."
- The learned part that matters — the **filler codes** and the **read-out cleanup** — is what generalizes (the
  held-out systematicity already established) and is exactly the idealization-removing piece: the composer's
  exact-inverse algebra is replaced by a learned, lossy, spiking read-out over learned codes.

## Honest scope

This step validates the spiking *forward* path of the binding (the bind nonlinearity + the bundled superposition
on real LIF rates), driven by external current with the filler/read-out weights trained off-substrate. The next
build step is the production wiring: a composer path that (a) projects the filler codes through real synapses
(the known-easy part — population codes carry graded values at ~94%, established CYCLE 91), (b) does store/query
with the no-confab moat, and (c) trains the read-out on the conversation. Whether any of that needs a protected
`sim/` edit (vs pure reuse-by-import of the brain-region + external-current machinery) is the next question;
Step 1 says the *mechanism* is already there.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_frlf_bundle_derisk \
    --dh 256 --seeds 42,43,44,100,101,102
```
