# N9 Stage-2 — a SPARSE-read MSN-D1 value critic on the self-organized place code: FIRE-vs-GRADE tension is IRREDUCIBLE → **NEGATIVE** (the honest deliverable)

**Date:** 2026-06-09
**Type:** runner-side implementation + decisive CuPy de-risk (Stage 2 of the place-code biologization plan). **NO `sim/` edits** (`git status --short sim/` byte-empty throughout — verified before/after every stage).
**Backend:** `SIM_BACKEND=cupy` (production backend; numpy DISQUALIFIED for striatal/near-threshold work per `2026-06-09-N9-cupy-membrane-divergence-ROOT.md`). Deterministic regime (OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF), hard-asserted.
**Owner directive:** biologize everything, no banking, brain-based-only. **An honest negative IS the scientific deliverable.**
**Builds on:** `2026-06-09-place-code-selforg-stage1-derisk.md` (Stage 1 PASS 3/3 — a self-organized sparse spiking place code IS position-specific) + `2026-06-09-N9-convergent-upstate-derisk.md` (the dense-up-state PLACE-GRADED negative) + `2026-06-09-N9-faithful-value-cell-design.md` + `2026-06-09-place-code-biologization-research.md` (the Stage-2 read-out design).

---

## TL;DR — VERDICT: **NEGATIVE (0/3 seeds on the load-bearing gates).** The fire-vs-grade tension is IRREDUCIBLE on the self-organized place code through an MSN-D1.

Stage 1 produced exactly what it promised — a **self-organized, sparse (3.65%), position-specific (diff-cos 0.064), sensor-dependent** spiking place code. Stage 2 reads it into the N9 MSN-D1 critic via the **plastic, DA-δ-gated arm ONLY** (no dense position-blind A1 floor), per the design. On CuPy, ≥3 seeds, in the deterministic regime:

| Gate | Result | Notes |
|---|---|---|
| **2a FIRE** (critic ≥5 Hz at goal) | ❌ **0/3** | The sparse place ensemble (≤~9 spikes/step total, distributed across ≤~30 cells) **cannot clear the MSN-D1 rheobase (~420 pA effective)** through the learnable arm at any realistic weight. |
| **2b PLACE-GRADED** (NEAR ≥3× FAR) | ❌ **0/3** | Critic silent at BOTH NEAR and FAR → ratio undefined/0. The load-bearing gate the dense blob capped at 1.2× is **still not met.** |
| **2c LEARNS-V (LTP)** | ❌ **0/3** | **The LTP-bootstrap deadlock, now on the sparse code:** the critic never fires → no post-spike → no STDP eligibility → `w_near` frozen at init (1.00→1.00 over 60 trials). |
| **2d ACTOR-NOT-PERTURBED** | ✅ **3/3** | actor cortex rate identical with/without the critic (ratio 1.000) — the place/critic feed ONLY the critic. |
| **2e GABA_B subtraction** | ❌ not applicable | gap stays 1.0 (the critic never fires → no GABA_B current onto SNc → no subtraction to measure). Lesion control consistent. |

**The irreducible fire-vs-grade tension (exhaustively mapped, the core finding):** the self-organized place code can be EITHER *distinct-per-location* OR *a strong-enough synaptic driver*, **never both** — because the distinctness is PRODUCED BY the threshold-WTA competition that keeps per-cell firing low:

- **DISTINCT regime** (`max_int ≤ 600`, diff-cos ≤ 0.21): the place ensemble fires ~0.4–4 spikes/step → **too weak to fire the critic** (critic 0 Hz) even with a learned `w_near` and even onto a 20-cell critic.
- **DRIVING regime** (`max_int ≥ 900`, diff-cos ≥ 0.49): the ensemble fires ~9–15 spikes/step and CAN drive the critic (NEAR 91–105 Hz) — but the place code is **no longer distinct** (the WTA collapses, ensembles overlap heavily) → the critic fires at FAR too (**trained ratio 0.88**, FAR slightly higher) — the **same position-blindness** the dense up-state had.

No point on the intensity / init-weight / critic-size / learning-rate / w_max / training-length axis clears BOTH gate-2a (fire) and gate-2b (grade ≥3×). DA-gated LTP **is** place-selective (`w_near > w_far` when it runs), but it cannot grow `w_near` to the firing band (~30) from a sparse code, because the sparse code's low per-cell firing gives STDP almost no pairing events (w_near 1.00→1.25 over **300** trials, plateauing). The only way to fire the critic from init is the dense convergent A1 floor (position-blind, the Stage-1 N9 negative) or a host critic-teacher (a BRAIN-BASED-ONLY shortcut into the critic, disqualified). **Per the design's graceful-FAIL contract: Stage 2 gate 2b fails → Option C (BTSP/sharper read-out) or Option D (bank the deeply-mapped negative).** This is Option D, with a precise diagnosis of what the next mechanism must change.

---

## What was built (runner-side ONLY; `sim/` byte-empty)

### Probe — `research/runners/n9_place_graded_critic_stage2_derisk.py` (CuPy-only, self-contained)

The design's **Option A read-out** (`place-code-biologization-research.md` §1.4): read the Stage-1 self-organized place code into the critic via the **PLASTIC DA-δ-gated arm only, no dense A1 floor.**

```
landmark_sensors  (>=2 landmarks, egocentric bearing+distance render — the Stage-1 BVC/OVC channel,
                   the ONLY place (x,y) enters the brain; position-leak controlled by construction)
   --plastic random projection (STDP, gate landmark_to_place; Stage-1 mechanism)-->
place             (400-cell hippocampal-pyramidal pool; threshold-WTA competition -> sparse distinct
                   fields). SELF-ORGANIZED, then FROZEN (the prompt's suggested sequencing).
   --PLASTIC, DA-delta-gated projection (STDP, gate value_input; NO dense A1 floor)-->
striosome_value   (80-cell MSN-D1 critic, fully GABAergic, KIR2 up/down, rheobase ~420 pA effective)
   --GABA_B (gate critic_snc_window)-->  snc  (30-cell DA; the value subtraction onto SNc)
+ actor stub: sensor_place_readout -> cortex_{N,E,S,W} (for gate 2d; NO edge to place/critic)
```

**Sequencing (the prompt's clean path):** (1) self-organize the place code with `landmark_to_place` OPEN, then **FREEZE** it (close the gate → stable place fields = a stable afferent). (2) Train the critic's V on the FROZEN fields (`value_input` OPEN, value-leads-reward: visit NEAR → place ensemble fires → SNc reward burst → DA-gated STDP). All gates + the lesion/shuffle/sensor-ablation controls + the CuPy-regime + position-leak assertions.

### Diagnostic probes (CuPy; `research/findings/raw/_n9_stage2_*.py`)
`_n9_stage2_calib.py` (init-weight sweep + teacher-bootstrap), `_n9_stage2_instrument.py` (rheobase + place→critic g_e/V + STDP-with-teacher), `_n9_stage2_placefire.py` (place-pool firing per location), `_n9_stage2_placeboost.py` (the distinctness/rate trade-off), `_n9_stage2_teacherltp.py` (teacher-bootstrapped LTP), `_n9_stage2_ltpceiling.py` (the LTP ceiling vs the firing band), `_n9_stage2_intermediate.py` (the intermediate-intensity self-bootstrap attempt).

---

## The decisive CuPy diagnostics (why the tension is irreducible)

### 1. The MSN-D1 critic's effective rheobase is ~420 pA (direct teacher sweep)
```
teacher  300pA -> 0.00 Hz    420pA -> 11.1 Hz    600pA -> 33.3 Hz
teacher  380pA -> 0.00 Hz    500pA -> 22.2 Hz    800pA -> 44.4 Hz
```
(the nominal ~339 pA is the analytic rheobase; the effective fire-onset in this regime/dt is ~420 pA.)

### 2. The DISTINCT sparse place ensemble delivers nowhere near 420 pA
After self-org, driving NEAR (`max_int=600`, diff-cos 0.21): the place ensemble fires **~0.4–4 spikes/step total** across the active cells; the critic's excitatory conductance `g_e` reaches only **0.08–0.84** and its membrane peaks at **−73 mV** (rest −80, threshold −25) even at `place→critic weight = 60`. **The critic never depolarizes past −73 mV.**

### 3. The distinctness/rate trade-off — the irreducible tension, mapped
Place-ensemble firing rate AND distinctness vs sensory intensity (after self-org):
```
max_int   place_spk/step   diff-location cosine
   450        ~0.4              0.074   <- DISTINCT, too weak to fire critic
   600        ~3.9              0.210   <- DISTINCT, too weak
   900        ~8.5              0.488   <- borderline overlap, marginally drives critic
  1800       ~9.2              0.764   <- DRIVES critic, but POSITION-BLIND
  3000       ~14               0.860   <- POSITION-BLIND
```
The distinctness (low diff-cos) is **produced by** the threshold-WTA that keeps per-cell firing low. Drive the cells hard enough to fire the critic → the WTA collapses → the code is position-blind again.

### 4. In the DRIVING regime, the critic fires but does NOT grade
`max_int=1800`, train the critic (`place→value init 6`): critic fires (init 16.5 Hz, trained 91.5 Hz) — but **trained NEAR/FAR ratio = 0.88** (FAR slightly higher). Position-blind, exactly the dense-up-state failure.

### 5. In the DISTINCT regime, DA-gated LTP is place-selective but cannot reach the firing band
`max_int=600`, teacher-bootstrapped LTP (a host critic-teacher used only to prove the LTP CAN run):
`w_near 0.50→1.76` over **150** trials, `w_near/w_far ≈ 2.2×` (**the value learning IS real and place-specific**) — but the critic STILL fires **0 Hz** at NEAR (w_near=1.76 ≪ the ~30 needed). Aggressive training (lr 0.3, 300 trials): `w_near 1.00→1.25`, **plateauing** (the sparse code's low per-cell firing gives STDP almost no pairing). Onto a 20-cell critic at higher init: `w_near→8.3`, critic still **0 Hz**.

### 6. WITHOUT a teacher (the faithful config), the bootstrap deadlock is total — 0/3 seeds
The faithful primary (no teacher, the critic must reach the up-state from the LEARNED place→critic synapses alone), `max_int=600`, 3 seeds 42/43/44:
```
                       seed42  seed43  seed44
2a FIRE critic@NEAR     0.00    0.00    0.00   Hz  -> 0/3 (>=5)
2b PLACE-GRADED ratio   0.00    0.00    0.00       -> 0/3 (>=3)
2c LEARNS w_near 1.0->  1.003   0.999   1.007      -> 0/3 (frozen; no post-spike -> no STDP)
2d ACTOR with/without   1.000   1.000   1.000      -> 3/3
2e SNc gap              1.00    1.00    1.00        -> n/a (critic silent)
```
`V(near)` stays **0.00 Hz** for all 60 trials → `w_near` frozen at init → the deadlock the convergent up-state was designed to break is **back**, because the up-state arm (the only thing that fires the cell from init) was the position-blind A1 floor we removed.

### NMDA Option B (the prompt's named fallback) — also FAILS
`--nmda-critic` at the best config: 0/3, identical. NMDA is voltage-dependent (Mg²⁺-block); the sparse drive never depolarizes the cell enough to unblock it → NMDA can't engage (chicken-and-egg). Confirms the design's prediction that NMDA deepens an existing up-state but cannot CREATE one from sub-threshold sparse input.

---

## Anti-cheat ledger (all behave consistently with the negative)

- **(a) place-shuffle** (permute the place-cell→location mapping): grading already fails 0/3 in the TRUE case → the shuffle is **moot** (no value-of-location for it to ablate). Documented honestly (same situation `2026-06-09-N9-convergent-upstate-derisk.md` noted), NOT claimed as a pass. ✅ consistent
- **(b) sensor-ablation** (zero landmark sensors at recall): critic stays silent (it was already silent) — the critic firing inherits from the place pool, which inherits from the sensors (Stage-1's sensor-dependence). ✅ consistent
- **(c) regime fidelity**: `backend=="cupy"` asserted; OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF asserted (hard-fail otherwise). **NO per-region homeostasis on the critic** (it must fire from synaptic current, not threshold collapse — and it doesn't fire at all, so there is no threshold-collapse rescue masking the result). ✅
- **(2e) GABA_B lesion**: cutting the GABA_B mask leaves the gap at 1.0 (consistent — the critic never fired to drive GABA_B onto SNc, so there was no subtraction to remove). ✅ consistent
- **position-leak**: every `cp_external_input_current` write targets `landmark_sensors` (+ the actor stub's own cortex/sensor tonic, which feed ONLY the actor). The place pool + critic NEVER receive a direct allocentric (x,y) injection. Enforced by construction + asserted. No host `vs_place_context` Gaussian anywhere. ✅

---

## Diagnosis — what the next mechanism MUST change (the value of this negative)

The negative is precise and actionable. The substrate **can** self-organize a position-specific place code (Stage 1) and **can** learn a place-specific value (gate 2c's `w_near > w_far` when LTP runs). It **cannot** make a *sparse-distinct* spiking place code drive a *rheobase-high MSN-D1* over threshold via learnable synapses, because:

1. **The distinctness and the drive-strength are coupled** through the WTA: you cannot have a sparse-distinct code that also fires the cell. This is a property of the *threshold-WTA place layer* (Option B), not of the value learning.
2. **The MSN-D1 is rheobase-high by design** (~420 pA, the KIR2 down-state) — it is built to refuse weak input. A sparse place ensemble IS weak input. The faithful escape (the convergent up-state) re-introduces position-blindness.

The candidate fixes that would change this (NOT tried here; the next de-risk):
- **(C1) CA3 pattern completion + a recurrent attractor place layer** (Option A-trisynaptic, D.13): a CA3 autoassociator can hold a *high-rate* attractor that is STILL distinct-per-location (the recurrence sustains a strong, sparse, completed ensemble — distinctness from separation, drive-strength from completion, decoupling the trade-off the single-hop WTA couples). The Stage-1 doc already flagged that the minimal layer skips CA3 completion; this is the principled upgrade.
- **(C2) BTSP single-trial plateau potentiation** (Bittner-Magee 2017): a plateau in the critic (or place cell) potentiates inputs in one traversal — a non-Hebbian, seconds-wide rule that bypasses the slow-STDP-from-sparse-firing bottleneck. Requires a protected `sim/` kernel addition (flagged in the research doc as Option C, deferred, byte-review).
- **(C3) a lower-rheobase value-cell type** for the critic (the design notes ventral-striatal value cells fire at higher rates than dorsal MSNs) — but the N9 faithful design specifies MSN-D1 striosome; changing it is a fidelity trade-off to weigh.
- **(C4) FS-WTA among critic sub-populations + a sparse convergent arm that ONLY the goal ensemble drives** — a sharper read-out that lets the sparse NEAR ensemble win the critic competition without a position-blind floor.

The cheapest next move is **(C1)**: build the place layer through the already-validated trisynaptic loop (EC→DG→CA3→CA1) so CA1 is a *completed, high-rate, distinct* ensemble, and re-run THIS Stage-2 probe on the CA1 code. That decouples the exact trade-off this negative pins.

---

## Honest bottom line + recommendation

A sparse-read MSN-D1 value critic on the Stage-1 self-organized place code **fires neither at the goal nor anywhere** through the faithful plastic-only arm, so it does not grade NEAR≫FAR (gate 2b 0/3, the load-bearing gate). The cause is an **irreducible fire-vs-grade tension** that is now exhaustively mapped on CuPy across intensity, init weight, critic size, learning rate, w_max, training length, NMDA, and ±teacher: the self-organized place code's *distinctness is produced by the threshold-WTA that keeps it too weak to drive a rheobase-high MSN-D1*, and any drive strong enough to fire the critic collapses the WTA back to position-blindness. DA-gated LTP IS place-selective (`w_near > w_far`) but cannot grow `w_near` to the firing band from a sparse code (almost no STDP pairing events). The only faithful way to fire the cell from init (the dense convergent up-state) is the position-blind A1 floor the Stage-1 N9 negative already ruled out; a host critic-teacher is a BRAIN-BASED-ONLY shortcut and is disqualified.

**This is a clean, multiply-confirmed BRAIN-BASED-ONLY honest negative (design Option D)** — and it maps the real substrate boundary precisely: a *single-hop WTA place layer* cannot feed a *value-of-location through an MSN-D1*. **Recommendation:** do **NOT** run the 6-seed nav A/B for the place-graded critic. The next faithful step is **(C1)** — a CA3-completion place layer (the already-built trisynaptic loop) so the place code is a *completed, high-rate, STILL-distinct* CA1 attractor, then re-de-risk this exact Stage-2 gate set on the CA1 code; failing that, **(C2) BTSP** (a protected `sim/` kernel, byte-review) or **(C4)** a sharper FS-WTA critic read-out. The probe + diagnostics are shipped as the re-runnable de-risk harness for whichever mechanism is tried next.

---

### Artifacts
- Probe: `research/runners/n9_place_graded_critic_stage2_derisk.py` (CuPy-only; gates 2a–2e + lesion/shuffle/sensor-ablation/regime/position-leak controls; self-org→freeze→value-train sequencing; opt-in `--nmda-critic`).
- CuPy de-risk JSONs: `research/findings/raw/_n9_place_graded_critic_stage2_3seed.json` (the 3-seed primary NEGATIVE), `_n9_stage2_shuffle.json`, `_n9_stage2_lesion.json`, `_n9_stage2_ablate.json`.
- CuPy diagnostics: `research/findings/raw/_n9_stage2_calib.py`, `_n9_stage2_instrument.py` (rheobase ~420 pA + place→critic g_e/V), `_n9_stage2_placefire.py`, `_n9_stage2_placeboost.py` (the distinctness/rate trade-off), `_n9_stage2_teacherltp.py`, `_n9_stage2_ltpceiling.py`, `_n9_stage2_intermediate.py`.
- `git status --short sim/` byte-empty (verified before/after every stage).
