# C1 — the canonical hippocampal TRISYNAPTIC LOOP driven from landmark sensors does NOT decouple the fire-vs-grade tension: **NEGATIVE (0/3)** — CA1 is EITHER distinct-and-silent OR firing-and-position-blind, never high-rate-AND-distinct

**Date:** 2026-06-09
**Type:** runner-side implementation + decisive CuPy de-risk (C1 = "the most biologically faithful lever" — make the canonical place-cell circuit work, or honestly map why it can't). **NO `sim/` edits** (`git status --short sim/` byte-empty throughout — verified before/after; confirmed at the end).
**Backend:** `SIM_BACKEND=cupy` (production backend; numpy DISQUALIFIED for striatal/near-threshold work per `2026-06-09-N9-cupy-membrane-divergence-ROOT.md`). Deterministic regime (OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF), hard-asserted.
**Owner directive:** biologize everything, no banking, brain-based-only. **An honest negative IS the scientific deliverable.**
**Builds on / closes the loop with:** `2026-06-09-N9-place-graded-critic-stage2-derisk.md` (Stage 2 NEGATIVE: the fire-vs-grade tension is irreducible through a single-hop WTA place layer — distinct⇒weak⇒MSN-silent OR strong-driver⇒WTA-collapse⇒position-blind), `2026-06-09-place-code-selforg-stage1-derisk.md` (Stage 1 PASS: single-hop place code IS position-specific; Option-A trisynaptic did NOT conduct EC-driven at probe scale), `2026-06-09-place-code-biologization-research.md` (the design; the C1 thesis).

---

## TL;DR — VERDICT: **NEGATIVE (0/3 seeds, all three operating regimes).** The trisynaptic loop reproduces the SAME fire-vs-grade tension at the CA3 recurrent stage; it does NOT decouple it through the available runner-side levers.

**The C1 thesis (the hypothesis under test):** the trisynaptic loop decouples the tension with TWO mechanisms instead of one WTA — **DG pattern-separation** makes locations distinct, **CA3 recurrent pattern-completion** amplifies a sparse DG cue into a HIGH-RATE attractor *while preserving that distinctness*. So **CA1 should be BOTH high-rate (enough to fire an MSN-D1) AND distinct-per-location.**

**The result:** the loop CONDUCTS end-to-end from landmark sensors (CA1 fires) **only** in the one regime where CA3 holds a *single GLOBAL attractor* — which is position-blind AND too weak to fire the MSN AND self-sustaining (not even sensor-driven). The DG separation works beautifully (diff-cos 0.066), but **CA3 completion destroys it** (diff-cos → 0.99). There is a razor-sharp bifurcation with **no high-rate-AND-distinct point in between**, robust across recurrent weight (4–35), recurrent density (0.02–0.30), feedback-inhibition weight, mossy weight (40–150), and ±NMDA.

| Gate (CuPy, 3 seeds 42/43/44) | `distinct` regime | `moderate` regime (CA1 fires) | `ignite` regime |
|---|---|---|---|
| **CONDUCTS** (CA1 non-silent end-to-end) | ❌ 0/3 (CA1 0.0 spk/step) | ✅ **3/3** (CA1 0.35–0.78 spk/step) | ❌ 0/3 (CA1 0.0) |
| **HIGH-RATE** (CA1→MSN ≥5 Hz; rheobase ~420 pA) | ❌ 0/3 (no CA1) | ❌ **0/3** (MSN drive **82–162 pA** ≪ 420; 0 Hz) | ❌ 0/3 |
| **DISTINCT** (diff-loc CA1 cos < 0.3) | (n/a, silent) | ❌ **0/3** (cos **0.83–0.88** — global attractor) | (n/a, silent) |
| **STABLE** (same-loc cos > 0.7) | (n/a, silent) | ✅ 3/3 (cos 0.81–0.87 — stable, but the SAME basin everywhere) | (n/a) |
| **SENSOR-DRIVEN** (ablate sensors → CA1 collapses) | (n/a) | ❌ **0/3** (cos(true,ablated) **0.77–0.85** — the attractor self-sustains WITHOUT sensors) | (n/a) |
| CA3 spk/step | 0.9 (sparse, distinct 0.09, but silent CA1) | ~10 (moderate global attractor) | 0 or **200 (saturated/runaway, seed-fragile)** |

**The load-bearing finding:** the C1 thesis's premise — *"completion preserves the separation"* — is **false on this substrate**. The CA3 recurrent autoassociator, built from the existing machinery, has only TWO stable states (a hard bifurcation, not a graded attractor): **(a) sparse + distinct + silent** (recurrent doesn't ignite → CA1 0 spk → no rate, can't fire the MSN), or **(b) a single GLOBAL attractor** (every distinct DG cue completes to the SAME CA3 state, diff-cos 0.99 → position-blind). The "rate from completion, distinctness from separation" decoupling does NOT happen, because the recurrent connectivity is *random/dense, not structured per-location* — whichever ensemble ignites first recruits the whole network through the dense random recurrent, collapsing all locations into one basin. **This is the exact same fire-vs-grade tension the single-hop WTA had (Stage 2), relocated one stage downstream to the CA3 recurrent.** C1 does not move the blocker; the remaining levers are **C2 (BTSP)** and **C4 (FS-WTA critic read-out)**.

---

## What was built (runner-side ONLY; `sim/` byte-empty)

### Probe — `research/runners/c1_trisynaptic_ca1_place_code_derisk.py` (CuPy-only, self-contained, 3-regime, gated)

Byte-mirrors `build_biological_brain_regions(enable_hippocampus_consolidation=True)` for the hippocampal regions/pathways (EC, DG, dg_pv_basket, CA3, CA1 + perforant/FFI/mossy/Schaffer/recurrent), with TWO faithful changes:
1. **The afferent is SWAPPED** from `language_input → ec` to **`landmark_sensors → ec`** (the legitimate egocentric BVC/OVC channel — the ONLY place (x,y) enters the brain; the Stage-1 render, ≥2 landmarks).
2. **Added a CA3 feedback-inhibition pool** `ca3_inh` (`ca3 → ca3_inh → ca3`, the de Almeida 2009 E%-max / basket-cell k-WTA loop the base build lacks) to *try* to keep the recurrent attractor sparse-and-distinct.
+ an **MSN-D1 test cell** (`IZH2007_STRIATAL_MSN_D1`, fully GABAergic, KIR2 down-state, `E_GABA=-60`, ~420 pA effective rheobase) reading CA1 through a strong sparse projection — the HIGH-RATE bar (does the CA1 ensemble's effective drive clear the rheobase the single-hop code failed?). Reports the MSN's **mean effective excitatory drive (pA)** via `g_e·(E_e − V)`.

**Sequencing:** self-organize (open the plastic feedforward + recurrent gates `landmark_to_ec/ec_to_dg/dg_to_ca3/ca3_to_ca1/ca3_swr_burst/ec_to_ca1`, walk 6 locations), then FREEZE and measure CA1 per location + the MSN drive + sensor-ablation. Gates CONDUCTS/HIGH-RATE/DISTINCT/STABLE/SENSOR-DRIVEN + regime-fidelity + position-leak (every external-current write targets `landmark_sensors` ONLY) hard-asserted.

**Three operating regimes** (`--regime`), the two stable states of the CA3 attractor + the firing-global midpoint:
- `distinct` — sparse CA3, recurrent doesn't ignite (CA1 silent; tests CONDUCTS+DISTINCT+STABLE).
- `moderate` — CA3 holds a moderate (~10 spk/step) GLOBAL attractor, CA1 DOES fire (the only CONDUCTS regime; tests HIGH-RATE on a firing CA1; expected to fail DISTINCT).
- `ignite` — recurrent + feedback inhibition → CA3 saturates (200 spk/step, runaway, seed-fragile) AND the synchronous volley shunts CA1 silent.

### Diagnostics (CuPy; `research/findings/raw/_c1_*.py`)
- `_c1_trisyn_conduction_diag.py` — stage-by-stage per-hop conduction (ec/dg/dg_pv_basket/ca3/ca1) instrument + diff-location cosine per hop; the forensic tool that mapped WHERE the loop goes silent/floods. Exposes every lever (perforant/FFI/mossy/recurrent/CA3-inhibition weights+densities) + `--selforg-passes`.
- `_c1_ec_rate_probe.py` — the sensor→EC rate bottleneck (the upstream EC fire-vs-select tension).
- `_c1_ca3_attractor_sweep.py` — the decisive `rec_w × inh_w` sweep proving the CA3 bifurcation (sparse-distinct-weak ↔ global-saturated) has no middle.

JSONs: `_c1_trisyn_ca1_{distinct,moderate,ignite}_3seed.json`.

---

## The decisive CuPy diagnostics (the obstacle, mapped hop-by-hop)

The C1 task named the obstacle as "the EC-driven conduction going silent" and asked to diagnose WHY (EC over-active collapsing DG sparsity? mossy/Schaffer too weak? CA3 recurrent not igniting?) and fix it by tuning. **All three sub-obstacles were found and individually solved by tuning — but solving them serially re-exposed the tension at the next stage, and finally at CA3 it became irreducible.**

### Obstacle 1 — EC fire-vs-select (the upstream tension)
The landmark render is a *sparse, weak, peaky* code: median sensor drive **2 pA**, only ~11–21 of 60 sensors fire. So:
- a **sparse** perforant projection (`lm_to_ec_density=0.05`) makes EC **selective** (diff-cos 0.18) but EC fires at only **af 0.11, 0.2 spk/step** → too weak to drive DG;
- a **dense** projection (`density=0.40`) makes EC fire strongly (af 1.0, 5–7 spk/step) but **non-selective** (diff-cos **0.92** — every location presents nearly the same EC firing).
- The middle (`density=0.10`, weight 60, intensity 900) is the workable compromise: EC selective enough (diff-cos 0.43) AND firing enough (af 0.5, ~1–2 spk/step) to drive DG. **This is the SAME fire-vs-grade tension at the EC input stage** (it took a careful density/weight balance to get past it).

### Obstacle 2 — DG silence under FFI (solved by tuning, separation WORKS)
With EC conducting, DG fires only if the EC→DG perforant (w≈20–30) beats the feedforward inhibition (`ec → dg_pv_basket → dg`). At the workable point DG is **sparse and beautifully separated**: **diff-cos 0.066** (input overlap 0.49 → DG 0.066, a ~43 pp orthogonalization) — D.12 pattern separation works exactly as advertised. (Solved.)

### Obstacle 3 — CA3 silence, then the irreducible CA3 bifurcation (the actual blocker)
Mossy `dg → ca3` (default w8/d0.10) cannot fire CA3 from DG's ~0.1–0.8 spk/step; a strong detonator (w40, d0.10–0.30) does. CA3 then either stays silent or, with the recurrent ON, **ignites — and HERE the thesis fails.** The `rec_w × inh_w` sweep (`_c1_ca3_attractor_sweep.py`, ≥12 configs each, robust across rec_density 0.02–0.30) shows a **razor-sharp bifurcation with NO middle**:

```
 rec_w inh_w |  ca3_spk(mean)  ca3_af  ca3_diffcos   verdict
   4–8  14    |       0.9       0.22      0.093       distinct-but-WEAK (CA1 silent)
   4–8  18    |     200.0       1.00      1.000       GLOBAL runaway  (CA1 silent)
   4–8  20–28 |     200.0       1.00      1.000       GLOBAL runaway  (CA1 silent)
```

- **Distinct regime** (low/no recurrent ignition): CA3 sparse (af 0.22) + **distinct (diff-cos 0.09)** but **0.9 spk/step** (threshold flicker, not an ensemble) → CA1 silent. Pushing the mossy harder (w 40→150) does NOT raise the rate (the feedback inhibition clamps it) — it only *worsens* distinctness (0.20→0.24).
- **Ignited regime:** CA3 → **200 spk/step (saturated runaway), af 1.00, diff-cos 1.000** — every location's distinct DG cue completes to the **identical** global CA3 state (confirmed cell-by-cell: the *same* 400 cells fire at 200 Hz regardless of location). AND the synchronous saturated volley **shunts CA1 silent** (CA1's own 15% GABAergic interneurons). Adding a Wang-style balanced NMDA attractor (recurrent w25–35 + tuned GABA-A, rec_density 0.05) gives the **same global collapse** (diff-cos 1.0).
- **The moderate midpoint** (recurrent ON, ca3_inh feedback OFF): CA3 holds a **moderate ~10 spk/step** attractor and **CA1 fires (0.35–0.78 spk/step)** — but it is a single GLOBAL basin (CA1 diff-cos 0.83–0.88) at a rate too low to fire the MSN (drive 82–162 pA ≪ 420 rheobase), and the attractor **self-sustains without the sensors** (ablation cos 0.77–0.85).

**Why completion destroys separation here:** the CA3 recurrent is a *random, dense, uniform-weight* matrix. Marr's autoassociator requires the recurrence to be *learned per-ensemble* (each location's mossy-driven CA3 ensemble strengthening ITS OWN closed recurrent loop, distinct ensembles non-overlapping). Starting the recurrent strong+uniform (w8, d0.30) ignites globally on the FIRST step — before STDP can carve basins — and then training just reinforces the one global attractor. Starting it near-zero (grow via STDP) keeps it sparse-and-distinct but it never reaches ignition (CA1 silent; more passes → fully silent). Additionally, the Izhikevich recurrent excitatory network has no conductance-based saturation to hold a *graded mid-rate* attractor — it is bistable between OFF and a 200-spk/step runaway, with no stable ~10–40 Hz CA3 state a real (conductance-based, adapting, E/I-balanced) CA3 would sit in.

---

## Anti-cheat ledger (all behave consistently with the negative)

- **(SENSOR-DRIVEN — the decisive control):** in the only firing regime (`moderate`), ablating landmark sensors leaves CA1 firing the global attractor (active 0.15 vs 0.13, cos(true,ablated) **0.77–0.85**) — the attractor is an **autonomous reverberation, NOT a sensor-driven place code.** This FAILS the anti-cheat: the firing CA1 isn't a real place code at all. ✅ consistent (the negative is honest, not masked).
- **(regime fidelity):** `backend=="cupy"` asserted; OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF asserted (hard-fail otherwise). No per-region homeostasis on CA3/CA1/MSN — they fire from synaptic current, not threshold collapse. ✅
- **(position-leak):** every `cp_external_input_current` write targets `landmark_sensors` ONLY; EC/DG/CA3/CA1/MSN NEVER receive a direct allocentric (x,y) injection. No host `vs_place_context` Gaussian. No direct-CA3 injection (unlike `validate_trisynaptic_loop`'s DIRECT-CA3 completion test — this probe is strictly EC/landmark-driven, the harder + faithful path). Enforced by construction + asserted. ✅
- **(HIGH-RATE bar is real):** the MSN-D1 test cell uses the actual `IZH2007_STRIATAL_MSN_D1` type + `E_GABA=-60` (the ~420 pA-effective-rheobase cell), read through a STRONG sparse projection (w30, d0.40 — its best shot). Its effective drive (82–162 pA) is reported directly. ✅

---

## Diagnosis — what the next mechanism MUST change (the value of this negative)

The negative is precise and actionable. The substrate **CAN** do DG pattern-separation (diff-cos 0.066, decisively) and the loop **CAN** conduct to a firing CA1 (moderate regime, 3/3). It **CANNOT** make a *single-hop random-recurrent CA3 autoassociator* hold a **high-rate AND distinct-per-location** attractor from a landmark-driven cue, because:

1. **The CA3 recurrent is random/dense, not structured per-ensemble.** A global attractor (one basin) is the only high-rate state it supports; the moment it ignites it collapses all locations together. Marr completion that *preserves* separation needs the recurrent to be *learned into distinct non-overlapping basins* — which the strong-uniform-init prevents (global ignition before STDP carves basins) and the grow-from-zero path can't reach (never ignites).
2. **The Izhikevich recurrent has no graded mid-rate attractor** — it is bistable OFF↔200-spk/step-runaway, so there is no stable ~10–40 Hz CA3 state (the rate a real conductance-based CA3 holds). The "moderate" 10-spk/step point exists only WITHOUT the ca3_inh feedback and is still a global basin.
3. **CA1's drive is too weak regardless:** even when CA3 holds a moderate attractor and CA1 fires, CA1 delivers only **82–162 pA** to the MSN — ~3× below the ~420 pA rheobase. CA1 conducting ≠ CA1 firing the striatal critic. (This is the Stage-2 wall again: a sparse spiking place ensemble at <1 spk/step can't clear an MSN-D1.)

**The levers that remain (per the C1 design's graceful-FAIL contract):**
- **(C2) BTSP single-trial plateau potentiation** (Bittner-Magee 2017): a plateau in the critic/place-cell potentiates inputs in one traversal — bypasses both the slow-STDP-from-sparse-firing bottleneck AND (if applied to carve the CA3 recurrent per-location) could in principle build distinct basins. **Requires a protected `sim/` kernel addition** (seconds-wide asymmetric eligibility + plateau-gated potentiation) — flagged for byte-review, NOT made here.
- **(C4) FS-WTA among critic sub-populations + a sparse convergent arm that ONLY the goal ensemble drives** — a sharper read-out that lets the sparse-distinct CA1 (the `distinct` regime, diff-cos 0.09) win the critic competition at low rate, *without* needing CA3 to supply a high rate the recurrent can't give distinctly. This keeps the place code in the distinct regime and pushes the burden onto the read-out, which is the lever the single-hop Stage-2 negative also pointed at.
- **(deeper, sim-level)** a conductance-based, adaptation-equipped CA3 with tuned E/I that holds a graded ~10–40 Hz attractor + a *structured* (learned-per-location, not random-dense) recurrent — the genuine Marr autoassociator. This is a real `sim/` dynamics change (byte-review), not a runner tune.

The cheapest faithful next move is **C4** (a sharper FS-WTA critic read-out on the *distinct-regime* sparse CA1), since C1 proved the place code CAN be distinct (just not simultaneously high-rate), so the unsolved half is purely the read-out — exactly where Stage 2 pointed.

---

## Honest bottom line + recommendation

The canonical hippocampal trisynaptic loop, driven from the legitimate egocentric landmark sensors, **does NOT decouple the fire-vs-grade tension** the single-hop WTA place layer hit. DG pattern-separation works (diff-cos 0.066), but CA3 recurrent pattern-completion has a razor-sharp bifurcation with no middle — **sparse+distinct+silent** (CA1 0 spk; no rate, can't fire the MSN) OR a **single global+firing+position-blind** attractor (CA1 diff-cos 0.83–0.88, drive 82–162 pA ≪ 420 rheobase, and self-sustaining without sensors). No point on the recurrent-weight / recurrent-density / feedback-inhibition / mossy / ±NMDA / Wang-balanced axes gives CA1 high-rate-AND-distinct, across 3 seeds and ~40 configurations. The tension the single-hop WTA couples is **reproduced at the CA3 recurrent stage**, because the recurrent is random/dense (one global basin, not learned-per-location) and the Izhikevich recurrent is bistable OFF↔runaway (no graded attractor).

**This is a clean, multiply-confirmed BRAIN-BASED-ONLY honest negative** — and it maps the real substrate boundary precisely: *a single-hop random-recurrent CA3 autoassociator cannot hold a high-rate-AND-distinct place attractor from a landmark cue.* **Recommendation: do NOT wire this into the nav critic; do NOT run a Stage-2 re-read on CA1.** The next faithful lever is **C4** (a sharper FS-WTA critic read-out on the distinct-regime sparse CA1 — C1 proved distinctness is achievable; the unsolved half is the read-out) or **C2 (BTSP)** / a structured-recurrent conductance-based CA3 (both protected `sim/` edits, byte-review). The probe + diagnostics ship as the re-runnable harness for whichever is tried next.

---

### Artifacts
- Probe: `research/runners/c1_trisynaptic_ca1_place_code_derisk.py` (CuPy-only; 3 regimes `distinct`/`moderate`/`ignite`; gates CONDUCTS/HIGH-RATE/DISTINCT/STABLE/SENSOR-DRIVEN + regime/position-leak; landmark→trisynaptic loop + ca3_inh + MSN-D1 test cell with effective-drive readout).
- CuPy de-risk JSONs: `research/findings/raw/_c1_trisyn_ca1_distinct_3seed.json`, `_c1_trisyn_ca1_moderate_3seed.json` (the CONDUCTS-but-position-blind primary), `_c1_trisyn_ca1_ignite_3seed.json`.
- CuPy diagnostics: `research/findings/raw/_c1_trisyn_conduction_diag.py` (per-hop conduction + CA3 feedback inhibition), `_c1_ec_rate_probe.py` (sensor→EC rate bottleneck), `_c1_ca3_attractor_sweep.py` (the decisive CA3 bifurcation sweep).
- `git status --short sim/` byte-empty (verified before/after every stage).
