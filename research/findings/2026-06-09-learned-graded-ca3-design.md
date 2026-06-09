# Design — a LEARNED, GRADED CA3 autoassociator (distinct per-location attractors at a stable ~10–40 Hz rate)

**Date:** 2026-06-09
**Type:** Deep-research + DESIGN pass. **READ-ONLY — zero `sim/` edits made.** This document is a *proposal* for the owner to byte-review before anything lands, per the project's protected-edit discipline (the GABA_B + per-region-NMDA + per-region-homeostasis + graded-lateral precedents).
**Owner directive:** biologize everything, no banking, brain-based-only. An honest negative IS the deliverable.
**Solves the blocker mapped by:** `2026-06-09-C1-trisynaptic-ca1-place-code.md` (CA3 completion destroys DG separation; the recurrent is random/dense → ONE global basin; the Izhikevich recurrent is bistable OFF↔runaway, no graded mid-rate state), `2026-06-09-N9-place-graded-critic-stage2-derisk.md` (the fire-vs-grade tension), `2026-06-09-place-code-selforg-stage1-derisk.md` (Stage-1 place code IS distinct, but cue-fragile = no completion).
**The lever the owner chose:** a learned/graded CA3 autoassociator — the most general + faithful fix (it makes the place code distinct-AND-high-rate without a host shortcut, which unblocks the N9 MSN-D1 critic).

---

## 0. TL;DR (the recommendation, up front)

The C1 negative has TWO coupled root causes; each maps to a **different** fix, and they must land **together**:

1. **Learned-per-location recurrent (Marr-Treves-Rolls).** *Mostly runner-side + ONE small `sim/` knob.* The `ca3 → ca3` pathway is ALREADY `plastic=True` (gate `ca3_swr_burst`), and the global STDP path already reaches it — so the recurrent CAN be Hebbian-learned today. What's missing is (a) the *protocol* (one-shot per-location storage with the recurrent gate timed so STDP carves distinct basins, not one global basin), and (b) a way to **clip the recurrent weight ceiling per-pathway** so a learned basin can't grow into the runaway regime. The STDP soft-bound `w_max` is global; the recurrent needs its OWN low ceiling. → **one additive per-pathway field** (`RegionPathway.stdp_w_max_override`) OR reuse the existing per-synapse plasticity-rate gate with a low global `w_max`. Cheapest: try runner-side first (it may already work with the right protocol); add the override only if needed.

2. **Graded attractor (the harder half) — NOT a new dynamics rule; a *routing* fix.** The project **already has a working Wang-2002 graded persistent attractor** — the dlPFC WM region (`--enable-pfc-nmda`, Cluster G v2.5): `BrainRegion.enable_nmda=True` + recurrent self-excitation + FS inhibition → stable graded persistence that survives dt=1.0 *at the genuinely-NMDA-dependent weight (~30), not the saturated AMPA-ping-pong weight (~50)*. The reason the C1 CA3 attractor still ran away **with `enable_nmda=True`** is a **routing defect, not a substrate limit**: the NMDA conductance increment is **GLOBAL** (`g_nmda_increase = g_e_increase * cfg.nmda_ratio`, `bridge.py:5524`) — it scales the **mossy detonator** the same as the recurrent. So the fast AMPA component of the *recurrent* drives a synchronous runaway *before* the slow NMDA can supply graded sustain, and the detonator gets NMDA-amplified too. The fix that makes the dlPFC attractor's mechanism work for CA3 is **per-pathway synaptic-receptor routing**: route the **recurrent** through a **slow NMDA-dominant** channel (graded, self-limiting via Mg block + adaptation), and keep the **mossy** AMPA (fast trigger). → **one additive per-pathway field** (`RegionPathway.exc_receptor ∈ {"ampa","nmda_slow"}`) + a guarded NMDA-route increment, mirroring EXACTLY the GABA_B `receptor=` per-pathway routing already shipped.

**Net `sim/` surface:** ~2 additive, default-OFF, byte-identical-when-off changes (one per-pathway receptor-routing field + its guarded conductance increment; one per-pathway STDP-ceiling override), ~120–180 lines incl. docstrings, mirroring the GABA_B/`receptor=` and per-region-NMDA precedents line-for-line. **Risk: MEDIUM** — the dlPFC existence-proof strongly de-risks "can Izhikevich hold a graded attractor at all" (YES), but whether the SPECIFIC CA3 geometry (sparse, structured, landmark-driven, distinct-preserving) sits in the graded band is the empirical question the de-risk gates answer. Honest fallback if NMDA-slow routing on point neurons still can't hold a *distinct-preserving* graded band: the genuine dynamics upgrade is an **AdEx/HH adapting CA3** (a real spike-frequency-adaptation current caps the rate) — a larger but well-scoped `sim/` change, flagged as the deeper fallback in §6.

**Build smallest-first:** Step A (runner-only) proves the learned-recurrent protocol on the EXISTING plastic gate; Step B adds the per-pathway NMDA-slow routing knob (the load-bearing graded-attractor fix); Step C adds the per-pathway STDP ceiling only if the learned basins overgrow. Each step has its own CuPy de-risk before the next.

---

## 1. Diagnosis — precisely why the current CA3 cannot hold a graded learned attractor

Mapped to the C1 numbers and the exact engine code.

### 1.1 Sub-problem A — the recurrent is random/dense, so it has ONE global basin (not learned-per-location)

- **C1 evidence:** DG separation works (diff-cos 0.066). But when the `ca3 → ca3` recurrent ignites, **every** distinct DG cue completes to the **identical** CA3 state (diff-cos → 0.99–1.00; cell-by-cell, the *same* 400 cells fire regardless of location). Robust across rec-weight 4–35, rec-density 0.02–0.30, ±NMDA, ±feedback-inhibition, mossy 40–150.
- **Code root cause:** the recurrent pathway is built by `RegionManager._build_pathway` (`regions.py:586`) as a **uniform-weight, random Erdős-Rényi** matrix (`weight_mean=ca3_recurrent_weight=1.5`, `density=0.30`, `text_minimal_isolation.py:1122-1131`). A uniform random recurrent with positive weight is a *single* Hopfield basin: whichever ensemble crosses threshold first recruits the whole network through the undifferentiated recurrence. Marr's autoassociator REQUIRES the recurrence to be **learned per-ensemble** (each location's CA3 ensemble strengthens ITS OWN closed recurrent loop; distinct ensembles' loops are non-overlapping). The project's TREVES-ROLLS reference: graded-rate attractor capacity is *only* realized when the recurrent stores *specific* sparse patterns via Hebbian LTP — never with a uniform random matrix.
- **Why STDP-from-zero didn't fix it (C1 "grow-from-zero never ignites"):** the pair-based soft-bound STDP (`fused_stdp_weight_update`, `kernels.py:285`) needs post-synaptic spikes to form eligibility; a near-zero recurrent never ignites → no CA3 spikes → no pairing → `w` frozen (the same LTP-bootstrap deadlock the N9 critic hit). And starting strong-uniform ignites globally on step 1, *before* STDP can differentiate basins, then training just deepens the one global basin.
- **The catalog confirms this is the known gap:** D.05 "*partial — `RegionPathway` CA3→CA3 would create the recurrent substrate, but no runner does this and no test verifies attractor convergence*"; D.13 "*missing — too much completion → confused episodes*" (the global-basin failure mode named exactly). O&N (D.05 supplemental): the recurrent is *learned via Hebbian LTP on co-active recurrents during exploration* — the storage protocol, not a random matrix.

**So sub-problem A is a PROTOCOL + INITIALIZATION problem on an already-plastic pathway, with one possible weight-ceiling knob — NOT a missing learning rule.**

### 1.2 Sub-problem B — the Izhikevich recurrent is bistable OFF↔runaway (no stable graded mid-rate)

- **C1 evidence:** the `rec_w × inh_w` sweep shows a **razor-sharp bifurcation, no middle**: `rec_w 4–8 / inh_w 14 → 0.9 spk/step` (sub-ignition, CA1 silent) vs `inh_w ≥18 → 200 spk/step` (saturated runaway, af 1.00). No stable ~10–40 Hz CA3 state. The one "moderate ~10 spk/step" point exists only with the feedback inhibition OFF and is a global basin.
- **Code root cause — the CA3 cell + the recurrent routing:**
  - `IZH2007_HIPPO_PYRAMIDAL` (`enums.py:659`): `C=100, k=0.7, vr=-65, vt=-40, vpeak=35, a=0.01, b=5.0, c=-55, d=50`. The adaptation is **slow and weak** (`a=0.01`); after a spike, `u` jumps by only `d=50` and decays back over ~100 ms. There is **no strong outward spike-frequency-adaptation current** to cap the population rate. So once recurrent excitation exceeds the (fast) feedback inhibition, every cell re-fires faster than `u` removes it → the population saturates at the refractory ceiling (≈200 spk/step). This is the textbook AMPA-recurrent instability the Wang/Compte literature warns about: *"the faster kinetics of AMPARs lead to dynamical instability and network collapse."*
  - **The NMDA that should fix this is mis-routed.** `bridge.py:5514-5533`: NMDA current is computed and masked per-NEURON (`cp_nmda_neuron_mask`), but the conductance **increment** is `g_nmda_increase = g_e_increase * cfg.nmda_ratio` (`:5524`) — a **single global ratio applied to ALL excitatory input** onto NMDA neurons. There is **no per-pathway NMDA routing** (confirmed: only `nmda_ratio` exists, grep'd `sim/`). Consequence for CA3: turning `enable_nmda=True` makes BOTH the recurrent AND the mossy detonator NMDA-scaled. The recurrent still has its full fast-AMPA component (it's `g_e`, not NMDA-only), so the synchronous AMPA volley still runs away; and the detonator is now slow-NMDA-amplified, making ignition *more* explosive, not gentler. **The C1 "±NMDA / Wang-balanced attempt → same global collapse (diff-cos 1.0)" is exactly this routing defect, not proof the substrate can't do it.**

- **The existence proof that the substrate CAN:** the dlPFC WM region (`g11_bg_runner.py:602-618`, `--enable-pfc-nmda`) is a `BrainRegion(enable_nmda=True)` recurrent (`internal_density>0`, `exc_weight_mean=2.0`) + FS inhibition, and the project documents it as a **working Wang-2002 graded persistent attractor** ("its NMDA-dependent WM latch survives dt=1.0 — de-risked at the genuinely NMDA-dependent attractor weight 30, not the saturated 50 = AMPA ping-pong", CLAUDE.md one-bridge-unification step3). **Same cell-class engine, same per-region-NMDA machinery — a graded NMDA attractor exists on this Izhikevich substrate.** The CA3 difference is (i) routing (the recurrent must be NMDA-dominant, the detonator AMPA) and (ii) the *distinctness* constraint (dlPFC holds one bump; CA3 must hold many non-overlapping ones — that's sub-problem A's structured recurrent + the E%-max sparsity).

- **Why de Almeida E%-max alone "wasn't enough" (C1 added `ca3_inh`):** E%-max (de Almeida-Idiart-Lisman 2009) is a **WTA SELECTION** rule — gamma feedback inhibition picks *which* cells fire (cells within E% of the max-excitation cell), keeping the code sparse. It does **not cap the per-cell RATE of the winners.** So it preserves distinctness/sparsity (good) but the selected winners still ride the recurrent to the runaway ceiling (bad). It addresses sub-problem A's sparsity, not sub-problem B's rate ceiling. The rate ceiling needs EITHER the NMDA-slow recurrent (self-limiting via the Mg-block: as V rises the drive saturates) OR a real adaptation current (the AdEx/HH fallback).

**So sub-problem B is a RECURRENT-ROUTING problem (per-pathway NMDA-slow vs AMPA), with an AdEx/HH-adaptation fallback if point-neuron NMDA still can't hold the distinct-preserving graded band.**

---

## 2. The mechanism design (grounded in the literature)

The fix is the **canonical CA3 autoassociator built the way Treves-Rolls + Wang specify**, realized on the existing engine with two additive routing knobs:

### 2.1 Learned-per-location recurrent (Marr 1971; Treves-Rolls; O'Keefe-Nadel pp.224-230)

- **Storage:** for each location, drive the place/DG cue → a sparse CA3 ensemble fires (selected by E%-max feedback inhibition → the existing `ca3_inh` / DG-PV-basket sparsity, 2–5%). With the recurrent plasticity gate (`ca3_swr_burst`) OPEN and the cue held, Hebbian/STDP LTP on the **co-active recurrent synapses within that ensemble** strengthens its closed loop. Distinct locations → distinct sparse ensembles → distinct non-overlapping recurrent loops. This is O&N's "Hebbian LTP on co-active recurrents during exploration."
- **Critical protocol details (the C1 lesson):**
  1. **Recurrent starts at/near ZERO** (`ca3_recurrent_weight≈0.0`, or a tiny prior). The basins are *grown by storage*, never present at init → no global ignition before differentiation.
  2. **Storage is cue-CLAMPED, not free-running.** During each location's storage window, the **mossy detonator drives the ensemble** (so the cells fire and STDP gets pairing events — defeating the bootstrap deadlock), while the recurrent gate is open. The recurrent learns to *reproduce* the mossy-driven ensemble. (This is the standard "teacher-forced" autoassociator storage; the *teacher* is the legitimate mossy afferent, NOT a host signal → brain-based-legal.)
  3. **Per-location, interleaved if possible**, with a **bounded recurrent ceiling** so no single basin overgrows into runaway (sub-problem B). The ceiling is the new `stdp_w_max_override` (Step C) OR a low global `stdp_w_max` if the recurrent is the only plastic pathway in the storage phase.
  4. **Recall:** present a partial/sensor cue → mossy lights a partial ensemble → the *learned* recurrent completes it to the stored ensemble (pattern completion, D.13) at a graded rate (sub-problem B), STILL distinct (because the recurrent loops are non-overlapping). Gate `ca3_swr_burst` can be lower at recall (completion needs less gain than storage).

This is **entirely runner-side on the existing plastic pathway** EXCEPT the optional per-pathway weight ceiling.

### 2.2 Graded attractor via NMDA-slow recurrent routing (Wang 2001/2002; Compte 2000)

- **Mechanism:** route the **`ca3 → ca3` recurrent** through a **slow NMDA-dominant** synaptic channel; keep **`dg → ca3` mossy** through fast AMPA. The Wang result: slow NR2B-NMDA recurrent reverberation gives a *stable graded* attractor because (a) the slow τ (~100 ms) low-pass-filters the recurrent feedback → no synchronous AMPA volley → no runaway; (b) the voltage-dependent Mg-block (`fused_nmda_update_and_current`, already in `kernels.py:229`) is *self-limiting* — as the ensemble depolarizes, the drive saturates rather than diverging; (c) FS feedback inhibition (the existing `ca3_inh`) sets the operating point. Result: a stable ~10–40 Hz graded band instead of OFF↔200.
- **Why per-pathway routing is the load-bearing change:** with today's GLOBAL `nmda_ratio`, you cannot make the recurrent NMDA-slow WITHOUT also making the mossy detonator NMDA-slow (which over-amplifies ignition). The mechanism *requires* the recurrent and the detonator to use different receptor kinetics. This is precisely the Wang/Compte architecture (recurrent = NMDA, feedforward/AMPA = trigger).
- **Engine realization:** mirror the **GABA_B `receptor=` per-pathway routing** that already ships (`regions.py:259-269`, `bridge.py:5535-5567`): a pathway tagged `exc_receptor="nmda_slow"` increments a *separate* slow-NMDA conductance (reuse `cp_conductance_g_nmda` / the existing NMDA kernel, OR a dedicated `cp_conductance_g_nmda_recurrent` to keep the slow τ independent) via a restricted matvec over only that pathway's synapses, while AMPA pathways feed `g_e` as today. The Mg-block + dual-exp kinetics are reused verbatim from `fused_nmda_update_and_current`.

### 2.3 Sparsity / distinctness preservation (de Almeida E%-max; DG-PV-basket)

- Keep the existing feedback-inhibition pool (`ca3_inh`, the C1 addition) as the E%-max selector so only the best-matched ~2–5% fire per location — this preserves distinctness during BOTH storage and recall (it's what keeps the learned basins non-overlapping). This is already runner-side; no `sim/` change. It is necessary but not sufficient (it selects, doesn't cap rate — §1.2); the NMDA-slow routing supplies the rate cap.

**Summary of the mechanism:** distinct sparse ensembles (DG separation + E%-max), each stored as its own learned recurrent loop (Hebbian LTP, cue-clamped, zero-init, bounded ceiling), reverberating at a graded rate via slow NMDA recurrence balanced by FS inhibition (Wang) — so CA1 reads a *completed, high-rate, distinct-per-location* ensemble that can fire the MSN-D1 critic. Every piece is a documented canonical CA3 feature; the only `sim/` additions are the two routing knobs that let the existing machinery be aimed correctly.

---

## 3. The exact byte-level `sim/` surface (proposal — owner byte-reviews)

All changes ADDITIVE, default-OFF, byte-identical-when-off, mirroring shipped precedents. **Separated into runner-side-only vs needs-`sim/`-edit.**

### 3.0 RUNNER-SIDE ONLY (no `sim/` edit) — Step A

- The learned-recurrent **storage protocol** (zero-init recurrent, cue-clamped mossy teacher, per-location interleaved storage with the `ca3_swr_burst` gate timed) is built ENTIRELY in a new probe runner (e.g. `research/runners/ca3_learned_graded_attractor_derisk.py`), reusing `build_biological_brain_regions(enable_hippocampus_consolidation=True)` + the existing gate helpers (`set_plasticity_gate("ca3_swr_burst", …)`). The `ca3 → ca3` pathway is already `plastic=True` and the global STDP path already updates it. **No `sim/` change for Step A.**
- If the recurrent overgrows into runaway during storage with only the global `stdp_w_max`, Step A can first try a **low global `cfg.stdp_w_max`** during the storage phase (the recurrent may be the dominant plastic pathway then) — still no `sim/` edit. Only if other pathways need a *different* ceiling simultaneously does Step C's per-pathway override become necessary.

### 3.1 `sim/` CHANGE 1 (load-bearing) — per-pathway excitatory receptor routing (NMDA-slow recurrent) — Step B

**Purpose:** let the recurrent be NMDA-slow while the detonator stays AMPA (§2.2). The graded-attractor fix.

**Mirror:** the shipped GABA_B `receptor=` routing — `regions.py:259-269` (the field) + `regions.py:654-655` (plumbed into the wiring-plan dict) + `bridge.py:5535-5567` (the guarded separate-conductance increment) + `config.py:142-145` (`enable_gabab` + propagation) + `bridge.py:233-241` (the conductance-array allocation). This change is the *excitatory mirror* of that pattern.

**(a) `sim/config.py`** — add (next to `nmda_ratio` line 134), default-OFF:
```python
# Per-pathway slow-NMDA-dominant recurrent routing (2026-06-09; Wang 2001/2002
# graded persistent attractor). When True AND a pathway sets exc_receptor=
# "nmda_slow", that pathway's excitatory increment feeds a SEPARATE slow-NMDA
# conductance (its own tau, Mg-block self-limiting) instead of the fast AMPA
# g_e -- so a CA3 recurrent can reverberate gradedly while the mossy detonator
# stays fast AMPA. Default False => the new increment block is unreached and
# total_input_current_pA is byte-identical to today (mirrors enable_gabab).
enable_nmda_recurrent: bool = False
nmda_recurrent_ratio: float = 1.0          # recurrent NMDA increment scale (AMPA component suppressed for nmda_slow pathways)
nmda_recurrent_tau_decay_ms: float = 100.0 # slow NR2B decay (Wang); >> AMPA ~5ms
nmda_recurrent_tau_rise_ms: float = 2.0
nmda_recurrent_propagation_strength: float = 0.105  # per-spike increment scale (mirrors gabab_propagation_strength)
```

**(b) `sim/regions.py`** — add to `RegionPathway` (next to `receptor` field, line 269), default byte-identical:
```python
# exc_receptor (2026-06-09): which excitatory receptor an EXCITATORY pathway uses.
#   "ampa" (default) -- fast ionotropic g_e (current behavior, byte-identical routing).
#   "nmda_slow" -- slow NR2B-NMDA-dominant: feeds a SEPARATE slow-NMDA conductance
#     (Mg-block self-limiting, tau ~100ms), AMPA component suppressed, so a recurrent
#     can hold a graded reverberatory attractor (Wang 2001/2002) without the fast-AMPA
#     synchronous runaway. Requires cfg.enable_nmda_recurrent=True; the pathway's
#     synapses are added to the per-synapse nmda-recurrent routing mask.
# See the GABA_B/receptor precedent (the inhibitory mirror of this).
exc_receptor: str = "ampa"
```
and plumb into `_build_pathway`'s returned dict (next to `"receptor": …`, line 655):
```python
"exc_receptor": getattr(pw, "exc_receptor", "ampa"),
```

**(c) `sim/bridge.py`** —
- **Allocate** (next to the GABA_B conductance arrays, ~line 239): `self.cp_conductance_g_nmda_recurrent = None`, `self.cp_conductance_g_nmda_recurrent_rise = None`, `self.cp_nmda_recurrent_synapse_mask = None` (bool per-synapse: True for `exc_receptor=="nmda_slow"` synapses; built from the wiring plan exactly like `cp_gabab_synapse_mask`). All None by default → the new block is unreached.
- **Build the mask** where `cp_gabab_synapse_mask` is built from pathway metadata (same wiring-ingest site), and **suppress the AMPA component** of nmda_slow synapses in the `g_e` matvec by zeroing their entries in the effective matrix used for `g_e_increase` (a masked copy, identical technique to the GABA_B restricted matvec).
- **The guarded increment + current** — a new block mirroring the GABA_B block (`bridge.py:5535-5567`), guarded `if getattr(cfg,"enable_nmda_recurrent",False) and self.cp_conductance_g_nmda_recurrent is not None:`. Restricted matvec over the nmda-recurrent-masked synapses → increment `g_nmda_recurrent` with `nmda_recurrent_propagation_strength`; decay + current via the EXISTING `fused_nmda_update_and_current` (reused verbatim) with the slow recurrent τ caches; add `I_nmda_recurrent` to `total_input_current_pA`. Byte-identical when the flag/array is off.

**Diff size:** ~90–130 lines incl. docstrings (most is the guarded increment block; the kernel is reused). **Byte-identity story:** with `enable_nmda_recurrent=False` (default) AND no pathway setting `exc_receptor="nmda_slow"`, every new array stays None, the new `if` blocks short-circuit, the `g_e` matvec is unmasked, and the Izhikevich/HH/AdEx paths are unreached-unchanged. Prove with: (i) `pytest tests/` green; (ii) a byte-diff harness running an existing g11 nav seed with/without the patched bridge → identical spike rasters (same technique the GABA_B/per-region-NMDA changes used); (iii) `git diff --stat sim/` shows only additive hunks.

### 3.2 `sim/` CHANGE 2 (optional, only if basins overgrow) — per-pathway STDP weight ceiling — Step C

**Purpose:** bound the learned recurrent so a basin can't grow into runaway, independent of the global `stdp_w_max` (sub-problem A point 3). Only needed if Step A can't keep the recurrent bounded with a global ceiling.

**(a) `sim/regions.py`** — add to `RegionPathway` (default None = use global), plumb into the wiring dict like the other per-pathway fields.
```python
stdp_w_max_override: Optional[float] = None  # per-pathway STDP soft-bound ceiling; None => cfg.stdp_w_max (byte-identical).
```
**(b) `sim/bridge.py`** — in the STDP application (`bridge.py:5949`, the `fused_stdp_weight_update` call), if any pathway sets an override, pass a per-synapse `w_max` array (built once from the wiring plan, defaulting to `cfg.stdp_w_max`) instead of the scalar. The kernel already takes `w_max` as a broadcastable arg (`kernels.py:285`), so this is a **scalar→array** swap guarded by `if self.cp_stdp_w_max_per_syn is not None`. **Diff size:** ~30–50 lines. **Byte-identity:** array is None unless a pathway sets the override → the scalar path is unchanged.

**Recommendation:** defer Change 2 — try a low global `stdp_w_max` in Step A's storage phase first.

---

## 4. De-risk + anti-cheats (the CuPy gates that confirm it)

A self-contained CuPy probe (`ca3_learned_graded_attractor_derisk.py`), 3 seeds (42/43/44), deterministic regime hard-asserted (OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF, `backend=="cupy"`), landmark→trisynaptic loop (the C1 harness, reused), storage→freeze→recall sequencing.

**Load-bearing gates (the C1 failure becomes a pass):**
| Gate | Pass criterion | What it proves |
|---|---|---|
| **G1 DISTINCT** | CA3 diff-location cos < 0.30 across ≥6 locations AFTER recurrent completion | beats C1's 0.99 global basin — the learned recurrent preserved separation |
| **G2 GRADED** | CA3 stable at ~10–40 Hz (NOT 0, NOT the ~200 spk/step ceiling), std-over-time bounded | beats C1's OFF↔runaway bifurcation — NMDA-slow recurrence holds a graded band |
| **G3 STABLE** | same-location repeat cos > 0.70 | a real attractor of the field |
| **G4 HIGH-RATE → fires MSN-D1** | CA1 effective drive to the `IZH2007_STRIATAL_MSN_D1` test cell ≥ ~420 pA → MSN ≥5 Hz | beats C1/N9's 82–162 pA ≪ 420 — the place code can drive the striatal critic |
| **G5 SENSOR-DRIVEN** | ablate landmark sensors → CA3/CA1 collapses (cos(true,ablated) low) | NOT an autonomous reverberation; a real sensor-driven place code (the C1 "moderate" regime FAILED this) |
| **G6 COMPLETION** | drop 1 of ≥3 landmarks → recall cos(true,partial) > 0.7 | D.13 pattern completion (Stage-1 Option-B's missing property) — the learned recurrent fills the partial cue |

**Anti-cheats (each must behave consistently):**
- **No host teacher.** The ONLY drive into CA3 during storage is the **mossy afferent** (the legitimate DG→CA3 detonator) — NOT a host-injected per-location pattern, NOT a `vs_place_context` Gaussian, NOT a direct allocentric (x,y) into CA3. Grep-assert every `cp_external_input_current` write targets `landmark_sensors` ONLY. (The cue-clamp teacher is the brain's own mossy pathway, which is brain-based-legal — same standing as DG driving CA3 in vivo.)
- **Genuinely learned, not hand-wired.** ABLATE the recurrent (zero `ca3 → ca3`) → completion (G6) and the graded sustain (G2) must COLLAPSE (CA3 falls back to the bare mossy-driven sparse code; CA1 silent). If G2/G6 survive recurrent ablation, the "attractor" was just feedforward drive → FAIL. Also: SHUFFLE the stored→location mapping → distinctness/completion must degrade to chance.
- **CuPy regime fidelity.** `backend=="cupy"` (numpy DISQUALIFIED per the divergence root-cause). No per-region homeostasis on CA3/CA1 (they must fire from synaptic current, not threshold collapse). Deterministic knobs OFF, hard-asserted.
- **The graded band is genuine, not threshold-flicker.** G2 requires a *population* at 10–40 Hz (≥ the C1 "0.9 spk/step threshold flicker" floor AND ≤ the saturation ceiling), measured over a sustained window, with the recurrent ON — and the SAME band at multiple distinct locations (distinctness + graded *simultaneously*, the exact pair C1 proved impossible with the random recurrent).

**Graceful-FAIL contract:** if G1+G2 can't BOTH pass (distinct AND graded simultaneously) on point-neuron NMDA-slow routing after the protocol+routing are correct, that is the honest negative that motivates the §6 fallback (AdEx/HH adapting CA3) — and it will be a *precise* negative (it will show WHICH of distinctness/rate the NMDA-slow recurrent couldn't hold).

---

## 5. Recommended build sequence (smallest-first protected edit + its de-risk)

1. **Step A — runner-only (NO `sim/` edit). The learned-recurrent protocol.**
   Build `ca3_learned_graded_attractor_derisk.py`: zero-init recurrent, cue-clamped mossy-teacher storage per location, `ca3_swr_burst` gate timed, E%-max `ca3_inh` sparsity, low global `stdp_w_max` for the storage phase. Run gates G1/G3/G5/G6 (the *learned-distinct* half). **This may already partially work** (the C1 negative never tried the zero-init cue-clamped storage protocol — it always started the recurrent strong-uniform or grew-from-zero free-running). De-risk decides whether sub-problem A is solved runner-side. **If G1 passes but G2 still bifurcates → proceed to Step B (expected).**

2. **Step B — `sim/` CHANGE 1 (per-pathway NMDA-slow routing). The graded-attractor fix.**
   Land §3.1 (owner byte-review). Tag the `ca3 → ca3` recurrent `exc_receptor="nmda_slow"`, set `enable_nmda_recurrent=True`, keep `dg → ca3` AMPA. Re-run ALL gates, especially **G2 GRADED + G4 HIGH-RATE**. This is the load-bearing step — the dlPFC existence-proof says the substrate can hold a graded NMDA attractor; this aims it at CA3. Byte-identity proof (§3.1) BEFORE behavioral runs.

3. **Step C — `sim/` CHANGE 2 (per-pathway STDP ceiling) — ONLY if Step A/B show recurrent overgrowth.** Land §3.2 if a single global ceiling can't simultaneously bound the recurrent and let other pathways learn.

4. **Then** re-run the **N9 Stage-2 MSN-D1 critic** (`n9_place_graded_critic_stage2_derisk.py`) on the new CA1 code — gate 2b (NEAR ≫ FAR ≥3×). This is the downstream payoff: a distinct-AND-high-rate place code is exactly the input the N9 critic needed.

Each step is independently gated; nothing proceeds to nav A/B until G1+G2+G4 pass.

---

## 6. Honest risk assessment + fallback

- **What's strongly de-risked:** "Can an Izhikevich point-neuron network hold a *graded* (~10–40 Hz, non-runaway) recurrent attractor?" — **YES, proven** by the shipped dlPFC WM region (Wang-2002 NMDA persistence at weight ~30). The graded attractor is not a new dynamics rule; it's a routing fix to aim the existing per-region-NMDA machinery at the recurrent.
- **The real residual risk (MEDIUM):** whether a graded NMDA-slow attractor can be *distinct-preserving* at CA3's geometry — i.e. hold MANY non-overlapping graded basins, not one. The dlPFC holds ONE bump; CA3 must hold a basin per location. The structured (learned, zero-init, sparse, E%-max-selected) recurrent is designed to make the basins non-overlapping, but on point neurons there's a real tension: enough recurrent gain to *complete* (G6) risks merging nearby basins (lower G1). The de-risk gates G1∧G2∧G6 measure exactly this; it may land in a workable band or it may show a narrower-than-wanted separation/completion trade-off (the D.13 "too much completion vs too little" knife-edge — which the catalog flags as the genuine biological tension, not an artifact).
- **Where it could fail on point neurons specifically:** Izhikevich point neurons lack a strong, tunable spike-frequency-adaptation current. The NMDA Mg-block provides *some* self-limiting, but if the graded band proves too narrow/seed-fragile to be useful, the honest cause is "point-neuron rate-capping is too weak."
- **The fallback (deeper `sim/`, well-scoped):** give CA3 a real adaptation current. Two options, both larger but bounded:
  (a) **Switch CA3 to AdEx** (`ADEX_RS`-derived with a tuned `a`/`tau_w` adaptation) — AdEx's `w` adaptation variable is a genuine spike-frequency-adaptation current that caps the population rate at a graded level (this is *the* canonical graded-attractor cell model in the Brunel/Compte tradition). The engine ALREADY HAS the AdEx kernel (`fused_adex_dynamics_update`, `kernels.py:184`) and per-region neuron-type override (`BrainRegion.adex_neuron_type`) — so a per-region AdEx CA3 is *also* mostly a config/routing change, not a new kernel. This is arguably a *cleaner* graded-attractor substrate than Izhikevich+NMDA and may be the better Step B if NMDA-slow proves marginal.
  (b) **HH CA3** (`HH_CA3_PYRAMIDAL_BURST` exists) with an M-current (`fused_hh_m_current_update` exists) — full biophysics, slowest, the last resort.
  Both are flagged as fallbacks, NOT first moves — Step B (NMDA-slow routing) is cheaper and rides a proven mechanism.
- **Net honest call:** the design has a proven existence-path for the graded half (dlPFC) and a proven plastic pathway for the learned half (the already-plastic recurrent); the genuine unknown is the *simultaneous* distinct+graded+completing band, which only the CuPy de-risk can settle. If it doesn't yield on Izhikevich+NMDA, the AdEx-CA3 fallback (cheap, kernel already present) is the principled next substrate — and either way the de-risk produces a precise, brain-based-only honest result.

---

### Engine references (for the byte-review)
- Per-region NMDA precedent: `sim/regions.py:100-112` (field), `sim/bridge.py:1152-1170` (mask build), `sim/bridge.py:5514-5533` (masked current).
- GABA_B per-pathway `receptor=` precedent (the exact mirror for §3.1): `sim/regions.py:259-269` + `:654-655`; `sim/config.py:142-145`; `sim/bridge.py:233-241` (alloc) + `:5535-5567` (guarded increment) + `fused_gabab_decay_and_current` `sim/kernels.py:217-226`.
- Graded-lateral precedent (template for a per-region opt-in dense-matrix pre-spike term): `sim/regions.py:176-188`; `sim/bridge.py:264-266`, `:1639-1747`, `:5481-5486`.
- NMDA kernel (reused verbatim): `sim/kernels.py:228-250`.
- CA3 cell + recurrent build: `sim/enums.py:659-664` (`IZH2007_HIPPO_PYRAMIDAL`); `research/runners/text_minimal_isolation.py:712-719` (CA3 region), `:1122-1131` (the already-`plastic` `ca3 → ca3` recurrent, gate `ca3_swr_burst`).
- STDP application (where §3.2's scalar→array ceiling goes): `sim/bridge.py:5908-6015`; kernel `sim/kernels.py:285-324`.
- The working graded-attractor existence proof: `research/runners/g11_bg_runner.py:602-618` (`--enable-pfc-nmda` dlPFC WM).
- `git status --short sim/` byte-empty (verified — this pass made ZERO `sim/` edits).
