# N9 — the SNc r−V subtraction (gate-2e): deep-research + diagnosis BEFORE building

**Date:** 2026-06-09
**Type:** READ-ONLY deep-research + diagnosis pass (no code changed, no GPU run). The institutionalized "deep research + catalog review FIRST" move that precedes building the last core N9 sub-arc.
**Scope:** the REMAINING N9 piece — the spiking SNc reward-prediction-error δ = r − V realized as a **GABA_B subtraction of the learned value V from the dopamine cell** (gate-2e in `n9_place_graded_critic_stage2_derisk.py`). The critic VALUE half (learns + grades V on the self-organized place code) is ALREADY validated this session (weighted-coincidence plateau, `2026-06-09-N9-weighted-coincidence-plateau-RESULT.md`); do NOT re-litigate it.
**Predecessors read in full:** the de-risk runner; `2026-06-08-gabab-girk-stageB-derisk-GO.md` (the GABA_B subtraction that PASSED 3/3 in the simpler Stage-B probe); `2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md` (the disinhibition circuit survey); `2026-06-09-N9-place-graded-critic-stage2-derisk.md` + `-place-grading-volley-RESULT.md` + `-weighted-coincidence-plateau-RESULT.md` (the value-half arc); `2026-06-09-N9-cupy-membrane-divergence-ROOT.md` (CuPy is authoritative). Catalog B.14/B.15/B.07/C.22/C.28/C.30; Kandel 6e Ch 43/38; Eshel-2015, Cohen-2012, Tepper-Lee-2007.

---

## TL;DR — the diagnosis, in one paragraph

Gate-2e is weak **NOT because the GABA_B is too weak** — it is the opposite. At the runner's parameters the GABA_B→GIRK conductance, when the critic fires, is **enormously strong** (the slow τ=150 ms conductance has a ~150× temporal-summation gain; even the runner's *reduced* `gabab_propagation_strength=0.02` delivers ~**−960 pA** onto the SNc at a 20 Hz critic, enough to fully silence a cell whose burst sits at ~100 Hz). The two observed failure modes are both consequences of that over-strong slow conductance interacting with **timing/calibration bugs in the runner**, plus the fact that the gate-2e protocol was written for the prior arc where the critic could not fire:
1. **Non-FS runs, "gap ≈ 0.98, SNc≈37 Hz at both":** ~37 Hz is the SNc at its *tonic* operating point (180 pA → 38.5 Hz on the IZH2007_DOPAMINE preset), NOT at the reward burst (480 pA → ~100 Hz). The cited numbers are the regime where the **critic does not fire during the `_snc_test`** (no volley / weak firing) → **zero differential GABA_B is delivered at EITHER near or far** → no subtraction → no gap. The subtraction has nothing to subtract because the critic is silent through the test window.
2. **FS-gating runs, "SNc tonic calibration → 0.0000":** the SNc reads 0 Hz at 180 pA in `_calibrate_da` **because of a residual-conductance bug** — `_calibrate_da` does **not** zero `cp_conductance_g_gabab` before measuring, and there is **no membrane/conductance reset between phases**. In the FS-gating config the critic fires hard (33–53 Hz, per the weighted-plateau finding), building a huge standing `g_gabab` that (τ=150 ms) is still hammering the SNc with hundreds of pA of hyperpolarizing current when calibration runs immediately after → SNc silent → threshold mis-set → the whole gate-2e collapses.

**The fix is therefore not "more GABA_B."** It is (a) make the gate-2e protocol drive the *graded, fireable* critic (the validated weighted-plateau arm) during the test so a differential GABA_B actually exists; (b) reset the SNc membrane + `g_gabab` between phases and re-calibrate; (c) **re-tune the operating point** to the Stage-B GO regime (the working Stage-B used `snc_reward_gain=120`, `snc_tonic_pa=220`, the **shipped default `gabab_propagation_strength=0.105`**, and **n_train≈18** — the Stage-2 runner instead uses rg=300, tonic=180, prop=0.02, n_train=40, all drifted from the working point and all documented to break the gap). The biology (Eshel-2015) says the subtraction is an **additive constant shift** carried by a **sustained** GABA expectation signal that **overlaps the reward**, which the runner's lead-then-burst protocol already mirrors structurally — the residual is calibration + protocol, not mechanism.

---

## 1. DIAGNOSIS — why gate-2e is weak (grounded in the runner + the parameters)

### 1.0 What gate-2e actually does (the protocol, verbatim from the runner)

`_snc_test(sensor_act, snc_pa)` (lines 757–779) runs, per location:
1. **ITI floor:** SNc at `snc_tonic_pa` (180 pA), no place drive, `hold_steps+20` = 60 steps.
2. **LEAD:** place drive at the location ON + SNc at tonic, for `lead_steps` (`lead_ms`/dt = 150 steps). *Intent:* the place ensemble fires → critic fires → GABA_B builds on the SNc BEFORE the reward.
3. **REWARD BURST:** place drive still ON + SNc at `snc_pa` (= `snc_tonic_pa + snc_reward_gain` = 180+300 = **480 pA**), `hold_steps` = 40 steps; count SNc spikes over this window → rate.

`pred_r = _snc_test(near, 480)` (predicted: V is high → critic fires → GABA_B subtracts → SNc should be LOWER). `unpred_r = _snc_test(far, 480)` (unpredicted: V≈0 → no GABA_B → SNc HIGH). `state_specific = (unpred_r > 1.30·pred_r) and (unpred_r ≥ 10)` (line 789). `--lesion` (`_lesion_gabab`, lines 456–466) zeros the mask AND `cp_conductance_g_gabab` → the gap must vanish.

This is a faithful **Rescorla-Wagner δ = r − V** protocol and it is structurally aligned with Eshel-2015 (§2): a sustained expectation signal during the lead that overlaps the reward burst.

### 1.1 The SNc dynamics — the operating point (IZH2007_DOPAMINE, dt=1.0)

Direct simulation of the preset (`C=100, k=0.9, vr=−65, vt=−45, vpeak=40, a=0.01, b=1.0, c_reset=−55, d=5`, dt=1.0):

| Input current (pA) | SNc rate (Hz) |
|---|---|
| 100 | 0.5 |
| 150 | 29.0 |
| **180 (tonic)** | **38.5** |
| 250 | 56.5 |
| 300 | 67.0 |
| **480 (tonic+reward 300)** | **~100** |
| 600 | 111.5 |
| 1000 | 166.5 |

Two things follow immediately:
- **The cited "predicted≈37.5, unpredicted≈36.67" are the SNc firing at ~TONIC (180 pA → 38.5 Hz), not at the reward burst (480 pA → ~100 Hz).** If the burst were registering and GABA_B were absent, BOTH locations would read ~100 Hz (gap ~1.0 at the saturated-burst level), not ~37 Hz. Reading ~37 Hz at both means the measured window is dominated by tonic-level firing — i.e. **the reward-burst response is not what's being differentiated, and the critic is not delivering any GABA_B at either location.** This is the **critic-silent** regime (the non-volley / weak-firing arm, which the gate-2e protocol predates — `2026-06-09-N9-place-graded-critic-stage2-derisk.md` explicitly logged gate-2e as "not applicable … the critic never fires → no GABA_B current onto SNc → no subtraction to measure").
- **The SNc reward response at 480 pA (~100 Hz) is NOT saturated** (headroom to 166 Hz at 1000 pA). So at rg=300 there *is* gap headroom — saturation is not the dominant blocker. (The Stage-B GO explicitly warned that rg=400 *does* saturate the 500 Hz ceiling; rg=300 is below that but the working point was rg=120 → 340 pA → ~75 Hz, even more headroom.)

### 1.2 The GABA_B conductance is OVER-strong, not weak — the decisive calculation

The per-step GABA_B increment (`sim/bridge.py:5815–5817`) is `g_gabab += (gb_mat.T @ prev_firing) · gabab_propagation_strength`, decaying by `exp(−dt/τ)` with τ=150 ms (`fused_gabab_decay_and_current`, `sim/kernels.py:218–226`), and the current is `I = g · (E_K − V)` with E_K=−90. At steady state `g_ss = increment / (1 − decay)`, and **`1/(1−decay) = 150.5`** — a slow conductance integrates 150 steps of input. With `strio_to_snc_weight=10`, density 0.5 → ~40 critic presyn cells per SNc neuron, and the driving force `(E_K − V) ≈ −40` at V≈−50:

| `gabab_propagation_strength` | critic rate | I_gabab onto SNc |
|---|---|---|
| **0.02 (runner)** | 10 Hz | **−482 pA** |
| 0.02 | 20 Hz | **−963 pA** |
| 0.02 | 30 Hz | −1445 pA |
| **0.105 (shipped default)** | 10 Hz | **−2528 pA** |
| 0.105 | 20 Hz | −5057 pA |

Recall (§1.1) that just **−400 pA silences the 480 pA burst to 0 Hz**. So **when the graded critic fires (the validated arm fires 33–53 Hz), the GABA_B is wildly over-strong — it would not "shift the burst down by a constant," it would clamp the SNc to 0 at the predicted location.** This is exactly what the Stage-B GO observed: predicted = **0.00 Hz**, unpredicted = 110.83 Hz (gap ∞) at seed 42. A clean **state-specific gap, but in the "all-or-none clamp" regime, not the Eshel "constant downward shift" regime.** The runner's reduction to `prop=0.02` (from the shipped 0.105) was an attempt to tame this, but 0.02 is still ~−960 pA at 20 Hz — far past clamp. **The conductance is not under-powered; it is operating in a saturating regime where a fired critic annihilates the SNc and a silent critic does nothing — there is no graded middle, and the place-code overlap that makes the critic fire a *little* at far (the value-half's residual) then leaks a clamp at far too.**

### 1.3 The FS-gating "SNc tonic = 0.0000" — a residual-conductance / no-reset bug

`_calibrate_da` (lines 440–453) drives ONLY the SNc at 180 pA and measures its firing fraction to set the DA-production threshold. By §1.1 that MUST read ~38.5 Hz **if the SNc starts clean**. It reads 0.0000 in the FS-gating config because:
- **`_calibrate_da` does NOT zero `cp_conductance_g_gabab` before measuring** (contrast `_lesion_gabab`, which explicitly does at line 464–465). There is **no membrane or conductance reset anywhere between phases** in `run_seed`.
- Calibration runs (line 625) **immediately after** the place self-org + the place-ensemble provenance probes, where — in the FS-gating + volley config — the critic fires **hard** (the weighted-plateau finding logged the critic at **33–53 Hz** in FS-gating). That builds a large standing `g_gabab` (by §1.2, critic@50 Hz → g_ss ≈ 60 nS → ~−2400 pA), which decays with τ=150 ms (halving only every ~100 ms). During the immediately-following 300-step calibration the SNc is still being hammered by hundreds of pA of residual GIRK current → it cannot fire at 180 pA → `tonic_frac = 0.0000` → the DA threshold is mis-set to 0 → every gate downstream that depends on calibrated DA is corrupted, and the SNc gap reads 0 because the SNc is suppressed at *both* locations by the standing conductance, not by a state-specific signal.

This is a **runner bug, fully explanatory of the FS-gating collapse**, and it is independent of the mechanism. (It also silently contaminates the non-FS runs to a lesser degree whenever the prior phase left any residual `g_gabab`.)

### 1.4 The reversal/override interaction is fine (ruled out as the cause)

The SNc's `syn_reversal_potential_i_override=−55.0` sets the **GABA_A** reversal (E_Cl) for the depolarized DA cell (catalog B.15: SNc lacks KCC2 → E_Cl≈−55). The **GABA_B** path is *separate* — it uses its own per-neuron `cp_gabab_reversal_per_neuron = −90` (E_K, set on GABA_B-post neurons at `sim/bridge.py:2381–2386`) and the `I = g·(E_K − V)` kernel. The two do not collide: `−55` governs the weak shunting GABA_A; `−90` governs the strong hyperpolarizing GIRK. This separation is the whole point of the GABA_B edit (it gives the KCC2-lacking SNc a genuinely hyperpolarizing, chloride-independent subtraction — Tepper-Lee-2007). **The override is correct and is not the failure.** (One subtlety, noted in the Stage-B GO §"GABA_A/GABA_B co-occurrence": a `receptor="gaba_b"` synapse ALSO drives the GABA_A `g_i` at E=−55, so the critic delivers a weak GABA_A + a strong GABA_B; the GABA_B dominates. This is a minor fidelity wart, not the gate-2e blocker.)

### 1.5 Is the critic firing during the lead window? (the engagement question)

For the subtraction to occur, the critic must fire during the LEAD + BURST windows so GABA_B is delivered while the SNc is being reward-driven (Eshel-2015: the expectation signal must **overlap** the reward). In the **non-volley** arm the critic is 0 Hz → no engagement → no subtraction (failure mode 1). In the **volley/weighted-plateau** arm the critic DOES fire at near (and, because of place-code overlap, a little at far) — so GABA_B *is* engaged, but then §1.2's saturation takes over: the fired critic clamps the SNc, and the same overlap that gives the value-half its residual (far fires a little) leaks a partial clamp at far. **So the gate-2e protocol has never been run against the fully-graded critic in a non-saturating GABA_B regime with clean calibration** — that combination is what the next de-risk must construct.

### 1.6 Summary of the diagnosis

| Failure mode | Root cause | Evidence |
|---|---|---|
| **(1) Non-FS gap ≈ 0.98, SNc ≈ 37 Hz at both** | Critic SILENT through the test window → no differential GABA_B; the ~37 Hz is tonic-level (180→38.5 Hz), the burst response isn't being differentiated | SNc preset sim (§1.1); the value-half arc shows the non-volley critic is 0 Hz |
| **(2) FS-gating SNc tonic = 0.0000** | `_calibrate_da` does not reset `cp_conductance_g_gabab`; no inter-phase membrane/conductance reset; FS-gating critic fires 33–53 Hz → huge standing GIRK current (τ=150 ms) still suppressing the SNc during calibration | runner lines 440–453 vs 464–465; weighted-plateau finding's 33–53 Hz; §1.2 conductance math |
| **(under both) no graded gap even when engaged** | GABA_B is OVER-strong at these params (150× summation; −960 pA at 20 Hz; −400 silences the burst) → all-or-none clamp, not Eshel constant shift; place-code overlap leaks a far-clamp | §1.2 calc; Stage-B GO's predicted=0.00 (clamp regime) |
| (drifted operating point) | runner uses rg=300/tonic=180/prop=0.02/n_train=40 vs the Stage-B GO rg=120/tonic=220/prop=0.105/n_train=18; n_train=40 is past the documented ≥30 saturation point | Stage-B GO §"Honest scope": "rg≈120 + 18 trials is the robust operating point; over-training (≥30) saturates" |

---

## 2. BIOLOGY — how the SNc r−V subtraction actually works

### 2.1 The computation: δ = r − V, and the subtraction is ARITHMETIC (additive), not divisive

**Schultz 1998 (J. Neurophysiol. 80:1; catalog C.22/C.28):** phasic DA encodes the TD error δ = r + γV(s′) − V(s); the three signatures (Fig. 2): burst on unpredicted reward (δ>0), no response to predicted reward (δ≈0 because r=P), dip on omission (δ<0). Eq. 6a (p.12): the effective reinforcement at reward time is `r̂(t) = r(t) − P(t−1)`. The cue-shift and omission-dip require a **learned prediction P** — i.e. a value-function critic (catalog C.30: "actor implemented, critic missing"). The Stage-2 runner's R-W protocol (V learned at near, subtracted at reward time) is the `r − V` half (the TD bootstrap cue-shift is a deeper later increment, out of scope).

**Eshel et al. 2015 (Nature 527:398) — the load-bearing mechanistic paper (NOT in the project catalog; sourced here).** Combining optogenetics + extracellular VTA recordings in mice doing classical conditioning, they show:
- **The subtraction is ADDITIVE/CONSTANT, not divisive.** "regardless of reward size, the odour cue simply shifted the dose-response curve by a constant amount" (Fig. 1d). An "output subtraction" model fit significantly better than divisive models, for both identified and putative DA populations. → **The expectation shifts the DA reward-response curve DOWNWARD by a constant**, the same number at every reward size. This is the canonical demonstration that DA cells compute δ = r − V by **subtraction**.
- **The source of the subtraction is VTA GABA neurons.** Selectively exciting neighboring VTA GABA neurons **suppressed** the DA reward response (P<0.001; 40 Hz opto added ~10 spikes/s → ~44% suppression, ≈ the ~53% that natural odour-expectation produced); inhibiting them **increased** the DA response to expected reward (DA "responds as if reward is less expected"). The GABA neurons are **local** and "synapse preferentially onto dendrites of dopaminergic neurons."
- **The expectation GABA is SUSTAINED and OVERLAPS the reward.** The VTA GABA neurons fire in a **sustained** manner across the entire cue→reward delay (Cohen 2012: persistent delay activity proportional to expected reward, NOT modulated by reward delivery/omission; 16/17 encode reward size, P<0.001), and that sustained inhibition is ON during the reward → it is the "burst-canceling expectation signal." Analysis window ~600 ms after reward onset.

**The arithmetic constraint this puts on the model (the key design lesson):** for the SNc to compute `r − V` (a *constant downward shift* of the reward response), the GABA inhibition must scale with **V** (the expectation), and the resulting SNc rate change must be **roughly proportional to the GABA input over the operating range** — i.e. the subtraction must live in a **near-linear / graded** regime, NOT a saturating clamp. The runner's GABA_B (§1.2) is in the clamp regime, which produces a *state-specific gap* (good for the gate's binary `state_specific` check) but is NOT the Eshel **arithmetic constant subtraction** — it is all-or-none. A faithful r−V wants the GABA_B operating point set so that a *graded V* produces a *graded* SNc suppression.

### 2.2 The anatomy: striosome → SNc, and why GABA_B/GIRK is the right channel

**Houk-Adams-Barto 1995 (the C.30 actor-critic mapping):** VTA/SNc DA = critic δ output; **striosome (patch) limbic striatum = the critic state-value V(s)**; matrix = actor; corticostriatal synapses on matrix = actor weights. The Stage-2 critic is exactly the **striosome_value** MSN-D1 cell (catalog B.07: "major input to SNc dopaminergic neurons arises from striatal patch/striosome compartment"; striosomes project to SNc DA cells — the classical striosomal projection).

**Why direct GABA_A onto the SNc is the WRONG tool, and GABA_B/GIRK is the RIGHT one (catalog B.15, B.07 supplemental; Tepper-Lee-2007):** SNc DA neurons **lack KCC2** → E_Cl ≈ −55 mV (near threshold) → GABA_A IPSPs are **only weakly hyperpolarizing / shunting** → DA cells are "remarkably resistant to direct striatal/pallidal GABA inhibition." Biology subtracts the value through **two routes**: (i) disynaptic disinhibition via KCC2-expressing SNr collaterals (the dominant phasic-DA route; but its sign is *wrong* for value subtraction — striosome↑ → SNr pause → DA burst↑); (ii) **local GABA → GABA_B → GIRK potassium channels, reversal E_K ≈ −90 mV** — a genuinely hyperpolarizing, chloride-independent conductance, which is **the exact mechanism Eshel-2015's local VTA GABA neurons use to subtract V from DA cells.** The project's GABA_B/GIRK `sim/` edit (shipped 2026-06-08, `enable_gabab`, `receptor="gaba_b"`, E_K=−90, τ=150 ms) implements precisely this. So **the project's striosome_value→snc GABA_B route IS the Eshel-2015 mechanism** — the issue is the operating point, not the route. (The Stage-B B′ research had recommended a *disinhibition workaround* because GABA_B didn't exist yet; now that it's shipped and PASSED 3/3 in Stage-B, the direct GABA_B route is the faithful and validated choice — supersedes B′-DISINHIBIT-EXC for this purpose.)

### 2.3 What biology needs that the runner must match

| Biological requirement (Eshel/Cohen/Schultz) | Runner status |
|---|---|
| V is a **graded** learned value (striosome) | ✅ validated this session (weighted-plateau: w_near/far 3–6×, NEAR≫FAR firing) |
| The subtraction is via **GABA_B/GIRK** (E_K=−90), not GABA_A onto the depolarized SNc | ✅ the route is `receptor="gaba_b"`, E_K=−90 |
| The GABA expectation is **sustained** and **overlaps the reward** | ✅ structurally: `_snc_test` LEAD (place ON, critic fires) → BURST (place still ON) |
| The subtraction is **arithmetic (constant downward shift)** — graded, near-linear, NOT a clamp | ❌ §1.2: GABA_B is in the saturating clamp regime (−960 pA at 20 Hz; −400 silences) |
| Clean DA baseline so δ=0 at no-RPE (calibrate threshold to tonic) | ❌ §1.3: calibration corrupted by residual `g_gabab` in FS-gating |
| The critic must FIRE during the test (else no GABA_B) | ❌ in the non-volley arm (failure mode 1); ✅ in the volley arm but then §1.2 clamps |

---

## 3. RANKED, biology-grounded options to make the subtraction state-specific + spiking

All reuse the **shipped GABA_B/GIRK route** (the Eshel-2015 mechanism) and the **already-validated weighted-coincidence-plateau graded critic**. Ranked cheapest-first.

### Option A — **Fix the protocol + calibration + operating point on the EXISTING GABA_B route (runner-only).** ★ RECOMMENDED

**Mechanism.** Keep the direct `striosome_value → snc` GABA_B subtraction. Make three runner-only corrections so the *fired, graded* critic drives a *graded* (not clamped) subtraction with a clean baseline:
1. **Run gate-2e against the validated graded critic** (`--enable-volley --weighted-drive --gate-fs-during-selforg`, the config that gives G_FIRE 3/3 + G_GRADE 3/3). The critic must fire NEAR≫FAR during the LEAD+BURST so a *differential* GABA_B exists. (Failure mode 1 is simply that gate-2e was being read in the silent-critic arm.)
2. **Reset `cp_conductance_g_gabab` (and the SNc membrane `cp_membrane_potential_v` to vr, and `cp_recovery_variable_u`) at every phase boundary** — before `_calibrate_da`, and inside `_snc_test` before the ITI floor. This kills the residual-GIRK calibration bug (§1.3) and makes each location's test independent. (Mirror `_lesion_gabab`'s existing `cp_conductance_g_gabab[:]=0.0`.)
3. **Move the operating point back into the Eshel "constant-shift" (graded), non-saturating regime**, and OUT of the clamp. The clamp is caused by the 150× summation × strong per-spike increment. Two knobs to bring it into the graded band:
   - **Lower `strio_to_snc_weight` and/or `gabab_propagation_strength` until the subtraction is GRADED** — target ~−100 to −250 pA at the NEAR critic rate (shifts ~100 Hz → ~50–65 Hz, a partial constant shift) rather than −960 pA (clamp to 0). Because the slow conductance is so high-gain, the right `prop` here is likely **≪ 0.02** (e.g. 0.002–0.005), or `strio_to_snc_weight` ~2–3. This is the *inverse* of "GABA_B too weak" — it is "GABA_B too strong, detune it."
   - **Calibrate the gap target to the Eshel ratio, not a clamp:** the gate's `state_specific = unpred > 1.30·pred` is satisfied by a constant shift (e.g. pred 55 Hz, unpred 100 Hz → 1.8×) — you do NOT need pred=0. Tuning for a *graded* gap is both more biologically faithful (Eshel constant subtraction) AND more robust (less seed-sensitive than the all-or-none clamp).
   - Optionally restore the Stage-B GO macro-params (`snc_reward_gain≈120`, `snc_tonic_pa≈220`, `n_train≈18`) which were validated to give the gap with headroom.

**Reuses:** GABA_B route (shipped); weighted-coincidence plateau (committed `e0818d2d`); FS-gating-during-self-org (runner infra); `_lesion_gabab` (anti-cheat). **Expected effect:** a **graded, state-specific** SNc gap (predicted ≪ unpredicted by a constant shift), surviving the `--lesion` (gap→1.0) and `--shuffle` (place→value decoupled → no value-of-location → gap→1.0) controls. **Cost:** RUNNER-ONLY, no `sim/` edit. CuPy. This is the smallest faithful experiment and directly tests whether the residual is purely operating-point.

### Option B — **A normal-reversal GABAergic relay (B′-DISINHIBIT-SNr-lite) to deliver a graded, non-saturating subtraction (runner-only).**

**Mechanism.** If Option A's direct GABA_B cannot be tuned out of the clamp (the 150× summation makes the graded band narrow/seed-fragile), insert the biology's actual subtraction topology: the critic does NOT hit the SNc directly; instead a **tonically-active GABAergic relay** (`snc_gaba_tonic`, normal E_Cl=−75) inhibits the SNc with a *graded, controllable* GABA_B, and the critic *sets the relay's rate* (via an intervening disinhibitory stage so more V → more relay GABA → less SNc — the odd-inhibitory-link chain from the B′ research §3). The relay's rate is bounded/tonic, so the GABA_B onto the SNc is graded-by-construction rather than the critic's bursty all-or-none output. This is the literal Houk-Adams-Barto / SNr-collateral disinhibition topology.

**Reuses:** GABA_B route; the GABAergic-relay recipe (gpe/gpi pattern, fully runner-side per the B′ research §4 table — "ZERO protected edits"). **Expected effect:** a graded gap that's less sensitive to the critic's burst amplitude (the relay tonic-rate is the controlled variable). **Cost:** RUNNER-ONLY (new relay BrainRegion + 2–3 pathways), no `sim/` edit. Heavier than A (more moving parts, its own calibration). Use only if A's graded band is too narrow.

### Option C — **A graded/divisive critic-side FS-WTA so the SNc input is bounded (runner-only), composed with A.**

**Mechanism.** The all-or-none problem is partly that the critic itself fires in unbounded bursts. Add a striatal PV-FSI pool on the critic (Tepper-2018, already the value-half's named next lever) implementing divisive normalization, so the critic's output rate (and thus the GABA_B drive) is **bounded and graded with V** rather than saturating. Then A's GABA_B operates on a tamer input. **Reuses:** FS-PING/FS-WTA infra already in the runner. **Expected effect:** smooths the clamp into a graded shift. **Cost:** RUNNER-ONLY. Secondary to A (A's GABA_B detune is the more direct lever).

### Option D (deferred, protected `sim/` edit) — **a saturating/ceiling on the GABA_B conductance (Destexhe-Sejnowski G-protein kinetics).**

**Mechanism.** The real GABA_B→GIRK has **cooperative, saturating** G-protein kinetics (Destexhe-Sejnowski) — `g` does not grow linearly without bound; it saturates. The current single-exponential `fused_gabab_decay_and_current` (no saturation) is *why* the 150× summation runs away. A saturating GABA_B kernel would make the subtraction *intrinsically graded* (V maps to a bounded conductance) — the principled biophysical fix the Stage-B GO flagged as a "ranked future refinement." **Reuses:** the GABA_B array/mask machinery. **Expected effect:** removes the clamp at the mechanism level → graded δ=r−V for free. **Cost:** a **protected `sim/` kernel edit** (byte-review), additive/default-off. Defer until A–C are exhausted; flag for the owner as the principled endpoint if the phenomenological detune (A) proves seed-fragile.

**Ranking:** **A ≻ C ≻ B ≻ D.** A is the smallest faithful experiment and tests the diagnosis directly (the residual is operating-point + calibration, not mechanism). C composes with A to widen the graded band. B is the relay fallback if direct GABA_B can't be detuned. D is the principled biophysical endpoint (protected edit) if the phenomenological route is fragile.

---

## 4. RECOMMENDED cheap-first de-risk

**The smallest experiment: Option A on the EXISTING `n9_place_graded_critic_stage2_derisk.py`, CuPy, ≥3 seeds (42/43/44), against the validated graded critic, with the calibration/reset fix + a GABA_B detune sweep.**

**Setup (runner-only):**
1. Run the **validated graded-critic config**: `--enable-volley --weighted-drive --gate-fs-during-selforg --n-place 800 --readout-weighted-k 26 --n-train 50` (the weighted-plateau GO point, G_GRADE 3/3).
2. **Add the resets** (the bug fix): zero `cp_conductance_g_gabab` and reset the SNc `cp_membrane_potential_v`→vr / `cp_recovery_variable_u`→0 (a) at the top of `_calibrate_da` and (b) at the top of `_snc_test` before the ITI floor. [This is the one new runner edit; everything else is flags.]
3. **Sweep the GABA_B strength DOWN into the graded band:** `--strio-to-snc-weight {2,3,5,10}` × `--gabab-propagation-strength {0.002, 0.005, 0.02}`. Target the operating point where predicted ≈ 0.5–0.65 × the no-GABA_B burst (a constant shift, ~50–65 Hz vs ~100 Hz), NOT pred=0 (clamp).

**The gate it must move (the deliverable):** a **state-specific SNc gap** at the *graded* operating point: `unpred_r > 1.30 · pred_r AND unpred_r ≥ 10` (the existing `state_specific`), with **pred_r > 0** (proving a graded constant shift, the Eshel signature, not an all-or-none clamp). Target ≥ 2/3 seeds (the value-half's robust bar), ideally 3/3.

**Anti-cheat controls (all must hold — these are the load-bearing falsifiers):**
- **`--lesion` (GABA_B-zero):** the gap MUST collapse to ~1.0 at every seed (the subtraction is carried by the GIRK conductance, not host arithmetic). `_lesion_gabab` already zeros both the mask and `cp_conductance_g_gabab`. **This is the decisive control** — if the gap survives lesion, the "subtraction" is an artifact.
- **`--shuffle` (place→location permute):** the gap MUST break (no value-of-location → no state-specific V → no differential GABA_B). The value-half already showed shuffle breaks the LTP; it must also break the gap.
- **Regime fidelity:** assert `backend=="cupy"` AND OU / conductance-noise / global-homeostasis / heterogeneity / STP OFF (the `_assert_cupy_regime` hard-fail, already in the runner). Deterministic (`CUBLAS_WORKSPACE_CONFIG`).
- **Provenance / no host r−V:** `current_reward_signal = 0.0` (already set, line 294); the SNc current is `tonic + reward burst − (synaptic GABA_B via the critic)` only; no host `V`/`reward_ema` reaches the SNc. Assert it.
- **Calibration sanity check (new, catches the §1.3 bug):** assert `tonic_frac ∈ [0.2, 0.6]` after `_calibrate_da` (the SNc MUST read ~38.5 Hz at 180 pA from a clean start) — a hard-fail here flags any residual-conductance contamination before the gates are trusted.

**Decision rule:**
- **If the gap is graded + state-specific ≥2/3 AND lesion+shuffle collapse it** → gate-2e is RESOLVED runner-side; the N9 r−V subtraction is validated on the production backend; proceed to the **6-seed nav A/B** (deploy `--spiking-snc --enable-neural-critic` with the critic→SNc routed GABA_B; acceptance = summed reward ≥ Stage A; an honest negative is still a valid deliverable mapping a neural-critic limit).
- **If the graded band is too narrow / seed-fragile** (the clamp keeps winning despite detune) → escalate to **Option C** (FS-WTA bounds the critic output) then **Option B** (relay), and only then **Option D** (the saturating-GABA_B protected kernel, byte-review).

**Wall-clock:** the value-half config is ~minutes/seed on CuPy (n_place=800, small critic/SNc); the sweep is a handful of runs. This is a same-session de-risk.

---

## 5. What NOT to do (shortcuts to avoid — BRAIN-BASED-ONLY)

- **Do NOT host-compute r − V** (read the critic rate in Python and inject `I = −k·rate` into the SNc). The subtraction MUST be **synaptic** (the GABA_B conductance the critic delivers). This is the B′ research's Option C ("value neural, subtraction host") — explicitly NOT the deliverable. The `--lesion` control exists precisely to catch this: a host subtraction would survive zeroing the GABA_B mask.
- **Do NOT read a host `reward_ema` / `_V_scaffold` as the prediction** (the Stage-A scaffold). The prediction MUST be the **spiking striosome critic's learned V**, delivered as spikes → GABA_B. Keep `current_reward_signal = 0.0`.
- **Do NOT "fix" the gap by cranking GABA_B UP.** The diagnosis is that it's already over-strong (clamp regime). Cranking it up deepens the clamp and makes it MORE all-or-none and seed-fragile — the opposite of the Eshel constant shift. The faithful move is to **detune into the graded band**.
- **Do NOT declare gate-2e PASS on the all-or-none clamp alone.** A binary "predicted=0, unpredicted=100" passes the `state_specific` boolean but is NOT the Eshel-2015 **arithmetic constant subtraction** — it's a winner-take-all silence. Report the gap AND whether pred_r is a graded constant shift (pred>0) vs a clamp (pred≈0); the graded version is the real r−V.
- **Do NOT add a host membrane reset that masks the calibration bug without fixing the root.** The resets in Option A are legitimate experiment hygiene (independent per-location tests, clean baseline), but the *finding* — that the slow GABA_B has a 150× summation and the runner never reset it — must be documented, because it also silently contaminates the non-FS runs.
- **Do NOT run the de-risk on numpy.** The MSN-D1/SNc near-threshold dynamics diverged on the numpy backend (an aliasing bug, since fixed, but the lesson stands): CuPy is the production + authoritative backend for striatal/near-threshold work (`2026-06-09-N9-cupy-membrane-divergence-ROOT.md`).
- **Do NOT switch the critic away from MSN-D1 striosome** to chase an easier fire (the N9 faithful design specifies the striosome MSN-D1 critic, Houk-Adams-Barto). The value-half is already validated on it; the gate-2e fix is downstream (the SNc subtraction), not the critic cell type.

---

## 6. Sources

### Project code (verified file:line this session)
- The de-risk runner: `research/runners/n9_place_graded_critic_stage2_derisk.py` — `_snc_test` (the lead-then-burst protocol) `:757–779`; gate-2e + `state_specific` `:785–789`; `_calibrate_da` (no g_gabab reset) `:440–453`; `_lesion_gabab` (zeros mask AND `cp_conductance_g_gabab`) `:456–466`; the snc region (`IZH2007_DOPAMINE`, `syn_reversal_potential_i_override=-55`) `:213–216`; the `striosome_value→snc` GABA_B pathway (`receptor="gaba_b"`, weight 10, density 0.5) `:236–239`; gabab config (`gabab_reversal_potential=-90`, `gabab_tau_decay=150`, `prop=0.02`) `:314–318`; calibration call `:625`; defaults (rg=300, tonic=180, n_train=40, lead_ms=150, prop=0.02) `:877–885`.
- GABA_B engine: kernel `fused_gabab_decay_and_current` (`I = g·(E_K−V)`, single-exp τ) `sim/kernels.py:218–226`; per-step GABA_B block (increment `= (gb_mat.T @ prev_firing)·prop`, decay τ=150) `sim/bridge.py:5789–5821`; per-neuron `E_gabab=−90` on GABA_B-post neurons `:2365–2386`; config defaults (`gabab_propagation_strength=0.105`) `sim/config.py:196–201`.
- SNc preset `IZH2007_DOPAMINE` (`C=100,k=0.9,vr=−65,vt=−45,vpeak=40,a=0.01,b=1.0,c_reset=−55,d=5`) `sim/enums.py:665–670`; MSN-D1 `IZH2007_STRIATAL_MSN_D1` `:671–676`.
- Signed-DA production rule `from_region_firing_signed` (the spiking-SNc dip half) `sim/neuromodulators.py:131–147, 774–817`.

### Project findings (the arc this builds on)
- `2026-06-08-gabab-girk-stageB-derisk-GO.md` — the GABA_B subtraction PASSED 3/3 in the simpler Stage-B probe; the **working operating point** (rg≈120, tonic=220, prop=0.105, n_train=18) + the documented saturation warnings (rg=400 saturates; n_train≥30 saturates).
- `2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md` — the disinhibition circuit survey (B′-DISINHIBIT-EXC/SNr), the KCC2/GABA_B biology, Eshel/Cohen/Tepper-Lee citations.
- `2026-06-09-N9-weighted-coincidence-plateau-RESULT.md` — the critic VALUE half VALIDATED (fires+learns+grades, G_GRADE 3/3 with FS-gating); explicitly flags gate-2e as the remaining piece ("SNc gap weak ~1.0; tonic calibration fragile in FS-gating").
- `2026-06-09-N9-place-grading-volley-RESULT.md` — gate-2e 0/3 because grading didn't open (critic NEAR≈FAR → no differential GABA_B).
- `2026-06-09-N9-place-graded-critic-stage2-derisk.md` — gate-2e "not applicable" (critic silent → no GABA_B).
- `2026-06-09-N9-cupy-membrane-divergence-ROOT.md` — CuPy is authoritative (numpy aliasing bug, since fixed); MSN-D1 rheobase ≈ 339 pA.

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **B.15** SNc lacks KCC2 → E_Cl≈−55 → direct GABA weak/shunting; disinhibition is the dominant phasic-DA route `:351–358`. **B.14** MSN E_GABA=−60 shunting `:342–349`. **B.07** striosome→SNc (the critic-value input to DA); SNc depolarized-GABA supplemental `:505–520`.
- **C.30** Actor-critic mapping (SNc=δ / striosome=V(s) / matrix=actor; Houk-Adams-Barto 1995; "critic missing"; acceptance = cue-shift + omission dip) `:592–599`. **C.28** δ=r+γV(s′)−V(s) (cue-shift needs a critic) `:574–581`. **C.22** Schultz RPE + HS98 cue-shift/omission criterion `:907–921`.

### Peer-reviewed literature (verified via search this session; Eshel-2015 is NOT in the project catalog — sourced here)
- **Eshel N. et al. (2015)** "Arithmetic and local circuitry underlying dopamine prediction errors", *Nature* 527:398. The subtraction is **ADDITIVE/constant** ("the odour cue simply shifted the dose-response curve by a constant amount", Fig. 1d; output-subtraction fits ≫ divisive); VTA GABA neurons are **the source** (opto-excite GABA → ~44% DA suppression; opto-inhibit → DA responds "as if reward less expected"); sustained, overlapping the reward. https://www.nature.com/articles/nature14855 ; PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC4567485/
- **Cohen J.Y. et al. (2012)** "Neuron-type-specific signals for reward and punishment in the VTA", *Nature* 482:85. VTA GABA neurons: **sustained delay activity proportional to expected reward**, NOT modulated by reward delivery/omission (16/17 encode reward size); local, synapse onto DA dendrites. https://pmc.ncbi.nlm.nih.gov/articles/PMC3271183/
- **Tepper J.M. & Lee C.R. (2007)** "GABAergic control of substantia nigra dopaminergic neurons", *Prog. Brain Res.* 160 (catalog PBR-160 ch 11). SNc GABA_A reversal −55 to −65 (no KCC2), shunting/weakly hyperpolarizing; ≥70% of SN DA afferents GABAergic; **GABA_B → GIRK K⁺ is the genuinely hyperpolarizing arm**; SNr→SNc disinhibition dominant. https://pubmed.ncbi.nlm.nih.gov/17499115/
- **Schultz W. (1998)** "Predictive reward signal of dopamine neurons", *J. Neurophysiol.* 80:1 (δ=r−P; cue-shift; omission dip). **Houk J.C., Adams J.L., Barto A.G. (1995)** (striosome=V(s); SNc=δ; matrix=actor). **Frémaux N. et al. (2013)** *PLoS CB* 9:e1003024 (spiking actor-critic; TD error modulates reward-STDP). **Kandel 6e** Ch 43 (dopamine/reward, Fig 43-2), Ch 38 (basal ganglia).

---

**Deliverable path:** `E:\Documents\Projects\sim\research\findings\2026-06-09-N9-SNc-rV-subtraction-research.md`
