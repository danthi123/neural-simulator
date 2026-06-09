# N9 robust value-learning diagnosis — WHY DA-gated place→value LTP is draw-fragile, and the biology-grounded fix

**Date:** 2026-06-09
**Type:** read-only diagnosis (deep-research + catalog/Kandel/literature). NO code edited.
**Backend assumed:** CuPy (RTX 3090), deterministic-nav regime (OU / conductance-noise / global-homeostasis OFF; cuBLAS/cusparse NOT pinned).
**Boundary being root-caused:** the N9 value critic (a striosome MSN-D1) learns V(location) via DA-gated STDP on `place → striosome_value` only **2/3** of place-code draws. Strong draws: `w_near` 0.5→~6.4, critic fires ~33 Hz. Weak draws (e.g. seed 44): `w_near` 0.5→~1.0, critic silent. This capped BOTH the value-grading arc (`2026-06-09-N9-weighted-coincidence-plateau-RESULT.md`, seed 43 LTP 1.86×) and the SNc r−V subtraction arc (`2026-06-09-N9-SNc-subtraction-derisk-RESULT.md`, seed 44 silent critic → no subtraction). The owner correctly rejects "2/3 with a documented hole."

Files quoted: `research/runners/n9_place_graded_critic_stage2_derisk.py` (the de-risk), `sim/bridge.py` (`_run_one_simulation_step`), `sim/kernels.py` (STDP + coincidence + eligibility-decay kernels), `sim/neuromodulators.py` (the DA modulator), `sim/enums.py` (the MSN-D1 preset). Biology: `sim-catalog/references/feature-catalog.md` (B.02, O.03, C.28/C.29/C.30), Kandel 6e, Yagishita-Kasai 2014, Frémaux-Gerstner 2016.

---

## 0. TL;DR

Two separate failures, **one root mechanism**: the runner's LEARN window pairs the **place volley** and the **DA reward burst SIMULTANEOUSLY** for only **40 steps (40 ms)**, on a cell that starts each trial cold in the MSN **down-state**. This violates the two biological prerequisites for corticostriatal LTP that the project's own machinery is built to honour:

1. **Puzzle 1 (init 58 Hz vs LEARN 0–39 Hz on the SAME near drive):** the MSN-D1 is a **bistable down-state cell** (`vr=−80`, `vt=−25`; catalog B.02: "silent at rest; requires substantial coordinated cortical/thalamic input to reach the up-state"). The isolated init read gives the cell **120 steps with a 30-step warm-up** to climb its slow coincidence plateau into the up-state. The LEARN window gives it **40 steps from a cold down-state start** (preceded by an ITI floor that drove only SNc → place silent → the critic's `cp_conductance_g_coincidence` plateau decayed and KIR2/`b=−20` pulled V back to −80). The SNc burst does **nothing** to depolarize the critic (DA targets `plasticity_rate`, not excitability — `sim/neuromodulators.py`), so the cell must reach the up-state from the place volley **alone, every trial, in 40 ms**. On a weak-drive draw it crosses threshold only intermittently → V(near) = 0,0,15,39…

2. **Puzzle 2 (slow/non-robust LTP even when it fires):** the DA burst is delivered **at the same instant** as the pre→post pairing, not **~0.3–2 s after** it. Yagishita-Kasai 2014 (and the project's O.03 = Schultz98 Eq. 9, which the eligibility trace is explicitly designed to implement) require DA to arrive **within a critical window AFTER the pre-before-post pairing**, gating a **silent eligibility trace** that the pairing laid down. The runner instead raises DA and lays down eligibility **on the same steps**, then converts eligibility→weight **on those same steps** while the trace is still tiny — so the high-DA window sees almost no accumulated eligibility, and by the time eligibility integrates, DA has decayed (200 ms tau). Worse, the eligibility is **signed STDP** (`Δw_LTP` from pre→post **minus** `Δw_LTD` from post→pre); with a 40 ms volley the LTP and LTD events nearly cancel, so net eligibility is small and seed-variable. **This is also exactly why the `--critic-teacher-pa` band-aid produces only `w_near`~1–2.8: continuous supra-threshold teacher firing is UNPHASED → LTP and LTD cancel.**

**The fix that makes the LEARNING robust (not a band-aid):** phase-separate the trial into a **pairing phase** (drive the place volley → critic into the up-state, lay down a positive eligibility trace, **DA at baseline**) followed by a **reward phase** (SNc burst → DA → convert the *already-accumulated* eligibility, **place still on so the cell stays up**), and give the cell enough steps to enter the up-state before scoring the pairing. This is the Yagishita protocol and the C.30 actor-critic the catalog says is missing. It reuses the existing eligibility-trace + neuromodulator + coincidence-plateau machinery with **runner-only** changes. The teacher, host-computed V, and "2/3 = pass" are all rejected below.

---

## 1. Diagnosis of Puzzle 1 — init 58 Hz vs LEARN 0–39 Hz on the same near drive

### 1.1 The two reads are NOT the same experiment

**Init read** (`_critic_rate_at_location`, runner L381-409):
```python
def _critic_rate_at_location(... n_steps=120, warmup=30, ...):
    if not jitter:
        _drive_landmarks(...)              # near sensors ON for the whole window
    for t in range(n_steps):               # 120 steps
        _tick(bridge)
        if t >= warmup:                    # score only steps 30..120 (90 steps)
            spk += int(bridge.cp_firing_states[crit_idx_gpu].sum()); ...
```
→ 120 steps of **uninterrupted** near drive, scoring after a **30-step warm-up**. The slow coincidence plateau (`cp_conductance_g_coincidence`, ~80 ms tail, `sim/kernels.py:fused_coincidence_plateau`) builds across those 30 steps; the MSN climbs into the up-state; the scored window sees a settled cell. This is why it reads ~58 Hz.

**LEARN window** (runner L693-716):
```python
for t in range(n_train):
    # ITI floor: SNc tonic, NO place drive
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_tonic_pa)
    _step(bridge, hold_steps)                                   # 40 steps, place SILENT
    if ... cp_eligibility_trace is not None:
        bridge.cp_eligibility_trace[:] = 0.0                    # reset
    # LEARN: near drive + reward burst (SIMULTANEOUS)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[sensor_idx_g] = ...near...  # place volley ON
    bridge.cp_external_input_current[snc_idx_g]   = snc_tonic + snc_reward_gain
    spk = 0
    for _ in range(hold_steps):            # only 40 steps, scored from step 0 (NO warm-up)
        _tick(bridge); spk += int(bridge.cp_firing_states[crit_idx_g].sum())
    near_v_curve.append(spk / ...)
```
→ the LEARN measurement is **40 steps, scored from step 0**, beginning from a **cold down-state** (the preceding 40-step ITI had place silent, so the plateau decayed and `cp_membrane_potential_v` of the critic relaxed toward `vr=−80`). The first ~10–20 of those 40 steps are the *missing warm-up*: the cell is climbing, not firing. On a strong-drive draw the volley re-ignites the plateau fast enough that the back half of the window fires (→ 39 Hz). On a weak-drive draw it doesn't reliably re-cross threshold in 40 ms → 0–15 Hz, intermittently.

### 1.2 Why the down-state start is decisive — the MSN-D1 preset

`sim/enums.py` `IZH2007_STRIATAL_MSN_D1`: `C=50, k=1.0, vr=−80, vt=−25, vpeak=40, a=0.01, b=−20, c_reset=−55, d_increment=150`.
- Resting potential **−80 mV**, spike threshold **−25 mV** → a **55 mV** gap the cell must be *driven across*. Catalog **B.02** (`feature-catalog.md:420-421`): *"bistable membrane (down-state ~−85 mV, up-state ~−55 mV)… Silent at rest; requires substantial coordinated cortical/thalamic input to reach up-state and fire."* L365: *"striatonigral neurons fire only during cortical Up states."*
- The cell is **stateful**: between the ITI (place off) and the LEARN onset, `cp_conductance_g_coincidence` decays (80 ms tail) and V relaxes to −80. **Nothing resets it back up** at LEARN onset — and crucially **the reward burst can't push it up**, because (next point) DA's only effect is on plasticity rate, not membrane drive.

### 1.3 The reward burst does NOT depolarize the critic — DA is plasticity-only

The DA modulator (runner L328-333; identical to `_default_dopamine_config`):
```python
NeuromodulatorConfig(name="dopamine", baseline=0.5, decay_tau_ms=200.0, ...,
    targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
    production_rules=[ProductionRule(rule_type="from_region_firing_signed", ...,
                                     source_regions=["snc"])])
```
`target_type="plasticity_rate"` → `compute_plasticity_rate_multiplier()` only multiplies `effective_reward_lr` in the three-factor block (`sim/bridge.py:6390-6392`). There is **no `excitability_drive` target** on the critic and **no SNc→critic synapse** in the pathway list (runner L218-240: the only critic afferents are `place→striosome_value` and `striosome_value→snc`). So during LEARN the critic's firing is driven **entirely by the place volley** — the +300 pA reward burst lands on `snc`, not on the critic. The cell has to reach its up-state from the place code alone, in 40 ms, from cold, on every trial. **This is the mechanistic root of the init-vs-LEARN discrepancy.**

### 1.4 Down-state-onset = the "KIR2 down-state at LEARN-window onset" the prompt asked about

Yes — the critic sits in the down-state at LEARN onset (V≈−80, plateau≈0). The Izhikevich `b=−20` is the project's KIR2 approximation (enums L618: *"the negative b=−20 approximates KIR2's contribution"*; catalog B.02 supplemental L433: KIR2 clamps RMP to −80…−95 mV). It is not a *bug* — it is the correct down-state physics — but the runner's protocol fights it: a real corticostriatal up-state is *entered and held* for hundreds of ms (Wilson & Kawaguchi; catalog L432 "Up-state… excitation-driven"), then plasticity is read out. The runner gives 40 ms with no hold.

**Summary of Puzzle 1 root cause:** the LEARN-window firing measurement is a **40-step, no-warm-up, cold-down-state-start** read on a bistable cell whose only depolarizing input is the place volley; the isolated init read is a **120-step, 30-warm-up, settled** read. The discrepancy is a **protocol artifact** (measurement window + missing up-state hold), not a property of the synapses.

---

## 2. Diagnosis of Puzzle 2 — slow / non-robust LTP even when the critic DOES fire

### 2.1 The two weight paths and the three-factor conversion

`sim/bridge.py` runs STDP and three-factor reward-modulation as two stages every step:

**Stage 4b — STDP (L6162-6267).** For synapses whose pre OR post fired this step, `fused_stdp_weight_update` (kernels L313-352, soft-bound) is applied and **written to the weight directly** (L6247), gated only by `cp_plasticity_rate_gain` (L6242-6245) — **NOT by DA**. The **signed** delta is also accumulated into eligibility (L6265-6267):
```python
if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
    weight_changes = updated_weights - current_weights      # SIGNED (LTP +, LTD −)
    self.cp_eligibility_trace[stdp_active_indices] += weight_changes
```

**Stage 4c — three-factor (L6310-6449).** Eligibility decays (`fused_eligibility_trace_decay`, tau = `reward_eligibility_tau_ms` = **1000 ms**, config L271), then converts to weight via DA:
```python
da_signal = float(da_conc) - float(da_baseline)             # DA deviation from baseline
effective_signal = da_signal                                # DA path takes precedence
effective_reward_lr = cfg.reward_learning_rate * compute_plasticity_rate_multiplier()
weight_updates = effective_reward_lr * effective_signal * cp_eligibility_trace[:nnz]   # L6424
... weight_updates *= cp_plasticity_rate_gain ...
self.cp_connections.data += weight_updates                  # L6449
```
So the **DA-gated** growth = `lr × (DA−baseline) × eligibility`, applied **every step**. Note DA enters **twice**: once as `effective_signal` (the δ), once inside `effective_reward_lr` via `compute_plasticity_rate_multiplier()` (both ≈ `1+(conc−baseline)`). Net: weight growth is roughly **quadratic in the DA deviation** and **linear in the eligibility integral**.

### 2.2 Failure mode A — DA and the pairing are simultaneous; the high-DA window sees a tiny eligibility

Yagishita-Kasai 2014 (Science 345:1616; confirmed via literature search) and Shindou-Kasai 2019 ("a silent eligibility trace"): the pre-before-post pairing lays down a **silent eligibility trace**; **dopamine applied within a critical window AFTER the pairing** (the trace decays over ~1 s) converts it to LTP. DA arriving *during* or *before* the pairing does **not** potentiate. The project's O.03 supplemental (`feature-catalog.md:4783`) states the eligibility machinery is the **intended** implementation of Schultz98 Eq. 9, `Δw = η·r̂·h(i,o)` where `h(i,o)` is *"an eligibility trace of conjoint pre/post activity that outlasts the events themselves."*

The runner does the opposite of the protocol:
- It drives the place volley (laying down eligibility) **at the same time** as the SNc reward burst (raising DA), for the same 40 steps.
- Stage 4c converts **every step**. On the early LEARN steps DA is already high (the SNc fires immediately on the +300 pA burst, and the `from_region_firing_signed` EMA + 200 ms DA decay keep it up), but the **eligibility integral is still near zero** (it has only had a few steps of STDP, and the cell is still climbing into the up-state — §1). So `lr × (DA−base) × eligibility ≈ lr × (large) × (≈0) ≈ 0` early.
- By the time eligibility has integrated (needs the cell firing for several steps), the 40-step window is ending and the next ITI zeroes eligibility (`cp_eligibility_trace[:]=0`) **before** the next high-DA window. There is no DA pulse *after* the pairing within the window — DA and pairing co-terminate.

Result: each trial deposits only a sliver of `w_near` growth → `w_near` crawls 0.5→1.0 over 40 trials on a weak draw (and reaches ~6 only on strong draws where the cell fires hard enough early that eligibility *and* DA briefly overlap). **The growth is small and seed-variable precisely because the eligibility-to-DA phase relationship is wrong.**

### 2.3 Failure mode B — signed eligibility from an unphased volley nearly cancels

Eligibility accumulates the **signed** STDP delta (§2.1). With a sparse FS-PING volley, place and critic both fire near-synchronously each gamma cycle, so for a given `place_i → critic` synapse the per-step `delta_t = t_post − t_pre` is sometimes **positive** (place slightly leads → `ltp_update`, kernels L335) and sometimes **negative** (critic leads → `ltd_update`, L343). With `stdp_a_plus=0.012` only marginally above `stdp_a_minus=0.01` and equal taus (20 ms), an **unphased** pre/post relationship gives **net eligibility ≈ 0**. The strong draws win because the place volley reliably *leads* the critic (the critic only fires *after* enough place input accumulates → consistent pre-before-post → net-positive eligibility). The weak draws have the critic firing sporadically and late, so the pre/post order is noisy → eligibility sign is seed-variable.

This is the **same reason the `--critic-teacher-pa` band-aid barely works** (runner L704-711; result doc: teacher → critic fires 15-39 Hz but `w_near` only ~1-2.8): a **continuous supra-threshold teacher current makes the critic fire on its own schedule, UNPHASED with the place volley** → for each `place→critic` synapse, LTP (place-leads) and LTD (critic-leads) events are ~balanced → net eligibility ≈ 0 → almost no LTP. The teacher fixes *firing* (Puzzle 1) but not *phasing* (Puzzle 2), so it cannot produce robust LTP. **The teacher is a band-aid; say so.**

### 2.4 Why DA-gating is "multiplying an eligibility that has already decayed" — confirmed

The prompt's hypothesis is correct in spirit but the precise failure is the **co-termination**, not decay-within-trial: with tau=1000 ms and a 40 ms window, intra-window decay is negligible (`exp(−40/1000)=0.96`). The decay that *does* kill it is the **`cp_eligibility_trace[:]=0` at the next ITI** (L698-699), which wipes the trace before the *next* reward could act on it, combined with **no post-pairing DA pulse inside the current window**. So the eligibility is never "aged into" a high-DA window the way Yagishita requires; it is born and converted and erased all within 40 simultaneous ms.

**Summary of Puzzle 2 root cause:** (a) DA is delivered *simultaneously with* the pairing instead of *after* it, so the high-DA × high-eligibility product never forms; (b) the signed eligibility from a 40 ms unphased volley nearly cancels on weak draws. The LTP is slow because each trial deposits a sliver; non-robust because both the firing (§1) and the pre/post phasing (§2.3) are draw-dependent.

---

## 3. The biology — how robust corticostriatal value-learning actually works

1. **The three-factor rule with a SILENT eligibility trace + a critical post-pairing DA window (Yagishita-Kasai 2014, Science 345:1616-1620; Shindou-Kasai 2019, EJN).** The pre→post pairing lays down a **silent** eligibility trace (CaMKII / Ca²⁺-based) that decays over **~1 s**; **dopamine arriving within ~0.3–2 s AFTER the pairing** converts it to LTP. DA *before/during* the pairing does not potentiate. The order is **pairing → (silent trace) → DA**, never DA-during-pairing. (Frémaux-Gerstner 2016, *Front. Neural Circuits* 9:85; Gerstner-Lehmann-Liakoni-Corneil-Brea 2018 review the neoHebbian three-factor formalism: `dw/dt = η · eligibility(t) · M(t)` with the neuromodulator `M` multiplying a *pre-existing* eligibility.)

2. **The MSN must be depolarized into the UP-STATE for the Ca²⁺ that LTP needs (Wilson & Kawaguchi 1996; catalog B.02 + supplemental, `feature-catalog.md:420-433`; Kandel 6e Ch 38 pp 933-938, 947-950).** Corticostriatal LTP is gated by NMDA-dependent Ca²⁺ influx, which requires the spine to be depolarized (Mg²⁺ unblock) — i.e. the cell in the up-state, **held** for the pairing, not flickering across threshold. The up-state is **entered by coordinated cortical/thalamic excitation and held**; plasticity rides on top of that sustained depolarization. A 40 ms cold-start window is too short to establish the up-state Ca²⁺ signal reliably.

3. **D1 LTP requires a DA BURST; the sign/direction is DA-state-dependent (Surmeier 2009; Schultz16-NRN; catalog O.03 supplemental L4784).** A phasic DA burst *after* coincident pre+post → D1 LTP. This is what the runner *intends* (SNc burst → DA up), but the **timing relative to the pairing** is what makes it work or not.

4. **The system the catalog says is MISSING is a critic that bootstraps a value estimate (C.28/C.30, `feature-catalog.md:574-608`).** *"the project does not bootstrap a value estimate… closer to a windowed Monte Carlo in which the window is the eligibility-trace decay length."* The N9 critic is exactly the attempt to add C.30. The robust way to train V is the **actor-critic / TD form** (Sutton & Barto Ch 6, 7, 11): a stable place afferent + a value readout whose weights move by `δ × eligibility` where δ is the *post-pairing* reward signal. The biology and the algorithm agree: **separate the eligibility-laying (pairing) from the value signal (DA), with DA following.**

**Net biological prescription:** (i) drive the place volley to bring the critic into the up-state and **hold it** long enough to lay a clean, net-positive (pre-before-post) eligibility trace, with **DA at baseline**; (ii) **then** deliver the SNc reward burst so DA rises and converts the *already-laid* trace; (iii) keep the place drive on during the reward so the cell stays up. This is Yagishita's protocol and a TD/actor-critic value update. It removes both the firing fragility (the cell is given time to enter the up-state) and the cancellation (a phased, held pairing gives consistent pre-before-post LTP).

---

## 4. Ranked, biology-grounded options (favouring robust LEARNING over band-aids)

### Option 1 (RECOMMENDED) — Phase-separate the trial into PAIR → REWARD (the Yagishita / actor-critic protocol). **Runner-only.**
**Mechanism.** Restructure the LEARN trial (runner L693-716) into three phases:
- **(a) Up-state induction + pairing (place ON, DA at baseline, ~80–120 steps).** Drive near sensors → place volley → critic; give it a **warm-up** (e.g. 30 steps unscored) so the coincidence plateau lifts the cell into the up-state and **holds** it; STDP lays a net-positive (place-leads-critic) eligibility trace. **Do NOT fire the SNc reward burst yet** (SNc at tonic only → DA ≈ baseline → Stage-4c conversion ≈ 0, so the trace accumulates *silently*, exactly Yagishita's silent trace). Because the cell is held up for 80–120 ms, the pre-before-post ordering is consistent → eligibility is net-positive and **draw-robust** (the volley always leads the integrating critic when given time).
- **(b) Reward / DA conversion (place STILL ON, SNc burst ON, ~40 steps).** Now fire the SNc reward burst → DA rises *after* the pairing → Stage-4c converts the accumulated eligibility (`lr × (DA−base) × eligibility`, now both factors large simultaneously) → `w_near` grows in a big, robust step. Keep place on so the cell stays in the up-state during conversion (the DA acts on a still-depolarized spine, as in vivo).
- **(c) ITI (place OFF, SNc tonic, reset eligibility).** As now.

**Reuses:** the existing eligibility trace (`cp_eligibility_trace`, tau 1000 ms already matches the ~1 s biological trace), the neuromodulator DA subsystem (`from_region_firing_signed` already produces the post-pairing DA), the coincidence plateau (it does the up-state induction), and `set_plasticity_gate` (unchanged). **No `sim/` edit** — it is purely a re-ordering + longer pairing window in `run_seed`'s training loop, plus exposing the warm-up/pair-length as args.
**Expected effect:** removes BOTH failure modes — (1) the cell is given time to enter the up-state before the pairing is scored (fixes Puzzle 1's intermittent firing), and (2) DA arrives *after* a clean, held, net-positive eligibility (fixes Puzzle 2's tiny-product + cancellation). `w_near` should grow robustly to a fireable value on weak draws too, because the up-state-hold makes the pairing draw-independent. Anti-cheat-clean (place-shuffle still breaks it; no host value).
**Cost:** runner-only; ~1 hr to implement + the existing 3-seed CuPy de-risk to verify.

### Option 2 — Pin CuPy determinism so the place-code DRAW is reproducible. **Runner-only, COMPOSE with Option 1.**
**Mechanism.** Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the CuPy import + CuPy deterministic algorithms (the g11 `--deterministic` pattern, already referenced in the SNc-subtraction result doc as "lever 1"). This makes STEP-1's self-organized place code reproducible across invocations, so a seed maps to ONE place-code draw (currently the same seed draws 0.120 or 0.138 between runs — `2026-06-09-N9-SNc-subtraction-derisk-RESULT.md` §blocker).
**Reuses:** existing regime knobs. **No `sim/` edit.**
**Expected effect:** does NOT by itself fix the learning fragility (a weak draw is still weak), but it makes the 3-seed result **reproducible and honest** — without it, "3/3" could be luck-of-the-draw per invocation. **Necessary for a credible multi-seed claim; pair with Option 1.**
**Cost:** trivial (env var + one CuPy call); ~10 min.

### Option 3 — Make eligibility net-positive by construction: a brief pre-lead on the place drive within the pairing. **Runner-only.**
**Mechanism.** Within Option-1's pairing phase, ensure the place volley **leads** the critic by starting the place drive a few ms before the critic can fire (it already does, via the FS-PING volley re-timing), and optionally widen the LTP/LTD asymmetry for this arm by **lowering `stdp_a_minus`** relative to `stdp_a_plus` *for the training phase only* (config is global, but the runner already toggles config fields per-phase, e.g. `coincidence_weighted_drive`, so it can set `cfg.stdp_a_minus` low during PAIR and restore it). Biology: corticostriatal STDP is LTP-biased under DA (Shen-Surmeier 2008); a modest pre-before-post LTP dominance is realistic.
**Reuses:** STDP kernel as-is; per-phase config toggling pattern already in the runner.
**Expected effect:** guarantees net-positive eligibility even on draws where the pre/post timing is noisy → removes §2.3 cancellation. Secondary to Option 1 (Option 1's up-state-hold already does most of this), but a cheap robustness booster.
**Cost:** runner-only; minutes. Anti-cheat note: keep `stdp_a_plus/minus` in a realistic band (don't zero LTD — that would let any coincidence grow w unboundedly and defeat the place-shuffle control's specificity).

### Option 4 — Sub-threshold PHASE-LOCKED teacher (replaces the unphased supra-threshold teacher). **Runner-only.**
**Mechanism.** If a teacher is wanted at all, make it **sub-threshold and phase-locked to the place volley** (a small depolarizing pulse delivered on the gamma cycle *just after* the place volley, never on its own), so it nudges the critic over threshold **only when the place volley has already fired** → preserves the pre-before-post order. The current teacher (`--critic-teacher-pa`, continuous supra-threshold) is unphased and gives ~0 net LTP (§2.3).
**Reuses:** the existing teacher current injection site (runner L704-711), retimed.
**Expected effect:** turns the band-aid into a biologically-defensible scaffold (the innate-reflex-teaches pattern, but *phased*). Use only if Option 1's up-state-hold proves insufficient on the weakest draws; **prefer Option 1**.
**Cost:** runner-only. The result doc itself flags the unphased teacher as "weaker, w_near ~2.8" and "a sub-threshold phase-locked teacher would be cleaner + more biological" — this is that.

### Option 5 (LAST RESORT, protected `sim/` edit) — explicit MSN up/down bistability kernel (KIR2 + Kv2 voltage-dependent leak).
**Mechanism.** The catalog (B.02 supplemental L433) notes Izhikevich `b=−20` only approximates KIR2; a faithful KIR2+Kv2 kernel would give the cell a true latched up-state (input resistance peaking ~6× at −60 mV), so once driven up it **stays up** with less drive — making the up-state hold cheap and draw-independent at the membrane level.
**Reuses:** the fused-kernel pattern (`sim/kernels.py`), guarded + default-off + byte-identity-proven (the project's standing bar).
**Expected effect:** the most biologically complete fix, but heavy. **Only if Options 1–3 leave a residual.** The owner wants byte-level diff review for any `sim/` edit; this is a real new kernel, not additive routing — high cost.
**Cost:** weeks; protected `sim/` edit + byte-review. Defer.

**Recommended stack:** **Option 1 + Option 2** (phase-separate the trial **and** pin determinism), with **Option 3** as a cheap robustness booster if a weak draw still misses. This makes the LEARNING robust and the result reproducible, entirely runner-only, no `sim/` edit.

---

## 5. Recommended cheap-first de-risk (the smallest runner-only experiment)

**Experiment.** Add a `--pair-then-reward` mode to `n9_place_graded_critic_stage2_derisk.py` (runner-only) that restructures the LEARN trial as Option 1: PAIR phase (place ON ~100 steps incl. 30-step warm-up, **SNc tonic only / DA baseline**, eligibility accumulating silently) → REWARD phase (place STILL ON, SNc burst ON, ~40 steps, eligibility→weight conversion) → ITI (reset). Run on **CuPy with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before import** (Option 2). Keep everything else (FS-gating-during-selforg, the weighted-coincidence read-out, θ) at the validated value-grading config.

**Exact metric (must hold to call it robust):**
1. **The weak-drive draw learns:** on **seed 44** (the documented weak draw), `w_near` reaches a **fireable** value — operationally `w_near_final ≥ 4.0` (the strong-draw seeds reached ~6; ≥4 is comfortably above the ~1.0 silent-critic floor) **AND** `crit_near ≥ 5 Hz` (gate 2a) **AND** `w_near ≥ 2× w_far` (gate 2c).
2. **Multi-seed 3/3:** seeds 42/43/44 all pass gates 2a + 2b (≥3×) + 2c (≥2×) — i.e. PRIMARY **3/3**, not 2/3. Report `w_near_final` per seed; all three must clear the fireable threshold.
3. Report the eligibility trajectory: `w_near` should now grow in **large per-trial steps** during the REWARD phase (confirming DA-after-pairing conversion), not the current 0.5→1.0 crawl.

**Anti-cheat controls (decisive):**
- **(a) Permuted place→reward mapping (`--shuffle`) MUST break the learned V.** Permuting which place cells the value arm tracks must drop gate-2c LTP below 2× (the value rides on weights learned at the *rewarded* location, not on "fired-on-any-drive"). This is the existing control; it must still hold under the new protocol.
- **(b) No host-computed V anywhere.** The critic's firing during read-out comes only from the LEARNED `place→striosome_value` synapses + the (spiking) coincidence plateau; the value signal δ is the **SNc firing** via the neuromodulator (`from_region_firing_signed`), never a Python scalar (`current_reward_signal=0.0`, runner L294, is preserved). Confirm `_critic_rate_at_location` reads `cp_firing_states`, not a host value (it does).
- **(c) CuPy-deterministic regime asserted.** `_assert_cupy_regime` already enforces backend==cupy + OU/cond-noise/homeostasis OFF; **add** the `CUBLAS_WORKSPACE_CONFIG` pin and assert the place-code draw is reproducible across two invocations of the same seed (the diff-cos must be byte-identical run-to-run, closing the non-determinism gap the SNc-subtraction doc flagged).
- **(d) DA-timing falsification (proves it's the timing, not just "more steps"):** run a control arm that delivers the SNc burst **simultaneously** with the pairing (the current protocol) but with the SAME total step budget as `--pair-then-reward`. If the simultaneous arm stays fragile (2/3) while the pair-then-reward arm is 3/3 at equal step budget, that isolates the **DA-after-pairing timing** as the fix — not merely the longer window. This is the decisive scientific control.

---

## 6. What NOT to do

- **Do NOT ship the `--critic-teacher-pa` band-aid as the fix.** A continuous supra-threshold teacher fires the critic UNPHASED with the place volley → LTP (place-leads) and LTD (critic-leads) eligibility events ~cancel → near-zero net LTP (the result doc's own `w_near`~1-2.8 confirms it). It fixes firing (Puzzle 1) but not phasing (Puzzle 2). If a teacher is used at all, it must be **sub-threshold and phase-locked** (Option 4), and even then Option 1's up-state-hold is preferred.
- **Do NOT host-compute V** (a Python `V(location)` table, a distance formula, an argmax-over-place-cells readout, or injecting a host-computed value current into the critic). That violates the BRAIN-BASED-ONLY standard — the *brain* (the MSN-D1's learned synapses + spiking) must compute V. The value must be read out of `cp_firing_states`, the δ out of SNc firing.
- **Do NOT declare 2/3 a pass.** A value system that learns V on only 2/3 of place-code draws is not a working value system (owner directive). The gate is **3/3 with a reproducible (determinism-pinned) place code**, including the weak-drive seed 44.
- **Do NOT "fix" it by raising `stdp_w_max` / `reward_learning_rate` / `place_to_value_weight` to brute-force `w_near` up.** That makes the critic fire on the rate-coded AMPA alone (a rate leak), defeating the coincidence/jitter property the arc validated, and inflating `w_far` alongside `w_near` (the place-overlap trade-off the value-grading doc already hit). The fix is the **protocol** (DA-after-pairing + up-state hold), not bigger numbers.
- **Do NOT reach for the KIR2 `sim/` kernel (Option 5) first.** It's the heaviest, protected-edit path; Options 1–3 are runner-only and should be exhausted (and almost certainly suffice) before touching `sim/`.

---

## Appendix — the load-bearing code/biology citations

| Claim | Source |
|---|---|
| LEARN window = 40 steps, scored from step 0, cold down-state start; init = 120 steps w/ 30 warm-up | runner L381-409 (`_critic_rate_at_location`), L693-716 (train loop) |
| MSN-D1 `vr=−80, vt=−25, b=−20`; silent at rest, needs coordinated drive to reach up-state | `sim/enums.py:671-676`; catalog B.02 `feature-catalog.md:420-426`, supplemental L433 |
| DA targets `plasticity_rate` only (not excitability); no SNc→critic synapse | runner L328-333, L218-240; `sim/neuromodulators.py:323-342, 546-560` |
| STDP writes weight directly (DA-independent) + accumulates SIGNED eligibility | `sim/bridge.py:6203-6267`; `sim/kernels.py:313-352` |
| Three-factor conversion `w += lr × (DA−base) × eligibility`, every step; eligibility tau 1000 ms | `sim/bridge.py:6310-6449`; `config.py:271`; `sim/kernels.py:354-365` |
| coincidence plateau is stateful (80 ms tail), NOT reset between ITI/LEARN | `sim/bridge.py:5741-5787` (`cp_conductance_g_coincidence`); runner `_reset_snc_subtraction_state` resets only GABA_B |
| eligibility wiped each ITI (`cp_eligibility_trace[:]=0`) before next reward | runner L698-699 |
| Silent eligibility trace + critical post-pairing DA window (~0.3–2 s; trace ~1 s) | Yagishita-Kasai 2014 Science 345:1616; Shindou-Kasai 2019 EJN; Frémaux-Gerstner 2016 |
| Project's eligibility machinery = intended impl of Schultz98 Eq. 9 three-factor rule | catalog O.03 supplemental `feature-catalog.md:4783-4784` |
| MSN up-state (Ca²⁺ for LTP) entered + HELD by coordinated excitation | catalog B.02 supplemental L432-433; Kandel 6e Ch 38 pp 933-938, 947-950 |
| The missing piece is a value-bootstrapping critic (C.30); current ≈ windowed Monte Carlo | catalog C.28/C.30 `feature-catalog.md:574-608` |
| Place-code draw non-deterministic between invocations (cuBLAS/cusparse unpinned) | `2026-06-09-N9-SNc-subtraction-derisk-RESULT.md` §blocker |

**Sources (literature):**
- [A critical time window for dopamine actions on the structural plasticity of dendritic spines — Yagishita et al. 2014 (Science) — via Shindou/Kasai 2019 EJN](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.13921)
- [Reinforcement determines the timing dependence of corticostriatal synaptic plasticity in vivo — Nat. Commun. 2017](https://www.nature.com/articles/s41467-017-00394-x)
- [Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules — Gerstner et al. 2018 (Front. Neural Circuits)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/)
- [Retroactive modulation of spike timing-dependent plasticity by dopamine — Brzosko et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626806/)
