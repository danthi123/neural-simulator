# Shortcut #9 + B4 — the delayed-reward, value-IS-load-bearing task scoping: design the genuine close + the #9↔B4 unification verdict (2026-06-21)

**Status:** READ-ONLY deep-research + design-first scoping (the standing "deep research + catalog review FIRST at a confirmed boundary / new direction" move, CLAUDE.md). NO `sim/` edits, NO build, NO GPU — single deliverable = this doc.
**Date:** 2026-06-21.
**Author role:** read-only research subagent. Every load-bearing project claim below was re-verified against the repo (the four #9 dendrite findings, the B4 scoping, the two limbic-load-bearing findings, the deploy raw JSONs **re-run** this pass) and the catalog/literature. The deploy "qualified NEGATIVE" was confirmed by **running** the deploy verdict aggregator on the raw per-seed JSONs (the numbers below are measured, not trusted from a summary).
**This is a scoping/decision doc, NOT a brain-based result and NOT a commitment to build.**

---

## 0. The one-paragraph answer (the rest is the evidence)

Shortcut #9's dendrite-graded VALUE read-out is a genuine GO in isolation (on-bridge graded V is clean/monotone/location-selective, 3/3 seeds, ~9× near/far, `sim/` edit byte-reviewed) — but its DEPLOY into the production nav critic is a **qualified NEGATIVE for the reason the task was designed to expose**: the moving-goal gridworld is **immediate-reward-solvable** (the SC/orienting/place machinery reaches a per-step-rewarded goal without any *predictive* value), so the value critic's actual function — **credit assignment over a temporal GAP** — is never exercised, the lesion (silence the value cell) barely moves navigation, and the SNc saturates (flat 50 Hz). The deploy aggregator I re-ran confirms it numerically: **dendcritic sum 8.47 ≈ value-lesion sum 9.08 (Δ7.2%)** — the value is NOT load-bearing — while the *whole* improvement over the point-neuron baseline (8.47 vs 20.9) comes from the **NMDA on the critic slice**, not from the value shaping behaviour (the `ctrl_nmda` arm with NO dendrite value also lands at 8.72). So the genuine #9 close is **a task where the value is PROVABLY load-bearing**, and the canonical biological paradigm for exactly that is **trace conditioning** (catalog F.22/F.23; Hesslow-Yeo 2002; the recent NAc-DA-encodes-the-trace-period result, eNeuro 2025): a reward separated from its predictive cue by a CS-free GAP, where the *only* way to predict/act correctly is to carry a learned value across the gap. The decisive design move — and the answer to the deploy's confound — is the catalog's own **2×2 factorial (TRACE vs DELAY × value-ON vs value-LESION)**: the **TRACE arm needs the value (lesion collapses it)**, while the **DELAY arm (no gap, US during CS) does NOT need the value** (an immediate-reward control that, by construction, the substrate solves *without* the critic) — which directly discriminates "the task exercises the critic" from "the task is immediate-reward-solvable," the exact confound that sank the nav deploy. On the **#9↔B4 unification:** they are the **same broad family** ("value / credit-assignment over a temporal gap") and a single delayed-reward harness can host both, BUT they are **distinct sub-problems with distinct bottlenecks**, and the project's own work has already split them: **B4 (the TD cue-shift) is a multi-seed POINT-NEURON GO** on the standalone probe (r = −0.80/−0.77/−0.89; `2026-06-18-TD-cueshift-dendrite-decision-scoping.md` closes the dendrite question NEGATIVE for B4) — its open work is *consolidation onto the merged bridge*; whereas **#9's genuine close needs the value to be BEHAVIOURALLY load-bearing**, a *task-design* gap orthogonal to whether the cue-shift signature can be produced. **Verdict: build ONE delayed-reward (trace-conditioning) harness that serves both — it gives B4 its consolidation test (does the cue-shift survive co-resident?) AND #9 its load-bearing test (does lesioning the dendrite-graded value collapse the trace-bridged behaviour?) — but score them as two separate gates.** The recommended close is cheap, reuse-heavy (the whole TD-CSC machinery + the limbic RPE battery + the value-train wiring already exist), numpy-first, and **is NOT where a real substrate wall is likely** — the substrate wall risk for #9 lives in the *separate* hidden-goal place→action arc (3× NEGATIVE), which the trace-conditioning task deliberately sidesteps by NOT requiring spatial credit assignment.

---

## 1. EXISTING-MACHINERY INVENTORY (the reuse map — almost everything needed already exists)

The single most important finding of the machinery survey: **the delayed-reward value task is ~90% reuse-by-import.** The TD critic, the eligibility traces, the spiking SNc r−V subtraction, the dendrite-graded value read-out, the Schultz RPE battery, the value-train wiring, AND the merged-bridge co-residence pattern are all built and validated. The genuine new work is the **task protocol** (a CS→gap→US trace schedule) + the **load-bearing gate** (lesion the dendrite-graded value, require collapse) + the **immediate-reward (DELAY) discriminating control**.

### 1.1 Value / TD / eligibility core (numpy + bridge)

| File | What it is | Reward timing | REUSE FOR the delayed-reward task |
|---|---|---|---|
| `sim/td_value_critic.py` | Pavlovian **TD(λ)** critic. `run_pavlovian(mode, seed, n_trials)`; `csc_features(onset)` = tapped-delay (complete-serial-compound) state; `analytic_vstar()`; `scale_free_transfer()`. `delta = r + gamma·V(s') − V(s)`; eligibility `e = gamma·lambda·e + phi(s)` reusing `fused_eligibility_trace_decay`. Frozen bars VALUE_RMSE≤0.05, TRANSFER≥0.90, US_DECAY≤0.15. | **DELAYED** (TRACE=4 steps; reward NOT immediate; jittered cue onset) | **The reference numpy TD engine.** Already implements the bootstrap + traces + the Schultz transfer metric over a delay. The trace-conditioning probe's numpy arm IS this (extend the schedule to a CS-then-GAP-then-US trace, add a value-lesion mode). |
| `sim/kernels.py::fused_eligibility_trace_decay` | The eligibility trace (catalog C.29 — "TD(λ) in all but name"). Geometric decay × DA gate × per-pathway plasticity gate. | substrate | **The temporal-credit carrier across the gap.** The lit (Yagishita-Kasai; eNeuro 2025) puts the useful eligibility window at ~0.2–2 s; the project's short-tau (~40 ms) bridges one tap/trial. Reuse verbatim; the gap length is a task knob. |
| `research/runners/td_critic_core.py` | The **THREE-STATE verdict harness** (PASS/FAIL/VOID): instrument-validity-first (VOID if V1 unsound OR controls don't fail), then science verdict. Frozen bars, MIN_SEEDS=3. | n/a (validation) | **The verdict framework, verbatim.** Copy the pattern for the trace-task's metrics (trace-CR-acquisition + lesion-collapse + delay-control-doesn't-need-value). |
| `research/runners/td_critic_gate.py` | Runnable gate: checkpoint (`sim.train_checkpoint`) + couples the TD delta into the NeuromodulatorManager as **phasic DA** (`_da_modulator_from_delta`, `from_reward` → `plasticity_rate` scope=all). | DELAYED 4-step | **The kill-safe checkpoint + the DA-coupling pattern** (proves the delta integrates with the validated phasic-DA substrate). |

### 1.2 Spiking limbic core / SNc / value-of-location (bridge — the brain-based pieces)

| File | What it is | Reward timing | REUSE FOR the delayed-reward task |
|---|---|---|---|
| `research/runners/_limbic_core_rpe_battery_derisk.py` | The **minimal ~170-neuron limbic organ** standalone: `cue (CS)→striosome_value (V, plastic)→snc (DA)` ⊕ `reward_us (US, spiking PPN-like)→snc (r)`, GABA_B/GIRK subtraction at SNc (E_K=−90 mV). `run_battery(...)` = the 6-gate Schultz battery (burst-on-US, graded-in-magnitude, omission-dip, predicted-US-shrinks, reward-lesion, critic-lesion). GO ≥5/6, lesions 3/3 clean. **Reward is SYNAPTIC (`reward_us`), not host current** = BRAIN-BASED-ONLY. | **DELAYED** (acquisition: ITI → PAIR(cue+tonic, 40 hold) → REWARD(cue on, reward_us fires + burst, 40 hold)) | **THE organ to extend.** This is the standalone limbic core that the trace-conditioning task slots into directly: add a CS-free GAP between the cue offset and the reward, and the cue-trace must bridge it. Its Schultz battery + lesion anti-cheats are the exact harness. **Honest note in the finding itself: it is NEGATIVE on the TD cue-shift (the R-W-vs-TD family) — i.e. it does R-W δ=r−V, not the bootstrap.** |
| `research/runners/snc_stageb_critic_probe_place.py` / `_navfaithful.py` / `snc_stageb_critic_probe.py --td-csc` | The **value-of-location** critic on a place code, AND (with `--td-csc`) the validated **TD cue-shift** (the A-CSC tapped-delay cue + B-2 conductance-derivative + multi-channel reward relay). The **`run_place_lead_sweep`** (train once, test at multiple leads) is the canonical de-risk for slow-GABA_B subtraction. | **DELAYED + nav-realistic LEAD** (place held tonic for `lead_steps` ms before reward → pre-builds V + slow GABA_B). The `--td-csc` arm is the cue→reward delay with the cue-shift. | **The B4 reference recipe + the lead-sweep protocol.** `--td-csc` is the validated cue-shift to lift; the lead-sweep is how you handle the GABA_B tau~150 ms integration over the gap. |
| `research/runners/_advantage_actor_critic_probe.py` | **Frémaux-Sprekeler-Gerstner spiking actor-critic** on a hidden goal: the deployed core `enable_neural_critic + spiking_snc + spiking_reward_us` routes **advantage δ = r − V(place)** as the actor's signed 3rd factor (`bridge.py:6901-6952`; `da_signal = conc(dopamine) − baseline`). `critic_warmup_trials` seeds V before the actor. | **IMMEDIATE on-nav + warmup DELAYED** | **The actor-critic credit path** — relevant to the *harder* #9 follow-on (instrumental trace conditioning where the agent must ACT, not just predict). NOTE: this probe is the one that came back PRELIMINARY NEGATIVE (§3.3 below) — it is the spatial-credit arm, NOT the Pavlovian trace arm. |

### 1.3 The dendrite-graded value mechanism (#9 itself) + the deploy

| File | What it is | REUSE FOR the delayed-reward task |
|---|---|---|
| `sim/config.py` + `sim/kernels.py` + `sim/bridge.py` (commits `d69cc0ab` + `f941a39b`) | The **guarded `sim/` edit** for #9: `enable_graded_dendritic_plateau` flag + `fused_graded_dendritic_plateau` (smooth non-saturating logistic, `V = max(sigmoid(slope·(c_w−center)) − floor, 0)`, `g_inc = strength·V`) + 4 guarded bridge sites. **Default-OFF byte-identical** (proven: the flag-off navfaithful regime reproduces Stage-0 EXACTLY + a 5-test guard suite). The value reads from `cp_conductance_g_graded_plateau`, subtracted at the SNc. | **The #9 mechanism under test.** The trace-conditioning load-bearing gate lesions THIS (`--...-graded-strength 0`) and requires the trace-bridged CR to collapse. NO new `sim/` edit needed for the task; the edit already ships. |
| `research/runners/_dendrite_stage1_onbridge_graded_plateau.py` | On-bridge validation: a `value_dendrite` MSN-D1 compartment bearing the graded plateau on the routed `vs_place_context → value_dendrite` pathway; the bridge's OWN reward-STDP grows the weights near-selectively. **Validated: graded V 3/3, ~9× near/far.** | **The on-bridge value read-out to wire into the trace cue.** Replace "place near/far" with "trace-cue value across the gap." |
| `research/runners/g11_bg_runner.py` (`--dendrite-critic`) | The DEPLOY wiring (runner-side, default-OFF): tags `vs_place_context → striosome_value` with `coincidence_detector`, sets the cfg dendrite block. | The deploy template — but the deploy is into the immediate-reward nav, which is the confounded task. The trace-conditioning task is the FIX for the confound. |
| `research/findings/raw/dendrite_critic/_verdict_aggregate.py` + per-seed JSONs | The deploy nav table (re-run this pass — see §2.2). | The numerical proof of the qualified-NEGATIVE. |

### 1.4 The B4 (TD cue-shift) + merged-bridge co-residence machinery

| File | What it is | REUSE FOR the unified task |
|---|---|---|
| `research/runners/_merged_td_cueshift_consolidation_derisk.py` | The **B4 consolidation probe**: lift the A-CSC TD slice (K=8 tapped-delay cue + critic + FS-clamp + reward relay + the B-2 conductance-derivative) onto the merged bridge as an additive default-off slice; run the Pavlovian battery + migration-r + cue-pathway-lesion + unpaired-timing anti-cheats. Frozen GO bar migration r<−0.7, ≥5/6 seeds; MOAT byte-intact; NAV byte-identity. CLI `--td-csc-n 8 --td-stdp-w-max 40 --td-gabab-conductance-max ...`. | **THE B4 half of the unified harness.** It already is the cue-shift-on-the-merged-bridge test. Pair it with the #9 load-bearing gate in the same delayed-reward family. |
| `research/runners/_merged_td_cueshift_opsearch.py` | Bounded coordinate-descent op-point search for the merged cue-shift (the per-tap weight clip `td_stdp_w_max=40` is the primary lever; global `stdp_w_max=400` runs the critic away). | The merged op-point tuner (engineering, not biology). |
| `research/runners/_merged_navcritic_valuetrain.py` | The merged-bridge **value-train** (DA-gated STDP on `vs_place_context → striosome_value`) + `check_moat` (asserts `what_does(dog,go)=='north'` AND `what_does(river,look) is None` under the dopamine `scope=all` broadcast) + `lesion_gabab` + the GIRK-cap op-point. **DELAYED protocol** (`REWARD_DELAY_STEPS=8`, PAIR-then-REWARD-after-8). | **The merged co-residence wiring + the MOAT anti-cheat verbatim.** The trace task reuses `check_moat` to assert the no-confab moat survives. |
| `research/runners/_merged_limbic_coresident_validate.py` | The 4-region limbic core co-resident on the merged bridge: zero `cp_connections` out-edges to non-limbic (nav-inert), default-off byte-preserved, the Schultz RPE arithmetic co-resident. | The co-residence template (additive masked-region slice, byte-identity-when-off). |
| `research/runners/nav_conv_merged_bridge.py` | `build_merged_nav_conv_bridge` + the `co_resident_*` masked-region pattern + the merged `dopamine` modulator over `["snc"]` + the GABA_B route + `value_input` gate. | The merge host (if the unified task is run on the one brain). |

### 1.5 The dendritic substrate (for completeness — NOT needed for this task)

`sim/dendritic_neuron.py` (`DendriticLayer`, Larkum BAC / Guerguiev-Lillicrap-Richards), `sim/dendritic_plasticity.py` (`urbanczik_senn_update`), `sim/dendritic_mlp.py`. **These are the SPATIAL (cross-neuron decorrelation / feedback-alignment) dendrite** — a single leak, fixed-random apical, no multi-timescale temporal eligibility. The B4 scoping established (and I re-verified) that this is the **WRONG dendrite** for temporal credit; the temporal-dendrite function (BTSP plateau-as-eligibility) is realized *functionally* by the B-2 GABA_B conductance EMA on point neurons. **Out of scope for the delayed-reward task** (which does not need a dendritic rewrite — see §6).

---

## 2. THE DIAGNOSIS — why the #9 deploy is a qualified NEGATIVE (re-verified numerically this pass)

### 2.1 The validate-by-function root cause (the same lesson as N5 reward)

The owner standard `feedback_validate_signal_by_its_function`: a signal looks validated by an A/B that the signal is not actually load-bearing for. N5 reward "passed" a nav A/B because the task was orient-solvable (the reward was not load-bearing). **The #9 dendrite-graded value has the identical confound**: the moving-goal nav delivers a **dense per-step reward** the moment the agent is at/near the goal, so the policy can be learned/driven by immediate reinforcement WITHOUT any *predictive* value-of-future-state. A value critic's distinctive function — propagate future reward back to the *current* state across a temporal gap (the bootstrap `γV(s′)`) — is **never required** because there is no gap to bridge. The task does not discriminate "the dendrite-graded value is doing work" from "immediate reward + NMDA-on-the-critic-slice is doing the work."

### 2.2 The deploy numbers (RE-RUN this pass — `_verdict_aggregate.py` on the raw per-seed JSONs)

```
arm         seed     sum atgoal  striov   snc  cw_final  quarters
dendcritic    42   8.507    828   263.0  50.0    119.96
dendcritic    43   8.489    828   264.3  50.0    121.58
dendcritic    44   8.413    829   261.5  50.0    124.26
baseline      42  21.476    255     0.0   3.9       0.2     (point-neuron LINEAR critic: 0 Hz, no V)
lesion        42   8.516    828     0.0   4.7       0.2     (dendrite value SILENCED: graded-strength=0)
lesion        43  10.053    810     0.0   6.8       0.2
lesion        44   8.658    827     0.0   6.1       0.2
ctrl_nmda     42   8.716    -      0.0   3.4       0.2     (baseline + global NMDA, NO dendrite value)

MEANS:  dendcritic 8.470 (sd 0.040)   baseline 20.883 (sd 1.596)   lesion 9.076 (sd 0.694)
  dendcritic vs lesion(value silenced): 8.470 vs 9.076 -> Δ 7.2%  (LESION ≈ DENDCRITIC => value NOT load-bearing)
  dendcritic vs baseline: 59.4% better -> BUT ctrl_nmda (NO value) = 8.716 ≈ dendcritic
```

Three facts pin the qualified-NEGATIVE:
1. **Value lesion ≈ deploy (8.47 vs 9.08, Δ7.2%).** Silencing the dendrite-graded value barely changes navigation. The value is not load-bearing.
2. **The whole gain over the point-neuron baseline is the NMDA on the critic slice, NOT the value.** `ctrl_nmda` (baseline + global NMDA, **no dendrite value**) = 8.72 ≈ the deploy's 8.47. The improvement from 20.9 → 8.5 is the NMDA's slow integration making the critic slice fire, NOT the graded value shaping the policy.
3. **The SNc saturates (flat 50.0 Hz across all dendcritic seeds; striov ~263 Hz).** The critic over-fires and the SNc is pinned — the value cannot GRADE the dopamine burst (the same n_snc-quantization + saturation the on-bridge Stage-1 finding flagged). Even if the task DID need the value, the read-out is saturated.

(The "1/12 phase-seed cells load-bearing" framing in the task prompt is the per-quarter-per-seed version of the same Δ7.2% — the lesion changes a single quarter on a single seed and is otherwise identical. The aggregate Δ is the cleaner statement.)

### 2.3 The genuine #9 close = (a) calibrate the SNc subtraction so the value GRADES, AND (b) a task where the value is PROVABLY load-bearing

Both halves are needed and they are separable:
- **(a) SNc-subtraction calibration** (read-out engineering): a denser SNc population OR a V-magnitude-scaled subtract gain so the small graded ΔV moves a *graded* (not saturated, not all-or-none) burst. The on-bridge Stage-1 handoff already names the levers (read the instantaneous logistic; widen c_w separation; keep the MSN sub-somatic). This is necessary for the value to even be EXPRESSED as graded dopamine.
- **(b) the delayed-reward task** (this doc's deliverable): a task whose correct behaviour REQUIRES the predictive value, so the lesion of the (now-graded) value collapses behaviour and an immediate-reward control does NOT need the value.

---

## 3. THE BIOLOGY — how biology does delayed credit assignment → the minimal task

### 3.1 The mechanisms (catalog + Sutton-Barto + literature, re-confirmed)

- **TD learning / bootstrapping (catalog C.28/C.31; S&B Ch 6).** δ = r + γV(s′) − V(s). The bracketed bootstrap is what lets a single trial shift the prediction (no episode-end wait). The empirical proof of bootstrapping in biology is the Schultz cue-shift (C.22): the DA burst migrates from US to CS over consecutive trials — **direct evidence the brain updates from one moment to the next using its current value estimate as the target.**
- **Eligibility traces (catalog C.29; S&B Ch 7; Yagishita-Kasai 2014; Gerstner-Lehmann-Liakoni-Corneil-Brea 2018).** `e_t(s)` decays by γλ and is gated by DA — the synaptic memory that **bridges the gap between a sensorimotor event and a later reward**. The literature (this pass) puts the useful window at ~0.2–2 s; a 5-HT/ACh-prolonged trace extends the coincidence window to bridge longer CS-US gaps (the insect DPM result; Frontiers 2018 NeoHebbian three-factor support). **This is the substrate the project already has (`fused_eligibility_trace_decay`).**
- **Dopamine ramps / value-coding across a gap (Schultz; eNeuro 2025 "Nucleus Accumbens Dopamine Encodes the Trace Period").** Cue-evoked DA encodes the trace period and tracks the reward value; the DA response to a reward-predictive cue signals value through bidirectional changes — i.e. **the value signal IS what carries the prediction across the empty gap.**
- **Actor-critic (catalog C.30; S&B Ch 11; Houk-Adams-Barto 1995; F-S-G 2013).** Critic (striosome) learns V(s); its δ updates both itself and the actor (matrix). The cleanest RL account of the BG-DA system. The catalog: "partial — actor implemented, critic missing... no separable population that outputs a learned V(s), and consequently no bootstrapping."

### 3.2 The canonical paradigm where a VALUE/credit MUST bridge a stimulus-reward gap — TRACE CONDITIONING (catalog F.22/F.23)

The catalog names the exact paradigm and — decisively — the exact *control*:

> **F.22 Trace conditioning — hippocampus-dependent CS-US bridging.** "When CS terminates before US onset (trace gap > 0), the [system] must associate a CS-driven signal with a US delivered hundreds of ms later... For longer traces, hippocampectomised rabbits fail entirely (Moyer et al. 1990 — 500 ms trace abolishes learning; 300 ms learns normally; H.M.-class amnesics fail traces but learn delay normally)."
>
> **Behavioral validation:** "CS-US gap parameter sweep... **Delay conditioning (no gap, US during CS) is unaffected** by [lesion]. **(Two-axis validation: delay vs trace × [lesion] vs no-[lesion] → 2×2 factorial with sharp predictions.)**"

This is the perfect template because the catalog's own validation logic gives BOTH the load-bearing gate AND the discriminating control:
- **TRACE (CS-free gap before US)** = the task that REQUIRES a value/trace to bridge the gap → **lesion the bridging mechanism (the dendrite-graded value), require collapse** (= the #9 load-bearing gate).
- **DELAY (no gap, US during CS)** = the same apparatus with the gap removed → solvable WITHOUT the bridge → **the lesion does NOT collapse it** (= the immediate-reward discriminating control that directly answers the deploy confound).

The biological dissociation (H.M. fails trace, learns delay) is the gold-standard proof that the gap-bridging mechanism is dissociable and load-bearing — exactly the contrast the #9 deploy lacked.

---

## 4. THE DESIGNED TASK — delayed-reward (trace-conditioning) value-IS-load-bearing

### 4.1 Protocol (the trace-conditioning schedule on the limbic core)

**Substrate (reuse `_limbic_core_rpe_battery_derisk.build_limbic_core` + the #9 graded-plateau on the critic afferent):**
`cue (CS) → striosome_value/value_dendrite (V, plastic, dendrite-graded plateau) → snc (DA)` ⊕ `reward_us (US, spiking PPN-like) → snc (r)`, GABA_B/GIRK subtraction at the SNc. Concept codes / cue drive are the substrate's own (orthogonal sparse cue patterns).

**A trial (the TRACE arm):**
1. **ITI floor** — SNc tonic only (baseline DA).
2. **CS window** (e.g. 40 steps) — drive the cue population; the cue's firing tags the `cue → critic` synapses (eligibility builds). The dendrite-graded value V begins to be read at the critic.
3. **TRACE GAP** (the load-bearing knob; e.g. 0 / 100 / 300 / 600 ms-equivalent steps, CS-FREE) — NO cue drive, NO reward. The eligibility trace + the slow critic conductance must CARRY the value across the empty gap. **This is the only place the predictive value can matter.**
4. **US window** — `reward_us` fires (spiking, synaptic — the r term); the critic's held value V is subtracted at the SNc (δ = r − V). DA-gated STDP converts the surviving eligibility to weight.

**The learned signature (what "the value is doing work" looks like):** across trials, (i) the `cue → critic` weight grows (V acquires), (ii) the **DA burst migrates onto the CS** and the omission of the US produces a **dip at the expected-reward time** (the Schultz/HS98 signature — the value is PREDICTING across the gap), (iii) a behavioural read-out (a CR-analogue: an anticipatory critic/actor response in the gap timed to the expected US) emerges ONLY when the value bridges the gap.

**The DELAY arm (the discriminating immediate-reward control):** the SAME apparatus with **GAP = 0** (US delivered DURING the CS — co-active, no trace interval). The reward coincides with the cue, so the association forms by immediate coincidence; **no predictive value across a gap is needed.**

### 4.2 The load-bearing GO bar (validate-by-function — the genuine #9 close)

Pre-registered, frozen, NOT tuned on the test (inherit the limbic-battery + TD-CSC bars):

| Gate | Criterion |
|---|---|
| **(G1) TRACE acquisition** | At gap ≥ 300 ms-equiv, the value acquires + the anticipatory CR-analogue (or the cue-shift migration r<−0.7 for the Pavlovian read) emerges across trials, ≥5/6 seeds. |
| **(G2) #9 LOAD-BEARING (the headline)** | **Lesion the dendrite-graded value** (`enable_graded_dendritic_plateau=False` / `--...-graded-strength 0`) → the TRACE-arm CR-analogue / migration **COLLAPSES** (to the no-bridge floor), ≥5/6 seeds. This is the gate the nav deploy failed (its Δ was 7.2%; here the lesion must move the trace-bridged behaviour from acquired to floor). |
| **(G3) DELAY control does NOT need the value (the discriminator)** | The SAME lesion on the **DELAY arm (gap=0)** does **NOT** collapse it (the immediate-coincidence association survives without the bridge), ≥5/6 seeds. **This is the direct answer to the deploy confound: it proves the task discriminates "needs the critic" from "immediate-reward-solvable."** |
| **(G4) graded, not saturated** | With the SNc-subtraction calibration (§2.3a), the value GRADES the burst (the unpredicted/predicted gap is graded with the trace fidelity, NOT the flat-50-Hz / 0-Hz saturation the deploy showed). |
| **(G5) dendrite-graded vs point-neuron value** | The dendrite-graded value supplies the gap-bridging credit BETTER than the point-neuron rate critic (the point-neuron LINEAR critic, per burndown-9, fires 0 Hz / can't grade — so the comparison is "graded-plateau bridges the gap" vs "point-neuron can't"). This is the #9-specific positive control. |

**Pass = G1 ∧ G2 ∧ G3 (+ G4 for the calibration close, + G5 for the dendrite-specific claim).** G2∧G3 together are the validate-by-function close: lesion collapses the gap-task AND does not collapse the no-gap-control → the value is provably load-bearing for the function it computes (credit over a gap), and the task is proven to discriminate.

### 4.3 Two task variants (Pavlovian-first, instrumental-as-follow-on)

- **(V-A) Pavlovian trace (RECOMMENDED FIRST — cheapest, cleanest, sidesteps the substrate wall).** The agent only PREDICTS (the CR-analogue is an anticipatory critic/actor response); no spatial credit assignment. This is the `_limbic_core_rpe_battery` + a trace gap + the value lesion. **Maps directly onto the validated machinery; the substrate-wall risk is LOW** (the cue-shift is already a point-neuron GO; the only new thing is the gap, and the eligibility trace is built for exactly that).
- **(V-B) Instrumental trace (FOLLOW-ON, higher-variance).** The agent must ACT during/after the gap to obtain the delayed reward (e.g. a "hold then act at the expected US time" or a 2-step "go-A-now to get reward-later" task). This routes the advantage δ = r − V into the actor's eligibility (the `_advantage_actor_critic_probe` path). **This is where the genuine substrate wall MIGHT appear** — it is the actor-critic-credit family that came back 3× NEGATIVE on the spatial hidden-goal (§3.3 of the actor-critic finding). Do V-B only after V-A is GREEN, and treat a V-B NEGATIVE as the honest characterized deliverable (it would localize the wall to ACT-over-gap, distinct from PREDICT-over-gap).

---

## 5. THE #9 ↔ B4 UNIFICATION VERDICT

**Same broad family, distinct sub-problems, ONE harness serves both, score as two gates.**

### 5.1 The family

Both #9 and B4 are "value / credit-assignment over a temporal gap." Both flow reward-modulated plasticity through the SAME neural substrate (eligibility traces + the spiking SNc r−V subtraction + the value critic). A single delayed-reward (trace-conditioning) harness — the CS→gap→US schedule on the limbic core — is the natural common home for both.

### 5.2 The distinction (the project's own work already split them)

| Axis | **B4 (TD cue-shift)** | **#9 (dendrite-graded value load-bearing)** |
|---|---|---|
| **The question** | Can the system PRODUCE the Schultz cue-shift SIGNATURE (burst migrates US→CS, omission dip)? | Is the (dendrite-graded) VALUE BEHAVIOURALLY LOAD-BEARING (does lesioning it collapse a value-requiring behaviour)? |
| **Dependent variable** | Dopamine burst TIMING (migration r) | Behaviour / CR-analogue collapse under lesion |
| **Point-neuron status** | **MULTI-SEED GO** (r = −0.80/−0.77/−0.89 standalone; A-CSC + B-2 conductance-derivative). Dendrite question **CLOSED NEGATIVE** for B4 (`2026-06-18-TD-cueshift-dendrite-decision-scoping.md`). | GO in isolation (graded V 3/3); DEPLOY qualified-NEGATIVE because the nav task is immediate-reward-solvable (the confound this doc fixes). |
| **The open work** | **CONSOLIDATION** — does the validated cue-shift survive co-resident on the merged "one brain" alongside the R-W limbic core? (a co-residence engineering test, NOT biology) | **TASK DESIGN** — a task where the value is provably load-bearing (the trace-conditioning load-bearing gate). |
| **Where a substrate wall might live** | NOT here (point-neuron GO already). | NOT in the Pavlovian trace (V-A); MAYBE in the instrumental act-over-gap (V-B) — but that is the *separate* actor-critic-credit family (the hidden-goal 3× NEGATIVE), which the Pavlovian trace deliberately avoids. |

### 5.3 The recommended consolidation

**Build ONE trace-conditioning harness on the limbic core (Pavlovian V-A first). It serves BOTH:**
- **For B4:** run the cue→gap→US schedule with the A-CSC tapped-delay cue + B-2 conductance-derivative → measure the migration-r (the cue-shift). This IS the B4 consolidation test (`_merged_td_cueshift_consolidation_derisk` already does it; the trace gap is the natural extension that makes it a *trace* cue-shift, not just a delay cue-shift).
- **For #9:** the SAME schedule with the dendrite-graded value on the critic afferent → lesion it, require the TRACE-arm behaviour to collapse while the DELAY-arm does not (the load-bearing gate).

They share the schedule, the limbic organ, the SNc r−V subtraction, the eligibility traces, the Schultz battery, the MOAT anti-cheat, and the co-residence pattern. They differ ONLY in the read-out (B4 = burst-timing migration; #9 = behaviour-under-lesion) and the manipulated variable (B4 = is the cue-shift produced; #9 = is the value lesion-collapsing). **Score them as two separate gates on one harness.** This is a clean consolidation, NOT a forced merge.

---

## 6. THE DE-RISK + ANTI-CHEAT PLAN

### 6.1 Cheapest-first ladder (numpy → bridge, reuse-heavy)

1. **(numpy probe, CPU)** Extend `sim/td_value_critic.run_pavlovian` to a **trace schedule** (CS window → CS-free gap → US) + a **value-lesion mode** (zero the bootstrap/value) + a **delay mode** (gap=0). Confirm the analytic prediction: TRACE acquisition needs the value (lesion → no transfer), DELAY does not (lesion → still associates). This is the pure-RL sanity check that the task DISCRIMINATES before any spiking. (~hours, no GPU.)
2. **(bridge probe, CPU smoke → GPU multi-seed)** Lift the trace schedule onto `_limbic_core_rpe_battery_derisk.build_limbic_core` with the #9 graded-plateau on the critic afferent. Run the 5-gate battery (§4.2) at gap ∈ {0, 300, 600 ms-equiv}, seeds 42/43/44 → 42..102. Reuse the lead-sweep protocol (`snc_stageb_critic_probe_place.run_place_lead_sweep`) for the slow-GABA_B integration over the gap.
3. **(merged-bridge consolidation, GPU)** Run BOTH gates co-resident via `_merged_td_cueshift_consolidation_derisk` (B4) + the #9 load-bearing extension, asserting the MOAT + NAV byte-identity. Only if steps 1-2 are GREEN.

### 6.2 Anti-cheats (the value must be NEURAL + the gap-bridge genuine)

- **(AC1, the headline) LESION-COLLAPSES** — silence the dendrite-graded value → the TRACE-arm behaviour collapses to the no-bridge floor (the gate the deploy failed). Reuse `_merged_navcritic_valuetrain.lesion_gabab` + the `--...-graded-strength 0` flag.
- **(AC2, the discriminator) DELAY-CONTROL-DOESN'T-NEED-VALUE** — the SAME lesion on the gap=0 arm does NOT collapse it. **This is the direct anti-confound** (proves the task is not immediate-reward-solvable in the trace arm). Without AC2, AC1 alone could be a generic "lesion breaks everything" artifact.
- **(AC3) NO-LEARNING control** — freeze the `cue → critic` STDP → no acquisition, no migration, no CR-analogue (the value must be LEARNED, not structural).
- **(AC4) PERMUTED / UNPAIRED-TIMING control** — random US timing (no CS→US contingency) → no acquisition (reuse the `_merged_td_cueshift_consolidation` `--unpaired` arm; the standalone got r = −0.28/−0.25/−0.28 paired-vs-unpaired).
- **(AC5) GABA_B-subtraction lesion** — zero the `striosome_value → snc` GABA_B mask → δ collapses to ≈ r (the subtraction is the conductance, load-bearing, not host arithmetic).
- **(AC6) MOAT byte-intact** — `check_moat`: `what_does(dog,go)=='north'` AND `what_does(river,look) is None` under the dopamine `scope=all` broadcast, byte-identical. **The conversational no-confab moat is ARRAY-DISJOINT from the nav/limbic critic by construction** (the RF complex weights `cp_rf_w_re/im` are separate arrays from `cp_connections`; the limbic regions have zero `cp_connections` out-edges to conversational slices; `enable_graded_dendritic_plateau` is default-OFF for the conversational slices). **NEVER weaken the moat** — it is preserved here by construction and re-asserted by the gate.
- **(AC7) REGIME FIDELITY** — deterministic regime (OU / conductance-noise / homeostasis OFF), faithful scale, asserted per seed (the #6 lesson: a permissive smoke misled before).
- **(AC8) HOST-PROVENANCE** — under the trace mode, the SNc drive is `tonic + reward_us(synaptic) + GABA_B(−V) + conductance-derivative(+dV/dt)` ONLY; `current_reward_signal == 0`, no host δ / value-EMA. The cue/US timing is world-presented (the legitimate environment boundary); the value/credit/burst/dip are 100% neural.

### 6.3 sim/-edit-or-not

**NO new `sim/` edit needed.** The #9 graded-plateau edit already ships (`d69cc0ab`+`f941a39b`, byte-reviewed, default-OFF). The B-2 conductance-derivative edit already ships (byte-reviewed). The eligibility traces, the GABA_B/GIRK subtraction, the spiking `reward_us`, the dopamine modulator, the merged co-residence pattern — all exist. The task is **runner-side only** (a trace-schedule extension + the lesion gate + the delay control). The ONLY residual `sim/`-adjacent question is the **SNc-subtraction calibration** (§2.3a) for G4 — and even that is parameter/read-out engineering on the runner side first (denser SNc / V-scaled gain / read the instantaneous logistic), not a new mechanism.

---

## 7. HONEST FRAMING — is this the genuine close, and is the substrate-wall risk real here?

### 7.1 Is this the genuine close for #9?

**Yes, for the Pavlovian trace (V-A).** The #9 deploy's qualified-NEGATIVE is a *task-design* failure (the nav task doesn't exercise the value's function), not a mechanism failure (the graded V is validated in isolation). A trace-conditioning task with the load-bearing lesion gate + the delay discriminating control is the direct, minimal fix: it forces the value to be the thing that bridges the gap, and it proves (via the delay control) that the task discriminates. If V-A passes G1∧G2∧G3∧G4, #9 closes as a **genuine GO** (the dendrite-graded value is provably load-bearing for delayed credit assignment, graded not saturated, the lesion collapses the gap-task and not the no-gap-control). If it fails, it fails for a *characterizable* reason (the gap-bridge SNR; the SNc quantization; the eligibility window) that is the honest deliverable.

### 7.2 Is the substrate-wall risk real here?

**Low for V-A (Pavlovian), genuine for V-B (instrumental) — and that is the useful distinction.** The value loop is the project's flagged "one place a real substrate wall might appear," but the survey shows the wall has a SPECIFIC location:
- **The cue-shift / Pavlovian prediction over a gap is a SOLVED point-neuron problem** (B4 GO; eligibility traces built for exactly this; the lit confirms DA encodes the trace period on point-neuron-modelable circuits). So **V-A is unlikely to hit a substrate wall** — the gap is just a longer eligibility window, which the substrate has.
- **The substrate wall the project actually keeps hitting is the ACTOR-CRITIC SPATIAL credit assignment** (hidden-goal place→action: 3× NEGATIVE — 2026-05-05 global-scalar + the limbic-load-bearing diagnostic + the advantage-routing de-risk). That is `r − V(place)` failing to carve a place→action policy on the point-neuron cascade — a DIFFERENT problem (spatial credit + structural-bias confound), which the named unlocker is the dendrite (apical-basal credit assignment). **The trace-conditioning task DELIBERATELY SIDESTEPS this** (V-A only predicts; it does not require spatial credit). **V-B (instrumental act-over-gap) is where the wall might re-appear** — and if it does, that NEGATIVE is the honest characterized boundary (act-over-gap distinct from predict-over-gap), and it is the legitimate juncture for the deferred dendritic substrate question — NOT the trace-conditioning Pavlovian close.

**So the genuine close is: V-A closes #9 cheaply and is unlikely to wall; V-B is the optional deeper probe where a real substrate wall would be a *finding*, not a failure.** The recommendation is V-A first (the genuine #9 close), V-B as the characterizing follow-on.

### 7.3 Non-claims / honest scope

- I did NOT re-run any probe; I re-ran the **deploy verdict aggregator** on the existing raw JSONs (the §2.2 numbers are measured) and read the four #9 findings + the B4 scoping + the two limbic-load-bearing findings + the catalog F.22/F.23 + the lit in full.
- The B4 "point-neuron GO / dendrite-CLOSED-NEGATIVE" verdict is the project's own (`2026-06-18-TD-cueshift-dendrite-decision-scoping.md`), re-verified against the A-CSC GO finding it cites; I did not relitigate it.
- The #9↔B4 unification is "one harness, two gates" — a consolidation, not a claim that they are the same problem (they are not; §5.2).
- The substrate-wall localization (V-A safe / V-B genuine) is a forward prediction grounded in the 3× actor-critic NEGATIVE + the B4 GO; the trace-conditioning de-risk is what would confirm it.
- The no-confab moat is preserved by construction (array-disjoint) and re-asserted by AC6; **NEVER weakened.**

---

## 8. Sources

### Project record (re-verified this pass; the deploy numbers RE-RUN)
- **#9 deploy qualified-NEGATIVE (the numbers, re-run):** `research/findings/raw/dendrite_critic/_verdict_aggregate.py` on the per-seed JSONs (`dendcritic/baseline/lesion/ctrl_nmda_seed{42,43,44}.json`) → dendcritic 8.47 ≈ lesion 9.08 (Δ7.2%); ctrl_nmda 8.72 (NMDA carries the gain, not the value); SNc flat 50 Hz. `research/findings/2026-06-20-shortcut9-dendrite-critic-deploy.md` (the deploy wiring, PENDING table now filled by the aggregator).
- **#9 mechanism GO-in-isolation:** `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md` (on-bridge graded V 3/3, ~9× near/far; the `sim/` edit byte-review; the SNc-quantization gap), `2026-06-20-dendrite-derisk-A-graded-plateau-readout.md` (Stage-0 δ=1.33 6/6), `2026-06-20-burndown-9-critic-graded-readout.md` (the point-neuron LINEAR critic can't fire the MSN; all-or-none over-clamps — the read-out FORK).
- **B4 (TD cue-shift) point-neuron GO + dendrite CLOSED-NEGATIVE:** `2026-06-18-TD-cueshift-dendrite-decision-scoping.md` (the full diagnosis; D2 = wrong dendrite; consolidation is the open work), citing `2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (r −0.80/−0.77/−0.89).
- **The validate-by-function diagnosis (the confound):** `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` (the hidden-goal NEGATIVE; structural-corner-drift confound), `2026-06-19-spiking-actor-critic-advantage-routing-derisk.md` (the 3rd actor-critic-credit NEGATIVE; the V-B substrate-wall location), `feedback_validate_signal_by_its_function`.
- **Reuse machinery:** `sim/td_value_critic.py`, `sim/kernels.py::fused_eligibility_trace_decay`, `research/runners/{td_critic_core,td_critic_gate,_limbic_core_rpe_battery_derisk,snc_stageb_critic_probe_place,_advantage_actor_critic_probe,_merged_td_cueshift_consolidation_derisk,_merged_navcritic_valuetrain,_merged_limbic_coresident_validate,nav_conv_merged_bridge}.py`. The #9 `sim/` edit `d69cc0ab`+`f941a39b`. The moat (`check_moat`; `cp_rf_w_re/im` array-disjoint from `cp_connections`).

### Feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **F.22** Trace conditioning — hippocampus-dependent CS-US bridging; the gap-sweep + the **delay-vs-trace × lesion-vs-no-lesion 2×2 factorial** (the load-bearing gate + the discriminating control). `:1922-1951`.
- **F.23** Hippocampus-dependent classical-conditioning six-pack (trace, reversal, latent inhibition, conditional discrimination, sensory preconditioning, blocking). `:1953-1994`.
- **C.28** TD error / **C.29** eligibility traces (implemented) / **C.30** actor-critic (critic missing) / **C.31** bootstrapping vs Monte Carlo / **C.22** Schultz RPE (the cue-shift + omission-dip acceptance numbers; HS98 graded transfer). `:574-611, :907-921`.

### Peer-reviewed literature (re-confirmed this pass)
- Schultz, Dayan, Montague (1997) *Science* 275:1593 — the cue-shift / TD-dopamine.
- Hollerman & Schultz (1998) *Nat. Neurosci.* 1:304 — graded cue-shift + omission dip.
- Moyer, Deyo, Disterhoft (1990) — hippocampectomy abolishes 500 ms trace, 300 ms learns; H.M. fails trace, learns delay (the dissociation).
- Hesslow & Yeo (2002) — trace conditioning + the six-pack (catalog F.22/F.23 source).
- Yagishita et al. (2014) *Science* — the ~1 s eligibility/sensitive-period for reward-DA in the NAc.
- Gerstner, Lehmann, Liakoni, Corneil, Brea (2018) *Front. Neural Circuits* 12:53 — eligibility traces on behavioural time scales (NeoHebbian three-factor).
- **NAc Dopamine Encodes the Trace Period during Appetitive Pavlovian Conditioning (2025)** *eNeuro* 12(5) ENEURO.0016-25 — cue-evoked DA encodes the trace period + tracks value (the gap-bridging value signal).
- Frémaux, Sprekeler, Gerstner (2013) *PLoS Comput. Biol.* 9:e1003024 — the spiking actor-critic (the V-B path).
- Sutton & Barto *RL* 2e — Ch 6 (TD/bootstrap), Ch 7 (eligibility/TD(λ)), Ch 11 (actor-critic), Ch 12 (CSC/cue-shift).
