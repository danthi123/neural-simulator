# Spiking-SNc Stage B — neural striosome value-critic: cheap-first de-risk result

**Date:** 2026-06-08
**Type:** Cheap-first de-risk (CPU/numpy, no nav build, no GPU) — per the standing practice, run the falsifier BEFORE committing the nav build + the heavy 6-seed GPU gate.
**Probe:** `research/runners/snc_stageb_critic_probe.py`
**Raw:** `research/findings/raw/_stageb_critic_derisk.json`
**Research scoping:** `2026-06-08-spiking-snc-stageB-striosome-critic-research.md`

## Verdict (honest, multi-seed)

> **The NEURAL value learning is VALIDATED (the hard part); the MEMBRANE SUBTRACTION hits the documented depolarized-SNc-GABA-reversal wall (§6 Q4). This is a design fork to resolve BEFORE the nav build, not a clean GO.**

| signature (3 seeds, 42/43/44) | result | meaning |
|---|---|---|
| **V-learned** (striosome rate on CS rises with training) | **2/3** (s42 weak at this tonic; s43 25→104 Hz, s44 20→89 Hz) | the critic learns value via the SNc-derived δ |
| **V cue-gated** (state-specific) | **3/3** (e.g. s43: 110.8 Hz with cue vs 9.2 without) | the value is NEURAL + STATE-DEPENDENT — a host global-EMA cannot do this |
| **omission-dip** (CS, no reward → SNc dips below tonic) | **3/3** | the signed DA + the value inhibition produce the dip |
| **state-specific SNc gap** (predicted burst < unpredicted) | **0/3** | the membrane subtraction does NOT cleanly cancel the reward |

So: **the striosome critic genuinely learns a cue-gated, state-dependent value from the SNc's own dopamine δ** (V rises 20-25 → 90-104 Hz on the CS, cue-gated, ω-dip 3/3). The host `reward_ema` scaffold is replaced by a real neural value. **What does NOT cleanly work is the value SUBTRACTING the reward at the SNc membrane.**

## The membrane-subtraction wall (the §6 Q4 calibration risk, materialized)

The SNc lacks the KCC2 chloride exporter, so its GABA_A reversal is depolarized (`E_GABA = −55 mV`, faithfully encoded). The striosome→SNc inhibitory current is `g_i·(V − E_inh) = g_i·(V + 55)`:
- when the SNc is **hyperpolarized** (V < −55, much of the inter-spike interval) GABA is **depolarizing (excitatory)**;
- only when **depolarized near threshold** (V > −55) is it hyperpolarizing.

Empirically the net effect at the reward operating point was **sign-inverted**: more striosome firing → *higher* SNc burst (e.g. raising the striosome→SNc weight 10→60 moved predicted 79→102 Hz, the wrong way). Raising the SNc tonic (350 pA) flipped the sign to *correct* (predicted 95 < unpredicted 102) but the gap stayed tiny — the depolarized reversal gives only weak shunting inhibition. This is exactly the calibration challenge the research review flagged (§2.2, §6 risk 1, §6 Q4): the depolarized SNc reversal makes a clean membrane value-subtraction finicky and weak.

This is a **biophysically-grounded honest finding**, not a bug: faithful SNc GABA biophysics resists a strong membrane subtraction.

## Honest scoping note — Rescorla-Wagner, not the full Schultz cue-shift

The minimal membrane scheme `I_snc = tonic + k_r·max(0,r) − inhibition(V)` implements **Rescorla-Wagner δ = r − V**, not the temporal-difference δ = r + γV(s′) − V(s). R-W gives US-burst-shrink + omission-dip (and a *dip*, not a burst, at the CS); the full Schultz burst-**migration onto the CS** needs the TD bootstrap (a deeper, later increment). So this de-risk tested the R-W-achievable + host-EMA-impossible signature (**CS-gated prediction**), not the textbook cue-shift. The research doc's "cue-shift" gate is split: the US-burst-shrink half is R-W; the CS-burst half is TD (deferred).

## Four real bugs found + fixed during the de-risk (reusable lessons)

1. **`current_time_ms` not advanced → STDP dead.** STDP reads `runtime_state.current_time_ms` for the pre/post `delta_t`. A probe that advances only `current_time_step` leaves `current_time_ms = 0`, so every `delta_t = 0`, STDP emits an exactly-zero update, and **no eligibility ever forms** (the critic cannot learn). The nav runner advances `current_time_ms` manually each step; any standalone learning harness must too. (This was the decisive fix — eligibility went 0 → non-zero, the weight grew, V learned.)
2. **STDP soft-bound `stdp_w_max=2.0` (CLAUDE.md gotcha).** `Δw_LTP = A_plus·(w_max − w)·exp(..)`: with a cue→striosome weight > 2, every LTP event goes strongly negative and the weight collapses to 2 — V can never rise. Fix: `stdp_w_max` well above the critic's working range (40).
3. **Short-term depression starves the cortico-striatal synapse.** At the cue rates needed to drive the MSN-typed striosome, the depressing E→I synapse (`stp_U=0.15, tau_d=200ms`) collapses transmission to near-zero. Disabled STP for this minimal mechanism probe (orthogonal feature; documented confound removal).
4. **Dopamine threshold mis-calibration.** The signed rule emits `sensitivity·(rate_ema − threshold)`; the static `0.30` default is above even the reward-burst firing fraction, so `da_signal` is negative throughout (pure LTD). Auto-calibrate the threshold to the measured tonic firing fraction so burst → LTP, dip → LTD.

## Ranked options for the subtraction (the design fork — present before the nav build)

- **Option A′ (membrane subtraction, as built) — honest partial.** Keep the GABAergic critic→SNc projection; accept that the faithful depolarized SNc reversal gives only a weak value subtraction. This is the most brain-based (subtraction at the membrane) but the cancellation is weak/operating-point-sensitive. Could be pushed with careful joint tonic/gain calibration, but the de-risk shows it is finicky.
- **Option C (neural value, host subtraction) — the research doc's calibration-crutch fallback.** Read the striosome rate (the **validated** cue-gated neural V) and inject `I_value = −k_v·rate` as a host hyperpolarizing current. The *value* is neural + state-dependent (the important claim); only the final subtraction is host arithmetic. Sidesteps the depolarized-reversal wall; clean function; weaker brain-based claim (documented honestly).
- **Option B′ (indirect inhibition via a normal-reversal interneuron).** Route striosome→SNc through a local GABAergic interneuron with a non-depolarized E_GABA, so the value subtraction arrives with a normal hyperpolarizing reversal. More biology + more build; defers the question.

**Recommendation:** the core deliverable (the value became NEURAL + state-dependent, learned from the SNc's own δ) is **achieved**. For the nav deployment, lead with **Option C** (neural value, host subtraction) as the functional path — it inherits the validated neural value without the membrane-subtraction wall — and document Option A′ (weak membrane subtraction) as the honest biophysical limit + Option B′ as the deeper brain-based fix. This keeps nav moving while being honest that the *subtraction* (not the value) is the residual host step.

## What this de-risk bought (the point of cheap-first)

For ~CPU-minutes and zero nav build, it: (a) validated the neural value-learning mechanism multi-seed, (b) surfaced + fixed four real harness/config bugs, (c) localized the exact remaining challenge (the depolarized-SNc-GABA membrane subtraction) that the research review predicted, and (d) produced a concrete, ranked design decision for the subtraction — all before committing the nav build + the heavy 6-seed GPU gate.
