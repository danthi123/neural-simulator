---
type: finding
status: retracted
date: 2026-07-20
mechanism: wkv-cortex
---

# gap#1 M1 — the on-bridge WKV state BEATS the fair trigram (graded `cp_ssm_state`), + M2 NEF input de-risk GO

**2026-07-20.** The session-long on-bridge boundary is **SURPASSED**. Produced by following the research gate
(`2026-07-19-onbridge-wkv-state-fidelity-research-gate.md`) instead of improvising more operating points.

## The reframe that unlocked it

Every prior on-bridge realization capped at deep-NLL **−0.9 .. −1.8** vs the fair interpolated trigram. The gate's
decisive finding: our bar — *"the recurrent state must be a mean FIRING RATE"* — was **SELF-IMPOSED and stricter than
BOTH SpikeGPT AND biology**:

- **No spiking LM realizes the recurrent state as spikes.** SpikeGPT (arXiv 2302.13939), SPikE-SSM (2410.17268),
  SpikingSSMs (2408.14909), SiLIF (2506.06374) ALL keep the SSM/WKV state real-valued/graded and spike only the I/O.
- **Biology holds integrator state in graded slow conductances** — NMDA plateaus, line-attractor persistent activity
  (Seung 1996, Goldman 2003, Wong-Wang 2006), not in a short-window spike count.

So the ~0.55 spike-rate-coded ceiling (9 read/state levers + 5 line-attractor de-risks, all capping) is a verdict on a
**METHOD**, and that method was the wrong target. The genuine residual was ONE stage: **encoding the input `v_t`**.

## M1 — the mechanism (NO `sim/` edit; reuses a SHIPPED asset)

The gate's RAG-check surfaced an asset we had **overlooked for this task**: `cp_ssm_state`
(`enable_selective_ssm_state`, `sim/config.py:266` + `sim/bridge.py:343/1372/5938`, RUNG4b) — a per-neuron GRADED leaky
integrator advanced by the bridge's own step, previously validated **byte-equal to numpy (1e-7)** but never pointed at
the WKV/trigram LM task.

**Exact mapping.** The bridge update is `s = lam*s + (1-lam)*inject`, `lam = clip(1 - k_leak*(1+shunt), 0, 1)`.
Setting `k_leak = 1-decay`, `shunt = 0` gives `lam = decay`; injecting `v_t/(1-decay)` reproduces **exactly**
`a_t = decay*a_{t-1} + v_t`. DUAL-NONNEG (two non-negative channels holding the integral of `relu(±v)`) keeps it
biology-faithful (two positive conductances; no signed-difference opponency).

**Read-out.** Because the state is byte-exact, the SSM's **OWN trained read-out runs on it UNCHANGED**
(`logits = head(sigmoid(Wr·LN(emb)) * (Wo_sp · state))`, `--use-ssm-readout`). A freshly re-fit post-hoc read-out is a
WEAKER, under-fit proxy that **MASKED the result** (−1.66 with a re-fit MLP at n_fit=1500 vs **+0.077** with the
trained read-out on the identical state) — a reminder that a post-hoc reservoir read-out can hide an exact state.

## Result — GO (6-seed), anti-cheats load-bearing

`--ssm-state --use-ssm-readout`, V=200 / d=64, n_fit 1500 / n_eval 300, deep bucket d10-99.

| seed | M1 (graded state) vs trigram | memoryless control (lam=0) |
|---|---|---|
| 42 | **+0.077** | −0.665 |
| 43 | **+0.107** | −0.623 |
| 44 | **+0.238** | −0.736 |
| 100 | **+0.146** | −0.687 |
| 101 | **+0.061** | −0.894 |
| 102 | **+0.128** | −0.712 |
| **mean** | **+0.126 (6/6 GO)** | **−0.720 (6/6 collapse)** |

**6/6 seeds BEAT the fair interpolated trigram**; the memoryless control collapses on **6/6**, a mean separation of
**0.846 nats**. For reference the same SSMs off-bridge score +0.060..+0.173 — i.e. the on-bridge realization is at
**parity with the off-bridge model**, as it must be when the state is byte-exact.

**verify-first:** `corr(cp_ssm_state, numpy dual-nonneg SSM state) = 1.000` every run (the reference was corrected to
match the deployed state FORM — integral-of-relu, not relu-of-integral — so the check is a genuine equivalence test).

**Anti-cheats (both load-bearing):**
1. **MEMORYLESS** (`k_leak=1` → `lam=0`, no integration) → **−0.62 .. −0.74**, i.e. collapses ~0.7-0.9 nats below the
   GO. The temporal integration is doing the work.
2. **RATE-READ control** (the OLD firing-rate path, SAME sentences/seed) → **−0.491** (map-corr 0.681) = reproduces the
   wall. ⇒ the **GRADED delivery** is what closed it, not a harness artifact.
3. Not run on the on-bridge path: a per-position prefix-PERMUTE control (no flag; would need O(T²) restatement). The
   MEMORYLESS control is the stronger form (it removes the memory entirely rather than scrambling it), and the
   off-bridge SSM's own perm control collapses **+4.45**. Recorded honestly as not-run rather than implied.

## M1 SCALES — and the margin GROWS with scale (the bridge toward the LLM-like target)

| scale | on-bridge vs trigram | verify corr | memoryless control |
|---|---|---|---|
| V=200, d=64 (6-seed) | +0.126 | 1.000 | −0.720 |
| **V=1000, d=128** (5× vocab, 2× width) | **+0.486** | **1.000** | **−0.919** (1.4-nat separation) |

The mechanism is exact, so it transfers by construction — and the *margin over the trigram grows* as vocabulary and
width increase, which is the direction the LLM-like target needs.

**⚠️ A SILENT FAILURE CAUGHT HERE (recorded).** The first V=1000 run read **−3.790 with corr = 1.000** — an impossible
combination (an exact state + the model's own read-out must reproduce the off-bridge +0.370). Cause: my script trained
the SSM with `--n-sentences 80000` but the on-bridge runner defaults to **40000**, so it rebuilt a **different vocab** →
token ids no longer matched the trained embedding. **The state-corr could not catch it** because both the on-bridge
state and the numpy reference were computed from the *same* mismatched tokens, so they agreed perfectly while both
were wrong. Re-run with matched `--n-sentences`: **+0.486**. Lesson for this harness: *a verify metric that consumes
the same upstream input as the thing it checks cannot detect an upstream error.*

## HONEST SCOPE (what this does and does NOT claim)

- **Claims:** a multi-channel GRADED recurrent LM state runs on the `SimulationBridge`, advanced by the bridge's own
  per-step update, and **beats the fair interpolated trigram at deep context** — the SpikeGPT/biology-faithful bar.
- **Does NOT claim** the state is spike-rate-coded (no spiking LM does this; it is the wrong target).
- **Residual:** the per-token `cp_ssm_inject` is written by the host, standing in for the upstream cortical
  population's graded synaptic drive. **That is what M2 closes.**

## M2 (input via a GENUINE spiking population) — off-bridge de-risk GO

The gate ranked M2 as the theory-backed fix for the *characterized* input failure (dead-zone + non-monotone +
refractory). Measured REAL Izhikevich tuning curves on a bridge over a dense `v` sweep, then solved the decoder:

| input encoding | corr(v̂, v) | flat/dead steps |
|---|---|---|
| **NEF heterogeneous encoders + OPTIMAL least-squares decoder** | **0.9993** (monotone, near-linear) | 9/40 |
| homogeneous pool + uniform-sum decode (the OLD path) | 0.8167 | **36/40 — dead-zoned** |

Heterogeneity = distributed **intercepts** (tile the range → kills the dead-zone) + **mixed-sign** preferred directions
+ distributed gains; the decoder is per-neuron least-squares, NOT a uniform sum. (The project's earlier `--hetero-gain`
was a half-measure: heterogeneous gains but still a uniform-sum decode — which is why it only moved 0.551→0.574.)

### M2 ON-BRIDGE — 6-seed: the NEF encoding is LOAD-BEARING, but the spiking delivery is still SHORT of the trigram

Wired on-bridge (`_emerge_wkv_m2_nef_onbridge_derisk.py`): an NEF pool projects to the state channel through synapses
**whose weights ARE the decoder** (the decode happens in the synapses), the state is frozen during the encode window via
`shunt=-1` (`lam=1`) and advanced by ONE step at `shunt=0`, so the per-token update is exactly `a_t = decay*a_{t-1}+v̂_t`.
**Dale's law forced a design change:** the bridge routes exc/inh **per presynaptic neuron**, so a mixed-sign decoder is
not expressible on the substrate → NNLS **sign-constrained** decoders on an excitatory pool.

| | mean (6 seeds: 42/43/44/100/101/102) |
|---|---|
| **M1** (exact graded inject) | **+0.126 — GO** |
| **M2** (spiking NEF input, n_enc=48) | **−0.345** (range −0.207..−0.557) |
| M2 **HOMOGENEOUS control** (the old encoding) | **−0.889** (range −0.630..−1.040) |
| M2 verify corr (post-rescale, held-out) | 0.613 |

⇒ **The NEF heterogeneity + optimal decode is LOAD-BEARING: +0.544 nats over the homogeneous control, 6/6 seeds.**
⇒ **But M2 does NOT reach the trigram.** The spiking input delivery has fidelity 0.613 (vs M1's exact 1.000), and that
costs ~0.47 nats. The residual is now precisely localized: **input-delivery fidelity**, not the state and not the read-out.

**⚠️ SELF-CAUGHT OVERCLAIM (recorded, per verify-first):** an n_eval=60 smoke of M2 read **+0.118 "GO"** — with only
**30 deep tokens**. The proper n_eval=250 run is **−0.319** on that same seed. A 30-token deep bucket is noise; the smoke
GO was a small-sample artifact and was retracted before it entered any claim. (Two further harness defects were caught
in the same pass: the per-channel gains were being fit **on eval sentences** against the reference — a leak, now
train-only — and the verify corr was measured **pre-rescale**, so it could not validate what actually feeds the
read-out; now post-rescale on held-out.)

### FOUR hypotheses for the ~0.6 input-fidelity ceiling — three REFUTED, one partial, all by direct measurement

| hypothesis | test | verdict |
|---|---|---|
| NEF error scales ~1/N (need ~100 neurons/dim) | n_enc 48→96→192→384 | **REFUTED** — fidelity got *worse* (0.615→0.578→0.551→0.594) |
| recurrent cross-talk inside the encoder pool | `pool_density` 0.05→0.001 | **REFUTED** — essentially identical (0.615→0.614) |
| decoder fit on the wrong basis (flat rate vs the deployed leaky CONDUCTANCE, `decay_e`=0.8187/step) | fit on the `g = g*decay_e + input` recursion | **marginal** (0.615→0.620) |
| **window quantization** (few spikes/neuron in a 6-step window) | t_step 6→12→24→48 | **CONFIRMED then BOUNDED** (below) |

**Window sweep (t_step at dt=1ms IS the per-token integration time):**

| t_step | fidelity | σ_rel | vs trigram |
|---|---|---|---|
| 6 | 0.620 | 0.771 | −0.276 |
| 12 | 0.677 | 0.727 | −0.266 |
| 24 | **0.786** | **0.622** | **−0.181 (best)** |
| 48 | 0.738 | 0.627 | −0.405 |

**⚠️ MY EXTRAPOLATION WAS REFUTED (recorded).** From the monotone 6→24 trend I predicted fidelity would keep rising and
cross zero near t_step≈150-200 (≈ normal speech rate) — and I queued a 96/192 run on that basis. **t_step=48 broke the
trend** (fidelity 0.786→0.738, NLL −0.181→−0.405), so the 96/192 run was KILLED rather than burn GPU on a dead
hypothesis. **Mechanism of the peak:** the read is the excitatory conductance with `tau_e`=5 ms, so it only "sees" the
last ~15 steps — a longer window adds NO information to the read while accumulating Izhikevich **spike-frequency
adaptation** (`cp_recovery_variable_u`, the same accumulation that bit EMERGE-61), which distorts the rate code.
⇒ **the optimum is a window long enough to sample but short enough to precede adaptation (~24 steps).**

### M3 (learn-through-the-substrate co-adaptation) — closes 83% of the remaining gap, but does NOT cross

Rather than fight the delivery noise, TRAIN THROUGH IT: the model is trained with input noise so the input map and
read-out co-adapt to the actual spiking delivery (the gap#4 lever), then deployed on the SAME spiking NEF path
(t_step=24, n_enc=48).

| co-adaptation noise | off-bridge (capability retained) | ON THE SPIKING INPUT PATH |
|---|---|---|
| none (naive M2) | +0.113 | −0.181 |
| 0.3 | +0.111 | −0.104 |
| 0.6 | +0.089 | −0.087 |
| 1.0 | +0.086 | −0.052 |
| **1.5** | +0.062 | **−0.030 ← PEAK** |
| 2.0 | +0.038 | −0.060 |
| 3.0 | +0.019 | −0.099 |

**⛔ RETRACTION (2026-07-20, blind-seed check) — the curve above is SEED 42 ONLY and its structure is BELOW the
cross-seed noise floor. I originally wrote "a clean inverted-U with a genuine optimum at σ≈1.5 … a real optimum, not a
tuning plateau." That claim is NOT SUPPORTED and is withdrawn.**

Re-running the σ=1.5 "peak" config on blind seeds:

| seed | off-bridge | ON THE SPIKING INPUT PATH |
|---|---|---|
| 42 (the tuned/dev seed) | +0.062 | −0.030 |
| 43 | +0.002 | −0.066 |
| 44 | +0.081 | **+0.080** |
| 100 | +0.022 | −0.124 |
| **mean** | +0.042 | **−0.035**, **spread 0.204** |

**The effect sizes I was tuning (~0.03–0.05 nats) are ~4-6× SMALLER than the cross-seed variance (~0.2 nats).** So the
entire M3 lever sweep — the monotone improvement, the peak, the inverted-U — was operating *below the noise floor* on a
single seed. 1 of 4 seeds actually crosses (+0.080); the mean is −0.035. Off-bridge capability at σ=1.5 is also
seed-fragile (+0.002 on seed 43 = nearly destroyed).

**What this changes:** the honest characterization of the fully-spiking-input path is **"at parity with the fair
trigram, seed-variance-dominated, straddling zero"** — NOT "short by 0.03", and NOT "co-adaptation has an optimum at
σ=1.5". Any future work here must pre-register seeds and report blind seeds separately, because the margin of interest
is smaller than the seed noise at this scale.

**What SURVIVES this correction (both well above the noise floor):**
- **M1's 6-seed GO** — +0.126 mean with **all 6 seeds positive**, and **+0.486** at V=1000. Effect ≫ variance.
- **M2's heterogeneity contrast** — +0.544 nats over the homogeneous control, **6/6 seeds**. Effect ≫ variance.

**SHARPENED (the variance is across MODEL INSTANCES, not across the lever).** Re-testing at the *mild* setting
(σ=0.6) on the same blind seeds separates the two explanations:

| seed | σ=0.6 (mild) | σ=1.5 (aggressive) |
|---|---|---|
| 43 | −0.211 | −0.066 |
| 44 | **+0.077** | **+0.080** |

Seed 44 **crosses at BOTH** noise levels; seed 43 **fails at both**. So the spread is not the co-adaptation
hyperparameter — it is *which trained model instance you got*. Some models transfer to the spiking-input path and beat
the fair trigram; others do not, and the co-adaptation lever does not decide it. ⇒ the honest statement is:
**the fully-spiking-input path beats the trigram for SOME model instances and not others, with instance variance
(~0.2 nats) dominating every lever tested (~0.03-0.05).** Chasing the lever further is below the resolution of the
experiment; the productive question, if resumed, is *what distinguishes a transferable instance* — not which σ to use.

This is the project's documented recurring failure mode (dev-seed selection) caught in my own work, by the control that
exists to catch it. It cost four extra runs and prevented a false characterization entering the record.

**M4 — combining the best of every lever does NOT cross either.** The encoder sweep had previously been run *without*
co-adaptation and at the *bad* window, so the combination was genuinely untested. Run at the co-adaptation peak
(σ=1.5) + the best window (t_step=24) + low pool density: n_enc **48 → −0.044**, **144 → −0.075**, **288 → −0.258**.
Encoder scaling *hurts* monotonically in every configuration tried — the 1/N expectation is refuted a third time, now
under the most favourable conditions available. The best result remains the base config at **−0.030**.

⇒ **All levers WITHIN the M2 design are now exhausted** (population, cross-talk, calibration basis, window,
co-adaptation, and all combinations). The
spiking input delivery tops out at fidelity **0.786** / σ_rel **0.622**, costing ~0.18-0.3 nats — real, bounded, and
precisely quantified. **The remaining fix is M3: stop fighting the noise and TRAIN THROUGH IT** — co-adapt the input map
and read-out to the *measured* delivery noise so the model is robust to it. That is precisely the
**gap#4 learn-through-the-substrate lever**, and the calibration it needs (σ_rel ≈ 0.62) is now measured.

## Process note

This is the research gate working as designed: it (a) killed the wrong target, (b) **prevented re-deriving** the
line-attractor (already run to a 5-de-risk verdict — all capping ≤0.55 with a rate read), and (c) surfaced a shipped
asset we had overlooked. The prior session's exhaustive operating-point/population sweeps were a genuine, honestly
recorded characterization — of a method that could not work.
