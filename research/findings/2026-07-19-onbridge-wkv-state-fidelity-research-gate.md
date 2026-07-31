---
type: finding
status: contributing
date: 2026-07-19
mechanism: wkv-cortex
---

# Research gate — carrying a HIGH-FIDELITY, MULTI-CHANNEL, GRADED recurrent LM state on the spiking substrate (on-bridge WKV/SSM)

**2026-07-19. READ-ONLY deep-research gate.** Fired by the multiply-confirmed boundary in
`2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-6seed-GO-...md`: the off-bridge WKV/SSM leaky recurrent LM
state `a_t = decay·a_{t-1} + v_t` beats a fair interpolated trigram at deep context (+0.10..+0.92 across scales),
but **every ON-bridge realization caps BELOW the trigram** (deep-NLL −0.9 to −1.8). The root cause was just
precisely measured: the multi-channel state is delivered via an **input pool whose few-spike firing rate**
(0–22 spikes over `T_STEP=6`) is a noisy, quantized, threshold-nonlinear, refractory-bounded, **dead-zoned**
map of each channel's value `v_t` — the documented **point-neuron rate-code wall**, now on the recurrent LM state.

> **BOTTOM LINE (verdict up front).** The trigram-beating multi-channel recurrent state **IS achievable on this
> point-neuron substrate, cheaply, and NOW** — but not as a *spike-rate-coded* state. The wall is a verdict on
> **ONE method (state and/or input delivered as a mean firing rate)**, not on the capability. Holding the state
> is already solved as a GRADED analog quantity (the entire spiking-LM literature AND biology do exactly this;
> the project already has `cp_ssm_state` [byte-equal to numpy] and `cp_conductance_g_graded_plateau` [0.98 for a
> clean input]). The genuine residual is one stage — **encoding the D-channel input `v_t` into the substrate** —
> and it has three well-understood escapes, cheapest-first below. A spike-*rate*-coded state is genuinely bounded
> at ~0.55 (Mikulasch–Priesemann), but that bar is **stricter than SpikeGPT and stricter than biology** and
> should be abandoned as the wrong target.

---

## (1) RAG-CHECK — what we have ALREADY concluded (avoid re-deriving)

### 1a. The on-bridge WKV arc is EXHAUSTIVELY characterized (source: `2026-07-19-gap1-WKV-...RUNG1a...md`, 594 lines)
The whole arc lives in ONE finding doc. Load-bearing conclusions, verbatim:

- **The state=firing-rate bar is SELF-IMPOSED and caps at ~0.55–0.57 (9+ levers, all exhausted).**
  > "NINE mechanisms (co-adapt, population, heterogeneous-population, read-window, conductance-read, latency,
  > feedforward-conductance, decay-match, scale) ALL cap at ~0.55-0.57. A >0.8 graded state on point neurons
  > genuinely requires a true graded persistent-activity INTEGRATOR ... OR end-to-end surrogate-BPTT ..."
  > — and the decisive reframe: *"SpikeGPT (Zhu et al. 2023 ...) does NOT hold the WKV state on spikes. Its
  > Spiking-RWKV keeps the leaky recurrence ... in REAL-VALUED FP32 across all timesteps ... So our 'state = mean
  > firing RATE' constraint is STRICTER than the SOTA spiking LM AND stricter than biology ... and it is the
  > DIRECT, SOLE cause of the ~0.55 floor."*

- **The dendritic GRADED PLATEAU already SURPASSES the point-neuron limit for a CLEAN input (0.98), but the FULL
  multi-channel WKV port does NOT transfer** — the bound is the INPUT DELIVERY, not the state hold:
  > "the plateau conductance tracks a leaky integral at corr **0.980** (decay-matched), vs the ~0.55 point-neuron
  > ceiling — a DECISIVE surpass." … then verify-first: "the plateau CORE-MECHANISM 0.98 does NOT transfer to the
  > full-WKV DEEP-NLL … the FULL WKV port loses it" … and finally: "the on-bridge multi-channel WKV state at
  > trigram fidelity is BOUNDED by the input-pool rate-code … hits the documented POINT-NEURON RATE-CODE WALL."

- **The DEMONSTRATED gap#1 close already exists off-substrate** (this is important — the *capability* is not open,
  only the strict on-substrate realization):
  > "the SpikeGPT-consistent architecture (graded local state + spike-coded output) beats the fair trigram at
  > scale (mean **+0.26, 6/6**) … GENERATES coherent open prose." (`--spike-output`, 6-seed GO, CI-guarded.)

- **The end-to-end fix precondition is MET off-bridge, and the exact remaining gap is named:**
  > "End-to-end training through the EXACT plateau transfer (`--plateau-exact`) BEATS the fair trigram OFF-bridge
  > at all 4 operating points (+0.111/+0.105/+0.104/+0.123) … BUT the on-bridge DEPLOY … is NEGATIVE: deep-NLL
  > **−1.169** … ROOT CAUSE … the on-bridge transfer input is `pathway_w * firing_rate` where
  > `firing_rate = f-I(drive_scale * relu(v))` — a SATURATING input-pool f-I … The off-bridge GO trained the
  > input map for a transfer input (relu(v)) that the on-bridge chain does NOT deliver."

### 1b. WAS the line-attractor population integrator actually RUN, or only scoped? — **RUN, to a decisive verdict (5 de-risks), but only with a POPULATION-RATE read.**
`_build_recur_channel_bridge` (`_emerge_wkv_onbridge_derisk.py:126`, `--recur-integrator`) is fully implemented and
was executed to **five** de-risks. Verbatim verdicts from the same finding doc:

- de-risk #1 (fixed sweep): *"corr stays 0.555 FLAT across recur_w 0.2→6.0 … the recurrence contributes negligibly."*
- de-risk #2 (transient-kick): *"corr 0.526, STILL FLAT … the population fires briefly on the kick then goes QUIET …
  precisely Seung-1996's famous LINE-ATTRACTOR FINE-TUNING PROBLEM."*
- de-risk #3 (tonic+high-recur): *"too-weak recurrence → NO effect (0.55, flat); strong-enough-to-matter →
  DISTORTION (0.32-0.40) … a knife's edge BETWEEN these that coarse parameter sweeps miss."*
- de-risk #4 (fine sweep): *"corr 0.44-0.49 — ALL below the 0.55 self-NMDA baseline … a FIXED-weight recurrent
  population integrator does NOT beat a single clean self-NMDA leaky conductance."*
- de-risk #5 (Hebbian-learned): *"also converges to corr 0.441 … even a LEARNED recurrent attractor does NOT beat
  the clean single-cell self-NMDA."*

**Critical nuance for this gate:** every line-attractor de-risk **read the population MEAN FIRING RATE** (the rate
code that hits the wall) and measured only `corr(state, rate-SSM)` — never a deep-NLL LM verdict, and never a
**conductance** read of the attractor's held value. So the line attractor is retired **as a rate-read state**, but
its combination with a graded-conductance read was never tested (see M4 — deprioritized, with reasons).

### 1c. A SECOND, cleaner on-bridge graded-state path ALREADY EXISTS — `cp_ssm_state` (RUNG4b), but it never met the trigram
The `2026-07-13-PAST-RESERVOIR-RUNG4b-*` arc added a **byte-identical-when-off additive `sim/` edit**
`enable_selective_ssm_state` (`sim/config.py:266`, `sim/bridge.py:343,5938`): three per-neuron FP32 arrays
`cp_ssm_state`/`cp_ssm_inject`/`cp_ssm_shunt` and one self-contained per-step block
`s = clip(1−k(1+shunt),0,1)·s + (1−lam)·inject`. Verbatim conclusions:

- *"the on-bridge `cp_ssm_state` mechanism is the SAME dynamical object as the validated numpy selective SSM …
  Max abs diff ~1e-7 (float32 round-off) on all 6 seeds → byte-equivalent."* (RUNG4b-iii-a)
- *"the transport-free selective diagonal SSM … now LEARNS end-to-end while its state lives on the spiking
  SimulationBridge … reproducing the validated numpy result exactly."* (RUNG4b-iii-b, 5/6 GO)
- **but:** *"the gate (which sets `cp_ssm_shunt`) and the read-out are currently host-computed by the runner"* and
  **no trigram comparison exists in any RUNG4b doc** (it beats a *fixed reservoir* on a retention task; the
  language-retention transfer even carries a ⚠️ SCOPE CORRECTION NEGATIVE for pure-retention-≠-prediction).

So: **a multi-channel graded leaky SSM state, byte-equal to numpy, already runs on the bridge** — it was simply
never pointed at the WKV/trigram LM task. This is the missing cheap experiment (M1).

### 1d. The dendritic plateau `sim/` edit — what it is
`fused_graded_dendritic_plateau` (`sim/kernels.py:289-338`) computes
`V = max(sigmoid(slope·(c_weighted − center)) − floor, 0)` → dual-exp NMDA kinetics → Jahr-Stevens Mg²⁺ block →
current toward E_e, writing `cp_conductance_g_graded_plateau` (`sim/bridge.py:6621-6633`). **The input `c_weighted`
is `Σ_j w_j · cp_prev_firing_states_j`** — a firing-rate-weighted sum. That IS the input-delivery wall: the plateau
integrates its input at 0.98 fidelity, but the input it is handed is a few-spike rate code. Validated only for **ONE**
graded channel to date (a scalar nav value); never for D simultaneous channels.

### 1e. Sibling walls already documented
- **Opponency / common-mode SNR wall** (`2026-06-05-B-...CONFIRMED.md`): *"the retina removes the common mode with
  GRADED signals before action potentials (Kandel p543) precisely because spike rates cannot."* Relevant because a
  **signed** WKV state on a non-negative conductance = a small signed difference of two large correlated integrals
  = this exact wall. **Already bypassed** in the arc: `--nonneg-state`/`--dual-nonneg` (a non-negative WKV state
  still beats the trigram +0.41–0.48), so the state need not be signed.
- **Off-diagonal recurrent credit** (`2026-07-16-MDGL-...DECISIVE-NEGATIVE.md`): doesn't cleanly spike-realize on
  point neurons; the biological path is **replay (SWR-replaces-BPTT)**, not an online dendritic op.

### RAG verdict
**Genuinely concluded / do NOT re-run:** state-as-mean-firing-rate (0.55 ceiling, 9 levers); fixed & Hebbian-learned
line-attractor with a *rate* read (5 de-risks, all ≤0.55); `--read-gnmda`, latency, feedforward spike-charged
conductance, naive scale-up (all NEGATIVE); the off-bridge SpikeGPT-faithful capability (DONE, 6-seed GO).
**Genuinely OPEN:** (i) the WKV state held in the EXISTING `cp_ssm_state`/plateau graded conductance, pointed at the
**trigram LM task** (never run); (ii) a **clean multi-channel INPUT ENCODING** of `v_t` into that graded state
(the localized wall); (iii) **end-to-end training through the CALIBRATED (measured) substrate transfer** (precondition
met off-bridge; deploy failed only on transfer mismatch).

---

## (2) DIAGNOSIS — is it achievable on THIS substrate, and by what mechanism class?

**Yes — because the problem was mis-scoped, and the residual is now localized to a single, well-understood stage.**

The task decomposes into three stages: **encode** `v_t` → **integrate/hold** the leaky state → **read** for the
next token. The arc's own data shows the hold and read are solved and the wall is entirely at the encode:

| stage | on-substrate status | evidence |
|---|---|---|
| **HOLD** the graded multi-channel state | **SOLVED, biologically-correct as GRADED** | `cp_ssm_state` byte-equal to numpy (1e-7); graded plateau 0.98 for a clean input; **and the ENTIRE spiking-LM literature holds it graded** (below) |
| **READ** the state → next token | solved when the state is faithful (post-hoc ridge/MLP + receptance) | `--exact-state`: given the exact state, corr-0.795 read → deep-NLL −0.34 (near trigram); off-bridge `--plateau-exact` GO +0.11 |
| **ENCODE** `v_t` → substrate drive | **THE WALL** (few-spike rate code: dead-zone + non-monotone + refractory) | fi_probe: 0–22 spikes/window; pop_k→500 stays non-monotone with a dead-zone below relu(v)~1 |

**The literature is unanimous and decisive: no spiking LM/SSM realizes the recurrent state as spikes.** All keep it
real-valued/graded and spike only the I/O — triple-confirmed this session:
- **SpikeGPT** (Zhu et al. 2023, arXiv 2302.13939): the Spiking-RWKV WKV recurrence is FP32 across timesteps; the
  Heaviside spike applies to the block OUTPUT only.
- **SPikE-SSM** (arXiv 2410.17268): *"The SSM hidden state remains real-valued float … the membrane potential u_t
  serves as the graded interface between analog SSM and binary output … Only the final output is converted to spikes."*
- **SpikingSSMs** (arXiv 2408.14909): *"the SSM hidden state (h_t) remains real-valued float. Only the output is
  spiked … y_t = C·h_t is treated as the input current of the neuron."*
- **Binary-S4D / SiLIF** (arXiv 2506.06374): SiLIF reparametrizes the LIF membrane to *be* the S4 state — i.e. the
  graded membrane IS the SSM state, spikes are the output nonlinearity.

And **biology holds integrator state in graded slow conductances**, not spike rates: NMDA plateaus, Wong–Wang 2006
recurrent-NMDA attractors, Seung-1996 / Goldman-2003 line-attractor persistent activity, Lisman–Idiart WM buffers.
A graded slow conductance charged by presynaptic spikes and read by driving downstream spikes **is** "neurons +
synapses + communication" — the project's BRAIN-BASED standard. Spikes carry the token I/O; the local integrator
state is analog. This is the same conclusion the project reached independently for the FHRR composer (analog
pre-spike computation) and the opponency wall.

**Mechanism class that works:** a **graded analog input delivery** (a smoothly-filtered synaptic conductance
∝ `v_t`, NOT a 6-step spike count) into a **graded analog integrator state** (`cp_ssm_state` or a leaky dendritic
plateau conductance), with spikes carrying only the token in/out. This is the SpikeGPT-faithful bar, realized on
the bridge with the state resident in a real bridge conductance array.

**Why the WKV is EASIER than a classic neural integrator:** a line/ring attractor is hard because it must SELF-SUSTAIN
a graded value across silence at α≈1 (Seung's knife-edge fine-tuning). **The WKV state is DRIVEN every token** and
decays with a fixed, known `decay` — so it needs only a **plain leaky conductance with a set tau** (τ = −T_STEP·dt/ln decay),
which the substrate already provides (NMDA tau, `graded_plateau_tau_decay_ms`, `ssm_k_leak`). No attractor tuning is
required. The 5 line-attractor de-risks were solving a harder problem than the WKV poses.

---

## (3) RANKED cheap-first mechanisms to de-risk next

### M1 — WKV on the EXISTING `cp_ssm_state` graded integrator, GRADED-conductance input  ⭐ cheapest, ~1 session, likely GO
**What it is.** Reuse the shipped `enable_selective_ssm_state` edit. Per token: set `cp_ssm_shunt = 0` (uniform
decay via `ssm_k_leak`, matched to the WKV `decay`), set `cp_ssm_inject = v_t` **as a graded quantity** (the
smoothly-filtered synaptic drive from the upstream population — in the one-brain system, the stream-cortex output's
graded conductance; here, the host `Wv@LN(emb)`, exactly the status the input already has). The bridge holds
`a_t = decay·a_{t-1} + v_t` in `cp_ssm_state` at FP32 fidelity. Read `cp_ssm_state`, apply the trained WKV read-out.
The state region's neurons still spike for the OUTPUT/downstream; the STATE is a graded conductance (an NMDA/Ca
plateau — the biological answer).

**Why it beats the rate-code wall.** It does **not encode `v_t` as a spike count** — it delivers `v_t` as a graded
synaptic quantity (biologically the graded current an upstream cortical population produces; the "6-step count" was
the artifact). The state IS the numpy SSM state (corr 1.0), which beats the trigram +0.6..+0.9 at the rate level —
so on-bridge it should beat it cleanly. This is the honest, biologically-legitimate close of "the graded
multi-channel recurrent state runs on the bridge and beats the trigram."

**Reusable machinery.** `enable_selective_ssm_state` + `cp_ssm_state/inject/shunt` (shipped, byte-identical off);
the RUNG4b runners (`_reslm_rung4b_iii*`); the WKV read-out + trigram harness in `_emerge_wkv_lm_derisk.py` /
`_emerge_wkv_onbridge_derisk.py` (Vocab / load_sentences / fit_interp_trigram / `_bucket` / `_feat`).

**Cheapest de-risk experiment.** Add a `--ssm-state` path to `_emerge_wkv_onbridge_derisk.py` that (a) builds a
bridge with `enable_selective_ssm_state`, (b) each token writes `cp_ssm_inject = v_t` (both signs, or the validated
`--dual-nonneg` pair so it stays non-negative), `cp_ssm_shunt = 0`, steps once, reads `cp_ssm_state`; (c) fits the
existing read-out; (d) reports deep-NLL vs the fair trigram. n_eval=200, single seed smoke → 3-seed → 6-seed.

**Anti-cheat controls (all required):**
1. **verify-first:** `corr(cp_ssm_state trajectory, numpy rate-SSM state) > 0.99` before any GO (it should be ~1.0).
2. **memoryless collapse:** set `ssm_k_leak` so `lam_eff→0` (no integration) → must collapse to ≈ bigram.
3. **perm collapse:** shuffle the prefix token order → the deep-context margin must collapse (+several nats).
4. **rate-read control (load-bearing):** the OLD input-pool firing-rate path on the SAME sentences must reproduce
   the ~−0.9 to −1.8 wall → proves the graded-conductance delivery (not some harness artifact) is what closed it.

**Honest scope / what it does and does NOT claim.** It closes "**a graded multi-channel recurrent LM state runs on
the SimulationBridge and beats the fair trigram at deep context**" — the SpikeGPT/biology-faithful bar. It does
**not** claim the state is *spiking-population-coded* (no spiking LM does this; it is the wrong target). The input
`cp_ssm_inject` is graded-synaptic; making that input a genuine **presynaptic-spike-driven synaptic conductance from
the upstream stream-cortex** is M2, and a spiking **read-out** (FS-WTA, already validated) is a small follow-on. M1
is the decisive cheap win that converts the arc's "on-bridge caps below trigram" into "on-bridge beats trigram."

---

### M2 — NEF heterogeneous-encoder + optimal-decoder input population  (moderate, ~1–2 sessions)
**What it is.** If the mission insists `v_t` be delivered by a genuine SPIKING input population (not a graded
injection), build the input pool the way the Neural Engineering Framework prescribes: per channel, ~64–128 neurons
with **heterogeneous encoders** — distributed gains, distributed intercepts/biases (so tuning curves TILE the `v_t`
range), and mixed-sign preferred directions — and set the `inp→state` synaptic weights to the **least-squares
OPTIMAL DECODE weights** (fit offline against a `v_t` sweep) instead of a uniform `pathway_w`. Then the postsynaptic
drive `c_weighted ≈ Σ_j d_j · rate_j` is a clean linear estimate of `v_t`, which the plateau/`cp_ssm_state`
integrates faithfully.

**Why it beats the wall.** The project's *characterized* failure — dead-zone below relu(v)~1, non-monotone,
refractory-bounded — **IS the homogeneous-population + uniform-decode failure mode.** NEF's core theorem is that
**heterogeneous tuning curves + an optimal linear decoder cancel the individual-neuron threshold nonlinearities**,
giving ~1% representation RMSE with ~100 neurons/dimension (Eliasmith & Anderson 2003, *Neural Engineering*). Two
mechanisms combine: distributed intercepts kill the dead-zone (some low-threshold neurons fire for small `v`); the
optimal decoder (per-neuron weights, NOT a uniform sum) linearizes the population response and averages spiking
noise ~1/√N. The project's `--hetero-gain` was a **half-measure** — heterogeneous gains but still a **uniform-sum
decode** — and only lifted 0.551→0.574. The **full** recipe (heterogeneous encoders AND the least-squares decoder,
plus distributed intercepts, plus signed encoders) is UNTRIED and is exactly the theory that predicts the observed
failure.

**Reusable machinery.** The plateau/`cp_ssm_state` bridge builders; the `--hetero-gain` scaffold (extend to
distributed intercepts + mixed-sign encoders); an offline least-squares decoder fit (numpy). Optionally Nengo's NEF
solver as an off-line reference for the decode weights (do NOT import at runtime; reference only).

**Cheapest de-risk (OFF-bridge FIRST).** Simulate the heterogeneous input pool's f-I over a dense `v_t` sweep, solve
the least-squares decoder, and check `c_weighted(v)` vs `v` is **linear, monotone, dead-zone-free (corr > 0.95)**.
Only if that passes, wire the decode weights as the `inp→chan` synapses and run on-bridge deep-NLL vs trigram.

**Anti-cheat controls:**
1. **homogeneous-pool control** reproduces the dead-zone/non-monotone map → heterogeneity is load-bearing.
2. **uniform-sum-decode control** reproduces the ~0.57 cap → the optimal decoder is load-bearing.
3. **held-out decode fit:** fit the decoder on one `v_t`/token set, test the transfer on a disjoint set (no
   train-on-test).
4. perm + memoryless collapse on the final LM deep-NLL.

---

### M3 — END-TO-END surrogate-gradient BPTT through the CALIBRATED substrate transfer  (deep; gap#1↔gap#4-convergent)
**What it is.** Measure the ACTUAL on-bridge chain transfer (input-pool f-I ∘ plateau/state) at the deploy operating
point, bake the **measured** f-I into a differentiable surrogate (`sim/surrogate_grad.py` + `sim/bptt_snn_gpu.py`),
train the WKV (input map + read-out) end-to-end through it, and deploy on-bridge with matching parameters. This
co-adapts the input map to keep `v_t` in the un-saturated faithful range and makes the read robust to the real
quantization/refractory/noise. e-prop (Bellec et al. 2020, *Nat. Commun.*, "A solution to the learning dilemma for
recurrent networks of spiking neurons") is the biologically-plausible eligibility-trace version of this
training-through-spikes — and is exactly the **gap#4 deep-credit lever**, so M3 advances both gaps at once.

**Why it should work (precondition already met).** The arc's `--plateau-exact` (end-to-end through a GENERIC plateau
transfer) already went **GO off-bridge (+0.10..+0.12, all 4 configs)**; the on-bridge deploy failed **only** because
the surrogate transfer (`relu(v)`) ≠ the measured saturating input-pool f-I (`f-I(drive_scale·relu(v))`). The named
fix is precisely to **calibrate the surrogate to the measured f-I** — a system-identification step already begun
(steady-state transfer measured: relu(v) 0.05..8 → plateau 0..1337).

**Reusable machinery.** `sim/surrogate_grad.py`, `sim/bptt_snn_gpu.py`, the `--plateau-exact`/`--input-noise`/
`--plateau-surrogate` scaffolds, the `fi_probe`, the measured transfer curve.

**Cheapest de-risk.** (1) System-id the FULL per-token transfer (transient over T_STEP, not just steady-state) at
the deploy operating point; (2) retrain `--plateau-exact` with the CALIBRATED transfer off-bridge (must stay GO);
(3) deploy, deep-NLL vs trigram, 3→6 seed.

**Anti-cheat controls:** perm + memoryless; a **generic-transfer control** must reproduce the −1.17 deploy failure
(the calibration is load-bearing); verify-first corr on the on-bridge state.

---

### M4 — Line-attractor population with a CONDUCTANCE read  (DEPRIORITIZED — do not build unless the mission needs drift-free hold across silence)
**Why deprioritized.** (a) The 5 prior line-attractor de-risks read the population **rate** and capped ≤0.55; a
conductance-read version is **redundant** with the plateau/`cp_ssm_state`, which already hold the value at 0.98/1.0
without any attractor. (b) A self-sustaining line attractor solves **persistence across silence at α≈1** — Seung's
knife-edge fine-tuning problem, confirmed empirically over 4 de-risks — a problem the **WKV does NOT have** (it is
driven every token and decays with a set tau). (c) It is the highest-cost, lowest-marginal-value option. Only
revisit if a future capability needs a genuinely drift-free hold across many silent steps (e.g. a discourse-level
WM latch); then a Goldman-2003 tuned-feedback / bistable-dendrite integrator or the theta-gamma Lisman–Idiart buffer
(already validated on-bridge in EMERGE-85/86) is the right machinery, not this fixed/Hebbian recurrent population.

---

## (4) VERDICT — surpassable, and how cheaply

**SURPASSABLE — cheaply and now.** The capability ("a multi-channel graded recurrent LM state that beats the fair
trigram at deep context, running on the SimulationBridge") is achievable via **M1** on already-shipped machinery in
~1 session, and should GO because `cp_ssm_state` is byte-equal to the numpy SSM state that already beats the trigram.
This is the biologically-faithful and field-standard realization: **the state is a graded slow conductance; spikes
carry the I/O** — exactly what SpikeGPT, SPikE-SSM, SpikingSSMs, SiLIF, and biology (NMDA plateaus, line-attractor
persistent activity) all do. The stricter "input delivered by a genuine spiking population" bar is surpassable via
**M2** (NEF heterogeneous-encoder + optimal-decoder — the precise, theory-backed fix for the characterized
dead-zone/non-monotone failure, moderate build, off-bridge-first). The strictest "whole chain co-adapted on the real
substrate" bar is **M3** (calibrated end-to-end surrogate-BPTT; off-bridge precondition already GO; converges with
gap#4).

**GENUINELY BOUNDED — and correctly abandoned.** A **spike-rate-CODED** recurrent state (state = mean firing rate on
point neurons) is bounded at ~0.55–0.57 — the Mikulasch–Priesemann point-neuron limit, exhaustively confirmed (9
read/state levers + 5 line-attractor de-risks, all cap). But per the mission law, **this is a verdict on a METHOD,
not the capability**: it is stricter than SpikeGPT AND stricter than biology, both of which keep the state graded.
The right move is to STOP pursuing a spike-rate-coded state and realize the state as the graded analog conductance
it should be (M1), delivering the input cleanly (M2), and — for a robust, fully-co-adapted on-substrate result —
training through the calibrated substrate (M3).

**Recommended sequence:** M1 (cheap, closes "beats trigram on-bridge") → M2 (closes "input via a genuine spiking
population") → M3 (robust end-to-end + gap#4). Each is verify-first, each has a load-bearing control that reproduces
the wall when the mechanism is ablated, each multi-seed before any generalization claim. No `sim/` edit is needed
for M1 (reuses `enable_selective_ssm_state`) or M2 (drives/reads public arrays); M3 reuses the surrogate-grad stack.

---

### Sources (external, cited above)
- Zhu et al. 2023, **SpikeGPT**, arXiv:2302.13939 — Spiking-RWKV; WKV state FP32, spike-coded output only.
- **SPikE-SSM**, arXiv:2410.17268 — SSM hidden state real-valued float; membrane = graded interface; output spiked; PMBC parallel reset.
- **SpikingSSMs**, arXiv:2408.14909 — S4D state float; `y=C·h` as neuron input current; surrogate dynamic network for parallel reset.
- **SiLIF**, arXiv:2506.06374 — LIF membrane reparametrized to BE the S4 state (graded membrane = SSM state).
- Eliasmith & Anderson 2003, **Neural Engineering** (MIT Press) — NEF: heterogeneous encoders + optimal linear
  decoder; ~100 neurons/dim → ~1% RMSE; representation error decreases with N.
- Bellec et al. 2020, **e-prop**, *Nat. Commun.* 11:3625 — eligibility-trace training of recurrent SNNs, BPTT
  approximation without backprop-through-time (the biological deep-credit lever = gap#4).
- Seung 1996 (*PNAS*) line attractor; Goldman et al. 2003 robust persistent activity (tuned feedback / bistable
  dendrites); Wong & Wang 2006 (*J. Neurosci.*) recurrent-NMDA integrator; Aksay/Goldman 2014 (*Nat. Neurosci.*)
  optogenetic oculomotor integrator — graded persistent activity held in slow conductances, α≈1 fine-tuning.
- Mikulasch, Priesemann et al. — dendritic/analog pre-spike computation the point-neuron soma cannot do (the
  project's documented point-neuron limit).
