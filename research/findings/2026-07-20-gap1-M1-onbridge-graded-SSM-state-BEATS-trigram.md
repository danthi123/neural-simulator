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
⇒ a genuine SPIKING input population CAN deliver `v_t` cleanly. Next: wire it on-bridge (pool → decoder-weighted
synapses → `cp_ssm_inject`), freezing the state during the encode window via `shunt=-1` (`lam=1`) so the per-token
update stays exactly `a_t = decay*a_{t-1} + v̂_t`.

## Process note

This is the research gate working as designed: it (a) killed the wrong target, (b) **prevented re-deriving** the
line-attractor (already run to a 5-de-risk verdict — all capping ≤0.55 with a rate read), and (c) surfaced a shipped
asset we had overlooked. The prior session's exhaustive operating-point/population sweeps were a genuine, honestly
recorded characterization — of a method that could not work.
