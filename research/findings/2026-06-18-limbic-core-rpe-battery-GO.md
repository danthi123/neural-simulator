# Shared limbic core (reward/value/dopamine) — standalone organ, Schultz RPE battery GO 6/6 (2026-06-18, CYCLE 206)

## Headline

The cheap-first de-risk for **roadmap item #1** (the shared reward/value/dopamine limbic core, the
single highest-leverage step of the TRUE-ONE-BRAIN directive) is **GO 6/6**. A minimal spiking
limbic organ — `reward_us` (a PPN-like reward afferent) → `snc` (dopamine) ← `striosome_value` (a
GABAergic value critic, subtracting V through the GABA_B/GIRK conductance) + the `dopamine`
signed-firing neuromodulator — computes a correct reward-prediction error **δ = r − V entirely in
spikes**, passing all four canonical Schultz signatures across six seeds plus the two decisive
lesions. `r`, `V`, and `δ` are all neural; the lesions prove neither is a re-hidden host scalar. The
organ is now validated to lift onto the merged "one brain" (which currently has **no limbic core at
all**).

## Why this de-risk mattered (the load-bearing gap)

The audit (`2026-06-18-full-spikeification-shared-substrate-roadmap.md`) established that the merged
bridge (`nav_conv_merged_bridge.py:506`) calls `build_bg_brain_regions` with **default kwargs** — so
the consolidated brain has the basal-ganglia actor + the conversational cortex co-resident but **no
reward, no value, no dopamine**. The spiking limbic organs exist + are validated, but only in the
*standalone* `g11_bg_runner` behind flags. Lifting them onto the merge as a shared limbic slice is
the step that turns "two skills sharing a GPU" into "one animal sharing a self." This de-risk
re-validates the organ standalone + pins the frozen GO bar before that lift.

## The organ under test

```
  state_cue (CS)  --plastic-->  striosome_value (GABAergic MSN-D1 critic; learns V)
                                       |  GABA_B/GIRK  (-V; E_K=-90mV, the value subtraction)
                                       v
  reward_us (US, PPN-like) --exc-->  snc (DOPAMINE)  ==>  delta = r - V  (the SNc FIRING)
                                       ^  tonic pacemaker
  dopamine modulator: from_region_firing_signed over [snc] -> plasticity_rate scope=all
  (the critic LEARNS V via three-factor: STDP eligibility x the SNc-derived DA delta)
```

~170 neurons; CPU/numpy. Reuse-by-import of the validated wiring (the `reward_us→snc` afferent from
`sc_n5_rpe_probe.py`; the `striosome_value` GABA_B critic from `snc_stageb_critic_probe.py`; the
`from_region_firing_signed` dopamine rule). **No new `sim/` edit** (the GABA_B/GIRK conductance is
the already-shipped, owner-approved edit).

## The decisive design choice — a SYNAPTIC reward, not a direct SNc current

The minimal `cue→striosome→snc` probe delivers the reward as a **400 pA direct SNc current that
saturates the SNc (~130 Hz)**, so the GABA_B −V cannot dent it (the predicted/unpredicted gap goes
the *wrong* direction). The audit specifies `reward_us → snc` for a reason: a **tunable synaptic
reward afferent** leaves the SNc headroom for a graded value subtraction (the N5 probe gets
corr=−0.99 this way) and is more faithful to the brain-based-only bar (the reward enters as a
*spike*, not a host current write). Building the organ this way resolved the saturation.

## Two systematic-debugging root causes found + fixed (the de-risk's real content)

1. **Cold-start trap (operating point).** At the default critic weight the MSN-D1 never crosses
   rheobase → no firing → no STDP eligibility → the weight never grows → permanently dead (0/6,
   `V(strio)=0`, `|elig|=0`). Fixed by a critic-afferent weight that fires the MSN from the start
   (≥8); the critic then learns V — cue-gated firing (predicted/omission fire, unpredicted/baseline
   silent), confirming **V is neural + state-specific + learned**.
2. **Order artifact (the read protocol).** Measuring `predicted` then `unpredicted` with no reset
   between them let the slow GABA_B conductance (τ≈150 ms) from the predicted window carry over and
   suppress the unpredicted read — making `predicted > unpredicted` (no subtraction) an artifact, not
   a mechanism failure. This is the exact bug the nav deployment hit
   (`2026-06-10-N9-nav-deployment-stageB-PASS-seed42.md`, the `_n9_reset_critic_read_state` fix).
   Adding a clean reset (zero the slow GABA_B + a silent inter-trial gap, biologically a real ITI)
   before each frozen measurement flipped the direction to `predicted < unpredicted` in **every**
   config — the value subtraction working.

## Results — the Schultz RPE battery (6 seeds: 42/43/44/100/101/102, CPU)

Frozen operating point: `reward_us→snc` weight 10, `striosome_value→snc` GABA_B weight 10,
`gabab_propagation_strength` 0.22, critic afferent weight 10, 40 training trials.

| Signature | gate | result |
|---|---|---|
| (1) burst on an unpredicted US | snc/tonic ≥ 3× | **6/6** (≈3.5–5×) |
| (2) graded in reward magnitude | corr ≥ +0.8 | **6/6** (corr +0.99–1.00) |
| (3) reward-omission dip | omission < tonic | **6/6** |
| (4) predicted-US burst shrinks | pred ≤ 0.5 × unpred | **6/6** |
| (5) reward lesion → burst vanishes | within ±15% of tonic | **3/3** (decisive) |
| (6) critic GABA_B lesion → gap collapses | gap ≤ 1.2× | **3/3** (gap → 0.95–1.00, decisive) |

All four core signatures clear the pre-registered ≥5/6 bar (here 6/6); the two lesions are
mechanistic (3 clean = conclusive). **Frozen GO bar met.**

The predicted-shrink is operating-point-graded, as the documented "critic-rate-dependent GABA_B
operating point" residual predicts: `gabab_prop` 0.10 → 3/6, 0.15 → 5/6, 0.22 → 6/6. Pushing the
GABA_B strength lifts the cold-critic seeds over the 50% bar *without* touching the unpredicted burst
(the critic is silent there) — biologically the strong-clamp regime is Schultz's "fully-predicted
reward → no dopamine error." gp 0.15 is the graded-δ intermediate (5/6); gp 0.22 the clean
strong-clamp 6/6 (the locked default).

## Anti-cheat controls (all passed)

- **Reward lesion (5):** zeroing `reward_us→snc` collapses the burst to tonic (3/3) — the RPE *is* the
  synaptic reward, not a host scalar. The build asserts `current_reward_signal == 0` (brain-based).
- **Critic GABA_B lesion (6):** zeroing the GABA_B routing mask collapses the predicted/unpredicted
  gap to ≈1.0 (3/3) — the value subtraction *is* the synaptic GABA_B conductance, not host arithmetic.
- **The value is cue-gated (neural + state-specific):** the critic fires on the cue (predicted +
  omission) and is silent without it (unpredicted + baseline) — a host global-EMA value cannot produce
  this, which is exactly the discriminator the Stage-B design names.

## Honest scope / what this is NOT

- This validates **Rescorla-Wagner** δ = r − V (the US burst shrinks as V cancels r). It does **not**
  test the **temporal-difference cue-shift** (the burst migrating from the US onto the earliest
  predictive cue) — that needs the TD bootstrap (roadmap #3) and is the expected, documented R-W-vs-TD
  boundary, not a failure here.
- The predicted-shrink is the operating-point-sensitive gate (the GABA_B-rate residual). At gp 0.22 it
  is robust 6/6; the residual is well-characterized (and the nav deployment handles it with the
  FS-clamp / GIRK cap / place-code sparsity).
- This is the **standalone organ**. The merge lift (the actual directive payoff) + the
  conversational-salience / nav-state wiring are the next increments.

## Next — the merge lift (item #1 payoff)

Lift the validated limbic slice onto `build_merged_nav_conv_bridge` as an additive, default-off
opt-in (mirroring `co_resident_rf` / `co_resident_perception`): append the `reward_us` / `snc` /
`striosome_value` regions + the three limbic pathways after the existing slices (index bases
preserved), register the `dopamine` signed-firing modulator on the merged cfg, then validate (a) the
RPE battery passes on the merged-bridge limbic slice (the organ works co-resident), (b) the
conversational no-confab moat + the nav byte-identity do not regress. Reuse-by-import; no new `sim/`
edit expected.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._limbic_core_rpe_battery_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_limbic_core_rpe_battery_6seed.json
# operating-point search:
SIM_BACKEND=numpy python -m research.runners._limbic_core_rpe_battery_derisk --opsearch
```

Runner: `research/runners/_limbic_core_rpe_battery_derisk.py`. Raw:
`research/findings/raw/_limbic_core_rpe_battery_6seed.{json,txt}`. Audit:
`2026-06-18-full-spikeification-shared-substrate-roadmap.md` §4.
