---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2d-uncertainty-gated-exploration-value-difference-coverage-decoupled-yoked
backend: numpy
runner: research/runners/_vocal_gateb_stage2d_uncertainty_gated.py
builds-on: 2026-08-06-gateB-stage2c-opponent-negative-rpe-NO-GO.md
surpasses-method-wall: 2026-08-06-gateB-stage2c-opponent-negative-rpe-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2d_uncertainty_gated/numpy.json
---

# Gate B Stage 2d: uncertainty-gated exploration removes the yoked-steering CONFOUND (mean D_yoked 1.0->0.0) but the per-seed steer gate still fails on yoked lock VARIANCE

## Verdict

**STAGE2D_NO_GO** (earned: preconditions hold; the byte-identical, lesion, and
reversal criteria PASS; one criterion fails). Adding a NEURAL uncertainty gate
(exploration OU amplitude on the spiking proposal + striatal MSN populations,
gated closed-loop by a NEURAL value-difference read-out from the str_d1 spiking
rates, with a coverage/novelty term forcing both actions to be sampled), plus
fixing the yoked control's confound, ELIMINATES the Stage-2c systematic yoked
steering: **mean D_yoked over exploring seeds 1.00 (2c) -> 0.00 (2d)** on the same
seeds, while contingent still steers perfectly (**D_contingent_exploring = 1.00**).
The mean divergence D_contingent - D_yoked = 1.00 (>=0.20). But the FROZEN GO gate
(**steer_seed_passes >= 5** of 6, per-seed) is met on only **3/6**: per-seed
D_yoked is now a NOISE term in {-1, 0, +1} (one decoupled-reward draw -> a WTA
lock), and seed 730605 coincidentally locked yoked to give D_yoked = 1.0 -> that
seed's D_contingent - D_yoked = 0 -> steer fail. Byte-identical reward-OFF guard,
both lesions, and reversal all PASS. Artifact:
`research/findings/raw/gateb_stage2d_uncertainty_gated/numpy.json`.

## Two distinct confounds were isolated at this wall (the scientific advance)

**(1) The Stage-2c yoked control was CONFOUNDED (master-yoking).** 2c delivered the
yoked reward on the MASTER's reward-trial indices. But the yoked brain shares
wiring + afferents with the master, so it does the target on the SAME trials the
master did; master-reward-indices then coincide with yoked-target-execution, so
the yoked brain experiences a REAL target->reward contingency. Measured in 2c: the
yoked brain learned the target IDENTICALLY (D_yoked = 1.00 on every exploring
seed, str_d1 route + reward count byte-equal to contingent). No uncertainty gate
can suppress learning a contingency that is genuinely present in the experience.
**Fix:** action-DECOUPLED reward (Hammond-1980 contingency degradation) -- the
same reward COUNT on RANDOM indices independent of the yoked brain's action, so
P(reward|target) = P(reward|other) = base rate. This alone drops mean D_yoked from
1.00 to 0.00: decoupled reward no longer steers toward the target.

**(2) The residual: decoupled reward still produces a FREQUENCY-driven route
asymmetry that the WTA test amplifies to a per-seed lock.** Route potentiation
scales with (times the action is taken) x (reward rate). The un-learned selector
takes its intrinsic-bias action ~70% of trials even at high OU amplitude (baseline
p0 held at 0.22..0.57 on exploring seeds; 2b already showed 40..600 pA cannot
equalise this). So ~70% of the decoupled rewards land on the bias route -> it
potentiates ~2x more events than the other -> the near-deterministic
winner-take-all frozen test converts that small, finite-sample asymmetry into a
full lock (p0 in {0, 1}, not ~0.5). Which route wins is set by which random-reward
draw concentrated on which action -> per-seed D_yoked is a coin flip in {-1, 0,
+1} (mean 0 over 6 seeds, but high variance). The single-draw per-seed steer gate
is defeated by this variance, not by any systematic yoked steering.

## The uncertainty gate is NEURAL and closed-loop (brain-based-only)

- **Signal:** per-action value V_c = EMA of the `str_d1_c` onset SPIKE rate on
  trials where action c was executed (the BG direct-pathway value/go read-out; its
  proposal->D1 route grows with reward, so V_c tracks expected reward). Read from
  spikes like the motor read-out that moves the body -- NOT a host EMA of the
  reward scalar. `conf = clip(|V0-V1|/(V0+V1)) x coverage`, coverage = per-action
  novelty (both actions must be sampled >= MIN_SAMPLES) so an UNSAMPLED action
  reads uncertain (its value unknown, not "init") -- the Bogacz-Brown novelty
  bonus that keeps exploration high until the alternative is tried.
- **Drive:** `sigma = SIGMA_CONFIDENT + (SIGMA_UNCERTAIN-SIGMA_CONFIDENT)(1-conf)`
  set on `bridge.ou_noise_std` -- the OU membrane-noise amplitude on the spiking
  proposal + striatal D1/D2 populations (tonic-neuromodulator-modulated MSN
  variability). The conf->sigma arithmetic is the abstracted tonic-DA/ACh
  controller, the same documented-residual class as Stage-2c's reward-V DA
  arithmetic. Kept from 2c: per-action DA, the opponent negative-RPE arm, the
  neural critic, the byte-identical reward-OFF guard.
- **Load-bearing check (honest).** The `conf_lesion` (gate off, sigma fixed) was
  run only on the lesion seed 730605, itself a coincidental-lock seed (gated
  D_yoked there is also 1.0), so it does NOT cleanly isolate the gate's marginal
  effect. What IS clean: the decoupled-reward fix + gate together drop mean D_yoked
  1.00->0.00 on the same seeds. Separating the gate's contribution from the
  decoupled-reward fix's contribution is not established here.

## Frozen criteria (unchanged from the Stage-2 preregistration)

- Reward-OFF byte-identical to Stage-1 (weights + raster): PASS.
- Acquisition lesion: contingent 1.00 vs acq-lesion 0.65 (delta 0.35 >= 0.15) PASS.
  Expression lesion: vs 0.35 (delta 0.65 >= 0.15) PASS.
- Same-brain reversal: P(B) 0.00 -> 1.00 (>= 0.60, and 1.00 > 0.00) PASS.
- Contingency steer_seed_passes: **3/6 (need >= 5) FAIL** -- the sole unmet gate.
  D_contingent_exploring 1.00, D_yoked_exploring mean 0.00, mean diff 1.00.
  Per-seed D_yoked {730601:0, 730602:-1, 730605:+1}: variance, not bias.

## Quantified residual + exact next mechanism

The residual is NOT the negative arm (validated 2c), NOT weight-level credit
specificity (2b), NOT the yoking confound (fixed here), and NOT the mean effect
(contingent steers 1.00, decoupled 0.00). It is the per-seed VARIANCE of the
yoked lock: a finite-sample, FREQUENCY-driven route asymmetry (bias action taken
~70% -> gets ~70% of decoupled rewards) amplified by the deterministic WTA test.

**Next mechanism (biology-grounded, in-substrate): DIRECTED novelty-biased
exploration that equalises action FREQUENCY.** Amplitude-only OU cannot (2b:
40..600 pA). A curiosity/novelty drive (the shipped `from_novelty` production
rule; Oudeyer/Schmidhuber) that adds excitatory drive to the LESS-sampled action's
proposal population until the two action frequencies equalise would make the
decoupled reward land ~50/50 -> yoked routes stay equal -> the yoked test reads
~0.5 per draw -> per-seed D_yoked -> ~0 with low variance -> steer passes. Care:
this manipulates selection, so it must be a NEURAL drive (extra current to a
spiking pool) gated by a NEURAL novelty read-out (the action-count deficit from
the motor read-out), documented as such, and must not become host action-picking.
Secondary (estimator, not mechanism): the yoked control's reward schedule is
stochastic, so a single draw has ~unit variance; averaging D_yoked over several
decoupled-reward draws estimates E[D_yoked] ~ 0 directly -- more rigorous, but a
metric-computation change, so deferred to the mechanism above. Closure is deferred
to a METHOD (frequency-driven route asymmetry under a WTA read-out), not the
CAPABILITY.
