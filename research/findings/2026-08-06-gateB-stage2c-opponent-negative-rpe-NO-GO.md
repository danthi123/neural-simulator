---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2c-opponent-bidirectional-credit-negative-RPE-reward-expectation-baseline
backend: numpy
runner: research/runners/_vocal_gateb_stage2c_opponent_rpe.py
builds-on: 2026-08-06-gateB-stage2b-per-action-da-NO-GO.md
surpasses-method-wall: 2026-08-06-gateB-stage2b-per-action-da-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2c_opponent_rpe/numpy.json
  - research/findings/raw/gateb_stage2c_opponent_rpe/calibrate_numpy_lr0.02.json
---

# Gate B Stage 2c: the opponent (negative-RPE) arm FIXES reversal but does NOT pass the contingency (yoked) gate

## Verdict

**STAGE2C_NO_GO** (earned: preconditions hold, the frozen criteria are measured;
one fails). Adding the NEGATIVE arm — a reward-EXPECTATION baseline so an
executed-but-unrewarded action yields DA BELOW baseline (negative RPE) →
D1-LTD — genuinely surpasses the Stage-2b appetitive-only wall on the control it
was built for: **same-brain reversal goes P(B) = 0.00 (2b) → 1.00 (2c)** on the
same seed (≥0.60 PASS; 1.00 > 0.00 PASS). But the FROZEN capability gate still
fails: **D_contingent − D_yoked = 0.00** on all 3 exploring dev seeds (need
≥0.20; steer 0/6). A newly isolated wall now blocks the GO: under DENSE reward
the selector LOCKS once learning starts, so the yoked (decoupled) dominant-action
reward rate (~0.9) ≈ the contingent rate (1.0) — the negative arm has too few
unrewarded dominant executions to punish, so contingent and yoked both saturate.
Reward-OFF at Stage-1 noise is byte-identical (weights + raster) to Stage-1.
Artifact: `research/findings/raw/gateb_stage2c_opponent_rpe/numpy.json`.

## What was built (brain-based, on the Stage-2b substrate)

On the Stage-2b per-action DA selector (kept: `dopamine_{N,E,S,W}`,
`from_action_specific_reward` gated by the neural motor read-out
`last_selected_action`, `compute_per_synapse_da_signal` routing each channel's DA
to its `action_index`-tagged `str_d1_c` afferents, NEURAL coactivity eligibility,
the neural OU exploration process), Stage 2c adds the opponent arm:

- **Reward-EXPECTATION baseline V(executed action).** Each trial the executed
  channel's striatal-D1 population FIRING RATE during onset is read as V
  (`VALUE_GAIN·spikes`, clipped). This is the basal-ganglia direct-pathway
  value/go signal; the proposal→D1 route grows with reward so the rate tracks
  expected reward. In the OUTCOME epoch (the expected-reward time) the DA
  production computes `reward − V`: a positive burst when rewarded (reward > V),
  a NEGATIVE DIP when the expected reward is omitted (0 − V) → D1-LTD.
- **Bidirectional substrate engaged.** `enable_d1_d2_asymmetry` (D1/D2 sign
  array) and `reward_aversive_scale = 0.5` — the latter now applies to the
  NEGATIVE entries of the per-action DA signal (Schultz/Fiorillo asymmetry: the
  dip drives LTD of smaller magnitude than the matching burst), a minimal
  additive `sim/bridge.py` hook gated by `enable_d1_d2_asymmetry` (byte-identical
  when off; no-op when no synapse dips). Both are gated OFF when reward is off, so
  the reward-OFF build stays byte-identical to Stage-1.

## The negative arm is REAL and it advances the wall

- **Reversal (the strongest control) now PASSES.** Seed 730605: train action 0 →
  P(A) = 1.00, P(B) = 0.00; reward action 1 in the SAME brain → **P(B) = 1.00**
  (2b was 0.00). The dominant, now-unrewarded action 0 is depressed on every
  execution (sustained dip → D1-LTD) so action 1 takes over — direct behavioural
  proof the DA dip causes LTD on the over-selected route.
- **Expectation cancels predicted reward (positive side).** Contingent D1 route
  grows only to ~63 (2b reached 227 on the same seed): as V rises, `reward − V`
  shrinks, so LTP self-limits — correct temporal-difference behaviour.
- **Both lesions pass** (learning frozen during the readout test, so the
  manipulation holds). Contingent test 1.00; acquisition lesion (eligibility off)
  0.55 (Δ 0.45 ≥ 0.15); expression lesion (routes restored to symmetric baseline)
  0.60 (Δ 0.40 ≥ 0.15).

## Why the contingency (yoked) gate still fails — the isolated residual

The negative arm is necessary but not sufficient. Per-seed (exploring seeds
730601/730602/730605): D_contingent = D_yoked = 1.00; the yoked reward COUNTS
equal the contingent counts (e.g. 730601 c=(37,25), y=(37,25)) and the yoked
brain learns the rewarded target IDENTICALLY (yoked P(a0|reward a0) = 1.00).

Root cause: reward is DENSE and the selector LOCKS. Once learning starts the
brain does its dominant action on ~37/40 trials, so the contingent action's
reward rate → 1.0. The yoked control forces the same ~37 rewards on the same
indices, decoupled; but the yoked brain (independent noise) ALSO does the
dominant action on most trials, so its dominant-action reward rate ≈ 37/40 ≈ 0.9
— nearly contingent. The negative arm punishes only the ~3/40 unrewarded
dominant executions, swamped by the ~37 rewarded ones → net potentiation → both
saturate → D_contingent − D_yoked = 0.00. Reversal escapes this precisely because
the dominant action is unrewarded on 100% of executions, so the dip is sustained.

## Is the baseline a NEURAL critic or a host stand-in? (honest)

The value estimate is NEURAL, not a host EMA / Python running-average: it is the
executed action's str_d1 SPIKING-population firing rate, read out like the motor
read-out that moves the body. Documented residual host stand-ins: (1) a fixed
read-out gain `VALUE_GAIN`/clip (a constant critic→DA weight, defensible like the
motor threshold); (2) the `reward − V` subtraction runs in the DA production rule
(the abstracted DA system), not a spiking SNc; (3) it reuses the ACTOR's own D1
population (advantage-style), not a dedicated learned critic. None of these is the
cause of the NO-GO — a perfect critic cannot make contingent diverge from yoked
under this dense-reward, locked-selector protocol.

## Quantified residual

- D_contingent − D_yoked (exploring seeds) = 0.00 (need ≥0.20); steer 0/6.
- Reversal P(B) = 1.00 (need ≥0.60) — PASS (2b: 0.00). Lesions Δ 0.45 / 0.40 ≥
  0.15 — PASS. Equivalence weights+raster match — PASS.
- Banked advance: the opponent arm converts reversal from FAIL→PASS and makes the
  expectation baseline temper LTP (227→63). The single unmet criterion is the
  contingency divergence, gated by SELECTOR LOCK-IN under dense reward, NOT by the
  negative arm (validated) nor by weight-level credit specificity (Stage 2b).

## Exact next mechanism (biology-grounded, in-substrate, no host shortcut)

**Sustained, uncertainty-gated exploration** — the 2b finding's still-unbuilt
companion process (point 2). Amplitude-only OU cannot keep sampling the
disfavoured action after a transient lead (measured 40..600 pA, 2b). Need a
neural exploration drive that PERSISTS while value is uncertain and decays only as
confidence rises: a Bogacz-Brown familiarity/uncertainty signal (the moat's D.04
gate) or tonic-DA-modulated exploration bonus on the striatal MSNs. Then
CONTINGENT (target reliably rewarded → confidence rises → exploit) and YOKED
(reward decoupled → value never confident → keeps sampling → the dominant action
is repeatedly UNREWARDED → the already-working negative arm punishes it) DIVERGE.
The negative arm stays; sustained exploration supplies the unrewarded dominant
executions it needs. Closure is deferred to a METHOD (dense-reward + locked
selector), not the CAPABILITY.
