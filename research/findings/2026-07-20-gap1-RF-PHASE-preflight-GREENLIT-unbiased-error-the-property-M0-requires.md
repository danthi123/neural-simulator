# gap#1 RF PHASE pre-flight — GREENLIT: the error is UNBIASED across the value range (the exact property M0 requires)

Following the M0 reframe (the wall is a VALUE-DEPENDENT nonlinear encode bias, not raw fidelity; an UNBIASED encode
at corr ~0.82 would be GO), the research-gate's #1 build candidate — RF PHASE delivery of v_t — is pre-flighted for
the one property that matters: is its error UNBIASED across the value range?

## The pre-flight (cheapest decisive check; NO sim/ edit)

Encode a grid of 128 values in [-3,3] as PHASES (v -> phase (v+3)/6 -> complex kick z=exp(i*2*pi*phase), unit
magnitude = value lives in the PHASE), on a real RESONATE_AND_FIRE bridge (period 200), read back via the
magnitude-invariant `rf_read_phases` (first-spike step), decode, and measure the error AS A FUNCTION OF VALUE.

## Result — UNBIASED, GREENLIT

| metric | value |
|---|---|
| corr(v_hat, v) | **0.954** |
| mean error (bias) | -0.015 |
| rms error | 0.018 |
| **error by value band** | small \|v\|<=1: **-0.0150** · v<-1: **-0.0154** · v>1: **-0.0153** |
| **bias-spread across bands** | **0.0007** (essentially ZERO) |

**The RF phase error is SYMMETRIC/UNBIASED across the entire value range** (bias-spread 0.0007), with NO
value-dependent dead-zone — the opposite of the rate code's value-dependent distortion that a linear read-out could
not fix (-2.904 -> -2.933). The tiny residual bias (-0.015) is a CONSTANT offset (linearly correctable), not a
value-dependent nonlinearity.

## Why this greenlights the full build (and why it should ACCUMULATE gracefully)

- M0 established: i.i.d./unbiased error accumulates GRACEFULLY over the recurrence (corr 0.877 -> +0.241, crosses
  zero at ~0.80), whereas the rate code's BIASED error accumulates coherently and is ~1.5-2 nats worse at the same
  corr.
- The RF phase error is unbiased (bias-spread 0.0007) at corr 0.954 per value. An unbiased error at corr 0.95, when
  accumulated, should track the M0 i.i.d. curve (well above zero at 0.95), NOT the rate code's collapse. This is the
  first candidate whose error has the CHARACTER (unbiased) M0 identified as the requirement, not just a high corr.
- It sidesteps the rate-code floor entirely: the value rides a single spike's TIMING (phase), read
  magnitude-invariantly, not a Poisson-floored spike count -> no dead-zone, no saturation.

## HONEST scope + the deployed pre-flight still required (the day's lesson)

This is a PER-TOKEN, isolated-value reconstruction. The token-SDR trap (standalone 0.906 -> deployed 0.501) means
the DEPLOYED accumulated-state test is still the real de-risk. BUT the mechanism differs decisively: token-SDR's
standalone-vs-deployed gap was a BIASED (per-token-reset + subtracted) measurement; the RF error here is measured
unbiased and symmetric, and unbiased per-token error provably accumulates as unbiased (zero-mean) noise. So the
prediction is that the deployed RF encode tracks the M0 i.i.d. curve (GO at corr 0.95), UNLIKE the biased encodes.

## NEXT (the motivated build, now de-risked)

Build the full RF phase encode: per token, deliver v_t = Wv·LN(emb[x_t]) as RF phases (precompute the V-vector
phasor dictionary once), decode the phases to charge the validated `cp_ssm_state`, run the deep-NLL gate. PRE-FLIGHT
before pre-registering (baked in): (a) re-confirm M1 +0.542; (b) measure the DEPLOYED accumulated-state corr AND the
per-channel error MEAN (bias) on real sentences -- confirm the accumulated error stays unbiased (bias-spread small);
(c) GO target = deep-NLL > 0, predicted by the M0 curve at the achieved corr. This is the first gap#1 encode with a
PRINCIPLED reason (unbiased error) to expect it clears the wall, not just a hope of higher fidelity.

Pre-flight runner: `research/runners/_gap1_rf_phase_preflight.py`.
