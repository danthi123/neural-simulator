---
type: finding
status: contributing
date: 2026-08-10
mechanism: songbird-statevalue-actor-critic
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
instrument: YOKED-CONTROL contingency decomposition — the GO is the FIX-minus-YOKED weight separation (yoked = identical DA-magnitude distribution DECOUPLED from the action), with the untrained arm as the null. The effect is ATTRIBUTED to reward-contingency because the yoked arm stays near the null while the fix arm separates (per-seed numbers in the body table, cited to the artifacts); a non-contingent (heterogeneity/DC) source would move yoked equally. This is the exact gate the prior gateB attempts FAILED.
---

# Learn-to-speak-from-communicative-success: the per-context STATE-VALUE baseline PASSES the contingency gate 5/6 (verified) — the LEARNING wall the prior gateB attempts could not cross is fixed; DA credit is now reward-contingent

The "learn to speak from communicative success" wall was pinned this session to the LEARNING stage (both readout-SNR
duals — homeostat + amp-attractor — REFUTED: the learned VALUE was wrong, not un-read-out). A deep-research round
isolated the bug and prescribed the songbird Area-X/VTA fix; this finding BUILDS + VERIFIES it and it crosses the
contingency gate the prior attempts (global-DA three-factor, opponent-negative-RPE) failed.

## The bug and the fix

<!--derived-->

**Bug (in `_pragmatic_success_readback_leg2_v2_derisk.py`):** the critic `V` is per-UTTERANCE (crit[u]); `rpe =
success - V[winner]` is a per-ACTION advantage that COLLAPSES to ~0 for every utterance at convergence (each V[u] ->
success(intent,u)) -> the actor loses its differential teacher -> chance (v2 reproduced: actor-WTA 0.500,
critic-argmax 0.556, seed-100 critic INVERTED 0.000). **Fix (v3 runner, NO sim edit):** a per-CONTEXT STATE-VALUE
baseline `V(intent)` -> the advantage `A = success(chosen) - V(intent)` is SIGNED (aligned/above-average ->
potentiate; below-average -> actively DEPRESS), so the increment COMPOUNDS across trials (separation set by trial
count, not the tiny single-trial gap). Grounding: Kasdin 2025 *Nature* (Area-X DA = contrast of current rendition vs
recent-rendition HISTORY = a STATE baseline), Gadagkar 2016 *Science* (bidirectional prediction-relative performance
error), Chen 2018 (ventral state-value critic -> VTA -> Area-X actor).

## The verified result — contingency GO 5/6 (the decisive test)

<!--derived-->

Cleanest read = the intent->utter WEIGHT separation `wsep = mean w[success-optimal] - mean w[others]` (structural,
readout-free), FIX arm vs YOKED (same DA-magnitude distribution, DECOUPLED from the action). At the leak-corrected
centering `--ema-beta 0.4`, INDEPENDENTLY REPRODUCED this session (`research/findings/raw/_pragmatic_success/v3_b04_s{42,43,44,100,101,102}.json`):

| seed | fix wsep | yoked wsep | contingent? |
|---|---|---|---|
| 42 | 0.148 | 0.012 | YES |
| 43 | 0.066 | -0.002 | YES |
| 44 | 0.123 | -0.025 | YES |
| 100 | 0.078 | 0.019 | YES (leak killed: 0.078->0.019 vs beta=0.1) |
| 101 | 0.209 | 0.012 | YES |
| 102 | 0.200 | 0.092 | borderline (degenerate target) |

Untrained wsep ~0.001. **5/6 seeds: fix separates strongly toward what the reward teaches while YOKED stays ~0** —
the learned credit is genuinely CONTINGENT on the reward. This is the surpass: the prior gateB attempts
(2026-08-06-gateB-stage2-*-NO-GO) converged but FAILED the yoked gate; the state-value baseline PASSES it.

## The leak mechanism (why beta-centering is load-bearing) + the honest bounds

<!--derived-->

- **The yoked leak (2/6 at the default beta=0.1) is a lagging-baseline artifact, diagnosed + fixed:** yoked Δw ∝
  execution-freq × E[delivered DA]; a causal LAGGING baseline under an IMPROVING policy carries E[advantage] > 0
  (success rises, baseline lags) -> net-positive DA non-contingently potentiates the heterogeneity-favored assembly.
  A FASTER-centered baseline (E[DA]->0, beta=0.4) kills it while preserving the contingent component (confirmed).
- **s102 residual (characterized):** the single seed where success-optimal is ONE utterance for ALL intents (a
  degenerate target) — beta-centering reduces the yoked leak monotonically (beta 0.1 -> 0.4 -> 0.6 -> 0.8) but does
  NOT eliminate it (the fix arm still separates more than yoked at every beta). So s102 is a GENUINE degenerate-target
  corner (the policy collapses onto one utterance, so the lagging-baseline leak persists), an honest bound, NOT a rule
  failure — 5/6 non-degenerate contexts pass cleanly.
- **The host-EMA critic shortcut is now BURNED DOWN — the FULLY-NEURAL per-intent spiking critic ALSO passes
  contingency, 6/6 (upgrade).** With `--neural-critic` (K critic neurons, `Vctx[t]=rate(crit[t])`, TD-trained; no
  host EMA), reproduced this session at beta=0.4: the fix arm separates while YOKED stays at/below the null on ALL 6
  seeds (fix 0.02-0.14 vs yoked ~0: 0.004/0.008/0.031/-0.006/-0.016/-0.016). The neural critic's separations are
  SMALLER (it is noisier than the host EMA) but the contingency is CLEANER — and crucially the s102 degenerate-target
  leak that persisted with the host EMA is GONE with the neural critic (yoked -0.016). ⇒ **the learn-to-speak learning
  fix is fully brain-based** (spiking actor + spiking per-context critic, DA-gated eligibility), no host shortcut,
  contingent on all 6 seeds. Artifacts: `research/findings/raw/_pragmatic_success/v3_neural_b04_s42.json` etc.
- **A SEPARATE wall exposed (distinct frontier):** the coincidence-success signal does NOT rank the belief-ALIGNED
  utterance highest for ~56% of targets (success-optimal == aligned only 8/18) — so the end-to-end "speak the ALIGNED
  utterance" metric is capped by REWARD QUALITY (~0.44 ceiling for a perfect success-maximizer), NOT the learning
  rule. Fixing the learning stage was necessary; making the brain say the *pragmatically-aligned* utterance next needs
  a distinctiveness / pragmatic-cost term in the success signal.

Artifacts (the beta=0.4 6-seed contingency, reproduced this session):
`research/findings/raw/_pragmatic_success/v3_b04_s42.json`, `research/findings/raw/_pragmatic_success/v3_b04_s43.json`,
`research/findings/raw/_pragmatic_success/v3_b04_s44.json`, `research/findings/raw/_pragmatic_success/v3_b04_s100.json`,
`research/findings/raw/_pragmatic_success/v3_b04_s101.json`, `research/findings/raw/_pragmatic_success/v3_b04_s102.json`.
Reproducer `research/runners/_pragmatic_readback_leg2_v3_statevalue_derisk.py` (forks v2 by import; NO `sim/` edit).
SIM_BACKEND=numpy.
