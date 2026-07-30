# Two measured results: density=1.0 destroys the BTSP learning structure, and lane D is 6/6 NEGATIVE

**Date:** 2026-07-30 · Seed 42 (gap#5), 6 seeds (lane D). GPU, `backend=cupy` recorded in the artifacts.

## 1. gap#5 — at density=1.0 learning writes the LARGEST weight change and the LEAST structure

Measured on `circ_dW` (the circular resultant of the weight CHANGE) — the quantity the 6-seed headline actually
reports, and which no artifact previously recorded. The `lr0` control reads exactly 0.0000 at every density, as it
must when `dW=0`, so the metric is behaving.

| density | `lr0` circ_dW | `btsp` circ_dW | dW |
|---|---|---|---|
| 0.25 | 0.0000 | **0.1933** | 560 |
| 0.50 | 0.0000 | 0.1222 | 1136 |
| 0.10 | 0.0000 | 0.0863 | 91 |
| **1.00** | 0.0000 | **0.0040** | **2251** |

**At density=1.0 the write is 48× less structured than at 0.25 while being the LARGEST in magnitude.** Maximum
weight change, essentially zero spatial structure. The optimum is non-monotonic, peaking at 0.25 — sparser is not
simply better.

**What this does and does not vindicate.** The DC-clamp mechanism I proposed is still REFUTED by the arm-A
measurement (reader spread 182-354 mV, not ~0). But the *direction* the adversarial round pointed at is confirmed:
density=1.0 is genuinely pathological, and it is a property of the WIRING. The account of HOW was wrong; the
identification of WHERE was right. Also consistent with the arm-A apical readings: physiological
(−72…+14 mV, spread 5-37) at low density, absurd (min −1024.72 mV, spread 305) at 1.0.

**Scope, stated plainly.** These circ_dW values (0.004-0.193) sit far below the headline's 0.6705, so this is an
UNTUNED operating point (different `laps`/`dwell`/`w_inh` defaults). The density result characterises THIS
configuration. It does not transfer to the tuned one without being re-run there, and it is single-seed.

## 2. Lane D (perception) — on-bridge V1 self-organization is 6/6 NEGATIVE

`_b1_v1_selforg_onbridge_derisk`, 6 seeds, GPU. Every seed returns `verdict: NEGATIVE`; on seed 42, learning
*degraded* orientation selectivity below random initialisation (OSI 0.0694 post-learned vs 0.1698 pre-random,
against a 0.0559 shuffle-control ceiling), with geometry 0.5216 against a 0.60 gate and orientation decode 0.0781
against the host reference's 1.000.

The off-substrate numpy version of this same mechanism was GO at OSI 1.0 and RSA-to-host 0.988. **That is the
board's own methodology lesson repeating exactly:** off-substrate toys sit at ceiling while the substrate spreads
the same configs, so a numpy GO cannot license an on-substrate claim.

Ruled out already: the standing Hebbian bound trap is NOT the cause — the runner explicitly sets
`cfg.hebbian_max_weight = hebb_max` and `cfg.stdp_w_max = hebb_max`, with a comment citing the `w_max` gotcha.

**Under the no-defer rule this is a verdict on the METHOD, not the capability.** The V1 Gabor weights remain
host-DESIGNED structure (a criterion-2 residual), and closing it now needs a different mechanism. That learning
made selectivity WORSE than random init — rather than merely failing to improve it — is the diagnostic lead: the
rule is actively destroying structure, which is a different failure from not building any.

## 3. A self-inflicted near-miss worth recording

I reported "1 crux cell completed" from `ls -1 <dir> | wc -l`, which counted a `PARALLEL_EXPERIMENT.txt` stamp I
had written myself minutes earlier. A `*.json` glob showed zero. Caught before it propagated, but it is the
cleanest example of the day's pattern: the number looked real, and the check that falsified it took one line.
Count the artifact TYPE you mean, never the directory.
