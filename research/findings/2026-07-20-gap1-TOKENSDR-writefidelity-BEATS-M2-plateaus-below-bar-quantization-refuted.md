# gap#1 TOKEN-SDR write-fidelity — BEATS the M2 wall (0.906 vs 0.786), plateaus below the 0.95 bar, quantization REFUTED

The research gate's #1 candidate (token-SDR SELECTION + fixed Wv value-synapses -> conductance, replacing M2's
regression encode), run on the regenerated V=1000/d=128 checkpoint, deployed held-out token stream. NO deep-NLL
(pre-registered: write-fidelity gate first). NO `sim/` edit (drives + reads public arrays; the only sim/ touch is
an additive default-off `--no-ou` DIAGNOSTIC that sets enable_ou_process=False).

## Result — the gate's reframe is DIRECTIONALLY VALIDATED, but it does not clear the bar

| condition | determinism | write-fidelity corr(v_t, true) |
|---|---|---|
| M2 wall (regression encode) | — | **0.786** |
| token-SDR, OU-on (deployment) | 44/60 | **0.883** (+0.097 over M2) |
| token-SDR, OU-off (ceiling) | **60/60** | **0.906** (+0.120 over M2) |
| pre-registered bar | | **0.95** |

**Selection beats regression** — the gate's central claim (spikes carry token IDENTITY; the fixed synapse delivers
the magnitude) is confirmed directionally: 0.906 vs M2's 0.786. OU noise costs ~0.023 (0.883 -> 0.906); the
remaining ~0.044 to the bar is the mechanism.

## The residual is NOT quantization — a K-sweep REFUTES it

Hypothesis: 13.8 spikes reconstructing a 128-dim value is finite-spike quantization; more spikes -> higher fidelity.
**FALSE.** Sweeping SDR neurons per token (OU-off ceiling):

| k_active | spikes/window | write-fidelity |
|---|---|---|
| 8 | 13.8 | **0.906** |
| 16 | 27.4 | 0.903 |
| 32 | 56.4 | **0.845** |

More spikes make it **monotonically WORSE**. So the ~0.90 ceiling is **structural to the finite-window conductance
readout** (per-neuron firing heterogeneity across a larger assembly adds variance to the weighted sum; the g_e
accumulation over the 6-step window under real membrane dynamics does not linearly reconstruct v), NOT a spike-count
limit. This is a clean, quantized characterization — and it means "add more neurons" is the WRONG lever (refuted),
so I am not pulling it.

## ⚠️ The pre-registered 0.95 bar may be CONSERVATIVE — the real test is the deep-NLL itself

The 0.95 bar was the gate's ESTIMATE ("the -0.345 deep-NLL gap needs near-exact input"). But the M2 finding
MEASURED two points of the write-fidelity -> deep-NLL mapping: **0.786 -> -0.030** and **1.000 -> +0.126**. Linear
interpolation puts **0.906 -> ~+0.06 (POSITIVE, crosses zero).** A 2-point linear interpolation is unreliable (the
true relation is likely threshold-like), so this is a HINT, not a claim — but it means the honest next step is to
**MEASURE the deep-NLL at 0.906**, not to tune write-fidelity toward an estimated proxy bar.

**This is a genuine methodological fork, recorded rather than resolved by fiat:** my pre-registration said "no
deep-NLL unless >= 0.95". Honoring that literally = record NO-GO and stop. But the bar itself is an unverified
estimate, and the mechanism cleanly beats M2 with a positive interpolated deep-NLL — so measuring the actual
deep-NLL (replacing the estimated proxy with the real quantity) is NOT goalpost-moving (I am not tuning to hit 0.95;
I am testing whether the mechanism's real capability is positive). The distinction: tuning knobs to reach 0.95 would
be chasing the proxy; running the deep-NLL once at the honestly-achieved 0.906 tests the actual capability.

## Verdict + next

- **Write-fidelity gate as pre-registered: NO-GO** (0.906 < 0.95). Recorded honestly.
- **The mechanism (token-SDR selection) genuinely beats M2** (0.906 vs 0.786) and closes the audit gap in principle
  (M1's host matmul -> spiking token-selection + real synaptic conductance), refuting quantization as the residual.
- **NEXT (the real test): wire the token-SDR path into the full deep-NLL eval and MEASURE whether 0.906 recovers a
  positive deep-vs-trigram** — the interpolation says it might (~+0.06). If positive: the conductance-drive input is
  a GO and the 0.95 estimate was conservative. If negative: the ~0.90 write-fidelity ceiling is genuinely
  insufficient and the structural readout limit is the wall.
- The honest caveat the gate flagged STANDS and is unresolved: even if it works, a skeptic calls this "M1 with a
  spiking veneer" (the token pool is effectively one SDR per token = a lookup). The gate's defense (world supplies
  the discrete token; the brain's fixed Wv synapses do the value projection as spiking conductance) is on record for
  owner judgement.
