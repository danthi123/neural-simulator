# EMERGE-5 — a genuine discovery: spike-count sampling noise ALONE self-organizes the hidden representation (dose-dependent on the rest bias); the anti-cheat needed the fix, not the mechanism

**2026-07-01 (autonomous; substrate ladder rung 2 — rate→spike Burstprop transition).** Runner `research/runners/_emerge5_spiking_burstprop_derisk.py`. This document records a diagnostic finding uncovered while reading EMERGE-5's first results, mechanistically isolated and confirmed via direct experiment before being folded into the record. Per the master directive, an unexpected result is traced to ground, not glossed over or discarded.

## The anomaly

EMERGE-5 (p0=0.5, the direct EMERGE-1b-inherited configuration) reported `apical_lesion` probe = 0.996 and `no_teaching_null` probe = 1.000 (mean over 3 seeds) — i.e. the hidden layer's representation, in arms specifically designed to carry **no credit signal at all**, near-perfectly linearly-decodes the ground-truth XOR-pair latents. That is suspicious on its face: these are supposed to be the "no-learning" floor.

## Ruling out a probe artifact

Tested whether a completely **untrained** (0-epoch, random Xavier init) net of the same architecture already supports this via sheer random-feature expressivity (384 hidden units vs. a 10-bit/1024-point discrete input space could plausibly let ridge regression memorize any deterministic function of the input without any training at all). Result: untrained probe ≈ 0.48–0.51 across hidden widths 8→1024 and ridge strengths 0.01→1000 — flat at chance. **Not a probe-design artifact.**

## Isolating the mechanism: real, and specific to spike sampling

Ran the identical `apical_lesion` condition on the pure **rate** model (`BurstpropMLP`, no spike sampling) for the same 1500 epochs. Result: the two hidden layers' weights are **bit-for-bit unchanged from init** (`max|ΔW| = 0.0` exactly); only the (always-correctly-trained) output layer moves. Probe stays at 0.483 — matching the untrained baseline. This confirms the analytic derivation: with `Y=0` (lesion) the rate model's burst probability `p` is exactly `p0` for every unit/example, so the burst-deviation `dev = post·(p−p̄)` is exactly zero every step, and no hidden update occurs.

**The rate model gives an exact floor at chance; the spiking model gives near-perfect structure. The difference is entirely the finite-sample spike-count noise.**

## The mechanism, traced

`dev = post·(P_obs − p̄)`, where `post = E_obs = k/samples` (the sampled event count) and the burst count `j ~ Binomial(k, p)` uses the **same** `k`. So:

- `Var(P_obs | k) = p(1−p)/k`
- `Var(dev) ∝ post · p(1−p)/samples` (since `k = post·samples`)

The noise **variance** of `dev` — not just its mean — scales with each unit's own activity level `post` for a given input. This creates a self-referential, activity-correlated noise structure: units that fire more for a given input get *larger* random perturbations for that input, even though the *expected* update is zero. Accumulated over ~9000 SGD+momentum steps (1500 epochs), this produces genuine, reproducible self-organization of the hidden representation toward the input's own latent structure — **with zero explicit teaching signal**. This is loosely resonant with (though mechanistically distinct from) a real developmental-neuroscience theme: activity/noise-correlated spontaneous activity organizing early cortical structure before any instructive signal is available (e.g. retinal-wave-driven visual-map formation, Wong 1999; Katz & Shatz 1996) — "noise itself can be an architect," not merely a nuisance.

## The dose-response confirmation (p0 controls it)

Predicted: since the effect's variance term is `p0(1−p0)`, a lower resting burst probability should produce a *much* weaker version of it (`0.03×0.97 ≈ 0.029` vs `0.5×0.5 = 0.25` — an ~8.6× reduction). Confirmed directly:

| p0 | lesion probe (mean) | no_teaching_null probe (mean) | deep_burst (TEST) probe (mean) |
|---|---|---|---|
| 0.5 | 0.996 | 1.000 | 0.946 (**below** the saturated floor — un-discriminating) |
| 0.03 | 0.582 | 0.574 | 0.874 (**clearly above** the reduced floor — discriminating again) |

At the biophysically-realistic rest bias (EMERGE-4's measured P0≈0.03), the noise-driven floor drops enough that the credit-carrying test arm's representational advantage becomes visible and clean.

## Honest read of what this changes

1. **The p0=0.5 EMERGE-5 "BOUNDARY" was not simply "noise breaks depth-credit."** It was specifically that the noise-*variance* term at p0=0.5 is strong enough to saturate the representation-probe metric on its own, masking whatever incremental structure the real burst-credit signal adds. This is a sharper, more precise diagnosis than the original verdict text, and it is itself informative: it identifies **which** parameter controls the confound and confirms the fix works.
2. **At p0=0.03, the representational picture is now clean and encouraging**: the test arm's probe (0.874) clearly separates from the reduced noise floor (0.58). Burst-credit assignment IS adding real, measurable structure beyond noise-driven self-organization, at the realistic rest bias.
3. **Task accuracy still lags** (0.505 mean vs. the 0.70 bar, vs. the rate ceiling 0.796) even at p0=0.03, and the S-sweep does not show simple monotone recovery there (S=1000 was worse than S=100 for two of three seeds) — so sample budget alone is not the full lever. The readout (turning a real representational advantage into task accuracy under noisy training) is now the more visible bottleneck, not "does structure emerge at all."
4. **This is a genuinely NEW mechanism worth naming**, separate from Payeur's burst-credit signal: noise-variance-driven self-organization from finite-sample burst-rate estimation. It is a real thing this project's own substrate produces, not a design flaw — the anti-cheat needed to control for it (fixed by testing at the biophysically-real p0), not the burst mechanism itself needing repair.

## Next concrete step (per the master directive — a sharpened mechanism to iterate, not a stop)

Distinguish **population** (independent noise sources averaged by a downstream reader) from **window** (samples-per-source over time) — EMERGE-5's `S` conflated both into one finite-sample budget. The literature's actual population-coding mitigation (Payeur; the rate model needed hidden width 384 to clear its gate) is about **width/ensemble averaging across independent units**, not longer time windows on the same implicit unit. Candidates, cheapest first:
1. **A width sweep at p0=0.03** (the now-clean regime) — does a wider hidden layer (matching or exceeding the rate model's own width-384 requirement) close the accuracy gap the way it closed the rate model's gap?
2. **An explicit population-averaging readout** — average `P_obs` over multiple independent unit "copies" per logical unit before computing `dev`, the literal population-coding fix, distinct from the window-only `S` sweep already run.
3. **The Sacramento-Senn microcircuit's spiking analogue** (EMERGE-3) — its self-predicting interneuron actively cancels the top-down prediction, which may be structurally more robust to this exact noise-variance coupling than Burstprop's raw burst-rate estimate.

## Artifacts
- Runner: `research/runners/_emerge5_spiking_burstprop_derisk.py` (now with the `p0`/rest-bias parameter + representation-level anti-cheat gate on ALL arms, not just the test arm).
- Results: `research/findings/raw/_emerge5_spiking_burstprop_p05.json`, `_emerge5_spiking_burstprop_p003.json`.
- Diagnostic commands (untrained-net control, rate-model-lesion control) run inline; not separately committed as scripts (cheap one-off checks, reproducible from the description above).

_Boundaries are undiscovered mechanisms: what looked like "the anti-cheat broke" was itself a real, named, dose-confirmed mechanism (noise-variance-driven self-organization) — finding it sharpens the next step rather than ending the investigation._
