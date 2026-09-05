---
type: finding
status: go
claim_check: measured
date: 2026-09-05
mechanism: the grounded-experience-stream Hebbian convergence with FOUR biological COMPANION processes the prior ungated rule proxied with constants -- (1) interoceptive relay POPULATION pooling (divisive normalization; SNR ~ sqrt(N)), (2) a label-free HOMEOSTATIC NOISE-FLOOR threshold per pooled channel (median + k*MAD, transmit only supra-baseline drive), (3) THREE-FACTOR US-gated eligibility (the Oja write scaled by the cleaned-arousal US salience), (4) HOMEOSTATIC SYNAPTIC SCALING toward an emergent population-mean-activity setpoint -- teaching the affect concept code read by the SAME validated separability CEILING; every threshold/setpoint read from the signal's OWN statistics (EMERGENT), never hand-set to the labels
lane: affect-learned-gate-retirement (rank-7)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_noise_robust_homeostatic_convergence_derisk.py
artifacts:
  - research/findings/raw/_affect_noise_robust_homeostatic_convergence_6seed.json
  - research/findings/raw/_affect_noise_robust_homeostatic_convergence.json
builds_on:
  - research/findings/2026-09-05-affect-grounded-experience-stream-hebbian-convergence-teaches-separable-code-derisk-PARTIAL.md
  - research/findings/2026-09-05-affect-gate-grounded-concept-code-lifts-ceiling-requirements-derisk-PARTIAL.md
  - research/findings/2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md
  - research/findings/2026-08-19-embodied-affect-interoception-GO.md
verdict: >
  GO / de-risk (6-seed). The grounded-experience-stream PARTIAL BUILT an emergent Hebbian convergence that teaches a
  fully separable, generalizing affect concept code at CLEAN grounding but stayed GO=False for ONE reason: at
  realistic interoceptive NOISE the strict worst-case zero-FP ceiling collapsed to near zero, and the whole noisy
  column (sigma>=0.5) failed at every coverage. This runner closes that residual by adding the biological COMPANION
  processes the ungated rule had proxied with constants -- the wall-reframe applied directly. The interoceptive relay
  is treated as a POPULATION whose downstream read POOLS it (divisive normalization; afferent-noise SNR ~ sqrt(N));
  each pooled channel ADAPTS to its own label-free baseline (a homeostatic noise-floor: median + k*MAD, transmitting
  only supra-baseline drive) so noise-only concepts fall to zero while a genuine US -- grounded affect concepts, which
  carry |val|>=0.5 by the _STRONG_MARGIN partition -- stays; plasticity is THREE-FACTOR US-gated (label-free) and
  assemblies are kept selective by homeostatic synaptic scaling toward an emergent activity setpoint. Every threshold
  and setpoint is read from the signal's OWN statistics (emergent), never hand-set to the labels. RESULT (6-seed, all
  four pre-registered gates pass, GO=True): at the SAME realistic noisy operating point where the prior ungated rule
  read near zero and the text code near zero, the noise-robust code clears the pre-registered worst-case bar; the
  clean/full code stays fully separable; held-out concepts at clean/full stay fully separable (TAUGHT, generalizes);
  and the no-body-state LESION and the shuffle-binding control both stay at the text baseline (the lift is grounding,
  not the machinery). The convergence is now essentially NOISE-INVARIANT up to the realistic noise level: at each
  coverage the worst-case is unchanged across sigma=0/0.5/1.0, and the residual worst-case tracks the COVERAGE bound
  (only grounded concepts can be recalled) rather than the noise. ATTRIBUTION (ablations, honest): the load-bearing
  companion process is POPULATION POOLING + the homeostatic noise-floor -- a 4-neuron relay fails the bar while a
  12+-neuron pooled relay clears it, and removing the three-factor gate or the synaptic scaling leaves the ceiling
  unchanged; the gate and scaling are biologically-faithful companions (the eligibility stopping-rule) that neither
  help nor hurt at this operating point. The threshold gain (k*MAD) is not cherry-picked (a k=2..5 sweep is flat).
  CONCLUSION: the grounded-TEACHER convergence arc is DE-RISKED at realistic interoceptive noise; the prior PARTIAL's
  named residual (a stronger homeostatic/three-factor rule + low-noise grounding) is closed with a measured mechanism.
  This is a de-risk GO, NOT a gate retirement: the residual is now purely COVERAGE (a real grounded world delivering
  broad per-concept US) and a fully-spiking on-substrate convergence -- both named next rungs. Additive, default-off,
  numpy-CPU, reuse-by-import, NO sim/ edit; affect_production_organ.py is byte-unchanged (_STRONG_MARGIN==2.0
  asserted; the population stream at n_relay=4 is byte-identical to the imported stream, np.array_equal-asserted).
lane_wall: affect salience gate (which words may move mood) -- rank-7 / affect-learned-gate-retirement
---

# Affect salience gate: a noise-robust homeostatic + three-factor convergence clears the strict bar at realistic interoceptive noise -- the grounded-teacher arc is de-risked

## The question this answers
`2026-09-05-affect-grounded-experience-stream-hebbian-convergence-teaches-separable-code-derisk-PARTIAL.md` built an
EMERGENT rate-Hebbian convergence that TEACHES a separable, generalizing affect concept code from a grounded-experience
stream, but was GO=False for exactly ONE reason: at realistic interoceptive NOISE the learned code's strict worst-case
zero-FP ceiling collapsed to near zero (the whole sigma>=0.5 column failed at every coverage), and -- unlike the
earlier oracle fusion -- relaxing the FP tolerance did not rescue it, because the noise was baked into the assemblies
at learning time. Its named next rung: a STRONGER homeostatic / three-factor rule that suppresses false-grounding on
neutral concepts, plus low-noise grounding. This runner asks: **does adding the biological COMPANION processes the
ungated rule proxied with constants -- population pooling, a homeostatic noise-floor, three-factor eligibility, and
synaptic scaling -- make the emergent convergence clear the strict worst-case zero-FP bar AT realistic interoceptive
noise, with grounding still lesion-load-bearing and the anti-cheats still holding?**

## The wall-reframe applied (what the proxy replaced with a constant)
The real interoceptive relay runs interacting processes the prior rule flattened to constants: it is a large POPULATION
whose downstream read POOLS many afferents (the prior used a 4-neuron relay fed raw, so per-neuron noise entered the
assemblies un-averaged); relay/sensory neurons ADAPT to their own spontaneous baseline (the prior had no adaptation, so
neutral concepts delivered "false grounding"); plasticity is NEUROMODULATOR-gated (the prior wrote every coincidence);
and postsynaptic homeostatic scaling keeps assemblies from becoming broad noise-responders. Each is a real biological
mechanism with an external anchor (Sources), added label-free and emergent.

## What ran
`research/runners/_affect_noise_robust_homeostatic_convergence_derisk.py` (SIM_BACKEND=numpy, 6-seed
42/43/44/100/101/102), on the SAME 164-word partition (102 affect + 62 neutral) and the SAME validated separability
CEILING instrument the prior rungs used. The baseline ungated convergence and the grounded-experience stream are reused
by import verbatim (like-for-like). The robust arm pools an N_RELAY=24 interoceptive population per channel, applies a
per-channel homeostatic noise-floor (median + k*MAD, k=3), scales the Oja-Hebbian write by the cleaned-arousal US
salience (three-factor), and applies per-epoch homeostatic synaptic scaling toward the population-mean-activity
setpoint. Arms: text-only baseline; the prior ungated convergence at the realistic + clean points (reproduces the
PARTIAL); the robust code across a (rho x sigma) grid; LESION (no body-state at learning); SHUFFLE (binding destroyed);
HELD-OUT (convergence trained on OTHER concepts -> generalization); text-only transfer; a POPULATION-size sweep and a
k*MAD-threshold sweep (isolate + de-cherry-pick the mechanism); per-companion ablations; a relaxed-FP sweep; and the
reused synthetic-clean instrument validation.

## Derived -- the measured numbers (all direct reads of research/findings/raw/_affect_noise_robust_homeostatic_convergence_6seed.json)
<!--derived: every value below is read directly from the cited 6-seed artifact -->
- **The strict bar is cleared at the realistic noisy point (G1 pass):** the robust code's worst-case ceiling at
  rho=0.6 / sigma=1.0 is **0.598 (0.605 mean), all 6 seeds**, vs the prior ungated rule's **0.010** and the text
  code's **0.059** at the same point. GO=True (G1, G2, G2b, G3 all pass; verdict_earned_status=GO).
- **Near NOISE-INVARIANCE up to the realistic level:** at each coverage the worst-case is unchanged across
  sigma=0/0.5/1.0 -- rho=0.6 reads 0.598 / 0.598 / 0.598; rho=0.8 reads 0.804 / 0.804 / 0.775; rho=1.0 reads
  1.000 / 1.000 / 0.912. The prior rule's sigma>=0.5 column was ~0 everywhere.
- **The residual is COVERAGE, not noise:** the worst-case tracks rho (0.2->0.196, 0.4->0.402, 0.6->0.598, 0.8->0.804,
  1.0->1.000) -- only grounded concepts can be recalled, the honest coverage bound. Required coverage spec now clears
  rho>=0.6 at EVERY tested sigma (0.0/0.5/1.0/2.0), vs the prior {0.0: rho>=0.6; 0.5/1.0/2.0: none}.
- **Grounding is LOAD-BEARING (G2 pass):** the no-body-state LESION reads **0.049** worst and the shuffle-binding
  control **0.127** worst -- both within margin of the text baseline (bar = text 0.059 + 0.15) -- while the robust
  treatment reads 0.598/1.000. The pooling/threshold/scaling machinery does not manufacture the lift.
- **TAUGHT not HANDED (G2b pass):** HELD-OUT concepts (never in the Hebbian weight training) separate at **1.000
  worst-case, all 6 seeds** at clean/full -- a reusable, generalizing text+body-state -> code map. (Held-out at the
  noisy point reads 0.467 worst, coverage-and-split-limited; the CLEAN held-out is the G2b criterion.)
- **The instrument is validated (G3 pass):** synthetic clean code **1.000** worst-case; text code **0.059** < the 0.2
  discrimination floor.
- **Attribution (ablations -- honest, the load-bearing companion is pooling + the noise-floor):** a 4-neuron relay
  reads **0.245** worst (fails); a 12/24/48-neuron pooled relay reads **0.598** (clears). Removing the three-factor
  gate leaves **0.598**; removing homeostatic synaptic scaling leaves **0.598**. So population pooling + the
  homeostatic noise-floor is the surpass; the three-factor gate and synaptic scaling are biologically-faithful
  companions (the eligibility stopping-rule) that neither help nor hurt at this operating point.
- **Not cherry-picked:** the k*MAD threshold-gain sweep (k=2/3/4/5) is flat at **0.598** worst; the thresholds
  themselves are label-free (median/MAD of each channel's own activity).
- **Extreme noise (sigma=2.0, 2x realistic) is the honest edge:** rho=0.6 still clears (0.529 worst) but high coverage
  degrades (rho=0.8 -> 0.451, rho=1.0 -> 0.353) -- pooling eventually saturates under extreme afferent noise.

## Reading it (no-defer)
The prior rung's one residual is closed with a measured mechanism, and the closure is the wall-reframe made concrete:
the ungated convergence's collapse under noise was not a property of Hebbian learning, it was the absence of the
COMPANION processes the real relay runs. Restoring population pooling + a homeostatic noise-floor makes the emergent
convergence essentially noise-invariant up to the realistic operating point, the grounding stays load-bearing
(lesion + shuffle at baseline), and the taught code still generalizes to held-out concepts. The honest attribution
matters: the ablations show the pooling + noise-floor is what clears the bar, while the three-factor gate and synaptic
scaling -- included for biological completeness and as the write-saturation stopping-rule -- do not move this operating
point. The remaining worst-case ceiling is the COVERAGE bound (only grounded concepts are recalled), which is exactly
what a real grounded world delivering broad per-concept US would raise -- the standing next rung, not a wall. This is a
de-risk GO for the grounded-TEACHER convergence arc, NOT a gate retirement: production is byte-unchanged.

## The scoped next build (named, not deferred)
1. **A fully-spiking on-substrate convergence** -- reuse `_genfrontier` build_propagation_bridge (rate-Hebbian ->
   NMDA concept spikes, held-out), GPU-queued, so the concept code is read off cp_firing_states, not a numpy matrix;
   the pooling, homeostatic floor, three-factor gate and synaptic scaling all have spiking realizations.
2. **A real grounded world** -- the standing residual across this whole arc: a world that experiences the
   conversational vocabulary with real bodily consequences, delivering the per-concept US (broad coverage raises the
   worst-case above the current coverage bound) that the interoceptive relay carries.
3. **Wire + lesion-test in production** -- only after (1): replace the fixed `_STRONG_MARGIN` host gate with the
   grounded-taught code as the affect salience readout, then a production lesion test (disable the grounded path ->
   the default answer changes) for integrated credit.

## Honest scope + residuals
Additive, default-off, numpy-CPU, reuse-by-import, no `sim/` edit; `_STRONG_MARGIN` unchanged (nothing wired -- the
population stream at n_relay=4 is byte-identical to the imported stream, np.array_equal-asserted; `_STRONG_MARGIN==2.0`
asserted). (1) The body-state US is a declared ORACLE STAND-IN for a grounded-perception teacher's output (the SAME
stand-in the prior rungs used); this measures whether the noise-robust convergence CAN teach a separable code GIVEN
such a stream and at what noise -- it does NOT deliver a real grounded world (named next build). (2) GO here means the
grounded-TEACHER convergence arc is de-risked, NOT that the gate is retired; the grounded-experience-stream finding
remains PARTIAL for its own (clean-grounding, generalization) question, which this rung extends to the noisy point.
(3) rate-Hebbian (numpy-CPU) convergence + rate-level pooling/homeostasis; a fully-spiking on-substrate convergence is
the named next rung. (4) N_RELAY=24 and k*MAD=3 are documented OPERATING POINTS (a modest population size + a standard
robust outlier cut), reported as sweeps so they are not cherry-picked; the thresholds themselves are label-free /
emergent. (5) the ceiling is a linear supervised upper bound (the spiking opponent's mild nonlinearity was measured
NOT to help by the prior boundaries). (6) the 164-word closed partition is inherited from the prior boundaries. (7) the
worst-case ceiling is the coverage bound, so a broader real grounded world is required to push it above ~rho.

## Sources (external -- deep_research_at_wall, lane affect-learned-gate-retirement)
- Turrigiano (2008, Cell) "The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses" -- multiplicative
  homeostatic scaling toward a firing-rate setpoint (the anchor for the homeostatic noise-floor + synaptic scaling).
- Fremaux & Gerstner (2016, Front Neural Circuits) "Neuromodulated STDP and Theory of Three-Factor Learning Rules";
  Gerstner et al. (2018, Front Neural Circuits) "Eligibility Traces and Plasticity on Behavioral Time Scales" -- the
  neuromodulator-gated three-factor eligibility rule.
- Shouval et al. (2025, Curr Opin Neurobiol) "Eligibility traces as a synaptic substrate for learning" -- the
  eligibility stopping-rule that preserves selectivity/representational power under write pressure.
- Carandini & Heeger (2012, Nat Rev Neurosci) "Normalization as a canonical neural computation" -- divisive
  normalization / population pooling (the anchor for afferent-noise averaging).
- Namburi, Tye et al. (2015, Nature) -- opponent valence populations bound to a real US, not lexical company.

## Reproduce
```
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_noise_robust_homeostatic_convergence_derisk --smoke
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_noise_robust_homeostatic_convergence_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_affect_noise_robust_homeostatic_convergence_6seed.json
```
