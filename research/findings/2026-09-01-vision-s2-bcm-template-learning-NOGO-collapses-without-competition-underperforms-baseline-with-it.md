---
type: finding
status: contributing
date: 2026-09-01
lane: perception (board #135 / #75)
mechanism: bcm-sliding-threshold-s2-template-learning
seeds: [42, 43, 100]   # EXPLORE only -- 0/3 beat at every hyperparameter point tried; no full-6 confirm spent
artifacts:
  - research/findings/raw/lanes/perception/vlin_bcm_baseline_none_explore.json
  - research/findings/raw/lanes/perception/vlin_bcm_g0.3_e1_cf0.35_explore.json
  - research/findings/raw/lanes/perception/vlin_bcm_g0.3_e1_cf0.35_kwta0.25_explore.json
  - research/findings/raw/lanes/perception/vlin_bcm_collapse_instrument.json
  - research/findings/raw/lanes/perception/vlin_bcm_tinynudge_explore.json
runner: research/runners/_vision_lindiscrim_readout_derisk.py
builds_on:
  - research/findings/2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md
  - research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md
---

# BCM sliding-threshold learning of the S2 templates is a NO-GO at explore: naive per-unit BCM collapses the whole bank to one shared direction without competition among the learners, and a Foldiak/Kohonen-style competitive gate that fixes the collapse still leaves every hyperparameter point tried UNDERPERFORMING the frozen-random baseline (0/3 explore seeds beat it) -- the mechanism doesn't beat "no learning at all"

## One-line verdict

The 2026-09-01 satdiv/ridge/k-WTA finding named **BCM S2-template learning** (Bienenstock, Cooper & Munro
1982) as the decisive next mechanism after every readout-side lever (normalization form, ridge strength,
k-WTA sparsification) plateaued -- the residual was diagnosed as the *information* the frozen random S2 bank
carries, not how it is read. This de-risk **built it** (`--s2-learn bcm` in the same runner, PORTED from
`sim/bridge.py`'s validated on-bridge `hebbian_bcm` equations, not re-derived) and ran an extensive
hyperparameter exploration (30+ configurations spanning gain, epochs, theta EMA rate, and a competitive
gate, on seed 42, then validated the finalists on all 3 explore seeds 42/43/100). The result is a clean
**NO-GO at explore**: naive single-cell BCM applied to a bank of templates that all share the SAME training-
patch pool (unlike the on-bridge V1 case, where each cell has its own retinotopic receptive field) collapses
the ENTIRE 96-template bank toward one shared direction (`theta_std` -> 0 across all templates, measured);
adding a competitive learning gate (Foldiak 1991 / Kohonen 1982, composed with BCM's own signed LTP/LTD)
fixes the collapse but does not rescue performance -- **every explored hyperparameter point sits at or below
the frozen-random baseline on the 3-seed explore set (best config: 0/3 beats the NO-GO floor by margin, mean
held accuracy 0.34 vs the baseline's 0.44)**. Given the explore set never shows a lead for BCM over "no
learning at all" -- not even at the gentlest possible perturbation -- no full-6-seed confirm was spent (the
established convention from the satdiv/ridge finding: a full-6 run is not spent chasing a lever with no
explore-set lead). **This is a negative on the METHOD** (unsupervised, purely local activity-dependent
tuning of a shared template bank), not a license to abandon S2-template learning as a capability -- the
decomposition below names why, and points to a supervised/relational next mechanism.

## The mechanism built (faithful port, not a re-derivation)

`_bcm_learn_s2_templates()` (`research/runners/_vision_lindiscrim_readout_derisk.py`) applies the EXACT
equations `sim/bridge.py`'s on-bridge `hebbian_bcm` branch implements (~L9849-9891) directly to the dense
`(n_S2, D)` S2 template matrix, since there is no CuPy sparse-connectivity object in this runner's numpy
patch-based architecture to hook the substrate's own inline implementation into:

```
y_i     = ReLU(w_i . x)                                  postsynaptic drive (the SAME 'drive' the
                                                           existing eval-time code computes)
theta_i <- (1-theta_alpha)*theta_i + theta_alpha*y_i^2    sliding metaplastic threshold EMA
dw_ij   = gain * x_j * y_i * (y_i - theta_i),  only where x_j > pre_floor (presynaptic gate)
```

Applied ONLINE (theta is a running average across presentations, so this must be sequential, exactly as
on-bridge), one training patch at a time, over `--s2-bcm-epochs` seeded-shuffled passes through every
(image, location) patch the FIXED spiking S1->C1 front end produces on the training split. Templates are
renormalized to unit L2 norm after every update (`--s2-bcm-renorm`, default on) -- the direct analog of the
substrate's own `[w_min,w_max]` excitatory clip, needed here because every downstream consumer
(`_apply_s2_norm`, `_kwta_over_templates`, the LIF gain calibration) treats S2 templates as UNIT-NORM
cosine-matching directions; without it, a magnitude confound (bigger templates -> bigger drive -> an
artificially easier ridge read) would be indistinguishable from genuinely learned template information. The
new flag is additive and default-off (`--s2-learn none`), byte-identical to every prior run of this file.

## The instrument-verification step that changed the design: naive population BCM collapses without competition

A first hyperparameter sweep, ported straight from the on-bridge gain scale (`--s2-bcm-gain 200`, the
validated V1 self-org operating point), produced a striking failure signature on seed 42: `theta_std=0.0000`
across ALL 96 templates (i.e. every template converged to an IDENTICAL sliding threshold), the RATE-code
ceiling DROPPED below the frozen-random baseline (0.25 vs 0.51), and `frac_theta_near_zero` reached 1.00 (all
templates end up firing near zero on essentially every training patch -- a dead-unit fixed point, since
`dw=0` whenever `y=0` regardless of `x`). Re-running this exact configuration on all 3 explore seeds
reproduces the pattern cleanly:

| seed | held (BCM, gain=200, no competition) | held (RANDOM control) | `theta_std` | `frac_theta_near_zero` |
|---|---:|---:|---:|---:|
| 42 | 0.2708 | 0.2708 | ~3e-68 | 1.00 |
| 43 | 0.3333 | 0.25 | ~7e-109 | 1.00 |
| 100 | 0.1979 | 0.2292 | ~4e-109 | 1.00 |

**Artifact:** `research/findings/raw/lanes/perception/vlin_bcm_collapse_instrument.json`.

The diagnosis: the on-bridge V1 case trains ONE weight vector per RETINOTOPIC cell, each with a spatially
distinct receptive field, so independent random initial conditions naturally symmetry-break into different
preferred features. This runner's `n_S2` templates instead all draw from the SAME shared pool of training
patches (a feature bank, not a spatial array) -- with no interaction between the units being trained, every
template's independent BCM trajectory is pulled toward the same dominant direction of the shared input
statistics. This is the textbook "grandmother-cell collapse" risk of applying a single-cell Hebbian/BCM rule
to a POPULATION of feature detectors with no lateral competition (which is exactly why Foldiak 1991 and
Kohonen's 1982 self-organizing-map competitive learning both pair a Hebbian/BCM-style update with an
explicit winner-relative gate). It is a genuine instance of this project's own wall-reframe: the companion
process missing here is **competition among the learners**, distinct from `--s2-kwta-frac`'s existing
read-time competition among an already-fixed bank's RESPONSES.

**The fix (`--s2-bcm-competitive-frac`):** restrict the weight update at each presentation to the top-`frac`
templates by their CURRENT drive `y` (a Foldiak/Kohonen-style winner-relative gate composed with BCM's own
signed rule; `theta` still updates for every template every presentation, unconditionally, matching the
substrate's own semantics where `cp_bcm_theta` updates regardless of the weight-branch gate). This
eliminates the collapse (`theta_std` becomes non-zero -- e.g. ~0.017-0.019 for the gentlest ablation and ~0.075-0.079 for the best-found configuration below, both cited in the next section's artifacts -- no more dead-unit fixed point) <!--derived--> -- but, as the next section shows, fixing the collapse does not fix the outcome.

## The decisive 3-seed explore result: every configuration underperforms "no learning at all"

An exploration on seed 42 covered gain in [0.001, 20], epochs in {1, 3, 5}, `theta_alpha` in {0.01, 0.02,
0.05}, and `competitive_frac` in {0.15, 0.25, 0.35} (30+ configurations total). The best point found
(`gain=0.3, epochs=1, theta_alpha=0.02, competitive_frac=0.35`) was then run, alongside the byte-identical
frozen-random baseline, a BCM+k-WTA composition, the pure-collapse instrument above, and a near-zero-gain
"tiniest possible nudge" ablation, on all 3 explore seeds:

| arm | held mean (42/43/100) | vs NO-GO floor (0.34) | vs RANDOM (load-bearing) | RATE ceiling |
|---|---:|---:|---:|---:|
| **baseline (`--s2-learn none`, byte-identical)** | **0.4445** (0.49/0.38/0.47) | beat **2/3** | lb **3/3** | 0.4687 |
| BCM best found (gain=0.3, e=1, cf=0.35) | 0.3403 (0.44/0.26/0.32) | beat **0/3** | lb 2/3 | 0.4062 |
| BCM best + k-WTA-at-S2 (frac=0.25, composed) | 0.316 (0.39/0.31/0.25) | beat **0/3** | lb 1/3 | 0.3646 |
| BCM tiniest nudge (gain=0.001, e=1, no competition) | 0.4271 (0.43/0.45/0.41) | beat **1/3** | lb 3/3 | 0.4444 |
| BCM collapse instrument (gain=200, no competition) | 0.2673 (0.27/0.33/0.20) | beat 0/3 | lb 0/3 | 0.25 |

**Artifacts:** `vlin_bcm_baseline_none_explore.json`, `vlin_bcm_g0.3_e1_cf0.35_explore.json`,
`vlin_bcm_g0.3_e1_cf0.35_kwta0.25_explore.json`, `vlin_bcm_tinynudge_explore.json` (all under
`research/findings/raw/lanes/perception/`).

**Every row that involves learning sits below the untouched frozen-random baseline.** The pattern is
monotonic in perturbation size: the gentlest possible nudge (gain=0.001, drift from init ~0.18, essentially
a rounding-error-scale change to the random templates) comes CLOSEST to the baseline (0.4271 vs 0.4445, beat
1/3 vs 2/3) but still does not match it; the best "real" learning configuration found (drift ~0.6-0.7) is
further below (0.3403, beat 0/3); the fully collapsed instrument (drift ~1.8) is worst (0.2673, at RANDOM).
Composing with the previously-best lever (k-WTA-at-S2, beat-3/6 on its own 6-seed confirm) does not rescue
BCM -- it makes it slightly WORSE (0.316), because k-WTA sparsifies an already-degraded template bank rather
than compensating for it. **No hyperparameter point explored beats the do-nothing control.**

## Why: the class signal here is CONFIGURAL, and local unsupervised statistics do not carry it

The task's class identity is defined by the ARRANGEMENT of the same small set of stroke orientations across
slots (`_object_classes` picks permutations of the SAME `n_slots` orientations for every class, so every
class has an IDENTICAL orientation histogram by construction -- this is precisely what makes it a
*configural*, not a local-feature, discrimination). A local, unsupervised, activity-dependent rule like BCM
can only sharpen a template's tuning to whatever LOCAL PATCH STATISTICS are most prevalent in the training
patches it sees -- and since every class shares the same orientation content, the dominant local statistics
are, by the task's own design, largely class-INDIFFERENT. Tuning templates toward that dominant structure
does not recover the fine, RELATIONAL signal (which slot holds which orientation) the class label actually
depends on; it instead moves the templates away from their "generically informative" random starting point
and toward a training-set-specific local optimum that generalizes WORSE to the held positions -- exactly the
monotonic degradation-with-drift pattern measured above. This is consistent with, and extends, the base
finding's own diagnosis (config-C's centroid read beating a "tuned" readout) and with #75b's granule-
expansion finding (a different re-basis of the same C2 code that also failed to move the underlying
separability ceiling): **structural re-projections of this C2 code, whether random-subsampled (granule),
re-normalized (satdiv/ridge), sparsified (k-WTA), or now unsupervised-retuned (BCM), all fail to add
separable structure that the direct 96-dimensional signed readout was not already extracting from the
UNTOUCHED random bank** -- the frozen random projection is, for this specific task, already close to the
best a purely bottom-up mechanism can do, because the discriminative axis is not a locally-detectable
feature at all.

## Honest scope against the three pre-registered risks

1. **Thin data (6 examples/class) makes the per-unit sliding threshold noisy.** Confirmed but not the
   dominant effect: `n_presentations` per seed is 2,346-2,360 for `epochs=1` (each of the 96 training images
   contributes ~24-25 patch locations, so the patch-level sample size is far larger than "6/class" alone) up
   to 11,730-11,800 for `epochs=5` -- yet MORE presentations (more epochs, seen in the original sweep) did
   not help; if anything the largest-drift configurations (more epochs, higher gain) generalized WORST. The
   residual is not principally an under-sampled `theta` estimate; it is the structural mismatch above.
2. **Learned templates could themselves plateau (a genuine informative negative, no-defer).** This is
   exactly what was found, but the failure mode is sharper than a plateau: BCM-learned templates
   (post-competitive-gate) do not merely fail to IMPROVE on the baseline, they consistently UNDERPERFORM it,
   monotonically with the size of the perturbation from the random start.
3. **Template-bank overfitting to 6 examples/class is a confound distinct from readout overfitting.** The
   held-out-of-tuning discipline (explore only on 42/43/100, confirm on 44/101/102) was the design meant to
   catch this -- but the explore set itself never shows a BCM lead to confirm, so the full-6 run was not
   needed to catch an overfitting regression that already shows up at 3 seeds. The held-position transfer
   test IS the mechanism that surfaces this: train accuracy for the best BCM config reaches 0.89-0.95 (not
   shown in the table above; see the per-seed rows in the artifact), close to the untouched baseline's own
   0.92-0.98 train accuracy, so BCM is not obviously MORE overfit on the TRAIN split -- it is worse
   specifically on the HELD positions, consistent with the configural-mismatch diagnosis rather than a
   simple sample-size story.

## Brain-based status

Somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory, per-step membrane noise) at
S1, S2, and the readout class populations -- unchanged from the base runner. The BCM template-learning rule
itself operates on RATE (`y = ReLU(w.x)`, an analog postsynaptic-drive proxy), matching the substrate's own
`hebbian_bcm` implementation, which likewise rides the RATE-window coactivity trace, not per-spike STDP.
FLAGGED scaffold: the competitive gate (`--s2-bcm-competitive-frac`) is a host-computed top-k selection
(an idealization of lateral inhibition among the template-bank units, not an emergent network dynamic) --
same status as `--s2-kwta-frac`'s existing read-time competition. No `sim/` edit; the BCM equations are
PORTED (same formula, verified against `sim/bridge.py`'s inline implementation) into a standalone numpy
function because this runner's dense template matrix has no CuPy sparse-connectivity object to hook the
substrate's own branch into.

## The next mechanism (no-defer)

The wall is not "BCM is broken" (the competitive-gated version learns cleanly, produces diverse non-collapsed
templates, and is a faithful, verified port of a rule already validated elsewhere on this substrate) -- it is
that **unsupervised, local, activity-dependent tuning has no way to discover a discriminative axis that is
defined by cross-location arrangement rather than local feature content**. Two concretely different next
mechanisms, in order of how directly they target this diagnosis:

1. **Supervised (label/error-guided) template shaping.** Extend the readout's own three-factor delta signal
   (already computed for the LEARNED linear discriminant `V`) back into the S2 template layer -- a genuine
   two-layer credit-assignment mechanism (e.g. a REINFORCE-style perturbation credit, or a learned feedback
   projection in the spirit of this project's gap#4 deep-credit work) so templates specialize on the axis the
   READOUT'S error actually needs, not on local input variance. This directly answers "what informs the
   templates about class identity" with the one signal BCM structurally lacks.
2. **A relational/conjunctive layer above S2, not a re-tuning of S2 itself.** Since the discriminative signal
   is the CROSS-LOCATION arrangement of features, a mechanism that explicitly conjoins S2 responses from
   DIFFERENT retinotopic locations (rather than pooling each template's response independently over
   locations, as the current MAX-pool C2 does) could expose the configural code without needing the
   individual local templates to change at all -- the templates may already carry sufficient local
   information; what may be missing is a stage that BINDS locations together, not one that tunes local
   feature detectors harder.

Both are DIFFERENT methods from what this de-risk falsified (unsupervised local BCM); per the project's
standing rule, this is a verdict on that one method, not a license to abandon S2-level plasticity as a
capability.

## Reproduce

```bash
# byte-identical baseline (unchanged default):
SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_readout_derisk \
  --seeds 42 43 100 --out research/findings/raw/lanes/perception/vlin_bcm_baseline_none_explore.json

# the best BCM configuration found (still NO-GO at explore):
SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_readout_derisk \
  --seeds 42 43 100 --s2-learn bcm --s2-bcm-gain 0.3 --s2-bcm-theta-alpha 0.02 \
  --s2-bcm-pre-floor 0.02 --s2-bcm-epochs 1 --s2-bcm-competitive-frac 0.35 \
  --out research/findings/raw/lanes/perception/vlin_bcm_g0.3_e1_cf0.35_explore.json

# the collapse instrument (naive port, no competition, on-bridge gain scale):
SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_readout_derisk \
  --seeds 42 43 100 --s2-learn bcm --s2-bcm-gain 200 --s2-bcm-competitive-frac 0 \
  --out research/findings/raw/lanes/perception/vlin_bcm_collapse_instrument.json
```

## Sources

- Bienenstock, E. L., Cooper, L. N. & Munro, P. W. (1982). Theory for the development of neuron selectivity.
  *J. Neurosci.* 2(1):32-48. (The sliding metaplastic threshold ported here.)
- Cooper, L. N. & Intrator, N. (2004). Theory of cortical plasticity. World Scientific. (BCM theory review.)
- Foldiak, P. (1991). Learning invariance from transformation sequences. *Neural Comput.* 3(2):194-200.
  (Competitive/winner-relative gate composed with BCM here; already used for `--s2-kwta-frac`.)
- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biol. Cybern.*
  43(1):59-69. (Competitive learning among a population of feature detectors sharing one input pool.)
- Prior on this substrate: `2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md` (the on-bridge BCM
  validation this port is faithful to); `2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-
  points-to-S2-template-learning.md` (this de-risk's mandate); `2026-08-25-vision-nonlinear-2layer-granule-
  expansion-readout-does-not-lift-the-c2-linear-ceiling.md` (a DIFFERENT re-basis of the same C2 code that
  also failed to move the separability ceiling -- independent corroboration that the ceiling is not a
  re-projection/re-tuning problem).
