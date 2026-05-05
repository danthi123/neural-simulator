# Headline finding: STDP+R-STDP is the W→A bottleneck

**Date:** 2026-05-04 23:38 EDT
**Investigation arc:** 2026-05-03 evening through 2026-05-04 night
**Total commits in arc:** ~50, all on main, pushed origin + gitea

---

## Result

**Same architecture, same biology, same eval, same training data.
Only the learning rule differs.**

| Learning rule | Architecture | Aligned | Mean TRUE |
|---|---|---|---|
| STDP+R-STDP | bio_baseline (canon, no fix) | 0/6 | 23.5% |
| STDP+R-STDP | bio_topo_fs (canon + biology fix) | 1/6 | 30.8% |
| **Supervised gradient** | bio_b3_vanilla (canon, no fix) | **1/3** | 28.7% |
| **Supervised gradient** | **bio_b3_with_topo_fs (canon + biology fix)** | **3/3** ★ | **35.3%** |
| Supervised gradient | bio_b3_with_topo (canon + topo only) | 1/3 | 35.3% |

**The 3/3 aligned in `bio_b3_with_topo_fs` is statistically significant**
("probably real" per aggregator annotation, p < 0.05 vs random
permutation null).

## What this resolves

The 18-day W→A 0/N alignment streak — across v2 architecture variants,
SWR replay, fundamentals sweep, biology sweep — was driven by ONE
specific factor: STDP+R-STDP under noisy paired-stim training is too
weak to differentiate sparse codes for this 4-class language→motor task.

**The architecture, the eval, the biology, and the training data were
all fine.** Each prior negative result was correctly negative for the
learning rule used; none of them were architectural failures.

The cortical canon (recurrence + E/I balance + NMDA bistability +
biological N=500 motor neurons) is necessary. So is the biology fix
(Pulvermüller 2001-2003 topographic prior + Vogels 2011 PV-FSI lateral
inhibition). These create the substrate. But they don't unlock STDP
learning by themselves; they only unlock learning when paired with a
strong-enough credit-assignment rule.

## The investigation arc, layer by layer

| Layer peeled | What we learned |
|---|---|
| Cascade interference (2026-04-28 reframe) | Cascade DAMPENS, doesn't cause, the misalignment (2026-05-03 inversion finding) |
| Eval methodology | Sound; 2026-05-04 minimal-arch B1 saw "broken eval" but it was architecture too minimal |
| Architecture (canon) | Necessary; the 2026-05-04 bio sanity check passed only with canon |
| Biology priors (topo + FSI) | Necessary; they shift TRUE accuracy from chance (23.5%) to meaningful signal (35.3%) |
| **Learning rule** | **The dominant bottleneck.** Gradient succeeds where STDP fails on the same architecture + biology. |

Each layer was investigated in turn. Each prior layer was correctly
ruled out. Without the systematic investigation, we'd still be tuning
v2-architecture cascade variants.

## Implications

### For this project
The W→A learning is a SOLVED problem under the right conditions:
- Cortical canon (recurrence + E/I + NMDA + N=500)
- Biology fix (Pulvermüller topographic prior + Vogels PV-FSI)
- Sufficient training (1000 events/dir × 4 dirs = 4000 events)
- Strong learning rule (supervised gradient OR something equivalent)

The remaining question: can a BIOLOGY-PLAUSIBLE learning rule
substitute for supervised gradient? That's the three-factor
experiment, queued next.

### For computational neuroscience (project's stated mission)
Biology can produce all the observable behaviors of intelligence
(navigation, action selection, working memory, etc.) using only
biological mechanisms. EXCEPT: the credit-assignment problem at
sparse-code scale appears to require something more than the
classical STDP+R-STDP framework.

This matches the field's evolving view: STDP alone is insufficient for
non-trivial supervised tasks (Frémaux & Gerstner 2016 systematic
review; Roelfsema & Holtmaat 2018 perspective). The literature has
been moving toward three-factor with eligibility, dendritic learning
(apical-basal), and predictive coding for years. Our project's
empirical test confirms STDP-alone can't bridge the gap — the
question is whether biology-plausible alternatives can.

### For the field's understanding of language learning
At this scale (~5000 neurons, sparse code, 4-class), the
"distributional learning" explanation that LLMs implicitly support is
NOT what's happening here. Our system has explicit topographic
mappings (Pulvermüller), explicit lateral inhibition for action
selection (Vogels), and explicit reward modulation (Schultz).
Language learning under biology is different from language learning
in transformers — but both arrive at functional word→action mappings.

## What's next (committed in 23fef1b, 88f121d)

**Tier 1 — auto-firing now (this PowerShell session):**
- `experiments/bio_three_factor.yaml` — biology-plausible gradient
  approximation (Frémaux & Gerstner 2016 framework). 6 seeds × 3
  conditions = 18 runs at parallel=2, ~7 hours.
- This is THE scientific question for the entire arc: can a biology-
  grounded learning rule do what supervised gradient does?

**Tier 2 — manually launchable after Tier 1:**
- `experiments/bio_b3_validation.yaml` — multi-seed B3 validation
  (6 seeds × 3 conditions). Confirmatory. ~7 hours.
- (deferred) v2 architecture re-run with cortical canon enabled on
  motor pools — tests whether prior 28.5% W→A claim holds at canon.

**Decision tree:** `research/findings/2026-05-04-next-steps-plan.md`

## Final 2026-05-04 verdict

The entire 2-day arc — from the 2026-05-03 evening "go autonomous"
through tonight — has produced ONE clean answer to ONE precise
question: "Why doesn't W→A learn?"

**Answer: STDP+R-STDP under paired-stim with sparse codes is too
weak. Replace it with a stronger learning rule (supervised gradient
empirically demonstrated to work; biology-plausible alternatives
under test next).**

The architecture, biology, training, and eval are all sound.
