---
type: plan
status: live
date: 2026-05-02
---

# 2026-05-02 — Strategic options for text I/O: biology-grounded paths forward

After overnight session establishing W→A 28.5% (p=0.027 robust) ceiling under
v2 config, the question is: **what biology-grounded changes realistically
push the system to functional text I/O at human-comparable performance?**

This doc thinks through options carefully, ranks them by leverage × tractability
× biological grounding, and recommends a sequenced path.

## Where we are vs. where biology is

| Aspect | Our system | Real biology | Gap |
|---|---|---|---|
| Language regions | 256 neurons | ~10⁶ Wernicke + 10⁶ Broca | 4000× |
| Training trials | 3000 (100 ep × 30 steps) | 10⁵-10⁶ in toddler-1yr | 30-300× |
| Time scales | ms STDP only | ms + s + min + hr + days | many missing |
| Cell type diversity | 1 main pyramidal type per region | 8+ in striatum alone (TK-2017) | rich |
| Top-down feedback | None for language | Predictive coding (Friston) | absent |
| Replay/consolidation | Cluster D v2 (SWR) untested for text | Sleep cycles | not integrated |
| Joint attention/grounding | Reward = Manhattan distance | Tomasello pragmatic context | weaker |

The biggest gaps are **trial count, region size, and time scales**. Closing
even one of these meaningfully changes capability.

## What's bottlenecking the current system?

From the overnight diagnostic:

1. **Weights saturate by 100 ep** (200-ep test gave identical weights — `2026-05-02-longer-training-NEGATIVE.md`)
2. **Cascade quality is poor** (training-time correct moves ~30%)
3. **High per-seed variance in language pathway formation.** Not structural
   N-bias — across 6 seeds, the per-direction means are: north -0.016
   (near zero), east +0.128, south +0.079, west +0.072. North only
   "appears" reversed in 4/6 seeds because it's HIGH VARIANCE around zero.
   On average it's not biased; it's unstable.
4. **I→W eval also high variance per seed** (different "lucky" direction each)

The cascade quality (#2) is the LEVERAGE POINT. With cascade at ~30% correct,
STDP gets a noisy training signal — even perfect plasticity machinery can't
extract differential learning from a near-chance source.

The variance problem (#3, #4) means ~150-200 trials per metric per seed isn't
enough to reliably detect 5-10pp learning effects. We need either lower-variance
training or more evaluation per seed (or both).

If cascade reached 60%+ correct moves, the language pathway would see:
- Clean motor_X firing patterns aligned with target words
- Strong reward correlations
- Reliable pre-post pairings

This is exactly what biology does via **scaffolding**: babies master
motor coordination (~6-12 months) before language production (~12+ months).
Vygotsky's zone of proximal development; Piaget sensorimotor stage.

## Ranked options (leverage × tractability × biology)

### Tier 1: Highest leverage, biology-grounded, tractable

**Option A: Curriculum — visuomotor first, then text I/O**

Biology source: Vygotsky scaffolding, Piaget sensorimotor stage, Tomasello
joint-attention prerequisites. Real children master action selection before
language production.

Concrete plan:
- Phase 1 (200 ep): visuomotor only — disable language pathway plasticity
  via existing `set_plasticity_gate()` infrastructure. Train cascade to
  60%+ correct moves on visuomotor task alone.
- Phase 2 (100 ep): enable language pathways. Bridge has been "scaffolded"
  by Phase 1's clean cascade dynamics; STDP now has clean training signal.

Effort: ~1-2 days implementation, ~3-5 hours per training run.
Expected impact: **20-40 pp accuracy boost** based on the principle that
the language pathway can only learn as cleanly as the cascade dynamics
allow.

Risk: low — uses existing plasticity gate infrastructure, doesn't change
underlying network.

**Option B: Variance reduction — motor lateral inhibition + multi-baseline eval**

Biology source: cortical FS interneurons + spinal Renshaw cells produce
lateral inhibition between competing motor commands (Kandel ch 35).
Lateral inhibition sharpens selection AND reduces variance.

Concrete plan:
- Resurrect `--enable-motor-lateral-inhibition` infrastructure for text
  training context (it exists but is DEPRECATED — "PARTIAL — net negative
  when stacked with adaptive DA" — but text doesn't use adaptive DA)
- Pathways: motor_X → motor_FS_X (excitatory), motor_FS_X → motor_Y for
  Y≠X (inhibitory). This creates WTA microcircuit per pool
- Combine with multi-baseline averaging in eval (3-5 baseline windows per
  trial, average) for cleaner per-trial reference

Effort: 1-2 days (mostly testing).
Expected impact: **per-seed variance halved**, north reversal becomes rare.
Could push W→A from 28.5% to ~33-35%, with much lower per-seed std.

Risk: low-medium. Reusable infrastructure. The earlier deprecation was for
navigation+DA context which doesn't apply here.

### Tier 2: Medium leverage, biology-grounded, moderate effort

**Option C: Population vector decoding (Georgopoulos 1986)**

Biology source: motor cortex encodes movement direction via population
vector — sum of preferred-direction vectors weighted by firing rate.
Georgopoulos & Schwartz et al. 1986. Used in BCI literature.

Concrete plan:
- Replace argmax-over-delta-spike-counts in evaluate_word_to_action
- Compute population vector from motor_X firing rates
- Read out direction as the angle/magnitude of the vector
- Doesn't require retraining — applies to existing checkpoints

Effort: ~half day (eval-only code).
Expected impact: **~5-10 pp accuracy boost**, possibly more if cascade
dynamics had partial signal lost in argmax.

Risk: minimal. New eval methodology; original eval preserved.

**Option D: Sleep replay consolidation (already have infra!)**

Biology source: Wilson & McNaughton 1994 hippocampal replay; Buzsáki SWR;
O'Reilly complementary learning systems 1995. Sleep replay strengthens
recent associations.

Concrete plan:
- Cluster D v2 (SWR-gated CA3 plasticity, commit 2026-04-30) already
  exists in g11_bg_runner
- Currently NOT integrated with text I/O training
- Add post-episode "sleep" phase: trigger SWR with text-pathway plasticity
  unlocked, replay recent (token, action, reward) tuples
- Could double effective trial count without doubling wall-clock time

Effort: ~3-5 days (integrate cluster D infra with text training, design
replay schedule).
Expected impact: **~10-15 pp accuracy boost**, more stable per-seed
results (consolidation reduces variance).

Risk: medium. Cluster D was tested for navigation, not text I/O. Some
risk of replay strengthening wrong correlations.

### Tier 3: Higher leverage, deeper biology, more effort

**Option E: Bigger language regions + sparser coding**

Biology source: Quian Quiroga 2005 sparse-distributed-representations;
"Jennifer Aniston neuron". Real cortex uses ~1% sparse codes for
concept neurons.

Concrete plan:
- Increase language_input/output regions: 256 → 1024 (4x)
- Decrease sparsity: 0.10 → 0.02 (5x sparser, ~20 active neurons per token)
- Each token's pattern much more distinct (less overlap)
- BigLang test in flight (256→512); results available shortly

Effort: 2-3 days incl. retuning drives, density.
Expected impact: **~5-10 pp from cleaner pattern separation**, depends on
how much the current 26-active-neurons overlap is limiting.

Risk: medium. Larger regions = more memory + slower training. Need to
retune drives.

**Option F: Multi-time-scale plasticity (synaptic tagging)**

Biology source: Frey & Morris 1997 synaptic tagging-and-capture;
Redondo & Morris 2011; Reymann & Frey 2007 LTP late phase. Real LTP
has fast (mins) and consolidated (hrs-days) phases.

Concrete plan:
- Currently STDP has only fast Δw + eligibility trace decay
- Add slow-consolidation tag: synapses with strong fast-LTP get tagged,
  later consolidate based on neuromodulator presence (DA, ACh, NE)
- Slow phase doesn't apply during normal training; activates during
  rest/sleep
- Substantial bridge code

Effort: 1-2 weeks.
Expected impact: **~10-20 pp** from learning persistence, but only with
sufficient training time. Most impactful at scale.

Risk: high. Major bridge change.

### Tier 4: Most foundational, highest effort

**Option G: Predictive coding feedback**

Biology source: Friston free-energy principle; Rao & Ballard 1999;
Bastos et al. 2012. Real cortex sends top-down predictions and bottom-up
prediction errors.

Concrete plan:
- Add top-down predictive feedback from PFC to lower cortical layers
- Predict expected sensory input, propagate prediction error
- Drives more efficient learning of features that resolve uncertainty

Effort: 2-3 weeks. Substantial architectural change.
Expected impact: **~15-30 pp**, esp. on I→W (high variance because
forward-only signal is noisy).

Risk: high. Bridges substantial network reorganization.

**Option H: Joint attention / mirror neurons**

Biology source: Tomasello shared intentionality (2003); Rizzolatti mirror
neurons. Language acquisition fundamentally requires social grounding.

Concrete plan:
- Modify reward to include joint-attention component:
  reward when (a) action correct AND (b) language matches what agent
  is "looking at" (gaze-modulated reward)
- Could implement gaze via beacon-perception-attention mechanism

Effort: 1-2 weeks.
Expected impact: **~10-20 pp on I→W** specifically.

Risk: medium-high. Reward shaping changes dynamics.

## Recommended sequence

Given the user's preference for biology-grounded changes and realistic
goals, here's the ranked path:

### Immediate (this week, 1-2 days each)
1. **Option B: Motor lateral inhibition** — fixes north reversal cleanly.
   Easy resurrection of existing infrastructure. Should bring 4/4 directions
   to LEARN.
2. **Option C: Population vector decoding** — half-day eval-only change.
   Test on existing 6 v2 checkpoints to see if it extracts more signal.
3. **Option A: Curriculum (visuomotor → text)** — most leverage. Pretrain
   cascade until 60%+ correct moves, then enable text I/O. Single biggest
   accuracy win expected.

### Near-term (next 1-2 weeks)
4. **Option D: Sleep replay** — leverages existing cluster D v2 infrastructure.
   Should improve generalization and reduce per-seed variance.
5. **Option E: Bigger regions + sparser** — straightforward scaling once
   above options stabilize.

### Longer-term (months)
6. **Option F: Multi-time-scale plasticity** — substantial bridge change,
   but biologically essential for long-term learning.
7. **Option G/H: Predictive coding + joint attention** — research-grade
   architectural changes.

## What NOT to spend time on

Based on overnight negative findings:
- Reward shaping (`wrong_move_reward=0`): NEGATIVE
- Stronger drives (lang_in 200→400, eval 200→500): NEGATIVE
- Naive bigger pools (10→30 motor neurons): NEGATIVE  
- Longer training alone (100→200 ep): NEGATIVE — weights saturated
- Reeval sweeps: cold-start state divergence makes them unreliable

These are exhausted. Don't repeat.

## Best single immediate action

If I had to pick ONE thing for the next 24 hours: **Option A (Curriculum)**.

Why: it directly addresses the LEVERAGE POINT (cascade quality) using a
well-established biology-grounded principle (developmental scaffolding).
The implementation is mostly orchestration on top of existing plasticity
gating infrastructure. The expected payoff (20-40 pp) is much larger than
any other single change at comparable effort.

Concrete first step: design a 2-phase training runner where phase 1 has
all language-pathway plasticity gates set to 0.0 (frozen), phase 2 unlocks
them. Same architecture, just different gating per phase.

The curriculum experiment also has no risk of breaking existing v2 result —
if it doesn't help, we still have 28.5% W→A.

## Open question for user

Which path appeals most? Curriculum (Option A) is my recommendation for
maximum leverage. But if you want quick eval-only wins first, Option C
(population vector decoding) is half a day and could surface signal we're
already throwing away.
