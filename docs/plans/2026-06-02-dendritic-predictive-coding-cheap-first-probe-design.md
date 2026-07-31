---
type: plan
status: live
date: 2026-06-02
---

# Dendritic / predictive-coding cheap-first generalization probe — design — 2026-06-02

## Premise (owner redirect)
Owner: stay 100% biology-faithful; pursue conversation AND artificial life; treat every negative as "we
haven't implemented the right brain structure yet"; use the reference catalog. Today's generative negative
(subword spiking LM, held-out perplexity ~200K = token-soup, OVERFIT) used **backprop-through-time (BPTT)**,
which `docs/biology.md` itself flags as non-biological ("Real brains have no equivalent of backpropagation
through time. They learn from local rules.").

## The missing structure (catalog-grounded)
Apical-basal dendritic neurons + predictive coding, learning by LOCAL rules:
- Pyramidal neurons are two-compartment: **basal** dendrites integrate bottom-up drive; **apical** dendrites
  integrate top-down predictions/feedback; plasticity at basal synapses is **gated by apical activity**
  (Bono-Clopath 2017; Sacramento 2018; Payeur-Naud 2021 burst-dependent plasticity).
- This implements **predictive coding** (Rao-Ballard 1999; Whittington-Bogacz 2017): each region predicts its
  input, learns from the local prediction error — close to what backprop computes but with LOCAL rules only.
- The project already designed this (`docs/plans/2026-05-05-dendritic-learning-design.md`) as the fix when
  global-scalar feedback failed, then deferred it (~multi-month). The catalog repeatedly flags
  "single-compartment ... cannot exhibit dendritic ..." and "column structure missing".

Why it should fix today's failure: predictive coding learns the GENERATIVE structure (it predicts inputs), so
it should GENERALIZE, exactly where BPTT memorized train and failed held-out.

## Goal of THIS doc: cheap-first validation BEFORE the multi-month build
Prove (or falsify) the load-bearing hypothesis cheaply, so the expensive biology-faithful spiking build is
only undertaken if the principle holds.

## Hypothesis (pre-registered, frozen)
On a small next-token prediction task with a held-out split, an apical-basal predictive-coding learner (local
rule) achieves a SMALLER train-vs-held-out generalization gap than a same-capacity BPTT learner (today's
failure mode). Decisive bar: PC held-out loss < BPTT held-out loss AND PC gap < BPTT gap, multi-seed.

## Design
- **Task:** a controlled next-token task with a clear generalization gap. Two variants for robustness:
  (a) a SYNTHETIC structured grammar (e.g. a small probabilistic regular/context-free grammar) where
  memorizing train sequences != generalizing to held-out sequences from the same generator — cleanest
  control; (b) a TINY real char-level TinyStories slice — directness to the actual goal. Start with (a)
  (clean signal), confirm direction on (b).
- **Learners (identical task / capacity / data; ONLY the learning rule differs):**
  1. **BPTT baseline** — reuse `sim/bptt_snn.py` (hand-written BPTT, the non-biological control).
  2. **Apical-basal predictive coding** — NEW minimal module: two-compartment units; basal integrates input;
     apical carries the top-down prediction error (target = actual next token, self-supervised); basal
     weight update = Hebbian/delta gated by apical activity (the 2026-05-05 rule). LOCAL only — no backprop.
- **Controls (anti-cheat):** an untrained/random net (both learners must beat it); the SAME task/capacity/data
  for both; pre-registered generalization metric; multi-seed. A "memorization control" (train on shuffled
  targets) confirms the task has a real generalization structure to learn.
- **Metric:** held-out next-token loss (perplexity) + train-vs-held-out gap. Frozen bar above.
- **Scale:** tiny (CPU/numpy), fast iteration; faithful to the apical-basal STRUCTURE + local rule, small
  scale for speed (validation purpose — owner-sanctioned shortcut for testing).

## Outcomes (three-state)
- **RESOLVES** (PC generalizes, gap << BPTT, multi-seed): the missing-mechanism hypothesis is validated ->
  proceed to the biology-faithful SPIKING multi-compartment build (the 2026-05-05 design), scaling toward
  generation. This is the path to biology-faithful conversation.
- **BOUNDARY / DOES-NOT-RESOLVE** (PC overfits too, gap ~ BPTT): the hypothesis is wrong at this scale ->
  honest NEGATIVE (saves the multi-month build) -> re-examine the catalog for a different missing mechanism
  (candidates: sparse coding, complementary-learning-systems consolidation for generation, theta-gamma
  sequence structure, cortical-column microcircuit).
- **CANNOT-CONCLUDE** (instrument-invalid: controls fail, task has no generalization structure) -> fix the
  instrument, do not propagate.

## Reuse / discipline
- Reuse `sim/bptt_snn.py` (BPTT baseline) + corpus/tokenizer infra for variant (b). NEW = a minimal
  `predictive_coding` local-learning module (NOT autograd — a hand-written local rule, which is the
  sanctioned new mechanism the owner asked for).
- No external deps. No protected/frozen/moat module change. Honest propagation of every outcome to both
  remotes. Scrutinize a PASS harder than a FAIL (a PC "win" must beat the controls + be multi-seed).

## After this probe
If RESOLVES: writing-plans for the spiking multi-compartment apical-basal build (extend the 2026-05-05 design),
integrate into the bridge, scale toward sequence generation. If NEGATIVE: catalog re-examination for the next
candidate missing mechanism. Either way: an honest, biology-faithful step on the owner's goal.
