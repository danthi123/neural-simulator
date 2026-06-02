# Cheap-first probe: the LEARNING RULE is NOT the missing generalization mechanism -- 2026-06-02

## Context
Owner redirect: stay biology-faithful, treat negatives as missing mechanisms (use the catalog). Today's
generative negative (spiking LM, held-out perplexity ~200K = overfit) used BPTT, which `biology.md` flags as
non-biological. Hypothesis under test: the missing mechanism is the brain's biological LOCAL learning rule
(apical-basal dendritic / predictive coding), and it would generalize where BPTT overfit.

## Cheap-first probe (research/findings/raw/_pc_vs_bptt_probe.py, CPU/numpy)
Isolate the LEARNING RULE on identical task/architecture/capacity, measure train-vs-held-out generalization.

**v1 -- hard compositional task** (next = (f[a]+g[b]) mod V, held-out = unseen pairs):
- backprop: train 0.05 (memorized), held-out 5.9 -- WORSE than random 2.49 (catastrophic overfit).
- feedback alignment (cheapest biological local rule): held-out 7.2 -- no better, slightly worse.
- Controls clean (untrained ~ chance; shuffled-target overfits). -> "any local rule generalizes" FALSIFIED.

**v2 -- achievable task** (next = f[a], b a distractor, held-out = seen-a/new-b):
- generic MLP + backprop: held-out 0.033 -- GENERALIZES fine.
- structured (separate a|b pathways) + backprop: held-out 0.061 -- also generalizes.
- Both far below chance; neither overfits.

## Honest conclusion (the probe did its de-risking job)
The variable that flipped overfit <-> generalize was **task difficulty**, NOT the learning rule and NOT the
structured-vs-generic architecture I tested:
- On the HARD task everything overfits; on the EASY task everything generalizes.
- backprop, feedback alignment behave the same; and predictive coding is mathematically ~ backprop
  (Whittington-Bogacz 2017) so it would behave the same too.

=> **The biological LOCAL learning rule (dendritic / predictive coding) is NOT, by itself, the missing
mechanism for generalization.** It makes learning biologically plausible (local, no weight transport) -- a real
virtue -- but it does not, on its own, close the generalization gap that overfitting opens. The multi-month
spiking-dendritic build is NOT justified by a "it fixes generalization" rationale. This is the honest negative
the cheap-first probe was designed to catch BEFORE the big build -- saving the multi-month investment.

## The better-grounded direction (the project's OWN evidence, not abstract MLPs)
Abstract MLP probes are confounded by task difficulty and are a poor proxy for the real question. The more
informative evidence is the project's own working-vs-failing contrast:
- **VSA composition WORKS** (320 concepts, generalizes to novel sentences) -- it uses STRUCTURED, DISTRIBUTED
  codes + binding/unbinding, and generation-by-COMPOSITION (assemble concepts), not next-token prediction.
- **The generative LM OVERFIT** -- generic spiking MLP + BPTT + next-token prediction over a flat output.

The difference is the REPRESENTATION + the COMPUTATION (structured/distributed/compositional vs generic
next-token), not the learning rule. Biologically this is right: the brain does not generate language with a
generic next-token net; it composes (dual-stream model; structured sequence generation through cortex/BG/
hippocampus; Pulvermuller distributed ensembles).

## Refined hypothesis -> next research step
The missing mechanism for biology-faithful GENERATION is **compositional / structured generation on the
distributed-code substrate that already works** -- i.e. generate language by COMPOSING and SEQUENCING concepts
(reusing the validated VSA bind/unbind + sequence mechanisms: SWR replay, theta-gamma ordering), rather than
next-token prediction over a generic net. This is catalog-grounded (sequence generation, hippocampal replay,
compositional language) and builds on the project's WORKING composition rather than a generic MLP.

Next: study the catalog's SEQUENCE-GENERATION + language-PRODUCTION mechanisms (Indefrey word production,
hippocampal/cortical sequence generation, theta-gamma ordering) and design a biology-faithful
generate-by-composition probe on the existing distributed substrate -- the cheap-first test of THIS refined
hypothesis, before any big build.

## Discipline
Honest negative propagated (the cheap probe correctly de-risked the dendritic-rule premise before a multi-month
build). No protected-module change; CPU/numpy; reuse-by-import. The dendritic LOCAL rule remains valuable as the
biology-faithful TRAINING method for whatever architecture we build -- it's just not the generalization lever.
