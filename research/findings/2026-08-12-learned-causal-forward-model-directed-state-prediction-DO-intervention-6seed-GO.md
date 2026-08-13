---
type: finding
status: live
date: 2026-08-12
mechanism: learned-causal-forward-model-directed-state-prediction
verdict: GO (6/6 seeds, runner-level de-risk; NOT yet wired/integrated to production)
lane: T1-4 · Learned causal forward model (the reasoning bottleneck)
artifacts:
  - research/findings/raw/_causal_forward_model_6seed.json
verification: >
  substrate seeded (cfg.seed set on all three RNGs); the runner's OWN tools.verdict.Verdict
  returns GO with all nine [ok] checks; the lesion HOLDS at measurement (enable_stdp +
  enable_reward_modulation frozen BEFORE the edges are zeroed, so they cannot regrow); the
  instrument was read in both directions (a saturating-runaway false regime and a
  no-propagation false-null were each caught by READING the block-rate reads + weight matrix,
  not the summary — see "the companion process"); every read is a cp_firing_states block-rate,
  no host transition table is consulted.
---

# A learned CAUSAL FORWARD MODEL on the spiking substrate — directed n-way state prediction, an UNSEEN consequence by forward-simulation, and CAUSE-vs-CORRELATION under a DO-intervention (6/6-seed GO, 2026-08-12)

## Result
The faculty audit's T1-4 ("the organ causal inference, counterfactual reasoning, and complete
deliberation all bottleneck on") is de-risked at a **6/6-seed GO** (local CPU, SIM_BACKEND=numpy,
seeds 42/43/44/100/101/102). The production "why" is today a HOST symbolic JOIN over stored
triples (`_causal_composition_chain_derisk.py`: `dir == obj_dir`) — a retrospective explanation,
no forward model, no intervention. This de-risk builds the missing rung: a **directed, queryable,
n-way STATE forward model** on the shared spiking substrate that

- **(a) predicts an UNSEEN consequence by forward-simulation** — hold state A; the substrate's own
  dynamics fire the successor B (150 Hz) AND the 2-step consequence D (98-100 Hz) though the
  direct A->D edge was NEVER taught (its weight stays at init 0.2), with every OFF-chain block
  (C,X,Y) silent (0 Hz). A host 1-step store given "A" returns only B; the substrate returns D by
  rolling its learned dynamics forward through the B intermediate. 6/6 seeds.
- **(b) recovers CAUSE vs CORRELATION under a DO-intervention** — after a common-cause confound
  (C->X, C->Y; X observed just before Y, so temporal-order STDP tags a SPURIOUS X->Y), a teacher
  DO-intervention prunes it: **do(X) -> Y = 0 Hz (X does NOT cause Y)** while **do(C) -> Y = 164-167
  Hz (C DOES cause Y)**. The genuine edges are untouched (C->Y and A->B constant at ~10; X->Y
  eroded 10 -> 0.5-0.7). 6/6 seeds.
- **(c) is a directed 1-step forward predictor** — cueing a state fires its LEARNED successor as
  the argmax on the chain edges (A->B, B->D), accuracy 1.00, DIRECTED (forward successor rate >>
  reverse predecessor rate ~0). 6/6 seeds.

This generalises the E2 affective world-model (a 2-channel VALENCE-sign predictor, 6/6 GO
2026-08-12) to a structured multi-event STATE prediction with directed edges + intervention — the
"n-way next-STATE recall" that E2's own finding named as its disabled next rung.

Runner: `research/runners/_causal_forward_model_derisk.py` (NEW; reuse-of-pattern, NO `sim/` edit).
Artifact: `research/findings/raw/_causal_forward_model_6seed.json` (+ provenance sidecar).

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._causal_forward_model_derisk \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_causal_forward_model_6seed.json
# fast 1-seed pipeline check: --smoke   (intact GO | lesion collapse | corr_only wrong)
```

## Mechanism (brain-based; a directed spiking next-state forward model)
One recurrent EVENT population `evt` (one block/assembly per event); cross-block edges weak +
plastic, NO within-block edges. Two brain-based factors make the transition edges DIRECTED and
CAUSAL:
1. **Temporal-order STDP** (Mehta-Blum-Abbott causal window; the gap#5 directed-band mechanism,
   6-seed GO): the teacher renders each experienced episode as an ORDERED pair (i then j), so
   block-i fires BEFORE block-j -> the asymmetric window tags i->j (pre-before-post LTP) and
   depresses j->i. Direction comes from temporal precedence.
2. **Phasic dopamine (DA-RPE)**, the substrate's three-factor rule
   (`reward_defer_stdp_weight_update`=True: STDP timing creates the eligibility TAG; a dopamine
   signal converts the tag to LTP/LTD). DA is held ON through each ordered episode, so the tag is
   converted AS it forms and the net never free-runs into spurious potentiation.

**The DO-intervention (Pearl do-calculus, teacher-scaffolded — T1-4's named mechanism).** In the
confound, observational learning tags the spurious X->Y (X precedes Y, both effects of C). The
teacher then forces **do(X) in isolation** (C absent): the learned X->Y fires Y (the false
prediction), and a NEGATIVE DA depresses the eligible X->Y (predicted-but-should-be-absent). The
chain edges and the genuine C->Y are never driven by X here, so they are untouched — the
invariance-across-interventions principle (Peters/Scholkopf) on spikes: only edges robust across
observation AND intervention survive.

**What is neural vs the declared boundary.** The prediction is a `cp_firing_states` block-rate
argmax (the n-way generalisation of E2's sign read); the forward-simulation is the substrate's own
propagation (D activates via B, A->D unlearned); the edges are STDP + DA three-factor weights. The
LEGITIMATE boundary (per T1-4 "teacher DO-interventions"): the teacher renders the event drive, the
temporal ORDER of the episode, and the phasic-DA SIGN (the teacher's reinforcement; the brain's own
dopamine channel converts it to a weight change). No host writes a transition table, computes the
prediction, or computes the causal verdict.

## Anti-cheats (all pass 6/6 or 3/3)
- **Lesion (load-bearing, decisive)** — zero the learned cross-block edges (learning frozen first,
  so it HOLDS): forward prediction collapses (acc 1.0 -> 0.5, argmax undefined), the unseen D -> 0,
  do(C) -> Y -> 0. 3/3. The whole faculty is carried by the learned edges.
- **Correlation-only control (the DO-prune is load-bearing)** — run WITHOUT the interventional
  phase: the spurious X->Y survives (weight ~10) so **do(X) WRONGLY fires Y (161-167 Hz), Xcause =
  True** — the model concludes X causes Y. WITH the intervention, do(X) -> Y = 0. The cause-vs-
  correlation separation is **106% attributable to the DO-prune** (the control moves OPPOSITE:
  do(X) fires Y HIGHER than do(C)). 3/3.
- **Shuffle (structure, not a fixed template)** — relabel the events by a non-identity permutation:
  the forward prediction scored vs the TRUE chain FAILS (acc 0.0-0.5, the true-role A->B/C->Y
  weights ~0.2 = unlearned). The model learns the structure it is SHOWN. 3/3.
- **Brain-based** — every read is a `cp_firing_states` block-rate; `current_reward_signal` is the
  teacher DA (declared boundary), zero during all reads; no host argmax over a stored table.

## The companion process (the wall-reframe, and two instrument saves)
Pure STDP+DA potentiation ran away exactly as CLAUDE.md warns: with a propagation gain, EVERY
co-active pair potentiated to w_max, so all edges became equal (A->D = A->B) and holding any block
ignited the whole net — the missing COMPETITION/normalization companion. The fix is gap#5's
protocol: **learn at LOW operating strength** (the net cannot self-ignite, so only the externally-
driven ORDERED pairs tag -> a selective structure, A->D stays ~0.2), then apply a **UNIFORM
maturation GAIN** for the reads (preserves the learned ratios; the SELECTIVITY is entirely the
learning's). Two instrument failures were caught by READING the substrate, not the summary: (1) a
saturating-runaway regime that fired all blocks at the refractory ceiling (read as spurious
"propagation"); (2) a no-propagation false-null (the successor never crossed threshold at the
training gain). A third bug — a phantom D->C edge — was a stale `cp_last_spike_time` cross-tagging
consecutive episodes (STDP pairs from it; my reset had missed it); resetting it per episode cleaned
the structure.

## Honest boundary + the next rungs (per THE LAW — first-class deliverables, not caveats)
- **The DA sign is teacher-delivered** (the environment boundary). Driving it from a SPIKING
  mismatch unit (E2's surprise read -> a from_reward/from_novelty DA) so the prediction-error is
  itself neural is the named next rung.
- **First-order transitions only** (state -> next). History-dependent / ambiguous-given-context
  transitions need the HTM-TM high-order predictor (EMERGE-15 GO) — a composition, not a new wall.
- **Forward-simulation is a held-cue steady-state propagation** (hold A -> B and its downstream D):
  a fully autonomous multi-step TEMPORAL rollout (a gap#5-style traveling packet through discrete
  event assemblies) is the refinement; the decisive property here (D activates via the intermediate
  though A->D is unlearned) already holds.
- **Events are delivered as block drive** (the environment boundary). Grounding them in the emergent
  relational/spatial code (2026-08-11 GO) is the follow-on that makes the states themselves learned.
- **The read-out is a firing-rate argmax** (a spiking read, like E2's sign read) — not a host
  computation of the prediction, but not yet a downstream spiking consumer either.

## Path to a production spiking "why / what-if" (the point of T1-4)
The production turn's "why" (`webapp/server.py`) can gain a co-resident forward-model organ, exactly
as Gate-B/E2 wired the valence forward model: on a "what happens if <state/action>?" query, cue the
state block and read the spiking successor (the what-if); on a "why did <state>?" query, read the
directed edge INTO the state (its cause) and confirm it survives a DO-probe. This is the rung a host
triple-JOIN cannot serve (it has no forward model and no intervention). Wiring it default-on +
moat-safe (reads/notices only, never manufactures a fact) is the integration follow-on.

## Provenance
`research/runners/_causal_forward_model_derisk.py` (modes: default 6-arm; `--smoke`; `--opsearch`).
Uses `tools.lab.attributable_to` + `tools.verdict.Verdict`. NO `sim/` edit. CPU/numpy (the
180-neuron model gets no GPU benefit — the E2 precedent, same scale, ran 6-seed on numpy CPU).
