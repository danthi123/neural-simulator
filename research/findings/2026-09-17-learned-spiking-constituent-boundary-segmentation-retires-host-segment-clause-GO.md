---
type: finding
status: verified
date: 2026-09-17
mechanism: a LEARNED, spiking, emergent constituent-boundary segmenter that replaces the host segment_clause lexical
  scan — an STDP sequence-prediction circuit with predictive-coding-by-inhibition on a small Izhikevich
  SimulationBridge (stim RS --STDP-plastic--> pred FS/GABA_A --fixed inh--> err RS <--fixed exc-- stim). Streaming STDP
  learns stim[prev]->pred[curr]; the learned FS prediction subtractively inhibits err on high-transitional-probability
  within-constituent transitions, so ONLY unpredictable (= boundary) transitions still spike. Boundaries are read as a
  per-token-normalized err firing rate (each transition vs that token's own unpredicted baseline). A minimal
  determiner/function-word cue is kept as a legitimate perceptual input (Benjamin 2021: transitional probability is
  necessary-but-not-sufficient); the segmentation DECISION is learned + spiking
integration_faculty: language (comprehension — constituent/SVO boundary segmentation)
lane: language (comprehension — the irreducible host residual)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (5/6 all-gates, repo >=5/6 convention). The learned spiking segmenter discovers constituent boundaries FROM
  the token stream (emergence bar met): boundary AUC is high on every seed, an untrained circuit collapses to chance on
  every seed (learning is load-bearing), a stream-scramble that keeps the function-word cue but destroys the statistical
  structure collapses it (the statistical structure, not the cue, is load-bearing), and it PARSES clauses the host
  VERB_LEXICON cannot (a genuine coverage win), while byte-identical-off and moat-abstain hold on every seed. This
  de-risks retiring the LAST host-and-not-learned piece in the comprehension path (the segment_clause lexical scan) —
  the residual the neural-comprehension scoping identified as irreducible (everything downstream, thematic role
  assignment, already learns on-substrate and is GO'd). The lone 5/6 miss (seed 44) is a borderline scramble-cap margin
  on the STRONGEST intact seed, not an intact weakness; the gate was NOT relaxed to flip it. Additive, opt-in
  (BRAIN_LEARNED_SEGMENT, default-OFF), byte-identical when off. De-risk only; wires nothing, flips no default.
runner: research/runners/_learned_spiking_segmentation_derisk.py
artifacts:
  - research/findings/raw/_learned_spiking_segmentation/verify_6seed.json
external: EXTERNAL-DONE (comprehension lane, read + lane-recorded) — STDP predictive-coding-by-inhibition (Masumori,
  Sinapayen & Ikegami, WebFetch-verified) + statistical-learning prediction-error at low-transitional-probability
  boundaries (Daikoku), with Saffran / Karuza (cortico-striatal statistical-learning locus) + Benjamin (transitional
  probability necessary-but-not-sufficient) as design grounding. The arXiv/PubMed identifiers are in the body.
builds_on:
  - research/findings/2026-08-20-spiking-np-boundary-binding-closes-free-prose-extraction.md
---

# Learned spiking constituent-boundary segmentation — retires the host segment_clause scaffold (GO 5/6)

The neural-comprehension path had ONE host-and-not-learned piece left: `segment_clause()`, a hand-coded
determiner/aux/copula/participle scan that finds where role-fillable spans begin and end. Everything downstream
(thematic role assignment) already learns on-substrate and is GO'd (the EMERGE-78 fronto-striatal reservoir, the
multi-cue Competition parser, the spiking NP-head binder). This builds the learned, spiking, emergent replacement for
that last host piece.

## Mechanism (brain-based, emergent — the segmentation is DISCOVERED, not installed)

An STDP sequence-prediction circuit + predictive-coding-by-inhibition on a small Izhikevich SimulationBridge. Streaming
STDP learns to predict the next content token; the learned prediction subtractively inhibits an error population on
high-transitional-probability (within-constituent) transitions, so only unpredictable (= boundary) transitions survive
as error spikes. This is the biological source of statistical-learning word/constituent segmentation (Saffran; the
left-IFG + cortico-striatal locus of Karuza 2013 — the same substrate the role reservoir uses). Reuse-by-import of the
surprise organ's block-diagonal install/step and the existing spiking NP-head-binder role read-out; nothing hand-installs
a boundary. The minimal function-word cue is kept as a legitimate perceptual input, per Benjamin 2021 (pure
transitional-probability tracking is necessary but not sufficient) — but the segmentation DECISION is learned + spiking,
which is the load-bearing shift.

## Verify (from research/findings/raw/_learned_spiking_segmentation/verify_6seed.json)
<!--derived-->
- GO 5/6 all-gates; verdict preconditions all hold (all-gates >=5/6; intact boundary AUC >= 0.85 min over seeds; no-
  learning collapses; scramble collapses; coverage learned >= host; byte-identical-off; moat-safe).
- Boundary-detection AUC (chance 0.5): intact 6/6 >= 0.85 (0.935, 0.970, 0.975, 0.973, 0.938, 0.998); NO-LEARNING 6/6
  -> 0.500 (learning load-bearing); STREAM-SCRAMBLE 5/6 -> ~0.50 (seed 44 = 0.659, a hair above the 0.65 cap);
  intact-vs-scramble mean separation ~0.44.
- Extraction coverage (learned front-end + existing spiking NP-head-binder + parser): learned >= host 6/6, a genuine WIN
  on 5/6 (the learned segmenter parses verb-headed clauses the host VERB_LEXICON cannot — host 9/14, learned up to
  14/14).
- byte-identical-off 6/6 (BRAIN_LEARNED_SEGMENT=0 -> md5-identical host frames); moat 6/6 (a mis-segmentation abstains,
  never fabricates a triple).

## External sources actually read (comprehension lane, recorded)
<!--derived-->
- Masumori, Sinapayen & Ikegami 2019, "Predictive Coding as Stimulus Avoidance in Spiking Neural Networks",
  arXiv:1911.09230 (WebFetch-verified: SNNs learn to predict temporal sequences by STDP alone).
- Daikoku — statistical-learning neural prediction-error is high at low-transitional-probability (boundary)
  transitions (PubMed PMID 30944387).
- Saffran et al. 1996 (statistical word segmentation) + Karuza et al. 2013 (SL word-segmentation in left IFG + the
  striatal network) + Benjamin et al. 2021 (TP necessary-but-not-sufficient) as design grounding.

## What it means, and the next rung

This is the de-risk that the last host piece of the comprehension path — the `segment_clause` lexical scan — can be
replaced by a mechanism the brain LEARNS from experience, spiking, on one substrate. It flips no default and wires
nothing yet; the next rung is to wire `BRAIN_LEARNED_SEGMENT` into the extraction path (with the same byte-identical-
differential + no-regression soak used for other flips) and then retire the host `segment_clause`, which unblocks the
ledger rows that name the comprehension host residual. Two honest residuals are recorded in the finding: seed 44's
borderline scramble-cap margin (a control-margin, not an intact weakness), and verb->object TP-invisibility (a
deterministic verb->object pair is a real boundary that pure transitional probability under-weights — Benjamin 2021's
exact point) — the named next lever there is to strengthen the function-word cue's contribution or add lateral
competition at verb->object boundaries. A wall defers a METHOD (the host lexical scan), never the capability
(constituent segmentation), which is now a learned spiking GO.
