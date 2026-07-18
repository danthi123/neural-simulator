# Gap #3 residual A1 — the host `content_bias_target` feature-lexicon EMERGES from corpus co-occurrence (cheap-first 6-seed GO)

**2026-07-18.** Per the owner's easiest-first strategy correction (fully close each gap before the harder ones), gap #3
(multi-referent disambiguation) is largely closed — the biased-competition WTA + D3 composed-focus are 6-seed GO and
wired. Its EMERGENCE-BAR residual A1: the `content_bias_target` host op (+ the `ANIMACY` / `VERB_SELECTS` feature
lexicons) that decides WHICH held referent gets the bias current is a HOST LOOKUP, flagged in-module for conversion to
a learned synaptic feature-compatibility map. This de-risks that the feature-compatibility is LEARNABLE (not host).

## Result — 6-seed GO
`research/runners/_gap3_learned_feature_compat_derisk.py` (numpy cheap-first). From an SVO corpus, JOINTLY learn each
concept's ANIMACY and each verb's SELECTIONAL preference by iterative co-occurrence (EM-like: a concept is animate iff
it is the patient of animate-selecting verbs; a verb selects-animate iff its patients are animate; + a weak
agents-tend-animate prior). The learned feature-compatibility then reproduces the host disambiguation:

| metric | mean(6 seeds) |
|---|---|
| learned == host `content_bias_target` (resolvable pronoun cases) | **1.00** |
| PERMUTED-corpus (shuffle patient animacy) anti-cheat | **0.00** (collapses) |

The learned path never reads `ANIMACY`/`VERB_SELECTS` (used only to generate the corpus + as eval ground truth). The
permuted-corpus control collapsing to 0.00 proves the feature-compatibility is CORPUS-DERIVED, not smuggled. ⇒ the host
lexicon is replaceable by a learned map: which animacy a verb selects + which animacy a concept has both EMERGE from
the distributional SVO structure (Bates-MacWhinney cue-from-distribution).

## Remaining A1 work (the spiking realization + wire-in)
The MECHANISM is validated (feature-compatibility is learnable). To fully close A1 per BRAIN-BASED-ONLY: (a) realize the
learned features + the compatibility as SPIKING neurons (reuse the 2026-06-15 on-bridge Hebbian co-occurrence for the
features + the validated spiking coincidence for the compatibility), (b) route the compatibility output as the bias
current into the existing `BiasedCompetitionContextBuffer` WTA (replacing the host `content_bias_target` call), keeping
the 6-seed GO competition + the no-confab moat. Then gap #3 residual A1 is fully closed on spikes.

## SPIKING realization — GO (2026-07-18)
`research/runners/_gap3_spiking_feature_compat_derisk.py`: the compatibility READOUT is now computed by SPIKING neurons
on a real `SimulationBridge` — two feature-detector pools (F_anim, F_inanim); the query verb drives its learned
selection pool, each candidate its learned animacy pool; the COMPATIBLE candidate co-drives the verb's selection pool
(coincidence) → highest firing → the bias target. **spiking==host `content_bias_target` 1.00 (6-seed); permuted-corpus
anti-cheat collapses 0.00.** The learned signs are the offline-EM scaffold (like the concept codes); the DECISION is
spiking. Remaining A1: wire the spiking bias-target into `BiasedCompetitionContextBuffer.read(bias_concept=...)` in
place of the host `content_bias_target` call, keeping the 6-seed GO competition + moat → A1 fully closed.
