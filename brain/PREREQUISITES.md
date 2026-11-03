# brain/ prerequisites: what the biology needs before a capability can be tested

Purpose: an INDEX into assets we already have (not a new map). It adds only two things per capability: our status in
`brain/` and whether a test is informative yet.
Use: before any test, read the capability's readiness; a failure with NOT-READY prerequisites is recorded as expected, not as a verdict.

Path keys. `CAT` = `/home/dant123/Projects/sim-catalog/references/feature-catalog.md` (entry ids like D.13);
`ROAD` = `/home/dant123/Projects/sim-catalog/references/biology-buildout-roadmap.md`;
`BIO/` = `/home/dant123/Projects/sim/research/biology/` (46 one-page bindings);
`MASTER` = `/home/dant123/Projects/sim/docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` (section 2 faculty map);
`F/` = `/home/dant123/Projects/sim/research/findings/`; Kandel = Principles of Neural Science 6e (RAG `--corpus kandel`;
full text `/home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`).
Brain-side files: `brain/hippocampus.py`, `brain.py`, `cortex.py`, `SCAFFOLDS.md`. Scale ladder:
`brain/experiments/results/2026-11-02-scale-ladder.txt` (0.1B / 0.4B / 1.5B). Brain-side record brief:
`brain/research/2026-11-02-record-brief-consolidation-and-selection.md`.

## Episodic recall
Biology:
- CAT D.01 episodic cycle; D.12 pattern separation; D.13 pattern completion; D.20 reactivation supports retrieval.
- BIO/dg-ca3-sparse-index.md, BIO/ca3-superposed-fact-attractor.md, BIO/awake-replay-pattern-completion.md.
- MASTER section 2.4 "Episodic STORAGE / retrieval (CA3 completion)"; Kandel Ch 52, 54-56 (via CAT D.01).
In brain/: have: one-shot store of an experience as a cortical state delta, cued recall by cosine, reinstated as state (`hippocampus.py` store/recall/reinstate).
In brain/: code-substitute: centred cosine on the layer-0 delta stands in for DG/CA3 completion; the "is it a telling" gate is a word list (`brain.py` `_is_telling`, SCAFFOLDS.md).
In brain/: missing: semantic (paraphrase-robust) cue; sequence/time context; separation of near-duplicate episodes.
Readiness: PARTIAL. Informative: same-words cue, fresh single facts, restart survival. Expected failure: paraphrase cues, many near-duplicate facts.
Scale note: paraphrase selection 8/12 at 0.1B, 1/12 at 0.4B, 5/12 at 1.5B (non-monotone, so a paraphrase result at one size is not a mechanism verdict); same-words 12/12, 7/12, 7/12.

## Memory strength and forgetting
Biology:
- CAT J.34 memory imperfections (schemas, gist); J.38 late-LTP; J.27 reconsolidation.
- BIO/repetition-retrieval-strengthen-same-trace.md, BIO/encoding-strength-and-allocation.md, BIO/gist-detail-graded-forgetting.md, BIO/importance-tagging-at-encoding.md.
- MASTER section 2.4 "Forgetting (active/adaptive)" and "Reconsolidation"; owner rule: measure savings, cued recall, transfer, never perfect retention.
In brain/: have: per-episode strength, power-law decay, use boost, permanent identity episodes (`Hippocampus.effective_strength`, `USE_BOOST`).
In brain/: code-substitute: decay constants are hand-set (TAU 1 day, exponent 0.5); salience is a number passed in, not computed.
In brain/: missing: re-telling strengthens the SAME episode (store appends a new one); gist vs detail; savings measurement; surprise/affect-driven salience (`modulators.py` not built).
Readiness: PARTIAL. Informative: use-strengthening and decay curves over simulated days; high-use facts stay correct. Expected failure: savings and gist (no consolidation yet), salience-from-surprise.
Scale note: not scale-sensitive in the ladder; strength logic is outside the LM, so any size gives the same mechanism result.

## Sleep consolidation
Biology:
- CAT N.12 sleep-dependent consolidation; N.14 hippocampal-neocortical dialogue; N.07 sharp-wave ripples; N.05/N.06 slow oscillation and spindles.
- BIO/systems-consolidation.md, BIO/cls-interleaved-consolidation.md, BIO/prioritized-replay-triage.md, BIO/sleep-load-dependent-renormalization.md.
- ROAD T1.B (sharp-wave ripple replay); record brief A1-A4 (prior results, traps, first experiment: distilled reinstatement adapter).
In brain/: have: `plastic_cortex.py` (trainable cortex, low-rank feed-forward adapters, equals the pretrained cortex at zero); `sleep.py` (strength-weighted interleaved replay + KL protection); `experiments/sleep_probe.py` (hippocampus-off recall, paraphrase, false recall, generic drift; no-sleep and shuffled arms).
In brain/: code-substitute: gradient descent on the adapters stands in for a local plasticity rule; replay uses the stored episode text (both in SCAFFOLDS.md).
In brain/: missing: varied replay (paraphrase/recombination), sleep wired into the live brain's restart, persistence of the adapters, the high-use-facts measure.
Readiness: PARTIAL. Informative: hippocampus-off recall vs no-sleep and shuffled arms, relative to the cortex's own in-context ceiling (0.1B 6/12, 0.4B 11/12). Expected failure: reworded-question recall from single-form replay at 0.1B (knowledge stated once is hard for small LMs to extract).
Scale note: operating points did not survive a scale change in the old track (record brief A3.6); derive gain per model size on dev data.

## Memory selection (pattern completion and separation)
Biology:
- CAT D.12 pattern separation; D.13 pattern completion; D.05 CA3 recurrent collaterals; D.14 engram cells.
- BIO/dg-ca3-sparse-index.md, BIO/ca3-superposed-fact-attractor.md, BIO/spiking-closed-class-pattern-completion.md, BIO/conjunction-competitive-selection.md.
- Record brief B (semantic episode selection); MASTER section 2.4 "Episodic STORAGE / retrieval".
In brain/: have: winner-take-all by strength-weighted centred cosine with a margin and familiarity floor.
In brain/: code-substitute: layer-0 delta key is lexical, so the key is word overlap in disguise (SCAFFOLDS.md row on pattern completion).
In brain/: missing: semantic key (deeper-layer or learned readout); false-recall rate measure; separation of confusable facts.
Readiness: PARTIAL. Informative: same-words cue and wrong-cue false-recall rate. Expected failure: paraphrase cue until the semantic key is built.
Scale note: selection same-words 12/12 at 0.1B but 7/12 at 0.4B and 1.5B; paraphrase 8/12, 1/12, 5/12. The key depends on the LM, so rerun the ladder after any key change.

## Working memory and topic holding
Biology:
- CAT G.06 / G.08 PFC working memory; G.09 imagination as constructive memory; G.03 object-based attention.
- BIO/wm-referent-focus-retrieval.md, BIO/attention-gated-readout.md.
- MASTER section 2.3 WORKING MEMORY (maintenance, capacity, gating).
In brain/: have: the cortex's recurrent state carries the conversation (`Brain.state` updated each turn).
In brain/: code-substitute: the live state IS working memory; there is no gating, capacity limit or separate maintenance.
In brain/: missing: topic tracking across long turns, interference handling, explicit hold/release.
Readiness: PARTIAL. Informative: in-context recall over a few turns; whether a reinstated memory stays usable next turn. Expected failure: long-span topic holding beyond the model's native context behaviour.
Scale note: in-context recall 6/12 at 0.1B, 11/12 at 0.4B and 1.5B; the 0.1B result here reflects the LM, not the design.

## Language
Biology:
- CAT G.10 hierarchical language; G.11 dual-stream; G.12 Broca; G.13 Wernicke; L.23 critical period for language.
- MASTER section 2.6 LANGUAGE + REASONING (comprehension, production, lexicon).
- Kandel language chapters: look up via CAT G.10-G.13 source lines (chapter not verified here).
In brain/: have: pretrained RWKV-7 speaks and reads (`cortex.py` read/speak).
In brain/: code-substitute: language is gradient-pretrained (declared trade, ARCHITECTURE.md rule 4), not learned in the brain.
In brain/: missing: learning of new words/constructions after birth; teacher-correction loop; fluency number on the weekly test.
Readiness: READY (as the declared pretraining trade). Informative: fluency and register of replies at a given size. Expected failure: anything asking the brain to acquire new grammar.
Scale note: fluency depends on size (0.1B is weak, 1.5B is the reference fluent size); judge relative to scale.

## Affect shaping speech
Biology:
- CAT O.01 amygdala; O.12 amygdala fear circuit; O.13 amygdala reward role; C.04 dopamine, C.05 norepinephrine.
- BIO/interoceptive-affect.md, BIO/affect-active-clear.md, BIO/locus-coeruleus-ne-adaptive-gain-swap.md.
- MASTER section 2.5 EMOTION / MOTIVATION / REWARD-VALUE (core affect region, emotion biases cognition); `F/2026-07-24-P0.3-affect-state-region-6seed-GO.md`.
In brain/: have: `affect.py` (default OFF): innate primaries -> valence axis in the cortex's fast memory, appraisal -> slow mood, additive signed bias on the matrix memory; probe `experiments/mood_probe.py`; design `brain/research/2026-11-03-mood-state-design.md`.
In brain/: code-substitute: innate primaries are a declared body scaffold (balanced bodily/social); appraisal is a linear two-pole read of the deep-layer fast-memory trace. Iteration 2 (11-03): held-out appraisal works 5/6 seeds; negative mood lesion-proven 6/6, positive 3/6 (`results/2026-11-03-mood2-0.1b.md`). Iteration 3 (mid-scale register, random-direction null): positive 6/6 at 0.1B but echo fails 2 seeds; 0.4B positive 1/6 vs null; negative partly a refusal mode (`results/2026-11-03-mood3.md`).
In brain/: missing: arousal acting on the cortex (NE gain), surprise/novelty input to arousal, salience to encoding gain, wiring into `Brain.hear`, a body model behind the primaries.
Readiness: PARTIAL. Informative: lesion changes tone of replies to the same question (mood probe) and, once wired, on the weekly prompts. Expected failure: appraisal of subtle or positive events (linear lexical read, no learned appraisal), arousal effects (not acting yet), any test needing a body.
Scale note: tone control is partly an LM property; check a small and a large size before attributing a tone change to the affect part.

## Curiosity
Biology:
- CAT D.23 misplace system (novelty drives exploration); C.32 two-component DA response; C.27 wanting vs liking.
- BIO/spiking-habituation-novelty.md, BIO/spiking-confidence-synaptic-integration.md.
- MASTER section 2.8 "Curiosity / intrinsic motivation"; `F/2026-07-23-DR1-curiosity-inversion-6seed-GO.md`.
In brain/: have: the cortex's own prediction error is available (logits) but unused; hippocampal familiarity score exists.
In brain/: code-substitute: none.
In brain/: missing: a surprise signal, a gap-to-want drive, questions asked because of it, a place to ask (the chat loop only answers).
Readiness: NOT-READY. Informative once surprise exists: does the brain ask about unfamiliar topics more than familiar ones. Expected failure: curiosity probes today.
Scale note: untested; prediction-error signal quality should be checked per size before use.

## Metacognition
Biology:
- CAT G.20 consciousness (arousal and content access); G.15 signal-detection decision rule; D.23 novelty detection.
- BIO/spiking-confidence-synaptic-integration.md, BIO/sacramento-selfpredicting-microcircuit.md.
- MASTER section 2.7 (first-order uncertainty monitor, second-order metacognition); `F/2026-06-11-familiarity-gate-v320-GO.md`, `F/2026-08-18-self-organized-metacog-monitor-GO.md`.
In brain/: have: the recall familiarity score and margin (a first-order "do I know this" signal) in `Hippocampus.recall`.
In brain/: code-substitute: fixed thresholds FAMILIARITY / MARGIN decide recall vs abstain.
In brain/: missing: a reported confidence, a commit/hedge/ask routing, honest read-out wording ("my familiarity reads this as new").
Readiness: PARTIAL. Informative: abstains on never-told facts instead of confabulating. Expected failure: calibrated second-order confidence.
Scale note: untested across the ladder; larger LMs confabulate differently, so measure false-recall per size.

## Self and identity
Biology:
- CAT P.05 hippocampal disorders of autobiographical memory; D.15 CA2 social memory; G.09 imagination and future simulation.
- BIO/systems-consolidation.md; BIO/importance-tagging-at-encoding.md.
- MASTER section 2.4 "Autobiographical / self memory" and 2.7 "Self-schema", "Narrative/autobiographical self"; `F/2026-07-23-DR3-self-schema-region-6seed-GO.md`.
In brain/: have: three seeded identity exchanges stored as permanent episodes (`Brain._grow_self`); name and friend persist across restart.
In brain/: code-substitute: identity is seeded at birth by host code (SCAFFOLDS.md), not learned.
In brain/: missing: identity learned in conversation and consolidated first; self/other tag; narrative of past turns.
Readiness: PARTIAL. Informative: own name and friend's name after restart and after many turns. Expected failure: any claim that identity is learned or self-modelled.
Scale note: identity Q/A traces loop above reinstatement gain ~4.5 (hippocampus.py GAIN comment); retune per size.

## Action selection and skills
Biology:
- CAT A.01-A.04 basal ganglia direct/indirect/hyperdirect and WTA; A.09 goal-directed vs habitual; O.04 goal-directed to habitual transition; J.33 implicit memory taxonomy.
- BIO/conjunction-competitive-selection.md, BIO/question-route-selection-wta.md.
- MASTER section 2.8 "Three-factor neuromodulated plasticity + reward"; ROAD T0.B (gpe to gpi pathway).
In brain/: have: nothing (the brain only converses; no body, no tasks).
In brain/: code-substitute: none.
In brain/: missing: choices with consequences, reward signal, skill learning, a place to act (tools or a world).
Readiness: NOT-READY. Informative: nothing yet; there is no action space. Expected failure: any skill or habit test.
Scale note: not applicable until an action space exists.

## Lifelong learning
Biology:
- CAT J.28 LTP/LTD; J.38 late-LTP; J.34 memory imperfections (schemas, gist); L.04 critical periods.
- BIO/cls-interleaved-consolidation.md, BIO/homeostatic-scaling-relative-strength.md, BIO/semantic-store-cortical-capacity.md.
- MASTER section 2.8 "Consolidation as offline credit"; record brief A (stability/plasticity trade).
In brain/: have: persistence of episodes and strength across restarts (`Brain.save/load`); fast memory grows with every telling.
In brain/: code-substitute: growth is episode count, not weight change; the cortex itself never changes.
In brain/: missing: consolidation into weights, forgetting that keeps language intact, transfer/schema measures, a long-run (weeks) test.
Readiness: NOT-READY (needs sleep first). Informative: multi-day store growth and recall over a simulated month. Expected failure: savings, transfer, schema formation.
Scale note: capacity and interference grow with model size; do not extrapolate a 0.1B result.

## Theory of mind
Biology:
- CAT H.18 mirror neurons; D.15 CA2 social memory; G.09 imagination.
- BIO/joint-attention-gaze-following.md, BIO/common-ground-ledger.md.
- MASTER section 2.7 (joint attention, common ground, belief attribution); `F/2026-07-24-W3-false-belief-register-6seed-GO-adversarially-verified-immunity-claim-corrected.md`.
In brain/: have: a "friend" name and one speaker label ("User") in memory.
In brain/: code-substitute: the LM's pretrained social knowledge, unmeasured.
In brain/: missing: a model of what the friend knows or believes, self/other tags on memories, multiple speakers.
Readiness: NOT-READY. Informative: none now. Expected failure: false-belief or perspective tests; at small LM sizes even the pretrained version fails.
Scale note: pretrained theory-of-mind behaviour is strongly size-dependent; check at 1.5B before concluding anything.
