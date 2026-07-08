# The KNOWLEDGE half of breadth, SPEAK rung (GO, 3-seed GPU): the brain SPEAKS its grounded answer ON SPIKES — the numpy breadth reasoner decides yes/no/idk, and the decision is produced as a spoken word decoded from `language_output` firing (the validated spiking A→W); on the unknown it abstains WITHOUT invoking the speaker (gate-first no-confab moat, 0 spike-renders). The full mission loop: discover→reason→SPEAK. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_speak_answer_onspikes_derisk.py` (reuse-by-import: the rung-4 emergent console + EMERGE-67 `NeuralSpell`, the validated spiking A→W read-out). Requires `SIM_BACKEND=cupy` (GPU). NO `sim/` edit.
**Verdict:** GO (3-seed GPU) — the mission-completing SPEAK rung of the breadth→knowledge arc.

## Why this ran (completing discover→reason→SPEAK)
The breadth→knowledge arc reasons end-to-end (discover a broad vocab from a real corpus → discover categories by clustering → teach a fact → answer a yes/no question about a held-out word → moat), but the answer was an internal token. This rung SPEAKS it on spikes: the decision is produced as a spoken word on a real `SimulationBridge` via the validated spiking A→W read-out (drive the answer's concept pool → decode the spoken word from `cp_firing_states[language_output]`). The full mission loop — discover, reason, and SPEAK — now runs end-to-end for the yes/no case.

## The result — 3-seed (42/43/44), GPU, TinyStories K=1024
- **spoken-accuracy 1.000** every seed — the reasoner's decision is faithfully produced as a distinct spiking-spelled word (decoded from `language_output` firing), every query.
- **moat-renders-on-abstain 0** every seed — on "idk" (an unknown word with no discovered code) the SPEAKER IS NEVER INVOKED (gate-first): the no-confab moat holds by construction (0 spike-renders on abstains).
- Transcript (seed 42): a held-out cluster member → reasoner YES → **spoke 'fly' ON SPIKES**; an other-category word → reasoner NO → **spoke 'swim' ON SPIKES**; 'zzzqqx' (unknown) → **"I don't know" [speaker not invoked, moat]**.

## The architecture (one process, gate-first)
The numpy breadth reasoner (pure numpy math — `learn_stream_codes` + the associative-memory read, no `SimulationBridge`) co-executes with the cupy spiking A→W (`NeuralSpell`, a real bridge) in ONE cupy process — the reasoner never touches the backend-global, so they coexist. GATE-FIRST: the reasoner decides first; only a yes/no decision invokes the speaker; an "idk" abstain routes to "I don't know" without invoking the spiking producer at all — so the moat is enforced BEFORE any speech, by construction.

## Honest scope
- The SPIKING PRODUCTION is real (decoded from `language_output` firing on a real bridge; spoken-accuracy 1.000) — the claim of this rung.
- The 3 answer tokens (yes/no/idk) are mapped to words of the validated 16-word A→W vocab (yes→'fly', no→'swim' — proxy surfaces). The SPIKING production + gate-first moat are the claim; a literal "yes"/"no" A→W retrain is cosmetic polish (a bounded GPU vocab-rebind, per EMERGE-67).
- The reasoner's DECISION accuracy on the coarse emergent clusters is the rung-4 reasoning quality (already characterized — emergent clusters at K=1024 are coarse; some queries mis-decided). The SPEAK wire faithfully speaks WHATEVER the reasoner decides — it does not change the reasoning accuracy; it produces the decision on spikes.
- GPU (cupy A→W). The reasoner is backend-agnostic numpy.

## What this establishes
The breadth→knowledge arc now runs end-to-end with SPEECH: **discover a broad vocab from real experience → discover its categories (probe-free) → teach a fact in plain terms → answer a yes/no question about a held-out real word → SPEAK the answer ON SPIKES → abstain on the unknown (gate-first moat)**, transformer-free, moat intact, NO `sim/` edit. Next: a literal yes/no A→W vocab (a bounded GPU rebind); the full "the dog can move" frame render (needs breadth-word A→W + a property verb — gated on a factual/definitional corpus, per rung-3); the fully-spiking reasoner (rung-2's spiking inheritance co-resident with the A→W).

## Files
`research/runners/_realcorpus_speak_answer_onspikes_derisk.py`; 3-seed `research/findings/raw/_rc_speak_onspikes.json`. Prior: rung-4 conversation `2026-07-08-knowledge-half-rung4-talk-about-real-corpus-vocab-GO.md`; EMERGE-67 `NeuralSpell` (the spiking A→W); rung-2 (spiking inheritance).
