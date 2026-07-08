# The KNOWLEDGE half of breadth, FULLY-SPIKING reason+speak (architecture GO, 3-seed GPU): the WHOLE conversational turn runs ON SPIKES in ONE cupy process — a spiking-inheritance reasoner (EMERGE-42 pooler + committed HTM kernel) classifies a held-out word, the answer is SPOKEN on spikes (A→W from `language_output`), and the unknown is abstained without invoking the speaker (gate-first moat). Spoken-fidelity 1.000, moat 0. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_fully_spiking_reason_speak_derisk.py` (reuse-by-import: rung-2's `RealCorpusPoolerProbe` spiking inheritance + EMERGE-67 `NeuralSpell` spiking A→W). Requires `SIM_BACKEND=cupy`. NO `sim/` edit.
**Verdict:** architecture GO (3-seed GPU) — the biology-purity completion of the SPEAK arc: reason AND speak on spikes, one brain, one process.

## Why this ran (make the reasoning spiking too)
The SPEAK rung reasoned on numpy + spoke on spikes. This makes the REASONING spiking: rung-2's spiking inheritance (the EMERGE-42 competitive pooler + the committed `sim.kernels.fused_htm_permanence_update` coincidence kernel on a real `SimulationBridge`) classifies a held-out word's category ON SPIKES (read from `cp_v_apical`); the yes/no decision follows; the answer is SPOKEN on spikes via the A→W read-out (decoded from `language_output` firing). Both bridges co-reside in one cupy process.

## The result — 3-seed (42/43/44), GPU, TinyStories K=256
- **Co-residence: OK** — both cupy `SimulationBridge`s (the rung-2 pooler-inference reasoner + the A→W speller) build and run in ONE cupy process (the EMERGE-70/71 one-brain pattern), verified.
- **spoken-fidelity 1.000** every seed — the spiking reasoner's decision is faithfully produced as a spiking-spelled word (decoded from `language_output`).
- **moat-renders-on-abstain 0** every seed — an unknown word (no codon) routes to "I don't know" WITHOUT invoking the speaker (gate-first): the no-confab moat holds by construction.
- ⇒ the WHOLE turn — reason (spikes) → decide → speak (spikes) → abstain (gate-first) — runs on spikes in one brain, one process.

## Honest scope (architecture vs. reasoning accuracy — the load-bearing distinction)
This validates the fully-spiking ARCHITECTURE + the wire + the moat, NOT the reasoning accuracy:
- The **spiking reasoner's DECISION accuracy is rung-2's** (~0.46 held-out inheritance, K=1024; the transcript shows several "expect yes" held-out members mis-decided as "no"). That is the characterized on-substrate CODON-ASSIGNMENT read-variance limit (CYCLE 958's diagnosis — the accuracy limiter is the codon match for held-out members, not the read or the wire). The wire faithfully speaks WHATEVER the (imperfect) spiking reasoner decides; spoken-fidelity 1.000 measures that faithfulness, not the decision correctness.
- The answer tokens are A→W-vocab proxies (yes→'fly', no→'swim'); a literal yes/no A→W is cosmetic polish.
- GPU (both bridges cupy).
So: the fully-spiking ONE-BRAIN turn is demonstrated (reason+speak+moat all on spikes, one process); lifting the spiking reasoner's decision accuracy is the codon-side read-variance lever (the diagnosed, still-open mechanism — diverse-subsampling codon readers / more pooler capacity), distinct from this architecture GO.

## What this establishes
The breadth→knowledge arc's turn is realizable FULLY on spikes in one brain: discover a broad vocab from real experience → a spiking reasoner (pooler + committed HTM kernel) classifies a held-out word on spikes → the answer is spoken on spikes (A→W) → the unknown is abstained gate-first, all in one cupy process, transformer-free, moat intact, NO `sim/` edit. The remaining lever is the spiking reasoner's accuracy (codon-side read variance), not the architecture.

## Files
`research/runners/_realcorpus_fully_spiking_reason_speak_derisk.py`; 3-seed `research/findings/raw/_rc_fully_spiking.json`. Prior: the numpy-reason SPEAK rung `2026-07-08-knowledge-half-SPEAK-grounded-answer-on-spikes-GO.md`; rung-2 spiking inheritance; the CYCLE-958 codon-variance diagnosis; EMERGE-70/71 (one-brain co-residence).
