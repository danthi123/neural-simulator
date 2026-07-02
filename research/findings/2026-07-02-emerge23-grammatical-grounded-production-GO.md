# EMERGE-23 / toward-language — GO (6/6 seeds, all anti-cheats airtight): the CAPSTONE. The emergent sequence cortex GENERATES full grammatical, grounded sentences, GENERALIZES them to similar cues, and ABSTAINS for ungrounded cues — the whole toward-language chain (EMERGE-15..22) unified into one producer on the real spiking `SimulationBridge`. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge23_grammatical_grounded_production_derisk.py`; CI guard `tests/test_emerge23_grammatical_grounded_production.py` (4 tests). Reuse-by-import (`_emerge14` on-bridge learner + `_emerge12` priming); NO `sim/` edit; CPU numpy-backend; 6-seed (42/43/44/100/101/102).

## The claim
On ONE real spiking bridge, emergent + unsupervised + no `sim/` edit, the sequence cortex generates a full sentence from a subject cue and does all three jobs at once:
- a **GROUNDED** cue → the correct grounded sentence, POS-grammatical: `dog → "dog chased ball"`, `cat → "cat ate fish"` (grounded-grammatical **1.00**, all seeds);
- a **SIMILAR untrained** cue → a grammatical grounded sentence, generalized via the family block: `wolf → "wolf chased ball"` (canine, like dog), `lion → "lion ate fish"` (feline, like cat) (generalized-grammatical **1.00**, all seeds);
- a **NOVEL** cue (fully-disjoint code, no family) → **ABSTAIN**: `zzz → <abstain>` (novel-abstain **1.00**, confabulation 0, all seeds) — the intrinsic no-confab moat.

## The mechanism — grammar and content are read from DIFFERENT blocks (the key insight)
Each word = a fixed sparse code over a shared micro-column pool with **three blocks**: a **POS-class** block (grammar, shared by all words of a part of speech), a **content** block (the specific word), and a **family** block (shared by similar words). The cortex learns the sentences with the committed `sim/` three-term kernel (`fused_htm_permanence_update`, the Bouhadjar rule) over the pre-allocated cross-column coincidence pool; generation rolls it out autoregressively.

The load-bearing realization: the shared POS-class block carries **grammar** (it is primed for every word of the class, so the frame generalizes) but **cannot pick the specific content word** — after any noun the shared class cells prime BOTH "chased" and "ate", tying. The specific continuation is carried by the **distinguishing** blocks (content + family): "cat" primes "ate" (via cat's own content+family → ate), not "chased" (which cat never followed). So content selection reads the prediction driven by the current word's **content+family cells only** (not the shared class cells). A similar cue (wolf shares dog's family block) inherits dog's continuation → generalizes. A novel cue (zzz, disjoint, no family) drives no distinguishing coincidence → nothing primed → abstain.

## Anti-cheats (all airtight, 6/6)
- **dAP-LESION** (coincidence off → no plateau → nothing primed): grounded production collapses to **0.00** (vs 1.00) — the prediction is genuinely the bridge's dendritic-plateau recurrence, not host bookkeeping.
- **FAMILY-DERANGEMENT** (swap the similar cues' family blocks so wolf shares feline, lion shares canine): generalization collapses to **0.00** (vs 1.00) — isolating the **family block** as the generalization carrier (not coincidence, not chance).
- **UNTRAINED** (epochs 0): nothing primed → abstain everywhere → grounded 0.
- **NOVEL-ABSTAIN = 1.00** (confabulation 0): the moat is intrinsic to the disjoint code, not a host check.
- No teacher; multi-seed unanimous.

## A debugging note (root cause, honest)
The first integration attempts produced grammatical-but-wrong content ("dog chased dog" for everyone; zzz did not abstain). Root cause (systematic-debugging, not a guess): the two-compartment apical compartment, coupled toward the soma rest (`apical_g_couple=2.0`), settles its **resting** potential at ~**-61.74 mV**, which is ABOVE the imported `coincidence_predict`'s read-line `apical_E_rest + 2.0 = -63.0` — so that reader returns EVERY cell as "primed" (the truly-primed plateau reaches ~+20 mV, but rest already clears -63). EMERGE-18 survives this because it *intersects* the primed set with each word's own cells; EMERGE-23's content scoring needs a clean primed set. Fix: a local reader (`Producer._predict_primed`) that thresholds the apical at **-40 mV** — between the ~-62 rest and the ~+20 plateau — isolating the genuinely-primed cells. NO `sim/` edit; the fix is a read-threshold in the runner.

## The toward-language chain — unified
On one spiking brain, emergent + unsupervised + no `sim/` edit: PREDICTION (15) · PRODUCTION (16) · GENERALIZATION (17/19) · HIGH-ORDER-GEN (18) · GROUNDED-MOAT (20) · INTERACTIVE CONSOLE (21) · SYSTEMATIC RECOMBINATION / GRAMMAR (22) · **GRAMMATICAL GROUNDED PRODUCTION (23, this doc)**. The cortex predicts, produces, generalizes, grounds (intrinsic no-confab moat), generates novel grammatical structure, and now **produces full grammatical grounded sentences that generalize and abstain** — the transformer's core language roles, biology-native, on the real spiking substrate.

## Honest scope + next
- The grounded facts use distinct verbs+objects so there is no shared-function-word high-order ambiguity in the object slot; unifying the full three-block content encoding WITH high-order context allocation (shared function words like "the" that carry the verb→object dependency across positions) is a separate, named integration step (the SeqGenLearner's context allocation + the three-block content codes combined). This is the next toward-language integration, not a wall.
- Grammar here is a fixed frame (NOUN VERB NOUN); EMERGE-22 already showed systematic recombination (held-out content in a learned frame). Coupling EMERGE-22's grammar prediction with EMERGE-23's content grounding (two parallel streams: class-block → POS, content/family-block → word) is the clean full-grammar+content producer.
- The genuinely-hard residual (NOT surface form, a separate faculty): **open-world SEMANTICS** — the knowledge-acquisition problem, managed by grounded knowledge + the no-confab moat, which is the artificial-life / experience-driven-learning direction (the master directive's core). That is the next deep-research gate.

## Artifacts
`research/runners/_emerge23_grammatical_grounded_production_derisk.py`, `tests/test_emerge23_grammatical_grounded_production.py`, `research/findings/raw/_emerge23_grammatical_grounded_production.json`. Prior: `2026-07-02-emerge22-pos-frame-grammar-GO.md`, `2026-07-02-emerge20-grounded-moat-GO.md`, `2026-07-02-emerge18-sequence-generalization-GO.md`, `2026-07-02-emerge14-onbridge-nseq-scaling-R1-surpassed-GO.md`.
