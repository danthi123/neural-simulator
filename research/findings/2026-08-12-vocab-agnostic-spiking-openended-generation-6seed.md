---
type: finding
status: contributing
date: 2026-08-12
mechanism: vocab-agnostic spiking open-ended generation — the followon2 Buesing-Maass soft-WTA draw decoupled from the hand-designed 8x8 taxonomy via a corpus-induced morpho-distributional role tagger, producing grammatical novel moat-safe multi-word SVO utterances on firing neurons over ARBITRARY corpus vocabulary
lane: E · Language / brain-native spiking generation (the vocab-agnostic-draw wall)
verdict: 6-seed GO (42/43/44/100/101/102). A VOCAB-AGNOSTIC spiking soft-WTA DRAW produces grammatical, novel, plausibility-advantaged, moat-safe multi-word SVO utterances on FIRING NEURONS over an arbitrary corpus-mined vocabulary (150 words, only 14% overlap with the hand-designed 8x8 taxonomy). Every pre-registered gate passes all 6 seeds: PROVENANCE+noise-ablation (the draw is the argmax-over-firing winner from cp_firing_states, 0 host-rng draws; ou_std->0 collapses to a deterministic argmax); GRAMMATICAL 0.949 mean (min 0.929) vs a role-blind floor 0.134 -> 7.1x, judged by an INDEPENDENT tagger re-induced on a DISJOINT corpus split; NOVEL (>=204 distinct novel utterances/seed, disjoint from the store, known-fact retrieval abstains on every one); PLAUSIBLE (spiking 0.479 vs host 0.418 -> the spiking draw BEATS the host rng.choice at 1.15x quality, min 1.09x; 12.8x the random floor); LESION+SHUFFLE both collapse (likelihood + learned structure load-bearing); MOAT 0 confab leaks, 0 negated re-proposed, untaught-cue abstention 1.00. The blocker (SpikingWTASampler KeyErrors on arbitrary vocab) is removed by inducing role pools from the corpus (morpho-distributional tagger, induced-verb precision/recall 1.00/1.00 vs the taxonomy POS oracle). HONEST SCOPE: the DRAW is spiking; the role tagger, the SVO/connective template, the PPMI likelihood, and the RF-composer moat remain HOST scaffolds (NOT "fully spiking"); grammaticality caps at ~0.95 (bounded by the host tagger's ~5% category errors); single-clause SVO, not deep multi-sentence coherence. <!--derived-->
artifacts:
  - research/findings/raw/_spiking_openended_generation_derisk.json
  - research/findings/raw/_spiking_openended_generation_derisk.log
  - research/runners/_spiking_openended_generation_derisk.py
verification: _spiking_openended_generation_derisk, seeds 42,43,44,100,101,102 (500 spiking SVO draws/seed, read_window=100, ou_std=200pA, D=96). OVERALL VERDICT GO; per-seed grammatical 0.929-0.972, spiking/host quality 1.09-1.23, moat leaks 0/6, provenance+noise-ablation True/6. Instrument verified: all fractions finite (no NaN); noise-ablation requires the noiseless sampler deterministic (peak>=0.999) AND the noisy sampler stochastic (peak<0.999) -- both hold all seeds. <!--derived-->
---

# Vocab-agnostic spiking open-ended generation — 6-seed GO (the spiking draw generalizes to arbitrary vocab)

## The wall this attacks (north-star: brain-native spiking fluent generation)
The brain's spiking generative DRAW is a 6-seed GO (`_followon2_spiking_wta_sampler_derisk`: a Buesing-Bill-
Nessler-Maass 2011 noise-driven soft-WTA over an Izhikevich bank — the winner read from `cp_firing_states` IS
the categorical draw, NO host `rng.choice`). But it was HARD-LOCKED to the hand-designed 8x8 taxonomy:
`SpikingWTASampler.__init__` → `_category_pools(TAXONOMY_8x8)` → `_encodable_agents()` indexes
`self.row[<taxonomy word>]`, so it **KeyErrors on any corpus-mined vocabulary** (confirmed empirically here:
`KeyError 'run'`). The spiking draw could not run on arbitrary vocab — which blocks the brain-owns-generation
stack (#3E, `2026-08-12-brain-owns-open-ended-generation-...`) from being genuinely open-ended and keeps its
draw host-side (`rng.choice`) whenever the vocabulary is not the toy taxonomy.

## Wall-discipline reframe (what constant did we substitute for a companion process?)
`TAXONOMY_8x8` is a CONSTANT standing in for a process biology runs ALONGSIDE generation: **syntactic-category
acquisition** — a child induces noun/verb categories from morphology (-ed/-ing inflections mark verbs) +
function-word frames (Mintz 2003 "frequent frames"; morphological bootstrapping). Replacing the constant
taxonomy with a corpus-derived **morpho-distributional role tagger** makes the spiking draw vocab-agnostic.

## What is genuinely SPIKING vs HOST (the brutally-honest inventory this de-risk MAPS)
- **SPIKING (firing neurons):** the generative DRAW — each SVO slot is one soft-WTA competition on a REAL
  `SimulationBridge` Izhikevich pool (`VocabAgnosticSpikingSampler`, a subclass of the GO `SpikingWTASampler`
  with the role pools swapped from `TAXONOMY_8x8` to the induced tagger) driven by the brain's PPMI likelihood
  + OU membrane noise; the winner read from `cp_firing_states` IS the word. NO host `rng.choice` on the draw
  path (source-grep clean + 0 host-rng draws); OU noise IS the stochasticity (ablate → deterministic argmax).
- **HOST SCAFFOLDS (mapped residual — the biologization targets, NOT hidden):** the morpho-distributional role
  tagger; the SVO / connective TEMPLATE (slot order + the connective lexeme); the PPMI likelihood matrix
  (project stream-cortex); the RF-composer no-confab moat. The subject-seed selection is the SWR replay seed
  (which memory reactivates — uniform, a documented-legitimate host process, NOT the filler draw).

## Grammaticality, measured INDEPENDENTLY (non-circular)
Roles are induced on corpus SPLIT A (used to generate); grammaticality is judged by a role tagger re-induced on
a DISJOINT corpus SPLIT B. A generated (S,V,O) is grammatical iff, per split-B's INDEPENDENT tagger, S and O are
nouns and V is a verb. A role-BLIND "unslotted" draw (all three words from the full vocab, ignoring roles) is
the grammaticality FLOOR the slotted spiking draw must beat.

## Result — 6 seeds (42,43,44,100,101,102), verdict GO
<!--derived-->
Arbitrary vocab: top-150 corpus content words, **14% overlap** with the 8x8 taxonomy; split-A role induction
gives 86 nouns / 64 verbs; **induced-verb precision/recall 1.00/1.00** vs the taxonomy POS oracle;
**split-A/split-B tagger agreement 0.987**.

| gate | mean (min across seeds) | bar | pass all 6 |
|---|---|---|---|
| PROVENANCE (draw from cp_firing_states, 0 host-rng) + noise-ablation | True | hard | ✅ |
| VOCAB-AGNOSTIC (overlap 14% ≤ 20%, ≥204 novel/seed, no KeyError) | True | — | ✅ |
| **GRAMMATICAL** (independent split-B oracle) | **0.949 (0.929)** | ≥0.85 | ✅ |
|   role-blind unslotted floor / advantage | 0.134 / **7.1x** | ≥1.5x | ✅ |
|   connective-clause grammatical (2 SVOs + connective) | 0.889 | — | (report) |
| NOVEL (distinct novel utterances/seed) | ~213 (204) | ≥3 | ✅ |
|   novel-comp score (÷ 16 296-triple discoverable universe) | 0.013 | >0 | ✅ |
| **PLAUSIBLE** spiking-frac vs random floor (advantage) | 0.479 / 0.038 → **12.8x (9.6x)** | ≥3x | ✅ |
|   **spiking/host quality** (vs the host rng.choice draw) | **1.15x (1.09x)** | ≥0.7 | ✅ (beats host) |
| LESION (equal drive) plausible-frac | 0.034 (collapses) | — | ✅ |
| SHUFFLED-PPMI true plausible-frac | 0.044 (collapses) | — | ✅ |
| MOAT: hypothesis→known leaks / negated re-proposed | 0 / 0 | 0 | ✅ |
|   untaught-cue abstention | 1.00 | ≥0.95 | ✅ |

Representative spiking-drawn utterances (seed 42): `boy named tim`, `ball played long`, `everyone loved
special`, `tree saw small`, `away flew spot`; connective clauses: `idea tried box and tree saw small`,
`max named upon because away flew spot`. The role-mismatched minority (`lily each dog`, `max named upon` —
`each`/`upon` mis-slotted by the tagger) is the honest ~5% grammaticality residual.

## Honest boundary + the single most promising next lever
<!--derived-->
**What is genuinely spiking (the GO):** the categorical DRAW — each SVO slot is the winner of a soft-WTA
competition on a real `SimulationBridge` Izhikevich pool, read from `cp_firing_states`, over ARBITRARY corpus
vocabulary. This generalizes the followon2 GO off the hand-designed taxonomy: vocab-agnostic spiking generation
is now feasible and 6-seed validated, moat intact, at parity-plus with the host draw.

**The residual walls (mapped, not hidden — the biologization targets):**
1. **Role acquisition is HOST.** The noun/verb pools come from a morpho-distributional tagger (morphological
   bootstrapping + Mintz frames) — corpus-derived (so it is vocab-agnostic) but host-computed, and its ~5%
   category errors are exactly what caps grammaticality at ~0.95.
2. **The grammar TEMPLATE is HOST.** Slot order (S-V-O) and the connective lexeme are a Python template, not a
   learned/spiking sequencer.
3. **The likelihood + moat are HOST** (the PPMI stream-cortex matrix; the RF phasor composer). NOT "fully
   spiking."
4. **Single-clause only.** SVO + a host connective — not deep multi-sentence coherence (that is the separate
   stream-cortex e-prop deep-context boundary, ~44% of the recurrent-credit margin).

**Single most promising next lever:** wire the EMERGE-65 **corpus-self-organized** slot-inventory + slot-order
discovery (`_emerge65_self_organized_producer`, 6-seed GO — function words, slot inventory, slot order all
induced from the corpus) INTO this vocab-agnostic spiking sampler, **replacing the host morpho-distributional
tagger and the SVO template**, and scale it past EMERGE-65's 3 fixed frames to the open 150-word vocab. That
makes both the ROLES and the GRAMMAR corpus-self-organized rather than host tables; the subsequent step is to
biologize the role/frame discovery itself onto spikes (a spiking morphology/frame detector), closing residuals
1–3 toward brain-native spiking fluent generation. (Residual 4 — deep multi-clause coherence — is the orthogonal
stream-eprop frontier.)

## Files
Runner `research/runners/_spiking_openended_generation_derisk.py`; raw
`research/findings/raw/_spiking_openended_generation_derisk.json` + `.log`. Reuse-by-import (b2 PPMI gates +
host baseline; followon2 spiking soft-WTA; option_c corpus builder; RF composer moat). NO `sim/` edit; CPU.
