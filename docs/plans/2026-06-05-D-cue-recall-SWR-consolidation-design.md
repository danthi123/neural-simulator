---
type: plan
status: live
date: 2026-06-05
---

# Arc: lift cue-direction associative recall via SWR sleep-replay consolidation — design — 2026-06-05

> Owner steer (after the cheat-conversion arc): "Pursue an A or D boundary." Picked **D — cue-direction recall** (over
> A's grounding) because it aligns with the owner's stated conversational-path vision (MEMORY.md: build conversation on
> **generative replay**, not static retrieval/RAG ranking — SWR replay IS generative replay), reuses validated
> machinery (hippocampus trisynaptic loop + SWR replay + Phase 1.3 consolidation 3/3), is measurable, and needs no
> external dataset (A's semantic grounding does).

## The boundary
`2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`: **co-STIMULATION** of an engram tag reactivates BOTH
associated concepts at **87.5% multi-seed** (drive the tag spanning apple+big → both fire). But **cue-only recall**
(drive concept A ALONE → expect associate B in the readout) is **27.5% multi-seed, barely above 20% chance**. The
finding's own open question: *"Can assoc-recall be pushed above chance via a different mechanism?"* Cross-pool plastic
pathways (v18/v19) and Tonegawa tagging did NOT lift it. This is the **heteroassociative asymmetry**: clean
cue→associate completion needs the association consolidated into a directed cortical pathway, which one-shot encoding
does not build (capacity ∝ recurrent-synapses / sparseness; Treves-Rolls).

## The hypothesis (biology-grounded)
**SWR sleep-replay consolidation of the associated PAIR builds the directed cortical association that lifts cue→
associate recall.** Biology: McClelland-McNaughton-O'Reilly 1995 (complementary learning systems — the hippocampus
fast-binds, then slow sleep replay consolidates into cortex); Buzsáki 2015 (sharp-wave ripples replay waking
sequences during NREM, driving cortical STDP); the project's Phase 1.3 consolidation (validated: hippo→cortex transfer
via SWR replay, 3/3 multi-seed). The specific mechanism: during simulated NREM, **co-replay** the associated concept
pairs (drive apple's + big's CA3 ensembles together, repeatedly) → STDP at the cross-concept cortical pathways grows a
DIRECTED apple→big association → driving apple alone now propagates to big. This is generative replay building the
heteroassociative link that one-shot encoding can't.

## The de-risk (cheap-first, the A-arc pattern)
1. **Reproduce the 27.5% baseline.** Encode N concept-pair associations (engram tags, the validated 87.5%-co-stim
   recipe); measure cue-only recall (drive A alone → B in the lang_output / pool-firing readout top-k). Confirm ~27.5%
   multi-seed (the boundary, reproduced honestly).
2. **Apply SWR co-replay consolidation.** During a simulated NREM phase, co-replay each associated pair (drive BOTH
   concepts' CA3 ensembles together, `run_concept_replay_phase`-style but PAIRED), with STDP ON at the
   cortical cross-concept pathways. Multiple replay cycles (Buzsáki ripple rate / Phase-1.3 schedule).
3. **Re-measure cue-only recall.** Drive A alone → B. **GATE: cue-recall lifts SIGNIFICANTLY above 27.5%** (target a
   clear margin, e.g. ≥ 50% multi-seed = well above chance), WITHOUT breaking the 87.5% co-stim or the no-confab moat
   (an UNassociated cue must still NOT recall a random concept — the anti-confab control).
4. **Anti-cheat:** a permuted-pair control (consolidate the WRONG pairings) must NOT lift the TRUE cue-recall — the
   lift must be specific to the consolidated associations, not a generic readout shift.

## Success criteria / honest outcomes
- **GO:** SWR co-replay lifts cue-direction recall well above the 27.5% floor, multi-seed, anti-cheat-clean, no-confab
  moat intact. → The heteroassociative boundary is liftable by the owner-aligned generative-replay mechanism.
- **BOUNDARY/NEGATIVE:** co-replay does NOT lift it (or only at the cost of confabulation). → An honest negative — the
  measured limit of one-bridge consolidation for cue-direction recall; documents WHY (e.g. dense-code capacity, the
  Treves-Rolls sparseness requirement) and what a fuller fix would need. Per the top-level goal, this negative IS a
  deliverable.

## Scope + reuse
Multi-week arc. Reuse-by-import: `consolidation_trainer.py` (`run_concept_replay_phase`, `run_swr_replay_phase`),
the engram API (`sim/bridge.py`), the v16 concept-pool bridges (`compose_concept_engram` encoder), the hippocampus
trisynaptic builder (`build_biological_brain_regions(enable_hippocampus_consolidation=True)`). GPU for real runs;
multi-seed validation; honest propagation to both remotes. Protected `sim/` edits only if strictly required (flagged).
First concrete step: the cheap-first de-risk above (reproduce baseline → SWR co-replay → re-measure), one bridge,
multi-seed, the parity/anti-cheat gates load-bearing (scrutinize a lift harder than a null).
