---
type: plan
status: live
date: 2026-06-13
---

# Phase 1 production build plan — the 32-bridge / 2,048-concept learned-graded cortex on the curated Option-B substrate

**Date:** 2026-06-13. **Status:** BUILD (Phase-1 de-risks complete; owner: "proceed autonomously, best judgment"). **Scope:** wire the validated ensemble machinery onto 32 REAL curated graded bridges + validate the full conversational matrix at 2,048 concepts. Mostly reuse; one new corpus generator. NO `sim/` edits.

## What's settled (the de-risk arc) → what the build adds
- **Mechanism:** 32-bridge fan-out GO, multi-seed 42/43/44 (cross-bridge V-tag composition + the no-confab moat hold at 2,048 concepts) — `2026-06-13-phase1-32bridge-fanout-derisk-GO.md`.
- **Architecture:** Route A (per-bridge composers + cross-bridge V-tag identity) — `2026-06-13-phase1-composer-architecture-routeA-GO.md`.
- **Vocabulary:** `g20_vocab_spec_2048.py` (32 semantic clusters of 64 = 2,048 concepts).
- **Corpus source:** Option B (curated within-cluster sub-taxonomy). Option C (learned-from-real-experience) is a logged follow-on — INCONCLUSIVE (`2026-06-13-option-c-...-INCONCLUSIVE.md`; the host-ceiling control fired, TinyStories is syntagmatic not paradigmatic).
- **Capability (validated at 3 bridges):** the conversational matrix + within-bridge generalization-in-conversation + the moat on the learned-graded cortex (`2026-06-12-cortex-conversation-{capability,3bridge-ensemble}-GO.md`).
- **The build adds:** the CURATED corpus (real words + meaningful sub-groups, so generalization is meaningful) + the SCALE (3→32 bridges) + the full conversational matrix at 2,048 concepts. **Key point: the per-bridge G1/G2 gate is label-AGNOSTIC (it recovers whatever sub-cluster structure the corpus encodes), so it cannot validate Option-B's meaningfulness — only the conversational matrix + generalization can. The matrix is the real validator.**

## Reuse map (the build is mostly wiring)
- `cortex_conversation_ensemble_derisk.py`: `EnsembleCortexAgent` (the {shard: CortexCodebook} dict + composer + matrix), `build_ensemble_cortices(all_corpora, seed, args)` (builds graded cortices from per-bridge corpora), `gate_X_conv`/`gate_X_vtag` (cross-bridge composition), `anticheat_C3_moat` (the no-confab moat).
- `phase1_composer_ab_derisk.py`: `PerBridgeCortexAgent` (route-A {shard: composer} dict + word→shard router), `gate_A_routeA_per_bridge` (within-bridge matrix), `run_gate_B_and_controls` (within-bridge generalization + C1/C4 controls).
- `multibridge_graded_derisk.py`: `build_bridge_corpus` (the corpus dict shape), `learn_bridge_graded` (the per-bridge graded learn), `_factor_subclusters` (8×8).
- `learned_graded_embedding_derisk_probe.py`: `build_toy_cooccurrence` (the hub/sub-cluster fact-structure generator).

## The ONE new piece: the curated corpus generator
`build_curated_bridge_corpus(cluster_name, seed)` — returns the `build_bridge_corpus` dict shape, but with the bridge's graded structure = the CURATED semantic sub-groups over the REAL cluster words:
1. Get the cluster's 64 words + their curated sub-group ids (0..7) from `g20_subtaxonomy_2048.cluster_sublabels(cluster_name)` (the sub-taxonomy spec, in authoring).
2. Call `build_toy_cooccurrence(n_sub=8, per_sub=8, seed)` → the validated hub-mediated 8×8 fact structure + S_true + second_order_pairs.
3. RELABEL the synthetic members (`c{sub}_m{i}`) → the real cluster words, grouping by the curated sub-groups (synthetic sub-cluster `sub` ↔ curated sub-group `sub`, its 8 members ↔ the 8 real words of that sub-group), namespaced by the cluster name.
4. ⇒ the bridge's graded code makes `lion≈tiger` (both in the curated `big_cats` sub-group) MEANINGFUL, not arbitrary — the difference from the de-risk's synthetic-arbitrary corpus.

## Build steps (cheap-first → scale)
1. **Curation** — `g20_subtaxonomy_2048.py` (32 clusters × 8 semantic sub-groups of 8). IN FLIGHT (subagent `a6d1187d401154ac4`).
2. **Corpus generator** — `build_curated_bridge_corpus` (above). Small new function; CPU-smoke against the dict-shape consumers.
3. **CHEAP integration validation** — a `production_cortex_build.py` runner at a FEW bridges (e.g. `--n-bridges 4`, real curated): build the cortices + run the conversational matrix (who/what/abstain/negation/clause) + within-bridge generalization + cross-bridge V-tag + the moat. **This is the real Option-B validation** (does the curated pipeline produce MEANINGFUL conversational behavior + generalization at real scale, moat intact?). GO → step 4.
4. **SCALE to 32 bridges** — the full conversational matrix at 2,048 concepts + cross-bridge + moat, multi-seed. The production deliverable: a working 2,048-concept conversational brain analogue on the learned-graded cortex.

## Cost + honest scope
- Per-bridge graded learn ~120 s (2300-2700 neurons); 32 bridges ~1 h/seed + the cross-bridge V-tag + moat eval. Not multi-day — ~hours/seed on the GPU. The integration code (steps 2-3) is GPU-free to write + a cheap-bridge smoke.
- **Honest scope:** Option B's similarity is host-CURATED (the agent's structured experience), with a brain-based learn on top — a principled stepping-stone, NOT learned-from-raw-experience (Option C, the follow-on). The binding mechanism + the moat are validated-spiking; the curation is the residual idealization (documented). The deliverable is the first 2,048-concept conversational brain analogue at this scale, with meaningful within-bridge generalization + the no-confab moat.

## Validation gates (the build's acceptance)
At the cheap step (4 bridges) and the full scale (32): the conversational matrix passes (≥5/6 cells + the moat, every bridge); within-bridge generalization meaningfully above chance with the C1-permuted-similarity + C4-random-shard controls collapsing; cross-bridge V-tag composition GO + the fixed anti-cheat collapsing; the no-confab moat zero-breach. A meaningfulness spot-check: generalization neighbours are semantic (cat→dog) not arbitrary.
