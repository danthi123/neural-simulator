# gap#5 RANK 3 — imaginative/generative replay (novel recombination): research gate

**2026-07-22.** RANK 1 (single-assembly spontaneous reactivation) = 6-seed GO; RANK 2 (ordered forward sequence replay)
= within-reactivation SOLVED+robust, forward-chain real (deterministic confirm running). RANK 3 is the "imagine episodes"
half of gap#5 — the last replay rung before the SWR-loop console.

## The capability (what "imagination" means here, mechanistically — not free fantasy)
The neuroscience is specific and matches this project's substrate: **SWR replay can traverse sequences the animal NEVER
experienced as a whole — it RECOMBINES stored transitions at shared states** (Ólafsdóttir/Gupta "shortcut/novel-path"
replay; Kay-Frank; the eLife Ecker-2022 CA3 model the project already uses for RANK 1/2). CA3's recurrent excitatory
chain, learned from experience, is the generative model; replay samples paths through it. So imagination = **novel-but-
consistent recombination of learned A→B / B→C transitions across a SHARED branch node**, NOT arbitrary generation.

## Diagnosis → the cheapest de-risk (reuses the RANK 1/2 primitives, both working)
Store TWO overlapping chains that SHARE a middle assembly B: **A→B→C** and **X→B→Y** (assemblies A,C,X,Y disjoint; B
shared). Each transition is a forward BTSP chain link (the RANK 2 mechanism); each assembly a bistable within-attractor
(the RANK 1 mechanism). During REST under weak noise (frozen plasticity, no cue), does the network sometimes generate the
**NOVEL recombination A→B→Y or X→B→C** — a path never stored as a whole — by entering B (from A or X) and exiting to
EITHER of B's two learned successors (C or Y)? That IS generative/imaginative replay: the shared node is a branch point,
and the recombination is the imagined episode.

## Ranked biology-based, spiking, one-brain methods (cheapest first)
1. **Shared-branch-node recombination (THE de-risk):** the above. Cheapest — a direct extension of the RANK 2 driver
   (add a shared assembly + a second chain; detect cross-chain transitions in replay). Biology: CA3 recurrent branch
   sampling (Ecker-2022; Gupta shortcut replay). The gap#5 catalog entry.
2. **Cue-driven imagined completion:** cue a PARTIAL/degraded pattern (e.g., A + weak B) → CA3 completes to a full
   imagined episode A→B→C (RANK 1 completion, now SEQUENTIAL). Complements #1.
3. **Preplay (Dragoi-Tonegawa):** replay of a to-be-experienced sequence from pre-configured assemblies. Highest-variance;
   deferred unless #1/#2 need it.

## Anti-cheats (mandatory, same family as RANK 1/2)
- **NO-RECOMBINATION control:** store A→B→C and X→D→Y (B ≠ D, NO shared node) → the recombination A→B→Y must NOT appear
  (it has no shared node to cross). If it does, the "recombination" is a noise artifact, not learned structure.
- **SCRAMBLE-between:** shuffle the cross-chain edges → recombination must break (load-bearing structure).
- **NO-NOISE acid:** without background noise, no spontaneous recombination (rules out a self-sustaining artifact).
- **NO-ENCODE:** without encoding, no recombination (learned weights load-bearing).
- **Consistency, not fantasy:** the recombined path must exit B to a LEARNED successor (C or Y), NOT to a random
  assembly. Report the fraction of B-exits that go to a learned vs unlearned target.
- **DETERMINISM:** the transition-ORDER metric is GPU-non-deterministic (RANK 2 lesson) → run the order/recombination
  claims on numpy or GPU+`CUBLAS_WORKSPACE_CONFIG`.

## Verdict / plan
Surpassable-and-cheap: RANK 3 recombination is a direct composition of the two validated primitives (bistable
within-attractor + forward BTSP chain) on the shared-node topology — no new `sim/` mechanism, an additive extension of
`_gap5_sequence_replay_derisk.py` (a shared-assembly draw + a second chain + a recombination detector). Build gated on
the RANK 2 deterministic confirm (running) so RANK 3 rests on a solid RANK 2. De-risk cheap-first on numpy (deterministic),
then 6-seed.
