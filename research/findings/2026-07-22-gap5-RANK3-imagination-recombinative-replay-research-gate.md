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

---

## DE-RISK RESULT #1 (2026-07-22, deterministic numpy) — the shared-node topology has a LEAKY-ATTRACTOR boundary; fixes in flight
Built `_gap5_recombination_derisk.py` (+ additive `chain_edges` branch topology in `_gap5_sequence_replay_derisk.py`,
default None = linear, byte-identical) using the proven RANK 1 (bistable within, `--rank1-encode`) + RANK 2 (forward BTSP
chain, `--within-refresh 8`) recipe on the 5-assembly shared-node topology A→B→C + X→B→Y.

**RESULT: events=0, per_asm(A,B,C,X,Y)=[0,0,0,0,0], w_within=21.7 — NONE of the 5 assemblies reactivate** (not just the
branch node B). vs RANK 2's refresh-8 within=143. **Diagnosis: the branch topology's many cross-links (A→B, X→B, B→C,
B→Y, plus the rank1-style refresh's cross-links spread across 5 assemblies) LEAK each within-attractor's recurrent
excitation below the self-sustaining threshold — a "leaky attractor" from over-connectivity.** Note w_within=21.7 is
ABOVE the 15.2 that reactivated at RANK 2 n_mem=2, so it is NOT a pure within-strength deficit — the CROSS-LINK LEAK is
the mechanism (more connected → the noise ignition drains to neighbours instead of igniting one assembly's basin).

This is a genuine, precisely-characterized boundary of the direct-composition approach (per THE LAW: a verdict on the
METHOD, not the capability). Ranked fixes (testing): (1) a MUCH stronger within-attractor to dominate the cross-leak
(within-events 60 + refresh 24); (2) a SPARSER chain (fewer chain iterations → weaker cross-links) + more refresh;
(3 — if 1/2 fail, the next research-gated method) assembly-SELECTIVE inhibition scaling with connectivity, OR a theta/
phase-gated read so only ONE branch is active per cycle (biology: SWR replay is phase-organized; the branch is sampled,
not co-active), OR sparser (partial-overlap) assemblies so B shares only a FEW cells with each chain rather than being a
full 4-edge hub. All numpy-deterministic (the order/recombination metric is GPU-non-deterministic).

## DE-RISK #1 fix attempts — the leaky attractor is a TOPOLOGY boundary, not within-strength
- **fix2 (sparse chain, chain-fwd 6 + refresh 16):** w_within **27.0** (HIGHER than the original 21.7) yet STILL
  **events=0, per_asm=[0,0,0,0,0]**. ⇒ decisively rules OUT within-strength as the cause — more within does not help.
- **fix1 (strong within, within-events 60 + refresh 24):** running (heavy on numpy); expected to confirm the same.
- **⇒ the spontaneous-replay method is EXHAUSTED** for the densely-wired 5-assembly hub (B has 4 edges; the cross-links
  drain every assembly's basin). Per THE LAW, the next METHOD: **cue-driven / triggered replay** — biology's SWR replay
  is TRIGGERED and theta-phase-organized, not free spontaneous chaos in an over-connected net. Cue predecessor A briefly,
  let the chain propagate, measure whether it reaches C (stored A→B→C) or Y (imagined A→B→Y); cue X → Y (stored) or C
  (imagined). This samples ONE branch per cue (no co-active leak) and directly tests the imagination capability. Cheaper
  fallback within this method: partial-overlap B (B shares a FEW cells with each chain, not a full 4-edge hub).

## ⛔ CORRECTION (same day) — fix1 REFUTES "topology boundary, not within-strength"; strong within DOES restore reactivation
I committed the "topology boundary not within-strength" conclusion off **fix2 alone (w_within 27 → events=0)** while
**fix1 was still running** — a premature conclusion (verify-not-assume violation: wait for all arms). **fix1 (within-events
60 + refresh 24, w_within 129.3) restores reactivation of ALL FIVE assemblies: per_asm=[3,3,3,3,3], events=4.** So:
- The shared-node topology RAISES the within threshold for reactivation (the over-connectivity leak IS real) — but it is
  OVERCOME by a strong-enough within (~129, vs RANK 2's linear chain reactivating at ~15-143). fix2's 27 was just too weak.
- **The genuine remaining issue is DIFFERENT: with strong within, the 5 assemblies reactivate INDEPENDENTLY but form NO
  ordered pred→B→succ TRANSITIONS (within=0, cross=0).** The spontaneous replay fires assemblies but does not TRAVERSE the
  chain in order — the same weak-chain-traversal signature as RANK 2's modest forward-order, amplified at the hub.
- ⇒ the right next method is unchanged (cue-driven / triggered replay) but for a SHARPER reason: not to *enable
  reactivation* (strong within already does) but to FORCE ordered chain TRAVERSAL — cue A strongly (RANK 1's 700pA×150
  completion cue) so A ignites → the A→B edge drives B → B→{C,Y} drives a successor; measure which. Use the fix1 strong-
  within params (ev60/refresh24) so the assemblies are reactivatable. Lesson (again): do not commit a conclusion off a
  partial arm set.
