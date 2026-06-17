# Multi-hop relational reasoning on the production composer — cheap-first de-risk GO (3 seeds × 3 D, unanimous)

**Date:** 2026-06-17
**Status:** **GO.** The role-structured pointer-chase reasons multi-hop on the deployed `RFPhasorComposer`, beating
every anti-cheat the 2026-05-14 transitive-inference retraction demands. 3 seeds (42/43/44) × 3 dimensions
(128/256/512), unanimous.

## What was de-risked

Per the scoping (`2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md`, Option 1): multi-hop reasoning is
**iterated single-hop retrieval** — each hop's cleaned-up output becomes the next hop's cue. The production
composer already does one validated hop (`query_patient`: match (agent, action), read patient, abstain on no
match). `query_chain(cue, [action, action, …])` iterates it. Crucially, the cleanup **re-discretizes** the
intermediate concept between hops, so retrieval error does **not** integrate multiplicatively across hops.

The retracted "90% transitive inference" (`2026-05-14-CRITICAL-bug-…`) was a leaky **spreading-activation**
artifact (2nd-degree co-occurrence neighbours, no role structure). So this de-risk is built entirely around the
five controls that defeat that trap.

## The corpus (and why the anti-cheat forced its design)

Eight food-chains of separate SVO facts (`dog eat cat`, `cat eat mouse`, …), **plus dense distractors**: each
chain concept also `play`/`see`s two *other-chain* concepts (different relations). The distractors pollute the
relation-blind co-occurrence graph — making a 2-hop co-occurrence neighbour **ambiguous** — while leaving the
functional `eat` relation a clean unique chain. The first (clean-chain) corpus was **correctly rejected** by the
spreading anti-cheat (spreading scored 1.00 at 2-hop, tying the chase — proving nothing); the dense-distractor
redesign is what makes the test discriminating. (The probe self-policed exactly as intended.)

## Result (held-out k-hop queries: cue = chain[0], answer = chain[k]; the k-hop composition is never a stored fact)

| metric (aggregate @ D=128, 3 seeds) | value | meaning |
|---|---|---|
| 2-hop relational chase | **1.00** | every held-out 2-hop query answered correctly |
| 2-hop spreading floor (control) | **0.08** | leaky co-occurrence baseline fails (gap **0.92** ≫ the 0.5 bar) |
| permuted-relation (control) | **0.00** | scrambling which patient binds each (agent, eat) collapses it → it reads *relations*, not concepts |
| between-hop re-cue lesion (control) | **~0.00** | severing the cleaned hand-off collapses it → the chain is load-bearing |
| moat (every config) | **holds** | unstored cue and over-run chain both abstain (`None`) — no confabulation across hops |
| depth at which chase stays ≥ 0.5 | **4** | chase = 1.00 through 4 hops at D=128 |

All 9 configs (3 seeds × D∈{128,256,512}) are identical to within a single lesion coincidence: chase 1.00 at
every depth, spreading 0.00–0.25, permuted 0.00, moat true. Chance = 1/40 = 0.025.

## Reading it honestly

- **The anti-cheats are the result.** A 2-hop accuracy that any control also achieved would be the retraction
  repeated. Here the chase beats spreading by 0.92, permutation and lesion drive it to chance, and the moat holds
  at every hop — so this is genuine **role-structured relational chaining**, not co-occurrence smearing.
- **Stronger than the scoping's conservative prediction.** The scoping expected "2 hops yes, 3+ a mapped SNR
  boundary." At this scale the chase holds 1.00 *through 4 hops at production D=128* — because the cleanup resets
  SNR each hop, depth is nearly free.
- **Honest scope (where the depth wall actually lives).** This is **40 well-separated concepts** on the numpy
  fast path (the spiking-cleanup parity is established multi-seed elsewhere). D-independence (128≡256≡512) shows
  cleanup is not yet stressed. The genuine depth/SNR limit will appear at **320-concept** scale and on
  **grounded/correlated** codes, where cleanup confusion rises — that is the bounded follow-on the scoping flagged
  (and it remains a *deliverable* either way: it maps exactly how many hops the point-neuron substrate sustains).
- **Not general relational reasoning.** This is chaining over *stored* relational facts (Eichenbaum–Cohen
  relational memory, catalog D.02; CA3 attractor cleanup, D.05). Generalizing a relation to never-seen items is a
  factorised relational code (TEM, Option 4) — months-scale, the strategic end-state, explicitly out of scope.

## Next (cheap, if pursued)

Promote `query_chain` from this probe to a production method on `RFPhasorComposer` / `BrainConversationalAgent`
(~10 lines, reuse-by-import of `query_patient`, no `sim/` edit), with the five controls as a regression test.
Multi-turn dialogue (carrying the validated NMDA loop-attractor working-memory state across turns) is a separable
Phase 2.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_multihop_query_chain_derisk --seeds 42 43 44 --dims 128 256 512
```

No `sim/` edit. Reuse-by-import: `RFPhasorComposer.query_patient` (the validated single hop + its abstention moat).
