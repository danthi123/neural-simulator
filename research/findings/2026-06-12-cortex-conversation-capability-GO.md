---
type: finding
status: contributing
date: 2026-06-12
---

# Cortex↔conversation CAPABILITY de-risk: GO — the learned-graded cortex delivers GENERALIZATION IN CONVERSATION with the no-confab moat intact (the lifted capability proven at small scale)

**Date:** 2026-06-12. **Runner:** `research/runners/cortex_conversation_capability_derisk.py` (built per `docs/plans/2026-06-12-cortex-conversation-integration-design.md`). **Backend:** `SIM_BACKEND=cupy` (GPU). **Raw:** `research/findings/raw/_cortex_conversation_capability_derisk_full.json` + `.log`. **Scope:** 1 shard × 64 concepts (animals, 424 within-shard facts), 3 seeds (42/43/44), the REAL learned cortex (`--cortex learned`).

> **Verdict: GO (3 seeds).** The learned-graded-similarity cortex, integrated into the conversational pipeline, delivers the capability the composer's idealized exact-inverse vector-symbolic algebra provably could NOT — **generalization IN conversation** (answer a query about a held-out concept via a *similar* known concept) — while the no-confab moat stays intact (zero confabulation on genuine absence). This is the "step 3 true cortex" goal, demonstrated at small scale, multi-seed, with all anti-cheats collapsing. The mechanism (multi-bridge cross-bridge + moat, GO + holds to 8-bridge fan-out) AND the capability are now both de-risked ⇒ the 32-bridge build is justified end-to-end.

## What was tested (the goal, not just the mechanism)
The prior multi-bridge de-risk validated the cortex MECHANISM (graded codes = a similarity metric; cross-bridge fact recall; the moat). This run tests the CAPABILITY: the conversational matrix running ON the learned cortex, generative role-filler binding using cortex-induced codes, and the NEW generalization — all on the integrated `CortexAugmentedAgent` (the learned cortex wired into `BrainConversationalAgent`: generalization reads the GRADED codes; a DG decorrelation feeds the composer's `grounded_codes` seam so FHRR binding + the moat run on clean codes; cortical reinstatement links binding-identity → graded code; a familiarity-gated graded fallback redefines abstention as "no SIMILAR known fact → abstain"). NO `sim/` edits.

## Results (3 seeds, learned cortex)

| Gate | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Learned cortex graded (within/between cos, margin) | 0.878 / 0.354 / +0.524 | (graded=True) | 0.887 / 0.365 / +0.522 |
| **A — conversational matrix** (who/what, abstention, negation, one-attribute, clause) | **6/6 cells, moat holds (0 breaches)** | 6/6, moat holds | 6/6, moat holds |
| **B1 — generalization in conversation** (held-out graded-neighbour, chance 0.25) | **0.988 (4.0×)** | **1.000** | **1.000** |
| **B1-conv** — generalization through the actual `what_does` conversational fallback | 0.984 | 1.000 | 1.000 |
| **B2 — moat on genuine absence** (64-cue floor) | **0 false-accepts, abstains all** | 0, abstains all | 0, abstains all |

**Anti-cheats — ALL collapse (3 seeds):**
- **C1 permuted-similarity** (scramble which concepts are similar) → B1 collapses to chance. **Headline: generalization is meaning-driven, not an artifact.**
- **C2 orthogonal codes** (no graded similarity) → B1 collapses (0.20–0.375 ≈ chance) **while the conversational matrix STILL passes (6/6)** — the generalization is *similarity*-driven; the binding is *similarity-independent* (works on orthogonal codes too).
- **C3 moat alongside host** → agreement 1.000, zero host-abstain/gate-accept breaches, lesion collapses → moat intact.
- **C4 random-shard** (destroy semantic co-location) → B1 collapses. Co-location is load-bearing.

`{'C1_collapses': True, 'C2_collapses_matrix_passes': True, 'C3_moat_intact_lesion_collapses': True, 'C4_collapses': True}` → **COMBINED VERDICT: GO.**

## The two halves of the deepest risk — both resolved
The design (§4) flagged: (a) can generative VSA binding run on the DG-decorrelated LEARNED graded codes (first time on learned, not synthetic), and (b) does the moat survive generalization (generalize vs don't-confabulate pull against each other)?
- **(a) Binding on learned decorrelated codes — works.** The DG decorrelation landed at between-cos ~0.173–0.176 (slightly above the ~0.15 "binding-clean" target I flagged while watching), but the conversational matrix (which REQUIRES binding) passes 6/6 on every seed, and the C2 orthogonal control collapses generalization *while the matrix still passes* — so the binding is robust on the learned codes despite the imperfect decorrelation.
- **(b) The moat survives generalization.** B2 (zero false-accepts on 64 genuine-absence cues) + C3 (zero host-abstain/gate-accept breaches, lesion collapses) hold on every seed *simultaneously with* B1 generalization 0.99–1.00. The familiarity-gated fallback (fire only when a SIMILAR known fact exists) makes "generalize" and "don't confabulate" coexist.

## Honest scope
- **Small scale: 1 shard × 64 concepts** (the cheap-first de-risk's purpose — prove the CAPABILITY before the expensive scale build). The 32-bridge × 64 = 2,048-concept build is the scale + cross-bridge-at-fan-out integration.
- **The parser is bypassed** (the `BridgeParser` is GPU-validated but degenerate on numpy): the de-risk exercises the load-bearing COMPOSER path (cortex → binder → generalization → moat), which is the capability question; the parser (comprehension) is upstream and separately validated by the existing matrix test suite. `--use-parser` wires the full `hear()` loop on GPU.
- The fallback's relational generalization is the demonstrated form; richer compositional generalization (multi-hop, attribute transfer) is build-time.

## Conclusion + next
**The lifted capability is proven:** the learned-graded cortex gives the conversational agent semantic generalization ("answer about a dog from what was learned about a cat") that the idealized composer algebra could not — multi-seed, anti-cheated, moat intact. Combined with the mechanism de-risk (multi-bridge cross-bridge + moat, GO, holds to 8-bridge fan-out), the dual/CLS learned-graded cortex is now de-risked END-TO-END (mechanism + capability). **The 32-bridge build (= 2,048 concepts, full integration + the conversational matrix at multi-bridge fan-out, ~2–4 weeks) is justified end-to-end and is the OWNER'S explicit-go gate** (build plan piece iii). No banking — the GO is reported with all anti-cheats collapsing and the deepest risk resolved; no `sim/` edits.
