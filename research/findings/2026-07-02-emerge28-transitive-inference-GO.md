# EMERGE-28 / toward-semantics — GO (6/6 seeds): TRANSITIVE relational INFERENCE. From only adjacent premises (A>B, B>C, C>D, D>E) the never-trained non-adjacent relations (B>D, ...) are INFERRED by chaining the overlapping premises into an integrated order on the spiking HTM cortex — the classic hippocampal transitive-inference paradigm, emergent, NO inference engine, NO `sim/` edit. Completes the inference triad (generalization · inheritance · transitivity).

**2026-07-02 (autonomous).** Runner `research/runners/_emerge28_transitive_inference_derisk.py`; CI guard `tests/test_emerge28_transitive_inference.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim
Teach ONLY the adjacent premises A>B, B>C, C>D, D>E. Then, on the emergent spiking HTM cortex, 6/6 seeds:
- **Non-adjacent inference (1.00 on held-out pairs):** the never-trained relations B>D, A>C, A>D, A>E, B>E, C>E are all inferred with the correct order — the full transitive order A>B>C>D>E is recovered from adjacent premises alone.
- **The CRITICAL internal pair B>D (1.00):** B and D each appear as BOTH a greater and a lesser item across the premises (A>**B**, **B**>C ; C>**D**, **D**>E), so B>D is *unsolvable by associative strength* — inferring it requires integrating the premises into an order (Dusek-Eichenbaum 1997; catalog D.02). This is the discriminating test that separates genuine inference from association, and it is 1.00.

## The mechanism
Each item = a disjoint content code; a premise "X > Y" is learned as X→Y via the committed `sim/` three-term kernel over the coincidence pool. The overlapping premises (B is the lesser in A>B and the greater in B>C) chain into a single learned sequence A→B→C→D→E. A transitive judgment `greater(X, Y)` = "is Y reachable downstream of X in the learned chain?" — read by rolling the chain out from X (autoregressive priming, EMERGE-16) and collecting every item reached, using the resting-vs-plateau apical threshold. B reaches C, D, E though B>D and B>E were never trained → the non-adjacent order is inferred by integration.

## Anti-cheats (all airtight, 6/6)
- **dAP-LESION** (coincidence off → no chaining): non-adjacent inference collapses to **0.00**.
- **BROKEN-CHAIN** (drop the middle premise C>D → B and D become uncomparable): B>D collapses to **0.00** — isolating the transitive chaining (integration of overlapping premises) as the cause, not any per-pair signal.
- **HELD-OUT**: the non-adjacent pairs are never trained (only the 4 adjacent premises).
- **Adjacent 1.00** (sanity); 6-seed unanimous.

## Significance — the inference triad is complete on the emergent brain
With EMERGE-17 (generalization across similar concepts), EMERGE-26/27 (Collins-Quillian inheritance, single + multi-level), and EMERGE-28 (transitive relational inference), the emergent spiking cortex now performs the three canonical forms of inference-beyond-told-facts — all emergent from shared/overlapping codes × the HTM next-state predictor, with no explicit inference engine, no `sim/` edit. Transitive inference is a hippocampal signature (Dusek-Eichenbaum; catalog D.02); here it emerges on the same substrate that stores, generalizes, grounds, produces grammar, grows, and inherits.

## Honest scope + next
- The items + premises are host-DESIGNED (disjoint codes hand-assigned); this is inference-OVER-structure, NOT acquisition-OF-structure-from-experience (the deferred R-c residual — the next deep-research gate: the codes/premises must arrive from experience/perception).
- Named next builds: couple the inference read-outs (EMERGE-26/27/28) into the EMERGE-25 conversational console (answer "does a robin fly?", "is B greater than D?"); and the R-c research gate (emergent structure from experience via the PPMI stream cortex + replay).

## Artifacts
`research/runners/_emerge28_transitive_inference_derisk.py`, `tests/test_emerge28_transitive_inference.py`, `research/findings/raw/_emerge28_transitive_inference.json`. Prior: `2026-07-02-emerge27-multilevel-taxonomy-GO.md`, `2026-07-02-emerge26-emergent-inheritance-GO.md`.
