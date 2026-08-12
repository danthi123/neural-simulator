---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — making the chat recall genuinely SPIKING (onebrain store) breaks the runtime new-word LEARN
lane: integration-first (#0 one-brain — a de-risk, not a clean flip)
integration_faculty: one-brain-substrate
verdict: NO-GO for the naive flip. The default chat recall runs the NUMPY fast path (RFPhasorComposer, enable_substrate_store=False, _scan_first_match uses np., not _resonate) — so CHOOSE+LEARN are substrate-VSA but not resonate-and-fire SPIKING. Flipping composer_kind="onebrain" (the #0 flip, enable_spiking_cleanup+substrate_store) makes the recall run on the spiking substrate (confirmed: much slower, OneBrainComposer) AND CHOOSE still works (dog→chase→cat, abstain correct) AND anaphora works — BUT it BREAKS the runtime new-word LEARN: teaching "otter chase crab" mis-renders ("ok — now dog is doing it") and the recall abstains ("I don't know"). The runtime code allocation (rf_phasor_composer _filler_phases: a new word gets a code in self.concepts + self.words) is NOT registered in the SPIKING SUBSTRATE store, so the spiking store cannot recall it. Genuinely-spiking recall therefore requires a de-risk: make runtime vocabulary growth register the new code in the substrate store (onebrain), not only the numpy codebook. Pre-baked facts recall fine on onebrain; only runtime-allocated ones fail.
artifacts:
  - research/runners/rf_phasor_composer.py
verification: full-chat test with composer_kind="onebrain" — CHOOSE "what does dog chase?"->"dog chase cat", abstain correct, anaphora "it eat"->"cat eat fish"; LEARN "otter chase crab" -> recall "I don't know about that." (runtime-allocated fact not in the spiking store).
---

# Making the chat recall genuinely SPIKING breaks the runtime new-word LEARN — a de-risk, not a clean #0 flip

## The brain-based-only gap this probes

The owner goal is ALL-SPIKING. The default chat recall is measured to run the NUMPY fast path: `RFPhasorComposer` with
`enable_substrate_store=False`, `enable_spiking_cleanup=False`, and `_can_batch_scan()=True` -> `_scan_first_match` uses
`np.` (numpy masking), NOT `_resonate` (the resonate-and-fire spiking step). So CHOOSE + LEARN, while substrate-VSA and
NOT the host keyword router, are not genuinely spiking. The #0 backlog item ("one-brain default") is meant to fix this.

## Result — the flip works for CHOOSE but breaks runtime LEARN

<!--derived-->
Building the tiny-demo chat with `composer_kind="onebrain"` (auto-enables spiking cleanup + substrate store): the
recall now runs on the spiking substrate (`OneBrainComposer`, and it is MUCH slower — the resonate step per query, which
is the evidence it is on the substrate; speed is secondary). CHOOSE recall ("what does dog chase?" -> "dog chase cat"),
CHOOSE abstain ("what does fish fly?" -> "I don't know"), and multi-turn anaphora ("what does it eat?" -> "cat eat fish")
all still work. **But runtime new-word LEARN breaks:** teaching "otter chase crab" mis-renders and the recall abstains
("I don't know"). The runtime code allocation registers the new word in the NUMPY codebook (`self.concepts` + `self.words`)
but NOT in the SPIKING SUBSTRATE store, so the spiking store has no trace to recall. Pre-baked (build-time) facts recall
fine; only runtime-allocated ones fail.

## The honest next de-risk

Genuinely-spiking recall (the brain-based-only requirement) needs runtime VOCABULARY GROWTH to register the new
concept's code in the SUBSTRATE store (the onebrain spiking store), not only the numpy codebook — so a fact taught this
conversation is laid down as a spiking trace and recalled from it. That is a real integration (the spiking store's
add-concept path), not a flag flip. Until it lands, the honest state is: CHOOSE + LEARN are on the endpoint and
substrate-VSA (numpy fast path), GENERATE (rich) uses the spiking elaborate; the resonate-and-fire recall + runtime
growth compose is the next de-risk for "all-spiking".
