---
type: finding
status: live
date: 2026-08-20
mechanism: spiking-np-boundary-binding
lane: D-pragmatics
seeds: [42]
seed-waiver: A deterministic mechanism de-risk — does a spiking NP-head binder collapse multi-word noun-phrase spans so the extractor reaches free-prose clause shapes it could not before. The evidence is a before/after coverage delta on a fixed labelled clause set with lesion-free/lesioned extraction paths, not a stochastic effect size across a seed population; the single seed is the substrate build seed.
instrument: NPHeadBinder (spiking Hebbian boundary-flag binder on a private Izhikevich SimulationBridge) feeding the unchanged BridgeParser, over the open-text extraction clause set
runner: research/runners/_spiking_np_boundary_extraction_derisk.py
artifacts:
  - research/findings/raw/_spiking_np_boundary_extraction_derisk.json
---
# A spiking NP-boundary binder closes the free-prose extraction gap the canonical-SVO parser left open

Artifact: research/findings/raw/_spiking_np_boundary_extraction_derisk.json

**One line.** The 2026-08-20 honest negative was: the on-brain spiking parser reaches artificially-canonical 3-word
SVO but fails on the multi-word proper-noun subjects, copula predicate nominals, and passives that make up real
free-generated prose. This de-risk builds the named next mechanism — a spiking Hebbian NP-head binder that collapses
an arbitrary-length noun-phrase span into ONE role-fillable unit before the position×voice read-out — and it closes
almost the whole gap: extraction coverage on the combined clause set rises **9/19 → 18/19** (0.474 → 0.947), the <!--derived-->
hard multi-word-NP/copula/passive subset **0/10 → 9/10**, and the moat verifier stays **precision/recall/F1 = 1.0**
on the newly-parsed clauses (0 regressions on the original 9 canonical items).

## The mechanism (genuinely spiking, genuinely Hebbian — not host regex for the role)
`NPHeadBinder` (`research/runners/_spiking_np_boundary_extraction_derisk.py`) generalizes `AttributedBridgeParser`'s
(from-start × from-end × voice) conjunction → Hebbian-role-ensemble architecture by DROPPING the from-start factor
and collapsing its roles to two — HEAD vs MODIFIER — keyed only on the from-END boundary flag (is this word the last
of the span). That makes it span-length-agnostic by construction: 2 conjunction units, Hebbian-trained once
(`enable_hebbian_learning`, firing-rate argmax read-out on a private Izhikevich `SimulationBridge`), cover a span of
ANY length, unlike AttributedBridgeParser's fixed 1–2-adjective window. The collapsed span (its HEAD word) fills the
subject/object slot; the resulting 3-slot frame feeds the UNCHANGED `BridgeParser.parse()` — including its existing
passive 0↔2 flip. `segment_clause()` is the one declared, minimal host lexical rule: it finds where a
determiner-headed span begins/ends (an aux/copula/participle/verb scan) — it only finds span BOUNDARIES, never
assigns a role. The role assignment stays 100% spiking.

## Result (artifact _spiking_np_boundary_extraction_derisk.json; reproduced independently)
| | BEFORE (plain BridgeParser) | AFTER (NP-binding) |
|---|---|---|
| Combined set (19 assertion clauses) coverage | 9/19 = 0.474 | 18/19 = 0.947 | <!--derived-->
| Hard subset (multi-word-NP / copula / passive, 10) | 0/10 = 0.0 | 9/10 = 0.9 |
| Original 11-clause subset (the prior finding) | 9/11 = 0.818 | 10/11 = 0.909 | <!--derived-->
| Verifier on newly-parsed clauses (9) | — | precision=1.0, recall=1.0, F1=1.0 |
| False-claim catch (whole set) | 5/9 = 0.556 | 8/9 = 0.889 | <!--derived-->

## The one honest residual + the next lever
Exactly one clause still fails, unchanged from the original finding: `"The Eiffel Tower was built in London"` — a
passive with NO `by`-agent, so there is nothing to fill the agent slot (a location PP is not an agent). Named next
lever: a locative/temporal-PP handler that either fills the missing agent with an abstained/underspecified marker or
routes agentless passives to a non-SVO relation type, instead of leaving them abstain-and-suppress. Morphological
normalization (orbits/orbit → different store keys) also remains out of scope, as previously documented.

## Sources (biological grounding for a dedicated phrase/NP-boundary representation)
- (Ding, Melloni, Zhang, Tian & Poeppel, 2016) "Cortical tracking of hierarchical linguistic structures in connected
  speech", Nat Neurosci 19:158–164 — cortical activity concurrently entrains to word, PHRASE, and sentence
  timescales, i.e. the brain builds phrase-level (NP/VP) constituents as a distinct neural representation. Grounds a
  dedicated spiking phrase/NP binder as biologically motivated, not a host convenience.
- Local record: 2026-06-19-embedded-clause-parsing-scoping (closed-class FUNCTION WORDS mark constituent boundaries —
  the cue our `segment_clause` determiner/aux scan uses) + 2026-07-08-neural-role-extraction-GO (a fronto-striatal
  reservoir already assigns thematic ROLE 6-seed GO — the read-out our binder feeds).

## Provenance note (trust-but-verify caught a test bug, not a parser bug)
The agent's first run showed one false positive that traced to a test-authoring error, not the mechanism:
`FactStore.store()` keys on `(agent, action)` only (one patient per relation), so two "great barrier reef supports X"
gold facts collided and the first was silently overwritten. Fixed with a distinct verb for the second fact and noted
inline in the runner; the numbers above are the corrected, independently-reproduced run.
