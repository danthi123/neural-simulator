---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-moat-verifier
lane: D-pragmatics
seeds: [42]
seed-waiver: A deterministic COMPOSITION check — do the three fluency routing rules (hedge/synonym/reporting), each GO in isolation, compose without interference on cross-style clauses. The evidence is per-item keep/suppress correctness + a same-input order-ablation on a fixed labelled set; single seed is the substrate build seed.
instrument: research/runners/_unified_fluent_verifier_derisk.py — one verify_clause_unified applying all three routing rules in front of the UNCHANGED extract (NPHeadBinder) + entail (FactStore) pipeline
runner: research/runners/_unified_fluent_verifier_derisk.py
artifacts:
  - research/findings/raw/_unified_fluent_verifier_derisk.json
external: NO-EXTERNAL-NEEDED — composes three in-repo lexical routing rules already individually landed + validated (dd1b76b6 / 620601a6 / 61d9dcbb); no new mechanism, capability wall, or paradigm claim.
---
# The fluency trilogy COMPOSES into one verifier, ready to wire into the live moat

Artifact: research/findings/raw/_unified_fluent_verifier_derisk.json

**One line.** The three fluency-moat fixes (hedge->verify-under-hedge, synonym expansion, reporting-clause strip)
were each GO in isolation; wiring them into the LIVE verifier needs proof they COMPOSE. This unifies them into one
`verify_clause_unified` and tests the union of all three de-risks' item sets PLUS 11 new CROSS-STYLE items (each
combining >=2 rules): they compose cleanly, zero regression, and — verified by an order-ablation — the order does not
matter. `rules_compose_cleanly: true`. Ready to wire live.

## Result (numpy, seed 42; reproduced; all three routing rules + FactStore imported UNCHANGED)
- **Union set (45 items)**: recall-on-faithful 18/19, precision-on-confab 19/19, subjective-bypass 7/7.
- **Cross-style (11 new, each exercises >=2 rules)**: recall-on-faithful 5/5, precision-on-confab 5/5, subjective 1/1.
- **Regression vs each rule in isolation** (re-ran each de-risk's own pipeline on its own set, per-item diff): hedge
  29/29, synonym 26/26, reporting 25/25 — ZERO regressions anywhere.
- **The single miss** is the inherited, already-documented nested-report residual ("Scientists confirmed that
  researchers reported that X", 0/1 faithful, fails safe) — reproduced honestly, not a new failure.

## Composition order (and why it is order-independent)
`verify_clause_unified`: (i) `is_subjective` bypass on the raw clause; (ii) `strip_reporting_frame` (token-anchored on
the token before the first "that"); (iii) `strip_hedge` (unanchored substring) on the remainder; (iv)
`extract_svo_npbind` (spiking NP-binding + BridgeParser role read-out); (v) `classify_claim_synonym_aware` at
entailment. An ablation (`verify_clause_alt_hedge_first`) run on every cross-style item AGREES 11/11 — structurally,
the reporting anchor cares only about the token before "that" while the hedge search is unanchored, so under this
construction family they never compete for the same span. On "I believe scientists confirmed that Mercury orbits the
sun" the reporting-strip removes the whole prefix (hedge included); on "Scientists confirmed that I believe whales
inhale air" reporting fires first, then hedge on the remainder, then synonym grounds "whales inhale air" — all three
genuinely exercised on one clause. A relative-clause guard confirms all three stay silent when they must not fire.

## Verdict / integration
Ready to wire the composed hedge+synonym+reporting verifier into the LIVE brain_chat moat path (which today guards
only single-fact SVO), then widen Qwen to free generation gated by it + lesion-test the faculties stay load-bearing
over open output (#99). The one open edge — nested reports — is a PRE-EXISTING coverage gap of the reporting rule
alone (not a composition failure), with a recursive-strip fix path already measured; track/close it separately before
claiming full reporting coverage, but it does not block the live-wiring. This closes the fluency DE-RISK arc: extract
(reaches free prose) -> entail (moat) -> hedge/synonym/reporting (all closed) -> COMPOSE (clean). What remains is
INTEGRATION (live-wiring + Qwen-widening) + the emergence burn-down of the host-lexical NP-boundary detection.
(Agent-built, independently re-run + reproduced.)
