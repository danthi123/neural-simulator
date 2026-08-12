# INTEGRATION: the CLAIM-LEVEL entailment moat wired into production — multi-clause fluent prose survives the no-confab moat (WIRED)

**Date:** 2026-08-12
**Status:** GO / WIRED (production-integration). Synchronous numpy-CPU verify (owner was gaming — no GPU/cupy). All
four verify checks pass through the production wiring.

## The wall

Production chat's no-confab moat re-parsed each rendered sentence back into EXACTLY ONE gated SVO triple and required
it to equal the single gated fact (`ChatBrain._verify` → `_extract_svo_from_prose` → `BridgeParser.parse` → `==
gate_svo`). That is a SINGLE-PROPOSITION verifier: it recovers only the FIRST (agent, action, patient) it finds, so any
SECOND clause — a connective, an added property, an injected false SVO — is NEVER checked. Genuinely free-form,
MULTI-CLAUSE fluent prose therefore could not survive the moat: either the extra clause was silently ignored (a confab
LEAK) or the mixed-order words broke the single re-parse (a false reject). The de-risk measured the old single-triple
moat leaking on multi-clause prose; the mechanism to close it (`ClaimEntailmentVerifier`) was already a de-risked GO on
main (0 confab leaks, 6 seeds) but was NOT wired into the production path.

## What changed (additive, guarded, no `sim/` edit)

The RichAnswerComposer multi-fact turn now verifies each rendered sentence with the de-risked
`ClaimEntailmentVerifier` over the SET of facts the turn gathered — imported from
`research/runners/_moat_claim_entailment_derisk.py`, NOT reimplemented. Two pieces:

- **`brain_chat_tui.py` — `ChatBrain._verify_claim_set` + `_build_claim_verifier`** (new). Lazily builds (and caches on
  the gated set) a `ClaimEntailmentVerifier` whose per-clause role assignment is `self.inner.parse` — the SAME on-brain
  spiking role parser the single-triple `_verify` uses. Nouns/verbs/inflection are derived from the gathered set; the
  tight verb-synonym table is reused. It returns `(accepted, result)`, or `(None, None)` when the claim moat is disabled
  (escape flag) or the verifier is unbuildable (a role-permutation collision in the gathered set → the caller falls back
  to the single-triple `_verify` in the SAFE direction).

- **`rich_answer_composer.py` — `render_paragraph` + `_render_one_verified` + `_verify_rendered`** (new helper). The
  multi-fact turn passes the gathered set down; a rendered sentence is accepted IFF every proposition it asserts is
  entailed by that set. This lets multi-clause fluent prose survive IFF every clause is grounded, and rejects a
  rendered unit carrying any ungrounded/contradictory clause. The single-fact turn (gated set of 1) and the escape flag
  both keep the exact single-triple `_verify`.

It is a strict **SUPERSET** of the old check: a single grounded sentence still passes byte-identically. The escape
`BRAIN_CLAIM_MOAT=0` reverts every rendered sentence to the pre-generalization single-triple behavior.

## Verify (synchronous, numpy-CPU, seed 42 — THROUGH the production wiring)

Built a `ChatBrain` over the de-risk gated world (cat/dog/fish) so the de-risk adversarial suite runs through the
production `chat._verify_claim_set` / `composer._verify_rendered`, plus the composer stub smoke default-vs-escape.

- **(a) grounded multi-clause PASSES where old single-triple REJECTS.** `"The cat eats fish and the dog chases the
  cat."` → claim-level `accepted=True` (grounded = both facts); the old single-triple `_verify` on the same prose
  (target `[dog,chase,cat]`) returns `False` (it recovers only the first SVO). ✅

- **(b) injected-false clause REJECTED — 0 confab leaks.** The full de-risk suite (30 cases, 14 with a false assertion)
  through `chat._verify_claim_set` → **0 leaks, 0 mismatches vs the de-risk expected verdicts**. Example: L1 `"The cat
  eats fish and the dog chases the bird."` → `accepted=False`, reason `asserted proposition ('dog','chase','bird') not
  entailed by the gated set`. ✅

- **(c) LESION (load-bearing).** With `BRAIN_CLAIM_MOAT=0`, `_verify_claim_set` returns `None` (verifier not consulted)
  and `_verify_rendered` falls back to the single-triple `_verify`, which DROPS the grounded multi-clause rendering
  `"the dog chases the cat and the cat eats fish."` (kept ON → `True`, escape → `False`). End-to-end `render_paragraph`
  with a multi-clause renderer: ON keeps both gathered facts; OFF drops the multi-clause one. ✅

- **(d) NO REGRESSION.** Single-fact recall (`what does the cat eat` → "The cat eats fish."), abstain on an untaught
  subject, and in-loop learn-then-recall (`bird chase fox` → recall "fox") all correct on the unmodified
  `ChatBrain.answer`/`gate` path. The single grounded sentence still passes the single-triple `_verify`. The composer
  CPU stub smoke transcript is identical with the claim moat ON vs the escape flag (the stub renders single-clause
  sentences, on which the generalization is inert — the pre-existing seed-dependent "how do you learn" gate-side abstain
  is upstream of this change and identical in both). ✅

## Scope / residuals

- The per-clause ROLE assignment is on-substrate (`BridgeParser` on `SimulationBridge`). The clause DECOMPOSITION,
  COVERAGE, and synonym/negation/hedge bookkeeping remain a HOST verification harness — a legitimate harness exactly
  like the existing `_verify`/`_extract_svo_from_prose`, and the same C1 residual the burn-down already tracks. This
  wires the de-risked GENERALIZATION; biologizing the host decomposition is a separate, low-priority research item.
- The gated set for a turn is the set the turn gathered (all brain-sourced), so a rendered clause referencing a fact
  NOT gathered this turn is correctly rejected — the gathered set is authoritative.

## Files

- `research/runners/brain_chat_tui.py` — `ChatBrain._verify_claim_set`, `_build_claim_verifier`, `_claim_moat_enabled`.
- `research/runners/rich_answer_composer.py` — `render_paragraph`, `_render_one_verified`, `_verify_rendered`.
- Escape: `BRAIN_CLAIM_MOAT=0` → exact single-triple per-sentence behavior.
