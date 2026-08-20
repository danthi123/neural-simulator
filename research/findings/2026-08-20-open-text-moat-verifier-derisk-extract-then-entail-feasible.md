---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-moat-verifier
lane: D-pragmatics
seeds: [42]
instrument: claim-extraction + entailment prototype over 10 synthetic mixed-claim paragraphs (24 gold clauses)
runner: research/runners/_open_text_moat_verifier_derisk.py
artifacts:
  - research/runners/_open_text_moat_verifier_derisk.py
  - research/findings/raw/_open_text_moat_verifier_derisk.json
---
# Open-text moat verifier de-risk: catching confabulation in FREE Qwen prose is feasible as extract-then-entail

Artifact: research/findings/raw/_open_text_moat_verifier_derisk.json

**One line (the hard part of fluent/open Qwen, #99).** Letting Qwen generate freely without reintroducing
confabulation needs the moat to check ARBITRARY free-text claims, not just one caller-selected triple. A prototype
shows this is feasible as a **two-layer extract-then-entail** architecture: split prose into clauses -> extract SVO
+ detect hedged opinion -> entail-check each claim against the store. On 10 synthetic mixed paragraphs (24 clauses:
grounded, unknown-ungrounded, contradicted-ungrounded, opinion) the confabulation-catch precision=1.0, recall=1.0,
with 0 opinion or grounded clauses wrongly flagged.

## What it proves — and what it does NOT
The ENTAILMENT layer transfers cleanly: it is a direct reuse of the production moat's query_patient / ask_yes_no
(abstain-on-unknown, same-SVO-opposite-polarity rejection), already 6-seed GO. So the honesty guarantee over open
text reuses the validated spiking check. HONEST BOUNDS: (a) the set is 10 hand-authored synthetic paragraphs, NOT
held-out or real Qwen output (no coreference, passive voice, multi-word entities, nested clauses); (b) the EXTRACTION
layer here is a host regex/lexicon stand-in — the genuinely hard, unproven part; (c) unparsed clauses are currently
treated as non-claims (unsafe — an unparsed clause could hide a confabulation).

## Concrete path to wire (fluent/open Qwen, #99)
(1) Replace the heuristic extractor with the SAME spiking SVO parser (BridgeParser, #43) run over each clause of
Qwen's output — keeps extraction brain-based, not host-regex. (2) Reuse routed_composer.ask_yes_no/query_patient
UNCHANGED as entailment (production, GO). (3) Treat unparsed/low-confidence extractions as ABSTAIN-and-suppress that
clause (never emit-uncheckable-as-fact — the moat's conservative bias). (4) Validate catch-rate on REAL Qwen prose
before production. Honesty = STATE-fidelity: hedged opinion is legitimate (marked); only ungrounded-stated-as-fact is
the failure the verifier catches.
