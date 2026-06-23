# 🎉 Grounded-language DE-RISK P2 = GO 6/6 — the brain re-encodes a Claude-authored curriculum + recalls it + abstains (the knowledge-teacher→brain→grounded-recall loop works; P2 + P3's gate de-risked, NO LLM touched) (2026-06-23)

**A ~30-fact structured curriculum (Claude-authored, the OFFLINE "textbook author") ingested through the brain's
validated parser→composer pipeline: structured RECALL 1.000 (taught facts) + the no-confab MOAT 0 false-accepts
(untaught cues abstain) + 2-hop chain + embedded clause, 6/6 seeds (42/43/44/100/101/102). ⇒ the WHOLE
knowledge+grounding half (P2 knowledge-teacher→brain re-encoding + P3's grounding gate) is de-risked with ZERO model
download / ZERO convert / ZERO new GPU — before any LLM is touched.** `research/runners/_grounded_lang_p2_derisk.py`,
numpy-CPU (~8s/seed), NO `sim/` edit. The cheapest-first step of the owner-chosen grounded-language arc (scoping
`2026-06-22-grounded-language-faculty-scoping.md` §4 Rank-1).

## Result (6/6 seeds)
| metric | result |
|---|---|
| structured recall (taught facts) | 9/9 = **1.000** every seed |
| no-confab moat (untaught cues) | **0/5 false-accepts** every seed |
| 2-hop chain (dog→cat→mouse) | PASS (returns 'mouse') |
| embedded clause (dog know [cat chase mouse]) | renders |

Curriculum ingested per seed: 22 SVO facts + 8 attribute facts + 2 embedded-clause facts + 2 chain facts (vocab 57
words). Moat probes: `lion eat` / `whale eat` / `dog drink` / `fly plane` → `None`; `apple is blue` → `unknown`.

## P2 ≠ the deprecated Path-3 (the owner's load-bearing concern, confirmed in practice)
The rich model (Claude) authored the curriculum OFFLINE; the BRAIN holds the knowledge + recalls + abstains at runtime
with ZERO external-LLM calls. The standalone-agent stance is preserved by construction — exactly the scoping's
P2≠Path-3 distinction (the rich model is a "textbook author," not the student's brain), now demonstrated end-to-end.

## Curriculum-format finding (a convention for future curricula)
The composer's `ask_yes_no` reads a bound AFFIRM/NEGATE polarity tag; a fact stored with `polarity=None` binds no tag
→ a seed-fragile AFFIRM/NEGATE coin-flip (yes-no failed on 1 seed before the fix). FIX: store every assertion with
`polarity="AFFIRM"` → yes-no deterministic 6/6, no recall regression. **CONVENTION:** all curriculum facts are
affirmative assertions (an explicit `polarity` field). Attribute facts `[noun, adj]` → `(noun, "is", adj)` SVO
triples; clause facts via `hear_clause_fact(ag, ac, Clause(s,v,o))` (the recursive-clause path).

## ⇒ next (the ranked plan)
P2 + P3's gate GO. P1a (the FREE Gen-F convert-mechanism check, scoping Rank 2) is ALREADY satisfied by the
fully-spiking-C1 work (Gen-F is fully spiking on the bridge, generating byte-identical — CYCLE 478-480), so the next
NEW steps are: **P3** (the grounding/gating loop — gate→constrain→verify — cheaply with a template-stub faculty,
closing the architecture that confines a faculty to grounded content) → **P1b** (convert a real fluent SLM,
Qwen2.5-0.5B, the production faculty; a model download). The brain-half is proven; no LLM touched yet.
