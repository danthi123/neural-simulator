---
type: finding
status: go
date: 2026-08-12
mechanism: moat generalization — a CLAIM-LEVEL entailment/abstain gate that lets multi-clause fluent prose through IFF every asserted proposition is grounded, replacing the single-triple ChatBrain._verify
lane: conversational-moat (fluency vs the no-confab honesty guarantee)
verdict: GO (de-risk, runner-level). The single-proposition moat generalizes to a proposition-SET entailment gate with ZERO confab leaks. Decompose the rendered prose into clauses, role-parse EACH clause on the SAME on-brain BridgeParser (position x voice -> agent/action/patient — the passive flip is where the substrate does non-trivial work), then require every ASSERTED (affirmative, un-hedged) proposition to be entailed by the gated fact set (exact or a tight verb-synonym that maps to a gated fact); a HEDGED proposition is allowed only as an explicitly FLAGGED guess; a negation of a gated fact is a contradiction; the leak-proof invariant is COVERAGE — every known content token must be consumed by an accepted proposition and there must be ZERO unknown content words. Measured over 6 seeds x 14 adversarial leak cases (+ a 15-case wider net + 4 old-moat-leaks-here controls): confab_leak_rate=0.0000, false_reject_rate=0.0000 on the supported grounded constructions, entailment-gate precision=1.0 recall=1.0. NOT yet wired into production `ChatBrain._verify` — this is a runner-level de-risk; the exact wire-in rule is recorded below. Honest boundary: unknown-to-the-brain content, object-relative fronted-object clauses, subject-conjunction expansion, and any negation all REJECT in the SAFE (abstain) direction — mapped, not leaked.
artifacts:
  - research/runners/_moat_claim_entailment_derisk.py
  - research/findings/raw/_moat_claim_entailment_derisk.json
verification: 6-seed GO-gate in the runner (seeds 42,43,44,100,101,102) — leaks=0/14, core=10/10, hyp=3/3, P=R=1.0 EVERY seed; plus a verify-go refutation: the OLD single-triple moat (first-SVO re-parse == gate_svo) LEAKS 4/4 on the same multi-clause false-carrying prose (proving the leak-detector is not silently passing), and a 15-case wider adversarial net (synonym abuse, deep so/because nesting, casing, passive grounded/false, hedge+unhedged-false, dangling predicate, object-relative) leaks 0 / false-rejects 0.
---

# Moat generalization: a claim-level entailment gate lets multi-clause fluent prose pass IFF grounded — GO (de-risk)

## The wall this closes (at runner level)

Production chat's no-confab moat is `ChatBrain._verify` (`research/runners/brain_chat_tui.py:612`). It re-parses
the rendered prose back into EXACTLY ONE gated SVO triple
(`_grounded_lang_integration_derisk._extract_svo_from_prose` recovers the FIRST agent/action/patient in surface
order → `BridgeParser.parse` → `== gate_svo`) and accepts only if that single triple equals the single gated
fact. It is a SINGLE-PROPOSITION verifier: any SECOND clause — a connective, an added property, an injected
false SVO — is never checked. So no free-form, multi-clause, or connective prose can survive the moat: either
the extra clause is silently dropped (a CONFAB LEAK — the honesty guarantee breaks) or the mixed-order words
break the single re-parse (a false reject). Fluency and the honesty guarantee collide at this one function.

The honesty boundary is a DELIVERABLE, not a caveat, so the fix must generalize the check WITHOUT weakening it.

## The mechanism (what the de-risk built)

`research/runners/_moat_claim_entailment_derisk.py` — `ClaimEntailmentVerifier.verify(prose)`:

1. **Decompose** the prose into clauses: split on sentence/segment punctuation (`. ! ? ; : ,`) and on the
   coordinator/subordinator/relativizer words (`and but or so then because which who that while since if …`).
2. **Classify** each clause's tokens: drop function words; detect NEG markers (`not/never/no/n't…`) and HEDGE
   markers (`perhaps/maybe/might…`); a verb is normalized through an inflection map (eats/ate/eaten/eating) or a
   TIGHT verb-synonym table (consume/devour==eat, pursue==chase); a noun is any known content noun; ANY token
   that is none of these is UNKNOWN.
3. **Build the candidate proposition(s)** per clause: a noun before the verb is the explicit subject, nouns
   after are objects; a coordinated object NP (`… seed and worm`) attaches to the previous predicate; a
   subjectless coordinated/relative clause (`… which chases the cat, eats meat`) carries the antecedent subject
   (a stranded leading NP first, else the previous clause's substrate-assigned agent). Passive (`copula + by`)
   feeds the surface triple with `voice="passive"`.
4. **Substrate role parse**: EACH candidate `[w0, w1, w2]` is role-assigned by the on-brain `BridgeParser.parse`
   (the SAME spiking role parser the production moat already uses). This is what catches a role swap — active
   `fish eats the cat` → `(fish,eat,cat)`; passive `the dog is chased by the cat` → the substrate FLIPS to
   `(cat,chase,dog)`.
5. **Adjudicate** each proposition: negation → reject (affirmative-fact scope); hedge → allow but return in a
   HYPOTHESES list surfaced as a guess (never as fact); otherwise require `(a,v,p)` IN the gated set (exact, or
   via the tight synonym) else reject.
6. **Coverage (the leak-proof invariant)**: every known content token must be consumed by an accepted
   proposition, and there must be ZERO unknown content tokens. A dangling predicate/reference, or a leading
   subject NP never consumed, rejects. ACCEPT iff no reject fired — an accepted response asserts ONLY grounded
   facts (+ explicitly flagged guesses).

A smuggled claim always trips one of these: it introduces an unknown content word (unrepresentable), a known
content word that forms an un-entailed proposition, or a role-swap the substrate parse exposes.

## Spiking vs host (the honest boundary)

- **Spiking (on-substrate):** the per-clause role assignment is the on-brain `BridgeParser` — 6 conjunction
  units → 3 role ensembles, Hebbian-trained, spiking Izhikevich on `SimulationBridge`. For ACTIVE voice the
  learned map is position-identity, so a host position-map would agree; the substrate does NON-TRIVIAL work on
  PASSIVE voice (it flips agent↔patient), and that is the SAME parser the production `_verify` already trusts.
- **Host (a legitimate verification harness, exactly like the existing `_verify`/`_extract_svo_from_prose`):**
  the decomposition, coverage, negation/hedge/synonym bookkeeping, and gated-set membership. This is the moat's
  own logic, not the brain's cognition — the brain SUPPLIES + role-parses the content; the harness VERIFIES it.
  Honest framing: the entailment/decomposition is host, as the task requires; only the role parse is spiking.

## Numbers (6 seeds; runner GO-gate)

Full per-seed detail (every case, trace, and verdict) is in the artifact
`research/findings/raw/_moat_claim_entailment_derisk.json` (provenance sidecar
`research/findings/raw/_moat_claim_entailment_derisk.json.prov.json`).

| metric | value |
| --- | --- |
| confab leak rate | 0.0000 (0 leaks / 14 leak-cases × 6 seeds) |
| false-reject rate (supported grounded prose) | 0.0000 (0 / (10 core + 3 hyp) × 6 seeds) |
| grounded-core pass | 10/10 every seed |
| hypothesis handling | 3/3 every seed (fact→reject, perhaps→allow+flag) |
| entailment-gate precision / recall | 1.0 / 1.0 every seed |

Adversarial leak set caught (0 accepted): injected false clause `(dog,chase,bird)`; unknown-verb `flies`;
active role-swap `fish eats the cat`; passive role-swap `the dog is chased by the cat`; added conjoined object
`… fish and meat`; negation of a fact; plausible-but-untaught `the dog eats fish` (as fact); a hedge that does
NOT license a second un-hedged ungrounded clause; property injection `… which is delicious`; antecedent-carry
abuse `the bird eats seed and chases the cat`; passive reverse `the cat is eaten by fish`; false middle clause;
false "synonym" `likes` (not in the tight table → unknown → reject); one-false-among-grounded.

Grounded-core that PASSES: single clause, 2/3-clause coordination, coordinated VP with antecedent carry,
relative clause, passive (substrate flip), verb synonym, progressive+past inflection, coordinated object, long
4-clause, grounded + one flagged hypothesis.

## verify-go refutation (why I believe it)

1. **The leak-detector genuinely sees leaks.** The OLD single-triple moat (recover the first SVO, require it ==
   a gated fact) ACCEPTS 4/4 of the multi-clause false-carrying cases (`the cat eats fish and the dog chases the
   bird`, `… and meat`, `the cat eats fish, the dog eats fish, and the bird eats seed`, `… and the cat chases
   the dog`) — it leaks exactly because it never checks the 2nd clause. The new gate catches all 4. So a 0 here
   is a real 0, not a silent pass.
2. **A wider net (15 fresh cases) holds:** synonym abuse (`pursues`→chase, `devours`→eat) rejected when the
   mapped triple isn't gated; deep `so`/`because` nesting of true clauses accepted; ALL-CAPS accepted; passive
   grounded accepted / passive false rejected; hedge+grounded accepted / hedge+unhedged-false rejected; dangling
   predicate rejected; object-relative rejected (safe). 0 leaks, 0 false-rejects.

## Mapped boundary (rejects SAFE — abstain, never leak; future scope)

- **Unknown-to-the-brain content** (a word outside the taught vocab, e.g. an adjective/property) → reject. In
  production the vocab is V=320+, but the same rule holds: content the brain cannot represent cannot be grounded,
  so it abstains. This is the honest edge — the gate passes only prose whose content the brain can ground.
- **Object-relative fronted-object clauses** (`the cat eats the fish which the dog chases`) → reject (the fronted
  object leaves the relative predicate object-less). Subject-conjunction expansion (`the cat and the dog eat
  fish`) → reject. Negation of any kind → reject (affirmative-fact scope). All in the SAFE direction; none leak.

## The exact rule to wire into `ChatBrain._verify` (main branch)

Replace the single-triple `_verify` with a proposition-SET gate: (1) decompose the rendered prose into clauses
(sentence/segment punctuation + coordinator/subordinator/relativizer words); (2) per clause classify tokens
(drop function words; detect NEG + HEDGE; any non-func/verb/noun token is UNKNOWN) and reject the WHOLE response
on any UNKNOWN content token; (3) build the candidate `(subj,verb,obj)` per clause — conjunction-reduce
coordinated objects, carry the previous clause's substrate-assigned AGENT into a subjectless coordinated/relative
clause — and role-parse EACH via the existing on-brain `BridgeParser.parse([w0,w1,w2], voice)` (voice=passive iff
copula+by); (4) adjudicate: negation → reject; hedge → allow but surface as `perhaps …` in a hypotheses list;
else require `(a,v,p)` in the gated set (exact, or a tight verb-synonym that maps to a gated triple) else reject;
(5) coverage: every known content token must be consumed by an accepted proposition (dangling → reject). ACCEPT
iff no reject fired. This is a strict SUPERSET of the current `_verify` — a 1-clause grounded sentence still
passes — that additionally lets grounded MULTI-clause prose through while rejecting any response carrying even
one ungrounded asserted clause. The gated set for `_verify` is the multi-fact set the `ChatBrain` supplied for
the turn (single-fact recall, a multi-hop chain, or the RichAnswerComposer's grounded sentence set), not one
triple.

## What this is NOT

NOT wired / integrated / production-default (`docs/TERMS.md`): the production `ChatBrain._verify` still runs the
single-triple check; this de-risk lives in a runner. The next step is to port the `ClaimEntailmentVerifier` into
`brain_chat_tui.py` behind the `RichAnswerComposer` multi-sentence path and re-run its per-sentence verify as one
claim-level pass, then lesion-test (disable → a multi-clause grounded answer that previously abstained now
emits; an injected false clause still rejects). Committed to branch `worktree-agent-a06b444310bb23ba2`; nothing
pushed.
