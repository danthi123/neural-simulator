# Tier-2 reasoning wired into the FIRST-CHAT CONSOLE — chain-of-thought + analogy (additive, moat-safe)

**Date:** 2026-06-27
**Scope:** Wire the two GO'd Tier-2 reasoning capabilities into `research/runners/first_chat_console.py` so the
owner can USE them in chat: **2.2 self-cued chain-of-thought** (commit `85814fe2`,
`2026-06-27-tier2.2-chain-of-thought-GO.md`) and **2.1-A analogy over factored relations** (commit `f122200d`,
`2026-06-27-tier2.1A-factored-relation-analogy-GO.md`). Reuse-by-import; **NO `sim/` edit**; the default chat is
byte-equivalent (the DEFAULT stub-rubric stays **10/10, 0 leaks, MIXED, PASS**).

## TL;DR

- **Chain-of-thought route** — HIGH value, fully working on the real brain. "starting from X, what follows?" /
  "what comes after X?" / "where does thinking about X lead?" → the brain SELECTS each hop by its own learned
  association and chases it via the validated single hop, rendering the self-generated chain or abstaining at a
  dead-end / unknown X. On the 24-fact first-chat brain it produces genuine **multi-hop** chains (`curry -> pine ->
  clover`, `navel -> celeriac -> tanker`), 1-hop chains, dead-end abstains (`forehead`), and unknown abstains
  (`florbglax`). Moat holds at every hop.
- **Analogy route** — wired and fully functional, but **data-limited (stated honestly)**. "A is to B as C is to?" /
  "A:B::C:?" answered over a **standalone curated factored-relation KB** of **64 items across 4 BIJECTIVE families**
  (gender 16, capital_of 16, past-tense 16, comparative 16). All four families resolve at conf 1.000
  (prince→princess, rome→italy, jump→jumped, fast→faster); is_a / unknown-operand queries ABSTAIN. **HONEST
  CAVEAT:** the analogy KB is the explicit factored-relation structure the agent is GIVEN (the GO'd **regime A**);
  it does NOT operate over the brain's 1454 corpus-LEARNED codes (**regime B = the documented NO-GO**,
  `2026-06-27-tier2.1-analogy-NEGATIVE.md`). So the route answers analogies whose items the curated KB tracks and
  abstains on everything else — it is NOT analogy over the brain's corpus knowledge. 16 of the 64 KB items happen
  to also be in the corpus vocab, but the analogy codes are the KB's own factored codes, not the corpus codes.
- **No-confab moat:** 0 leaks across the full live transcript (14 probes incl. both new routes, their abstains,
  and 3 old Tier-0/1 probes). Both new-route records carry empty `emitted_propositions`, so `audit_moat` is clean
  by construction; the moat is enforced INSIDE the chain/analogy ops (abstain on dead-end / untracked / low-conf).
- **Default regression:** the stub `--rubric` is byte-identical to the pre-change baseline — **10/10, moat leaks
  0, MIXED, VERDICT PASS** — and `--demo` is **0 leaks (CLEAN)**. The two routes are additive (new regexes +
  handlers + a lazily-built guarded analogy KB); no existing route changed.

## What was added (`first_chat_console.py`, +112 lines, 0 deletions)

1. **Imports:** `build_knowledge_base` (the analogy KB). Chain-of-thought uses the composer's OWN
   `chain_of_thought` method (the exact op `self_cued_chain_demo.think` and `BrainConversationalAgent.chain_of_thought`
   delegate to) — reused directly on the console's composer; no demo import needed.
2. **Route regexes:** `_CHAIN_RE` (starting-from / what-comes-after / where-does-thinking-about-X-lead),
   `_ANALOGY_PROSE_RE` ("A is to B as C is to ?"), `_ANALOGY_COLON_RE` ("A:B::C:?"). Verified to NOT misfire on any
   existing-route phrasing (what-is-X, is-X-like-Y, which-X, what-does-X-Y, greeting, tell-me-more) and to match all
   intended forms.
3. **`_build_analogy_kb(brain)`** — lazy + guarded (any failure → `self.analogy_kb=None` → graceful abstain),
   mirroring the entity-instance layer pattern. Same seed as the brain, D=256 (the KB's validated dimension).
4. **`_chain_response(start)`** + **`_analogy_response(a,b,c)`** — both return `(paragraph, record)` with a
   moat-safe record; both ABSTAIN honestly (chain dead-end / unknown start; analogy untracked-operand / no-KB /
   low cleanup confidence).
5. **`respond()` wiring:** analogy + chain routes inserted after greeting/tell-me-more and BEFORE the wh / which /
   about routes (so the distinctive triggers are not consumed by a generic content-word opinion). An unknown chain
   start word routes to the existing specific `_clarify_unknown` ("I don't know X yet"). If the composer lacks
   `chain_of_thought` (a non-RF composer), the chain route falls through to the normal routes.

## Live real-brain transcript (`bridges/firstchat/brain1454_w7000_seed42.npz`, stub faculty, numpy-CPU)

```
[CHAIN (multi-hop)]   YOU: starting from curry, what follows?
  BRAIN: Starting from "curry", my thoughts run: curry -> pine -> clover. That's 2 self-cued hops, ending at "clover".
[CHAIN (multi-hop)]   YOU: where does thinking about navel lead?
  BRAIN: Starting from "navel", my thoughts run: navel -> celeriac -> tanker. That's 2 self-cued hops, ending at "tanker".
[CHAIN (1-hop)]       YOU: what comes after dragonfly?
  BRAIN: Starting from "dragonfly", my thoughts run: dragonfly -> cod. That's 1 self-cued hop, ending at "cod".
[CHAIN abstain dead-end]  YOU: starting from forehead, what follows?
  BRAIN: Starting from "forehead", nothing follows -- I have no association to chase from there.
[CHAIN abstain unknown]   YOU: starting from florbglax, what follows?
  BRAIN: I don't know the word "florbglax" yet -- it's not in what I've learned.

[ANALOGY gender]      YOU: king is to queen as prince is to?     BRAIN: "king" is to "queen" as "prince" is to "princess".
[ANALOGY capital]     YOU: paris:france::rome:?                  BRAIN: "paris" is to "france" as "rome" is to "italy".
[ANALOGY tense]       YOU: walk is to walked as jump is to what  BRAIN: "walk" is to "walked" as "jump" is to "jumped".
[ANALOGY comparative] YOU: big:bigger::fast:?                    BRAIN: "big" is to "bigger" as "fast" is to "faster".
[ANALOGY abstain is_a]    YOU: dog is to mammal as robin is to?
  BRAIN: I can't answer that analogy -- I don't track that kind of relation. I only do analogies over relations I
         know explicitly (gender, capital-of, past-tense, comparative).
[ANALOGY abstain unknown] YOU: king:queen::florbglax:?
  BRAIN: I can't answer that analogy -- I don't track a relation for 'florbglax'. ...

[OLD Tier-0 what-does]  YOU: what does dragonfly hum?  BRAIN: Good question. The dragonfly hums cod.       [moat OK]
[OLD Tier-0.3 wh]       YOU: who eat bear?             BRAIN: the canary eats the bear                     [moat OK]
[OLD Tier-0 engage]     YOU: what is bison?            BRAIN: I don't have settled facts about bison, but
                                                              it tends to come up alongside plantain, lynx,
                                                              and buffalo -- I'd be guessing past that.    [moat OK]

==> TOTAL MOAT LEAKS across the live transcript: 0
```

## Honest reporting on the analogy route (the key caveat)

The analogy route is **wired and works perfectly on its curated bijective relations**, but it is **data-limited**:
the brain's first-chat corpus codes are regime-B (raw learned concept codes), where producing meaningful relational
geometry is the documented open problem (`2026-06-27-tier2.1-analogy-NEGATIVE.md`). The analogy capability is GO
ONLY for **explicit factored bijective relations** (regime A), which the agent is GIVEN as a curated KB. So the
console answers analogies over gender / capital_of / past-tense / comparative and **abstains** on anything else —
it does NOT reason analogically over the brain's corpus knowledge. This is the honest result: a real, anti-cheated
relational-reasoning capability, scoped exactly to what the GO'd mechanism supports, made usable in chat. No
relations were fabricated to make the route look broader than it is.

## Verification

- **DEFAULT rubric** (`--rubric`, stub): **10/10, moat leaks 0, MIXED, VERDICT PASS** — byte-identical to the
  pre-change baseline (re-run before and after the edits).
- **DEFAULT demo** (`--demo`, stub): **0 leaks (CLEAN)**.
- **Live transcript** (above): chain (multi-hop + 1-hop + 2 abstains) + analogy (4 families + 2 abstains) + 3 old
  Tier-0/1 probes — **0 moat leaks**.
- **Imported-module CI guard:** `tests/test_factored_relation_analogy.py` — **7 passed, 1 GPU-skipped** (the module
  I import is unbroken).
- **Regex isolation:** verified the new triggers do not fire on any existing-route phrasing and do fire on all
  intended forms.

## Scope / NO `sim/` edit

Reuse-by-import only. The two routes are additive: new regexes + two handler methods + a lazily-built guarded
analogy KB on `FirstChatConsole`. No existing route, the DiscursiveTurn pipeline, the composer, the agent, or any
`sim/` file was changed. Default `--faculty stub` path is byte-unchanged (numpy-CPU; no torch needed).
