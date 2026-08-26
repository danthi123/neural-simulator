---
type: finding
status: contributing
lane: conversation-integration
date: 2026-08-25
seed-waiver: single-session qualitative re-run (one warm brain, one seed, seed=42) confirming INTEGRATION -- that two independently-verified landings (reasoning-frontier hardening's 37/37-turn battery; the DA-axis cupy-interop fix's own before/after + numpy-regression verify) cohere together live on the real endpoint. The individual mechanisms already carry their own multi-part verification elsewhere (cited below); this document's unique contribution is the live-composition check, which is inherently single-session by nature. No new capability GO is claimed here beyond what those two findings already established. The chain-route lesion (F1/F2) and the untaught/ambiguous-hop moat checks (E2/E6) are exact, deterministic, verified in-data (not seed-sensitive) -- see docs/TERMS.md "lesion".
---

# Integrated conversational-state diagnostic #2 -- reasoning + DA-axis landings confirmed LIVE together on the production brain (2026-08-25)

**Status:** contributing (single-session re-run of
[`2026-08-25-integrated-conversational-state-diagnostic.md`](2026-08-25-integrated-conversational-state-diagnostic.md),
confirming two landings that shipped since -- the compositional chain route and the DA-axis cupy-interop fix --
work together live through the real `/api/brain-chat` handler, and naming the next wall).

**Base:** `origin/main` @ `3bb9bfdf7`. Branch `research/integrated-convo-diagnostic-2`.
**Artifacts:** `research/findings/raw/_integrated_convo_diagnostic_2/transcript_2026-08-25.json` (24 turns; same
data as `.jsonl`), `run_log_2026-08-25.txt`, driver `_diag_driver_2.py`. Auto-provenance sidecars
(`*.prov.json`) record `git_sha: 3bb9bfdf7`, `sim_backend: cupy`. Every quote/number below is verbatim from that
transcript.

## Method

REUSED the original diagnostic's harness style exactly: one warm brain, reused across all 24 turns, driven
through the REAL production handler (`webapp/server.py brain_chat()` via FastAPI `TestClient`, in-process).
`SIM_BACKEND=cupy` (the owner's real path), `BRAIN_CHAT_RENDERER=stub` (this isolated worktree lacks the Qwen
priming corpus and `bridges/developed/*`, same as the original run -- confirmed absent before starting), opt-in
`BRAIN_SOURCE_PROVENANCE_HONESTY=1`, everything else production default. Seed 42, brain `tiny-demo`. All 24
turns returned HTTP 200; the run log shows zero tracebacks/exceptions; every response's `renderer` field reads
`"template-stub (GPU-free)"` throughout -- **the Qwen-mouth hang never occurred; no fallback to numpy was
needed.** The substrate build (first turn) took 232.6s, matching the original diagnostic's 278.2s first-turn
cost (building ~10 small Izhikevich organ networks); no single call exceeded that.

New phases beyond the original's regression turns: **D3/F1/F2** (the reasoning DERIVE + its live lesion via
`BRAIN_CHAIN_ROUTE=0`, toggled mid-session with no restart -- `chain_route_enabled()` reads the env var fresh on
every call) and **E1-E6** (the chain route's own moat hardening: an untaught second hop, and a genuinely
ambiguous first hop with two conflicting taught facts). Full turn list and exact bot replies are in the raw
transcript; representative ones are quoted below.

## 1. Per-faculty before/after (vs the 2026-08-25 original)

| Faculty | Original diagnostic (pre-fix) | This re-run (post-fix) |
|---|---|---|
| Multi-hop reasoning / own-conclusion | NEVER fires; every compositional question abstains or echoes one stored fact | **DERIVES.** `"what does the wolf's prey eat?"` -> `"I derived this from: wolf hunt deer; deer eat grass. The deer eats grass."` (D3, F2) |
| Reasoning route's moat | untested (route did not exist) | HOLDS. Untaught 2nd hop -> abstain (E2); a genuinely ambiguous 1st hop (eagle hunts BOTH fish and snake) -> abstain the WHOLE chain even though one branch (snake eats mice) is fully resolvable (E6) |
| Reasoning route load-bearing-ness | n/a | PROVEN by live ABA lesion: `BRAIN_CHAIN_ROUTE=0` reverts the identical question to abstain (F1); unset restores the derivation (F2), same warm brain, no restart |
| Dopamine axis | INERT every turn: `da_drives.reason = "error:ValueError: non-scalar numpy.ndarray cannot be used for fill"`, `da_level: 0.0`, `mode: rest` always | **LIVE.** `reason` is `"engaged"` or `"low_engagement"` on all 22 non-empty turns (never `error:`); `da_level` ranges 0.30-0.96, tracking `mode` (`rest`/`neutral`/`focus`) with message content |
| DA-gated encoding | inert (`g:1.0` always, no modulation) | modulated by the now-live `da_level` (not separately re-derived here; see the DA-axis fix finding's own 2(b)/2(c) for the isolated dose-response + SNc lesion) |
| Curiosity follow-up (D3/board) | DORMANT: `curious:false` on every abstain, no exception observed | **LIVE.** `curious:true` on **6 of 6** abstains this run (A2, C2, D1, E2, E6, F1), each appending an honest follow-up question; gated by `curiosity_da.da_level` matching that turn's `da_drives.da_level` exactly |
| Verb-inflection recall fragility | FRAGILE: `"the wolf hunts the deer"` taught, immediate `"what does the wolf hunt?"` ABSTAINED (hunts != hunt) | FIXED (a third, un-asked-for landing on the same arc: `aaf1ad5bc`, verb lemmatization at store+query). B2 recalls the just-taught fact immediately, no idle gap needed |
| In-loop learning | worked | still works (B1, D2, E1, E3-E5 teach cleanly) |
| Affect-drives (#84) | LOAD-BEARING: neutral 0.0 -> engaging message 0.407 <!--derived-->, lead `"Gladly --"`, persists via EMA | **UNCHANGED, still LOAD-BEARING.** Identical value: `ema_valence` 0.0 -> **0.40713**, `level` 0 -> 2, lead `"Gladly -- "` persists across every later turn (C1-F2) |
| Swap-drives (#85) | LOAD-BEARING: `"On wolf, then --"` / `"On dog, then --"` transition leads; `same_topic_hold` silent | **UNCHANGED.** `C4-topic-swap`: `reason: topic_change_swap`, lead `"On dog, then -- "`; same-topic turns hold silent |
| Continuous-state engine (wander) | LOAD-BEARING: `"(I'd been mulling over cat.)"` surfaced after an idle gap | **UNCHANGED, byte-identical text.** `B4-recall-after-idle2`: `"(I'd been mulling over cat.) The wolf hunts deer."` |
| Self-initiation | WORKS: genuine selection (pooled_member 0.417 vs pooled_random 0.036) <!--derived--> | **UNCHANGED, byte-identical utterance.** `D5`: `"Something's been on my mind -- cat eat worm. What does cat eat?"` (pooled_member 0.446 vs pooled_random 0.030 <!--derived--> -- same order of magnitude, same seed; exact: 0.44635240170522433 / 0.029737360160747257) |
| Comprehension/other-repair | WORKS: honest clarifying question on out-of-vocab tokens | **UNCHANGED, byte-identical.** `D6`: `"I followed the shape of that, but I don't know the words 'say' or 'mind' yet -- what do they refer to?"` |
| Source-provenance honesty (#129) | WIRED but off the default (rich) path; `provenance: null` on every rich turn | **UNCHANGED.** Still `provenance: null` on every turn (this run used the default `rich=True` path throughout, same as the original) -- an independent, still-open wiring gap, not a regression |
| 15k knowledge core on default brain | not reachable (`tiny-demo` has no LTM attached) | not re-tested this run (orthogonal to the 2 landings; already characterized) |

## 2. Reasoning DERIVE -- confirmed live end-to-end, with a clean lesion proof

Teaching `(wolf, hunt, deer)` (B1) then `(deer, eat, grass)` (D2), the possessive-clause question:

```
D3: "what does the wolf's prey eat?"
BOT: "Gladly -- I derived this from: wolf hunt deer; deer eat grass. The deer eats grass. -- worth going further here."
```

`resp.derived == True`, `resp.recalled_svo == None`, `resp.derived_from == [["wolf","hunt","deer"],
["deer","eat","grass"]]` -- exactly the documented `ChainedSVO` API contract (moat-hardening audit reqs #4/#5:
distinct shape, GENERATED not PERCEIVED, excluded from episodic/discourse-WM writes -- confirmed `episodic:
null` on this turn). The answer is framed as the brain's OWN inference ("I derived this from: ..."), not an
abstain, not a first-hop echo of `(wolf, hunt, deer)` alone.

**Live lesion (F1/F2), same warm brain, no restart** -- `BRAIN_CHAIN_ROUTE` is read fresh from `os.environ` on
every call, so the driver toggled it mid-session:

```
F1 (BRAIN_CHAIN_ROUTE=0): "what does the wolf's prey eat?" -> "...I don't know about that. My curiosity is
    piqued -- I haven't learned about wolf's yet..."   [reverts to abstain -- resp.derived == False]
F2 (env restored):        "what does the wolf's prey eat?" -> "I derived this from: wolf hunt deer; deer eat
    grass. The deer eats grass."                         [resp.derived == True, identical derived_from]
```

This is the load-bearing proof from the task, obtained directly rather than re-derived: the SAME question on
the SAME warm brain flips between abstain and derive purely on the lesion flag.

**Moat hardening confirmed live** (not merely read from code):
- **Untaught 2nd hop** (E1 taught only `fox hunts rabbit`, no fact for what a rabbit eats): `E2 "what does the
  fox's prey eat?"` -> abstain, `derived: false`. No fabrication.
- **Genuinely ambiguous 1st hop** (E3 taught `eagle hunts fish`, E4 taught `eagle hunts snake` -- two DISTINCT
  patients under the same `(eagle, hunt)` key -- E5 then taught a COMPLETE answer for one branch, `snake eats
  mice`): `E6 "what does the eagle's prey eat?"` -> abstain, `derived: false`, **even though the snake branch
  alone would have answered "mice."** This is the audit's ranked req #1 (multi-valued-hop abstain) holding on
  the live endpoint, not just in the runner's own unit tests: the route refuses to silently cherry-pick a
  resolvable branch when the first hop itself is contested.

**Precise scope of the new capability, named honestly.** `D4` asked the identical logical chain, spelled out
explicitly instead of as a possessive clause: `"the wolf hunts the deer and the deer eats grass; so what does
what the wolf hunts eat?"` -> abstain (misparsed "wolf" as an unknown topic, curiosity fired on the wrong thing).
The route detects exactly ONE regex shape (`"what does X's ROLE V?"`); a logically-identical question phrased
differently gets zero benefit from any of the hop machinery that just worked twice on D3/F2. See section 4.

## 3. DA axis -- confirmed live, no longer erroring, load-bearing on engagement

`da_drives.reason` was `"error:ValueError: non-scalar numpy.ndarray cannot be used for fill"` on **all 19**
turns of the original diagnostic. In this run it is `"engaged"` or `"low_engagement"` on **all 22** non-empty
turns (the 2 empty-message turns, D5/D6, short-circuit before the DA block, unrelated to this fix, exactly as
noted in the DA-axis fix finding's own 2(a) observation) -- **zero occurrences of `error:` anywhere in the
transcript.**

`da_level` tracks engagement within this one session (all EMA-folded, so absolute values drift with history --
this is a qualitative confirmation, not the isolated dose-response the DA-axis fix finding's own 2(b)/2(c)
already ran):

| turn | message shape | da_level | mode |
|---|---|---|---|
| C1 (rich/engaging: "Wow, I absolutely love wolves...") | high engagement | 0.7973 | focus |
| C3 (flat/curt: "wolf hunt what") | low engagement, same topic | 0.4579 | neutral |
| B3 (after a 25s idle gap, plain recall) | low engagement | 0.4944 | neutral |
| B4 (after a 2nd 25s idle gap) | low engagement | 0.3222 <!--derived--> | rest |

Across the run `da_level` spans **0.30 to 0.96**, and every turn's `mode` (`rest`/`neutral`/`focus`) tracks it
monotonically (rest <~0.36, neutral ~0.45-0.58, focus >~0.63 in this session) -- the isolated SNc-lesion proof
that this is load-bearing rather than coincidental is already in the DA-axis fix finding (2(c): lesion collapses
`da_level` to the exact same floor regardless of message content) and was not re-run here, since re-deriving an
already-verified mechanism would not add information.

## 4. Curiosity -- reversed from DORMANT to LIVE

The original diagnostic found `curious:false` on every one of its abstains, with no exception, and named this
"downstream of the DA bug." This run's abstains:

| turn | topic asked about | curious | curiosity_da.da_level |
|---|---|---|---|
| A2 | dragon | true | 0.8953 |
| C2 | phoenix (right after an engaging turn) | true | 0.7200 |
| D1 | "might" (a mis-parse, see below) | true | 0.5044 |
| E2 | fox's (untaught 2nd hop) | true | 0.5768 |
| E6 | eagle's (ambiguous 1st hop) | true | 0.4491 <!--derived--> |
| F1 | wolf's (chain route lesioned) | true | 0.3596 |

**6 of 6 abstains fired curiosity**, each appending an honest follow-up question ("My curiosity is piqued -- I
haven't learned about X yet: what can you tell me about X?"). `curiosity.curiosity_da.da_level` matches that
turn's top-level `da_drives.da_level` exactly on every occurrence, confirming the documented DA-gating (the
threshold scales with `da_level`; a low-DA turn like F1 at 0.36 still cleared it here, so the effective
threshold was not so strict as to make curiosity DA-only-in-name). **Curiosity is now genuinely live on the
integrated endpoint** -- a direct reversal of the original's finding, and exactly what fixing the DA bug was
expected to unlock (three default-ON faculties named "silently neutralized by one caught error").

**One honest wrinkle, newly visible now that curiosity fires.** On D1 ("what might the wolf chase?"),
curiosity attaches to the WRONG thing: the comprehension front end's fixed-arity truncation (see section 5,
unrelated to today's two landings) misreads "might" as an unknown topic noun, so the honest-sounding follow-up
("what can you tell me about might?") is itself a symptom of a mis-parse, not a genuine unknown-concept
abstain. Curiosity's SELECTION mechanism is doing its job (DA-gated, fires only on abstain); the CONTENT it
asks about inherits whatever the (separately known, unfixed) parser handed it.

## 5. No regression -- every original faculty re-confirmed, two byte-identical

Affect-drives, swap-drives, continuous-wander, self-initiation, and comprehension-repair all fired exactly as
in the original diagnostic (section 1's table); self-initiation's utterance (`"cat eat worm"` / `"what does cat
eat?"`) and comprehension-repair's clarifying question are **byte-identical strings** to the original run
(same seed, same tiny-demo facts, same deterministic paths -- see docs/TERMS.md "byte-identical": asserted in
the data, not inferred). The no-confab moat holds throughout: every unknown-topic question (A2 dragon, C2
phoenix) still abstains honestly rather than fabricating, now simply with an honest curiosity follow-up
attached where it did not fire before.

## 6. The single biggest next wall

**The original wall was "no multi-hop reasoning." That is now addressed for exactly ONE sentence shape.** The
new wall: **reasoning fires on a syntactic template, not on the logical structure of the question.** D4 proves
this directly -- the identical two facts, the identical 2-hop chain, phrased as an explicit conjunction instead
of a possessive clause, gets ZERO benefit from the hop machinery that had just correctly derived "grass" twice
(D3, F2) on the exact same facts moments earlier. It abstains and mis-fires curiosity onto a parser artifact
("might" in D1 shows the identical symptom from an unrelated angle).

The root cause is named, not hidden, in the shipped code's own documentation
(`research/runners/compositional_chain_route.py`, "PARSER TRUNCATION (audit req #6) -- CONFIRMED, not
re-fixed"): `ChatBrain._neural_question_parse` is **position-only over exactly 2 content tokens** and silently
discards a 3rd+; the new route works ONLY because it runs its own regex over the raw string BEFORE that
truncating parser ever sees the question, for ONE hand-matched shape (`"what does X's ROLE V?"`). Every
additional phrasing needs its own bespoke host regex -- this does not generalize, and the module's own honesty
section already names the real fix: "a spiking relation-extraction circuit is the named next rung," not another
regex. Until the comprehension front end itself can parse an arbitrary multi-clause/N-token question (ideally
on the substrate, per the project's brain-based-only standard, rather than as a widening whitelist of host
patterns), the brain's reasoning is a single hard-coded template wrapped around two genuinely-spiking hop reads
-- real, verified, load-bearing (section 2), but not yet GENERAL reasoning over compositional language.

**Ranked next mechanisms:**
1. **Fix the arity limit at the comprehension front end** (`ChatBrain._neural_question_parse`/`_extract_route`
   in `brain_chat_tui.py`), not by adding more regex shapes to the chain route. This is the single change that
   would let D4's phrasing (and any other N-token compositional question) reach the SAME hop machinery that
   already works, without a new bespoke detector per sentence shape.
2. **Replace the host regex/role-noun-hint detection layer itself with a learned relation-extraction
   circuit** on the substrate -- explicitly named as the honest next rung in the shipped module's own docstring,
   not a new observation from this diagnostic.
3. **Extend past 2 hops** and thread confidence across hops (both named "STILL OPEN" in the shipped module;
   untested by this diagnostic, which only exercised the validated 2-hop case).

## Caveats

Single session, single seed (42) -- see the frontmatter seed-waiver for why that is the right scope for an
INTEGRATION check specifically (the underlying mechanisms carry their own separate multi-part verification,
cited above). Stub renderer (this worktree lacks the Qwen priming corpus, confirmed absent before the run,
matching the original diagnostic's own documented caveat) -- fluency on the real Qwen mouth was not assessed;
every faculty coupling wraps the rendered text, so the couplings themselves are fully exercised regardless of
which mouth renders it. The knowledge-core-not-attached-to-default-brain gap and the provenance-null-on-rich-path
gap are both still present and were not re-derived in depth here, since neither is touched by the two landings
under test and both were already characterized in the original diagnostic.
