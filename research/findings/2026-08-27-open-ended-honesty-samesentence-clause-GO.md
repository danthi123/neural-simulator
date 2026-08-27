---
type: finding
status: contributing
date: 2026-08-27
mechanism: open-ended-clause-granularity-contradiction-repair
lane: E-language-open-ended-honesty
seeds: [42]
seed-waiver: A deterministic filter-logic verify over SAVED open-ended replies + hand-built same-sentence
  test items (no new Qwen render) — the evidence is keep/strip/leak counts against a fixed ground truth run
  through the REAL webapp.open_ended_chat.post_filter, not a stochastic effect, so a seed sweep would not
  change the answer (same waiver shape as the 2026-08-21/08-27 filter findings this arc builds on).
instrument: research/runners/_open_ended_clause_contradiction_filter_verify.py — runs the WIRED
  `webapp.open_ended_chat.post_filter` over same-sentence correct+wrong items, the 3 saved known-topic
  replies (10-item MUST_DROP regression), and no-salvage items, with tools.verdict.Verdict.
runner: research/runners/_open_ended_clause_contradiction_filter_verify.py
external: NO-EXTERNAL-NEEDED — a mechanical generalization (sentence- to clause/span-granularity) of an
  already-GO filter (2026-08-21/08-27), not a new mechanism; no literature dependency.
artifacts:
  - research/findings/raw/_open_ended_clause_contradiction_filter_verify.json
  - research/findings/raw/_open_ended_chat_known_supplement_wiring_verify.json
---
# Clause-granularity contradiction repair keeps the correct detail, strips only the wrong one (GO)

Artifact: research/findings/raw/_open_ended_clause_contradiction_filter_verify.json (GO).

**One line.** Vikunja #112's carried-over honesty limit — a CORRECT detail and a WRONG detail in the SAME
sentence were both dropped together, because `sentence_contradicts` (2026-08-21) works at whole-sentence
granularity — is closed. `_open_ended_clause_contradiction_filter_derisk.clause_filter_sentence` tries two
declared, span-level repairs (an appositive/relative-clause strip; a coordinated relation-object-list strip)
and re-verifies every edit against the UNCHANGED `sentence_contradicts` before ever keeping it, falling back
to the pre-existing whole-sentence drop whenever a repair can't be verified clean. Wired into
`webapp.open_ended_chat.post_filter`'s known-topic branch (BRAIN_OPEN_ENDED, still default-OFF).

## The two repairs (research/runners/_open_ended_clause_contradiction_filter_derisk.py)
**Repair 1 (appositive/relative clause).** "Ottawa, which was founded in 1867" → drops ", which was founded
in 1867", keeps "Ottawa". **Repair 2 (coordinated relation-object list).** "bordered by the United States
... and Mexico" → `_bad_relation_tokens` locates the store-WRONG token(s) by re-running
`sentence_contradicts`'s own border/continent gazetteer checks (imported unchanged: `COUNTRIES`,
`CONTINENTS`, `_obj`) — locating a SPAN is new, the wrongness DECISION is not — then `_remove_bad_tokens`
removes just that token plus its connecting comma/"and" glue and directional modifier, keeping the
store-correct token(s). A defense-in-depth re-check (`sentence_contradicts` on the edited text) and a
dangling-function-word guard (a cleaned sentence that would end/start on "by"/"and"/"which"/etc., e.g. every
border-list item was wrong) both fall back to None (whole-sentence drop) — never less safe than before.

## A real pre-existing gap this verify surfaced, fixed in the same file
`post_filter`'s old `" ".join(keep).strip() or reply.strip()` fell back to the RAW, unfiltered reply
whenever every sentence was dropped — leaking exactly what the filter exists to catch. Verified this
predates the clause work (the SAME leak reproduces against both `_base_post_filter`'s own known-topic branch
and the pre-clause sentence-level filter, unchanged code); it never fired on the 3 saved multi-sentence
replies because some sentence always survived. Fixed in `webapp/open_ended_chat.py`'s live `post_filter`
(the only code path a real turn takes) by falling back to `_empty_known_fallback(topic)`, a fixed,
non-fabricating honest string, instead of the raw reply.

## The verdict (research/runners/_open_ended_clause_contradiction_filter_verify.py) — GO
<!--derived-->
Run through the ACTUAL `webapp.open_ended_chat.post_filter`, over 4 same-sentence correct+wrong items (2
from the real saved canada reply — the border-list and capital-date sentences named above — plus 2
synthetic france/morocco items in the same shape, to check generalization beyond the motivating reply):
**4/4 keep the correct detail (united states/ottawa/paris/spain) and strip the wrong one
(mexico/1867/1523/algeria)** — 0/4 fall back to a full drop. **No regression**: the 10/10 known-wrong-detail
catch (mexico, 35 million, 1867, italy, germany, switzerland, algeria, tunisia, libya, egypt) from the
2026-08-27 wiring verify's own MUST_DROP ground truth still catches **10/10 with 0 leaks** through the
NOW-clause-aware filter, and **canada's specificity improves from 0 (the prior wiring verify's own disclosed
scope limit) to 4** — the correct Ottawa/North America/United States content that used to be lost with its
co-located wrong supplements now survives. **MOAT-safe**: 3 no-salvage items (france's fully-wrong
Italy/Germany/Switzerland border list, morocco's fully-wrong Algeria/Tunisia/Libya/Egypt border list,
canada's bare-number "35 million" sentence with no relative-clause boundary) leak **0/8** tracked wrong
tokens — two fall back to the pre-existing whole-sentence drop, all three now route through the
newly-fixed non-fabricating empty-fallback rather than the old raw-reply leak. **Byte-identical off**:
`webapp/server.py` is untouched by this change (confirmed via `git diff --stat -- webapp/server.py` showing
zero lines changed, the strongest available byte-identical evidence — not inferred from reading code), so an
off run imports nothing different from before; `open_ended_enabled()` still reads False unset / True at "1".
Re-running the 2026-08-27 wiring verify unmodified against this change also stays GO (10/10, 0 leaks,
canada specificity now 4 there too), confirming no regression on the prior arc's own test.

## Honest scope
**Disabled, named**: a general (non-gazetteer) entity check or NLI-based clause extraction. v1 makes exactly
the two OBSERVED same-sentence residual shapes (appositive/relative-clause date; coordinated relation-object
list) clause-safe; a bare unsupported claim with no relative-clause/list boundary (canada's "35 million"
sentence) has no declared-safe repair and keeps falling back to whole-sentence removal, same as before this
file existed — the SAME general-entity-check/NLI next rung the 2026-08-21 and 2026-08-27 findings already
name, now with a smaller residual (2 of the original 3 same-sentence loss cases) left to close. The
relative-clause repair is non-greedy to the next comma or sentence end (declared simplification; no case in
the observed data needs the fully general form of a relative clause followed by more independent content).

NEXT: the general entity-check/NLI rung named above, and re-pointing at the live-retrieval topic-routing gap
the 2026-08-27 finding already flagged separately (unrelated to this filter). NO `sim/` edit.
