---
type: finding
status: contributing
date: 2026-08-27
mechanism: open-ended-chat-known-supplement-contradiction-wiring
lane: integration
seeds: [42]
seed-waiver: A deterministic filter-logic wiring verify over the SAVED open-ended replies (no new render) — the
  evidence is catch/leak counts against a fixed wrong-supplement ground truth run through the REAL webapp entry
  point, not a stochastic effect.
instrument: research/runners/_open_ended_chat_known_supplement_wiring_verify.py — runs the WIRED
  `webapp.open_ended_chat.post_filter` and the UNWIRED (lesioned) base post-filter over the saved known/hard
  open-ended replies and compares catch/leak/specificity/byte-identity, with tools.verdict.Verdict +
  tools.lab.attributable_to.
runner: research/runners/_open_ended_chat_known_supplement_wiring_verify.py
external: NO-EXTERNAL-NEEDED — this is a wiring verify of two already-GO de-risks (2026-08-21), not a new
  mechanism; reuses the prior de-risks' saved replies + store-fact ground truth.
artifacts:
  - research/findings/raw/_open_ended_chat_known_supplement_wiring_verify.json
---
# The GO known-topic contradiction filter is wired into `webapp.open_ended_chat.post_filter` (GO)

Artifact: research/findings/raw/_open_ended_chat_known_supplement_wiring_verify.json (GO).

**One line.** `webapp/open_ended_chat.py`'s post-filter (BRAIN_OPEN_ENDED, default-OFF) reused the base VERIFY
post-filter's `contradicts()` — a declared STUB that always returns False — so a KNOWN-topic reply's wrong
parametric supplements (Canada "borders ... Mexico" when the store holds "united states") previously survived.
The already-GO 2026-08-21 contradiction filter (`_open_ended_known_supplement_filter_derisk.sentence_contradicts`,
catch_rate 1.0 / 0 leaks on the saved known-topic replies) is now wired into `webapp.open_ended_chat.post_filter`,
closing that named gap. Still additive + default-off: the flag, the gating, and the unknown-topic moat are
byte-identical to before this change.

## The wiring (webapp/open_ended_chat.py)
`post_filter(reply, topic, known, facts)` is redefined in `webapp/open_ended_chat.py` (shadowing the base
import, now bound as `_base_post_filter`). On a brain-UNKNOWN topic it is `_base_post_filter` unchanged
(persona-strip + hedge/abstain). On a KNOWN topic it mirrors the base filter's OWN structure (persona-leak-strip
the reply's sentences, drop the ones a contradiction check flags, rejoin) but swaps the stub `contradicts()` for
the de-risked `sentence_contradicts`, imported verbatim from `_open_ended_known_supplement_filter_derisk.py`. An
adapter (`_facts_as_relation_pairs`) converts the retrieval's `(agent, action, patient)` triples into the
`(relation, object)` pairs `sentence_contradicts` expects — the SAME shape the 2026-08-21 de-risk's own
ground-truth FACTS table used. No detection logic is reimplemented; both `persona_leak` and
`sentence_contradicts` are imported, not rewritten.

**A real bug caught by dogfooding this wiring, before it shipped:** the first version re-split
`_base_post_filter`'s OWN output. `_sentences()` splits on `[.!?]+` and DROPS the delimiters, so the base
filter's `" ".join(keep)` output has NO punctuation left — re-splitting it collapsed the whole reply into ONE
giant "sentence," making every per-sentence check either fire on the entire blob (dropping everything, then
falling back to the unfiltered text via the `or` guard) or never fire at all. Operating on the RAW reply's own
sentence split instead (mirroring the base filter's structure rather than composing through its output) fixed
this; the runner's own module docstring documents it so it cannot silently regress.

## The verdict (research/runners/_open_ended_chat_known_supplement_wiring_verify.py) — GO
<!--derived-->
Over the 3 saved known-topic replies (canada/france/morocco) run through the ACTUAL
`webapp.open_ended_chat.post_filter`, with their store facts expressed as production-shaped
`(agent, action, patient)` triples: **10/10 wrong supplements caught, 0 leaked** (catch_rate_wired 1.0), and
every filtered reply stays non-empty. Lesioning back to the unwired base filter (its stub `contradicts()`
restored) on the SAME raw replies: **catch_rate_lesioned 0.0, all 10 leak** — `attributable_to("known-topic
contradiction catch: wiring vs lesioned stub", 1.0, 0.0)` = **1.0**, i.e. the entire catch is attributable to
this wiring, not something else in the pipeline. The 8 saved brain-unknown ("hard") replies are
**byte-identical** through the wired vs. base filter and fabrication stays suppressed on all 8 — the unknown-topic
moat is untouched. Flag-off: `webapp/server.py`'s `open_ended_chat` import is still nested directly under the
unchanged `BRAIN_OPEN_ENDED` truthy guard (structurally verified, not just asserted), and `open_ended_enabled()`
reads False when unset and True when set — so a default-off run never imports the changed module.

## Honest scope
**Correct substance is preserved for france/morocco** (specificity 3 and 4 respectively — capital + continent
survive), but **not for canada in this saved reply**: Qwen's canada reply put its correct facts (Ottawa, North
America, United States) in the SAME sentences as its wrong ones ("bordered by the United States ... and Mexico";
"Ottawa, which was founded in 1867"), so the per-SENTENCE filter drops both together (specificity 0, though the
reply stays non-empty/conversational). This is **inherited, not introduced by this wiring** — the standalone
2026-08-21 de-risk measured the identical canada `n_kept=1` before this wiring existed, and its own "Honest
scope" already names per-clause splitting / a store-backed entity check / an NLI model as the general next rung.
Wiring the v1 filter as approved does not change that scope; it makes the ALREADY-DISCLOSED tradeoff live.

**A second, pre-existing gap noticed but out of scope here:** the CURRENT shipped default LTM bundle
(`~/Projects/sim-data/knowledge_bundles/wikidata_core_15k`) keys country entities like `canada_portal`, not the
bare `canada` a user types — so `webapp.open_ended_chat.retrieve('canada')` against TODAY's bundle returns `[]`
independent of this wiring (a topic-extraction/retrieval-routing mismatch, not a contradiction-filter defect).
This verify used the 2026-08-21 de-risk's own documented store-fact ground truth (in production triple shape)
run through the real `post_filter` entry point, which is what proves the WIRING; it does not re-validate today's
bundle's exact-match coverage. Flagged separately (spawn_task) rather than folded into this change's scope.

NEXT: the per-clause / store-backed-entity generalization named above; otherwise this closes the known-topic
honesty gap the 2026-08-21 wiring commit itself disclosed. NO `sim/` edit.
