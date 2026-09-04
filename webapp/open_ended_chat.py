"""BRAIN_OPEN_ENDED live-chat mode — the wiring of three proven de-risks into `/api/brain-chat` (default-OFF).

WHAT THIS IS. The owner reframed (2026-08-19): Qwen = a FORM scaffold; honesty = STATE-FIDELITY. Three de-risks made
that concrete and all are GO on `main`:
  * `_open_ended_state_driven_generation_derisk.OpenEndedGenerator` — assemble the brain's STATE (retrieved
    KNOWLEDGE + affect/familiarity/curiosity/self) and let the off-bridge spiking Qwen write a FREE, first-person,
    multi-sentence reply (conversational, V1 GO), instead of the telegraphic per-SVO strict composer.
  * `_open_ended_verify_postfilter_derisk.post_filter` — the VERIFY moat MOVED from a pre-hoc SVO constraint to a
    POST-FILTER on the free reply: strip persona leaks; on a brain-UNKNOWN topic keep only the honest hedges (and
    prepend an honest abstain if none remain); on a KNOWN topic drop sentences that its own `contradicts()` flags
    (a declared STUB that always returns False — see the 2026-08-21 wiring commit's own named gap). This dropped
    fabrication on Qwen-known/brain-unknown topics 1.0 -> 0.0 while keeping known substance 1.0 (GO).
  * `_open_ended_known_supplement_filter_derisk.sentence_contradicts` — the named fix for that stub: a per-sentence
    CONTRADICTION check (a stored relation — borders/continent/capital — asserted with a DIFFERENT object than the
    store holds, plus any bare number/year never in the store). Validated GO on the saved known-topic replies
    (catch_rate 1.0, 0 leaks, no reply emptied — research/findings/2026-08-21-contradiction-filter-catches-known-
    topic-wrong-supplements-GO.md). `post_filter()` below layers this on top of the base post-filter's known-topic
    path, closing the stub gap; the base post-filter's persona-strip + unknown-topic hedge/abstain logic is
    untouched.
  * `_open_ended_clause_contradiction_filter_derisk.clause_filter_sentence` (2026-08-27, Vikunja #112) — the SAME-
    SENTENCE residual the wiring above disclosed as "Honest scope": `sentence_contradicts` flags a whole sentence,
    so when a CORRECT detail and a WRONG detail land in the SAME sentence ("bordered by the United States [correct]
    ... and Mexico [wrong]"; "Ottawa [correct], which was founded in 1867 [wrong]"), the correct detail was dropped
    with it. This drops only the store-WRONG clause/span, re-verifies against the UNCHANGED `sentence_contradicts`
    before ever keeping edited text, and falls back to the prior whole-sentence drop whenever a repair can't be
    verified clean — never less safe than before this file existed.
  * `_open_ended_gen_time_consensus_veto_derisk.generate_with_generation_time_veto` (2026-08-28, Vikunja #112
    follow-on) — GENERATION-TIME honesty, additive + default-OFF (`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`, see
    `gen_time_honesty_enabled` below). Everything above runs AFTER the whole reply is written (generate, THEN
    strip). This mode steps the mouth ONE SENTENCE AT A TIME and checks each candidate against what the LIVE
    LTM-exempt organ-B/C spiking CONSENSUS (`webapp.gnw_two_organ_bus`/`gnw_three_organ_bus`, the SAME
    machinery that already authors the strict/rich recall path, production default-ON) actually COMMITS for
    that (topic, relation) — a genuine neural verdict, not the static python `FACTS` table — BEFORE the
    sentence is fixed into the context later sentences are generated from. An unsupported clause is
    suppressed/repaired there, so it never shapes what the mouth says next. Requires a live, organ-wired `chat`
    object (see `answer_turn`'s `chat=` parameter); with no `chat`, or the flag off, or an unknown topic, this
    mode never activates and the one-shot path above runs unchanged. The string post-filter (`post_filter`,
    everything above) still runs afterward on whatever this mode emits — a SAFETY NET, never bypassed.
  * `_open_ended_gen_time_consensus_veto_derisk._generate_tokenid_continuation_skip` (2026-08-28,
    `BRAIN_HONESTY_SKIP_CONTINUE`, default-OFF, see `skip_continue_enabled` below) — SKIP-AND-CONTINUE, stacked
    on top of `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` (all three flags must be truthy for anything to change). The
    generation-time veto above conservatively STOPS the whole reply at the first sentence it cannot repair; this
    mode instead DROPS that one sentence and CONTINUES generating the next one in the SAME reply (the token-id
    context still advances past the dropped span, so the mouth's own state moves forward), so a later,
    verifiable sentence is still reached instead of being truncated away with the unverifiable one. Same
    `clause_filter_sentence` veto, same string safety net afterward — only what happens AFTER a drop changes.
  * `webapp.wkv_mouth_generator` (2026-08-28, `BRAIN_OPEN_ENDED_WKV_MOUTH`, default-OFF, see
    `wkv_mouth_enabled` below) — a genuine CRUTCH-BURNDOWN lever, not a fourth honesty layer: swaps the FORM
    generator itself, for IN-VOCAB prompts only, from the literal Qwen2.5-0.5B model to a from-scratch,
    home-grown WKV/SSM spiking cortex (`bridges/wkv_ckpt`, V=1000, D=128, trained on TinyStories,
    architecturally unrelated to Qwen) reading its own next-word decision via a GENUINE few-spike Izhikevich
    soft-WTA population read (`research.runners._wkv_fewspike_read_derisk`, GO-verified), not a host argmax.
    `webapp.wkv_mouth_generator.in_vocab_scope(msg)` gates it: an out-of-vocab prompt (this checkpoint's
    vocabulary is V=1000 TinyStories-domain words, not general-purpose) is NEVER forced through it — `answer_turn`
    falls straight back to the existing Qwen path, unchanged. Every response field downstream (topic/known/facts/
    post_filter) is identical either way; only WHICH generator wrote `raw` differs. See that module's own
    docstring for the two honest residuals this rung does NOT resolve (the specific e-prop-learned read-out head
    was never persisted to disk, so this uses the checkpoint's own native head; the checkpoint's recurrent-store
    training method — local rule vs host-BPTT — is unverified).
  * `webapp.np_entailment_moat_gate` (2026-09-01, `BRAIN_OPEN_ENDED_NP_ENTAILMENT`, default-OFF, see
    `np_entailment_enabled` below) — wires NPHeadBinder (spiking NP-boundary binding,
    `_spiking_np_boundary_extraction_derisk.py`) + entailment classification (`FactStore`/`classify_claim`,
    `_open_text_moat_verifier_derisk.py`) into the KNOWN-topic branch of `post_filter`, ON TOP OF the existing
    gazetteer-based `_clause_filter_sentence`. The gazetteer only recognizes THREE relation shapes (borders/
    continent/capital) plus a bare number/year regex, so a wrong supplement on any OTHER relation ("Mercury
    discovered Neptune" against a store holding only mercury/orbits/sun) trips no branch and leaks through
    unedited. This gate extracts an (agent, action, patient) triple with the SAME spiking, vocabulary-agnostic
    role assignment other de-risks already validated (BridgeParser + NPHeadBinder, both reused UNCHANGED) and
    checks it with the SAME entailment semantics production's single-triple moat already uses — catching that
    whole class of wrong-relation confabulation the gazetteer is structurally blind to. Additive and
    MONOTONIC-ONLY: it can only drop a sentence the earlier stages already kept, is a no-op on anything it
    cannot confidently parse, on a claim not about the retrieved topic, and on a copula predicate nominal (an
    elaborative descriptive object the strict entailment check would false-reject — see that module's own
    docstring for why). See research/findings/2026-09-01-np-entailment-moat-gate-wired-into-live-open-ended-
    postfilter.md for the load-bearing before/after/lesion proof.
  * `webapp.wkv_mouth_generator.render_fact_sentence` (2026-09-01, `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE`,
    DEFAULT-ON as of 2026-09-01 — auto-flipped, GO, see `wkv_fact_sentence_enabled` below) — board #112 rung-3
    "clean unlock" WIRE-IN. The 2026-09-01 lexicon lever
    (`research/findings/2026-09-01-wkv-fact-to-sentence-lexicon-and-np-lever.md`, 6-seed GO) built a curated
    relation->predicate lexicon + slug->NP surfacer driving the already-6-seed-GO `SpikingClauseProducer` to
    render a real recalled fact as a coherent English clause, but left it a PARALLEL renderer never reachable
    from the mouth's own decode. This flag closes that: on a KNOWN, in-vocab topic,
    `webapp.wkv_mouth_generator.generate(sentence_facts=...)` tries the clause render FIRST, and when the
    topic's relation is lexicon-covered, THAT coherent sentence becomes the WKV mouth's actual reply (skipping
    free generation for the turn) — a known-topic reply now preserves the real fact instead of trading it for
    fact-thin TinyStories free-gen, for the narrow slice of real topics that are both in-vocab and lexicon-
    covered. INDEPENDENT of `wkv_fact_grounding_enabled()` (a separate `generate()` parameter, still default-
    OFF); falls straight through to the pre-existing free-gen/fact-boost path when no covered relation is
    found. ZERO PRODUCTION RISK today: gated two levels deep behind `BRAIN_OPEN_ENDED` (default OFF). See
    research/findings/2026-09-01-wkv-mouth-fact-sentence-wirein.md for the measured coverage + before/after.
  * `fact_clause_fallback_enabled` (2026-09-02, `BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK`, default-OFF) — the
    board #112 residual the rung-3 wire-in above explicitly named and left untouched: `render_fact_sentence`
    ALSO tried on a known topic the WKV mouth did NOT already handle (off, out-of-vocab, or an exception) —
    i.e. the real-traffic MAJORITY of known topics, which route to Qwen. Closes diagnosis (c) of
    `research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`: the retrieved facts
    ARE assembled into Qwen's prompt, but a pretrained Qwen does not reliably obey the "use ONLY KNOWLEDGE"
    instruction and supplements/overrides with its own confident, sometimes-wrong parametric detail, which the
    existing post-hoc string filters often cannot even parse to catch (copula/participial/pronoun prose). This
    flag ADDS the brain's own fact directly, moat-safe by construction, instead of relying on catching Qwen's
    fabrication after the fact. See research/findings/2026-09-02-open-ended-qwen-routed-fact-clause-fallback.md.
  * AFFECT INTO THE WKV MOUTH (2026-09-03, `BRAIN_WKV_MOUTH_AFFECT`, default-ON — see `webapp.wkv_mouth_generator.
    wkv_mouth_affect_enabled`) — closes an affect-hollow gap the 2026-09-03 linattn live-verification measured
    twice (an isolated valence sweep AND a live `BRAIN_AFFECT_LESION` pipeline test): this function already
    assembles the LIVE valence/arousal read off the real spiking affect organ into `state`/`system`/`user` (the
    Qwen prompt), but the WKV-mouth branch below never passed it to `_WKV.generate()` at all — `_free_gen`/
    `_free_gen_linattn` took no affect parameter, full stop. That call now passes `valence=float(valence),
    arousal=float(arousal))`, which drives a mood-congruent additive decode-time logit bias over a Warriner-
    gated, DR-2-learned-value word lexicon (`wkv_mouth_generator._apply_affect_bias`) — the same decode-control
    category the existing fact-boost/repetition levers already occupy; the genuine few-spike spiking WTA read
    still makes the actual word selection. `valence=0.0` (neutral mood, and exactly what `BRAIN_AFFECT_LESION=1`
    clamps the real organ's read to) is an exact no-op, so this is additive/byte-identical-at-neutral by
    construction. Fixes BOTH `BRAIN_WKV_MOUTH_RECURRENCE` families (`ssm`/`linattn`) from the one shared
    `generate()` entry point. Does NOT touch `render_fact_sentence`'s fact-clause path (facts stay tone-neutral
    by construction, matching Gate-B's own honesty floor). See research/findings/2026-09-03-affect-wiring-into-
    wkv-mouth-*.md for the vary/lesion load-bearing proof.
  * GENERATOR-TRACE MISLABEL FIX (2026-09-04, additive/guarded, no reply-content change — found during the
    2026-09-03 linattn live verification) — `webapp.wkv_mouth_generator.generate()`'s OWN `sentence_facts`
    branch (see the rung-3 wire-in bullet above) can render via `render_fact_sentence` INSIDE the WKV-mouth
    try-block below, but this function used to infer the `generator` trace label purely from WHICH try-block
    called `generate()`, not from what `generate()` itself did — so whenever that inner branch fired (the
    common case once `BRAIN_WKV_MOUTH_SCOPE=broad` routes nearly every prompt into the WKV-mouth try-block
    first, starving the separate FACT-CLAUSE FALLBACK branch below of ever running), the reply was actually
    written by the SAME SpikingClauseProducer mechanism the fallback branch names, but `generator` read
    `"wkv_mouth"` instead of `"spiking_clause"` — corrupting the per-touchpoint Qwen-vs-substrate provenance
    the one-brain roadmap's de-risk #2 depends on (research/findings/2026-09-03-one-brain-mouth-integration-
    ROADMAP.md SS3). Fixed by threading a `trace` dict through `_WKV.generate()` (see that function's own
    `trace` parameter) so THIS caller reads back which of `generate()`'s OWN internal branches produced `raw`
    and labels `generator`/`fact_clause_used`/`wkv_mouth_used` from THAT, independent of which of `answer_turn`'s
    two try-blocks reached it. `trace=None` (every pre-existing call site) is an exact no-op in `generate()`
    itself; here, the new `wkv_attempted` variable (see the WKV-mouth block below) additionally fixes a second-
    order bug the mislabel would otherwise have caused once the fallback branch's guard is corrected: gating the
    fallback on `not wkv_used` alone (now `wkv_used` no longer covers the inner-sentence-fact case) would have
    made it re-render the SAME fact a second time. See research/findings/2026-09-04-generator-trace-mislabel-
    fix.md for the root cause, the fix, and the byte-identical-reply verification.

THE LIVE RECIPE (per turn): extract the TOPIC from the user message -> RETRIEVE the grounded facts the live brain
holds about it (the LTM / chat bundle) -> ASSEMBLE a StateContext from the LIVE affect read (valence/arousal) +
familiarity/novelty/curiosity grounded in whether the store knows the topic -> `build_prompt` -> generate the reply
(FORM: `BRAIN_OPEN_ENDED_WKV_MOUTH` + an in-vocab prompt -> the WKV mouth's few-spike spiking decode; else —
`BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK` + a known topic -> the SAME brain-based fact->sentence render, reached
regardless of vocab; else the one-shot `OpenEndedGenerator.generate`, or — `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` +
a live `chat` + a known topic — the sentence-by-sentence generation-TIME consensus veto) -> `post_filter`
(HONESTY safety net: base persona-strip/hedge/abstain + the known-topic contradiction filter, always applied
either way, REGARDLESS of which generator wrote the reply) -> return the filtered reply.

MEMORY / COST DISCIPLINE (the two hard lessons this session).
  (1) ONE Qwen. `OpenEndedGenerator.__init__` would load a SECOND Qwen-0.5B. Instead we REUSE the server's already
      warm `SpikingQwenFaculty` (the one the `qwen` renderer loaded, `_get_warm_qwen_renderer()._fac`) by binding it
      onto an `OpenEndedGenerator` built with `__new__` (no model load). Exactly one Qwen lives in the process.
  (2) LOW MEMORY retrieval. The proven de-risk retrieves from the store's `facts.json` `by_agent` index (its own
      comment: "ground-truth agent index from the persisted facts.json ... for fast retrieval"). We read ONLY that
      11 MB `facts.json` (NOT the 96 MB phasor `composites.npz`, which the live brain already holds) into a cached
      `by_agent` map — the SAME retrieval source the de-risk used, without a second heavy store load.

BRAIN-BASED-ONLY NOTE (unchanged from the de-risk). The topic->facts router and the state->prompt assembly are the
declared HOST scaffolds (the same boundary the SVO parser occupies — host comprehension of the world input); the
retrieval CONTENT + no-confab abstain are the store's genuine facts, the valence is the real spiking affect organ's
differential, and Qwen is the owner-sanctioned conditioned-articulation FORM mouth. NO `sim/` edit; everything here
is reuse-by-import of modules already on `main`.
"""
from __future__ import annotations

import collections
import json
import os
import re
import threading
import time

# reuse-by-import: the three GO de-risk modules (on main). No local reimplementation of the mechanism.
from research.runners._open_ended_state_driven_generation_derisk import (
    StateContext, build_prompt, OpenEndedGenerator, _valence_from_differential, n_sentences, _sentences, persona_leak,
)
from research.runners._open_ended_verify_postfilter_derisk import post_filter as _base_post_filter
from research.runners._open_ended_clause_contradiction_filter_derisk import (
    clause_filter_sentence as _clause_filter_sentence,
)

# The self-model line the state carries (the de-risk's default; a held string, exactly as declared there).
SELF_MODEL = "a spiking brain that learns from conversation"


# ── HONESTY: the base VERIFY post-filter + the GO known-topic contradiction filter (2026-08-21) ──────────────────
def _facts_as_relation_pairs(facts):
    """Adapt the retrieval's (agent, action, patient) triples into the (relation, object) pairs
    `sentence_contradicts` expects -- the SAME (action, patient) shape the de-risk's own ground-truth FACTS table
    used (e.g. canada -> [("borders", "united states"), ("capital", "ottawa"), ...]). An interface adapter only;
    the contradiction-detection LOGIC stays entirely inside the imported de-risk module."""
    return [(str(v).lower(), str(p).lower()) for (_a, v, p) in facts]


def post_filter(reply, topic, known, facts):
    """The live post-filter: on a brain-UNKNOWN topic this IS `_base_post_filter` (persona-strip + the hedge/abstain
    path), unchanged. On a KNOWN topic it swaps the base filter's own stub `contradicts()` (declared to always
    return False) for the de-risked `sentence_contradicts` (2026-08-21, catch_rate 1.0 / 0 leaks / no reply emptied
    on the saved known-topic replies), closing the named gap: a KNOWN-topic reply's wrong parametric supplements
    (e.g. Canada "borders ... Mexico" when the store holds "united states") previously survived the stub unchanged.

    Mirrors `_base_post_filter`'s OWN known-topic structure (persona-leak-strip the reply's sentences, then drop
    the ones the contradiction check flags, then rejoin) rather than re-splitting `_base_post_filter`'s OUTPUT:
    `_sentences()` splits on [.!?]+ and DROPS the delimiters, so a second split over the base filter's own
    already-joined, punctuation-stripped text collapses back into ONE sentence and the per-sentence check goes
    inert (caught by this wiring's own verify -- see research/runners/_open_ended_chat_known_supplement_wiring_
    verify.py). Reuse-by-import only: `persona_leak` and `sentence_contradicts` are both imported verbatim from
    their GO de-risk modules; nothing here reimplements persona-leak or contradiction DETECTION.

    Per sentence, `_clause_filter_sentence` (2026-08-27) decides what to KEEP: the sentence unchanged when nothing
    contradicts; an EDITED sentence with only the store-wrong clause/span removed when a safe repair verifies clean
    (the same-sentence correct+wrong residual); or None (the sentence is dropped whole) whenever no repair can be
    verified clean -- the prior, still-correct behavior. `sentence_contradicts` itself is not reimplemented here or
    in that helper; only the SPAN to drop is newly located.

    A REAL, PRE-EXISTING gap this file's own MOAT-safety verify surfaced (2026-08-27): when EVERY sentence drops
    (a reply with no salvageable clause at all -- e.g. a single wholly-fabricated sentence), the old
    `" ".join(keep).strip() or reply.strip()` fell back to the RAW, UNFILTERED reply -- leaking exactly the
    fabricated content the filter exists to catch. This was ALREADY true of both `_base_post_filter`'s own
    known-topic branch and the pre-clause sentence-level filter this replaced (verified directly against both,
    unchanged code, before this fix); it simply never fired on the 3 saved multi-sentence replies because some
    sentence always survived. Fixed HERE (the only code path a live known-topic turn actually takes) by falling
    back to `_empty_known_fallback(topic)` -- a fixed, non-fabricating honest string -- instead of the raw reply.

    `BRAIN_OPEN_ENDED_NP_ENTAILMENT` (default OFF, see `np_entailment_enabled`): when truthy, a sentence the
    gazetteer-based `_clause_filter_sentence` KEPT is additionally screened by the spiking NPHeadBinder-
    extraction + entailment gate (`webapp.np_entailment_moat_gate.gate_sentence`) -- it can only drop that
    sentence further (monotonic-only; see that module's own docstring for the exact no-op scope), never restore
    what the gazetteer already removed. Flag off: this branch is never reached and `post_filter` is
    byte-identical to before this parameter existed."""
    if not known:
        return _base_post_filter(reply, topic, known, facts)
    pairs = _facts_as_relation_pairs(facts)
    sents = [s for s in _sentences(reply) if not persona_leak(s)]
    np_gate_on = np_entailment_enabled()
    if np_gate_on:
        from webapp.np_entailment_moat_gate import gate_sentence as _np_gate_sentence
    keep = []
    for s in sents:
        k = _clause_filter_sentence(s, topic, pairs)
        if k and np_gate_on:
            k = _np_gate_sentence(k, topic, facts)
        if k:
            keep.append(k)
    return " ".join(keep).strip() or _empty_known_fallback(topic)


def _empty_known_fallback(topic: str) -> str:
    """The KNOWN-topic honest fallback when every generated sentence turned out unverifiable: a FIXED,
    non-fabricating string (never the raw Qwen reply, which is exactly the leak this closes) -- the same honest-
    abstain category as `_base_post_filter`'s unknown-topic hedge-prepend, applied to "I generated something about
    a topic I know, but none of it held up.\""""
    return f"I don't have a version of what I just said about {topic} that I can actually stand behind."


# ── env flags (default-OFF) ─────────────────────────────────────────────────────────────────────────────────────
def open_ended_enabled() -> bool:
    """`BRAIN_OPEN_ENDED` truthy -> the open-ended state-driven + VERIFY-post-filter reply path. Default OFF (unset
    or 0/false/no/off) -> the block is skipped entirely and the existing strict/rich path runs byte-identically."""
    return os.environ.get("BRAIN_OPEN_ENDED", "0").strip().lower() in ("1", "true", "on", "yes")


def wkv_mouth_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_WKV_MOUTH` truthy -> `answer_turn` tries the from-scratch WKV mouth (genuine few-spike
    spiking-WTA decode, `webapp.wkv_mouth_generator`) FIRST, for IN-VOCAB prompts only, before falling back to the
    existing Qwen path. DEFAULT-ON as of 2026-08-30 (rung-3 crutch-burndown): unset now reads as ON, so when the
    open-ended channel is active the from-scratch mouth is the default in-vocab generator (set the flag to 0 to
    force the Qwen path). This is ZERO PRODUCTION RISK: it is a SECOND gate UNDER `BRAIN_OPEN_ENDED`, which is
    default-OFF and is the ONLY thing that imports this module (webapp/server.py) — with BRAIN_OPEN_ENDED off the
    open-ended block never runs, so production `answer_turn`/chat is BYTE-IDENTICAL to before the flip. When the
    channel IS on, an out-of-vocab prompt or any exception from the WKV path still falls back to Qwen and never
    crashes the turn."""
    return os.environ.get("BRAIN_OPEN_ENDED_WKV_MOUTH", "1").strip().lower() in ("1", "true", "on", "yes")


def gen_time_honesty_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` truthy -> `answer_turn` routes a KNOWN-topic reply through the
    sentence-by-sentence LTM-exempt organ-B/C spiking CONSENSUS VETO (generation-TIME honesty; see
    `research.runners._open_ended_gen_time_consensus_veto_derisk.generate_with_generation_time_veto`) instead of
    the one-shot `OpenEndedGenerator.generate` + string-only post-filter. Default OFF (unset/0/false/off/no):
    `answer_turn` runs its EXISTING one-shot path, byte-identical, regardless of whether a `chat` is passed in --
    this is a SECOND, independent gate on top of `BRAIN_OPEN_ENDED` (both must be truthy for anything to change).
    Even with the flag ON, this mode activates ONLY when `answer_turn` receives a live, organ-wired `chat` AND
    the topic is KNOWN (facts retrieved) -- an unknown topic's honest-abstain path is untouched either way, and
    with no `chat` (e.g. a caller that never passes one) the one-shot path runs unchanged. The string post-filter
    (`post_filter`) still runs afterward on whatever this mode emits, unconditionally -- a safety net, never
    bypassed, so this flag can only ever ADD a generation-time suppression, never remove the existing one."""
    return os.environ.get("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", "0").strip().lower() in ("1", "true", "on", "yes")


def np_entailment_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_NP_ENTAILMENT` truthy -> `post_filter`'s KNOWN-topic branch additionally screens
    every sentence the existing gazetteer-based `_clause_filter_sentence` KEPT through the spiking
    NPHeadBinder-extraction + entailment gate (`webapp.np_entailment_moat_gate.gate_sentence`) -- see that
    module's docstring for what it catches (wrong-relation confabulations the 3-relation gazetteer cannot see)
    and its exact no-op scope (monotonic-only: it can only drop a sentence, never restore one; a no-op on
    anything it cannot confidently parse, off-topic, or a copula predicate nominal). Default OFF
    (unset/0/false/off/no): `post_filter` never imports `webapp.np_entailment_moat_gate` (which in turn would
    pull in the spiking BridgeParser/NPHeadBinder build + their own `SIM_BACKEND` default) and is
    BYTE-IDENTICAL to before this flag existed. A SECOND, independent gate on top of `BRAIN_OPEN_ENDED`."""
    return os.environ.get("BRAIN_OPEN_ENDED_NP_ENTAILMENT", "0").strip().lower() in ("1", "true", "on", "yes")


def skip_continue_enabled() -> bool:
    """`BRAIN_HONESTY_SKIP_CONTINUE` truthy -> when generation-time honesty is ALSO active (see
    `gen_time_honesty_enabled` -- flag ON + a live `chat` + a KNOWN topic), a sentence the LIVE consensus veto
    cannot safely repair is DROPPED and generation CONTINUES to the NEXT sentence (the token-id context
    advances past the dropped span; see `research.runners._open_ended_gen_time_consensus_veto_derisk.
    _generate_tokenid_continuation_skip`) instead of truncating the whole reply there. A THIRD, independent
    gate stacked on top of `BRAIN_OPEN_ENDED` + `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` -- all three must be
    truthy for anything to change. Default OFF (unset/0/false/off/no): `answer_turn` passes
    `skip_continue=False` to `generate_with_generation_time_veto`, which is BYTE-IDENTICAL
    to the pre-existing conservative-truncate token-id path (the parameter's own default) -- this flag can only
    ADD a behavior (a later, verifiable sentence in the same reply is still reached after an earlier one was
    vetoed), never remove the existing truncating one, and never bypasses the string `post_filter` safety net,
    which still runs afterward on whatever text is ultimately emitted, unconditionally, exactly as before."""
    return os.environ.get("BRAIN_HONESTY_SKIP_CONTINUE", "0").strip().lower() in ("1", "true", "on", "yes")


def wkv_fact_grounding_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND` truthy -> when the WKV mouth (see `wkv_mouth_enabled`) is about to
    generate a KNOWN-topic reply, `answer_turn` passes the already-retrieved `facts` (the SAME triples the
    post-filter's contradiction check already uses -- no new retrieval, no new store access) into
    `webapp.wkv_mouth_generator.generate(facts=...)`, which gives their in-vocab CONTENT-word tokens an additive
    decode-time logit boost (`fact_grounding_ids` / `_apply_fact_boost`) so the genuine few-spike spiking WTA is
    more likely to actually select the TRUE recalled word, where this checkpoint's closed V=1000 vocabulary has
    one at all (board #112's named "clean unlock": let a known-topic free generation surface the brain's
    recalled fact). A FOURTH, independent gate: only reached when `BRAIN_OPEN_ENDED` + `wkv_mouth_enabled()` are
    ALSO truthy, the prompt is in-vocab, AND the topic is KNOWN. Default OFF (unset/0/false/off/no): `answer_turn`
    calls `_WKV.generate()` with `facts=None`, byte-identical to before this flag existed -- see
    research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md for the measured coverage ceiling (this
    checkpoint's vocabulary structurally cannot express the large majority of real Wikidata facts; this lever
    only helps the minority whose content word already exists in-vocab) and the before/after generation samples."""
    return os.environ.get("BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND", "0").strip().lower() in ("1", "true", "on", "yes")


def wkv_fact_sentence_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE` truthy -> when the WKV mouth (see `wkv_mouth_enabled`) is
    about to generate a KNOWN-topic reply, `answer_turn` passes the already-retrieved `facts` into
    `webapp.wkv_mouth_generator.generate(sentence_facts=...)`, which — via the merged board #112 rung-3 lexicon
    (`RELATION_LEXICON` + `slug_to_np`) driving the already-6-seed-GO `SpikingClauseProducer` — renders the
    FIRST fact whose relation the lexicon covers as a coherent factual clause (e.g. "the Isaac Asimov works for
    the University Of Boston"), FROM THE MOUTH's own `generate()` call, replacing that turn's free generation
    entirely (board #112's "clean unlock": a known-topic reply now preserves the actual recalled fact instead
    of trading it for fact-thin TinyStories free-gen — see research/findings/2026-09-01-wkv-mouth-fact-
    grounding-lever.md for the word-boost lever this supersedes for covered relations).

    A FIFTH, independent gate: only reached when `BRAIN_OPEN_ENDED` + `wkv_mouth_enabled()` are ALSO truthy,
    the prompt is in-vocab for the WKV checkpoint, AND the topic is KNOWN. INDEPENDENT of
    `wkv_fact_grounding_enabled()` — the two flags gate two SEPARATE `webapp.wkv_mouth_generator.generate()`
    parameters (`sentence_facts` here vs `facts`/the decode-time boost there); either, both, or neither may be
    on. When `sentence_facts` finds no covered relation (an honest degrade, not a crash), `generate()` falls
    straight through to its pre-existing free-generation path — with the boost lever still applied if THAT
    flag is also on.

    DEFAULT-ON as of 2026-09-01 (the wire-in's own 6-seed verify GO, auto-flipped per the 2026-09-01 owner
    policy: validated-GO + load-bearing + moat-safe + byte-identical-off + no-regression -- see
    `research/findings/raw/_wkv_mouth_fact_sentence_wirein_verify.json`, GO on all 6 seeds, 48/48 real cases:
    `generate()`'s own raw output is readable=faithful=moat_safe=1.0 on every seed; the end-to-end post-filtered
    answer is ALWAYS either that exact clause or the fixed honest-abstain fallback, never a corrupted hybrid,
    even on the one pre-existing-filter interaction named below). Unset now reads as ON; set the flag to 0 to
    force the pre-existing free-gen/boost path even when a covered relation exists. ZERO PRODUCTION RISK today:
    gated two levels deep behind `BRAIN_OPEN_ENDED` (default OFF) -- with that top-level channel off, this flag
    is never even read.

    HONEST SCOPE, unaffected by the flip: this covers only the narrow slice of real known topics that are ALSO
    in-vocab for the WKV checkpoint's closed V=1000 TinyStories vocabulary (empirically ~3% of a real 400-agent
    live-store sample, ALL of which had a lexicon-covered relation when in-vocab at all — see the 2026-09-01
    wire-in finding) — it does NOT touch the much larger Qwen-routed (out-of-vocab) known-topic grounding
    regression `research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md` measured,
    which is a different generator on a different code path entirely. A NAMED, MAPPED residual (not a blocker
    to the flip -- it fails SAFE): a pre-existing (2026-08-21) known-topic contradiction filter
    (`_open_ended_known_supplement_filter_derisk.sentence_contradicts`) flags ANY bare 3+-digit number/year in
    a sentence as "not in store," with no exemption for a number that is part of the topic's OWN slug/name
    (e.g. `1974_football_world_cup`) -- a documented pre-existing SCOPE limit of that filter, not of this wire-
    in. It causes the post-filter to over-cautiously fall back to the honest-abstain string on an otherwise-
    correct rendered clause for that narrow sub-class (1/48 sampled cases here) -- never a leak, always the
    safe degrade (verified 1.0 on every seed). Fixing that filter's number check is a separate, un-built next
    step, out of this wire-in's own scope. **This is exactly the residual `fact_clause_fallback_enabled` (below,
    2026-09-02) closes** -- the SAME `render_fact_sentence` mechanism, reached on a known topic regardless of
    whether it also happens to pass this checkpoint's free-gen `in_vocab_scope` gate."""
    return os.environ.get("BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE", "1").strip().lower() in ("1", "true", "on", "yes")


def fact_clause_fallback_enabled() -> bool:
    """`BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK` truthy -> the SAME mechanism `wkv_fact_sentence_enabled` wires into
    the WKV mouth (`webapp.wkv_mouth_generator.render_fact_sentence`) is ALSO tried on a KNOWN topic that the WKV
    mouth did NOT already handle (it is off, or the prompt failed `in_vocab_scope`, or it raised) -- i.e. the
    real-traffic MAJORITY of known topics, which route to Qwen. Closes the residual `wkv_fact_sentence_enabled`'s
    own docstring named but explicitly left untouched: `research/findings/2026-09-01-open-ended-bundle-moat-
    safety-soak-fabrication-delta.md` measured that on real Qwen-routed known-topic turns, the retrieved facts
    ARE assembled into the prompt (`build_prompt`'s KNOWLEDGE block, "Use ONLY the facts under KNOWLEDGE") but a
    pretrained Qwen does not reliably obey that instruction -- it supplements with confident, specific, WRONG
    parametric detail (`castleford_f_c`: "a professional **football** club" when the store's only sport fact is
    `rugby_leauge`) or sometimes ignores the facts entirely ("I don't have any information on..." despite 2 real
    stored facts) -- diagnosis (c) from that soak: retrieved AND injected, but overridden, not (a) never
    retrieved or (b) never injected. The existing post-hoc moat (`post_filter`, `NP_ENTAILMENT`, `GEN_TIME_
    HONESTY`) can only SUBTRACT a sentence it catches as wrong; on real Qwen prose (copula/participial/pronoun-
    heavy) it often cannot even parse the wrong clause to catch it (the soak's own measured example: `NP_
    ENTAILMENT` changed ZERO of 12 real known-topic replies). This flag instead ADDS the brain's own fact,
    unconditionally correct by construction (`render_fact_sentence` builds the surface ONLY from the fact's own
    subject/object NP + the fixed closed-class `RELATION_LEXICON` predicate/determiner -- no token Qwen or any
    other model chose can appear), so the reply is GROUNDED (attributable to the retrieved fact) rather than
    merely not-yet-caught-as-wrong.

    `render_fact_sentence` has NO dependency on the WKV checkpoint's V=1000 free-gen word-overlap vocabulary --
    that gate (`in_vocab_scope`) only scopes the checkpoint's OWN next-word spiking decode (`_free_gen`); the
    clause render is driven entirely by the closed-class `RELATION_LEXICON`/`slug_to_np` lookup + the already-
    6-seed-GO `SpikingClauseProducer`, a structurally different mechanism. That is WHY it generalizes here: a
    2026-09-01 scan (see the wire-in finding) found `RELATION_LEXICON` already covers 34/34 live relation types
    in the shipped `wikidata_core_15k` store, so this reaches essentially every real known topic with >=1 fact,
    not just the ~3% that also happen to pass the checkpoint's free-gen vocabulary gate.

    Reached only when `known` (facts were retrieved) AND the WKV mouth did NOT already produce `raw` for this
    turn BY EITHER of its own internal mechanisms (`not wkv_attempted`, 2026-09-04 -- was `not wkv_used` before
    the generator-trace-mislabel fix, research/findings/2026-09-04-generator-trace-mislabel-fix.md; the earlier
    name under-counted the case where `_WKV.generate()` itself already rendered via `sentence_facts` -- when it
    did, `wkv_fact_sentence_enabled`'s own gate on the identical mechanism already ran first; this flag never
    re-renders on top of that, and `answer_turn` labels that case `"spiking_clause"` too, from inside the
    WKV-mouth try-block). A hit means the rendered clause becomes `raw` and
    NEITHER Qwen NOR the generation-time consensus veto runs for this turn (a guaranteed-correct single fact
    replaces a free-form paragraph that might fabricate around it) -- `generator` reports `"spiking_clause"`
    and the new `fact_clause_used` trace key is `True`. A miss (no lexicon-covered relation in `facts`, or the
    clause producer did not genuinely spike) falls straight through to the PRE-EXISTING generation-time-honesty/
    Qwen path below, completely unchanged -- this flag can only ADD a generator choice, never remove one, and
    any exception here degrades safely to that same pre-existing path rather than crashing the turn. The
    pre-existing string `post_filter` (persona-strip + the known-topic contradiction filter) still runs
    afterward on whatever this path emits, unconditionally, exactly as for every other generator.

    DEFAULT-ON as of 2026-09-02 (this task's own 6-seed verify GO, auto-flipped per the SAME 2026-09-01 owner
    policy that flipped `wkv_fact_sentence_enabled`: validated-GO + load-bearing + moat-safe + byte-identical-
    off + no-regression -- see `research/findings/raw/_open_ended_qwen_fact_clause_fallback_verify.json`, GO on
    all 6 seeds, 48/48 real known+out-of-vocab+covered-relation cases sampled from the live store: raw
    readable=faithful=moat_safe=1.0 on every seed; the fake-Qwen stub fires on ZERO cases when the fallback
    handled the turn (a genuine bypass, not a decoration) and on EVERY case with the flag off (byte-identical
    routing, poison-pill-confirmed); unknown-topic honesty and routing are both unaffected (known=False
    short-circuits this branch regardless of flag state). Unset now reads as ON; set the flag to 0 to force the
    pre-existing generation-time-honesty/Qwen path even on a covered-relation known topic.

    HONEST TRADE-OFF, stated plainly (not hidden by the flip): because `RELATION_LEXICON` already covers 34/34
    live relation types in the shipped `wikidata_core_15k` store (see the wire-in finding's own coverage check),
    this reaches the large majority of real known-topic Qwen-routed turns, not a narrow slice -- a known topic
    with any covered fact now gets ONE short, terse, guaranteed-correct clause instead of Qwen's richer
    (but potentially fabricating) multi-sentence paragraph. This project's own standing priority (facts MUST
    drive the answer; an honest boundary is a deliverable, not a caveat) favors this trade explicitly, but it IS
    a real reduction in conversational richness on the turns it fires, disclosed here rather than left implicit.
    ZERO PRODUCTION RISK today regardless: gated behind `BRAIN_OPEN_ENDED` (default OFF) -- with that top-level
    channel off, this flag is never even read. Setting the flag to 0/false/off/no reduces `if wkv_used or
    fact_clause_used` to exactly `if wkv_used` -- the pre-existing branch, BYTE-IDENTICAL (verified directly, not
    inferred, by this task's own 6-seed verify: a poison-pill on `render_fact_sentence` never trips, and a
    fake-Qwen stub fires on every case), and `render_fact_sentence` is never imported by this branch (it may
    still be imported by the SEPARATE,
    pre-existing `wkv_fact_sentence_enabled` branch inside the WKV-mouth block, unaffected by this flag). See
    research/findings/2026-09-02-open-ended-qwen-routed-fact-clause-fallback.md for the 6-seed verify through
    the real `answer_turn` and the auto-flip decision."""
    return os.environ.get("BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK", "1").strip().lower() in ("1", "true", "on", "yes")


# ── topic extraction (host comprehension of the world input — the declared scaffold boundary) ────────────────────
# Longest-first so "what do you know about" wins over "what is"; each strips a natural lead-in to the bare entity.
_LEADINS = sorted([
    "what can you tell me about", "what do you know about", "what do you think about",
    "can you tell me about", "do you know anything about", "do you know about", "tell me about",
    "what is a", "what is an", "what is the", "what are the", "what's a", "what's the",
    "what is", "what's", "what are", "who is", "who was", "who's", "tell me", "describe", "explain", "about",
], key=len, reverse=True)


def extract_topic(msg: str) -> str:
    """Extract the bare topic entity from a natural user message. Host-side comprehension of the world input (the
    SAME boundary the SVO parser occupies). Exact-match retrieval keys off this, so a clean strip matters:
    'Tell me about Canada' -> 'canada'; 'what do you know about iron?' -> 'iron'; 'What is a zorplaxian?' ->
    'zorplaxian'. When nothing strips (a bare noun) the message itself is the topic."""
    t = (msg or "").strip().lower().rstrip("?.!").strip()
    for p in _LEADINS:
        if t.startswith(p + " "):
            t = t[len(p) + 1:].strip()
            break
    for art in ("a ", "an ", "the "):
        if t.startswith(art):
            t = t[len(art):].strip()
            break
    return t.rstrip("?.!").strip()


# ── low-memory retrieval index (the de-risk's `by_agent`, read from facts.json ONLY) ─────────────────────────────
_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[tuple, dict] = {}


def _read_facts_json(path: str) -> list:
    """Load a bundle's facts.json, tolerating both persisted schemas: the LTM store's list of
    {"shard": .., "fact": {agent, action, patient, polarity}} records, and a developed-brain bundle's
    {"schema_version": .., "facts": [{agent, action, patient, polarity}, ...]}."""
    fj = os.path.join(path, "facts.json")
    if not os.path.exists(fj):
        return []
    with open(fj, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):                       # developed-brain bundle schema
        return list(data.get("facts", []))
    return list(data)                                # LTM store schema (list of records)


def build_index(ltm_bundle: str | None, brain_bundle: str | None) -> dict:
    """A cached agent->[(a,v,p)] index over the AFFIRM facts in the LTM store (bulk knowledge) + the developed-brain
    bundle (self / working-set). This is the retrieval source the de-risk used (facts.json `by_agent`) — read WITHOUT
    the 96 MB phasor store, which the live brain already holds. Cached per (ltm, brain) path pair."""
    key = (ltm_bundle or "", brain_bundle or "")
    hit = _INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    with _INDEX_LOCK:
        hit = _INDEX_CACHE.get(key)
        if hit is not None:
            return hit
        by_agent: dict = collections.defaultdict(list)
        for path in (ltm_bundle, brain_bundle):
            if not path:
                continue
            for rec in _read_facts_json(path):
                f = rec.get("fact", rec) if isinstance(rec, dict) else None
                if not isinstance(f, dict):
                    continue
                if f.get("polarity", "AFFIRM") != "AFFIRM":
                    continue
                a, v, p = f.get("agent"), f.get("action"), f.get("patient")
                if a is not None and v is not None and p is not None:
                    by_agent[str(a).strip().lower()].append((str(a), str(v), str(p)))
        by_agent = dict(by_agent)
        _INDEX_CACHE[key] = by_agent
        return by_agent


def retrieve(by_agent: dict, topic: str) -> list:
    """The grounded facts the brain holds about `topic` (exact agent match; empty -> the genuine abstain/moat)."""
    return list(by_agent.get((topic or "").strip().lower(), []))


# ── one shared open-ended generator over the ONE warm faculty ─────────────────────────────────────────────────────
_GEN_LOCK = threading.Lock()
_GEN: OpenEndedGenerator | None = None


def get_generator(warm_faculty) -> OpenEndedGenerator:
    """Return the process-shared OpenEndedGenerator, binding the SERVER'S already-warm SpikingQwenFaculty onto it via
    `__new__` (NO second model load). `generate()` uses only `self.fac`, so this reuses the proven generation code
    verbatim with exactly one Qwen in the process. Built once under a lock."""
    global _GEN
    if _GEN is not None:
        return _GEN
    with _GEN_LOCK:
        if _GEN is None:
            gen = OpenEndedGenerator.__new__(OpenEndedGenerator)   # bypass __init__ -> no model load
            gen.fac = warm_faculty
            gen.name = "open-ended state-driven (reusing the warm server faculty)"
            _GEN = gen
        return _GEN


def valence_from_affect(differential) -> float:
    """Map the live affect organ's signed differential to a valence in [-1, 1] — the de-risk's own mapping."""
    try:
        return _valence_from_differential(float(differential))
    except Exception:
        return 0.0


# ── the turn ──────────────────────────────────────────────────────────────────────────────────────────────────
def answer_turn(msg: str, warm_faculty, valence: float, arousal: float, *,
                ltm_bundle: str | None, brain_bundle: str | None,
                seed: int = 42, max_new_tokens: int = 110, chat=None) -> dict:
    """One open-ended turn: STATE + retrieved knowledge -> a free reply (FORM) -> VERIFY post-filter (HONESTY
    safety net, always applied). The FORM generator is Qwen (`SpikingQwenFaculty`) UNLESS
    `wkv_mouth_enabled()` is truthy AND `msg` is in-vocab for the from-scratch WKV mouth's V=1000 checkpoint (see
    `webapp.wkv_mouth_generator`) — in which case that genuinely different, non-Qwen spiking cortex writes `raw`
    instead, via a real few-spike Izhikevich soft-WTA decode. Off, or out-of-vocab, or on any WKV-path exception:
    the Qwen path runs exactly as before this parameter existed.

    `chat` (default None): the LIVE, organ-wired production ChatBrain (already passed through
    `install_two_organ_gate`/`install_three_organ_gate` by the server). Consulted ONLY when the WKV mouth was NOT
    used AND `gen_time_honesty_enabled()` is truthy AND the topic is KNOWN — see that flag's docstring for the
    full gate. In that case the reply is generated sentence-by-sentence through the LTM-exempt organ-B/C spiking
    CONSENSUS VETO instead of one-shot (generation-TIME honesty); otherwise (WKV used, flag off, `chat is None`,
    or an unknown topic) the existing one-shot `OpenEndedGenerator.generate` path runs, byte-identical to before
    this parameter existed. Either way, `post_filter` still runs afterward — the safety net is never skipped,
    regardless of which generator wrote `raw`. When the gen-time veto path IS taken, `skip_continue_enabled()`
    (a FOURTH, independent, default-OFF gate — `BRAIN_HONESTY_SKIP_CONTINUE`) additionally controls whether a
    sentence the veto cannot repair truncates the reply there (default) or is dropped-and-skipped so a later
    verifiable sentence in the same reply is still reached — see that flag's own docstring.

    Returns a dict with the final `answer` (the filtered reply) plus a trace (`raw`, `filtered`, `topic`, `known`,
    `facts`, the assembled `state`, `gen_seconds`, `gen_time_honesty_used`, `gen_time_trace`, `generator` —
    `"wkv_mouth"`, `"spiking_clause"`, or `"qwen"` — `wkv_mouth_used`, and `fact_clause_used`). `known` is True
    iff the store held facts about the topic — the caller maps it to `abstained = not known` / `verified = known`
    (an unknown topic is an honest abstain). ALL THREE of `generator`/`wkv_mouth_used`/`fact_clause_used` follow
    the ACTUAL PRODUCER of `raw`, independent of which internal try-block reached it (fixed 2026-09-04, see
    research/findings/2026-09-04-generator-trace-mislabel-fix.md): `wkv_mouth_used` is True only when the
    genuine WKV/linattn free-gen spiking decode wrote `raw`; `fact_clause_used` (see `fact_clause_fallback_enabled`)
    is True whenever the SAME brain-based fact->sentence render (`render_fact_sentence`, the already-6-seed-GO
    `SpikingClauseProducer`) wrote `raw` instead — whether reached from the WKV mouth's OWN `sentence_facts`
    path (the mouth was attempted, and it happened to render via that mechanism) or the separate fact-clause
    FALLBACK below (the mouth was never attempted, or attempted-but-declined) — closing the much larger
    Qwen-routed known-topic grounding regression
    `research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md` measured."""
    by_agent = build_index(ltm_bundle, brain_bundle)
    topic = extract_topic(msg)
    facts = retrieve(by_agent, topic)
    known = len(facts) > 0
    fam = 0.9 if known else 0.1
    novelty = 1.0 - fam
    curiosity = 0.5 + 0.3 * novelty
    state = StateContext(topic=(msg or "").strip(), facts=facts, valence=float(valence), arousal=float(arousal),
                         familiarity=fam, confidence=fam, novelty=novelty, curiosity=curiosity,
                         self_model=SELF_MODEL, affect_source="real-organ")
    system, user = build_prompt(state)
    gen = get_generator(warm_faculty)

    # ── WKV MOUTH (crutch-burndown, default-OFF) ──────────────────────────────────────────────────────────────
    # `BRAIN_OPEN_ENDED_WKV_MOUTH` truthy AND the prompt is IN-VOCAB for the checkpoint's V=1000 TinyStories
    # vocabulary -> generate via the from-scratch WKV spiking mouth instead of Qwen. Flag off: `wkv_mouth_enabled()`
    # short-circuits before any import -> byte-identical to the pre-existing path below. Flag on + out-of-vocab, or
    # any exception from the WKV path: falls straight through to the existing Qwen path, unchanged -- this branch
    # can only ADD a generator choice, never remove or alter the fallback.
    wkv_used = False
    # `wkv_attempted` (2026-09-04, see the `generator`-trace-mislabel fix below): True whenever `_WKV.generate()`
    # itself successfully returned `raw` -- via EITHER of ITS OWN two internal branches (the genuine WKV/linattn
    # free-gen spiking decode, OR its own `sentence_facts`-driven `render_fact_sentence` call, see that
    # function's `trace` parameter). This gates the FACT-CLAUSE FALLBACK block below (never re-render when the
    # WKV call already produced `raw` by ANY mechanism) and is DELIBERATELY SEPARATE from `wkv_used`, which now
    # means only "the free-gen spiking decode was the actual producer" -- see the root-cause note at the
    # `wkv_trace` check below.
    wkv_attempted = False
    fact_clause_used = False
    generator_name = "qwen"
    if wkv_mouth_enabled():
        try:
            from webapp import wkv_mouth_generator as _WKV
            if _WKV.in_vocab_scope(msg, seed=seed):
                # de-risked 2026-08-28 (_wkv_rep_penalty_derisk.json, GO decode_suppressible): a decode-time
                # repetition guard kills the learned-head looping residual (2/2 -> 0 loops, byte-identical-off);
                # only reached when BRAIN_OPEN_ENDED_WKV_MOUTH is on, so the default path is unaffected.
                # board #112 fact-grounding lever (default OFF, see `wkv_fact_grounding_enabled`): on a KNOWN
                # topic, pass the ALREADY-retrieved `facts` (no new store access) so their in-vocab content
                # words get a decode-time boost. Flag off, or `known` False (nothing was retrieved): `facts=None`,
                # identical to before this parameter existed.
                ground_facts = facts if (known and wkv_fact_grounding_enabled()) else None
                # board #112 rung-3 wire-in (default OFF, see `wkv_fact_sentence_enabled`): a SEPARATE, INDEPENDENT
                # gate on the SAME already-retrieved `facts` -- `generate()` tries rendering a coherent clause
                # FIRST (via the merged lexicon lever + SpikingClauseProducer) before any free generation. Flag
                # off, or `known` False: `sentence_facts=None`, identical to before this parameter existed.
                sentence_facts = facts if (known and wkv_fact_sentence_enabled()) else None
                # AFFECT (2026-09-03, closes the affect-hollow gap named by research/findings/2026-09-03-linattn-
                # mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md (ii-c)): `state` above
                # already carries the LIVE valence/arousal read off the real spiking affect organ (this
                # function's own `valence`/`arousal` parameters -- see `answer_turn`'s docstring/caller in
                # webapp/server.py, `_OE.valence_from_affect(affect_info["differential"])`); `system`/`user`
                # (the Qwen prompt) already condition on it via `build_prompt`, but this WKV branch never passed
                # it to `_WKV.generate()` at all before this line -- a structural gap, not a bug in an existing
                # wire. `_WKV.generate()`'s own `valence=0.0`/`arousal=0.0` defaults are an exact no-op, so
                # passing the real floats here can only ADD a mood-congruent decode-time bias (see
                # `wkv_mouth_generator._apply_affect_bias`); it never changes behavior when the organ reads
                # neutral (including under `BRAIN_AFFECT_LESION=1`, which clamps the organ's differential -- and
                # therefore this `valence` -- to exactly 0.0).
                # `trace` (2026-09-04, the generator-trace-mislabel fix -- research/findings/2026-09-04-
                # generator-trace-mislabel-fix.md): a fresh dict `_WKV.generate()` fills in with which of ITS
                # OWN internal branches produced `raw`, so THIS caller no longer has to (wrongly) infer the
                # producer from which of ITS OWN branches called `generate()`.
                wkv_trace: dict = {}
                raw, secs = _WKV.generate(msg, seed=seed, max_new_tokens=max_new_tokens,
                                          repetition_penalty=1.3, no_repeat_ngram_size=3,
                                          facts=ground_facts, sentence_facts=sentence_facts,
                                          valence=float(valence), arousal=float(arousal), trace=wkv_trace)
                wkv_attempted = True
                if wkv_trace.get("sentence_fact_used"):
                    # ROOT CAUSE OF THE 2026-09-03 MISLABEL: `_WKV.generate()` itself tried `sentence_facts`
                    # FIRST (see `wkv_fact_sentence_enabled`) and its `render_fact_sentence` call found a
                    # lexicon-covered relation -- the SAME SpikingClauseProducer mechanism the FACT-CLAUSE
                    # FALLBACK block below wires in, reached from INSIDE this try-block instead. Under
                    # `BRAIN_WKV_MOUTH_SCOPE=broad`, `in_vocab_scope` admits nearly every prompt, so this
                    # branch (not the outer fallback below, which `wkv_attempted` now short-circuits) is what
                    # actually fires on most known-topic turns -- the label must follow the PRODUCER, not
                    # WHICH TRY-BLOCK called `generate()`.
                    fact_clause_used = True
                    generator_name = "spiking_clause"
                else:
                    wkv_used = True
                    generator_name = "wkv_mouth"
        except Exception:
            wkv_used = False               # never let a WKV failure crash the turn -- degrade to the Qwen path
            wkv_attempted = False

    # ── FACT-CLAUSE FALLBACK (board #112 residual, default-OFF) ──────────────────────────────────────────────
    # `BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK` truthy AND the topic is KNOWN AND the WKV mouth did NOT already
    # produce `raw` above (by EITHER of its own internal mechanisms -- `wkv_attempted`, see the block above,
    # 2026-09-04; was `not wkv_used` before the generator-trace-mislabel fix, which under-counted the case where
    # `_WKV.generate()` itself already rendered via `sentence_facts`) -> try the SAME brain-based fact->sentence
    # render (`render_fact_sentence`) on the already-retrieved `facts`, independent of `in_vocab_scope` (that
    # gate only scopes the checkpoint's OWN free-gen word decode; the clause render uses its own closed-class
    # lexicon + the already-6-seed-GO SpikingClauseProducer). This is what reaches the real-traffic MAJORITY of
    # known topics that route to Qwen -- see `fact_clause_fallback_enabled`'s docstring for the diagnosis this
    # closes. Flag off, `known` False, `wkv_attempted` True, no lexicon-covered relation, or any exception:
    # `fact_clause_used` stays False and control falls straight through to the UNCHANGED branch below -- this
    # can only ADD a generator choice.
    if not wkv_attempted and known and fact_clause_fallback_enabled():
        try:
            from webapp import wkv_mouth_generator as _WKVFC
            t0fc = time.time()
            sentence = _WKVFC.render_fact_sentence(facts, seed=seed)
            if sentence is not None:
                raw, secs = sentence, round(time.time() - t0fc, 3)
                fact_clause_used = True
                generator_name = "spiking_clause"
        except Exception:
            fact_clause_used = False        # never let this path crash a turn -- degrade below, unchanged

    gen_time_used = False
    gen_time_trace = None
    if wkv_used or fact_clause_used:
        pass                                # raw/secs already set above
    elif known and chat is not None and gen_time_honesty_enabled():
        try:
            from research.runners._open_ended_gen_time_consensus_veto_derisk import (
                generate_with_generation_time_veto,
            )
            t0 = time.time()
            raw, gen_time_trace, _consensus_info = generate_with_generation_time_veto(
                gen, chat, topic, seed, system, user, max_new_tokens=max_new_tokens,
                skip_continue=skip_continue_enabled())
            secs = round(time.time() - t0, 2)
            gen_time_used = True
        except Exception:
            # never let a generation-time-honesty failure crash a turn -- degrade to the one-shot path (still
            # honesty-safe: the SAME post_filter safety net runs on whatever this produces, unconditionally).
            raw, secs = gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)
    else:
        raw, secs = gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)

    filtered = post_filter(raw, topic, known, facts)
    return {
        "answer": filtered,
        "raw": raw,
        "filtered": filtered,
        "topic": topic,
        "known": known,
        "facts": [list(f) for f in facts],
        "n_sentences": n_sentences(filtered),
        "gen_seconds": secs,
        "gen_time_honesty_used": gen_time_used,
        "gen_time_trace": gen_time_trace,
        "generator": generator_name,
        "wkv_mouth_used": wkv_used,
        "fact_clause_used": fact_clause_used,
        "state": {"valence": float(valence), "arousal": float(arousal), "familiarity": fam,
                  "novelty": novelty, "curiosity": curiosity},
    }
