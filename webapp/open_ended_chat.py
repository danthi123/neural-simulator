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

THE LIVE RECIPE (per turn): extract the TOPIC from the user message -> RETRIEVE the grounded facts the live brain
holds about it (the LTM / chat bundle) -> ASSEMBLE a StateContext from the LIVE affect read (valence/arousal) +
familiarity/novelty/curiosity grounded in whether the store knows the topic -> `build_prompt` -> `OpenEndedGenerator.
generate` (FORM) -> `post_filter` (HONESTY: base persona-strip/hedge/abstain + the known-topic contradiction filter)
-> return the filtered reply.

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

# reuse-by-import: the three GO de-risk modules (on main). No local reimplementation of the mechanism.
from research.runners._open_ended_state_driven_generation_derisk import (
    StateContext, build_prompt, OpenEndedGenerator, _valence_from_differential, n_sentences, _sentences, persona_leak,
)
from research.runners._open_ended_verify_postfilter_derisk import post_filter as _base_post_filter
from research.runners._open_ended_known_supplement_filter_derisk import (
    sentence_contradicts as _known_supplement_contradicts,
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
    their GO de-risk modules; nothing here reimplements persona-leak or contradiction DETECTION."""
    if not known:
        return _base_post_filter(reply, topic, known, facts)
    pairs = _facts_as_relation_pairs(facts)
    sents = [s for s in _sentences(reply) if not persona_leak(s)]
    keep = [s for s in sents if not _known_supplement_contradicts(s, topic, pairs)]
    return " ".join(keep).strip() or reply.strip()


# ── env flag (default-OFF) ──────────────────────────────────────────────────────────────────────────────────────
def open_ended_enabled() -> bool:
    """`BRAIN_OPEN_ENDED` truthy -> the open-ended state-driven + VERIFY-post-filter reply path. Default OFF (unset
    or 0/false/no/off) -> the block is skipped entirely and the existing strict/rich path runs byte-identically."""
    return os.environ.get("BRAIN_OPEN_ENDED", "0").strip().lower() in ("1", "true", "on", "yes")


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
                seed: int = 42, max_new_tokens: int = 110) -> dict:
    """One open-ended turn: STATE + retrieved knowledge -> free Qwen reply (FORM) -> VERIFY post-filter (HONESTY).

    Returns a dict with the final `answer` (the filtered reply) plus a trace (`raw`, `filtered`, `topic`, `known`,
    `facts`, the assembled `state`, `gen_seconds`). `known` is True iff the store held facts about the topic — the
    caller maps it to `abstained = not known` / `verified = known` (an unknown topic is an honest abstain)."""
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
        "state": {"valence": float(valence), "arousal": float(arousal), "familiarity": fam,
                  "novelty": novelty, "curiosity": curiosity},
    }
