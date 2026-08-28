"""GENERATION-TIME consensus-veto honesty for open-ended chat (Vikunja #112 follow-on, 2026-08-27/28/29).

TOKEN-ID CONTINUATION (2026-08-28, this file's own NEXT rung from the 2026-08-27-open-ended-generation-time-
honesty-PARTIAL.md finding). The original v1 stepped the mouth sentence-by-sentence by DECODING the growing
accepted text back to a string and RE-TOKENIZING `prompt_string + accepted_text_string` on every single step
(`_continue_chunk`, kept below for direct A/B comparison) -- an honest, disclosed text-roundtrip confound that
made the live decode NOT byte-identical to one-shot generation even when nothing was ever suppressed, capping
the live-mouth vary/lesion demonstration at 1/3 topics. `generate_with_generation_time_veto` now defaults to
`continuation="token_id"` (`_generate_tokenid_continuation` / `_continue_chunk_ids`): the growing context is
carried as TOKEN IDS, never decoded-and-reencoded. A KEPT sentence (the common case: nothing to suppress) has
its own model-generated ids appended to the context DIRECTLY -- zero retokenization, byte-identical continuation
to what one-shot generation would have produced up to that point. Only a REPAIRED sentence (an actual text
edit -- the store-wrong span was removed) requires a re-encode, and only of that one repaired span, not the
whole accumulated reply -- a narrow, disclosed, unavoidable exception (you cannot continue token IDs through an
edit that changed the text) instead of the old confound's every-step-every-sentence roundtrip. `continuation=
"text"` still runs the original v1 path unchanged, for a direct same-run comparison of the two technique's
live-mouth divergence rates (see `run_battery`). NO `sim/` edit; no change to `clause_filter_sentence` /
`sentence_contradicts` / `consensus_facts_for_topic` / the string safety net -- only the ORCHESTRATION of how
the off-bridge Qwen mouth is stepped between them.

THE GAP THIS CLOSES. `webapp/open_ended_chat.py`'s honesty moat runs AFTER the free Qwen reply is fully written:
generate the WHOLE thing, then strip/repair whatever `sentence_contradicts` (a host STRING match against a static
FACTS table) flags. The brain's own knowledge is not shaping the mouth AS it speaks -- it is cleaning up after it.
This file makes the LTM-exempt organ-B/C spiking CONSENSUS VETO (`webapp.gnw_two_organ_bus.two_organ_combine` /
`webapp.gnw_three_organ_bus.three_organ_combine`, the SAME machinery that already authors the strict/rich recall
path and is production DEFAULT-ON) a GENERATION-TIME signal: the off-bridge Qwen mouth is stepped ONE SENTENCE AT
A TIME, and each candidate sentence is checked against what the SUBSTRATE's own consensus ignition actually
COMMITS for that (topic, relation) pair -- not a python dict -- BEFORE the sentence is fixed into the context that
shapes the next one. An unsupported clause is suppressed/repaired THERE; only the survivors ever become part of
what later sentences are generated from. The string post-filter (`_open_ended_known_supplement_filter_derisk` /
`_open_ended_clause_contradiction_filter_derisk`) is UNCHANGED and stays layered on top as a safety net.

THE MECHANISM.
  (1) A lightweight (numpy, CPU, no GPU), genuinely-spiking chat: the existing tiny-demo buffer composer
      (`brain_chat_tui._build_tiny_demo`, an UNRELATED brain/dog/cat vocabulary) + a FRESH, small
      `ShardedPhasorStore` LTM seeded with exactly the canada/france/morocco relation facts the string-based
      no-regression battery (`_open_ended_known_supplement_filter_derisk.FACTS`) already uses, composed via
      `TieredFactStore` -- the identical drop-in the production knowledge-in-chat flip uses. A query for one of
      these (agent, action) pairs is a genuine LTM-TIER recall (`query_patient_source` reports "ltm"), so
      `BRAIN_GNW_ORGANB_LTM_EXEMPT` is exercised faithfully (not vacuously) -- organ B/C corroborate a stable
      cortical fact instead of withholding on a buffer-only expectation registry that never covers it.
  (2) `consensus_facts_for_topic(chat, topic, seed)` -- for each of the structural relations the string filter
      checks (capital / continent / borders), ask the CONSENSUS what it commits (via `two_organ_combine` /
      `three_organ_combine`, NOT `_open_ended_known_supplement_filter_derisk.FACTS`). This produces the SAME
      shape of (relation, object) pairs `sentence_contradicts` / `clause_filter_sentence` already expect, but
      SOURCED from the substrate's own ignition instead of a static dict -- a drop-in provenance swap, reusing
      both functions BY IMPORT, unmodified.
  (3) `generate_with_generation_time_veto(...)` -- the off-bridge Qwen mouth generates ONE SENTENCE at a time
      (greedy continuation from the growing ACCEPTED text, deterministic given the seed). Each candidate sentence
      is run through the imported `clause_filter_sentence(candidate, topic, consensus_facts)`: unchanged if
      nothing contradicts; REPAIRED (only the store-wrong span removed) if a safe repair verifies clean; dropped
      (conservative truncation -- generation STOPS there, v1 scope, see below) if neither. Only the ACCEPTED
      (possibly repaired) text is ever appended to what subsequent sentences are generated from -- a wrong clause
      literally never becomes part of the mouth's own context, vs. a post-filter that always saw it.

THE COUPLING LESION (the load-bearing lever, distinct from the organs' own internal biology levers already
proven in `_gnw_two_distinct_organs_derisk.py` / the LTM-exempt de-risks). `lesion_coupling=True` makes
`consensus_facts_for_topic` return `([], {})` WITHOUT ever calling the consensus -- the generation-time step has
nothing to check a candidate against, so it never intervenes (byte-identical to a naive sentence-by-sentence
mouth with no honesty coupling at all). This isolates exactly what THIS wiring contributes, separate from
whether the organs' own spiking reads are themselves intact.

SCOPE (honest, v1). (a) Relation-object contradictions only (capital/continent/borders), matching the string
filter's own structural checks -- a bare unsupported number/date (e.g. "35 million") is caught by
`sentence_contradicts`'s OWN facts-independent branch regardless of this file's consensus signal (unaffected
either way -- not this file's contribution, reported honestly, not attributed to it). (b) On a repair-fail, v1
conservatively STOPS generation (truncates the reply there) rather than skip-and-continue past the dropped
sentence -- never fabricates past an unverifiable point, but a later sentence within the SAME reply may go
untested by this pass (an honest, disclosed scope limit, not a moat gap: the string post-filter still runs over
whatever text a caller does emit). (c) Only KNOWN topics are in scope (an unknown topic's abstain path is
untouched, exactly as `webapp/open_ended_chat.py`'s existing post-filter already scopes it).

BRAIN-BASED-ONLY NOTE. The relation set + sentence-boundary chunking are HOST scaffolds (the same boundary the
SVO parser and the string post-filter already occupy); the CONSENSUS VERDICT itself (organ A recall + organ B
surprise-corroboration + organ C comprehension, coincidence-gated in the shared GNW workspace) is the genuine
spiking read, reused by import, unmodified. NO `sim/` edit. Reuse-by-import only: `two_organ_combine` /
`three_organ_combine` / `clause_filter_sentence` / `sentence_contradicts` / `OpenEndedGenerator` are all imported
verbatim; nothing here reimplements the consensus, the repair, or the mouth.

    SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -m \
        research.runners._open_ended_gen_time_consensus_veto_derisk \
        --out research/findings/raw/_open_ended_gen_time_consensus_veto_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# reuse-by-import: the string moat (unchanged), the mouth, and the consensus buses -- NO reimplementation.
from research.runners._open_ended_known_supplement_filter_derisk import (   # noqa: E402
    sentence_contradicts, FACTS as STRING_FACTS, MUST_DROP,
)
from research.runners._open_ended_clause_contradiction_filter_derisk import clause_filter_sentence  # noqa: E402
from research.runners._open_ended_state_driven_generation_derisk import (   # noqa: E402
    StateContext, build_prompt, OpenEndedGenerator, _sentences,
)
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_gen_time_consensus_veto_derisk.json"

# the structural relations the string filter checks (matches STRING_FACTS' own shape) -- the SAME three; only the
# PROVENANCE of the (relation, object) pairs changes (substrate consensus vs static dict).
RELATIONS = ("capital", "continent", "borders")

# the EXACT facts `_open_ended_state_driven_generation_derisk.json`'s saved canada/france/morocco runs retrieved
# from the real 100k store (verified against the saved artifact) -- reproduced here as a tiny, self-contained LTM
# so this file needs no heavy store load. (agent, action, patient) triples, fed to a fresh ShardedPhasorStore.
LTM_FACTS = {
    "canada":  [("canada", "capital", "ottawa"), ("canada", "continent", "north america"),
                ("canada", "borders", "united states")],
    "france":  [("france", "capital", "paris"), ("france", "continent", "europe"),
                ("france", "borders", "spain")],
    "morocco": [("morocco", "capital", "rabat"), ("morocco", "continent", "africa"),
                ("morocco", "borders", "spain")],
}

_SENT_END_RE = re.compile(r"[.!?]+")

# torch imported lazily inside the functions that need it (kept out of module scope so this file's pure-logic
# pieces -- consensus_facts_for_topic, run_controlled_unit_battery -- stay importable/runnable with no GPU/torch
# dependency at all, exactly as before this token-id-continuation addition).


# =====================================================================================================
# (1) A lightweight, genuinely-spiking chat with a real LTM tier over the SAME topics the string battery uses.
# =====================================================================================================
def build_consensus_chat(seed: int):
    """The tiny-demo buffer composer (brain/dog/cat -- unrelated vocabulary) + a FRESH small `ShardedPhasorStore`
    LTM holding exactly `LTM_FACTS`, composed via `TieredFactStore` (the identical drop-in the production
    knowledge-in-chat flip uses). A query for a canada/france/morocco relation is a genuine LTM-tier recall, so
    the LTM-exemption lever is exercised faithfully. NO GPU; numpy; a few CPU-seconds to build."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    vocab = sorted({w for facts in LTM_FACTS.values() for (a, v, p) in facts for w in (a, v, p)})
    ltm = ShardedPhasorStore(n_shards=1, seed=seed, D=128, vocab=vocab)
    for facts in LTM_FACTS.values():
        for (a, v, p) in facts:
            ltm.store(a, v, p, polarity="AFFIRM")
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


# =====================================================================================================
# (2) THE GENERATION-TIME HONESTY SIGNAL: per-relation consensus verdict, NOT a static python dict.
# =====================================================================================================
def consensus_facts_for_topic(chat, topic: str, seed: int, *, lesion_coupling: bool = False, bus: str = "three",
                              organb_lesion: bool = False, organc_lesion: bool = False,
                              ws_lesion: bool = False):
    """Ask the LTM-exempt organ-B/C spiking consensus what IT commits for (topic, relation), for every relation
    the string filter structurally checks. Returns `(facts, info)`: `facts` is `[(relation, committed), ...]` --
    the SAME shape `sentence_contradicts`/`clause_filter_sentence` already expect, in the SAME provenance role
    `STRING_FACTS[topic]` played before, but now sourced from `two_organ_combine`/`three_organ_combine`'s own
    ignition-committed patient. `info` is the raw per-relation combine() dict (for the trace).

    `lesion_coupling=True` -- THIS module's own wiring-level lesion, distinct from the organs' internal biology
    levers passed through below: the consensus is never even called, so the generation-time step has nothing to
    suppress against (facts=[]) -- the direct analogue of severing the mouth's coupling to the brain's own
    consensus veto."""
    if lesion_coupling:
        return [], {}
    from webapp.gnw_two_organ_bus import two_organ_combine
    from webapp.gnw_three_organ_bus import three_organ_combine
    combine = three_organ_combine if bus == "three" else two_organ_combine
    kwargs = dict(seed=seed, organb_ltm_exempt=True, organb_lesion=organb_lesion, ws_lesion=ws_lesion)
    if bus == "three":
        kwargs["organc_lesion"] = organc_lesion
    facts, info = [], {}
    for rel in RELATIONS:
        r = combine(chat, topic, rel, **kwargs)
        info[rel] = r
        if r.get("committed") is not None:
            facts.append((rel, str(r["committed"])))
    return facts, info


# =====================================================================================================
# (3) SENTENCE-BY-SENTENCE generation: continue from the ACCEPTED context, one candidate sentence at a time.
# TWO continuation techniques, selectable via `continuation=` on `generate_with_generation_time_veto`:
#   "text"     -- v1 (2026-08-27): decode accepted ids -> string, re-tokenize prompt+string every step. Kept
#                 verbatim below for A/B comparison; this is the technique the PARTIAL finding's "NEXT" named.
#   "token_id" -- v2 (2026-08-28, NEW DEFAULT): accepted context stays TOKEN IDS; a kept sentence's own
#                 generated ids are appended directly (zero retokenization); only a repaired sentence re-encodes
#                 its own (edited) span. See the module docstring for the full rationale.
# =====================================================================================================
def _continue_chunk(gen: OpenEndedGenerator, system: str, user: str, accepted_text: str, seed: int,
                    budget_tokens: int):
    """v1 TEXT continuation (kept for A/B comparison -- see `continuation="text"`). Generate up to
    `budget_tokens` MORE tokens, continuing the assistant turn from `accepted_text` (what has already been
    committed) by DECODING it to a string and RE-TOKENIZING `prompt + accepted_text` from scratch every step.
    Greedy + re-seeded exactly like `OpenEndedGenerator.generate` -- deterministic given (system, user,
    accepted_text, seed), but NOT byte-identical to a continuous decode (the retokenization confound the
    token-id path below removes). Returns (new_text_chunk, eos_reached)."""
    torch = gen.fac._torch
    B1 = gen.fac._B1
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = gen.fac.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_prompt = prompt + (accepted_text + " " if accepted_text else "")
    ids = gen.fac.tok(full_prompt, return_tensors="pt").to(gen.fac.device)
    torch.manual_seed(int(seed))
    if B1.SPK.gen is not None:
        B1.SPK.gen.manual_seed(1000 + int(seed))
    with torch.no_grad():
        out = gen.fac.model.generate(**ids, max_new_tokens=int(budget_tokens), do_sample=False,
                                     pad_token_id=gen.fac.tok.eos_token_id)
    new = out[0, ids.input_ids.shape[1]:]
    eos_id = gen.fac.tok.eos_token_id
    eos_reached = bool(eos_id is not None and (new == eos_id).any().item())
    txt = gen.fac.tok.decode(new, skip_special_tokens=True)
    return txt, eos_reached


def _generate_text_continuation(gen: OpenEndedGenerator, topic: str, seed: int, system: str, user: str,
                                facts, max_new_tokens: int, sentence_budget: int, max_sentences: int):
    """v1 TEXT-continuation sentence loop (unchanged behavior from the 2026-08-27 PARTIAL finding). Factored out
    of `generate_with_generation_time_veto` verbatim so `continuation="text"` reproduces it exactly, byte-for-
    byte, for A/B comparison against the new token-id path."""
    accepted = ""
    trace = []
    tokens_used = 0
    for _ in range(max_sentences):
        remaining = max_new_tokens - tokens_used
        if remaining <= 0:
            break
        budget = min(sentence_budget, remaining)
        chunk, eos = _continue_chunk(gen, system, user, accepted, seed, budget)
        tokens_used += budget
        if not chunk.strip():
            break
        m = _SENT_END_RE.search(chunk)
        incomplete = m is None
        candidate = (chunk[: m.end()] if m else chunk).strip()
        if not candidate:
            break
        repaired = clause_filter_sentence(candidate, topic, facts)
        if repaired is None:
            trace.append({"raw": candidate, "kept": None, "action": "dropped_stop", "consensus_facts": facts,
                          "continuation": "text"})
            break                                   # conservative truncation -- v1 scope, see module docstring
        action = "kept" if repaired.strip() == candidate.strip() else "repaired"
        accepted = (accepted + " " + repaired).strip()
        trace.append({"raw": candidate, "kept": repaired, "action": action, "consensus_facts": facts,
                      "continuation": "text"})
        if eos or incomplete:
            break
    return accepted.strip(), trace


def _find_sentence_boundary_ids(tok, ids_1d) -> tuple[int, bool]:
    """Find the smallest token-count `k` such that decoding `ids_1d[:k]` ends (after right-stripping
    whitespace) on a sentence terminator (`.`/`!`/`?`, one or more) -- i.e. the token that completes the FIRST
    sentence in this freshly-generated chunk. Works purely by incremental DECODE (never re-encodes), so the
    returned `k` slices the model's own generated ids exactly, with nothing lost or altered. Returns
    `(k, incomplete)`; `incomplete=True` (k = len(ids_1d)) if no terminator is found within the chunk (mirrors
    the v1 text path's `m is None` case)."""
    n = int(ids_1d.shape[-1])
    for k in range(1, n + 1):
        text = tok.decode(ids_1d[:k], skip_special_tokens=True)
        stripped = text.rstrip()
        if not stripped:
            continue
        m = _SENT_END_RE.search(stripped)
        if m is not None and m.end() == len(stripped):
            return k, False
    return n, True


def _continue_chunk_ids(gen: OpenEndedGenerator, prompt_ids, accepted_ids, seed: int, budget_tokens: int):
    """v2 TOKEN-ID continuation. Generate up to `budget_tokens` MORE tokens continuing directly from
    `cat(prompt_ids, accepted_ids)` -- IDS concatenation, never a decode-then-re-encode string roundtrip. Greedy
    + re-seeded exactly like `OpenEndedGenerator.generate` / the v1 text path. Returns
    `(new_ids: LongTensor[1, k], eos_reached)`."""
    torch = gen.fac._torch
    B1 = gen.fac._B1
    if accepted_ids is not None and accepted_ids.numel() > 0:
        full_ids = torch.cat([prompt_ids, accepted_ids], dim=1)
    else:
        full_ids = prompt_ids
    attn = torch.ones_like(full_ids)
    torch.manual_seed(int(seed))
    if B1.SPK.gen is not None:
        B1.SPK.gen.manual_seed(1000 + int(seed))
    with torch.no_grad():
        out = gen.fac.model.generate(input_ids=full_ids, attention_mask=attn, max_new_tokens=int(budget_tokens),
                                     do_sample=False, pad_token_id=gen.fac.tok.eos_token_id)
    new_ids = out[:, full_ids.shape[1]:]
    eos_id = gen.fac.tok.eos_token_id
    eos_reached = bool(eos_id is not None and (new_ids == eos_id).any().item())
    return new_ids, eos_reached


def _generate_tokenid_continuation(gen: OpenEndedGenerator, topic: str, seed: int, system: str, user: str,
                                   facts, max_new_tokens: int, sentence_budget: int, max_sentences: int):
    """v2 TOKEN-ID-continuation sentence loop (2026-08-28, NEW DEFAULT). Same suppress/repair/stop contract as
    `_generate_text_continuation` (same `clause_filter_sentence` call, same conservative-stop-on-drop v1 scope),
    but the context that carries between sentences is TOKEN IDS, not a decoded string: a KEPT sentence's own
    generated ids are appended to `accepted_ids` directly (`torch.cat`, no decode/re-encode at all); a REPAIRED
    sentence re-encodes ONLY its own (edited) text span (unavoidable -- the text itself changed) and appends
    THAT. This removes the every-step retokenization confound for the common (kept) case."""
    torch = gen.fac._torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = gen.fac.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = gen.fac.tok(prompt, return_tensors="pt").to(gen.fac.device).input_ids
    accepted_ids = torch.zeros((1, 0), dtype=prompt_ids.dtype, device=prompt_ids.device)
    trace = []
    tokens_used = 0
    for _ in range(max_sentences):
        remaining = max_new_tokens - tokens_used
        if remaining <= 0:
            break
        budget = min(sentence_budget, remaining)
        new_ids, eos = _continue_chunk_ids(gen, prompt_ids, accepted_ids, seed, budget)
        tokens_used += budget
        if new_ids.shape[1] == 0:
            break
        k, incomplete = _find_sentence_boundary_ids(gen.fac.tok, new_ids[0])
        candidate_ids = new_ids[:, :k]
        candidate = gen.fac.tok.decode(candidate_ids[0], skip_special_tokens=True).strip()
        if not candidate:
            break
        repaired = clause_filter_sentence(candidate, topic, facts)
        if repaired is None:
            trace.append({"raw": candidate, "kept": None, "action": "dropped_stop", "consensus_facts": facts,
                          "continuation": "token_id"})
            break                                   # conservative truncation -- v1 scope, see module docstring
        action = "kept" if repaired.strip() == candidate.strip() else "repaired"
        if action == "kept":
            accepted_ids = torch.cat([accepted_ids, candidate_ids], dim=1)    # the model's OWN ids, unaltered
        else:
            lead = " " if accepted_ids.numel() > 0 else ""
            repair_ids = gen.fac.tok(lead + repaired, add_special_tokens=False,
                                     return_tensors="pt").to(gen.fac.device).input_ids
            accepted_ids = torch.cat([accepted_ids, repair_ids], dim=1)       # the ONLY re-encode: an edit
        trace.append({"raw": candidate, "kept": repaired, "action": action, "consensus_facts": facts,
                      "continuation": "token_id"})
        if eos or incomplete:
            break
    accepted_text = (gen.fac.tok.decode(accepted_ids[0], skip_special_tokens=True).strip()
                     if accepted_ids.numel() else "")
    return accepted_text, trace


def generate_with_generation_time_veto(gen: OpenEndedGenerator, chat, topic: str, seed: int, system: str,
                                       user: str, *, max_new_tokens: int = 160, sentence_budget: int = 64,
                                       max_sentences: int = 6, lesion_coupling: bool = False, bus: str = "three",
                                       continuation: str = "token_id"):
    """Generate the reply ONE SENTENCE AT A TIME. Before each candidate sentence is fixed into the ACCEPTED
    context (what every later sentence is generated from), it is run through the imported `clause_filter_
    sentence` against the LIVE, per-topic `consensus_facts_for_topic` verdict (not a static dict). A sentence
    that cannot be safely repaired conservatively STOPS generation (v1 scope: truncate, never skip-and-continue
    past an unverifiable point -- see the module docstring). `continuation` selects the stepping technique:
    `"token_id"` (default, 2026-08-28) carries context as token ids (no retokenization on a kept sentence);
    `"text"` reproduces the original 2026-08-27 decode-and-re-encode path verbatim, for A/B comparison. Returns
    `(accepted_text, trace, consensus_info)`."""
    facts, consensus_info = consensus_facts_for_topic(chat, topic, seed, lesion_coupling=lesion_coupling, bus=bus)
    if continuation == "text":
        accepted, trace = _generate_text_continuation(gen, topic, seed, system, user, facts, max_new_tokens,
                                                       sentence_budget, max_sentences)
    elif continuation == "token_id":
        accepted, trace = _generate_tokenid_continuation(gen, topic, seed, system, user, facts, max_new_tokens,
                                                          sentence_budget, max_sentences)
    else:
        raise ValueError(f"unknown continuation technique: {continuation!r}")
    return accepted, trace, consensus_info


# =====================================================================================================
# BATTERY: vary vs coupling-lesion, and no-regression against the EXISTING string post-filter safety net.
# =====================================================================================================
def _wrong_present(text: str, topic: str) -> set:
    low = text.lower()
    return {m for m in MUST_DROP[topic] if m in low}


def _string_post_filter(text: str, topic: str) -> str:
    """The EXISTING, UNCHANGED string moat (`sentence_contradicts` + `clause_filter_sentence` over the STATIC
    `STRING_FACTS` table) -- the safety net this file must never be less safe than. Mirrors
    `webapp.open_ended_chat.post_filter`'s own known-topic loop exactly (persona-strip is out of scope here --
    these probes never contain a persona leak)."""
    facts = STRING_FACTS[topic]
    kept = [k for s in _sentences(text) for k in [clause_filter_sentence(s, topic, facts)] if k]
    return " ".join(kept).strip()


# =====================================================================================================
# (4) CONTROLLED unit-level battery: the EXACT MUST_DROP adversarial clauses, decoupled from whatever a live
# (and, as measured below, non-byte-identical-to-one-shot) Qwen continuation happens to say. This is the
# DIRECT, deterministic, repeatable demonstration that a candidate carrying the store's own named wrong-detail
# battery is suppressed when the LIVE consensus supplies `facts`, and survives when the coupling is severed --
# independent of live generation's own token-level variability. `clause_filter_sentence` is reused UNCHANGED;
# only the PROVENANCE of `facts` (consensus vs none) is under test, exactly as in the live path above.
# =====================================================================================================
ADVERSARIAL_SENTENCES = {
    "canada":  "Canada is bordered by the United States to the south and Mexico to the west.",
    "france":  "France is bordered by Spain to the south and Italy to the east.",
    "morocco": "Morocco is bordered by Spain to the north and Algeria to the east.",
}
ADVERSARIAL_MUST_KEEP = {"canada": "united states", "france": "spain", "morocco": "spain"}
ADVERSARIAL_MUST_DROP = {"canada": "mexico", "france": "italy", "morocco": "algeria"}


def run_controlled_unit_battery(chat, seed: int, bus: str):
    """For each topic's fixed adversarial sentence (a correct border + the MUST_DROP wrong border, the SAME
    coordinated-list shape `_open_ended_clause_contradiction_filter_verify.py`'s own SAME_SENTENCE_ITEMS use):
    run `clause_filter_sentence` with `facts` sourced from the LIVE LTM-exempt consensus (ON) vs the severed
    coupling (LESIONED). Deterministic (no generation, no seed-dependent decoding) -- reproducible on every run
    at any seed the consensus chat is built with."""
    rows = []
    for topic, sentence in ADVERSARIAL_SENTENCES.items():
        facts_on, info_on = consensus_facts_for_topic(chat, topic, seed, lesion_coupling=False, bus=bus)
        facts_les, info_les = consensus_facts_for_topic(chat, topic, seed, lesion_coupling=True, bus=bus)
        kept_on = clause_filter_sentence(sentence, topic, facts_on)
        kept_les = clause_filter_sentence(sentence, topic, facts_les)
        must_keep, must_drop = ADVERSARIAL_MUST_KEEP[topic], ADVERSARIAL_MUST_DROP[topic]
        on_low = (kept_on or "").lower()
        les_low = (kept_les or "").lower()
        rows.append({
            "topic": topic, "seed": seed, "sentence": sentence, "must_keep": must_keep, "must_drop": must_drop,
            "consensus_facts_ON": facts_on, "consensus_facts_LESIONED": facts_les,
            "kept_ON": kept_on, "kept_LESIONED": kept_les,
            "ON_drops_wrong": must_drop not in on_low, "ON_keeps_correct": must_keep in on_low,
            "LESIONED_keeps_wrong": must_drop in les_low,
            "consensus_committed_borders_ON": info_on.get("borders", {}).get("committed"),
        })
    return rows


def run_battery(seed: int, T: int, max_new_tokens: int, sentence_budget: int, max_sentences: int, bus: str,
                device: str, gen: OpenEndedGenerator | None = None):
    """`gen=None` builds a fresh off-bridge Qwen (the original single-seed entry point, unchanged). A caller
    doing a MULTI-SEED sweep (see `main`'s `--seeds`) passes an already-built `gen` so the (expensive: model
    load + calibration pass) Qwen faculty is loaded ONCE and reused across seeds -- generation is re-seeded
    per-call regardless (`torch.manual_seed(seed)` inside `_continue_chunk_ids`/`_continue_chunk`), so reusing
    `gen` does not confound the per-seed comparison. `chat` (the lightweight numpy consensus organs) is always
    rebuilt fresh per seed -- it IS the thing whose seed-dependence this battery is measuring."""
    print("[gt-veto] building the lightweight consensus chat (numpy, no GPU) ...", flush=True)
    chat = build_consensus_chat(seed)
    if gen is None:
        print("[gt-veto] building the off-bridge spiking Qwen (calibration pass) ...", flush=True)
        gen = OpenEndedGenerator(T=T, max_new_tokens=max_new_tokens, seed=seed, device=device)
        print(f"[gt-veto] Qwen ready (load {gen.fac.load_seconds}s)", flush=True)

    rows = []
    for topic in ("canada", "france", "morocco"):
        # the SAME facts (+isa) the saved run injected -- reproduces its prompt so raw generation is a genuine,
        # live, deterministic cross-check against the saved artifact (not required to match; reported either way).
        facts_for_state = [(topic, "isa", "country")] + [(topic, rel, obj) for (rel, obj) in STRING_FACTS[topic]
                                                          if rel != "isa"]
        st = StateContext(topic=f"what do you know about {topic}?", facts=facts_for_state, valence=0.1,
                          arousal=0.4, familiarity=0.9, confidence=0.9, novelty=0.1, curiosity=0.53)
        system, user = build_prompt(st)

        t0 = time.time()
        raw_one_shot, _secs = gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)
        wrong_raw = _wrong_present(raw_one_shot, topic)

        # Run BOTH continuation techniques (token_id = new default, text = the original 2026-08-27 path) so
        # this run reports a DIRECT, same-seed, same-prompt A/B of whether removing the retokenization confound
        # actually raises the live-mouth vary/lesion divergence rate -- not asserted, measured.
        per_technique = {}
        for tech in ("token_id", "text"):
            gt_on_text, gt_on_trace, info_on = generate_with_generation_time_veto(
                gen, chat, topic, seed, system, user, max_new_tokens=max_new_tokens,
                sentence_budget=sentence_budget, max_sentences=max_sentences, lesion_coupling=False, bus=bus,
                continuation=tech)
            gt_les_text, gt_les_trace, info_les = generate_with_generation_time_veto(
                gen, chat, topic, seed, system, user, max_new_tokens=max_new_tokens,
                sentence_budget=sentence_budget, max_sentences=max_sentences, lesion_coupling=True, bus=bus,
                continuation=tech)

            final_on = _string_post_filter(gt_on_text, topic) if gt_on_text else ""
            final_les = _string_post_filter(gt_les_text, topic) if gt_les_text else ""
            wrong_final_on = _wrong_present(final_on, topic)
            wrong_final_les = _wrong_present(final_les, topic)

            # GENERAL (topic-agnostic), UNAMBIGUOUS live-mouth signal: does the coupling change what the mouth
            # itself emitted, comparing the two PRE-safety-net texts directly (NOT routed through `_sentences()`
            # + `" ".join`, which drops sentence-ending punctuation on rejoin regardless of whether anything was
            # actually flagged -- an unrelated, pre-existing property of the safety net's own known-topic loop
            # that would otherwise read as a false "something was caught" on every multi-sentence reply). A
            # topic where the mouth's own decode produced the SAME text either way had nothing for the coupling
            # to have a chance to suppress this run -- UNDEFINED for the live vary/lesion check, not a pass.
            live_diverged = bool(gt_on_text != gt_les_text)

            per_technique[tech] = {
                "gen_time_veto_ON": {"text": gt_on_text, "trace": gt_on_trace, "chars": len(gt_on_text)},
                "gen_time_veto_LESIONED": {"text": gt_les_text, "trace": gt_les_trace, "chars": len(gt_les_text)},
                "final_after_string_safety_net": {
                    "ON_then_filtered": final_on, "LESIONED_then_filtered": final_les,
                    "wrong_present_ON": sorted(wrong_final_on), "wrong_present_LESIONED": sorted(wrong_final_les),
                },
                "consensus_info_ON": info_on,
                "live_mouth_output_diverged_on_vs_lesioned": live_diverged,
                "safety_net_never_leaks_ON": bool(len(wrong_final_on) == 0),
                "safety_net_never_leaks_LESIONED": bool(len(wrong_final_les) == 0),
            }
            print(f"[gt-veto] {topic} [{tech}]: gt_ON  ({len(gt_on_text)}c)={gt_on_text!r}", flush=True)
            print(f"[gt-veto] {topic} [{tech}]: gt_LES ({len(gt_les_text)}c)={gt_les_text!r}", flush=True)
            print(f"[gt-veto] {topic} [{tech}]: live_diverged={live_diverged} "
                  f"final_ON_wrong={sorted(wrong_final_on)} final_LESIONED_wrong={sorted(wrong_final_les)}",
                  flush=True)

        rows.append({
            "topic": topic, "seed": seed, "facts_injected": facts_for_state,
            "must_drop": sorted(MUST_DROP[topic]),
            "raw_one_shot": raw_one_shot, "wrong_in_raw_one_shot": sorted(wrong_raw),
            # top-level keys mirror the (now default) token_id technique for backward-compatible readers of
            # this artifact's prior shape; `by_continuation` carries the full token_id vs text A/B.
            **{k: v for k, v in per_technique["token_id"].items()},
            "by_continuation": per_technique,
        })

    unit_rows = run_controlled_unit_battery(chat, seed, bus)
    for r in unit_rows:
        print(f"[gt-veto] UNIT {r['topic']}: ON->{r['kept_ON']!r} LESIONED->{r['kept_LESIONED']!r} "
              f"drops_wrong={r['ON_drops_wrong']} keeps_correct={r['ON_keeps_correct']} "
              f"lesioned_keeps_wrong={r['LESIONED_keeps_wrong']}", flush=True)

    return rows, unit_rows, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seed list (e.g. 42,43,44,100,101,102) -- overrides --seed; the "
                         "off-bridge Qwen faculty is loaded ONCE and reused across seeds (generation is "
                         "re-seeded per-call regardless), so a 6-seed sweep costs one model load, not six.")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--sentence-budget", type=int, default=64)
    ap.add_argument("--max-sentences", type=int, default=6)
    ap.add_argument("--bus", default="three", choices=["two", "three"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    rows, unit_rows = [], []
    gen = None
    for i, sd in enumerate(seeds):
        print(f"[gt-veto] === seed {sd} ({i + 1}/{len(seeds)}) ===", flush=True)
        r, u, gen = run_battery(sd, args.T, args.max_new_tokens, args.sentence_budget, args.max_sentences,
                                args.bus, args.device, gen=gen)
        rows.extend(r)
        unit_rows.extend(u)

    n = len(rows)
    n_safety_on_ok = sum(r["safety_net_never_leaks_ON"] for r in rows)
    n_safety_les_ok = sum(r["safety_net_never_leaks_LESIONED"] for r in rows)

    # PRIMARY (decisive) load-bearing evidence: the CONTROLLED unit battery -- deterministic, all 3 topics, the
    # store's own exact MUST_DROP adversarial clause, decoupled from the live mouth's own token-level variability.
    n_unit = len(unit_rows)
    n_unit_drops_wrong = sum(r["ON_drops_wrong"] for r in unit_rows)
    n_unit_keeps_correct = sum(r["ON_keeps_correct"] for r in unit_rows)
    n_unit_lesioned_keeps_wrong = sum(r["LESIONED_keeps_wrong"] for r in unit_rows)

    # SECONDARY (opportunistic) confirmation AT the real, live off-bridge Qwen mouth: honest scope limit (named
    # in the module docstring) -- the sentence-by-sentence continuation decode is NOT necessarily byte-identical
    # to one-shot, so a given run's ON and LESIONED decodes may or may not diverge at all. Only topics where the
    # mouth's own OWN output actually differed (`live_mouth_output_diverged_on_vs_lesioned`) are a genuine test
    # of the vary/lesion property; the rest are UNDEFINED for this check, not a pass -- reported honestly rather
    # than forced into a 3/3 the live decode did not actually exercise. `n_live_diverged` (top-level) is the
    # NEW DEFAULT token_id technique; `n_live_diverged_text` is the same measurement for the original v1 text
    # technique, computed in the SAME run at the SAME seed/prompts -- a direct, disclosed A/B of whether the
    # 2026-08-28 token-id-continuation change actually raises the divergence rate over the 2026-08-27 baseline.
    n_live_diverged = sum(r["live_mouth_output_diverged_on_vs_lesioned"] for r in rows)
    n_live_diverged_text = sum(r["by_continuation"]["text"]["live_mouth_output_diverged_on_vs_lesioned"]
                               for r in rows)
    n_safety_on_ok_text = sum(r["by_continuation"]["text"]["safety_net_never_leaks_ON"] for r in rows)
    n_safety_les_ok_text = sum(r["by_continuation"]["text"]["safety_net_never_leaks_LESIONED"] for r in rows)

    v = Verdict("generation-time LTM-exempt organ-B/C consensus veto suppresses a known-supplement clause AT "
               "generation (controlled unit battery, decisive) and vanishes when the coupling is lesioned; "
               "string safety net unaffected; live-mouth confirmation reported honestly (opportunistic); "
               "token-id continuation (2026-08-28) vs the original text continuation (2026-08-27) A/B'd "
               "same-run")
    v.require("(PRIMARY, controlled) coupling ON: clause_filter_sentence drops the store-wrong border on the "
              "fixed adversarial sentence, every topic", n_unit_drops_wrong, expect=lambda x: x == n_unit)
    v.require("(PRIMARY, controlled) coupling ON: the store-correct border is KEPT, every topic",
              n_unit_keeps_correct, expect=lambda x: x == n_unit)
    v.require("(PRIMARY, controlled) coupling LESIONED: the store-wrong border REAPPEARS (kept unchanged), "
              "every topic", n_unit_lesioned_keeps_wrong, expect=lambda x: x == n_unit)
    v.require("(no-regression) string safety net still catches everything on the LIVE end-to-end output, "
              "coupling ON", n_safety_on_ok, expect=lambda x: x == n)
    v.require("(no-regression) string safety net still catches everything on the LIVE end-to-end output, "
              "coupling LESIONED (never less safe than before this file existed)", n_safety_les_ok,
              expect=lambda x: x == n)
    v.disabled("a live-mouth vary/lesion demonstration on EVERY probed topic",
               why=f"even with token-id continuation removing the retokenization confound for kept sentences, "
                   f"the mouth's own greedy decode may still legitimately produce the SAME text ON vs LESIONED "
                   f"when the coupling has nothing to suppress this run -- an honest scope limit, not a bug. "
                   f"token_id technique: ON/LESIONED actually diverged on {n_live_diverged}/{n} topics this "
                   f"run; text technique (the original 2026-08-27 path, same seed/prompts): diverged on "
                   f"{n_live_diverged_text}/{n}. The PRIMARY, decisive evidence remains the controlled unit "
                   f"battery above, which is deterministic and topic-complete -- see `by_continuation` on each "
                   f"row for the full token_id-vs-text comparison.")
    go = (n_unit_drops_wrong == n_unit and n_unit_keeps_correct == n_unit
         and n_unit_lesioned_keeps_wrong == n_unit and n_safety_on_ok == n and n_safety_les_ok == n)
    decided = v.decide(go=go)

    art = {
        "probe": "open_ended_gen_time_consensus_veto_derisk", "backend": "numpy(chat)+cuda(qwen)",
        "seed": args.seed, "seeds": seeds, "T": args.T, "max_new_tokens": args.max_new_tokens,
        "sentence_budget": args.sentence_budget, "max_sentences": args.max_sentences, "bus": args.bus,
        "rows": rows, "unit_rows": unit_rows, "n_topics": n,
        "n_unit_drops_wrong": n_unit_drops_wrong, "n_unit_keeps_correct": n_unit_keeps_correct,
        "n_unit_lesioned_keeps_wrong": n_unit_lesioned_keeps_wrong, "n_unit": n_unit,
        "n_safety_net_ok_ON": n_safety_on_ok, "n_safety_net_ok_LESIONED": n_safety_les_ok,
        "n_live_diverged": n_live_diverged,
        "continuation_default": "token_id",
        "n_live_diverged_text": n_live_diverged_text,
        "n_safety_net_ok_ON_text": n_safety_on_ok_text, "n_safety_net_ok_LESIONED_text": n_safety_les_ok_text,
        "verdict": decided, "preconditions": decided.get("preconditions", []), "GO": bool(go),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, ensure_ascii=False)
    print(f"\n[gt-veto] OVERALL {'GO' if go else 'NO-GO/PARTIAL'} "
          f"(unit: drops_wrong {n_unit_drops_wrong}/{n_unit} | keeps_correct {n_unit_keeps_correct}/{n_unit} | "
          f"lesioned_keeps_wrong {n_unit_lesioned_keeps_wrong}/{n_unit} || live[token_id]: safety-ON "
          f"{n_safety_on_ok}/{n} | safety-LESIONED {n_safety_les_ok}/{n} | mouth_diverged {n_live_diverged}/{n} "
          f"|| live[text]: mouth_diverged {n_live_diverged_text}/{n})", flush=True)
    print(f"[gt-veto] wrote {os.path.relpath(args.out, _REPO)}", flush=True)
    return decided["status"]


if __name__ == "__main__":
    main()
