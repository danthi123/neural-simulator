"""VERIFY: common entity ALIASES resolve to their canonical token so alias-form natural questions reach recall
-- the residual `2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md` names in its own
"Next rungs": "The entity side still leans on the naive underscore-join because the shipped bundle ships ZERO
alias_of facts (a separate residual: entity aliasing)."

VERIFY-FIRST FINDING (do not re-derive): the alias MECHANISM already exists in full --
  - build time: `research/runners/_knowledge_core_curate.py`'s `build_alias_facts` (+ `_all_other_aliases` +
    `_alias_quality`) already emits `{agent: alias_token, action: "alias_of", patient: canonical_token}` facts
    from the KB's own raw Wikidata alias lists (host code building the alias table, per this arc's brief).
  - query time: `research/runners/brain_chat_tui.py`'s `_alias_hop`/`_ground_content_words` already resolve a
    surface-form span to its canonical token via ONE MORE genuinely-spiking `composer.query_patient(candidate,
    "alias_of")` hop -- already wired EAGERLY (`min_span=1`) into `_kb_relation_question_route`,
    `_relation_fronted_route`, AND `_definitional_copula_route`'s own entity capture, not merely a last-resort
    fallback.
  - an alias-EXTENDED bundle was already BUILT at full production scale (2026-08-26, see
    `2026-08-26-knowledge-grounding-natural-language.md`): `~/Projects/sim-data/knowledge_bundles/
    wikidata_core_15k_grounded_v1`, SAME curation parameters as the shipped `wikidata_core_15k`
    (8000 entities/40 relations/15000 facts/seed 42), PLUS 30,804 `alias_of` facts (44 ambiguous aliases
    correctly DROPPED), verified 60/60-correct after a save/reload round-trip.

What was NOT yet done (this script's job): that bundle's alias facts had never been exercised with an actual
BATTERY of alias-FORM natural-language QUESTIONS through the live `_kb_relation_question_route`/
`_relation_fronted_route`/`_definitional_copula_route` machinery (the 2026-08-26 finding's own alias-grounding
turns were 2 hand-picked examples at SMOKE scale, built before `_kb_relation_question_route` existed at all --
2026-09-01, THIS SAME DAY). This closes that verification gap. NO new alias-generation mechanism, NO `sim/`
change, NO production-default flip (`_default_ltm_bundle_dir()` in webapp/server.py is untouched -- the
alias-extended bundle is loaded here via an explicit path, exactly as the 2026-08-26 finding's own verify did;
swapping the shipped default remains the owner's call, unchanged).

MOCK-SELF ROUTING (per GAP_CLOSURE_MISSION.md's 2026-09-01 OPS LESSON, commit e4680f4e3: "verify deterministic
ROUTING via the unbound route method + a mock `self` (pure parsing, no brain build, seconds) and confirm RECALL
separately... instead of a full-brain rebuild per case"): Part A calls `ChatBrain._kb_relation_question_route`/
`_relation_fronted_route`/`_definitional_copula_route` UNBOUND against a trivial mock `self` whose
`.inner.composer` is the REAL (loaded-once, never rebuilt) `ShardedPhasorStore` -- so `query_patient`/the
alias-hop is the genuine substrate primitive, not a host dict, while routing/segmentation is exercised without
ever constructing a full ChatBrain. Part B builds the REAL production `ChatBrain` (onebrain composer + the
alias-extended LTM via `TieredFactStore`, mirroring `_nl_parser_real_kb_relations/verify_kb_relations.py`
exactly) exactly ONCE per arm (INTACT, LESIONED) -- not per test case -- confirming `gate()`'s full pipeline
recall end-to-end for a representative subset, at seed 42.

Usage:
  SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python \\
      research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.py \\
      --out research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

GROUNDED_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k_grounded_v1"
SEED = 42

# Idiom-first / generic-fallback natural-question TEMPLATE per KB relation -- mirrors
# `_nl_parser_real_kb_relations/verify_kb_relations.py`'s own `_QUESTION_TEMPLATES` (same relation set, same
# shape choices) so this arc's battery is drawn from the SAME already-validated question surface, only with an
# ALIAS entity phrase substituted for the canonical one.
_QUESTION_TEMPLATES = {
    "located_in_time_zone": "what is {entity}'s located in time zone?",
    "located_in_the_administrative_territoria": "what is {entity}'s administrative territorial entity?",
    "subclass_of": "what is {entity} a subclass of?",
    "headquarters_location": "where is {entity} headquartered?",
    "shares_border_with": "what borders {entity}?",
    "language_of_work_or_name": "what is the language of work or name of {entity}?",
    "member_of": "what is {entity} a member of?",
    "part_of": "what is {entity} part of?",
    "taxon_rank": "what is {entity}'s taxon rank?",
    "country_of_citizenship": "what is {entity}'s nationality?",
    "followed_by": "what is {entity} followed by?",
    "contains_administrative_territorial_enti": "what is {entity}'s administrative territorial entities it contains?",
    "country_of_origin": "what is {entity}'s country of origin?",
    "languages_spoken_written_or_signed": "what languages does {entity} speak?",
    "award_received": "what award did {entity} receive?",
    "participant_of": "what did {entity} participate in?",
    "given_name": "what is {entity}'s given name?",
    "place_of_birth": "where was {entity} born?",
    "place_of_death": "where did {entity} die?",
    "educated_at": "where was {entity} educated?",
    "record_label": "what record label is {entity} signed to?",
    "work_location": "where does {entity} work?",
    "original_language_of_film_or_tv_show": "what is the original language of film or tv show of {entity}?",
    "member_of_political_party": "what political party is {entity} a member of?",
    "position_held": "what is {entity}'s position held?",
    "parent_taxon": "what is {entity}'s parent taxon?",
    "family_name": "what is {entity}'s family name?",
    "employer": "who does {entity} work for?",
    "occupation": "what is {entity}'s occupation?",
}

# A few single-bare-word relations (NOT in _KB_UNDERSCORED_RELATIONS -- already reachable via
# `_relation_fronted_route`'s "what <relation> is <entity>?" shape) used to confirm alias-hop entity resolution
# ALSO drives that pre-existing (2026-08-27) route, not just the brand-new (2026-09-01) KB-relation route.
_RELATION_FRONTED_TEMPLATES = {
    "country": "what country is {entity} from?",
    # NOT "known for" -- `_relation_fronted_route`'s regex captures entity as EVERYTHING between "is" and an
    # optional single TRAILING PREPOSITION (`_REL_FRONTED_TRAILING_PREPS`); "known" is not a preposition, so it
    # would be swallowed into the entity span. "in" fits the route's own documented grammar exactly.
    "sport": "what sport is {entity} in?",
}

# Adversarial phrases NO alias fact covers -- the moat probe. 'chelsea' is a REAL naturally-ambiguous common
# alias this exact core's own curation correctly DROPS (chelsea_fc / chelsea_kensington / chelsea_middlesex all
# present -> >=2 distinct canonical concepts -> ambiguous -> no alias fact emitted, per `build_alias_facts`'
# own documented ambiguity policy) -- the strongest adversarial case: a common short alias that IS genuinely
# ambiguous in this core, correctly abstaining rather than guessing one of the 3.
_MOAT_PROBES = ["chelsea", "purple elephant bicycle", "zorblexia quintar"]
# NOTE: a longer (5-word) nonsense phrase ("the glorble house of nonexistence") was ALSO tried and is reported
# separately in `single_word_collision_probe.json` -- it triggers `_ground_content_words`'s O(n^2) span sweep
# (many query_patient calls) and is genuinely slow under CPU contention, without adding a distinct failure mode
# from the 3-word case already covered here (both are non-len==1 partial groundings that correctly abstain at
# the ROUTE level, per that probe's finding). Kept out of the main battery for wall-clock cost, not evidentiary
# need.


class _MockRouter:
    def __init__(self, self_aliases):
        self.self_aliases = self_aliases


class _MockInner:
    def __init__(self, composer):
        self.composer = composer


class _MockSelf:
    """The minimal `self` the THREE route methods under test actually read: `self.router.self_aliases` and
    `self.inner.composer` (for the alias-hop). No agent, no stored_facts, no neural parser -- these routes
    never touch them. Passed as the unbound methods' `self` argument directly (duck typing)."""

    def __init__(self, composer, self_aliases):
        self.router = _MockRouter(self_aliases)
        self.inner = _MockInner(composer)


def _load_bundle_maps():
    """(alias->canonical, canonical->[alias,...], [(agent,relation,patient), ...]) straight from the
    alias-extended bundle's own facts.json -- ground truth for every assertion below, not hand-typed examples."""
    facts_path = os.path.join(GROUNDED_BUNDLE, "facts.json")
    with open(facts_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    alias_fwd, alias_rev, plain = {}, {}, []
    for row in raw:
        f = row["fact"]
        if f["action"] == "alias_of":
            alias_fwd[f["agent"]] = f["patient"]
            alias_rev.setdefault(f["patient"], []).append(f["agent"])
        else:
            plain.append((f["agent"], f["action"], f["patient"]))
    return alias_fwd, alias_rev, plain


def _sample_aliased_facts(plain, alias_rev, relations):
    """First real (agent, relation, patient) fact per target relation whose AGENT has >=1 alias_of fact --
    the entity side is genuinely resolvable through the alias-hop, not just present verbatim."""
    samples = {}
    for a, r, p in plain:
        if r in relations and r not in samples and a in alias_rev:
            samples[r] = (a, r, p, alias_rev[a][0])
    return samples


def part_a_mock_routing(store, alias_fwd, alias_rev, plain, log):
    """Pure ROUTING verification: the three entity-alias-hop-using route methods, called UNBOUND against a mock
    `self` whose composer is the real (loaded-once) store. No ChatBrain, no agent, no neural net -- seconds."""
    self_on = _MockSelf(store, {"you", "your", "yours", "i", "me", "my", "it", "its"})

    from research.runners.brain_chat_tui import (
        ChatBrain, _KB_UNDERSCORED_RELATIONS, _ground_content_words, _alias_hop,
    )

    kb_samples = _sample_aliased_facts(plain, alias_rev, set(_KB_UNDERSCORED_RELATIONS))
    log(f"KB-relation alias-covered samples: {len(kb_samples)}/{len(_KB_UNDERSCORED_RELATIONS)}")

    kb_rows = []
    os.environ.pop("BRAIN_KNOWLEDGE_GROUNDING", None)   # default-ON
    for relation, (agent_canon, _r, patient, alias) in kb_samples.items():
        q = _QUESTION_TEMPLATES[relation].format(entity=alias.replace("_", " "))
        route_on = ChatBrain._kb_relation_question_route(self_on, q)
        # LESION: grounding off -> entity stays the naive underscore-join of the ALIAS phrase, which must NOT
        # equal the canonical token (proves the alias-hop, not naive-join, is what makes this resolve).
        os.environ["BRAIN_KNOWLEDGE_GROUNDING"] = "0"
        route_off = ChatBrain._kb_relation_question_route(self_on, q)
        os.environ.pop("BRAIN_KNOWLEDGE_GROUNDING", None)
        resolved = route_on == [agent_canon, relation]
        naive_join = "_".join(alias.replace("_", " ").split())
        load_bearing = (route_off is not None and route_off[0] == naive_join
                         and route_off[0] != agent_canon)
        kb_rows.append({
            "relation": relation, "question": q, "alias": alias, "canonical": agent_canon,
            "route_grounding_on": route_on, "route_grounding_off": route_off,
            "resolved": resolved, "load_bearing": load_bearing,
        })
        log(f"  [kb-relation] {relation}: {q!r} -> ON={route_on!r} OFF={route_off!r} "
            f"{'PASS' if resolved and load_bearing else 'FAIL'}")

    # relation-fronted (pre-existing 2026-08-27 route) + definitional-copula (2026-08-26 route) -- confirm the
    # SAME alias-hop drives these two OLDER routes as well, not just the newest KB-relation table.
    rf_samples = _sample_aliased_facts(plain, alias_rev, set(_RELATION_FRONTED_TEMPLATES))
    rf_rows = []
    for relation, (agent_canon, _r, patient, alias) in rf_samples.items():
        q = _RELATION_FRONTED_TEMPLATES[relation].format(entity=alias.replace("_", " "))
        route_on = ChatBrain._relation_fronted_route(self_on, q)
        resolved = route_on == [agent_canon, relation]
        rf_rows.append({"relation": relation, "question": q, "alias": alias, "canonical": agent_canon,
                         "route": route_on, "resolved": resolved})
        log(f"  [relation-fronted] {relation}: {q!r} -> {route_on!r} {'PASS' if resolved else 'FAIL'}")

    # definitional copula: pick 5 aliased entities (any relation) and ask "what is <alias>?". Skip an alias
    # containing " of " -- `_definitional_copula_route` deliberately excludes that shape as RELATIONAL, not
    # definitional (its own docstring), so such an alias is not a fair copula test case (by design, not a bug).
    copula_rows = []
    seen_canon = set()
    for a, _r, _p in plain:
        if len(copula_rows) >= 5:
            break
        if a in alias_rev and a not in seen_canon:
            alias = next((al for al in alias_rev[a] if " of " not in al.replace("_", " ")), None)
            if alias is None:
                continue
            q = f"what is {alias.replace('_', ' ')}?"
            route_on = ChatBrain._definitional_copula_route(self_on, q)
            resolved = route_on == [a, "isa"]
            copula_rows.append({"question": q, "alias": alias, "canonical": a,
                                 "route": route_on, "resolved": resolved})
            log(f"  [copula] {q!r} -> {route_on!r} {'PASS' if resolved else 'FAIL'}")
            seen_canon.add(a)

    # MOAT: an alias no fact covers must NOT resolve (content unchanged) -- and, for the query-time primitive,
    # must not recall a WRONG patient (the store's own already-established false-hop=0 property).
    moat_rows = []
    for phrase in _MOAT_PROBES:
        toks = phrase.split()
        grounded = _ground_content_words(store, toks, min_span=1)
        unresolved = grounded == toks   # unchanged -> the phrase did not alias-hop to anything
        hop = _alias_hop(store, "_".join(toks))
        moat_rows.append({"phrase": phrase, "grounded": grounded, "alias_hop": hop,
                           "abstains": unresolved and hop is None})
        log(f"  [moat] {phrase!r} -> grounded={grounded!r} alias_hop={hop!r} "
            f"{'PASS(abstains)' if (unresolved and hop is None) else 'FAIL'}")

    return {
        "kb_relation": kb_rows, "relation_fronted": rf_rows, "definitional_copula": copula_rows,
        "moat": moat_rows,
        "n_kb_resolved": sum(1 for r in kb_rows if r["resolved"]),
        "n_kb_load_bearing": sum(1 for r in kb_rows if r["load_bearing"]),
        "n_kb_total": len(kb_rows),
        "n_rf_resolved": sum(1 for r in rf_rows if r["resolved"]), "n_rf_total": len(rf_rows),
        "n_copula_resolved": sum(1 for r in copula_rows if r["resolved"]), "n_copula_total": len(copula_rows),
        "n_moat_abstain": sum(1 for r in moat_rows if r["abstains"]), "n_moat_total": len(moat_rows),
    }


def _build_chat(seed, ltm_bundle):
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(ltm_bundle)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


def part_b_full_production(plain, alias_rev, log, t0, n_relations=10):
    """ONE real ChatBrain build per arm (INTACT, LESIONED) -- not per test case (the 2026-09-01 OOM lesson,
    GAP_CLOSURE_MISSION.md anchor). A representative subset of KB relations, alias-form questions, through the
    REAL `gate()` (/api/brain-chat) call against the alias-extended production-scale bundle."""
    from research.runners.brain_chat_tui import _KB_UNDERSCORED_RELATIONS

    kb_samples = _sample_aliased_facts(plain, alias_rev, set(_KB_UNDERSCORED_RELATIONS))
    subset = dict(list(kb_samples.items())[:n_relations])

    os.environ.pop("BRAIN_KNOWLEDGE_GROUNDING", None)     # default-ON (INTACT arm)
    chat = _build_chat(SEED, GROUNDED_BUNDLE)
    log(f"[{time.time()-t0:.0f}s] seed {SEED}: built onebrain tiny-demo + ALIAS-EXTENDED 15k LTM "
        f"(vocab-with-aliases scale)")

    rows = []
    for relation, (agent_canon, _r, patient, alias) in subset.items():
        q = _QUESTION_TEMPLATES[relation].format(entity=alias.replace("_", " "))
        gated = chat.gate(q)
        recalled_ok = bool(gated) and len(gated) == 3 and gated[0] == agent_canon and gated[1] == relation \
            and gated[2] == patient
        moat_ok = (gated is None) or (gated[2] == patient)
        rows.append({"relation": relation, "question": q, "alias": alias, "canonical": agent_canon,
                      "expected_patient": patient, "gated": gated, "recalled_ok": recalled_ok, "moat_ok": moat_ok})
        log(f"  [full-gate INTACT] {relation}: {q!r} -> {gated!r} {'PASS' if recalled_ok else 'FAIL'}")

    # regression: an ORDINARY already-working canonical-form question (no alias needed) must still work,
    # unaffected by this arc.
    canon_relation, (canon_agent, _r, canon_patient, _al) = next(iter(subset.items()))
    canon_q = _QUESTION_TEMPLATES[canon_relation].format(entity=canon_agent.replace("_", " "))
    canon_gated = chat.gate(canon_q)
    canon_ok = bool(canon_gated) and canon_gated == [canon_agent, canon_relation, canon_patient]
    log(f"  [full-gate INTACT canonical-form regression] {canon_q!r} -> {canon_gated!r} "
        f"{'PASS' if canon_ok else 'FAIL'}")

    # moat via the FULL pipeline: a fabricated alias-form question must abstain, never invent. A SHORT (2-word)
    # nonsense entity is used deliberately -- Part A already establishes the moat holds for LONGER nonsense
    # phrases too (via the raw alias-hop + route-level check, which is far cheaper to run broadly); a short
    # phrase here keeps this specific full-gate() round-trip check fast without weakening what it tests (an
    # unresolvable entity still must not fabricate a fact through the complete production pipeline).
    moat_q = "what is zorblexia quintar's country of origin?"
    moat_gated = chat.gate(moat_q)
    moat_full_ok = moat_gated is None
    log(f"  [full-gate INTACT moat] {moat_q!r} -> {moat_gated!r} {'PASS' if moat_full_ok else 'FAIL'}")

    os.environ["BRAIN_KNOWLEDGE_GROUNDING"] = "0"          # LESIONED arm -- fresh build, same seed/LTM
    chat_lesion = _build_chat(SEED, GROUNDED_BUNDLE)
    log(f"[{time.time()-t0:.0f}s] seed {SEED}: built LESIONED onebrain tiny-demo + ALIAS-EXTENDED 15k LTM")
    for row in rows:
        gated_lesion = chat_lesion.gate(row["question"])
        row["gated_lesion"] = gated_lesion
        row["lesion_abstains"] = gated_lesion is None
        log(f"  [full-gate LESIONED] {row['relation']}: {row['question']!r} -> {gated_lesion!r} "
            f"{'PASS(abstains)' if row['lesion_abstains'] else 'FAIL'}")
    canon_gated_lesion = chat_lesion.gate(canon_q)
    canon_lesion_unaffected = canon_gated_lesion == canon_gated
    log(f"  [full-gate LESIONED canonical-form regression] {canon_q!r} -> {canon_gated_lesion!r} "
        f"{'PASS(unaffected)' if canon_lesion_unaffected else 'FAIL'}")
    os.environ.pop("BRAIN_KNOWLEDGE_GROUNDING", None)

    return {
        "seed": SEED, "ltm_bundle": GROUNDED_BUNDLE, "n_relations": len(rows),
        "n_recalled_ok": sum(1 for r in rows if r["recalled_ok"]),
        "n_moat_ok": sum(1 for r in rows if r["moat_ok"]),
        "n_lesion_abstains": sum(1 for r in rows if r["lesion_abstains"]),
        "canonical_form_regression_intact": canon_ok,
        "canonical_form_regression_lesion_unaffected": canon_lesion_unaffected,
        "fabricated_alias_question_abstains": moat_full_ok,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_entity_aliasing.json"))
    ap.add_argument("--n-full-relations", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

    alias_fwd, alias_rev, plain = _load_bundle_maps()
    log(f"bundle loaded: {len(plain)} plain facts, {len(alias_fwd)} alias_of facts "
        f"({len(alias_rev)} canonical concepts with >=1 alias)")

    from research.runners.sharded_phasor_store import ShardedPhasorStore
    store = ShardedPhasorStore.load(GROUNDED_BUNDLE)
    log("ShardedPhasorStore loaded (ONCE, reused for every mock-routing call below)")

    part_a = part_a_mock_routing(store, alias_fwd, alias_rev, plain, log)
    log(f"PART A (mock-self routing, no brain build) done: "
        f"kb {part_a['n_kb_resolved']}/{part_a['n_kb_total']} resolved, "
        f"{part_a['n_kb_load_bearing']}/{part_a['n_kb_total']} load-bearing; "
        f"relation-fronted {part_a['n_rf_resolved']}/{part_a['n_rf_total']}; "
        f"copula {part_a['n_copula_resolved']}/{part_a['n_copula_total']}; "
        f"moat {part_a['n_moat_abstain']}/{part_a['n_moat_total']} abstain")

    del store   # free before the full-brain build (Part B loads its own copy inside TieredFactStore)

    part_b = part_b_full_production(plain, alias_rev, log, t0, n_relations=args.n_full_relations)
    log(f"PART B (full production gate()) done: "
        f"{part_b['n_recalled_ok']}/{part_b['n_relations']} recalled, "
        f"{part_b['n_moat_ok']}/{part_b['n_relations']} moat-ok, "
        f"{part_b['n_lesion_abstains']}/{part_b['n_relations']} lesion-abstains, "
        f"canonical-regression intact={part_b['canonical_form_regression_intact']} "
        f"lesion-unaffected={part_b['canonical_form_regression_lesion_unaffected']}, "
        f"fabricated-moat-abstains={part_b['fabricated_alias_question_abstains']}")

    all_pass = (
        part_a["n_kb_resolved"] == part_a["n_kb_total"]
        and part_a["n_kb_load_bearing"] == part_a["n_kb_total"]
        and part_a["n_rf_resolved"] == part_a["n_rf_total"]
        and part_a["n_copula_resolved"] == part_a["n_copula_total"]
        and part_a["n_moat_abstain"] == part_a["n_moat_total"]
        and part_b["n_recalled_ok"] == part_b["n_relations"]
        and part_b["n_moat_ok"] == part_b["n_relations"]
        and part_b["n_lesion_abstains"] == part_b["n_relations"]
        and part_b["canonical_form_regression_intact"]
        and part_b["canonical_form_regression_lesion_unaffected"]
        and part_b["fabricated_alias_question_abstains"]
    )

    out = {
        "probe": "entity ALIAS-form question resolution+recall against the alias-extended wikidata_core_15k_grounded_v1 "
                  "bundle (30,804 alias_of facts, already built 2026-08-26) -- closes the board #94 residual named "
                  "2026-09-01 ('the shipped bundle ships ZERO alias_of facts')",
        "seed": SEED, "ltm_bundle_used": GROUNDED_BUNDLE,
        "production_default_flip": "NOT DONE (owner-UX-gated, unchanged) -- webapp/server.py _default_ltm_bundle_dir() "
                                     "still resolves to the un-aliased shipped wikidata_core_15k",
        "part_a_mock_routing": part_a,
        "part_b_full_production_gate": part_b,
        "verdict": "GO" if all_pass else "PARTIAL",
        "all_pass": all_pass,
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"wrote {args.out}  verdict={out['verdict']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
