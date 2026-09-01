"""6-seed verification: the REAL shipped `wikidata_core_15k` KB's underscored multi-word relations are now
QUERYABLE through the live NL question parser (board #94 frontier, named in
2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md).

Mirrors research/findings/raw/_knowledge_chat_veto/repro_veto.py's construction EXACTLY: SIM_BACKEND=numpy,
`_build_tiny_demo(seed, composer_kind="onebrain")` (the TRUE production composer,
webapp.server._COMPOSER_KIND_DEFAULT) + the REAL shipped ShardedPhasorStore LTM attached via TieredFactStore
(the same attach webapp.server._build_chat_brain uses for 'tiny-demo +LTM', the out-of-the-box default brain) +
`ChatBrain.gate()` (the exact call /api/brain-chat makes). No fixture, no toy facts -- every (entity, relation,
patient) tested is sampled directly from the shipped facts.json.

For each of the 29 underscored/idiomatic relations `_kb_relation_question_route` now covers (research/runners/
brain_chat_tui.py), pick the FIRST real fact using that relation from the shipped bundle and pose ONE natural-
English question about it (idiom template where the table has one, else the generic possessive form) -- the
entity phrase is the canonical token's own underscore->space inverse (an exact round-trip through the SAME
naive-join path `_relation_fronted_route`/`_definitional_copula_route` already rely on for an entity with no
alias fact; this bundle currently ships with ZERO alias_of facts -- see the module docstring finding for why).

Per seed, per relation:
  - INTACT (BRAIN_KB_RELATION_QUESTIONS unset -> default-ON): `chat.gate(Q)` must recall exactly the real
    stored patient -- [entity_final, relation, patient] -- not merely "not abstain" (the moat: a recall of the
    WRONG patient is scored as a FAIL, identically to an abstain).
  - LESIONED (BRAIN_KB_RELATION_QUESTIONS=0): the SAME question, freshly re-routed, must ABSTAIN (gate() ->
    None) -- proving the route (not some other pre-existing path) is what makes the question answerable, i.e.
    the coupling this arc adds is load-bearing, not a no-op alongside an already-working route.
  - MOAT: every INTACT recall's patient is checked against the bundle's OWN stored fact for that (entity,
    relation) -- never an invented value.

Byte-identical-off (separate, cheap, single-seed check): with the flag OFF, `_extract_route` on a battery of
ordinary already-working questions (unrelated to this table) returns EXACTLY the same route as with the flag
unset/ON -- the new pass never perturbs anything it does not itself resolve.

Usage:
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_nl_parser_real_kb_relations/verify_kb_relations.py \\
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_nl_parser_real_kb_relations/verify_kb_relations.json
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

LTM_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"

# (relation, natural_question_TEMPLATE with {entity} placeholder) -- one per relation covered by
# research.runners.brain_chat_tui._KB_UNDERSCORED_RELATIONS. Mixes idiom + generic possessive shapes so both
# code paths in _build_kb_relation_patterns get real coverage, not just the generic fallback.
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

# Ordinary, ALREADY-WORKING questions unrelated to the new table -- used for the byte-identical-off sanity
# check (the new pass must never perturb a route it does not itself resolve).
_UNRELATED_QUESTIONS = [
    "what does the brain use?",
    "what is a country?",
    "what country is chelsea fc from?",
]


def _sample_facts():
    """First real (agent, relation, patient) fact per target relation from the shipped bundle's facts.json."""
    facts_path = os.path.join(LTM_BUNDLE, "facts.json")
    with open(facts_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    samples = {}
    for row in raw:
        f = row["fact"]
        r = f["action"]
        if r in _QUESTION_TEMPLATES and r not in samples:
            samples[r] = (f["agent"], f["action"], f["patient"])
    missing = sorted(set(_QUESTION_TEMPLATES) - set(samples))
    return samples, missing


def _make_question(relation, agent_token):
    entity_phrase = agent_token.replace("_", " ")
    return _QUESTION_TEMPLATES[relation].format(entity=entity_phrase)


def _build_chat(seed):
    """Mirrors repro_veto.py's construction exactly: tiny-demo onebrain composer + the REAL shipped LTM
    attached via TieredFactStore (the SAME attach webapp.server._build_chat_brain uses for the out-of-the-box
    'tiny-demo +LTM' default brain)."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(LTM_BUNDLE)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


def run_seed(seed, samples, t0):
    os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)   # default-ON (unset) for the INTACT arm
    chat = _build_chat(seed)
    print(f"[{time.time()-t0:.0f}s] seed {seed}: built onebrain tiny-demo + real 15k LTM", flush=True)

    per_relation = {}
    for relation, (agent_tok, _rel, patient_tok) in samples.items():
        q = _make_question(relation, agent_tok)
        route = chat._extract_route(q)
        gated = chat.gate(q)
        recalled_ok = bool(gated) and len(gated) == 3 and gated[1] == relation and gated[2] == patient_tok
        moat_ok = (gated is None) or (gated[2] == patient_tok)   # never a WRONG (invented) patient
        per_relation[relation] = {
            "question": q, "expected_agent": agent_tok, "expected_patient": patient_tok,
            "route": route, "gated": gated, "recalled_ok": recalled_ok, "moat_ok": moat_ok,
        }
        print(f"  [{relation}] {q!r} -> route={route!r} gate={gated!r} "
              f"{'PASS' if recalled_ok else 'FAIL'}", flush=True)

    # LESIONED arm: the identical questions, freshly re-routed, must ABSTAIN (proves THIS route is load-bearing,
    # not a no-op beside some other already-working path). Fresh ChatBrain (same seed, same LTM) so no state
    # bleeds across arms.
    os.environ["BRAIN_KB_RELATION_QUESTIONS"] = "0"
    chat_lesion = _build_chat(seed)
    print(f"[{time.time()-t0:.0f}s] seed {seed}: built LESIONED onebrain tiny-demo + real 15k LTM", flush=True)
    for relation, (agent_tok, _rel, patient_tok) in samples.items():
        q = per_relation[relation]["question"]
        gated_lesion = chat_lesion.gate(q)
        per_relation[relation]["gated_lesion"] = gated_lesion
        per_relation[relation]["lesion_abstains"] = gated_lesion is None
    os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)

    # BYTE-IDENTICAL-OFF sanity: unrelated, already-working questions route identically flag-on vs flag-off.
    byte_id_rows = []
    for q in _UNRELATED_QUESTIONS:
        os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)
        r_on = chat._extract_route(q)
        os.environ["BRAIN_KB_RELATION_QUESTIONS"] = "0"
        r_off = chat_lesion._extract_route(q)
        os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)
        byte_id_rows.append({"question": q, "route_on": r_on, "route_off": r_off, "same": r_on == r_off})

    n_recall_ok = sum(1 for v in per_relation.values() if v["recalled_ok"])
    n_moat_ok = sum(1 for v in per_relation.values() if v["moat_ok"])
    n_lesion_abstains = sum(1 for v in per_relation.values() if v["lesion_abstains"])
    n_byte_id = sum(1 for r in byte_id_rows if r["same"])
    seed_pass = (n_recall_ok == len(samples) and n_moat_ok == len(samples)
                 and n_lesion_abstains == len(samples) and n_byte_id == len(byte_id_rows))
    return {
        "seed": seed, "n_relations": len(samples), "n_recall_ok": n_recall_ok, "n_moat_ok": n_moat_ok,
        "n_lesion_abstains": n_lesion_abstains, "n_byte_identical_off": n_byte_id,
        "n_byte_identical_off_total": len(byte_id_rows),
        "per_relation": per_relation, "byte_identical_off": byte_id_rows, "seed_pass": seed_pass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_kb_relations.json"))
    args = ap.parse_args()

    t0 = time.time()
    samples, missing = _sample_facts()
    print(f"[{time.time()-t0:.0f}s] sampled {len(samples)} real facts from the shipped bundle "
          f"(missing relations: {missing})", flush=True)
    assert not missing, f"every targeted relation must have a real sample fact; missing {missing}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def _checkpoint(results, done):
        """Write partial progress after EVERY seed -- an interrupted run (session boundary, kill, crash) still
        leaves a valid, honestly-partial JSON on disk instead of losing completed seeds' work."""
        all_pass_so_far = all(r["seed_pass"] for r in results) if results else False
        partial = {
            "runner": "research/findings/raw/_nl_parser_real_kb_relations/verify_kb_relations.py",
            "seeds": args.seeds, "seeds_completed": done, "seeds_requested": args.seeds,
            "ltm_bundle": LTM_BUNDLE, "composer_kind": "onebrain",
            "n_target_relations": len(samples), "target_relations": sorted(samples),
            "verdict": ("GO" if (all_pass_so_far and done == args.seeds) else
                        ("PARTIAL" if done else "NO-GO")),
            "all_seeds_pass": all_pass_so_far, "complete": done == args.seeds,
            "total_seconds": round(time.time() - t0, 1),
            "results": results,
        }
        with open(args.out, "w") as fh:
            json.dump(partial, fh, indent=2, default=str)
        return partial

    results = []
    done = []
    for seed in args.seeds:
        r = run_seed(seed, samples, t0)
        results.append(r)
        done.append(seed)
        print(f"[{time.time()-t0:.0f}s] seed {seed} done: recall {r['n_recall_ok']}/{r['n_relations']}, "
              f"moat {r['n_moat_ok']}/{r['n_relations']}, lesion-abstains {r['n_lesion_abstains']}/{r['n_relations']}, "
              f"byte-id-off {r['n_byte_identical_off']}/{r['n_byte_identical_off_total']}, "
              f"seed_pass={r['seed_pass']}", flush=True)
        _checkpoint(results, done)   # incremental -- survives an interruption after THIS seed

    out = _checkpoint(results, done)
    print(f"[{time.time()-t0:.0f}s] wrote {args.out}  verdict={out['verdict']}", flush=True)
    return 0 if out["all_seeds_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
