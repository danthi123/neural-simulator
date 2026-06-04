"""Smoke tests for the unified-agent benchmark (the converge-not-add measurement).

Asserts the frozen conversational test set is well-formed, and that the validated constructed-codes path holds
at 320-concept scale on the robust categories (flat / 1-attr / 2-attr / depth-1 clause / who / abstain). The
depth-2 clause category is the documented boundary -- surfaced by the benchmark's report, NOT hard-asserted
here (it may cost a seed). Abstention is asserted total: the project's no-confabulation moat must hold.

Run: SIM_BACKEND=numpy python -m pytest tests/test_unified_agent_benchmark.py -q
"""
from research.runners.unified_agent_benchmark import (
    run_seed, ALL_FACTS, ABSTAIN_QUERIES, WHO_QUERIES, CORE_NOUNS, CORE_VERBS, CORE_ADJS, build_vocab)
from research.runners.nested_composition_agent import Clause


def _patient_words(p):
    if isinstance(p, Clause):
        return _patient_words(p.agent) + [p.action] + _patient_words(p.patient)
    if isinstance(p, tuple):
        mods = p[0]
        mm = list(mods) if isinstance(mods, (tuple, list)) else [mods]
        return mm + [p[1]]
    return [p]


def test_frozen_test_set_well_formed():
    """Every (agent,action) key is globally unique (query_patient well-defined); every word is in vocabulary;
    no abstention probe is a stored key (so a None there is genuine no-confabulation, not a missing fact)."""
    N, V, A = set(CORE_NOUNS), set(CORE_VERBS), set(CORE_ADJS)
    keys = set()
    for ag, ac, pa in ALL_FACTS:
        assert ag in N and ac in V
        for w in _patient_words(pa):
            assert w in N or w in V or w in A, f"{w} not in vocab"
        assert (ag, ac) not in keys, f"duplicate key {(ag, ac)}"
        keys.add((ag, ac))
    for ag, ac in ABSTAIN_QUERIES:
        assert (ag, ac) not in keys, f"abstention probe {(ag, ac)} is actually stored"
    # who-queries must resolve uniquely among flat facts
    assert len({(ac, pn) for ac, pn, _ in WHO_QUERIES}) == len(WHO_QUERIES)


def test_constructed_codes_robust_categories_seed42():
    """The validated constructed-codes path: robust categories all perfect at 320-concept scale, abstention
    total (no confabulation), overall well above chance."""
    r = run_seed(42, D=2048, mode="constructed")
    cats = r["categories"]
    for name in ("flat", "1-attribute", "2-attribute", "clause-depth1", "who-query", "abstain"):
        ok, tot = cats[name]
        assert ok == tot, f"{name} regressed: {ok}/{tot}  (misses: {[w for w in r['wrong'] if w['cat']==name]})"
    # no-confabulation moat: every abstention probe returns None
    assert cats["abstain"][0] == len(ABSTAIN_QUERIES)
    # overall (including the depth-2 boundary) must be high
    gok = sum(v[0] for v in cats.values())
    gtot = sum(v[1] for v in cats.values())
    assert gok / gtot >= 0.9, f"overall {gok}/{gtot} below 0.9"


def test_320_concept_codebook():
    """The benchmark genuinely faces a 320-concept codebook (the scale claim)."""
    nouns, verbs, adjs = build_vocab()
    assert len(nouns) + len(verbs) + len(adjs) == 320
