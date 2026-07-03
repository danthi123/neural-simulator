"""CPU tests for EMERGE-88 -- functional integration: the form->role reservoir COMPREHENDS -> the composer ANSWERS.

A light structural test (the reservoir parses a transitive sentence into an (agent, action, patient) fact) + a slow
single-seed gate (the reservoir's comprehension drives correct who/what recall, the no-confab moat holds, and a
comprehension-lesion collapses the whole turn). CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge88_reservoir_comprehends_composer_answers_derisk as m88  # noqa: E402


def _fit_small_comprehender(seed=42):
    import research.runners._emerge62_discover_function_words_derisk as m62
    from research.runners._emerge78_reservoir_form_to_role_derisk import _content_pools, _gen, _TRAIN_KINDS
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    comp = m88.ReservoirComprehender(seed, discovered)
    rng = np.random.default_rng(seed * 101 + 5)
    comp.fit(_gen(_TRAIN_KINDS, 120, rng, subj, verb, obj))     # light fit for the structural test
    return comp, subj, verb, obj


def test_comprehend_parses_transitive_into_a_fact():
    comp, subj, verb, obj = _fit_small_comprehender(42)
    s, v, o = str(subj[0]), str(verb[0]), str(obj[0])
    fact = comp.comprehend(["the", s, v + "s", "the", o])
    assert fact.get("agent") == s
    assert fact.get("action") == v + "s"
    assert fact.get("patient") == o


def test_role_field_map_covers_the_svo_roles():
    assert m88._ROLE2FIELD["AGENT"] == "agent"
    assert m88._ROLE2FIELD["PREDICATE"] == "action"
    assert m88._ROLE2FIELD["THEME"] == "patient"


@pytest.mark.slow
def test_seed42_reservoir_comprehends_and_composer_answers():
    d = m88._derisk_one(42)
    assert d["parse_acc"] >= 0.90            # the reservoir parses transitive sentences into the right roles
    assert d["recall"] >= 0.90               # its comprehension drives correct who/what answers
    assert d["moat_false_accept"] <= 0.05    # the no-confab moat holds on the reservoir-parsed facts
    assert d["lesion_recall"] <= 0.55        # comprehension is load-bearing (lesion collapses the whole turn)
