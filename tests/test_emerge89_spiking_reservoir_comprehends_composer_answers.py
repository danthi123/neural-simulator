"""CPU tests for EMERGE-89 -- fully-spiking comprehension -> composition: the ON-BRIDGE spiking reservoir COMPREHENDS
-> the composer ANSWERS.

A light structural test (the on-bridge `OnBridgeLSM` drops into EMERGE-88's `ReservoirComprehender` and parses a
transitive on spikes) + a slow single-seed gate. CPU/numpy, offline; the bridge step is heavy so the fit uses a
reduced train.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge89_spiking_reservoir_comprehends_composer_answers_derisk as m89  # noqa: E402


def test_onbridge_reservoir_drops_into_the_comprehender_and_parses_on_spikes():
    import research.runners._emerge62_discover_function_words_derisk as m62
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _content_pools, _gen, _TRAIN_KINDS
    from research.runners._emerge82_onbridge_lsm_derisk import OnBridgeLSM
    from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import ReservoirComprehender
    seed = 42
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    enc = Encoder(discovered)
    res = OnBridgeLSM(enc.dim, seed=seed, n=120)                 # small spiking reservoir for the structural test
    comp = ReservoirComprehender(seed, discovered, res=res, enc=enc)
    rng = np.random.default_rng(seed * 101 + 5)
    comp.fit(_gen(_TRAIN_KINDS, 40, rng, subj, verb, obj))
    assert res._last_mean_spikes > 0.3                          # the reservoir genuinely spiked during the fit
    s, v, o = str(subj[0]), str(verb[0]), str(obj[0])
    fact = comp.comprehend(["the", s, v + "s", "the", o])
    assert fact.get("agent") == s and fact.get("action") == v + "s" and fact.get("patient") == o


@pytest.mark.slow
def test_seed42_spiking_reservoir_comprehends_and_composer_answers():
    d = m89._derisk_one(42)
    assert d["mean_spikes_per_neuron"] > 0.5     # genuinely spiking
    assert d["parse_acc"] >= 0.90
    assert d["recall"] >= 0.90
    assert d["moat_false_accept"] <= 0.05
    assert d["lesion_recall"] <= 0.55
