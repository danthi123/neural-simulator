"""Structural guards for the v2 referent detector (`lexicon_spiking_frame_category`): the decision circuit is COUPLED
(reciprocal inhibition exists and its lesion removes it), the decision is driven by LEARNED synapses (not a host
score/sign), each lesion hits exactly its named edge, and the production flag reaches v2. Tiny synthetic corpus, numpy,
a few seconds; no real corpus needed.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.d6_multiref_wm_production_organ as D6  # noqa: E402
from research.runners import lexicon_spiking_frame_category as L  # noqa: E402

NOUNS_TRAIN = ["dog", "cat", "ball", "cup"]
VERBS_TRAIN = ["run", "jump", "sing", "eat"]
NOUN_NOVEL = "owl"
VERB_NOVEL = "swim"


def _corpus(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    nouns = NOUNS_TRAIN + [NOUN_NOVEL]
    verbs = VERBS_TRAIN + [VERB_NOVEL]
    toks = []
    for _ in range(n):
        t = rng.integers(3)
        n1, n2 = rng.choice(nouns, 2)
        v = rng.choice(verbs)
        if t == 0:
            toks += ["the", n1, "saw", "a", n2, "."]
        elif t == 1:
            toks += ["they", v, "and", "then", "we", v, "."]
        else:
            toks += ["i", "like", "to", v, "with", "the", n1, "."]
    return [str(x) for x in toks]


@pytest.fixture(scope="module")
def trained():
    env = L.FrameEnvironment(_corpus(), min_count=3)
    lex = L.SpikingFrameCategoryLexicon(42, env, n_replicas=1)
    words = NOUNS_TRAIN + VERBS_TRAIN
    labels = np.array([1.0] * len(NOUNS_TRAIN) + [-1.0] * len(VERBS_TRAIN))
    lex.train(words, labels[:, None])
    return lex


def test_circuit_is_coupled_by_reciprocal_inhibition(trained):
    from sim.backend import to_host
    data = np.asarray(to_host(trained.b.cp_connections.data))
    assert trained.S_inh.size > 0 and np.all(data[trained.S_inh] > 0), "no FSI cross-inhibition synapses"
    trained.set_lesion("competition")
    data = np.asarray(to_host(trained.b.cp_connections.data))
    assert np.all(data[trained.S_inh] == 0.0)
    trained.set_lesion(None)
    data = np.asarray(to_host(trained.b.cp_connections.data))
    assert np.array_equal(data[trained.S_inh], trained.inh_init.astype(np.float32))


def test_no_host_score_and_novel_words_decided_by_learned_synapses(trained):
    assert not hasattr(trained, "scores"), "v2 must not carry a host category score"
    assert trained.classify(NOUN_NOVEL) is True
    assert trained.classify(VERB_NOVEL) is False
    # the decision rides the graded synaptic drive: noun drive margin > 0 > verb drive margin
    assert trained.graded_drive(NOUN_NOVEL)[0] > 0 > trained.graded_drive(VERB_NOVEL)[0]


def test_learned_edge_lesion_restores_exact_pre_learning_weights(trained):
    from sim.backend import to_host
    assert not np.array_equal(trained.W, trained.W_init), "training did not change the learned edge"
    trained.set_lesion("learned_edge")
    data = np.asarray(to_host(trained.b.cp_connections.data))
    assert np.array_equal(data[trained.S], trained.W_init.astype(np.float32))
    trained.set_lesion("afferent_zero")
    assert trained.classify(NOUN_NOVEL) is None and trained.classify(VERB_NOVEL) is None
    trained.set_lesion(None)
    data = np.asarray(to_host(trained.b.cp_connections.data))
    assert np.array_equal(data[trained.S], trained.W.astype(np.float32))


def test_unheard_word_abstains(trained):
    assert trained.classify("zebra") is None


def test_production_flag_reaches_v2(monkeypatch, trained):
    monkeypatch.setattr(L, "_LEXICON", trained)
    monkeypatch.setenv("BRAIN_LEARNED_REFERENT_LEXICON", "1")
    monkeypatch.delenv("BRAIN_LEARNED_REFERENT_LESION", raising=False)
    assert D6.extract_referents("the wolf watches the owl") == ["wolf", "owl"]
    monkeypatch.setenv("BRAIN_LEARNED_REFERENT_LESION", "1")
    D6.extract_referents("the wolf watches the owl")
    assert trained.lesion == "learned_edge"
    trained.set_lesion(None)
