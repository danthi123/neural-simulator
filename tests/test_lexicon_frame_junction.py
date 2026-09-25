"""Structural guards for the frame-junction referent lexicon (`lexicon_frame_junction`, default OFF): every junction
has exactly its two frame inputs and nothing reaches the category pools except through junctions; the AND holds and
the `coincidence` lesion turns it into an OR; lesions hit their named edge; `get_lexicon()` switches variant only on
the flag. Tiny synthetic corpus, numpy; no real corpus needed."""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import lexicon_spiking_frame_category as L  # noqa: E402
from research.runners import lexicon_frame_junction as J  # noqa: E402

NOUNS_TRAIN = ["dog", "cat", "ball", "cup"]
VERBS_TRAIN = ["run", "jump", "sing", "eat"]


def _corpus(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    nouns = NOUNS_TRAIN + ["owl"]
    verbs = VERBS_TRAIN + ["swim"]
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
def junction():
    env = L.FrameEnvironment(_corpus(), min_count=3)
    return J.FrameJunctionLexicon(42, env)


def test_every_junction_has_exactly_its_two_frame_inputs(junction):
    from sim.backend import to_host
    coo = junction.b.cp_connections.tocoo()
    pre, post = np.asarray(to_host(coo.row)), np.asarray(to_host(coo.col))
    C = junction.C
    is_fj = np.isin(post, junction.fj)
    assert is_fj.sum() == 2 * C * C
    for k in (0, 7, C * C - 1):
        a, b = divmod(k, C)
        src = set(pre[post == junction.fj[k]].tolist())
        assert src == {int(junction.fr[J._OFF_L * C + a]), int(junction.fr[J._OFF_R * C + b])}


def test_no_frame_afferent_reaches_a_category_pool_directly(junction):
    from sim.backend import to_host
    coo = junction.b.cp_connections.tocoo()
    pre, post = np.asarray(to_host(coo.row)), np.asarray(to_host(coo.col))
    pools = np.concatenate([junction.cn[0], junction.cx[0]])
    assert not np.any(np.isin(pre, junction.fr) & np.isin(post, pools))
    assert np.all(np.isin(pre[np.isin(post, pools) & ~np.isin(pre, np.concatenate(junction.inn + junction.ixx))],
                          junction.fj))


def test_and_holds_and_coincidence_lesion_makes_it_an_or(junction):
    sm = J.and_smoke(junction, n_sample=12)
    assert sm["and_holds"], sm
    junction.set_lesion("coincidence")
    try:
        pairs = [(1, 2), (3, 4), (5, 0)]
        left, _ = junction.junction_response(pairs, "left")
        assert (left > 0).all(), "the lesioned junction must fire to one afferent alone"
    finally:
        junction.set_lesion(None)
    assert J.and_smoke(junction, n_sample=12)["and_holds"], "the lesion must be fully reversible"


def test_learned_edge_lesion_settles_near_start_weights(junction):
    """AMENDMENT 2 (R4 fix): the lesion no longer installs raw W_init -- it runs a ONE-TIME homeostatic settle
    (`_r4_homeostatic_settle`) starting from W_init and returns the RESULT, `W_lesion_settled`. The installed
    weights must therefore (a) NOT be the artificially-inflated `W` this test set, and (b) stay within the settle's
    own multiplicative clip bound of the pre-learning start weights (never far from them, just not IDENTICAL)."""
    from sim.backend import to_host
    junction.W = junction.W_init * 1.5
    junction._install()
    junction.set_lesion("learned_edge")
    data = np.asarray(to_host(junction.b.cp_connections.data))
    installed = data[junction.S].astype(np.float64)
    assert not np.allclose(installed, (junction.W_init * 1.5).astype(np.float32))
    ratio = installed / junction.W_init
    assert np.all(ratio >= 0.5 - 1e-6) and np.all(ratio <= 2.0 + 1e-6), \
        "the settled weights must stay within the homeostatic scale's own clip bound of W_init"
    junction.set_lesion(None)
    data = np.asarray(to_host(junction.b.cp_connections.data))
    assert np.array_equal(data[junction.S], junction.W.astype(np.float32))
    junction.reset_learning()


def test_get_lexicon_switches_variant_only_on_the_flag(monkeypatch, tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text(" ".join(_corpus(seed=1, n=600)))
    monkeypatch.setattr(L, "_LEXICON", None)
    monkeypatch.setattr(L, "EPOCHS", 1)
    monkeypatch.delenv("BRAIN_LEARNED_REFERENT_JUNCTION", raising=False)
    import research.runners.lexicon_spiking_frame_category as mod
    orig_train = mod.SpikingFrameCategoryLexicon.train
    monkeypatch.setattr(mod.SpikingFrameCategoryLexicon, "train",
                        lambda self, w, lab, order_seed=None: None)   # structure only: skip training
    lex = L.get_lexicon(seed=3, corpus_path=str(corpus))
    assert type(lex) is L.SpikingFrameCategoryLexicon and lex.variant == "frame"
    assert L.get_lexicon(seed=3, corpus_path=str(corpus)) is lex          # singleton reused while the flag is unset
    monkeypatch.setenv("BRAIN_LEARNED_REFERENT_JUNCTION", "1")
    lexj = L.get_lexicon(seed=3, corpus_path=str(corpus))
    assert isinstance(lexj, J.FrameJunctionLexicon) and lexj.variant == "junction"
    monkeypatch.setattr(mod.SpikingFrameCategoryLexicon, "train", orig_train)
    monkeypatch.setattr(L, "_LEXICON", None)
