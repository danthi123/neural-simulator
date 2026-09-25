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


# ── AMENDMENT 3: the elemental partial-match edge (default OFF) ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def junction_el():
    env = L.FrameEnvironment(_corpus(), min_count=3)
    return J.FrameJunctionLexicon(42, env, elemental=True)


def _pre_post(lex):
    from sim.backend import to_host
    coo = lex.b.cp_connections.tocoo()
    return np.asarray(to_host(coo.row)), np.asarray(to_host(coo.col))


def test_elemental_flag_off_by_default_and_adds_nothing(junction, monkeypatch):
    monkeypatch.delenv(J.ELEMENTAL_ENV, raising=False)
    assert not J.elemental_enabled()
    assert junction.elemental is False and junction.variant == "junction" and not hasattr(junction, "S_E")
    # the same build with the flag explicitly off installs the identical connection data (no group, no draw)
    from sim.backend import to_host
    other = J.FrameJunctionLexicon(42, junction.env, elemental=False)
    assert np.array_equal(np.asarray(to_host(other.b.cp_connections.data)),
                          np.asarray(to_host(J.FrameJunctionLexicon(42, junction.env).b.cp_connections.data)))


def test_elemental_edge_reads_only_the_minus1_plus1_afferents(junction_el):
    lex = junction_el
    pre, post = _pre_post(lex)
    C, pools = lex.C, np.concatenate([lex.cn[0], lex.cx[0]])
    from_fr_to_pools = np.isin(pre, lex.fr) & np.isin(post, pools)
    assert from_fr_to_pools.sum() == 2 * C * 2 * L.N_CAT
    assert set(pre[from_fr_to_pools].tolist()) == set(J.elemental_afferents(lex.fr, C).tolist())
    minus2_plus2 = np.concatenate([lex.fr[L.OFFSETS.index(-2) * C:(L.OFFSETS.index(-2) + 1) * C],
                                   lex.fr[L.OFFSETS.index(2) * C:(L.OFFSETS.index(2) + 1) * C]])
    assert not np.any(np.isin(pre, minus2_plus2)), "the -2/+2 afferents must still project nowhere"
    assert np.isin(post, lex.fj).sum() == 2 * C * C, "the junctions keep exactly their two inputs each"
    assert lex.variant == "junction_elemental" and lex.S_E.shape == (2 * C, 2 * L.N_CAT)
    assert np.all(lex.WE_init > 0) and abs(lex.WE_init.mean() / J.W_INIT_E - 1.0) < 0.1


def test_elemental_lesions_hit_only_their_named_edge(junction_el, junction):
    from sim.backend import to_host
    lex = junction_el

    def edges():
        d = np.asarray(to_host(lex.b.cp_connections.data))
        return d[lex.S].copy(), d[lex.S_E].copy(), d[lex.S_j].copy(), d[lex.S_inh].copy()
    s0, e0, j0, i0 = edges()
    lex.set_lesion("elemental")
    s, e, j, i = edges()
    assert np.all(e == 0) and np.array_equal(s, s0) and np.array_equal(j, j0) and np.array_equal(i, i0)
    lex.set_lesion("conjunctive")
    s, e, j, i = edges()
    assert np.all(j == 0) and np.array_equal(s, s0) and np.array_equal(e, e0) and np.array_equal(i, i0)
    lex.set_lesion("afferent_zero")
    s, e, j, i = edges()
    assert np.all(s == 0) and np.all(e == 0) and np.array_equal(j, j0)
    lex.set_lesion(None)
    assert all(np.array_equal(x, y) for x, y in zip(edges(), (s0, e0, j0, i0))), "lesions must be reversible"
    for kind in J.ELEMENTAL_ONLY_LESIONS:
        with pytest.raises(ValueError):
            junction.set_lesion(kind)


def test_conjunctive_lesion_silences_every_junction_pair(junction_el):
    lex = junction_el
    lex.set_lesion("conjunctive")
    try:
        both, _ = lex.junction_response([(1, 2), (3, 4), (0, 5)], "both")
        assert (both == 0).all()
    finally:
        lex.set_lesion(None)
    assert J.and_smoke(lex, n_sample=12)["and_holds"], "the AND must hold again once the lesion is lifted"


def test_elemental_learning_moves_the_elemental_edge(junction_el):
    lex = junction_el
    words = NOUNS_TRAIN[:2] + VERBS_TRAIN[:2]
    lex.epochs = 1
    try:
        lex.train(words, np.array([1.0, 1.0, -1.0, -1.0])[:, None])
        assert not np.allclose(lex.WE, lex.WE_init), "the Oja rule must update the elemental edge"
        dec, rn, rx = lex.decide("dog")
        d = lex.drive_of("dog")
        assert d is not None and {"junction_cn", "junction_cx", "elemental_cn", "elemental_cx", "total"} <= set(d)
        lex.set_lesion("elemental")
        lex.decide("dog")
        de = lex.drive_of("dog")
        assert de["elemental_cn"] == 0.0 and de["elemental_cx"] == 0.0 and de["total"] <= d["total"] + 1e-9
        lex.set_lesion("conjunctive")
        lex.decide("dog")
        dc = lex.drive_of("dog")
        assert dc["junction_cn"] == 0.0 and dc["junction_cx"] == 0.0 and dc["total"] <= d["total"] + 1e-9
    finally:
        lex.set_lesion(None)
        lex.reset_learning()
        lex.epochs = L.EPOCHS
    assert np.array_equal(lex.WE, lex.WE_init)


def test_learned_edge_lesion_settles_both_edges_cell_wide(junction_el):
    """AMENDMENT 3: learned_edge resets BOTH learned edges to their start weights and the homeostatic settle scales
    both by the SAME per-postsynaptic-neuron factor (synaptic scaling is cell-wide)."""
    from sim.backend import to_host
    lex = junction_el
    lex.set_lesion("learned_edge")
    try:
        d = np.asarray(to_host(lex.b.cp_connections.data)).astype(np.float64)
        rj = d[lex.S] / lex.W_init
        re_ = d[lex.S_E] / lex.WE_init
        assert np.allclose(rj.mean(axis=0), re_.mean(axis=0), rtol=1e-4), "one scale per postsynaptic neuron"
        assert np.all(re_ >= 0.5 - 1e-4) and np.all(re_ <= 2.0 + 1e-4)
    finally:
        lex.set_lesion(None)


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
    monkeypatch.setenv(J.ELEMENTAL_ENV, "1")                              # ignored while the junction is off
    assert L.get_lexicon(seed=3, corpus_path=str(corpus)) is lex
    monkeypatch.delenv(J.ELEMENTAL_ENV, raising=False)
    monkeypatch.setenv("BRAIN_LEARNED_REFERENT_JUNCTION", "1")
    lexj = L.get_lexicon(seed=3, corpus_path=str(corpus))
    assert isinstance(lexj, J.FrameJunctionLexicon) and lexj.variant == "junction" and not lexj.elemental
    monkeypatch.setenv(J.ELEMENTAL_ENV, "1")                              # AMENDMENT 3: both flags -> elemental
    lexe = L.get_lexicon(seed=3, corpus_path=str(corpus))
    assert isinstance(lexe, J.FrameJunctionLexicon) and lexe.variant == "junction_elemental" and lexe.elemental
    monkeypatch.delenv(J.ELEMENTAL_ENV, raising=False)
    assert L.get_lexicon(seed=3, corpus_path=str(corpus)).variant == "junction"
    monkeypatch.setattr(mod.SpikingFrameCategoryLexicon, "train", orig_train)
    monkeypatch.setattr(L, "_LEXICON", None)
