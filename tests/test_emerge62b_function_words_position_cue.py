"""CI for EMERGE-62b -- ADD the 3rd distributional cue (PHRASE-BOUNDARY / SYNTACTIC-POSITION ALIGNMENT) to the
function-word discovery, so the closed-class inventory self-organises on the REAL noisy corpus (Yang-Getz 2026 3rd
universal property; Redington/Cartwright-Brent left-neighbour role; catalog G.12 Broca open/closed dissociation).

CPU/numpy, offline. Small-stream smoke of: the sentence-aware positional stats, the asymmetric-exclusion 3D discovery
(controlled NOT regressed), the position-shuffle collapse (load-bearing), the freq-shuffle + no-stream + held-out
controls, the producer render + moat, and (skip-if-absent) the REAL-corpus precision lift with recall held.
"""
import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge62b_function_words_position_cue_derisk import (  # noqa: E402
    sentences_from_controlled, compute_stats_positional, discover_2d, discover_3d,
    real_corpus_position_check, render_on_discovered, GROUND_TRUTH_CLOSED, FRAME_FUNCTION_WORDS,
    TF_PCT, TC_PCT, TP_EXCL, MIN_FREQ, _prf,
)


def _cstats(seed, n=9000):
    """Sentence-segment the controlled stream + compute the positional stats (freq, coverage, posscore)."""
    # build a smaller stream for CI speed by re-running the segmenter over EMERGE-62's build_stream.
    from research.runners._emerge62_discover_function_words_derisk import build_stream, SENT_PERIOD
    toks = build_stream(seed, n_sentences=n)
    sents, cur = [], []
    for t in toks:
        if t == SENT_PERIOD:
            if cur:
                sents.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        sents.append(cur)
    return sents, compute_stats_positional(sents, MIN_FREQ)


def test_controlled_stream_sentence_segmentation():
    """The controlled stream segments into sentences with BOTH content AND function words present (the closed/open
    split is a real discovery problem)."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42, n=4000)
    assert len(sents) > 100
    vocab = set(words)
    assert all(fw in vocab for fw in FRAME_FUNCTION_WORDS)     # function words present
    assert "owl" in vocab and "fly" in vocab                    # content words present


def test_3d_does_not_regress_controlled_and_recovers_frame_words():
    """On the controlled stream the 3rd (position) cue does NOT regress the 2D discovery (3D F1 >= 2D F1), still
    recovers ALL frame function words, and still excludes clear content words."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42)
    gt = GROUND_TRUTH_CLOSED & set(words)
    d2, _ = discover_2d(words, freq, cover)
    d3, _, excluded = discover_3d(words, freq, cover, posscore)
    _, R2, F2 = _prf(d2, gt)
    P3, R3, F3 = _prf(d3, gt)
    assert F3 >= F2 - 1e-9, f"3D regressed the controlled domain (2D {F2:.3f} -> 3D {F3:.3f})"
    # recall NOT regressed vs 2D (the position cue must not drop any closed-class word the 2D rule found). The
    # controlled 2D recall is stream-dependent (the low-coverage pronoun 'it' is not a 2D candidate at this scale) --
    # the load-bearing property is that the 3rd cue does not LOSE any 2D find.
    assert R3 >= R2 - 1e-9, f"3D dropped a closed-class word the 2D rule found (2D R {R2:.3f} -> 3D R {R3:.3f})"
    for fw in FRAME_FUNCTION_WORDS:
        assert fw in d3, f"frame function word {fw!r} lost by the position cue"
    for cw in ("owl", "fly", "trout", "pond"):
        if cw in words:
            assert cw not in d3, f"content word {cw!r} wrongly discovered"


def test_position_shuffle_control_collapses():
    """POSITION-SHUFFLE (permute the position statistic<->identity mapping) destroys the 3rd cue -> discovery collapses
    (the load-bearing input-destruction anti-cheat proving the position cue is real, not spurious)."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42)
    gt = GROUND_TRUTH_CLOSED & set(words)
    d3, _, _ = discover_3d(words, freq, cover, posscore)
    _, _, F3 = _prf(d3, gt)
    rng = np.random.default_rng(777)
    perm = rng.permutation(len(words))
    d3_shuf, _, _ = discover_3d(words, freq, cover, posscore[perm])
    _, _, F3_shuf = _prf(d3_shuf, gt)
    assert F3 >= F3_shuf + 0.10, f"position-shuffle did not collapse (3D {F3:.3f} vs shuffle {F3_shuf:.3f})"


def test_frequency_shuffle_control_collapses():
    """FREQUENCY-SHUFFLE (permute the freq/coverage/position<->identity mapping) collapses discovery far below main."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42)
    gt = GROUND_TRUTH_CLOSED & set(words)
    d3, _, _ = discover_3d(words, freq, cover, posscore)
    _, _, F3 = _prf(d3, gt)
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(words))
    d3_shuf, _, _ = discover_3d(words, freq[perm], cover[perm], posscore[perm])
    _, _, F3_shuf = _prf(d3_shuf, gt)
    assert F3 >= F3_shuf + 0.30, f"freq-shuffle did not collapse (3D {F3:.3f} vs shuffle {F3_shuf:.3f})"


def test_no_stream_control_yields_empty_set():
    """NO-STREAM (empty stream) -> no statistics -> no discovery (empty set)."""
    words, freq, cover, posscore, ff, pbn = compute_stats_positional([], MIN_FREQ)
    d3, _, _ = discover_3d(words, freq, cover, posscore)
    assert d3 == set()


def test_heldout_word_classified_by_own_stats_with_position_gate():
    """A function word (does) and a content word (trout) WITHHELD from the fitting slice are still classified correctly
    by their OWN stats vs frozen freq/coverage/position thresholds (generalisation, not memorisation)."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42)
    assert "does" in words and "trout" in words
    keep = [w for w in words if w not in ("does", "trout")]
    ki = [words.index(w) for w in keep]
    logfk = np.log(freq[ki])
    ck = cover[ki]
    posk = posscore[ki]

    def classify(w):
        i = words.index(w)
        pf = float((logfk < math.log(freq[i])).mean())
        pc = float((ck < cover[i]).mean())
        pp = float((posk < posscore[i]).mean())
        return (pf >= TF_PCT) and (pc >= TC_PCT) and (pp >= TP_EXCL)

    assert classify("does") is True          # held-out function word -> closed (survives the position gate)
    assert classify("trout") is False        # held-out content word -> open (excluded)


def test_producer_renders_on_discovered_3d_set_and_moat_holds():
    """The DISCOVERED (3D) set feeds the EMERGE-59 frames: held-out facts render correctly on the discovered function
    words, and the gate-first no-confab moat holds (0 producer invocations on abstains)."""
    sents, (words, freq, cover, posscore, ff, pbn) = _cstats(42)
    d3, _, _ = discover_3d(words, freq, cover, posscore)
    facts = [{"subject": "owl", "ability_verb": "fly", "intr_verb": "walks"},
             {"subject": "penguin", "ability_verb": "fly", "intr_verb": "walks"}]
    render_ok, moat_calls, answer_produced, frame_covered = render_on_discovered(42, d3, facts)
    assert frame_covered is True
    assert render_ok >= 0.99
    assert moat_calls == 0
    assert answer_produced is True


def test_real_corpus_precision_lift_recall_held():
    """On the REAL corpus, the 3rd cue lifts precision above the 2D level with recall HELD at 1.00 + frame-recall 1.00,
    and the position-shuffle control collapses below the 2D level (load-bearing). Skips if the corpus is absent."""
    rc = real_corpus_position_check()
    if not rc.get("available"):
        pytest.skip(f"real corpus unavailable: {rc.get('reason')}")
    n = rc["narrow_gt"]
    assert n["R_3d"] == pytest.approx(1.0), "real recall must stay 1.00 for the true closed class"
    assert n["P_3d"] > n["P_2d"] + 1e-6, "position cue must lift real precision"
    assert n["F1_3d"] > n["F1_2d"] + 0.02, "real F1 must rise materially"
    assert rc["frame_recall_3d"] == pytest.approx(1.0), "all frame function words must survive"
    ps = rc["position_shuffle"]
    assert n["F1_3d"] - ps["F1"] >= 0.05, "position-shuffle must collapse below the 3D result"
    assert ps["F1"] <= n["F1_2d"] + 1e-6, "position-shuffle must fall to/below the 2D level (load-bearing)"
    assert rc["n_excluded_true_content"] > 0, "the position cue must exclude genuine content-word false positives"
