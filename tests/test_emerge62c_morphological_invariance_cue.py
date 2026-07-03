"""CI guard for EMERGE-62c -- the 4th (MORPHOLOGICAL-INVARIANCE) cue on the function-word discovery.

Fast, offline, CPU/numpy. Asserts the load-bearing properties of the morphological cue without running the full
6-seed real-corpus de-risk (the real-corpus checks are gated on the corpus file existing; skipped if absent).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._emerge62c_morphological_invariance_cue_derisk import (  # noqa: E402
    _base_stems, morphological_variant_flags, discover_4d, _derisk_controlled, real_corpus_morph_check,
)
from research.runners._emerge62b_function_words_position_cue_derisk import (  # noqa: E402
    sentences_from_controlled, compute_stats_positional, discover_3d,
)
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    GROUND_TRUTH_CLOSED, MIN_FREQ, _prf,
)


def test_base_stems_regular_inflections():
    # -s / -es / -ies / -ed / -ing base stems are proposed
    assert "give" in _base_stems("gives")
    assert "hug" in _base_stems("hugs")
    assert "make" in _base_stems("makes")
    assert "carry" in _base_stems("carries")
    assert "wash" in _base_stems("washes")
    assert "want" in _base_stems("wanted")
    assert "make" in _base_stems("making")
    assert "walk" in _base_stems("walking")


def test_base_stems_guards_non_inflectional_s():
    # double-s / -us / -is are NOT stripped (not 3sg/plural morphology) -> guards is/was/has/this from false stemming
    assert _base_stems("is") == set()
    assert _base_stems("us") == set()
    assert "grass" not in _base_stems("grass") and _base_stems("grass") == set()


def test_morph_flag_excludes_content_verbs_protects_function_inflections():
    # a realistic-scale synthetic vocab so the Goldilocks percentile scale is meaningful: many low-freq/low-coverage
    # filler content words, a couple of content verb paradigms (give/gives, hug/hugs), and the auxiliary do/does with
    # `do` placed at the TOP of the freq+coverage distribution (function-like).
    fillers = [f"w{i}" for i in range(20)]
    words = ["do", "does", "give", "gives", "hug", "hugs", "the", "cat"] + fillers
    # `do`+`the` at the top of freq+coverage (function-like); content bases mid; fillers low.
    freq = np.array([2000.0, 400.0, 200.0, 60.0, 200.0, 60.0, 2500.0, 150.0] + [15.0 + i for i in range(20)])
    cover = np.array([0.85, 0.40, 0.30, 0.20, 0.30, 0.20, 0.95, 0.25] + [0.05 + 0.005 * i for i in range(20)])
    flags, base_of = morphological_variant_flags(words, freq, cover)
    fd = dict(zip(words, flags))
    # inflected content verbs flagged (content base present + base NOT function-like)
    assert fd["gives"]
    assert fd["hugs"]
    # `does` PROTECTED because its base `do` is itself function-like (top of the freq+coverage Goldilocks distribution)
    assert not fd["does"]
    # bare function words / bare content words not flagged (no present inflected base of themselves)
    assert not fd["the"]
    assert not fd["do"]


def test_controlled_stream_not_regressed_and_moat_intact():
    d = _derisk_controlled(42)
    # 4-cue does not regress the 3-cue F1 on the controlled EMERGE stream
    assert d["F1_4d"] >= d["F1_3d"] - 1e-9
    # recall NOT regressed vs the 3-cue level (the controlled narrow-GT recall inherits EMERGE-62/62b's `it` miss --
    # `it` is high-freq but low-coverage in the controlled stream; the morphological cue must not drop it further), and
    # ALL frame function words are recovered (the load-bearing frame-feed property)
    assert d["R_4d"] >= d["R_3d"] - 1e-9
    assert d["frame_recall_4d"] >= 0.999 and d["frame_covered"]
    # producer renders on the discovered set; gate-first moat intact (0 producer invocations on abstains)
    assert d["render_ok"] >= 0.99
    assert d["moat_calls_on_abstain"] == 0 and d["answer_produced"]


def test_controlled_freqshuffle_and_nostream_collapse():
    d = _derisk_controlled(42)
    # frequency-shuffle destroys the signal -> F1 far below the main 4-cue F1
    assert d["F1_freq_shuffle"] <= d["F1_4d"] - 0.30
    # no-stream -> empty discovered set
    assert d["nostream_empty"]


def test_controlled_heldout_generalises():
    d = _derisk_controlled(42)
    # a withheld function word (does) is still classified CLOSED (protected by the base-is-function guard);
    # a withheld content word (trout) is still classified OPEN -- generalisation, not memorisation
    assert d["heldout_fw_closed"] is True
    assert d["heldout_cw_open"] is True


def test_morph_cue_is_asymmetric_exclusion_of_3cue_set():
    # the 4-cue discovered set is a SUBSET of the 3-cue set (exclusion-only, never adds)
    sents = sentences_from_controlled(42)
    words, freq, cover, posscore, _, _ = compute_stats_positional(sents, MIN_FREQ)
    morph, _ = morphological_variant_flags(words, freq, cover)
    d3, _, _ = discover_3d(words, freq, cover, posscore)
    d4, _, excl = discover_4d(words, freq, cover, posscore, morph)
    assert d4 <= d3
    assert excl == (d3 - d4)


@pytest.mark.skipif(
    not (_REPO / "data" / "corpus" / "ra_finetune_corpus.txt").exists(),
    reason="real corpus ra_finetune_corpus.txt absent",
)
def test_real_corpus_precision_up_recall_held_shuffle_collapses():
    rc = real_corpus_morph_check()
    assert rc.get("available")
    n = rc["narrow_gt"]
    # precision up from the 3-cue level, recall held at 1.00, frame function words all recovered
    assert n["P_4d"] > n["P_3d"] + 1e-6
    assert n["R_4d"] >= 0.999
    assert rc["frame_recall_4d"] >= 0.999
    # the excluded false positives are exactly inflected content surfaces (the named boundary class)
    for w in ("gives", "hugs", "makes"):
        assert w in rc["excluded_are_inflected_surfaces"]
    # MORPHOLOGY-SHUFFLE collapses: random morph flag deletes true function words (recall breaks) + no purification
    ms = rc["morphology_shuffle"]
    assert ms["R"] < n["R_4d"] - 1e-6            # shuffle breaks recall (input-destruction)
    assert ms["F1"] <= n["F1_3d"] + 1e-6         # shuffle cannot purify above the 3-cue baseline
