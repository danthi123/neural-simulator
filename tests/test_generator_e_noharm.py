"""LOAD-BEARING no-harm: Generator-E is PURELY ADDITIVE; the validated
deliverable + the FROZEN hardened gate_core bars are byte-untouched;
NgramTeacher reused UNMODIFIED; NO new bar; no song_g1_core pull."""
import sys


def test_hardened_bars_frozen_and_no_g1():
    import research.runners.subword_lm_gate_core as g
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20, 1.5, 0.5, 0.20,
                                                3, 1.0)
    assert not hasattr(g, "_G1_MARGIN")


def test_generator_e_does_not_pull_song_g1_core():
    before = "research.runners.song_g1_core" in sys.modules
    import sim.ngram_generate  # noqa: F401
    import sim.ngram_ppl  # noqa: F401
    import research.runners.generator_e_gate  # noqa: F401
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after


def test_ngram_teacher_reused_unmodified_contract():
    # Generator-E treats NgramTeacher as the runtime model; pin its
    # contract (it is reused byte-UNMODIFIED).
    import numpy as np
    from sim.ngram_teacher import NgramTeacher
    t = NgramTeacher()
    t.train([1, 2, 3, 1, 2, 3, 1, 2, 4] * 30, vocab_size=8)
    q = t.soft_dist((1, 2))
    assert q.shape == (8,) and abs(float(q.sum()) - 1.0) < 1e-9
