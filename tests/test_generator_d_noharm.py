"""LOAD-BEARING no-harm: Generator-D is PURELY ADDITIVE; the validated
deliverable + the FROZEN (hardened) gate_core bars are byte-untouched,
and Generator-D adds NO new bar and pulls in NOTHING that mutates or
shadows song_g1_core / gate_core."""
import sys


def test_gate_core_bars_frozen_and_g1_untouched():
    import research.runners.subword_lm_gate_core as g
    # the HARDENED frozen set (post caa3816) is byte-stable; Generator-D
    # introduced NO new bar.
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20, 1.5, 0.5, 0.20,
                                                3, 1.0)
    assert not hasattr(g, "_G1_MARGIN")
    assert not hasattr(g, "_G1_ABS_FLOOR")


def test_generator_d_does_not_pull_song_g1_core():
    before = "research.runners.song_g1_core" in sys.modules
    import research.runners.distill_subword_lm_train  # noqa: F401
    import research.runners.generator_d_gate  # noqa: F401
    import sim.ngram_teacher  # noqa: F401
    import sim.soft_xent  # noqa: F401
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after, (
        "Generator-D must not import song_g1_core")


def test_soft_xent_is_faithful_generalization_of_validated_hard_CE():
    # the distill gradient's correctness is load-bearing: when q is
    # one-hot it MUST equal the validated hard CE/grad (pin it here so
    # a future regression to soft_xent is caught by no-harm too).
    import numpy as np
    from sim.soft_xent import soft_xent_loss, soft_xent_grad
    from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np
    logits = np.array([[0.4, -1.1, 2.0, 0.3]])
    q = np.array([0.0, 0.0, 1.0, 0.0])
    assert abs(soft_xent_loss(logits, q)
               - cross_entropy_loss_np(logits, 2)) < 1e-6
    assert np.allclose(soft_xent_grad(logits, q),
                       softmax_grad_np(logits, 2), atol=1e-6)
