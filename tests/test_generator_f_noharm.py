"""LOAD-BEARING no-harm: Generator-F is PURELY ADDITIVE; the FROZEN
hardened gate_core bars + the validated assets are byte-untouched;
NO new bar; no song_g1_core pull; torch is a pre-existing dep."""
import sys


def test_hardened_bars_frozen_and_no_g1():
    import research.runners.subword_lm_gate_core as g
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20, 1.5, 0.5,
                                                0.20, 3, 1.0)
    assert not hasattr(g, "_G1_MARGIN")


def test_generator_f_does_not_pull_song_g1_core():
    before = "research.runners.song_g1_core" in sys.modules
    import sim.tiny_transformer  # noqa: F401
    import research.runners.tiny_transformer_train  # noqa: F401
    import research.runners.generator_f_gate  # noqa: F401
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after


def test_torch_is_preexisting_dependency():
    import importlib.util
    assert importlib.util.find_spec("torch") is not None
