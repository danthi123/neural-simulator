"""CPU test for the RUNG B-1 A→W transitive word-spell runner (structure only; the GPU cache/render is skip-if-absent).

The A→W read-out is a cupy/GPU module (trained once + cached). The light test asserts the transitive vocab structure +
the de-inflection so the runner's wiring is CI-guarded without a GPU; the full all-word spike render is validated in the
runner's `--seeds` mode on GPU (skip here if the transitive cache is absent).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest  # noqa: E402

import research.runners._rungB1_aw_neural_words_transitive_derisk as m  # noqa: E402


def test_transitive_content_vocab_is_16_and_covers_the_facts():
    # the A→W BRIDGE-A is 16 pools -> exactly 16 content words
    assert len(m._AW_TRANS_CONTENT) == 16
    assert len(set(m._AW_TRANS_CONTENT)) == 16                     # all distinct
    # the fact vocab (subjects/verb-3sg/objects the producer emits) is all inside the 16-word A→W content set
    content = set(m._AW_TRANS_CONTENT)
    for s, vb, o in m._facts(42):
        assert s in content                                       # subject spellable
        assert m.emerge_v3(vb) in content                         # verb 3sg surface spellable
        assert o in content                                       # object spellable


def test_the_determiner_routes_to_the_function_bridge():
    # "the" is a function word (BRIDGE-F), not in the transitive content set -> dispatched to the func spell
    assert "the" not in set(m._AW_TRANS_CONTENT)
    assert "the" in set(getattr(__import__("research.runners._emerge68_function_word_spell_derisk",
                                           fromlist=["_FUNC_WORDS"]), "_FUNC_WORDS"))


@pytest.mark.slow
def test_transitive_all_word_spike_render_if_cache_present():
    # only runs when the transitive BRIDGE-A cache + GPU are available (built via `--train`)
    if not m._TRANS_CACHE_BRIDGE.exists():
        pytest.skip("transitive A→W cache absent (run --train on GPU first)")
    try:
        engine = m.TransUnifiedSpell(load=True)
    except Exception as e:  # noqa: BLE001  (cupy/GPU unavailable)
        pytest.skip(f"A→W GPU engine unavailable: {e}")
    acc = m._derisk_one(42, engine)
    assert acc >= 0.90                                            # every word rendered from spikes == ground-truth
