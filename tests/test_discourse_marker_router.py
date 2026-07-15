"""CI guard for the OPEN-VOCAB discourse-marker router (the FluidChat `open_vocab_dispatch` wire-in).

Ckpt-free / GPU-free / pure-numpy: guards the router's decisions directly (the FluidChat end-to-end path needs the
local 21M gen artifacts and is exercised where those exist). De-risk: 6-seed held-out-synonym 1.000 / OOD->None 1.000
(`2026-07-15-discourse-marker-routing-is-semantic-nearest-intent-plus-novelty-threshold-...`).
"""
import os
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._discourse_marker_router import DiscourseMarkerRouter, INTENT_MARKERS

SEEDS = [42, 43, 44, 100, 101, 102]


@pytest.mark.parametrize("seed", SEEDS)
def test_attested_markers_route_to_their_intent(seed):
    r = DiscourseMarkerRouter(seed=seed)
    for intent, markers in INTENT_MARKERS.items():
        for m in markers:
            assert r.route([m]) == intent, f"attested {m!r} -> {r.route([m])} != {intent}"


@pytest.mark.parametrize("seed", SEEDS)
def test_novel_synonyms_route_open_vocabulary(seed):
    """The capability the CLOSED keyword set lacks: a synonym NOT in any keyword set routes by semantic proximity."""
    r = DiscourseMarkerRouter(seed=seed)
    for syn, intent in [("versus", "COMPARE"), ("differ", "COMPARE"), ("alike", "SHARE"), ("akin", "SHARE"),
                        ("lineage", "TAXONOMY"), ("taxonomy", "TAXONOMY")]:
        assert r.route([syn]) == intent, f"novel synonym {syn!r} -> {r.route([syn])} != {intent}"


@pytest.mark.parametrize("seed", SEEDS)
def test_ood_and_wh_tokens_fall_through(seed):
    """OOD / wh / content tokens -> None (the moat: the caller falls through to the neural wh-parse, not a marker intent)."""
    r = DiscourseMarkerRouter(seed=seed)
    for utt in [["what", "does", "the", "dog", "eat"], ["who", "chases", "the", "cat"], ["zzz", "qqq"]]:
        assert r.route(utt) is None, f"{utt} -> {r.route(utt)} (expected None fallthrough)"


@pytest.mark.parametrize("seed", SEEDS)
def test_first_recognised_marker_wins_in_a_phrase(seed):
    r = DiscourseMarkerRouter(seed=seed)
    assert r.route(["how", "are", "dogs", "and", "cats", "different"]) == "COMPARE"
    assert r.route(["what", "do", "dogs", "and", "cats", "share"]) == "SHARE"
    assert r.route(["trace", "the", "dog", "ancestry"]) == "TAXONOMY"
