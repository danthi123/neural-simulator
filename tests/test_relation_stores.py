"""CI guard for the RELATIONAL SCHEMA BREADTH beyond binary SVO: a 4-role FHRR store binds + recovers
DITRANSITIVE (ternary: agent-verb-recipient-theme) and PP (spatial: agent-verb-goal/location) relations,
abstaining on the unstored (no-confab moat). Skips gracefully if the corpus is absent. numpy-only, fast."""
import os
import pytest

CORPUS = "data/corpus/tinystories.txt"
pytestmark = pytest.mark.skipif(not os.path.exists(CORPUS), reason="needs the TinyStories corpus (regenerable)")


def test_ditransitive_ternary_store_recovers_and_abstains():
    """The dog GIVES the cat a bone -> both argument queries recover; the unstored abstains; permuted collapses."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_ditransitive_store_derisk import run_seed
    from research.runners.corpus_stream import load_token_stream_multi
    stories = load_token_stream_multi(CORPUS, max_stories=None)
    r = run_seed(42, stories, 256, n_facts=10)
    assert r["theme_acc"] >= 0.9 and r["recip_acc"] >= 0.9      # both argument queries recover
    assert r["moat_abstain"] >= 0.9                             # unstored -> abstain (no-confab moat)
    assert r["permuted_acc"] <= 0.1                             # wrong-verb query -> miss


def test_pp_spatial_store_discriminates_goal_from_location():
    """The owl flies TO the pond (goal) vs ON the rock (location) -> recovers + distinguishes goal/location."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_pp_relation_store_derisk import run_seed
    from research.runners.corpus_stream import load_token_stream_multi
    stories = load_token_stream_multi(CORPUS, max_stories=None)
    r = run_seed(42, stories, 256, n_facts=10)
    assert r["answer_acc"] >= 0.9                               # recovers the destination
    assert r["goal_loc_discrim"] >= 0.9                        # goal vs location kept separable
    assert r["moat_abstain"] >= 0.9                            # unstored -> abstain
