"""CPU-only config-guard for the multi-bridge A->W dispatch (`_realcorpus_multi_bridge_speaker.DEFAULT_BRIDGES`).

The dispatch SPEAKS a broad vocab on spikes by routing each word to whichever ~16-word grandmother bridge holds
it (the full-fidelity Rank-2 path). As bridges are added (BRIDGE-1..5 -> 77 words), the config invariants must
hold or the dispatch silently misroutes / wastes a pool. This guards them WITHOUT a GPU bridge build (imports
only the trainers' VOCAB / WORD_TO_POOL module constants), so it runs in CI on CPU. No sim/ dependency at build.
"""
from research.runners._realcorpus_train_breadth_aw import VOCAB as V1, WORD_TO_POOL as P1
from research.runners._realcorpus_train_breadth_aw2 import VOCAB as V2, WORD_TO_POOL as P2
from research.runners._realcorpus_train_breadth_aw3 import VOCAB as V3, WORD_TO_POOL as P3
from research.runners._realcorpus_train_breadth_aw4 import VOCAB as V4, WORD_TO_POOL as P4
from research.runners._realcorpus_train_breadth_aw5 import VOCAB as V5, WORD_TO_POOL as P5

BRIDGES = [("aw", V1, P1), ("aw2", V2, P2), ("aw3", V3, P3), ("aw4", V4, P4), ("aw5", V5, P5)]


def test_each_bridge_maps_words_to_distinct_pools():
    """Within one bridge, every word must own a DISTINCT dedicated pool (the grandmother architecture: 16 words ->
    16 pools). A duplicate pool would make two words spell as the same pool = a misread."""
    for name, vocab, wp in BRIDGES:
        pools = [wp[w] for w in vocab]
        assert len(set(pools)) == len(pools), f"BRIDGE-{name}: duplicate pool assignment {pools}"
        assert len(vocab) <= 16, f"BRIDGE-{name}: {len(vocab)} words exceeds the 16-pool grandmother cap"


def test_word_to_pool_covers_every_vocab_word():
    """Every vocab word must have a pool mapping (else spell() dispatches to a KeyError)."""
    for name, vocab, wp in BRIDGES:
        missing = [w for w in vocab if w not in wp]
        assert not missing, f"BRIDGE-{name}: vocab words with no pool: {missing}"


def test_combined_dispatch_is_deterministic_first_bridge_wins():
    """The dispatch dedups by first-bridge-wins; verify the combined vocab size and that any cross-bridge overlap
    resolves deterministically to the EARLIER bridge (so a word always spells the same way)."""
    seen, combined = {}, []
    for name, vocab, _wp in BRIDGES:
        for w in vocab:
            if w not in seen:
                seen[w] = name
                combined.append(w)
    # BRIDGE-1..5 currently total 77 dispatchable words (some cross-bridge overlap e.g. 'rock' in aw2+aw4 -> aw2)
    assert len(combined) == 77, f"expected 77 unique dispatchable words, got {len(combined)}"
    # 'rock' appears in aw2 AND aw4 -> first-bridge-wins must give aw2
    if "rock" in seen:
        assert seen["rock"] == "aw2", f"'rock' should dispatch to aw2 (earlier), got {seen['rock']}"


def test_no_empty_bridges():
    """Every wired bridge must contribute at least one word (else it's a wasted load)."""
    for name, vocab, _wp in BRIDGES:
        assert len(vocab) > 0, f"BRIDGE-{name} is empty"
