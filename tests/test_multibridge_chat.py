"""Unit tests for multibridge_chat.py — multi-bridge ensemble vocab routing.

Tests the lightweight pieces (vocab tables, routing, cosine helpers)
without loading actual bridges. Bridge-load smoke tests in a separate
suite gated by checkpoint presence.
"""
from __future__ import annotations
import importlib
import numpy as np
import pytest

import research.runners.multibridge_chat as mbc


# ----------------------------------------------------------------------
# Vocab table structure
# ----------------------------------------------------------------------

ALL_SET_VOCABS = {
    "set1": mbc.SET1_VOCAB,
    "set2": mbc.SET2_VOCAB,
    "set3": mbc.SET3_VOCAB,
    "set4": mbc.SET4_VOCAB,
    "set5": mbc.SET5_VOCAB,
}


class TestVocabStructure:
    """Each vocab set has the right structure: 4 motors at [0..3],
    12 concept words at [4..15], all keys present."""

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_word_to_idx_size(self, name):
        """word_to_idx has exactly 16 entries (4 motors + 12 concepts)."""
        v = ALL_SET_VOCABS[name]
        assert len(v["word_to_idx"]) == 16, (
            f"{name}: word_to_idx has {len(v['word_to_idx'])} entries, "
            f"expected 16"
        )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_motors_at_front(self, name):
        """The 4 motor words are at indices 0-3."""
        v = ALL_SET_VOCABS[name]
        motors = ["north", "east", "south", "west"]
        for i, w in enumerate(motors):
            assert v["word_to_idx"][w] == i, (
                f"{name}: motor {w} at idx {v['word_to_idx'][w]}, expected {i}"
            )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_concept_words_count(self, name):
        """concept_words list has exactly 12 entries."""
        v = ALL_SET_VOCABS[name]
        assert len(v["concept_words"]) == 12, (
            f"{name}: {len(v['concept_words'])} concept words, expected 12"
        )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_concept_words_have_pools(self, name):
        """Every concept word has a pool mapping."""
        v = ALL_SET_VOCABS[name]
        for w in v["concept_words"]:
            assert w in v["word_to_pool"], (
                f"{name}: concept word '{w}' missing from word_to_pool"
            )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_concept_words_have_idx(self, name):
        """Every concept word has a word_to_idx entry."""
        v = ALL_SET_VOCABS[name]
        for w in v["concept_words"]:
            assert w in v["word_to_idx"], (
                f"{name}: concept word '{w}' missing from word_to_idx"
            )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_word_to_idx_concept_range(self, name):
        """Concept words are at indices 4-15."""
        v = ALL_SET_VOCABS[name]
        for w in v["concept_words"]:
            idx = v["word_to_idx"][w]
            assert 4 <= idx <= 15, (
                f"{name}: concept '{w}' at idx {idx}, expected 4-15"
            )

    @pytest.mark.parametrize("name", list(ALL_SET_VOCABS.keys()))
    def test_pool_naming_convention(self, name):
        """Pool names follow noun_pool_X / verb_pool_X / adjective_pool_X."""
        v = ALL_SET_VOCABS[name]
        valid_prefixes = ("noun_pool_", "verb_pool_", "adjective_pool_")
        for w, pool in v["word_to_pool"].items():
            assert pool.startswith(valid_prefixes), (
                f"{name}: pool '{pool}' for word '{w}' has invalid prefix"
            )


class TestCrossSetUniqueness:
    """Vocab sets are mutually exclusive on concept words.

    This is the foundational invariant for multi-bridge scaling: each
    bridge owns its 12 words, no overlap. Total vocab = sum of set sizes.
    """

    def test_concept_words_mutually_exclusive(self):
        """No two sets share a concept word."""
        seen = {}
        for name, v in ALL_SET_VOCABS.items():
            for w in v["concept_words"]:
                if w in seen:
                    pytest.fail(
                        f"Word '{w}' appears in both {seen[w]} and {name}"
                    )
                seen[w] = name
        # Total = 5 sets * 12 words = 60 concept words
        assert len(seen) == 60, (
            f"Expected 60 unique concept words, got {len(seen)}"
        )

    def test_motors_shared_across_sets(self):
        """All sets share the same motor words (north/east/south/west)."""
        motors_set = set(["north", "east", "south", "west"])
        for name, v in ALL_SET_VOCABS.items():
            assert motors_set.issubset(set(v["word_to_idx"].keys())), (
                f"{name} missing one or more motors"
            )

    def test_pools_unique_per_set(self):
        """Within each set, pool names are unique (no two words share a pool)."""
        for name, v in ALL_SET_VOCABS.items():
            pools = list(v["word_to_pool"].values())
            assert len(pools) == len(set(pools)), (
                f"{name}: duplicate pool names within set"
            )

    def test_pool_names_mutually_exclusive_across_sets(self):
        """Each set has DIFFERENT pool names (e.g. noun_pool_APPLE vs
        noun_pool_TREE), so a single bridge architecture only handles
        one set's pools.
        """
        seen_pools = {}
        for name, v in ALL_SET_VOCABS.items():
            for pool in v["word_to_pool"].values():
                if pool in seen_pools:
                    pytest.fail(
                        f"Pool '{pool}' shared between {seen_pools[pool]} "
                        f"and {name} -- bridges would collide"
                    )
                seen_pools[pool] = name


class TestRoutingHelpers:
    """find_bridge_for_word and find_bridges_for_words route correctly."""

    def _make_member(self, set_name):
        """Build a BridgeMember stub (no bridge load) for routing tests."""
        m = mbc.BridgeMember(
            bridge_path=f"fake_{set_name}.h5",
            vocab_set=ALL_SET_VOCABS[set_name],
            n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24,
            sparsity=0.05, n_words_for_orthogonal=16,
            encoding_steps=500, balanced_teacher_pA=500.0,
            top_k=100, name=set_name,
        )
        return m

    def test_find_bridge_for_word_intra_set(self):
        """Word in one set's vocab routes to that bridge."""
        m1 = self._make_member("set1")
        m2 = self._make_member("set2")
        m3 = self._make_member("set3")
        members = [m1, m2, m3]
        # set1 has 'apple', set2 has 'tree', set3 has 'house'
        assert mbc.find_bridge_for_word(members, "apple") is m1
        assert mbc.find_bridge_for_word(members, "tree") is m2
        assert mbc.find_bridge_for_word(members, "house") is m3

    def test_find_bridge_for_word_unknown(self):
        """Word not in any set returns None."""
        m1 = self._make_member("set1")
        members = [m1]
        assert mbc.find_bridge_for_word(members, "xyzzy") is None

    def test_find_bridges_for_words_intra_set(self):
        """Both words in one set returns that bridge."""
        m1 = self._make_member("set1")
        m2 = self._make_member("set2")
        members = [m1, m2]
        # apple+big both in set1
        assert mbc.find_bridges_for_words(members, ["apple", "big"]) is m1
        # tree+fast both in set2
        assert mbc.find_bridges_for_words(members, ["tree", "fast"]) is m2

    def test_find_bridges_for_words_cross_set(self):
        """Words split across two sets returns None (cross-set case)."""
        m1 = self._make_member("set1")
        m2 = self._make_member("set2")
        members = [m1, m2]
        # apple in set1, fast in set2 -- no single bridge has both
        assert mbc.find_bridges_for_words(members, ["apple", "fast"]) is None

    def test_motors_in_all_sets(self):
        """Motors are in every set; routing picks the first."""
        m1 = self._make_member("set1")
        m2 = self._make_member("set2")
        members = [m1, m2]
        # 'north' is in BOTH; routing picks the first (set1)
        # Note: BridgeMember.vocab only contains concept_words (not motors)
        # so motors aren't routable via find_bridge_for_word. Verify:
        assert mbc.find_bridge_for_word(members, "north") is None


class TestCosineToWordWithVocab:
    """Per-bridge cosine helper takes word_to_idx as arg (not global)."""

    def test_returns_zero_for_unknown_word(self):
        """Word not in word_to_idx returns 0.0 cosine."""
        word_to_idx = {"north": 0, "apple": 4}
        pat = np.random.RandomState(42).rand(100)
        score = mbc.cosine_to_word_with_vocab(
            pat, "xyzzy", n_lang_out=100,
            word_to_idx=word_to_idx,
        )
        assert score == 0.0

    def test_returns_zero_for_zero_pattern(self):
        """Zero pattern -> 0 cosine."""
        word_to_idx = {"apple": 4}
        pat = np.zeros(100)
        score = mbc.cosine_to_word_with_vocab(
            pat, "apple", n_lang_out=100,
            word_to_idx=word_to_idx,
        )
        assert score == 0.0

    def test_matches_self(self):
        """Pattern matched to its own orthogonal_drive_pattern returns ~1.0."""
        from sim.text_embeddings import orthogonal_drive_pattern
        word_to_idx = {"apple": 4}
        pat = orthogonal_drive_pattern(
            cue_idx=4, n_cues=16, n_neurons=2048,
            drive_max_pA=1.0, sparsity=0.05,
        )
        score = mbc.cosine_to_word_with_vocab(
            pat, "apple", n_lang_out=2048,
            word_to_idx=word_to_idx,
            n_words_for_orthogonal=16, sparsity=0.05,
        )
        assert score > 0.99, f"self-match cos={score}, expected >0.99"


class TestQuerySentenceTemplate:
    """query_sentence_template matches tag-name templates across bridges."""

    def _make_member_with_tags(self, set_name, tags):
        """Make a BridgeMember stub with pre-populated encoded_tags."""
        m = mbc.BridgeMember(
            bridge_path=f"fake_{set_name}.h5",
            vocab_set=ALL_SET_VOCABS[set_name],
            n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24,
            sparsity=0.05, n_words_for_orthogonal=16,
            encoding_steps=500, balanced_teacher_pA=500.0,
            top_k=100, name=set_name,
        )
        m.encoded_tags = list(tags)
        return m

    def test_subject_query_single_match(self):
        """'who ate apple?' template ['*','ate','apple'] finds 'alice_ate_apple'."""
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        results = mbc.query_sentence_template([m1], ["*", "ate", "apple"])
        assert len(results) == 1
        assert results[0]["wildcards"] == ["alice"]
        assert results[0]["tag"] == "alice_ate_apple"
        assert results[0]["bridge"] == "set1"

    def test_subject_query_multiple_matches(self):
        """Multiple sentences with same verb+obj find multiple subjects."""
        m1 = self._make_member_with_tags(
            "set1", ["alice_ate_apple", "bob_ate_apple"])
        results = mbc.query_sentence_template([m1], ["*", "ate", "apple"])
        assert len(results) == 2
        subjects = sorted(r["wildcards"][0] for r in results)
        assert subjects == ["alice", "bob"]

    def test_subject_query_across_bridges(self):
        """Same template matches tags in DIFFERENT bridges (cross-set)."""
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        m2 = self._make_member_with_tags("set2", ["bob_ate_apple"])
        results = mbc.query_sentence_template(
            [m1, m2], ["*", "ate", "apple"])
        assert len(results) == 2
        # Both bridges contributed
        bridges = sorted(r["bridge"] for r in results)
        assert bridges == ["set1", "set2"]
        subjects = sorted(r["wildcards"][0] for r in results)
        assert subjects == ["alice", "bob"]

    def test_object_query(self):
        """'what did alice eat?' template ['alice','ate','*'] finds objects."""
        m1 = self._make_member_with_tags(
            "set1", ["alice_ate_apple", "alice_ate_cake", "bob_ate_apple"])
        results = mbc.query_sentence_template(
            [m1], ["alice", "ate", "*"])
        assert len(results) == 2
        objects = sorted(r["wildcards"][0] for r in results)
        assert objects == ["apple", "cake"]

    def test_no_match_returns_empty(self):
        """Template that doesn't match any tag returns empty list."""
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        results = mbc.query_sentence_template(
            [m1], ["*", "drank", "water"])
        assert results == []

    def test_length_mismatch_skips(self):
        """Tag with wrong length doesn't match template."""
        # 'alice_ate_apple' is 3 words; template is 4-word
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        results = mbc.query_sentence_template(
            [m1], ["*", "ate", "the", "apple"])
        assert results == []

    def test_4word_template(self):
        """4-word templates work (subj verb mod obj)."""
        m1 = self._make_member_with_tags(
            "set1", ["alice_ate_big_apple", "bob_ate_red_apple"])
        # what did *_ate_big_*: subject of 'ate big apple' (any)
        results = mbc.query_sentence_template(
            [m1], ["*", "ate", "big", "apple"])
        assert len(results) == 1
        assert results[0]["wildcards"] == ["alice"]

    def test_multiple_wildcards(self):
        """Multiple wildcards return all wildcard positions in order."""
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        # template ['*', 'ate', '*']: subject + object
        results = mbc.query_sentence_template([m1], ["*", "ate", "*"])
        assert len(results) == 1
        # First wildcard is alice (subject), second is apple (object)
        assert results[0]["wildcards"] == ["alice", "apple"]

    def test_no_wildcards_exact_match(self):
        """Template with no wildcards is exact-match check."""
        m1 = self._make_member_with_tags("set1", ["alice_ate_apple"])
        results = mbc.query_sentence_template(
            [m1], ["alice", "ate", "apple"])
        assert len(results) == 1
        assert results[0]["wildcards"] == []

    def test_empty_members_returns_empty(self):
        """No bridges -> no results."""
        results = mbc.query_sentence_template([], ["*", "ate", "apple"])
        assert results == []


class TestSetWrapperImportability:
    """The per-set patch modules can be imported without crashing."""

    @pytest.mark.parametrize("module", [
        "research.runners.concept_pool_demo_set2",
        "research.runners.concept_pool_demo_set3",
        "research.runners.concept_pool_demo_set4",
        "research.runners.concept_pool_demo_set5",
    ])
    def test_set_wrapper_imports(self, module):
        """Each set-wrapper module imports cleanly + monkey-patches vocab."""
        # Reload concept_pool_demo first to get clean state
        import research.runners.concept_pool_demo as cpd
        importlib.reload(cpd)
        # Save original vocab
        orig_noun = dict(cpd.NOUN_VOCAB)
        try:
            m = importlib.import_module(module)
            # Verify the patch happened: NOUN_VOCAB on cpd should no
            # longer match the original set1 (APPLE/RIVER/DOG/CAT)
            assert cpd.NOUN_VOCAB != orig_noun, (
                f"{module}: import did not patch NOUN_VOCAB"
            )
            assert len(cpd.NOUN_VOCAB) == 4, (
                f"{module}: NOUN_VOCAB has {len(cpd.NOUN_VOCAB)} entries, "
                f"expected 4"
            )
        finally:
            # Restore original to avoid cross-test contamination
            cpd.NOUN_VOCAB = orig_noun
            cpd.NOUN_NAMES = list(orig_noun.values())
