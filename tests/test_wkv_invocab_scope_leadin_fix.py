"""Regression pin for the `in_vocab_scope` lead-in-phrase loophole (found + fixed 2026-09-01).

WHAT BROKE. `webapp/wkv_mouth_generator.py::in_vocab_scope`'s `min_content_hits` check counted ANY vocab hit
outside `_FUNCTION_WORDS` as genuine domain content -- including the caller's OWN conversational lead-in
template ("tell me about", "what do you know about", ..., the same phrases
`webapp.open_ended_chat._LEADINS` strips before topic extraction). Several of those template words
("tell"/"me"/"about"/"know"/"think"/"describe"/"explain"/"an"/"what's"/"who's") sit in this checkpoint's
V=1000 vocabulary AND were not `_FUNCTION_WORDS`, so a FIXED lead-in phrase alone satisfied
`min_content_hits=2` regardless of the actual topic -- `in_vocab_scope("tell me about " + <any nonsense>)`
measured True 68.17% of the time on a real `wikidata_core_15k` sample (`research/findings/2026-09-01-wkv-
mouth-fact-grounding-lever.md` Part 2; `research/FAILURE_LOG.md` 2026-09-01 entry).

THE FIX. A new `_LEADIN_WORDS` set (every word appearing across `_LEADINS`'s phrases) is now ALSO excluded
from `content_hits`, alongside `_FUNCTION_WORDS`. `min_hits`/`min_frac` are unchanged (still scored over the
full original text) -- only which hits count as CONTENT changed, so a genuinely content-bearing message is
not penalized for carrying a lead-in.

FULL stratified real-store verification (n=600, catch-rate + no-regression by genuine-content bucket) lives
in `research/runners/_wkv_invocab_scope_leadin_fix_verify.py` /
`research/findings/raw/_wkv_invocab_scope_leadin_fix_verify.json` (bucket_0 old=63.45% -> new=0.0%,
bucket_2plus stays 100% -> 100%) -- this file is the CHEAP, ALWAYS-RUN regression pin; that runner is the
(slower, real-data) evidence-gathering companion, per the same split `test_wkv_mouth_learned_head_path.py`
uses for the learned-head rung.
"""
from __future__ import annotations

import pytest

from webapp import wkv_mouth_generator as WKV

SEED = 42   # cheap: loads one ~1.4 MB checkpoint, no store/bundle needed for these cases


# The exact documented loophole words this fix must exclude from content-hit scoring -- if any of these ever
# drop out of `_LEADIN_WORDS`, the specific bug this test class exists to pin can return for that word.
_MUST_BE_EXCLUDED = {"tell", "me", "about", "know", "think", "describe", "explain", "an", "what's", "who's"}


def test_leadin_words_constant_covers_the_documented_loophole():
    missing = _MUST_BE_EXCLUDED - WKV._LEADIN_WORDS
    assert not missing, f"_LEADIN_WORDS dropped documented loophole word(s): {missing}"


@pytest.mark.parametrize("phrase", [
    "tell me about", "what can you tell me about", "what do you know about",
    "what do you think about", "can you tell me about", "do you know anything about",
    "do you know about",
])
@pytest.mark.parametrize("tail", ["zzznonsenseword qqqgibberish", "xkcdplaceholder blorptastic"])
def test_leadin_phrase_alone_no_longer_passes_on_nonsense(phrase, tail):
    """THE regression pin: a recognized lead-in phrase followed by total nonsense must read False -- before
    this fix it read True for every one of these (phrase, tail) pairs."""
    msg = f"{phrase} {tail}"
    assert WKV.in_vocab_scope(msg, seed=SEED) is False, (
        f"lead-in-phrase loophole regressed: {msg!r} passed in_vocab_scope again")


def test_genuine_multiword_content_still_passes_with_a_leadin():
    """No-regression: a real TinyStories-register sentence carrying BOTH a recognized lead-in AND genuine
    multi-word domain content must still pass -- the fix must not penalize legitimate content-bearing
    messages just because they also happen to start with a lead-in phrase."""
    msg = "tell me about the big dog and the little cat and their ball"
    assert WKV.in_vocab_scope(msg, seed=SEED) is True


def test_genuine_multiword_content_still_passes_without_a_leadin():
    """No-regression, isolating the lead-in variable: the SAME kind of genuine content, with no recognized
    lead-in prefix at all, is unaffected by this fix (it never touched this code path)."""
    msg = "once upon a time there was a little boy named tim who had a dog and a ball"
    assert WKV.in_vocab_scope(msg, seed=SEED) is True


def test_fact_grounding_ids_unaffected_by_the_leadin_exclusion():
    """Precision check: the fix is scoped to `in_vocab_scope` only. `fact_grounding_ids` (a different
    function, used by the fact-grounding lever to pull content words OUT of retrieved facts, not to gate a
    user message) still excludes only `_FUNCTION_WORDS` -- a fact triple legitimately containing one of the
    `_LEADIN_WORDS` tokens (e.g. an action/patient literally being "think" or "know") must not be silently
    dropped by a fix that was never meant to touch this function."""
    triples = [("agent_x", "action_y", "think")]
    ids = WKV.fact_grounding_ids(triples, seed=SEED)
    _, vocab, word_to_id = WKV._get_readout(SEED)
    if "think" in vocab:
        assert word_to_id["think"] in ids, (
            "fact_grounding_ids must not exclude a genuine fact content word just because it's a "
            "_LEADIN_WORDS member -- that exclusion belongs to in_vocab_scope only")


def test_single_content_word_topic_is_an_honest_residual_not_silently_hidden():
    """Documents the fix's known limit rather than hiding it: a topic carrying exactly ONE genuine in-vocab
    content word (e.g. a single real TinyStories word as the whole topic) cannot reach `min_content_hits=2`
    from that word alone, so it correctly fails post-fix -- it was only ever passing before via the lead-in
    loophole, never via genuine 2-word support. Uses a checkpoint-vocabulary word verified present so this
    assertion is not vacuous."""
    _, vocab, _ = WKV._get_readout(SEED)
    single_word = next((w for w in vocab if w.isalpha() and w not in WKV._FUNCTION_WORDS
                        and w not in WKV._LEADIN_WORDS), None)
    assert single_word is not None, "checkpoint vocab has no non-function content word to test with"
    msg = f"tell me about {single_word}"
    assert WKV.in_vocab_scope(msg, seed=SEED) is False
