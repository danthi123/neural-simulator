"""CI for EMERGE-62 -- DISCOVER the closed-class function-word set from distributional statistics (frequency +
context-coverage, the Goldilocks signature; Yang-Getz 2026 / Redington / Dominey-Hinaut; catalog G.12 Broca), then
feed the discovered set into the EMERGE-59 spiking-Broca frames.

CPU/numpy, offline. Small-stream smoke of the discovery + the input-destruction controls + the producer-render + moat.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class, _prf, render_on_discovered,
    GROUND_TRUTH_CLOSED, FRAME_FUNCTION_WORDS, TF_PCT, TC_PCT,
)


def _stats(seed, n=9000):
    tokens = build_stream(seed, n_sentences=n)
    return tokens, compute_stats(tokens)


def test_stream_has_content_and_function_words():
    """The controlled stream contains BOTH content words AND function words (so the closed/open split is a real
    discovery problem, not a trivial partition)."""
    tokens, (words, freq, cover, content) = _stats(42, n=4000)
    vocab = set(words)
    assert FRAME_FUNCTION_WORDS  # {the,can,does,not}
    assert all(fw in vocab for fw in FRAME_FUNCTION_WORDS)     # function words present
    assert "owl" in vocab and "fly" in vocab                    # content words present
    # the ground-truth closed class is a small MINORITY of the vocab (else discovery is trivial)
    gt = GROUND_TRUTH_CLOSED & vocab
    assert 5 <= len(gt) <= len(vocab) // 2


def test_discovery_recovers_frame_function_words_and_excludes_content():
    """The discovery rule recovers ALL frame function words (R on the frame set == 1.0) and excludes the clear content
    words (owl/fly/trout are NOT discovered)."""
    tokens, (words, freq, cover, content) = _stats(42)
    disc, pred, fp, cp = discover_closed_class(words, freq, cover)
    for fw in FRAME_FUNCTION_WORDS:
        assert fw in disc, f"frame function word {fw!r} not discovered"
    for cw in ("owl", "fly", "trout", "pond"):
        if cw in words:
            assert cw not in disc, f"content word {cw!r} wrongly discovered as closed-class"
    P, R, F1 = _prf(disc, GROUND_TRUTH_CLOSED & set(words))
    assert R == pytest.approx(1.0)         # all ground-truth closed class recovered
    assert F1 >= 0.70                       # clearly better than chance


def test_frequency_shuffle_control_collapses():
    """FREQUENCY-SHUFFLE (permute the statistic<->identity mapping) destroys the signal -> discovery collapses far
    below the main F1 (the load-bearing input-destruction anti-cheat)."""
    tokens, (words, freq, cover, content) = _stats(42)
    disc, _, _, _ = discover_closed_class(words, freq, cover)
    gt = GROUND_TRUTH_CLOSED & set(words)
    _, _, F1_main = _prf(disc, gt)
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(words))
    disc_shuf, _, _, _ = discover_closed_class(words, freq[perm], cover[perm])
    _, _, F1_shuf = _prf(disc_shuf, gt)
    assert F1_main >= F1_shuf + 0.30, f"shuffle did not collapse (main {F1_main:.3f} vs shuffle {F1_shuf:.3f})"


def test_no_stream_control_yields_empty_set():
    """NO-STREAM (empty stream) -> no statistics -> no discovery (empty set)."""
    words, freq, cover, content = compute_stats([])
    disc, _, _, _ = discover_closed_class(words, freq, cover)
    assert disc == set()


def test_heldout_word_classified_by_own_stats():
    """A function word (does) and a content word (trout) WITHHELD from the threshold-fitting slice are still classified
    correctly by their OWN stats vs frozen thresholds (generalization, not memorization)."""
    import math
    tokens, (words, freq, cover, content) = _stats(42)
    assert "does" in words and "trout" in words
    keep = [w for w in words if w not in ("does", "trout")]
    ki = [words.index(w) for w in keep]
    logfk = np.log(freq[ki])
    ck = cover[ki]

    def classify(w):
        i = words.index(w)
        pf = float((logfk < math.log(freq[i])).mean())
        pc = float((ck < cover[i]).mean())
        return (pf >= TF_PCT) and (pc >= TC_PCT)

    assert classify("does") is True          # held-out function word -> closed
    assert classify("trout") is False        # held-out content word -> open


def test_producer_renders_on_discovered_set_and_moat_holds():
    """The DISCOVERED set feeds the EMERGE-59 frames: held-out facts render correctly on the discovered function words,
    and the gate-first no-confab moat holds (0 producer invocations on abstains)."""
    tokens, (words, freq, cover, content) = _stats(42)
    disc, _, _, _ = discover_closed_class(words, freq, cover)
    facts = [{"subject": "owl", "ability_verb": "fly", "intr_verb": "walks"},
             {"subject": "penguin", "ability_verb": "fly", "intr_verb": "walks"}]
    render_ok, moat_calls, answer_produced, frame_covered = render_on_discovered(42, disc, facts)
    assert frame_covered is True
    assert render_ok >= 0.99                 # renders correctly on the discovered function words
    assert moat_calls == 0                    # abstain never invokes the producer (moat)
    assert answer_produced is True            # an answer DOES invoke it (counter meaningful)


def test_missing_function_word_breaks_render():
    """If a required function word is NOT in the discovered set, the render is degraded (the discovery is load-bearing
    for the frames -- not a host insertion)."""
    tokens, (words, freq, cover, content) = _stats(42)
    disc, _, _, _ = discover_closed_class(words, freq, cover)
    # drop a frame function word from the discovered set -> its frame slot cannot be filled -> render degrades
    disc_missing = set(disc) - {"can"}
    facts = [{"subject": "owl", "ability_verb": "fly", "intr_verb": "walks"}]
    render_ok, _, _, frame_covered = render_on_discovered(42, disc_missing, facts)
    assert frame_covered is False
    assert render_ok < 1.0                    # the modal frame can no longer render "the owl can fly"
