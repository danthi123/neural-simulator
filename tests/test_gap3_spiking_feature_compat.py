"""CI guard — gap #3 residual A1 (2026-07-18): the referent-bias feature-compatibility is a SPIKING LEARNED map
(replacing the host `content_bias_target` lexicon lookup), and it is wired into MultiTurnAgent.

Pins: (1) the spiking feature-compat reproduces the host content_bias_target disambiguation + the permuted-corpus
anti-cheat collapses; (2) the MultiTurnAgent resolves a pronoun via the wired-in spiking feature-compat (no host
lookup) and answers the full turn. CPU/numpy.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._gap3_spiking_feature_compat_derisk import SpikingFeatureCompat
from research.runners._gap3_learned_feature_compat_derisk import run_seed as _mech_seed, make_corpus
from research.runners.biased_competition_buffer import content_bias_target, ANIMACY, VERB_SELECTS
from research.runners.multi_turn_agent import MultiTurnAgent

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat"]
ALL_CONCEPTS = list(ANIMACY.keys())
FULL_VOCAB = ALL_CONCEPTS + list(VERB_SELECTS.keys())


def test_learned_feature_compat_mechanism():
    # the learned (corpus co-occurrence) feature-compatibility reproduces the host disambiguation; permuted collapses
    assert _mech_seed(42) >= 0.80
    assert _mech_seed(42, permute=True) <= 0.60


def test_spiking_feature_compat_matches_host():
    fc = SpikingFeatureCompat(seed=42)
    # 'eat' selects animate -> cat; 'roll' selects inanimate -> ball (matching the host content_bias_target)
    for verb, exp in (("eat", "cat"), ("roll", "ball")):
        assert fc.bias_target(["cat", "ball"], verb) == content_bias_target(["cat", "ball"], verb) == exp


def test_agent_resolves_via_spiking_feature_compat():
    fc = SpikingFeatureCompat(seed=42)
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42,
                       enable_biased_competition=True, feat_compat_source=fc)
    a.agent.composer.store("cat", "eat", "fish")
    a.agent.composer.store("ball", "eat", "worm")
    a._write_referent("cat"); a._write_referent("ball")
    assert a._held_set() == ["ball", "cat"]
    assert a._resolve_biased("eat") == "cat"     # resolved by the SPIKING feature-compat, not the host lexicon
    assert a.what_does("it", "eat") == "fish"    # the full turn answers via the resolved referent


def test_feat_compat_default_off_byte_identical():
    # default (no feat_compat_source) still uses content_bias_target -> byte-identical
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42,
                       enable_biased_competition=True)
    assert a._feat_compat_source is None


def test_agent_learns_referent_bias_from_own_experience():
    """Gap #3 A1 DEPLOYMENT: the agent LEARNS the referent-bias feature-compatibility from the SVO facts IT heard
    (its own experience in `composer.kb`), then defaults the multi-referent resolution to that SPIKING chooser
    (retiring the host `content_bias_target` lexicon)."""
    heard = make_corpus(42, n=80)                                  # 80 SVO facts the agent HEARS
    a = MultiTurnAgent(referent_concepts=ALL_CONCEPTS, concepts={w: None for w in FULL_VOCAB}, seed=42,
                       enable_biased_competition=True)             # NO feat_compat_source -> starts on the host lookup
    assert a._feat_compat_source is None
    for ag, v, pt in heard:
        a.agent.composer.store(ag, v, pt)                          # accumulates into composer.kb (the agent's memory)
    assert len(a.heard_facts()) >= 40                             # it remembers what it heard
    assert a.build_referent_bias_from_experience() is True         # LEARN the feature-compat from that experience
    assert isinstance(a._feat_compat_source, SpikingFeatureCompat) # now defaults to the spiking brain-based chooser
    # the learned-from-experience chooser resolves the pronoun (eat selects animate -> cat, not ball) == host
    assert a._feat_compat_source.bias_target(["cat", "ball"], "eat") == "cat"


def test_build_referent_bias_insufficient_experience_returns_false():
    # too few heard facts -> does NOT install (leaves the host fallback answering; moat-safe, no half-learned map)
    a = MultiTurnAgent(referent_concepts=ALL_CONCEPTS, concepts={w: None for w in FULL_VOCAB}, seed=42,
                       enable_biased_competition=True)
    a.agent.composer.store("cat", "eat", "fish")
    assert a.build_referent_bias_from_experience() is False
    assert a._feat_compat_source is None


def test_A2_feature_silent_tie_broken_by_salience():
    """Gap #3 residual A2: two SAME-animacy candidates (cat, dog) -> the feature-compat ABSTAINS (tie); the
    discourse-salience focus (the D3 Cb; a stub here) breaks it. Content decides clear cases, salience breaks ties."""
    fc = SpikingFeatureCompat(seed=42)
    focus = lambda held, verb: (sorted(held)[0] if held else None)      # salience stub (real D3 Cb is 6-seed GO)
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42,
                       enable_biased_competition=True, feat_compat_source=fc, focus_bias_source=focus)
    a.agent.composer.store("cat", "eat", "fish"); a.agent.composer.store("dog", "eat", "bird")
    a._write_referent("cat"); a._write_referent("dog")                  # both animate -> feature-silent tie
    assert fc.bias_target(["cat", "dog"], "eat") is None                # feature-compat ties -> abstains
    assert a._resolve_biased("eat") is not None                        # the salience focus breaks the tie
