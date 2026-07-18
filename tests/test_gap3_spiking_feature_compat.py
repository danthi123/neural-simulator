"""CI guard — gap #3 residual A1 (2026-07-18): the referent-bias feature-compatibility is a SPIKING LEARNED map
(replacing the host `content_bias_target` lexicon lookup), and it is wired into MultiTurnAgent.

Pins: (1) the spiking feature-compat reproduces the host content_bias_target disambiguation + the permuted-corpus
anti-cheat collapses; (2) the MultiTurnAgent resolves a pronoun via the wired-in spiking feature-compat (no host
lookup) and answers the full turn. CPU/numpy.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._gap3_spiking_feature_compat_derisk import SpikingFeatureCompat
from research.runners._gap3_learned_feature_compat_derisk import run_seed as _mech_seed
from research.runners.biased_competition_buffer import content_bias_target
from research.runners.multi_turn_agent import MultiTurnAgent

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat"]


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
