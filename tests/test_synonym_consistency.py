"""Regression tests for synonym vocab consistency across modules.

Two modules currently maintain separate copies of SYNONYM_GROUPS_8/12/16:
- research/runners/text_eval.py: canonical for evaluate_word_to_action
- research/runners/consolidation_synonym_trainer.py: used by training loop

These MUST stay in sync. If they drift, training and eval would use
different word-to-action mappings, producing nonsense results that
might not be obviously broken.

This test file catches drift early.
"""
from __future__ import annotations

import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use importlib to avoid stale-text matching false positives
_te = importlib.import_module("research.runners.text_eval")
_cst = importlib.import_module("research.runners.consolidation_synonym_trainer")


@pytest.mark.parametrize("vocab_size", [8, 12, 16])
def test_synonym_groups_consistent_across_modules(vocab_size):
    """SYNONYM_GROUPS_N must agree exactly between text_eval and trainer."""
    if vocab_size == 16:
        a = _te.SYNONYM_GROUPS_16
        b = _cst.SYNONYM_GROUPS_16
    elif vocab_size == 12:
        a = _te.SYNONYM_GROUPS_12
        b = _cst.SYNONYM_GROUPS_12
    else:
        a = _te.SYNONYM_GROUPS  # canonical 8-word
        b = _cst.SYNONYM_GROUPS_8
    assert a == b, (
        f"SYNONYM_GROUPS_{vocab_size} drift between text_eval "
        f"and consolidation_synonym_trainer:\n"
        f"  text_eval:  {a}\n"
        f"  cst:        {b}"
    )


@pytest.mark.parametrize("vocab_size", [8, 12, 16])
def test_synonym_groups_have_4_actions(vocab_size):
    """Every vocab size must have exactly 4 action keys (N/E/S/W)."""
    sg = _te.get_synonym_groups(vocab_size)
    assert set(sg.keys()) == {"N", "E", "S", "W"}


@pytest.mark.parametrize("vocab_size,expected_per_action", [
    (8, 2), (12, 3), (16, 4),
])
def test_synonyms_per_action_count(vocab_size, expected_per_action):
    """Each action has the expected number of synonyms."""
    sg = _te.get_synonym_groups(vocab_size)
    for a, words in sg.items():
        assert len(words) == expected_per_action, \
            f"Action {a} has {len(words)} words at vocab={vocab_size}, " \
            f"expected {expected_per_action}"


def test_primary_word_is_first_in_each_group():
    """Convention: synonym_groups[a][0] is the canonical "primary" word.
    Tests downstream depend on this ordering."""
    canonical_primaries = {"N": "north", "E": "east", "S": "south", "W": "west"}
    for vocab_size in [8, 12, 16]:
        sg = _te.get_synonym_groups(vocab_size)
        for a, expected in canonical_primaries.items():
            assert sg[a][0] == expected, \
                f"vocab={vocab_size} action={a}: first synonym is " \
                f"{sg[a][0]!r}, expected {expected!r}"


def test_extended_word_to_action_round_trip():
    """For each vocab size, get_extended_word_to_action(N) maps every
    word in get_synonym_groups(N) to the right action."""
    for vocab_size in [8, 12, 16]:
        sg = _te.get_synonym_groups(vocab_size)
        eta = _te.get_extended_word_to_action(vocab_size)
        for action, words in sg.items():
            for w in words:
                assert eta[w] == action, \
                    f"vocab={vocab_size}: word {w!r} should map to " \
                    f"{action!r}, got {eta[w]!r}"
