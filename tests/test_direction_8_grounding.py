"""Direction 8 grounding pin - mirror of test_direction_4_grounding.py
with V=128 per bridge (640 cross-bridge concepts total).

These tests pin the contracts the Direction 8 cross-bridge-on-bio_brain_-
regions runner MUST satisfy. They lock the V=128 per-bridge architecture
+ frozen 0.80 multi-seed bar + 5 categories x V=128 = 640 unique
cross-bridge concepts.

The grounding-pin pattern is the same disciplined pattern used by prior
project arcs (Task 0 of the (c) generative-replay TDD plan; Direction Q's
grounding pin; Direction 3's grounding pin; Direction 4's grounding pin).
The contracts are codified UP FRONT so the implementation cannot drift
silently from the design bar.

Bar UNCHANGED at 0.80 multi-seed (same as pillars n=93+, Directions
Q, 3, 4). 4 frozen thresholds: OB_MIN=0.80, OI_MIN=0.80, LOADS=[2,3,5],
MIN_SEEDS=3.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_8_vocab_spec_module_exists():
    """Task 1: the 5-category vocab spec module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_vocab_spec.py",
    )
    assert os.path.exists(path), (
        "Task 1 not yet landed: " + path + " does not exist"
    )


def test_direction_8_bridge_builder_module_exists():
    """Task 2: the per-bridge builder wrapper exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_bridge_builder.py",
    )
    assert os.path.exists(path), (
        "Task 2 not yet landed: " + path + " does not exist"
    )


def test_direction_8_verdict_module_exists():
    """Task 3: the verdict module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_8_cross_bridge_probe_module_exists():
    """Task 4: cross-bridge probe runner exists (CPU-only scaffold).

    SKIP if Task 4 hasn't been added yet.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_cross_bridge_probe.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet (CPU-only probe scaffold)")


def test_direction_8_vocab_spec_has_5_categories_v128_each():
    """Task 1: vocab spec exposes 5 category lists, each V=128 = 640 unique
    cross-bridge concepts. Pre-registered design: nouns (64) + verbs (64)
    + adjectives (64) + spatial (64) + functional (64) = 640 distinct
    cross-bridge concepts. Vocab matches Direction M G.20 sparse
    production deliverable byte-for-byte.

    Global uniqueness enforced - no word appears in more than one bridge.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_vocab_spec.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 1 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_8_vocab_spec", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pre-registered category constants must exist
    assert hasattr(mod, "DIRECTION_8_NOUN_VOCAB"), (
        "vocab spec missing DIRECTION_8_NOUN_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_8_VERB_VOCAB"), (
        "vocab spec missing DIRECTION_8_VERB_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_8_ADJECTIVE_VOCAB"), (
        "vocab spec missing DIRECTION_8_ADJECTIVE_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_8_SPATIAL_VOCAB"), (
        "vocab spec missing DIRECTION_8_SPATIAL_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_8_FUNCTIONAL_VOCAB"), (
        "vocab spec missing DIRECTION_8_FUNCTIONAL_VOCAB"
    )
    # Pre-registered V=128 per category
    assert len(mod.DIRECTION_8_NOUN_VOCAB) == 128, (
        "DIRECTION_8_NOUN_VOCAB must be exactly 128"
    )
    assert len(mod.DIRECTION_8_VERB_VOCAB) == 128, (
        "DIRECTION_8_VERB_VOCAB must be exactly 128"
    )
    assert len(mod.DIRECTION_8_ADJECTIVE_VOCAB) == 128, (
        "DIRECTION_8_ADJECTIVE_VOCAB must be exactly 128"
    )
    assert len(mod.DIRECTION_8_SPATIAL_VOCAB) == 128, (
        "DIRECTION_8_SPATIAL_VOCAB must be exactly 128"
    )
    assert len(mod.DIRECTION_8_FUNCTIONAL_VOCAB) == 128, (
        "DIRECTION_8_FUNCTIONAL_VOCAB must be exactly 128"
    )
    # Pre-registered total = 640
    total = (
        len(mod.DIRECTION_8_NOUN_VOCAB)
        + len(mod.DIRECTION_8_VERB_VOCAB)
        + len(mod.DIRECTION_8_ADJECTIVE_VOCAB)
        + len(mod.DIRECTION_8_SPATIAL_VOCAB)
        + len(mod.DIRECTION_8_FUNCTIONAL_VOCAB)
    )
    assert total == 640, (
        "5-category V=128 spec must total exactly 640 concepts; got "
        + str(total)
    )
    # Global uniqueness - no word appears in two bridges
    all_words = (
        list(mod.DIRECTION_8_NOUN_VOCAB)
        + list(mod.DIRECTION_8_VERB_VOCAB)
        + list(mod.DIRECTION_8_ADJECTIVE_VOCAB)
        + list(mod.DIRECTION_8_SPATIAL_VOCAB)
        + list(mod.DIRECTION_8_FUNCTIONAL_VOCAB)
    )
    assert len(set(all_words)) == 640, (
        "Direction 8 vocab spec has duplicate words across bridges; "
        "expected 640 unique; got " + str(len(set(all_words)))
    )
    # Pre-registered total constant exists
    assert hasattr(mod, "DIRECTION_8_TOTAL"), (
        "vocab spec missing DIRECTION_8_TOTAL constant"
    )
    assert mod.DIRECTION_8_TOTAL == 640, (
        "DIRECTION_8_TOTAL tampered: design fixes this at 640"
    )


def test_direction_8_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present and match design-doc
    bar (0.80 multi-seed; same as pillars n=93+ + Directions Q, 3, 4)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_8_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must exist)
    assert hasattr(mod, "_DIRECTION_8_OB_MIN"), (
        "verdict module missing _DIRECTION_8_OB_MIN"
    )
    assert hasattr(mod, "_DIRECTION_8_OI_MIN"), (
        "verdict module missing _DIRECTION_8_OI_MIN"
    )
    assert hasattr(mod, "_DIRECTION_8_LOADS"), (
        "verdict module missing _DIRECTION_8_LOADS"
    )
    assert hasattr(mod, "_DIRECTION_8_MIN_SEEDS"), (
        "verdict module missing _DIRECTION_8_MIN_SEEDS"
    )
    # Pre-registered values (tampering caught here)
    assert mod._DIRECTION_8_OB_MIN == 0.80, (
        "_DIRECTION_8_OB_MIN tampered: design fixes this at 0.80"
    )
    assert mod._DIRECTION_8_OI_MIN == 0.80, (
        "_DIRECTION_8_OI_MIN tampered: design fixes this at 0.80"
    )
    assert list(mod._DIRECTION_8_LOADS) == [2, 3, 5], (
        "_DIRECTION_8_LOADS tampered: design fixes this at [2, 3, 5]"
    )
    assert mod._DIRECTION_8_MIN_SEEDS == 3, (
        "_DIRECTION_8_MIN_SEEDS tampered: design fixes this at 3"
    )


def test_direction_8_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID_MALFORMED (instrument-
    validity failure) from PASS / PARTIAL / NEGATIVE. Same discipline
    pattern as Direction Q + Direction 3 + Direction 4."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert (
        "VOID" in src or "void" in src
    ), (
        "verdict module must include VOID_MALFORMED branch for instrument-"
        "validity failure case"
    )


def test_direction_8_verdict_compute_verdict_callable():
    """Task 3: verdict module exposes compute_verdict callable returning a
    DIRECTION_8_* tag string."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_8_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compute_verdict"), (
        "verdict module missing compute_verdict() callable"
    )
    # Sanity: VOID branch reachable with empty input
    result = mod.compute_verdict(None)
    assert result == mod.DIRECTION_8_VOID_MALFORMED, (
        "compute_verdict(None) must return DIRECTION_8_VOID_MALFORMED"
    )
    # Below MIN_SEEDS is VOID
    result = mod.compute_verdict([])
    assert result == mod.DIRECTION_8_VOID_MALFORMED, (
        "compute_verdict([]) must return DIRECTION_8_VOID_MALFORMED"
    )


def test_direction_8_bridge_builder_has_five_functions():
    """Task 2: per-bridge builder wrapper exposes 5 functions, one per
    bridge (A_nouns / B_verbs / C_adj / D_spatial / E_functional). Each
    builds a fresh SimulationBridge for that bridge's V=128 category."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "def build_direction_8_bridge_A_nouns" in src, (
        "bridge_builder must define build_direction_8_bridge_A_nouns"
    )
    assert "def build_direction_8_bridge_B_verbs" in src, (
        "bridge_builder must define build_direction_8_bridge_B_verbs"
    )
    assert "def build_direction_8_bridge_C_adj" in src, (
        "bridge_builder must define build_direction_8_bridge_C_adj"
    )
    assert "def build_direction_8_bridge_D_spatial" in src, (
        "bridge_builder must define build_direction_8_bridge_D_spatial"
    )
    assert "def build_direction_8_bridge_E_functional" in src, (
        "bridge_builder must define build_direction_8_bridge_E_functional"
    )


def test_direction_8_bridge_builder_uses_protected_builder_byte_unchanged():
    """Task 2: per-bridge builder must REUSE build_biological_brain_regions
    byte-unchanged (the protected builder). The wrapper loads each
    category's V=128 vocab via the existing noun_pool_names /
    verb_pool_names / adjective_pool_names parameters; the builder itself
    is NOT modified."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "build_biological_brain_regions" in src, (
        "bridge_builder must reuse build_biological_brain_regions (the "
        "protected builder) via import; no other path acceptable"
    )


def test_direction_8_bridge_builder_has_seed_offsets():
    """Task 2: per-bridge builder defines per-bridge seed offsets (analog
    of D5 c4e18f2 + D4 fix) - mandatory to avoid byte-identical weight
    initialization across the 5 bridges."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_8_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "_DIRECTION_8_BRIDGE_LABEL_SEED_OFFSETS" in src, (
        "bridge_builder must define _DIRECTION_8_BRIDGE_LABEL_SEED_OFFSETS "
        "(analog of D4 fix); each of the 5 bridges needs a unique seed "
        "offset to avoid byte-identical weight init across bridges"
    )
    # 100k spacing per the discipline binding
    assert "100000" in src, (
        "bridge_builder seed offsets must use 100k spacing (per D5 + D4 fix)"
    )
