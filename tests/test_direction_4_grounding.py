"""Direction 4 grounding pin - intentionally RED until Tasks 1-3 land.

These tests pin the contracts the Direction 4 cross-bridge-on-bio_brain_-
regions runner MUST satisfy. They are RED on Task 0 commit; turn GREEN as
Tasks 1-3 land per docs/plans/2026-05-25-direction-4-cross-bridge-bio_-
brain_regions-implementation.md. Final tests keep the contract permanent.

The grounding-pin pattern is the same disciplined pattern used by prior
project arcs (Task 0 of the (c) generative-replay TDD plan; Direction Q's
grounding pin; Direction 3's grounding pin): the contracts are codified
UP FRONT so the implementation cannot drift silently from the design doc
bar.

Bar UNCHANGED at 0.80 multi-seed (same as pillars n=93+ and Directions
Q, 3). 4 frozen thresholds: OB_MIN=0.80, OI_MIN=0.80, LOADS=[2,3,5],
MIN_SEEDS=3.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_4_vocab_spec_module_exists():
    """Task 1: the 5-category vocab spec module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_vocab_spec.py",
    )
    assert os.path.exists(path), (
        "Task 1 not yet landed: " + path + " does not exist"
    )


def test_direction_4_bridge_builder_module_exists():
    """Task 2: the per-bridge builder wrapper exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_bridge_builder.py",
    )
    assert os.path.exists(path), (
        "Task 2 not yet landed: " + path + " does not exist"
    )


def test_direction_4_verdict_module_exists():
    """Task 3: the verdict module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_4_cross_bridge_probe_module_exists():
    """Task 4: cross-bridge probe runner exists (CPU-only scaffold).

    SKIP if Task 4 hasn't been added yet — Task 4 is in scope for
    this subagent only as a scaffold; the decisive run is Task 6
    controller-only after Task 5 trains.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_cross_bridge_probe.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet (CPU-only probe scaffold)")


def test_direction_4_vocab_spec_has_5_categories_v16_each():
    """Task 1: vocab spec exposes 5 category lists, each V=16 = 80 unique
    cross-bridge concepts. Pre-registered design (per design doc Approach
    A): nouns (16) + verbs (16) + adjectives (16) + spatial (16) +
    functional (16) = 80 distinct cross-bridge concepts.

    Global uniqueness enforced — no word appears in more than one bridge.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_vocab_spec.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 1 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_4_vocab_spec", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pre-registered category constants must exist
    assert hasattr(mod, "DIRECTION_4_NOUN_VOCAB"), (
        "vocab spec missing DIRECTION_4_NOUN_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_4_VERB_VOCAB"), (
        "vocab spec missing DIRECTION_4_VERB_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_4_ADJECTIVE_VOCAB"), (
        "vocab spec missing DIRECTION_4_ADJECTIVE_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_4_SPATIAL_VOCAB"), (
        "vocab spec missing DIRECTION_4_SPATIAL_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_4_FUNCTIONAL_VOCAB"), (
        "vocab spec missing DIRECTION_4_FUNCTIONAL_VOCAB"
    )
    # Pre-registered V=16 per category
    assert len(mod.DIRECTION_4_NOUN_VOCAB) == 16, (
        "DIRECTION_4_NOUN_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_4_VERB_VOCAB) == 16, (
        "DIRECTION_4_VERB_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_4_ADJECTIVE_VOCAB) == 16, (
        "DIRECTION_4_ADJECTIVE_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_4_SPATIAL_VOCAB) == 16, (
        "DIRECTION_4_SPATIAL_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_4_FUNCTIONAL_VOCAB) == 16, (
        "DIRECTION_4_FUNCTIONAL_VOCAB must be exactly 16"
    )
    # Pre-registered total = 80
    total = (
        len(mod.DIRECTION_4_NOUN_VOCAB)
        + len(mod.DIRECTION_4_VERB_VOCAB)
        + len(mod.DIRECTION_4_ADJECTIVE_VOCAB)
        + len(mod.DIRECTION_4_SPATIAL_VOCAB)
        + len(mod.DIRECTION_4_FUNCTIONAL_VOCAB)
    )
    assert total == 80, (
        "5-category V=16 spec must total exactly 80 concepts; got "
        + str(total)
    )
    # Global uniqueness — no word appears in two bridges
    all_words = (
        list(mod.DIRECTION_4_NOUN_VOCAB)
        + list(mod.DIRECTION_4_VERB_VOCAB)
        + list(mod.DIRECTION_4_ADJECTIVE_VOCAB)
        + list(mod.DIRECTION_4_SPATIAL_VOCAB)
        + list(mod.DIRECTION_4_FUNCTIONAL_VOCAB)
    )
    assert len(set(all_words)) == 80, (
        "Direction 4 vocab spec has duplicate words across bridges; "
        "expected 80 unique; got " + str(len(set(all_words)))
    )


def test_direction_4_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present and match design-doc
    bar (0.80 multi-seed; same as pillars n=93+ + Directions Q, 3)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_4_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must exist)
    assert hasattr(mod, "_DIRECTION_4_OB_MIN"), (
        "verdict module missing _DIRECTION_4_OB_MIN"
    )
    assert hasattr(mod, "_DIRECTION_4_OI_MIN"), (
        "verdict module missing _DIRECTION_4_OI_MIN"
    )
    assert hasattr(mod, "_DIRECTION_4_LOADS"), (
        "verdict module missing _DIRECTION_4_LOADS"
    )
    assert hasattr(mod, "_DIRECTION_4_MIN_SEEDS"), (
        "verdict module missing _DIRECTION_4_MIN_SEEDS"
    )
    # Pre-registered values (tampering caught here)
    assert mod._DIRECTION_4_OB_MIN == 0.80, (
        "_DIRECTION_4_OB_MIN tampered: design fixes this at 0.80"
    )
    assert mod._DIRECTION_4_OI_MIN == 0.80, (
        "_DIRECTION_4_OI_MIN tampered: design fixes this at 0.80"
    )
    assert list(mod._DIRECTION_4_LOADS) == [2, 3, 5], (
        "_DIRECTION_4_LOADS tampered: design fixes this at [2, 3, 5]"
    )
    assert mod._DIRECTION_4_MIN_SEEDS == 3, (
        "_DIRECTION_4_MIN_SEEDS tampered: design fixes this at 3"
    )


def test_direction_4_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID_MALFORMED (instrument-
    validity failure) from PASS / PARTIAL / NEGATIVE. Same discipline
    pattern as Direction Q + Direction 3."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_verdict.py",
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


def test_direction_4_bridge_builder_has_five_functions():
    """Task 2: per-bridge builder wrapper exposes 5 functions, one per
    bridge (A_nouns / B_verbs / C_adj / D_spatial / E_functional). Each
    builds a fresh SimulationBridge for that bridge's V=16 category."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "def build_direction_4_bridge_A_nouns" in src, (
        "bridge_builder must define build_direction_4_bridge_A_nouns"
    )
    assert "def build_direction_4_bridge_B_verbs" in src, (
        "bridge_builder must define build_direction_4_bridge_B_verbs"
    )
    assert "def build_direction_4_bridge_C_adj" in src, (
        "bridge_builder must define build_direction_4_bridge_C_adj"
    )
    assert "def build_direction_4_bridge_D_spatial" in src, (
        "bridge_builder must define build_direction_4_bridge_D_spatial"
    )
    assert "def build_direction_4_bridge_E_functional" in src, (
        "bridge_builder must define build_direction_4_bridge_E_functional"
    )


def test_direction_4_bridge_builder_uses_protected_builder_byte_unchanged():
    """Task 2: per-bridge builder must REUSE build_biological_brain_regions
    byte-unchanged (the protected builder). The wrapper loads each
    category's V=16 vocab via the existing noun_pool_names /
    verb_pool_names / adjective_pool_names parameters; the builder itself
    is NOT modified."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "build_biological_brain_regions" in src, (
        "bridge_builder must reuse build_biological_brain_regions (the "
        "protected builder) via import; no other path acceptable"
    )
