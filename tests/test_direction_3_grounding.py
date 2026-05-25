"""Direction 3 grounding pin - intentionally RED until later tasks land.

These tests pin the contracts the Direction 3 V=32 vocab-scaling-on-bio_-
brain_regions runner MUST satisfy. They are RED on Task 0 commit; turn
GREEN as Tasks 1-4 land per docs/plans/2026-05-25-direction-3-vocab-
scaling-bio_brain_regions-design.md. Final tests keep the contract
permanent.

The grounding-pin pattern is the same disciplined pattern used by prior
project arcs (Task 0 of the (c) generative-replay TDD plan; Direction Q's
grounding pin): the contracts are codified UP FRONT so the implementation
cannot drift silently from the design doc bar.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_3_vocab_spec_module_exists():
    """Task 1: the V=32 vocab spec module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_vocab_spec.py",
    )
    assert os.path.exists(path), (
        "Task 1 not yet landed: " + path + " does not exist"
    )


def test_direction_3_bridge_builder_module_exists():
    """Task 2: the V=32 bridge builder wrapper exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_bridge_builder.py",
    )
    assert os.path.exists(path), (
        "Task 2 not yet landed: " + path + " does not exist"
    )


def test_direction_3_verdict_module_exists():
    """Task 3: the verdict module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_3_runner_module_exists():
    """Task 4: the V=32 multi-seed runner module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_v32_runner.py",
    )
    assert os.path.exists(path), (
        "Task 4 not yet landed: " + path + " does not exist"
    )


def test_direction_3_vocab_spec_has_v32_lists():
    """Task 1: vocab spec exposes V=32 word lists totalling 32 distinct
    concepts. Pre-registered design choice: 4 motor + 12 noun + 12 verb +
    4 adjective = 32 pools (one pool per concept; mirrors v14/v16 1-pool-
    per-concept architecture; total scale 2x V=16)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_vocab_spec.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 1 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_3_vocab_spec", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pre-registered names of the four vocab kinds
    assert hasattr(mod, "DIRECTION_3_MOTOR_VOCAB"), (
        "vocab spec missing DIRECTION_3_MOTOR_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_3_NOUN_VOCAB"), (
        "vocab spec missing DIRECTION_3_NOUN_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_3_VERB_VOCAB"), (
        "vocab spec missing DIRECTION_3_VERB_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_3_ADJECTIVE_VOCAB"), (
        "vocab spec missing DIRECTION_3_ADJECTIVE_VOCAB"
    )
    # Pre-registered V=32 layout: 4 + 12 + 12 + 4 = 32 distinct concepts
    assert len(mod.DIRECTION_3_MOTOR_VOCAB) == 4, (
        "DIRECTION_3_MOTOR_VOCAB must be exactly 4 (Tier 1 cardinal directions)"
    )
    assert len(mod.DIRECTION_3_NOUN_VOCAB) == 12, (
        "DIRECTION_3_NOUN_VOCAB must be exactly 12 (3x v14 nouns)"
    )
    assert len(mod.DIRECTION_3_VERB_VOCAB) == 12, (
        "DIRECTION_3_VERB_VOCAB must be exactly 12 (3x v14 verbs)"
    )
    assert len(mod.DIRECTION_3_ADJECTIVE_VOCAB) == 4, (
        "DIRECTION_3_ADJECTIVE_VOCAB must be exactly 4 (v14 baseline)"
    )
    # Total = 32
    total = (
        len(mod.DIRECTION_3_MOTOR_VOCAB)
        + len(mod.DIRECTION_3_NOUN_VOCAB)
        + len(mod.DIRECTION_3_VERB_VOCAB)
        + len(mod.DIRECTION_3_ADJECTIVE_VOCAB)
    )
    assert total == 32, (
        "V=32 spec must total exactly 32 concepts; got " + str(total)
    )
    # Global uniqueness assertion - no word appears in two kinds
    all_words = (
        list(mod.DIRECTION_3_MOTOR_VOCAB)
        + list(mod.DIRECTION_3_NOUN_VOCAB)
        + list(mod.DIRECTION_3_VERB_VOCAB)
        + list(mod.DIRECTION_3_ADJECTIVE_VOCAB)
    )
    assert len(set(all_words)) == 32, (
        "V=32 spec has duplicate words across kinds; expected 32 unique"
    )


def test_direction_3_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present and match design doc
    bar (0.80 multi-seed; same as pillars n=93+ + Direction Q)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_3_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must be present)
    assert hasattr(mod, "_DIRECTION_3_V32_OB_MIN"), (
        "verdict module missing _DIRECTION_3_V32_OB_MIN"
    )
    assert hasattr(mod, "_DIRECTION_3_V32_OI_MIN"), (
        "verdict module missing _DIRECTION_3_V32_OI_MIN"
    )
    assert hasattr(mod, "_DIRECTION_3_V32_LOADS"), (
        "verdict module missing _DIRECTION_3_V32_LOADS"
    )
    assert hasattr(mod, "_DIRECTION_3_V32_MIN_SEEDS"), (
        "verdict module missing _DIRECTION_3_V32_MIN_SEEDS"
    )
    # Pre-registered values (must equal design doc; tampering caught here)
    assert mod._DIRECTION_3_V32_OB_MIN == 0.80, (
        "_DIRECTION_3_V32_OB_MIN tampered: design fixes this at 0.80"
    )
    assert mod._DIRECTION_3_V32_OI_MIN == 0.80, (
        "_DIRECTION_3_V32_OI_MIN tampered: design fixes this at 0.80"
    )
    assert list(mod._DIRECTION_3_V32_LOADS) == [2, 3, 5], (
        "_DIRECTION_3_V32_LOADS tampered: design fixes this at [2, 3, 5]"
    )
    assert mod._DIRECTION_3_V32_MIN_SEEDS == 3, (
        "_DIRECTION_3_V32_MIN_SEEDS tampered: design fixes this at 3"
    )


def test_direction_3_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID (malformed input)
    from PASS / PARTIAL / NEGATIVE. Same discipline pattern as Direction Q."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert (
        "VOID" in src or "void" in src
    ), (
        "verdict module must include VOID branch for malformed input case"
    )


def test_direction_3_runner_uses_protected_builder_byte_unchanged():
    """Task 4: the runner must REUSE build_biological_brain_regions
    byte-unchanged (the protected builder). The wrapper extends the vocab
    via the noun_pool_names / verb_pool_names parameters; the builder
    itself is NOT modified."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_3_v32_runner.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # The runner must use either the bridge_builder wrapper OR the
    # existing build_concept_bridge (both of which call the protected
    # builder via reuse-by-import).
    assert (
        "direction_3_bridge_builder" in src
        or "build_concept_bridge" in src
        or "build_biological_brain_regions" in src
    ), (
        "runner must call the V=32 bridge builder or reuse the validated"
        " build_concept_bridge / build_biological_brain_regions; no other"
        " path acceptable"
    )
