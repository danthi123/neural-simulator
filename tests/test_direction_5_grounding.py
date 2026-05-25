"""Direction 5 grounding pin - intentionally RED until Tasks 1-4 land.

These tests pin the contracts the Direction 5 HYBRID sparse-distributed
shared pool on bio_brain_regions runner MUST satisfy. They are RED on
Task 0 commit; turn GREEN as Tasks 1-3 land per
docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-implementation.md.
Final tests keep the contract permanent.

The grounding-pin pattern is the same disciplined pattern used by prior
project arcs (Task 0 of the (c) generative-replay TDD plan; Direction Q's
grounding pin; Direction 3's grounding pin; Direction 4's grounding pin):
the contracts are codified UP FRONT so the implementation cannot drift
silently from the design doc bar.

Bar UNCHANGED at 0.80 multi-seed (same as pillars n=93+ and Directions
Q, 3, 4). 4 frozen thresholds: OB_MIN=0.80, OI_MIN=0.80, LOADS=[2,3,5],
MIN_SEEDS=3.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_5_vocab_spec_module_exists():
    """Task 1: the 5-category vocab spec module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_vocab_spec.py",
    )
    assert os.path.exists(path), (
        "Task 1 not yet landed: " + path + " does not exist"
    )


def test_direction_5_bridge_builder_module_exists():
    """Task 2: the per-bridge HYBRID builder wrapper exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_bridge_builder.py",
    )
    assert os.path.exists(path), (
        "Task 2 not yet landed: " + path + " does not exist"
    )


def test_direction_5_verdict_module_exists():
    """Task 3: the verdict module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_5_cross_bridge_probe_module_exists():
    """Task 4: cross-bridge probe runner exists (CPU-only scaffold).

    SKIP if Task 4 hasn't been added yet - Task 4 is NOT in scope for
    THIS subagent (Tasks 0-3 only); a follow-up subagent ships Task 4
    once Task 5 GPU runner produces cached activity to test against.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_cross_bridge_probe.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet (CPU-only probe scaffold)")


def test_direction_5_vocab_spec_has_5_categories_v16_each():
    """Task 1: vocab spec exposes 5 category lists, each V=16 = 80 unique
    cross-bridge concepts. Pre-registered design (per design doc Approach
    A): nouns (16) + verbs (16) + adjectives (16) + spatial (16) +
    functional (16) = 80 distinct cross-bridge concepts.

    Global uniqueness enforced - no word appears in more than one bridge.

    The vocab is DELIBERATELY IDENTICAL to direction_4_vocab_spec.py so
    that the Direction 5 HYBRID test is directly comparable to the
    Direction 4 NEGATIVE result on the SAME concept set.
    """
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_vocab_spec.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 1 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_5_vocab_spec", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pre-registered category constants must exist
    assert hasattr(mod, "DIRECTION_5_NOUN_VOCAB"), (
        "vocab spec missing DIRECTION_5_NOUN_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_5_VERB_VOCAB"), (
        "vocab spec missing DIRECTION_5_VERB_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_5_ADJECTIVE_VOCAB"), (
        "vocab spec missing DIRECTION_5_ADJECTIVE_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_5_SPATIAL_VOCAB"), (
        "vocab spec missing DIRECTION_5_SPATIAL_VOCAB"
    )
    assert hasattr(mod, "DIRECTION_5_FUNCTIONAL_VOCAB"), (
        "vocab spec missing DIRECTION_5_FUNCTIONAL_VOCAB"
    )
    # Pre-registered V=16 per category
    assert len(mod.DIRECTION_5_NOUN_VOCAB) == 16, (
        "DIRECTION_5_NOUN_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_5_VERB_VOCAB) == 16, (
        "DIRECTION_5_VERB_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_5_ADJECTIVE_VOCAB) == 16, (
        "DIRECTION_5_ADJECTIVE_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_5_SPATIAL_VOCAB) == 16, (
        "DIRECTION_5_SPATIAL_VOCAB must be exactly 16"
    )
    assert len(mod.DIRECTION_5_FUNCTIONAL_VOCAB) == 16, (
        "DIRECTION_5_FUNCTIONAL_VOCAB must be exactly 16"
    )
    # Pre-registered total = 80
    total = (
        len(mod.DIRECTION_5_NOUN_VOCAB)
        + len(mod.DIRECTION_5_VERB_VOCAB)
        + len(mod.DIRECTION_5_ADJECTIVE_VOCAB)
        + len(mod.DIRECTION_5_SPATIAL_VOCAB)
        + len(mod.DIRECTION_5_FUNCTIONAL_VOCAB)
    )
    assert total == 80, (
        "5-category V=16 spec must total exactly 80 concepts; got "
        + str(total)
    )
    # Global uniqueness - no word appears in two bridges
    all_words = (
        list(mod.DIRECTION_5_NOUN_VOCAB)
        + list(mod.DIRECTION_5_VERB_VOCAB)
        + list(mod.DIRECTION_5_ADJECTIVE_VOCAB)
        + list(mod.DIRECTION_5_SPATIAL_VOCAB)
        + list(mod.DIRECTION_5_FUNCTIONAL_VOCAB)
    )
    assert len(set(all_words)) == 80, (
        "Direction 5 vocab spec has duplicate words across bridges; "
        "expected 80 unique; got " + str(len(set(all_words)))
    )


def test_direction_5_vocab_spec_matches_direction_4_vocab():
    """Task 1: Direction 5 vocab MUST match Direction 4 vocab exactly
    (deliberate design choice; same 80 concept set is the basis for the
    direct A/B comparison between dedicated-only and HYBRID substrates).
    Any future PR that diverges either vocab silently triggers this test.

    This test ALSO skips if Direction 4 vocab spec is absent (defensive;
    in practice Direction 4 always ships first per the dependency chain).
    """
    d5_path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_vocab_spec.py",
    )
    d4_path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_vocab_spec.py",
    )
    if not os.path.exists(d5_path):
        pytest.skip("Task 1 (D5 vocab spec) not landed yet")
    if not os.path.exists(d4_path):
        pytest.skip("Direction 4 vocab spec absent (defensive skip)")
    spec5 = importlib.util.spec_from_file_location(
        "direction_5_vocab_spec", d5_path,
    )
    mod5 = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(mod5)
    spec4 = importlib.util.spec_from_file_location(
        "direction_4_vocab_spec", d4_path,
    )
    mod4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(mod4)
    # Per-category dict equality (same keys + same values)
    assert dict(mod5.DIRECTION_5_NOUN_VOCAB) == dict(mod4.DIRECTION_4_NOUN_VOCAB)
    assert dict(mod5.DIRECTION_5_VERB_VOCAB) == dict(mod4.DIRECTION_4_VERB_VOCAB)
    assert dict(mod5.DIRECTION_5_ADJECTIVE_VOCAB) == dict(mod4.DIRECTION_4_ADJECTIVE_VOCAB)
    assert dict(mod5.DIRECTION_5_SPATIAL_VOCAB) == dict(mod4.DIRECTION_4_SPATIAL_VOCAB)
    assert dict(mod5.DIRECTION_5_FUNCTIONAL_VOCAB) == dict(mod4.DIRECTION_4_FUNCTIONAL_VOCAB)
    # All-words union equal in order
    assert list(mod5.DIRECTION_5_ALL_WORDS) == list(mod4.DIRECTION_4_ALL_WORDS)


def test_direction_5_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present and match design-doc
    bar (0.80 multi-seed; same as pillars n=93+ + Directions Q, 3, 4)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_5_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must exist)
    assert hasattr(mod, "_DIRECTION_5_OB_MIN"), (
        "verdict module missing _DIRECTION_5_OB_MIN"
    )
    assert hasattr(mod, "_DIRECTION_5_OI_MIN"), (
        "verdict module missing _DIRECTION_5_OI_MIN"
    )
    assert hasattr(mod, "_DIRECTION_5_LOADS"), (
        "verdict module missing _DIRECTION_5_LOADS"
    )
    assert hasattr(mod, "_DIRECTION_5_MIN_SEEDS"), (
        "verdict module missing _DIRECTION_5_MIN_SEEDS"
    )
    # Pre-registered values (tampering caught here)
    assert mod._DIRECTION_5_OB_MIN == 0.80, (
        "_DIRECTION_5_OB_MIN tampered: design fixes this at 0.80"
    )
    assert mod._DIRECTION_5_OI_MIN == 0.80, (
        "_DIRECTION_5_OI_MIN tampered: design fixes this at 0.80"
    )
    assert list(mod._DIRECTION_5_LOADS) == [2, 3, 5], (
        "_DIRECTION_5_LOADS tampered: design fixes this at [2, 3, 5]"
    )
    assert mod._DIRECTION_5_MIN_SEEDS == 3, (
        "_DIRECTION_5_MIN_SEEDS tampered: design fixes this at 3"
    )


def test_direction_5_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID_MALFORMED (instrument-
    validity failure) from PASS / PARTIAL / NEGATIVE. Same discipline
    pattern as Direction Q + Direction 3 + Direction 4."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_verdict.py",
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


def test_direction_5_bridge_builder_has_five_functions():
    """Task 2: per-bridge HYBRID builder wrapper exposes 5 functions, one
    per bridge (A_nouns / B_verbs / C_adj / D_spatial / E_functional).
    Each builds a fresh SimulationBridge for that bridge's V=16 category
    on the HYBRID substrate (bio dedicated pools + shared sparse pool)."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "def build_direction_5_bridge_A_nouns" in src, (
        "bridge_builder must define build_direction_5_bridge_A_nouns"
    )
    assert "def build_direction_5_bridge_B_verbs" in src, (
        "bridge_builder must define build_direction_5_bridge_B_verbs"
    )
    assert "def build_direction_5_bridge_C_adj" in src, (
        "bridge_builder must define build_direction_5_bridge_C_adj"
    )
    assert "def build_direction_5_bridge_D_spatial" in src, (
        "bridge_builder must define build_direction_5_bridge_D_spatial"
    )
    assert "def build_direction_5_bridge_E_functional" in src, (
        "bridge_builder must define build_direction_5_bridge_E_functional"
    )


def test_direction_5_bridge_builder_uses_protected_builder_byte_unchanged():
    """Task 2: per-bridge HYBRID builder must REUSE
    build_biological_brain_regions byte-unchanged (the protected
    dedicated-substrate builder). The wrapper loads each category's V=16
    vocab via the existing noun_pool_names / verb_pool_names /
    adjective_pool_names parameters; the protected builder itself is
    NOT modified."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "build_biological_brain_regions" in src, (
        "bridge_builder must reuse build_biological_brain_regions (the "
        "protected dedicated builder) via import; no other path acceptable"
    )


def test_direction_5_bridge_builder_uses_g20_sparse_primitives_byte_unchanged():
    """Task 2: per-bridge HYBRID builder must REUSE the G.20 sparse pillar
    n=95 primitives byte-unchanged for the shared sparse substrate:
    generate_sparse_patterns + apply_sparse_topographic_prior. The
    primitives themselves are NOT modified in
    concept_pool_sparse_distributed.py; this builder imports them and
    composes."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "generate_sparse_patterns" in src, (
        "bridge_builder must reuse generate_sparse_patterns (G.20 sparse "
        "pillar n=95 primitive) via import; no other path acceptable"
    )
    assert "apply_sparse_topographic_prior" in src, (
        "bridge_builder must reuse apply_sparse_topographic_prior (G.20 "
        "sparse pillar n=95 primitive) via import; no other path acceptable"
    )
    # Confirms the import is from the G.20 sparse module (not a copy in
    # this file)
    assert "concept_pool_sparse_distributed" in src, (
        "bridge_builder must import the G.20 sparse primitives FROM "
        "research.runners.concept_pool_sparse_distributed (not copy them)"
    )


def test_direction_5_bridge_builder_cpu_light_import():
    """Task 2: importing the bridge builder module MUST be CPU-light
    (must NOT import cupy / sim.bridge / sim.config / sim.regions /
    research.runners.text_minimal_isolation /
    research.runners.concept_pool_sparse_distributed at module load
    time). Those imports are deferred to inside the constructor function
    bodies so that the module can be loaded for inspection on CPU-only
    machines (CI, NumPy-backend dev) without triggering a CuPy
    initialization.

    Uses Python's ast module to inspect MODULE-LEVEL Import / ImportFrom
    nodes only (function-body imports are correctly ignored by the AST
    walk; coarse text-line indent tracking is unreliable when a function
    body itself contains a multi-line continuation).
    """
    import ast
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_5_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    forbidden_module_prefixes = (
        "cupy", "cupyx",
        "sim.bridge", "sim.config", "sim.regions",
        "research.runners.text_minimal_isolation",
        "research.runners.concept_pool_sparse_distributed",
    )
    # Walk MODULE-LEVEL statements only (tree.body); function-body
    # imports are inside FunctionDef.body and are intentionally allowed.
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modn = alias.name or ""
                for pref in forbidden_module_prefixes:
                    if modn == pref or modn.startswith(pref + "."):
                        offenders.append(("import " + modn, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            modn = node.module or ""
            for pref in forbidden_module_prefixes:
                if modn == pref or modn.startswith(pref + "."):
                    offenders.append(
                        ("from " + modn + " import ...", node.lineno),
                    )
    assert not offenders, (
        "Top-level forbidden imports in bridge_builder (must defer "
        "inside constructor function bodies): " + repr(offenders)
    )
