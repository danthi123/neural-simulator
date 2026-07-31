---
type: plan
status: live
date: 2026-05-25
---

# Direction 4 Implementation Plan — cross-bridge composition on bio_brain_regions (5 bridges × V=16 = 80 concepts)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build 5 bio_brain_regions bridges, each on a DIFFERENT vocab category (noun / verb / adjective / spatial / functional, V=16 each = 80 distinct cross-bridge concepts). Train them (controller-only; GPU-bound; Task 5). Run cross-bridge parallel-matching mode-unification probe (controller-only; Task 6) and emit pre-registered frozen-threshold verdict. Multi-seed [42, 43, 44]. Bar UNCHANGED at 0.80 multi-seed (same as pillars n=93+ and Directions Q, 3).

**Architecture:** 5 separate SimulationBridges, each built via `build_biological_brain_regions` (the protected builder) byte-unchanged. Each bridge has the standard 4 motor pools (N/E/S/W; Tier 1 canon hard-assumption in the builder) PLUS its category's 16 concept pools (loaded via the existing `noun_pool_names` / `verb_pool_names` / `adjective_pool_names` parameters). For spatial / functional bridges (no dedicated pool kind in the builder), reuse the `noun_pool_names` slot — the substrate's concept-pool architecture is category-agnostic at the pool level (each pool is a 200-neuron concept attractor with FS interneurons + lang_input/lang_output pathways). This preserves the protected builder byte-unchanged.

**Cross-bridge probe (Task 6):** Mirror the G.20 sparse 5-bridge cross-bridge parallel-matching pattern (pillar n=95) BYTE-UNCHANGED in primitive. The probe operates on cached trained activity per bridge; each composite samples K items uniformly from the 80-concept union; parallel-matching decodes per-slot identification.

**Tech Stack:** CuPy GPU (Task 5 training only), NumPy (Task 6 cross-bridge probe; CPU-only per pillar n=95 pattern), pre-registered fixed-threshold verdict module (Task 3).

**Reuse-by-import only.** No protected/frozen/moat modifications. `build_biological_brain_regions` remains byte-unchanged. The pillar n=95 cross-bridge probe primitive is reused byte-unchanged (Task 6).

**Net-new modules (5):**
1. `tests/test_direction_4_grounding.py` (Task 0)
2. `research/findings/raw/direction_4_vocab_spec.py` (Task 1)
3. `research/findings/raw/direction_4_bridge_builder.py` (Task 2)
4. `research/findings/raw/direction_4_verdict.py` + `tests/test_direction_4_verdict.py` (Task 3)
5. `research/findings/raw/direction_4_cross_bridge_probe.py` (Task 4; CPU-only)
6. `research/findings/raw/direction_4_5bridge_runner.py` (Task 5; GPU-bound, controller-only)

**Decisive run (Task 6) is CONTROLLER-ONLY** — orchestrates 5 bridges × 3 seeds = 15 bridge trainings (~5 hours GPU per design doc estimate), then cross-bridge probe + verdict emission.

---

## Pre-launch grep (DONE 2026-05-25, this subagent invocation)

Confirmed NET-NEW: no prior `direction_4_*.py` files exist; only design doc + roadmap + state files mention "cross-bridge bio_brain_regions" (which is what THIS direction implements). The G.20 sparse 5-bridge pattern (pillar n=95) is on a DIFFERENT substrate (sparse Kanerva SDM, not bio_brain_regions concept pools).

---

### Task 0: Grounding pin (intentionally RED until Tasks 1-4 land)

**Files:**
- Create: `tests/test_direction_4_grounding.py`

**Goal:** pin the contracts the Direction 4 subsystem MUST satisfy (module existence + threshold-frozen at the design-doc bar). RED on commit; turn GREEN as Tasks 1-3 land (Tasks 4 and 5 also have grounding tests).

**Step 1: Write the failing tests**

The grounding tests follow the Direction Q + Direction 3 pattern (file-existence assertions per task + threshold-frozen tests via importlib.util). They are RED on the Task 0 commit (none of the target files exist yet); they turn GREEN incrementally as later tasks land.

```python
# tests/test_direction_4_grounding.py
"""Direction 4 grounding pin — intentionally RED until Tasks 1-4 land.

Pins the contracts the Direction 4 cross-bridge-on-bio_brain_regions
runner MUST satisfy. Discipline pattern matches Direction Q + Direction 3.
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
    """Task 4: cross-bridge probe runner exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_cross_bridge_probe.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet (CPU-only probe scaffolding)")


def test_direction_4_vocab_spec_has_5_categories_v16_each():
    """Task 1: vocab spec exposes 5 category lists, each V=16 = 80 unique
    cross-bridge concepts. Pre-registered design (per design doc Approach A):
    nouns (16) + verbs (16) + adjectives (16) + spatial (16) + functional
    (16) = 80 distinct cross-bridge concepts."""
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
    # Pre-registered category names
    assert hasattr(mod, "DIRECTION_4_NOUN_VOCAB")
    assert hasattr(mod, "DIRECTION_4_VERB_VOCAB")
    assert hasattr(mod, "DIRECTION_4_ADJECTIVE_VOCAB")
    assert hasattr(mod, "DIRECTION_4_SPATIAL_VOCAB")
    assert hasattr(mod, "DIRECTION_4_FUNCTIONAL_VOCAB")
    # Pre-registered V=16 per category
    assert len(mod.DIRECTION_4_NOUN_VOCAB) == 16
    assert len(mod.DIRECTION_4_VERB_VOCAB) == 16
    assert len(mod.DIRECTION_4_ADJECTIVE_VOCAB) == 16
    assert len(mod.DIRECTION_4_SPATIAL_VOCAB) == 16
    assert len(mod.DIRECTION_4_FUNCTIONAL_VOCAB) == 16
    # Pre-registered total = 80
    total = (
        len(mod.DIRECTION_4_NOUN_VOCAB)
        + len(mod.DIRECTION_4_VERB_VOCAB)
        + len(mod.DIRECTION_4_ADJECTIVE_VOCAB)
        + len(mod.DIRECTION_4_SPATIAL_VOCAB)
        + len(mod.DIRECTION_4_FUNCTIONAL_VOCAB)
    )
    assert total == 80
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
        "expected 80 unique"
    )


def test_direction_4_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds match design-doc bar (0.80
    multi-seed; same as pillars n=93+ + Directions Q, 3)."""
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
    assert hasattr(mod, "_DIRECTION_4_OB_MIN")
    assert hasattr(mod, "_DIRECTION_4_OI_MIN")
    assert hasattr(mod, "_DIRECTION_4_LOADS")
    assert hasattr(mod, "_DIRECTION_4_MIN_SEEDS")
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
    """Task 3: verdict module must distinguish VOID_MALFORMED from
    PASS / PARTIAL / NEGATIVE."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "VOID" in src or "void" in src, (
        "verdict module must include VOID_MALFORMED branch"
    )


def test_direction_4_bridge_builder_has_five_functions():
    """Task 2: builder wrapper exposes 5 functions, one per bridge."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_4_bridge_builder.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 2 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "def build_direction_4_bridge_A_nouns" in src
    assert "def build_direction_4_bridge_B_verbs" in src
    assert "def build_direction_4_bridge_C_adj" in src
    assert "def build_direction_4_bridge_D_spatial" in src
    assert "def build_direction_4_bridge_E_functional" in src
```

**Step 2: Run tests to verify they fail / skip appropriately**

Run: `pytest tests/test_direction_4_grounding.py -v`

Expected on Task 0 commit (before Tasks 1-3): 3 FAILED (module-existence), 4 SKIPPED (gated on missing modules), 1 FAILED implicitly (cross-bridge module).

Expected after Tasks 1-3 land (this subagent invocation): 7-8 PASSED, 0 SKIPPED.

**Step 3: Commit**

Commit Task 0 grounding pin alongside Tasks 1-3 in a single subagent commit.

---

### Task 1: Vocab spec module (5 × V=16 = 80 unique cross-bridge concepts)

**Files:**
- Create: `research/findings/raw/direction_4_vocab_spec.py`

**Goal:** Frozen 5-category vocab lists; global uniqueness guaranteed; no runtime override path.

**Pre-registered word lists (DESIGN DOC Approach A):**

- BridgeA (nouns, 16): apple, river, dog, cat (v14 baseline) + tree, bird, sun, moon, book, chair, house, wheel, ball, cup, lamp, road
- BridgeB (verbs, 16): go, come, stop, look (v14 baseline) + walk, run, eat, sleep, sit, stand, jump, climb, throw, catch, lift, pull
- BridgeC (adjectives, 16): big, small, hot, cold (v14 baseline) + fast, slow, bright, dark, loud, quiet, sweet, sour, heavy, light, sharp, soft
- BridgeD (spatial, 16): north, east, south, west, up, down, left, right, in, out, near, far, top, bottom, center, side
- BridgeE (functional, 16): i, you, he, she, the, a, and, or, with, for, this, that, these, those, what, when

**Total = 80 unique cross-bridge concepts. Global uniqueness enforced by Task 0 grounding test.**

Discipline: stdlib + typing imports only; module-level constants; no runtime override.

```python
# research/findings/raw/direction_4_vocab_spec.py
"""Direction 4 vocab spec — 5 categories × V=16 = 80 cross-bridge concepts.

Pre-registered FROZEN word lists per design doc
docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-design.md
Approach A. Each list maps to a separate bio_brain_regions bridge.

DISCIPLINE: this module is data only (no imports beyond typing). The words
are FROZEN at module load time; no runtime override path. Any PR that
silently changes a word triggers test_direction_4_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# Bridge A — nouns (V=16)
DIRECTION_4_NOUN_VOCAB: Dict[str, str] = {
    # v14 baseline
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
    # extension (12)
    "tree": "TREE",
    "bird": "BIRD",
    "sun": "SUN",
    "moon": "MOON",
    "book": "BOOK",
    "chair": "CHAIR",
    "house": "HOUSE",
    "wheel": "WHEEL",
    "ball": "BALL",
    "cup": "CUP",
    "lamp": "LAMP",
    "road": "ROAD",
}

# Bridge B — verbs (V=16)
DIRECTION_4_VERB_VOCAB: Dict[str, str] = {
    # v14 baseline
    "go": "GO",
    "come": "COME",
    "stop": "STOP",
    "look": "LOOK",
    # extension (12)
    "walk": "WALK",
    "run": "RUN",
    "eat": "EAT",
    "sleep": "SLEEP",
    "sit": "SIT",
    "stand": "STAND",
    "jump": "JUMP",
    "climb": "CLIMB",
    "throw": "THROW",
    "catch": "CATCH",
    "lift": "LIFT",
    "pull": "PULL",
}

# Bridge C — adjectives (V=16)
DIRECTION_4_ADJECTIVE_VOCAB: Dict[str, str] = {
    # v14 baseline
    "big": "BIG",
    "small": "SMALL",
    "hot": "HOT",
    "cold": "COLD",
    # extension (12)
    "fast": "FAST",
    "slow": "SLOW",
    "bright": "BRIGHT",
    "dark": "DARK",
    "loud": "LOUD",
    "quiet": "QUIET",
    "sweet": "SWEET",
    "sour": "SOUR",
    "heavy": "HEAVY",
    "light": "LIGHT",
    "sharp": "SHARP",
    "soft": "SOFT",
}

# Bridge D — spatial (V=16). Mapped via noun_pool_names slot in the
# protected builder (no dedicated spatial pool kind; concept-pool
# architecture is category-agnostic at the pool level).
DIRECTION_4_SPATIAL_VOCAB: Dict[str, str] = {
    "north": "NORTH",
    "east": "EAST",
    "south": "SOUTH",
    "west": "WEST",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "in": "IN",
    "out": "OUT",
    "near": "NEAR",
    "far": "FAR",
    "top": "TOP",
    "bottom": "BOTTOM",
    "center": "CENTER",
    "side": "SIDE",
}

# Bridge E — functional (V=16). Same noun_pool_names slot mapping.
DIRECTION_4_FUNCTIONAL_VOCAB: Dict[str, str] = {
    "i": "I",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    "the": "THE",
    "a": "A",
    "and": "AND",
    "or": "OR",
    "with": "WITH",
    "for": "FOR",
    "this": "THIS",
    "that": "THAT",
    "these": "THESE",
    "those": "THOSE",
    "what": "WHAT",
    "when": "WHEN",
}


# Frozen pool-name lists (in builder-API ingestion order).
DIRECTION_4_NOUN_NAMES: List[str] = list(DIRECTION_4_NOUN_VOCAB.values())
DIRECTION_4_VERB_NAMES: List[str] = list(DIRECTION_4_VERB_VOCAB.values())
DIRECTION_4_ADJECTIVE_NAMES: List[str] = list(DIRECTION_4_ADJECTIVE_VOCAB.values())
DIRECTION_4_SPATIAL_NAMES: List[str] = list(DIRECTION_4_SPATIAL_VOCAB.values())
DIRECTION_4_FUNCTIONAL_NAMES: List[str] = list(DIRECTION_4_FUNCTIONAL_VOCAB.values())

# Per-bridge ordered word lists (the word-as-key order; used by training
# schedules + decoders that consume the FROZEN word order).
DIRECTION_4_BRIDGE_A_WORDS: List[str] = list(DIRECTION_4_NOUN_VOCAB.keys())
DIRECTION_4_BRIDGE_B_WORDS: List[str] = list(DIRECTION_4_VERB_VOCAB.keys())
DIRECTION_4_BRIDGE_C_WORDS: List[str] = list(DIRECTION_4_ADJECTIVE_VOCAB.keys())
DIRECTION_4_BRIDGE_D_WORDS: List[str] = list(DIRECTION_4_SPATIAL_VOCAB.keys())
DIRECTION_4_BRIDGE_E_WORDS: List[str] = list(DIRECTION_4_FUNCTIONAL_VOCAB.keys())

# Frozen union word order (cross-bridge probe consumes this).
DIRECTION_4_ALL_WORDS: List[str] = (
    DIRECTION_4_BRIDGE_A_WORDS
    + DIRECTION_4_BRIDGE_B_WORDS
    + DIRECTION_4_BRIDGE_C_WORDS
    + DIRECTION_4_BRIDGE_D_WORDS
    + DIRECTION_4_BRIDGE_E_WORDS
)

# Pre-registered total
DIRECTION_4_TOTAL: int = 80
```

**Step 4: Verify**

After landing, `pytest tests/test_direction_4_grounding.py::test_direction_4_vocab_spec_has_5_categories_v16_each -v` → PASSED.

---

### Task 2: Per-bridge builder wrapper (5 functions, CPU-only spec)

**Files:**
- Create: `research/findings/raw/direction_4_bridge_builder.py`

**Goal:** 5 pure constructor functions — `build_direction_4_bridge_A_nouns`, `build_direction_4_bridge_B_verbs`, `build_direction_4_bridge_C_adj`, `build_direction_4_bridge_D_spatial`, `build_direction_4_bridge_E_functional`. Each returns a fresh SimulationBridge configured with that bridge's V=16 vocab on the bio_brain_regions concept-pool substrate. No training in this module — just construction. The protected builder `build_biological_brain_regions` is byte-unchanged; this wrapper only sets parameters.

For bridges A / B / C: use the corresponding noun_pool_names / verb_pool_names / adjective_pool_names slot.

For bridges D / E: use the noun_pool_names slot (no spatial / functional pool kind exists; substrate's concept-pool architecture is category-agnostic at the pool level).

**v14/v16 production recipe** (per CLAUDE.md):
- n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24
- weak_concept_dynamics: density=0.05, exc=0.3, inh=0.8
- motor canon: density=0.10, exc=2.0, inh=4.0
- enable_motor_fs=True, enable_language_output=True
- NMDA on; tau_decay=100ms (Wang 2002 calibration)
- stdp_w_max=8.0
- enable_short_term_plasticity=False, enable_hebbian_learning=False,
  enable_structural_plasticity=False, enable_per_type_stp=False
- fast_spike_reset=True

```python
# research/findings/raw/direction_4_bridge_builder.py
"""Direction 4 per-bridge builder wrappers (5 functions, CPU-only spec).

Each function builds a fresh SimulationBridge with ONE category's V=16
vocab on the validated v14/v16 bio_brain_regions concept-pool architecture.
Mirrors the Direction 3 V=32 wrapper pattern but loads only ONE category
per bridge (the cross-bridge probe takes the union of 5 such bridges).

Reuses validated infrastructure byte-unchanged:
- sim.bridge.SimulationBridge (protected)
- research.runners.text_minimal_isolation.build_biological_brain_regions
  (protected; wrapper passes V=16 category vocab via existing
  noun_pool_names / verb_pool_names / adjective_pool_names parameters;
  the builder itself is NOT modified)
- v14/v16 production recipe defaults (weak_concept_dynamics, NMDA,
  motor canon, FS interneurons)

Bridges A / B / C map to dedicated pool kinds (noun / verb / adjective).
Bridges D / E (spatial / functional) reuse the noun_pool_names slot —
the substrate concept-pool architecture is category-agnostic at the pool
level. This preserves the protected builder byte-unchanged.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_4_vocab_spec import (
    DIRECTION_4_NOUN_NAMES,
    DIRECTION_4_VERB_NAMES,
    DIRECTION_4_ADJECTIVE_NAMES,
    DIRECTION_4_SPATIAL_NAMES,
    DIRECTION_4_FUNCTIONAL_NAMES,
)


def _build_bridge_core(
    seed: int,
    n_lang_input: int,
    n_per_pool: int,
    n_fs_per_pool: int,
    weak_dynamics: bool,
    noun_pool_names: list = None,
    verb_pool_names: list = None,
    adjective_pool_names: list = None,
    verbose: bool = False,
    label: str = "",
):
    """Shared bridge constructor body. Caller passes ONE non-None pool name
    list per call; the others stay None (= pool kind off)."""
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    concept_internal_density = 0.05 if weak_dynamics else None
    concept_exc_weight = 0.3 if weak_dynamics else None
    concept_inh_weight = 0.8 if weak_dynamics else None
    motor_internal_density = 0.10
    motor_exc_weight = 2.0
    motor_inh_weight = 4.0

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=motor_internal_density,
        motor_exc_weight_mean=motor_exc_weight,
        motor_inh_weight_mean=motor_inh_weight,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=(noun_pool_names is not None),
        noun_pool_names=noun_pool_names,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=(verb_pool_names is not None),
        verb_pool_names=verb_pool_names,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=(adjective_pool_names is not None),
        adjective_pool_names=adjective_pool_names,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    if verbose:
        rm = bridge.region_manager
        n_total = int(getattr(cfg, "num_neurons", 0)) or sum(
            r.n_neurons for r in cfg.brain_regions
        )
        print("[BUILD-D4-" + label + "] V=16 bridge: "
              + str(n_total) + " neurons total; n_lang_input="
              + str(n_lang_input) + ", n_per_pool=" + str(n_per_pool)
              + ", n_fs_per_pool=" + str(n_fs_per_pool)
              + ", weak_dynamics=" + str(weak_dynamics), flush=True)
    return bridge


def build_direction_4_bridge_A_nouns(seed, n_lang_input=2048,
                                       n_per_pool=200, n_fs_per_pool=24,
                                       weak_dynamics=True, verbose=False):
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_4_NOUN_NAMES,
        verbose=verbose, label="A_nouns",
    )


def build_direction_4_bridge_B_verbs(seed, n_lang_input=2048,
                                       n_per_pool=200, n_fs_per_pool=24,
                                       weak_dynamics=True, verbose=False):
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        verb_pool_names=DIRECTION_4_VERB_NAMES,
        verbose=verbose, label="B_verbs",
    )


def build_direction_4_bridge_C_adj(seed, n_lang_input=2048,
                                     n_per_pool=200, n_fs_per_pool=24,
                                     weak_dynamics=True, verbose=False):
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        adjective_pool_names=DIRECTION_4_ADJECTIVE_NAMES,
        verbose=verbose, label="C_adj",
    )


def build_direction_4_bridge_D_spatial(seed, n_lang_input=2048,
                                         n_per_pool=200, n_fs_per_pool=24,
                                         weak_dynamics=True, verbose=False):
    # Spatial vocab loaded via noun_pool_names slot (no dedicated spatial
    # pool kind in the protected builder; concept-pool architecture is
    # category-agnostic at the pool level).
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_4_SPATIAL_NAMES,
        verbose=verbose, label="D_spatial",
    )


def build_direction_4_bridge_E_functional(seed, n_lang_input=2048,
                                            n_per_pool=200, n_fs_per_pool=24,
                                            weak_dynamics=True, verbose=False):
    # Functional vocab loaded via noun_pool_names slot. Same rationale as
    # bridge D.
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_4_FUNCTIONAL_NAMES,
        verbose=verbose, label="E_functional",
    )
```

**Step 4: Verify**

After landing, `pytest tests/test_direction_4_grounding.py::test_direction_4_bridge_builder_has_five_functions -v` → PASSED.

**Important:** the bridge construction itself is NOT tested in this subagent invocation (would require importing CuPy / SimulationBridge which is GPU-bound and could compete with the running Direction 3 GPU work). Construction is verified via Task 5 controller-only training launch.

---

### Task 3: Frozen verdict module + adversarial test matrix (>=12 cases)

**Files:**
- Create: `research/findings/raw/direction_4_verdict.py`
- Create: `tests/test_direction_4_verdict.py`

**Goal:** pure stdlib-only verdict function. Takes recorded per-seed cross-bridge accuracy data (parallel-matching OB + OI at each load); returns frozen-threshold verdict tag in {DIRECTION_4_PASS, DIRECTION_4_PARTIAL, DIRECTION_4_NEGATIVE, DIRECTION_4_VOID_MALFORMED}. Instrument-validity check FIRST. No runtime override.

**Pre-registered thresholds (frozen):**
- `_DIRECTION_4_OB_MIN = 0.80`
- `_DIRECTION_4_OI_MIN = 0.80`
- `_DIRECTION_4_LOADS = (2, 3, 5)`
- `_DIRECTION_4_MIN_SEEDS = 3`

**Implementation:** mirror `direction_3_verdict.py` byte-pattern (rename constants/tags but otherwise identical structure). Stdlib-only imports.

**Test matrix (>=12 cases in `tests/test_direction_4_verdict.py`):**

1. `test_thresholds_frozen_at_design_values` — all 4 frozen thresholds
2. `test_threshold_tamper_detection` — explicit constant-value asserts
3. `test_pass_when_all_cells_clear_bar` — happy path (all loads + readouts >= 0.80)
4. `test_negative_when_no_cell_clears_bar` — opposite (all < 0.80)
5. `test_partial_when_some_cells_clear` — only some pass (e.g., OB but not OI)
6. `test_partial_when_one_load_fails_oi` — boundary case
7. `test_void_on_none_input` — None → VOID_MALFORMED
8. `test_void_on_empty_list` — empty list → VOID_MALFORMED
9. `test_void_on_fewer_than_min_seeds` — 1-2 seeds → VOID
10. `test_void_on_missing_load_key` — missing L=3 → VOID
11. `test_void_on_nan_ob` — NaN value → VOID
12. `test_void_on_inf_oi` — Inf value → VOID
13. `test_void_on_string_in_value_slot` — type error → VOID
14. `test_void_on_non_dict_seed_entry` — list-of-list → VOID
15. `test_boundary_exactly_at_threshold` — value == 0.80 → counts as PASS
16. `test_below_threshold_eps_fails` — value 0.79 → not PASS
17. `test_verdict_never_raises_on_malformed_garbage` — fuzz-style: weird inputs return VOID, don't crash

**Step 4: Verify**

`pytest tests/test_direction_4_verdict.py -v` → 16+ PASSED.

---

### Task 4: Cross-bridge probe runner (CPU-only)

**Files:**
- Create: `research/findings/raw/direction_4_cross_bridge_probe.py`

**Goal:** reuse the pillar n=95 G.20 sparse cross-bridge parallel-matching mode-unification probe pattern BYTE-UNCHANGED in primitive. Operates on cached trained activity per bridge (per-seed JSON dumps containing per-concept activity vectors). Computes per-load OB + OI accuracy and writes JSON in the verdict module's expected shape.

**Status: scaffolding only in this subagent invocation.** The full probe relies on cached trained-bridge activity which doesn't yet exist (Task 5 generates it). The scaffolding includes the function signature + the verdict-shape JSON emit logic, but trips clearly with a NotImplementedError if invoked without cached activity.

This task is in scope for THIS subagent (CPU-only by design) but the decisive run is Task 6 (controller-only after Task 5 trains).

**The probe primitive itself** must be located in the existing project (per pre-launch grep at design time confirming pillar n=95 already shipped). The probe file in `research/findings/raw/` reuses-by-import (no copy-paste). If the existing pillar n=95 primitive is in a file like `research/findings/raw/cross_bridge_mode_unification_probe.py` or similar, this Task 4 file imports it and adapts for the 80-concept union.

**Step 1: Locate the existing pillar n=95 probe primitive**

Defer file discovery to the implementing agent — confirm in the file via comment which existing module is reused, with a note that "no modification" is the discipline.

**Step 2: Write the scaffold**

The scaffold:
- imports the pillar n=95 cross-bridge probe primitive (reuse-by-import)
- defines `run_direction_4_cross_bridge_probe(per_seed_bridge_activity_paths, out_path)` that:
  - loads per-bridge per-concept activity vectors from cached JSON
  - constructs the 80-concept union
  - calls the reused probe primitive at each load in `_DIRECTION_4_LOADS`
  - emits verdict-shape JSON

**Step 3: Commit**

This scaffold is part of the subagent commit.

---

### Task 5: CONTROLLER-ONLY 5-bridge multi-seed training launch (GPU-bound)

**NOT a subagent task. Controller orchestrates.**

After Direction 3 V=32 smoke completes and GPU is free, controller (or watchdog spawn) launches:

```bash
python -u -m research.findings.raw.direction_4_5bridge_runner \
    --seeds 42 43 44 \
    > research/findings/raw/direction_4_5bridge_training.log 2>&1 &
```

Expected wall ~30 min/bridge train (per CLAUDE.md v14/v16 production timing); 5 bridges × 3 seeds = 15 bridge trainings = ~5 hours.

This task is mentioned in the implementation plan for completeness only. Subagent does NOT create the runner module (which would require integration with GPU/CuPy infrastructure not under subagent scope per the discipline). The decisive run is the controller's responsibility.

Note: an alternate scaffold could be added later if the controller wants a CPU-only smoke before the GPU launch.

---

### Task 6: CONTROLLER-ONLY decisive cross-bridge probe + verdict emission

**NOT a subagent task. Controller orchestrates after Task 5 completes.**

1. Load per-bridge per-seed trained activity dumps from Task 5
2. Run `direction_4_cross_bridge_probe.run_direction_4_cross_bridge_probe()`
3. Compute verdict via `direction_4_verdict.compute_verdict(per_seed)`
4. Emit `research/findings/raw/direction_4_decisive.json` with verdict + per-seed cells
5. Adversarial-reviewer subagent pass
6. If PASS → pillar n=106 candidate; honest propagation both remotes

---

## Subagent commit scope (THIS subagent invocation)

Per the discipline binding (CPU-only; no GPU competition with running Direction 3 V=32 smoke), this subagent ships Tasks 0-3 (+ Task 4 scaffold) in a single commit:

**Files:**
- `tests/test_direction_4_grounding.py` (Task 0)
- `research/findings/raw/direction_4_vocab_spec.py` (Task 1)
- `research/findings/raw/direction_4_bridge_builder.py` (Task 2)
- `research/findings/raw/direction_4_verdict.py` (Task 3)
- `tests/test_direction_4_verdict.py` (Task 3)
- `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-implementation.md` (this plan)

**Commit message:**

```
Direction 4 scaffolding (Tasks 0-3 CPU-only): writing-plans + grounding pin
+ vocab spec + bridge builder + frozen verdict module.
```

Task 4 scaffold + Task 5/6 controller-only runs land in subsequent subagent commits / controller actions.

---

## Discipline (binding throughout)

- Bar UNCHANGED: `_DIRECTION_4_OB_MIN=0.80`, `_DIRECTION_4_OI_MIN=0.80`, `_DIRECTION_4_LOADS=(2,3,5)`, `_DIRECTION_4_MIN_SEEDS=3`. Set ONCE in `direction_4_verdict.py` at Task 3; never tuned by results.
- No protected / frozen / moat modification. `build_biological_brain_regions` remains byte-unchanged. Direction 4 uses its OWN per-bridge constructor wrappers. The pillar n=95 cross-bridge probe primitive is reused byte-unchanged (Task 4).
- No autograd.
- GPU/CuPy for Task 5 training only (controller-only). NumPy for Task 4 cross-bridge probe (CPU-only per pillar n=95 pattern).
- Honest propagation EVERY outcome both remotes.
- Pre-launch grep confirmed (this subagent invocation): no prior cross-bridge bio_brain_regions work; G.20 sparse 5-bridge is on DIFFERENT substrate.
- Reviewer-style scrutiny applied at the time of result (Task 6), not deferred.

---

## Post-Task chain (per verdict)

- **DIRECTION_4_PASS**: write findings doc; dispatch adversarial reviewer subagent; if CLEAR record pillar n=106; update AUTONOMOUS_STATE + capability_status.json. Cross-bridge composition on bio_brain_regions VALIDATED → conversational vocabulary substantially extended to biology-faithful 80 concepts.
- **DIRECTION_4_PARTIAL**: precise per-load breakdown; biology-translatable comparison to G.20 sparse n=95 (which mechanism is the bottleneck: bio substrate geometry vs sparse coding?).
- **DIRECTION_4_NEGATIVE**: cross-bridge requires sparse coding; pivot to Direction 5 (sparse-on-bio hybrid) or revisit architectural assumption.
- **DIRECTION_4_VOID_MALFORMED**: instrument-validity failure; diagnose recorded JSON shape; do not propagate as a pillar.
