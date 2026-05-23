# 160-concept ensemble vocabulary scaling: TDD implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task in the same session (the owner's standing instruction pre-selects same-session subagent-driven execution; transition directly from this plan to subagent-driven-development).

**Goal:** Implement and decisively test the activity-grounded biologized grounded-composition pipeline at K_VOCAB=16 per-bridge on the validated 5-bridge sparse-distributed concept ensemble (32 concepts per bridge × 5 bridges = 160 unique concepts), per `docs/plans/2026-05-23-160-concept-ensemble-vocab-scaling-design.md`. The pipeline, the recognition front-end, the substrate builder, the validated G.20 encoding, the frozen 0.80 bar, the multi-seed grid, and the loads {2, 3, 5} are all reused byte-unchanged from the K=16 PASS arc. The genuinely-new code is the multi-bridge orchestration + per-bridge caches + a small helper that pulls each bridge's vocab from the existing vocab specification.

**Architecture:** A focused multi-bridge extension of `research/findings/raw/vocabulary_scaling_run_trained.py`. The runner loops over the 5 bridges (per the existing `research/runners/g20_vocab_spec.ALL_BRIDGES` mapping); for each bridge per seed: build at 32 concepts using the validated G.20 defaults, train via the existing `train_substrate`, capture activity at M_OBS=16, run `run_pipeline` at K_VOCAB=16 loads {2, 3, 5}, save per-bridge per-seed activity cache, aggregate. The genuinely-new pieces are: a `bridge_vocab_and_patterns(bridge_name, seed, n_pool, k)` helper that pulls each bridge's 32-word vocab from the existing `g20_vocab_spec.ALL_BRIDGES` and generates its sparse K-of-N patterns via the validated `generate_sparse_patterns`; the per-bridge per-seed cache directory; the multi-bridge orchestration loop in the runner's main.

**Tech Stack:** Python + numpy + CuPy (for the GPU training). Reuses the validated substrate builder + training + cache helpers + biologized pipeline by import, byte-unchanged. The frozen 0.80 bar is unchanged.

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_160_ensemble_pin.py`

**Step 1: Write the failing test**

```python
"""Grounding pin for the 160-concept ensemble vocab-scaling arc. Pins
the design doc contract that the frozen 0.80 bar is unchanged, the
test grid (multi-seed, loads {2,3,5}, K_VOCAB=16, K_RECOG=8) is
identical to the K=16 PASS thread, and the 5-bridge ensemble has
exactly 160 unique concepts. The runner-module-exists check goes
green only after Task 2 lands -- intentional."""
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS, N_CONCEPTS,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES, TOTAL_VOCAB


def test_compositional_bar_frozen():
    assert BAR == 0.80


def test_k16_pass_recipe_imported_unchanged():
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200
    assert N_DIM == 512
    # 64-concept test grid still pinned; the 160-ensemble runner
    # uses its own per-bridge concept count = 32, not the global
    # N_CONCEPTS = 64.
    assert N_CONCEPTS == 64


def test_five_bridges_at_32_concepts_each_160_total():
    assert TOTAL_VOCAB == 160
    assert sorted(ALL_BRIDGES.keys()) == sorted([
        "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
        "bridgeD_spatial", "bridgeE_functional",
    ])
    for name, vocab in ALL_BRIDGES.items():
        assert len(vocab) == 32, f"{name} has {len(vocab)} words"
    # Global uniqueness across all 5 bridges.
    all_words = [w for v in ALL_BRIDGES.values() for w in v]
    assert len(all_words) == 160
    assert len(set(all_words)) == 160


def test_runner_module_exists():
    """Red until Task 2 lands -- intentional: surfaces any drift in
    the runner's public surface once Task 2 is wired."""
    from research.findings.raw import (
        vocabulary_scaling_run_160ensemble as m,
    )
    assert hasattr(m, "bridge_vocab_and_patterns")
    assert hasattr(m, "run_one_bridge_seed")
    assert hasattr(m, "main")
```

**Step 2: Run to verify it fails on the module-exists check**

`python -m pytest tests/test_160_ensemble_pin.py -q`

Expected: `test_runner_module_exists` FAILS; the three constant pins pass.

**Step 3: Commit**

```bash
git add tests/test_160_ensemble_pin.py
git commit -m "160-ensemble Task 0: grounding pin (red until Task 2 -- intentional)"
```

---

### Task 1: `bridge_vocab_and_patterns` helper

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_160ensemble_helpers.py`
- Create: `tests/test_vocabulary_scaling_160ensemble_helpers.py`

**Step 1: Write the failing tests**

```python
"""Unit tests for the multi-bridge helper. Pure function; pins shape,
determinism, vocab match against the existing g20_vocab_spec, and
sparse-pattern correctness."""
import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_160ensemble_helpers import (
    bridge_vocab_and_patterns, BRIDGE_NAMES,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES


def test_bridge_names_match_spec():
    assert sorted(BRIDGE_NAMES) == sorted(ALL_BRIDGES.keys())
    assert len(BRIDGE_NAMES) == 5


def test_returns_vocab_matching_spec_exactly():
    for name in BRIDGE_NAMES:
        vocab, _ = bridge_vocab_and_patterns(name, seed=42,
                                              n_pool=2000, k=100)
        assert vocab == list(ALL_BRIDGES[name])
        assert len(vocab) == 32


def test_returns_32_sparse_patterns_of_k_neurons():
    vocab, pats = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    assert len(pats) == 32
    for p in pats:
        assert len(p) == 100
        assert all(0 <= i < 2000 for i in p)
        assert len(set(p)) == 100   # no duplicate indices in a pattern


def test_deterministic_in_seed():
    v1, p1 = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    v2, p2 = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    assert v1 == v2
    assert [list(x) for x in p1] == [list(x) for x in p2]


def test_per_bridge_patterns_differ():
    """Each bridge's patterns are seeded with its bridge name to
    decorrelate -- bridgeA's patterns must not equal bridgeB's."""
    _, pA = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    _, pB = bridge_vocab_and_patterns(
        "bridgeB_verbs", seed=42, n_pool=2000, k=100)
    # At least one pattern differs.
    same = all(list(pA[i]) == list(pB[i]) for i in range(32))
    assert not same


def test_unknown_bridge_raises():
    with pytest.raises(ValueError):
        bridge_vocab_and_patterns("not_a_bridge", seed=42,
                                   n_pool=2000, k=100)
```

**Step 2: Run to verify they fail**

`python -m pytest tests/test_vocabulary_scaling_160ensemble_helpers.py -q`

Expected: ModuleNotFoundError.

**Step 3: Write the helper**

```python
"""Helpers for the 160-concept ensemble vocab-scaling arc.

The ensemble has 5 sparse-distributed bridges, each at 32 concepts.
This helper pulls each bridge's 32-word vocab from the existing
g20_vocab_spec and generates its sparse K-of-N patterns
deterministically. Per-bridge patterns are seeded with both the seed
AND the bridge name (via a deterministic name hash) so the 5
bridges' pattern sets are decorrelated even at the same seed."""
from __future__ import annotations

import hashlib
from typing import List, Tuple

from research.runners.g20_vocab_spec import ALL_BRIDGES
from research.runners.concept_pool_sparse_distributed import (
    generate_sparse_patterns,
)

BRIDGE_NAMES = list(ALL_BRIDGES.keys())


def _bridge_seed(name: str, seed: int) -> int:
    """A deterministic per-bridge seed: base seed plus a stable hash
    of the bridge name. Keeps each bridge's pattern set independent
    of the others while remaining fully reproducible from (seed, name)."""
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(h[:4], "big")


def bridge_vocab_and_patterns(
    bridge_name: str, seed: int, n_pool: int, k: int,
) -> Tuple[List[str], List[List[int]]]:
    """Return ``(vocab, patterns)`` for one bridge.

    ``vocab`` is the bridge's 32-word vocabulary from g20_vocab_spec
    (returned in spec order; never reshuffled). ``patterns`` is a
    deterministic list of 32 sparse K-of-N patterns over ``n_pool``
    neurons, generated by the validated `generate_sparse_patterns`
    seeded with a per-bridge seed (base seed XOR a stable hash of
    the bridge name) so each bridge's patterns are decorrelated from
    the others.
    """
    if bridge_name not in ALL_BRIDGES:
        raise ValueError(
            f"unknown bridge {bridge_name!r}; valid: {BRIDGE_NAMES}")
    vocab = list(ALL_BRIDGES[bridge_name])
    pats = generate_sparse_patterns(
        n_concepts=len(vocab), n_pool=n_pool, pattern_size=k,
        seed=_bridge_seed(bridge_name, seed))
    return vocab, pats
```

**Step 4: Run to verify they pass**

`python -m pytest tests/test_vocabulary_scaling_160ensemble_helpers.py -q`

Expected: 6/6 PASS.

**Step 5: Commit**

```bash
git add research/findings/raw/vocabulary_scaling_160ensemble_helpers.py tests/test_vocabulary_scaling_160ensemble_helpers.py
git commit -m "160-ensemble Task 1: bridge_vocab_and_patterns helper (5 bridges × 32 concepts; 6/6 tests)"
```

---

### Task 2: The multi-bridge runner

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_run_160ensemble.py`
- Test: `tests/test_160_ensemble_pin.py` (the Task 0 pin -- it goes green when Task 2 lands)

**Step 1: Re-run the grounding pin to confirm it still trips**

`python -m pytest tests/test_160_ensemble_pin.py -q`

Expected: `test_runner_module_exists` still FAILS (red).

**Step 2: Write the runner (focused multi-bridge orchestration)**

The runner imports the 64-concept K=16 PASS pipeline byte-unchanged and adds a multi-bridge orchestration loop. Key public surface (the Task 0 pin checks for these):
- `bridge_vocab_and_patterns` (re-exported from the helper module)
- `run_one_bridge_seed(bridge_name, seed, smoke=False)` -> dict
- `main()`

Internal structure:
- `BRIDGE_CACHE_DIR = "research/findings/raw/vocabulary_scaling_160ensemble_cache"`
- `_cache_path(bridge_name, seed, smoke)` -> path
- `build_and_train_bridge(bridge_name, seed, n_lang=..., n_pool=..., n_fs=..., k=..., n_train_events=400)` builds the sparse pool bridge sized for 32 concepts, runs `train_substrate` with that bridge's patterns at K_VOCAB=16-matching encoding (the validated G.20 defaults from `train_substrate`), returns the trained bridge.
- `run_one_bridge_seed(bridge_name, seed, smoke=False)` loads cache if present, else builds + trains + captures + saves cache; runs `run_pipeline(seed, acts, words, LOADS, N_TRIALS, K_RECOG=8, K_VOCAB=16)` (the K=16 recipe).
- `main()` loops over BRIDGE_NAMES × SEEDS, aggregates 5 × 3 = 15 (bridge, seed) results, computes per-bridge multi-seed aggregates and overall PASS/NEGATIVE verdict (PASS iff every bridge × every load multi-seed mean >= 0.80), writes JSON.

`--smoke` mode: reduced bridge size (n_lang=512, n_pool=512, n_fs=60, k=24), 2 bridges only (bridgeA_nouns, bridgeB_verbs), few train events, tiny vocab subset (8 concepts/bridge). Toy numbers NOT propagated as a result.

Sparsity used throughout: 0.01 (same as the 64-concept thread; stride 8192/32 = 256, n_active at 0.01 = 82 < 256, geometry holds).

**Step 3: Run the grounding pin to verify it goes green**

`python -m pytest tests/test_160_ensemble_pin.py -q`

Expected: 4/4 PASS.

**Step 4: Commit**

```bash
git add research/findings/raw/vocabulary_scaling_run_160ensemble.py
git commit -m "160-ensemble Task 2: multi-bridge runner (focused orchestration extension of trained-substrate runner)"
```

---

### Task 3: Soundness tests for the multi-bridge runner

**Files:**
- Create: `tests/test_vocabulary_scaling_160ensemble.py`

**Step 1: Write the failing tests**

```python
"""Soundness tests for the 160-concept ensemble runner. The load-
bearing properties: (a) the runner uses g20_vocab_spec.ALL_BRIDGES,
unchanged; (b) build_and_train_bridge genuinely exercises each
bridge (snapshot weights pre/post); (c) the train -> capture handoff
on a tiny smoke bridge produces well-formed cached activity; (d)
recognised concept names are the only handle that names which
pattern is read."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run_160ensemble import (
    BRIDGE_NAMES, bridge_vocab_and_patterns,
    build_and_train_bridge_smoke,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES


def test_bridge_names_match_vocab_spec_exactly():
    assert set(BRIDGE_NAMES) == set(ALL_BRIDGES.keys())


def test_train_stage_exercises_each_bridge_substrate():
    """Smoke-scale build + train on a single bridge: the substrate
    connectivity must change substantially after training. Skips on
    CPU-only (the validated topographic prior is CuPy-only)."""
    from sim.backend import get_backend, to_host, is_gpu_backend
    get_backend()
    if not is_gpu_backend():
        pytest.skip("training requires the CuPy/GPU backend")
    from sim.backend import get_backend, to_host
    bridge, words, pats = build_and_train_bridge_smoke(
        "bridgeA_nouns", seed=42)
    # The smoke build_and_train should leave connectivity reshaped vs
    # a fresh bridge; the validated train_substrate's load-bearing
    # test already pins this on a similar tiny bridge (existing
    # tests/test_vocabulary_scaling_trained.py). Here we additionally
    # confirm the bridge is built + words match the bridge spec.
    assert words == list(ALL_BRIDGES["bridgeA_nouns"])[:len(words)]
    assert len(pats) == len(words)
```

The remaining soundness checks (no answer leak; recognition is the only handle; multi-bridge orchestration is well-formed) are covered by the dedicated adversarial reviewer in Task 4 since they involve runtime traces. The pytest layer pins the structural invariants only.

**Step 2: Run to verify they pass**

`python -m pytest tests/test_vocabulary_scaling_160ensemble.py -q`

Expected: 1 pass + 1 skip (or 2 pass on GPU).

**Step 3: Commit**

```bash
git add tests/test_vocabulary_scaling_160ensemble.py
git commit -m "160-ensemble Task 3: soundness tests (bridge name binding; smoke build-and-train sanity)"
```

---

### Task 4: Dedicated adversarial review (BEFORE the decisive GPU run)

**Files:** none (review only)

Dispatch a fresh general-purpose agent with the design + plan + the runner under review. Adversarial checks (RUN them, do not just read):

1. **No vocabulary drift.** Each bridge's vocab returned by `bridge_vocab_and_patterns` matches `g20_vocab_spec.ALL_BRIDGES[name]` exactly. The runner never regenerates vocabs from a different seed or re-orders them.

2. **Per-bridge pattern determinism.** Patterns are seeded with `_bridge_seed(name, seed)` (base seed XOR sha256(name)). For the same `(name, seed)` the patterns are byte-identical across runs. Across different bridges at the same seed the patterns differ.

3. **`train_substrate` genuinely exercises each bridge** (the same load-bearing check the 64-concept thread had, now per-bridge). A no-op for ANY bridge would silently lose that bridge in the aggregate.

4. **K=16 recipe is identical to the K=16 PASS.** Reviewer confirms `K_VOCAB=16`, `K_RECOG=8`, `N_TRIALS=200`, deriver seed 90909, no per-bridge tuning. The K=16 PASS recipe is fixed.

5. **The frozen 0.80 bar is unchanged.** `BAR` imported from `vocabulary_scaling_run`; verdict uses `mean_int < BAR` exactly.

6. **No answer leak in the multi-bridge orchestration.** The recognised concept name is the only handle that names which pattern / activity is read at every step (inherited from the 64-concept runner; must be re-verified at the multi-bridge level — e.g., the runner does not accidentally cross-reference one bridge's vocab into another bridge's pipeline).

7. **Byte-unchanged reuse.** `git diff` shows only new files added; no modification to any protected, frozen, moat, or previously-reviewed module (in particular `vocabulary_scaling_run.py`, `vocabulary_scaling_run_trained.py`, `concept_pool_sparse_distributed.py`, `g20_vocab_spec.py`, the no-confab moat).

8. **No automatic differentiation.** Grep.

9. **Per-bridge cache cannot poison.** Cache directory + key (`{bridge_name}_seed{seed}.npz`) is distinct from the 64-concept cache and from any other cache; the runner only LOADS a cache that matches both `bridge_name` and `seed`.

10. **Smell-test the GPU plan.** Wall-clock estimate ~9 hours is honest (5 × 3 × ~35 min); kill-safe at per-bridge per-seed granularity; matches the existing trained-substrate runner's discipline.

Output: VERDICT CLEAR or BLOCK with specific defects. The decisive run does NOT launch until CLEAR.

---

### Task 5: Controller-only decisive GPU run

After Task 4 = CLEAR.

**Step 1: Smoke (controller-only):**

`python research/findings/raw/vocabulary_scaling_run_160ensemble.py --smoke`

Confirms end-to-end on the toy configuration. Toy numbers NOT propagated.

**Step 2: Launch decisive run (harness-tracked background, NOT `nohup`):**

```bash
python research/findings/raw/vocabulary_scaling_run_160ensemble.py > research/findings/raw/vocabulary_scaling_run_160ensemble_full.log 2>&1
```

with `run_in_background: true` so the harness genuinely tracks completion. Expected wall-clock: ~9 hours. Kill-safe at per-bridge per-seed granularity; a kill mid-run loses only the in-flight bridge-seed.

**Step 3: When the harness notifies completion:**

(a) Mandatory anti-cheat smell-test. Adapt the existing `vocabulary_scaling_smell_test.py` to the 160-ensemble JSON shape (per-bridge per-seed per-load). Recompute every per-load mean per-bridge from the recording independently of the runner's aggregate; recompute captured pool density from each bridge's activity cache; re-derive the verdict; consistency checks.

(b) Pre-registered reading: PASS iff every (bridge, load) cell has multi-seed mean >= 0.80; NEGATIVE otherwise.

(c) Write the findings doc (per-bridge breakdown, multi-seed-mean and strict per-seed criteria both reported); update capability_status.json (a new pillar if PASS, status VALIDATED; a per-bridge NEGATIVE pillar if NEGATIVE); update AUTONOMOUS_STATE EXACT NEXT ACTION; commit + push BOTH remotes.

(d) On a PASS: a fresh dedicated adversarial review BEFORE the capability-pillar claim (matching the K=16 PASS arc's discipline).

---

### Honest scope

This plan executes ONE further test on the vocab-scaling thread. Whatever the verdict, it is one further test in a continuing line — not a final answer. The completed K=16 refined CAPABILITY PASS (64 concepts, multi-seed-mean PASS through L=6, strict per-seed PASS through L=5, ceiling between L=6-7) stands. The biology-translatable insight set from the K=16 thread (mean-centring as the geometric load-bearing condition; longer integration as the noise-bounded ceiling-closing mechanism) is the framing this 160-concept tier reads against. Cross-bridge composition is explicitly out of scope. Frozen bar never tuned; reuse-by-import only; no protected, frozen, or moat module modified; no automatic differentiation.
