---
type: plan
status: live
date: 2026-05-25
---

# Direction 5 Implementation Plan — HYBRID sparse-distributed shared pool ON bio_brain_regions (5 bridges × V=16 = 80 cross-bridge concepts; biology-faithful + Kanerva sparse on a unified architecture)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build 5 hybrid bridges, each on a DIFFERENT vocab category (noun / verb / adjective / spatial / functional, V=16 each = 80 distinct cross-bridge concepts). Each bridge keeps the validated bio_brain_regions dedicated 200-neuron concept pools PLUS adds a NEW 2000-neuron shared sparse pool (per G.20 sparse pillar n=95) with per-concept K=100 random patterns. Cross-bridge probe (controller-only; Task 6) reads OUT of the shared sparse pool ONLY, where the geometry is sufficient per pillar n=95. Multi-seed [42, 43, 44]. Bar UNCHANGED at 0.80 multi-seed (same as pillars n=93+ + Directions Q, 3, 4).

**Architecture:** 5 separate SimulationBridges. Per bridge: build dedicated concept-pool substrate via `build_biological_brain_regions` (the protected builder; byte-unchanged) → add `shared_concept_pool` + `shared_FS` regions AND the `language_input → shared_concept_pool` plastic pathway AS ADDITIONAL ENTRIES in the `cfg.brain_regions` / `cfg.region_pathways` lists (the wrapper appends; the protected builder remains byte-unchanged). At training time, the SAME lang_input drive that teaches the dedicated pool ALSO drives the shared sparse pattern (via a one-time `apply_sparse_topographic_prior` at pillar n=95 strength: factor 10.0 / off-target 0.1). Cross-bridge probe captures activity FROM the shared_concept_pool region only.

**Cross-bridge probe (Task 4 + Task 6):** Mirror the pillar n=95 + Direction 4 cross-bridge probe primitive BYTE-UNCHANGED. The probe operates on cached per-bridge per-seed shared_concept_pool activity (per-word; uniform 2000-feature substrate across all 5 bridges). Parallel-matching decodes per-slot identification via batched phase similarity on FHRR-bound positional codes.

**Tech Stack:** CuPy GPU (Task 5 training only), NumPy (Task 4 cross-bridge probe; CPU-only per pillar n=95 + Direction 4 pattern), pre-registered fixed-threshold verdict module (Task 3).

**Reuse-by-import only.** No protected/frozen/moat modifications. `build_biological_brain_regions` remains byte-unchanged. `concept_pool_sparse_distributed.py` primitives (`generate_sparse_patterns`, `apply_sparse_topographic_prior`) reused byte-unchanged. The pillar n=95 + Direction 4 cross-bridge probe primitive is reused byte-unchanged (Task 4).

**Net-new modules (6, this subagent scaffolds first 4):**
1. `tests/test_direction_5_grounding.py` (Task 0)
2. `research/findings/raw/direction_5_vocab_spec.py` (Task 1)
3. `research/findings/raw/direction_5_bridge_builder.py` (Task 2)
4. `research/findings/raw/direction_5_verdict.py` + `tests/test_direction_5_verdict.py` (Task 3)
5. `research/findings/raw/direction_5_cross_bridge_probe.py` (Task 4; CPU-only; in scope for a follow-up subagent, NOT this one — keeps THIS commit small + verifiable)
6. `research/findings/raw/direction_5_5bridge_runner.py` (Task 5; GPU-bound; controller-only)

**Decisive run (Task 6) is CONTROLLER-ONLY** — orchestrates 5 bridges × 3 seeds = 15 bridge trainings (~5-6 hours GPU per design doc estimate), then cross-bridge probe + verdict emission.

---

## Pre-launch grep (DONE 2026-05-25, this subagent invocation)

Confirmed NET-NEW: no prior `direction_5_*.py` files exist; no prior "hybrid sparse on bio_brain_regions" work in findings/* or docs/plans/* (Grep on `hybrid.*sparse` OR `sparse.*bio_brain_regions` OR `shared.*pool.*bio_brain_regions` returned only OPTION 3 single-substrate validation and the existing G.20 sparse + bio_brain_regions cross-bridge files which are SEPARATE substrates from this hybrid). The G.20 sparse 5-bridge pattern (pillar n=95) is on a DIFFERENT substrate (sparse Kanerva SDM, no dedicated bio pools). The bio_brain_regions cross-bridge 5-bridge (Direction 4 NEGATIVE) is on a DIFFERENT substrate (dedicated bio pools, no sparse shared pool).

---

### Task 0: Grounding pin (intentionally RED until Tasks 1-3 land)

**Files:**
- Create: `tests/test_direction_5_grounding.py`

**Goal:** pin the contracts the Direction 5 subsystem MUST satisfy (module existence + threshold-frozen at the design-doc bar). RED on commit; turn GREEN as Tasks 1-3 land (Tasks 4 and 5 also have grounding tests).

The grounding tests follow the Direction Q + Direction 3 + Direction 4 pattern (file-existence assertions per task + threshold-frozen tests via importlib.util). They are RED on the Task 0 commit (none of the target files exist yet); they turn GREEN incrementally as later tasks land.

```python
# tests/test_direction_5_grounding.py
"""Direction 5 grounding pin — intentionally RED until Tasks 1-4 land.

Pins the contracts the Direction 5 HYBRID sparse-distributed shared
pool on bio_brain_regions runner MUST satisfy. Discipline pattern
matches Direction Q + Direction 3 + Direction 4 grounding pins.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_5_vocab_spec_module_exists():
    """Task 1: the 5-category vocab spec module exists."""
    ...


def test_direction_5_bridge_builder_module_exists():
    """Task 2: the per-bridge hybrid builder wrapper exists."""
    ...


def test_direction_5_verdict_module_exists():
    """Task 3: the verdict module exists."""
    ...


def test_direction_5_cross_bridge_probe_module_exists():
    """Task 4: cross-bridge probe runner exists (CPU-only scaffold).

    SKIP if Task 4 hasn't been added yet — Task 4 is NOT in scope for
    THIS subagent (Tasks 0-3 only); a follow-up subagent ships Task 4.
    """
    ...


def test_direction_5_vocab_spec_has_5_categories_v16_each():
    """Task 1: vocab spec exposes 5 category lists, each V=16 = 80 unique
    cross-bridge concepts (matches Direction 4; the hybrid substrate
    extends the SAME vocab to a new architecture)."""
    ...


def test_direction_5_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds match design-doc bar (0.80
    multi-seed; same as pillars n=93+ + Directions Q, 3, 4)."""
    ...


def test_direction_5_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID_MALFORMED from
    PASS / PARTIAL / NEGATIVE."""
    ...


def test_direction_5_bridge_builder_has_five_functions():
    """Task 2: builder wrapper exposes 5 functions, one per bridge."""
    ...


def test_direction_5_bridge_builder_uses_protected_builder_byte_unchanged():
    """Task 2: per-bridge builder must REUSE build_biological_brain_regions
    byte-unchanged + reuse G.20 sparse primitives byte-unchanged."""
    ...
```

**Expected on this subagent's commit (after Tasks 1-3 land alongside):** 8/9 PASSED, 1 SKIPPED (the Task 4 module-existence test which legitimately skips since Task 4 is out of scope for this subagent invocation).

---

### Task 1: Vocab spec module (5 × V=16 = 80 unique cross-bridge concepts)

**Files:**
- Create: `research/findings/raw/direction_5_vocab_spec.py`

**Goal:** Frozen 5-category vocab lists matching the Direction 4 vocab (mirror — extends the SAME vocab to a NEW architecture so any PASS comparison is byte-equivalent on the concept set). Global uniqueness guaranteed; no runtime override path.

**Pre-registered word lists** (IDENTICAL to Direction 4 vocab spec, deliberately, so the Direction 5 hybrid test is directly comparable to the Direction 4 NEGATIVE result on the SAME concept set; the only architectural difference is the substrate, not the vocab):

- BridgeA (nouns, 16): apple, river, dog, cat (v14 baseline) + tree, bird, sun, moon, book, chair, house, wheel, ball, cup, lamp, road
- BridgeB (verbs, 16): go, come, stop, look (v14 baseline) + walk, run, eat, sleep, sit, stand, jump, climb, throw, catch, lift, pull
- BridgeC (adjectives, 16): big, small, hot, cold (v14 baseline) + fast, slow, bright, dark, loud, quiet, sweet, sour, heavy, light, sharp, soft
- BridgeD (spatial, 16): north, east, south, west, up, down, left, right, in, out, near, far, top, bottom, center, side
- BridgeE (functional, 16): i, you, he, she, the, a, and, or, with, for, this, that, these, those, what, when

**Total = 80 unique cross-bridge concepts.** Global uniqueness enforced by Task 0 grounding test.

The vocab module mirrors `direction_4_vocab_spec.py` byte-pattern with all `DIRECTION_4_*` renamed to `DIRECTION_5_*`. Module is stdlib + typing only.

---

### Task 2: Per-bridge HYBRID builder wrapper (5 functions, CPU-only spec)

**Files:**
- Create: `research/findings/raw/direction_5_bridge_builder.py`

**Goal:** 5 pure constructor functions — `build_direction_5_bridge_A_nouns`, `build_direction_5_bridge_B_verbs`, `build_direction_5_bridge_C_adj`, `build_direction_5_bridge_D_spatial`, `build_direction_5_bridge_E_functional`. Each returns a fresh SimulationBridge configured with that bridge's V=16 vocab on the HYBRID substrate (bio dedicated pools + shared sparse pool). No training in this module — just construction.

**Hybrid construction recipe** (per bridge):

1. **Dedicated substrate construction** (REUSED BYTE-UNCHANGED):
   - Call `build_biological_brain_regions(...)` with the bridge's V=16 vocab in the
     appropriate slot (`noun_pool_names` for A / D / E; `verb_pool_names` for B;
     `adjective_pool_names` for C). v14/v16 production recipe defaults.
   - Returns `(dedicated_regions, dedicated_pathways)` lists.

2. **Shared sparse substrate addition** (NEW):
   - Construct 2 additional `BrainRegion` objects:
     - `shared_concept_pool` (2000 neurons, exc_fraction=0.8, internal_density=0.05,
       weak dynamics 0.3/0.8; matches G.20 sparse pillar n=95)
     - `shared_FS` (300 neurons, exc_fraction=0.0, density 0.0 — WTA via the
       below pathways; matches G.20 sparse)
   - Construct 3 additional `RegionPathway` objects:
     - `language_input → shared_concept_pool` (density 0.30, weight_mean 3.0, jitter 0.5,
       plastic=True, plasticity_gate="language_input_to_shared")
     - `shared_concept_pool → shared_FS` (density 0.30, weight 1.0, jitter 0.2,
       plastic=False)
     - `shared_FS → shared_concept_pool` (density 0.30, weight 4.0, jitter 0.2,
       plastic=False)
   - APPEND these to `dedicated_regions` and `dedicated_pathways`. (CRITICAL: the
     protected builder is NOT touched; the wrapper assembles the combined
     `cfg.brain_regions` / `cfg.region_pathways` list AFTER the protected builder
     returns.)

3. **Config + bridge construction:**
   - Build `CoreSimConfig` with combined regions / pathways.
   - v14/v16 production parameters: NMDA on (tau_decay 100ms), STDP w_max 8.0,
     fast_spike_reset=True, no Hebbian / structural / per-type STP.

4. **Sparse pattern + topographic prior** (NEW, applied POST-construction):
   - Generate per-concept K=100 sparse patterns via
     `generate_sparse_patterns(n_concepts=16, n_pool=2000, pattern_size=100, seed=seed)`
     (REUSED BYTE-UNCHANGED from `research/runners/concept_pool_sparse_distributed.py`).
   - Apply `apply_sparse_topographic_prior(bridge, n_concepts=16, n_lang_input=2048,
     sparse_patterns, sparsity=0.05, topographic_factor=10.0, off_target_factor=0.1)`
     (REUSED BYTE-UNCHANGED). This boosts language_input → shared_concept_pool
     weights into each concept's K=100 sparse pattern at pillar n=95 strength.
   - Return both the bridge AND the per-concept sparse patterns (the runner / probe
     needs the patterns to verify cross-bridge probe activity reads from the
     RIGHT sparse pattern per concept).

**v14/v16 production parameters** (pinned in wrapper, mirrored from Direction 4 builder):
- n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24
- Weak concept dynamics (density=0.05, exc=0.3, inh=0.8)
- Motor canon (density=0.10, exc=2.0, inh=4.0)
- enable_motor_fs=True, enable_language_output=True
- NMDA on; tau_decay=100ms
- stdp_w_max=8.0

**Sparse substrate parameters** (pinned in wrapper, mirrored from G.20 sparse pillar n=95):
- n_shared_pool=2000, n_shared_fs=300
- pattern_size=100 (K=100), sparsity=0.05 (~102 active lang_input per drive)
- topographic_factor=10.0, off_target_factor=0.1

**Deferred imports:** the wrapper MUST NOT import cupy at module load time. All sim/sim-bridge/sparse-primitive imports are inside the constructor function bodies.

---

### Task 3: Frozen verdict module + adversarial test matrix (≥12 cases)

**Files:**
- Create: `research/findings/raw/direction_5_verdict.py`
- Create: `tests/test_direction_5_verdict.py`

**Goal:** pure stdlib-only verdict function. Mirror `direction_4_verdict.py` byte-pattern with `DIRECTION_4_*` → `DIRECTION_5_*` rename throughout.

**Pre-registered thresholds (frozen):**
- `_DIRECTION_5_OB_MIN = 0.80`
- `_DIRECTION_5_OI_MIN = 0.80`
- `_DIRECTION_5_LOADS = (2, 3, 5)`
- `_DIRECTION_5_MIN_SEEDS = 3`

**Test matrix (≥12 cases in `tests/test_direction_5_verdict.py`):** mirror `tests/test_direction_4_verdict.py` byte-pattern with module-name + tag-constant renames throughout.

---

### Task 4: Cross-bridge probe runner (CPU-only) — NOT in this subagent's scope

**Files (in scope for a FOLLOW-UP subagent):**
- Create: `research/findings/raw/direction_5_cross_bridge_probe.py`

**Goal:** reuse the pillar n=95 + Direction 4 cross-bridge parallel-matching mode-unification probe pattern BYTE-UNCHANGED in primitive. Operates on cached per-bridge per-seed activity from the shared_concept_pool ONLY (the bio dedicated pools are NOT in the cross-bridge probe activity vector). Computes per-load OB + OI accuracy and writes JSON in the verdict module's expected shape.

The key change vs `direction_4_cross_bridge_probe.py`: activity is captured from the
NEW `shared_concept_pool` region instead of from each bridge's dedicated noun /
verb / adjective pool union. The d_act = 2000 per bridge (uniform across all 5
bridges) which is also a cleaner substrate than D4's variable d_act per bridge.

**Status: NOT in this subagent commit.** A follow-up subagent (or controller-only
scaffolding) ships Task 4 once Task 5 GPU runner produces cached activity to test against.

---

### Task 5: CONTROLLER-ONLY 5-bridge multi-seed training launch (GPU-bound)

**NOT a subagent task. Controller orchestrates.**

After this subagent's commit lands + GPU is free, controller (or watchdog spawn) launches:

```bash
python -u -m research.findings.raw.direction_5_5bridge_runner \
    --seeds 42 43 44 \
    > research/findings/raw/direction_5_5bridge_training.log 2>&1 &
```

Expected wall ~17-20 min/bridge train (per v14/v16 + small overhead for the additional
sparse pathway init); 5 bridges × 3 seeds = 15 bridge trainings = ~5-6 hours total.

This task is mentioned in the implementation plan for completeness only. Subagent does NOT
create the runner module itself in this invocation.

---

### Task 6: CONTROLLER-ONLY decisive cross-bridge probe + verdict emission

**NOT a subagent task. Controller orchestrates after Tasks 4 + 5 complete.**

1. Load per-bridge per-seed trained shared_concept_pool activity dumps from Task 5
2. Run `direction_5_cross_bridge_probe.run_direction_5_cross_bridge_probe()`
3. Compute verdict via `direction_5_verdict.compute_verdict(per_seed)`
4. Emit `research/findings/raw/direction_5_decisive.json` with verdict + per-seed cells
5. Adversarial-reviewer subagent pass
6. If PASS → pillar n=106 candidate; honest propagation both remotes

---

## Subagent commit scope (THIS subagent invocation)

Per the discipline binding (CPU-only; no GPU competition with anything currently running), this subagent ships Tasks 0-3 in a single commit:

**Files:**
- `tests/test_direction_5_grounding.py` (Task 0)
- `research/findings/raw/direction_5_vocab_spec.py` (Task 1)
- `research/findings/raw/direction_5_bridge_builder.py` (Task 2)
- `research/findings/raw/direction_5_verdict.py` (Task 3)
- `tests/test_direction_5_verdict.py` (Task 3)
- `docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-design.md` (design)
- `docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-implementation.md` (this plan)

**Commit message:**

```
Direction 5 (NEW hybrid sparse-distributed on bio_brain_regions): design + writing-plans + Tasks 0-3 CPU-only scaffolding
```

Task 4 scaffold + Task 5/6 controller-only runs land in subsequent subagent commits / controller actions.

---

## Discipline (binding throughout)

- Bar UNCHANGED: `_DIRECTION_5_OB_MIN=0.80`, `_DIRECTION_5_OI_MIN=0.80`, `_DIRECTION_5_LOADS=(2,3,5)`, `_DIRECTION_5_MIN_SEEDS=3`. Set ONCE in `direction_5_verdict.py` at Task 3; never tuned by results.
- No protected / frozen / moat modification. `build_biological_brain_regions` remains byte-unchanged. `concept_pool_sparse_distributed.py` primitives reused byte-unchanged. Direction 5 uses its OWN per-bridge HYBRID constructor wrappers.
- No autograd.
- GPU/CuPy for Task 5 training only (controller-only). NumPy for Task 4 cross-bridge probe (CPU-only per pillar n=95 + Direction 4 pattern).
- Honest propagation EVERY outcome both remotes.
- Pre-launch grep confirmed (this subagent invocation): no prior hybrid sparse-on-bio_brain_regions work; G.20 sparse 5-bridge is on DIFFERENT substrate; Direction 4 bio cross-bridge is on DIFFERENT substrate.
- Reviewer-style scrutiny applied at the time of result (Task 6), not deferred.
- The hybrid wrapper MUST NOT import cupy at module load time (CPU-light import).
- The grounding pin's "uses protected builder" test verifies BOTH `build_biological_brain_regions` AND the sparse primitives (`generate_sparse_patterns`, `apply_sparse_topographic_prior`) are imported.

---

## Post-Task chain (per verdict)

- **DIRECTION_5_PASS**: write findings doc; dispatch adversarial reviewer subagent; if CLEAR record pillar n=106; update AUTONOMOUS_STATE + capability_status.json. The hybrid architecture is validated; conversational vocabulary substrate extended to 80 biology-faithful + sparse-distributed cross-bridge concepts on a unified architecture (FIRST such validated architecture in the project).
- **DIRECTION_5_PARTIAL**: precise per-load breakdown; biology-translatable comparison to G.20 sparse n=95 + Direction 4 NEGATIVE (which mechanism is the bottleneck: dedicated pool geometry, sparse pool geometry, or dedicated → shared coupling?). Likely PARTIAL outcome triggers Approach C learned dedicated → shared projection (the next iterative refinement).
- **DIRECTION_5_NEGATIVE**: hybrid composition fails; sparse readout doesn't rescue cross-bridge; the dual-substrate hypothesis at this scale is falsified. Diagnose: capture dedicated pool activity side-by-side and run the SAME probe primitive on it to confirm the dedicated activity ALSO doesn't carry cross-bridge information (separating "substrate-bound" from "decoder-bound" failure modes).
- **DIRECTION_5_VOID_MALFORMED**: instrument-validity failure; diagnose recorded JSON shape; do not propagate as a pillar.
