# Cluster B.2 — Striatal FSIs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Add a dedicated fast-spiking interneuron population to striatum providing millisecond-scale broadcast inhibition. Real BG striatum has ~1% FSIs (parvalbumin-positive, very fast-firing) that bias which action's MSN pool wins via fast convergent GABAergic inhibition. Different from the v3 MSN-MSN lateral inhibition we already have (slower, more local).

**Architecture:** Per-action FS pool (`str_FS_{N,E,S,W}`, ~5 neurons each). Each FS pool receives excitatory drive from its same-action cortex pool (`cortex_X → str_FS_X`). Each FS pool inhibits ALL striatal MSN pools, not just same-action (`str_FS_X → str_D{1,2}_Y` for all Y including X — broadcast inhibition). All pathways `plastic=False` (FSIs are static gating, not plastic).

**Tech stack:** Python 3.12, CuPy, pytest with `pytest.importorskip("cupy")`. Builder in `research/runners/g11_bg_runner.py:build_bg_brain_regions`; tests in `tests/test_g11_bg_runner_flags.py`.

**Reference:** Cluster B design at [`docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`](2026-04-28-cluster-b-striatal-microcircuit-design.md). B.1 partial signal at [`research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`](../../research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md).

---

## Task 1: Add `str_FS_X` regions + cortex→FS + FS→MSN broadcast pathways

**Files:**
- Modify: `research/runners/g11_bg_runner.py:build_bg_brain_regions`
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

Append to `tests/test_g11_bg_runner_flags.py`:

```python
def test_striatal_fsis_default_off():
    """When --enable-striatal-fsis is off, no str_FS_* regions exist."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(enable_striatal_fsis=False)
    fs_regions = [r for r in regions if r.name.startswith("str_FS_")]
    assert len(fs_regions) == 0, "FS regions should not exist when flag off"


def test_striatal_fsis_pathways_built():
    """When --enable-striatal-fsis is on:
       - 4 str_FS_X regions added (one per action)
       - 4 cortex_X → str_FS_X pathways added (excitatory drive)
       - 32 str_FS_X → str_D{1,2}_Y pathways added (broadcast inhibition,
         4 FS × 4 D-pool target × 2 D-types = 32). Includes same-action
         (X→X) since real FSIs broadcast indiscriminately, not just to
         non-self pools.
    All FS-related pathways are plastic=False (static gating)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(enable_striatal_fsis=True)

    fs_regions = [r for r in regions if r.name.startswith("str_FS_")]
    assert len(fs_regions) == 4, f"Expected 4 FS regions; got {len(fs_regions)}"
    fs_names = sorted(r.name for r in fs_regions)
    assert fs_names == ["str_FS_E", "str_FS_N", "str_FS_S", "str_FS_W"]

    cortex_to_fs = [p for p in pathways
                    if p.from_region.startswith("cortex_") and p.to_region.startswith("str_FS_")]
    assert len(cortex_to_fs) == 4, \
        f"Expected 4 cortex→FS pathways; got {len(cortex_to_fs)}"
    for p in cortex_to_fs:
        # Same action only: cortex_N→str_FS_N etc.
        assert p.from_region.split("_")[1] == p.to_region.split("_")[2], \
            f"cortex→FS pathway should be same-action; got {p.from_region}→{p.to_region}"
        assert not p.plastic, "cortex→FS should be plastic=False"

    fs_to_msn = [p for p in pathways
                 if p.from_region.startswith("str_FS_")
                 and (p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_"))]
    assert len(fs_to_msn) == 32, \
        f"Expected 32 FS→MSN pathways (4 FS × 4 D-pool × 2 D-types); got {len(fs_to_msn)}"
    for p in fs_to_msn:
        assert not p.plastic, "FS→MSN broadcast inhibition should be plastic=False"


def test_striatal_fsis_disabled_by_default():
    """build_bg_brain_regions default: no FS regions or pathways."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions()  # all defaults
    assert not any(r.name.startswith("str_FS_") for r in regions)
    assert not any(p.from_region.startswith("str_FS_") or p.to_region.startswith("str_FS_")
                   for p in pathways)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py -k "striatal_fsi" -v
```

Expected: FAIL — `enable_striatal_fsis` is not a `build_bg_brain_regions` kwarg.

**Step 3: Implementation**

a) **Add kwarg** to `build_bg_brain_regions` signature near other Cluster B-style flags (e.g. adjacent to `enable_motor_lateral_inhibition` if present, or `enable_bg_lateral_inhibition`):

```python
    enable_striatal_fsis: bool = False,
    n_striatal_fs_per_action: int = 5,
    cortex_to_fs_weight: float = 30.0,  # strong excitatory drive
    fs_to_msn_weight: float = 8.0,  # broadcast inhibition; tuned to suppress losers
```

b) **Add FS regions** in the region-creation loop (find where `gpe_X`, `gpi_X`, etc. are created — search for `str_D1_` region creation around line 302-311). Add a parallel block:

```python
    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons.
    # ~1% of striatal cells; PV-positive; broadcast inhibition.
    if enable_striatal_fsis:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"str_FS_{action}",
                n_neurons=n_striatal_fs_per_action,
                exc_fraction=0.0,  # all inhibitory (GABAergic interneurons)
                izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON",
                # FSIs have no internal recurrence; just receive cortex
                # input and broadcast to MSNs.
                internal_density=0.0,
            ))
```

c) **Add cortex → str_FS_X pathways** (excitatory, dense, plastic=False) — same-action only. Find the cortex→striatum pathway block (around line 486-520 where `cortex_X → str_D1_X` is built). Add a parallel block:

```python
    if enable_striatal_fsis:
        for cortex_action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_FS_{cortex_action}",
                density=1.0,
                weight_mean=cortex_to_fs_weight,
                weight_jitter=0.2,
                plastic=False,
            ))
```

d) **Add str_FS_X → str_D{1,2}_Y broadcast pathways** (inhibitory — auto-derived since str_FS regions have exc_fraction=0; plastic=False). Place this near the v3 lateral inhibition block (around line 525):

```python
    if enable_striatal_fsis:
        for fs_action in ACTION_NAMES:
            for str_action in ACTION_NAMES:
                # Broadcast: every FS pool inhibits every MSN pool,
                # including same-action (real FSIs don't selectively
                # spare same-action).
                for d_type in ("D1", "D2"):
                    pathways.append(RegionPathway(
                        from_region=f"str_FS_{fs_action}",
                        to_region=f"str_{d_type}_{str_action}",
                        density=1.0,  # dense within-pool
                        weight_mean=fs_to_msn_weight,
                        weight_jitter=0.2,
                        plastic=False,
                    ))
```

**Step 4: Verify pass + regression sweep**

```bash
pytest tests/test_g11_bg_runner_flags.py -k "striatal_fsi or bg_cross or pretraining or d1_d2" -v
```

Expected: 3 new tests pass + all earlier tests pass.

**Step 5: Commit + push**

```bash
git add research/runners/g11_bg_runner.py tests/test_g11_bg_runner_flags.py
git commit -m "feat(g11): Cluster B.2 — striatal fast-spiking interneurons

Adds 4 str_FS_{N,E,S,W} regions + 4 cortex→FS excitatory pathways +
32 FS→MSN broadcast inhibitory pathways behind --enable-striatal-fsis
flag (default off). All FS pathways plastic=False (static gating).

Real BG striatum has ~1% PV-positive FSIs providing fast convergent
GABAergic inhibition. Different from v3 MSN-MSN lateral (slower,
more local) — FSIs broadcast indiscriminately on a millisecond
timescale to bias which action's MSN pool wins.

Plan: docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md
Cluster: docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md"
git push origin main
```

---

## Task 2: CLI flag + kwarg plumbing on `run_moving_goal_episode`

**Files:**
- Modify: `research/runners/g11_bg_runner.py`
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write failing test**

```python
def test_striatal_fsis_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_striatal_fsis without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_striatal_fsis=True,
    )
```

**Step 2: Verify failure**

**Step 3: Implementation**

a) Add `enable_striatal_fsis: bool = False` to `run_moving_goal_episode` signature near `enable_d1_d2_asymmetry` (Cluster B siblings).

b) Pass through to `build_bg_brain_regions` call inside the function — find the existing call, add the kwarg.

c) Add argparse near `--enable-d1-d2-asymmetry`:

```python
ap.add_argument("--enable-striatal-fsis", action="store_true",
                help="Cluster B.2: striatal fast-spiking interneurons "
                     "(broadcast inhibition). See "
                     "docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.")
```

d) Pass-through in `main()` near the other arg passes.

**Step 4: Run test → PASS + regression sweep**

**Step 5: Commit:**
```
feat(g11): wire --enable-striatal-fsis CLI + kwarg
```

---

## Task 3: Biology probe — FSI suppression timing

**Files:**
- Create: `research/probes/striatal_fsi_probe.py`

**Goal:** Verify that adding FSIs causes faster MSN suppression in response to a competing cortex drive.

**Probe design:**

1. Build TWO bridges: one with `enable_striatal_fsis=False`, one with `=True`. All other flags identical.
2. Drive `cortex_N` strongly (simulate "agent picks N"). Drive `cortex_E` weakly (simulate "competitor that should be suppressed").
3. Run for ~200ms simulation time. Measure firing rates of `str_D1_N`, `str_D1_E`, `str_D2_N`, `str_D2_E` in 10ms bins.
4. Compute: time to "suppression" — first 10ms bin where `str_D1_E` firing rate drops below 50% of its peak.
5. Compare baseline vs +FSI. Expected: with FSIs, suppression-time is shorter.

**Output:** stdout summary + JSON at `research/findings/raw/striatal_fsi_probe/probe_results.json`.

**Step 5: Commit:**
```
feat(probe): striatal FSI broadcast-inhibition timing probe
```

---

## Task 4: Cheat-5 multi-goal re-eval

### 4a — v3 + B.1 + B.2 baseline (no cross-projections)

3 seeds, multi-goal. Should be ≤ 7.08 baseline. If significantly worse, B.2 hurts the cascade.

Bash:

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward --bg-lateral-inhibition \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --enable-d1-d2-asymmetry --enable-striatal-fsis \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_v3_b1_b2.json
done
```

### 4b — patch-matrix + B.1 + B.2 (the cheat-5 signal test)

Same flag set + `--bg-cross-projections --cross-projection-density 0.25 --cross-projection-topology-seed 0 --cross-projection-weight 5.0`.

### Decision matrix

Compared to patch-matrix + B.1 (7.62 ± 1.23):

| Result | Verdict |
|---|---|
| Mean ≤ 7.0 + std < 1.0 | **stronger cluster signal**; B.2 helps further |
| Mean similar to B.1 alone (~7.5 ± 1.0) | B.2 didn't add much; proceed to B.3 |
| Mean > 8.5 | B.2 introduced regression; debug |

---

## Task 5: Findings doc + propagation

After Task 4 lands:

- Create `research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md` (mirror B.1's findings template).
- Update CLAUDE.md, SCIENCE_ROADMAP §4.7, INDEX, CHANGELOG, memory.

## Done criteria

- [ ] 3 unit tests pass (Task 1) + 1 kwarg test (Task 2)
- [ ] Biology probe shows faster suppression with FSIs on (Task 3)
- [ ] v3+B.1+B.2 baseline ≤ 7.08 (Task 4a non-regression)
- [ ] patch-matrix + B.1+B.2 multi-goal numbers reported (Task 4b)
- [ ] Findings doc + propagation (Task 5)
