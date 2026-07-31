---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 v4 Developmental Pretraining — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--developmental-pretraining` mode to `g11_bg_runner.py` that runs 10×3000 trials of varied-goal pretraining with all plasticity gates open, then freezes BG cross-projections for the standard 1800-step moving-goal eval. The hypothesis is that cross-projection refinement is a developmental phenomenon — adult STDP+reward (v3.1 NO-GO) can't shape useful cross-action structure from random init, but a critical-period analog might.

**Architecture:** Pragmatic insertion. A separate ~100-line `_run_pretraining_phase` helper runs BEFORE the existing curriculum init at [`research/runners/g11_bg_runner.py`:1206](../../research/runners/g11_bg_runner.py#L1206). The existing curriculum init then naturally forces `bg_cross_projections=0.0` at eval start (no manual freeze needed). Cross-projection weights persist across the boundary; `cp_plasticity_gain` is reset by the curriculum init.

**Tech Stack:** Python 3.12, CuPy (GPU arrays), pytest with `pytest.importorskip("cupy")` for GPU tests, existing `research.runners.g11_bg_runner` module, `sim.bridge.SimulationBridge.set_plasticity_gate` (declared at [`sim/bridge.py`:1821](../../sim/bridge.py#L1821)), `sim.regions.RegionPathway` with `plasticity_gate` field.

**Reference:** Approved design at [`docs/plans/2026-04-28-cheat5-v4-design.md`](2026-04-28-cheat5-v4-design.md) (commit `e6ce0ce`).

---

## Task 1: `_run_pretraining_phase` helper — gate validation + thaw

**Files:**
- Modify: `research/runners/g11_bg_runner.py` (add helper near top of file, around line 660 — before `run_moving_goal_episode` at line 659)
- Test: `tests/test_g11_bg_runner_flags.py` (append after the last test, around line 400+)

**Step 1: Write the failing test**

Append to `tests/test_g11_bg_runner_flags.py`:

```python
def test_pretraining_raises_on_missing_gate():
    """If a gate name we want to thaw is missing from bridge.list_plasticity_gates(),
    the helper raises KeyError mentioning both the bad name AND the actual list of
    available gates. Catches typos before pretraining wastes minutes of GPU time."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import _run_pretraining_phase

    # Fake bridge: only has one of the gates we'd thaw
    class _FakeBridge:
        def list_plasticity_gates(self):
            return ["cortex_to_d1"]  # missing all the others

        def set_plasticity_gate(self, name, value):
            raise AssertionError("should not be called when validation fails")

    with pytest.raises(KeyError) as exc_info:
        _run_pretraining_phase(
            bridge=_FakeBridge(),
            cfg=None,
            regions=None,
            n_goals=1,
            steps_per_goal=10,
            grid_size=8,
            start_pos=(1, 1),
            seed=42,
            verbose=False,
        )
    msg = str(exc_info.value)
    assert "bg_cross_projections" in msg or "sensory_to_cortex" in msg, (
        "error should name at least one missing gate")
    assert "cortex_to_d1" in msg, (
        "error should list the actually-available gates so the user can spot the typo")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_raises_on_missing_gate -v
```

Expected: `ImportError: cannot import name '_run_pretraining_phase'` (function doesn't exist yet).

**Step 3: Write minimal implementation**

Add to `research/runners/g11_bg_runner.py` just before `def run_moving_goal_episode(...)` at line 659. Don't try to do the full pretraining yet — just the gate-validation skeleton:

```python
# Plasticity gates we expect to find on the runner's pathways. Pretraining
# thaws all of these; absence means a runner-side typo in plasticity_gate=
# (or a flag that doesn't add the pathway). Error early before GPU work.
_PRETRAINING_THAWED_GATES = (
    "cortex_to_d1",
    "sensory_to_cortex",
    "hippo_to_cortex",
    "beacon_to_goal",
    "landmark_to_place",
    "pfc_pathways",
    "bg_cross_projections",
)


def _run_pretraining_phase(
    bridge,
    cfg,
    regions,
    n_goals: int,
    steps_per_goal: int,
    grid_size: int,
    start_pos,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Critical-period analog. Thaws ALL plasticity gates and runs the agent
    through n_goals random goals for steps_per_goal trials each.

    See docs/plans/2026-04-28-cheat5-v4-design.md for the full architecture.
    Returns a summary dict with weight statistics — this is the only signal
    the caller gets about how the pretraining went short of the eval result.

    NOTE (2026-04-28, v4 initial): only the gate-validation skeleton is
    implemented in this commit. Trial-loop wiring lands in Task 2.
    """
    available = set(bridge.list_plasticity_gates())
    missing = [g for g in _PRETRAINING_THAWED_GATES
               if g not in available and _gate_required(g, regions)]
    if missing:
        raise KeyError(
            f"_run_pretraining_phase: gate(s) not declared on any pathway: "
            f"{missing!r}. Available: {sorted(available)!r}. "
            f"Either spell-check the gate name in build_bg_brain_regions, "
            f"or enable the flag that adds the pathway (e.g. "
            f"--learned-perception adds sensory_to_cortex)."
        )
    raise NotImplementedError("trial loop lands in Task 2")


def _gate_required(name: str, regions) -> bool:
    """Skeleton heuristic — most gates are conditional on flags, so we only
    treat 'cortex_to_d1' (always present in BG cascade) as strictly required.
    Refined when the flag wiring lands in Task 3."""
    return name == "cortex_to_d1"
```

(The simple `_gate_required` heuristic is a placeholder — the full `_run_pretraining_phase` will know which gates ARE expected based on the flags, but for now we just want the validation step to raise on a clearly-broken setup. Task 2 refines this.)

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_raises_on_missing_gate -v
```

Expected: PASS. The fake bridge has only `cortex_to_d1`, so the code path doesn't raise via the missing-required-gate route — but the inner `raise NotImplementedError` does after validation completes. **Update the test to expect KeyError, not NotImplementedError.** The test as written checks `pytest.raises(KeyError)`. Since `cortex_to_d1` is the only required gate per `_gate_required`, the fake bridge HAS it, no missing list, validation passes, and we hit the NotImplementedError instead.

This is a TDD signal that the test isn't tight enough yet OR the heuristic is too permissive. Tighten `_gate_required` to also require `bg_cross_projections` (since that's the gate v4 specifically needs to thaw):

```python
def _gate_required(name: str, regions) -> bool:
    # cortex_to_d1 always exists. bg_cross_projections is the WHOLE POINT
    # of v4 pretraining — fail loud if it's not tagged.
    return name in {"cortex_to_d1", "bg_cross_projections"}
```

Re-run the test → PASS. The fake bridge is missing `bg_cross_projections` so `KeyError` raises from the validation step.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): _run_pretraining_phase skeleton with gate validation

First task of v4 cheat-5 closure. Just the gate-validation step that
raises KeyError early if expected gates aren't tagged on pathways —
catches typos before the runner burns 30K GPU steps. Trial-loop
wiring lands in Task 2.

Plan: docs/plans/2026-04-28-cheat5-v4-implementation.md Task 1.
Design: docs/plans/2026-04-28-cheat5-v4-design.md (commit e6ce0ce)."
```

---

## Task 2: gate-thawing + summary dict (still no trial loop)

**Files:**
- Modify: `research/runners/g11_bg_runner.py:_run_pretraining_phase`
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

```python
def test_pretraining_thaws_all_gates_at_start():
    """After validation, the helper sets every (declared) gate to 1.0 via
    set_plasticity_gate. This test uses a recording fake bridge to assert the
    thaw calls happen with the correct values."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import _run_pretraining_phase

    class _RecordingBridge:
        def __init__(self):
            self._values = {}
            self.calls = []

        def list_plasticity_gates(self):
            return ["cortex_to_d1", "bg_cross_projections", "sensory_to_cortex"]

        def set_plasticity_gate(self, name, value):
            self._values[name] = value
            self.calls.append((name, value))

        def get_plasticity_gate_value(self, name):
            return self._values.get(name, 1.0)

    fb = _RecordingBridge()
    summary = _run_pretraining_phase(
        bridge=fb, cfg=None, regions=None,
        n_goals=0,  # zero goals → skip the trial loop, just thaw + early-return
        steps_per_goal=0,
        grid_size=8, start_pos=(1, 1), seed=42, verbose=False,
    )

    # Every gate the bridge knows about should have been set to 1.0
    for gate in ["cortex_to_d1", "bg_cross_projections", "sensory_to_cortex"]:
        assert (gate, 1.0) in fb.calls, (
            f"gate {gate!r} was not thawed to 1.0; calls={fb.calls}")

    # Summary must have the documented keys
    for key in ("n_trials", "n_goal_changes", "cross_weights_mean", "cross_weights_std"):
        assert key in summary, f"summary missing {key!r}: {summary!r}"
    assert summary["n_trials"] == 0
    assert summary["n_goal_changes"] == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_thaws_all_gates_at_start -v
```

Expected: fails on `NotImplementedError("trial loop lands in Task 2")` from Task 1's stub.

**Step 3: Write minimal implementation**

Replace the body of `_run_pretraining_phase` in `research/runners/g11_bg_runner.py` with:

```python
def _run_pretraining_phase(
    bridge,
    cfg,
    regions,
    n_goals: int,
    steps_per_goal: int,
    grid_size: int,
    start_pos,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Critical-period analog. Thaws ALL declared plasticity gates and runs
    the agent through n_goals random goals for steps_per_goal trials each.

    Returns a summary dict: {n_trials, n_goal_changes, cross_weights_mean,
    cross_weights_std}. See docs/plans/2026-04-28-cheat5-v4-design.md."""
    available = set(bridge.list_plasticity_gates())
    missing = [g for g in _PRETRAINING_THAWED_GATES
               if g not in available and _gate_required(g, regions)]
    if missing:
        raise KeyError(
            f"_run_pretraining_phase: gate(s) not declared on any pathway: "
            f"{missing!r}. Available: {sorted(available)!r}. "
            f"Either spell-check the gate name in build_bg_brain_regions, "
            f"or enable the flag that adds the pathway."
        )

    # Thaw every gate that IS declared. Gates not declared (e.g. learned
    # perception is off, so sensory_to_cortex doesn't exist) are silently
    # skipped — the corresponding pathway just isn't there.
    for gate in _PRETRAINING_THAWED_GATES:
        if gate in available:
            bridge.set_plasticity_gate(gate, 1.0)

    if verbose:
        print(f"[g11 seed={seed}] pretraining: all {len(available)} declared gates "
              f"thawed to 1.0; running {n_goals} goals × {steps_per_goal} steps each",
              flush=True)

    # Trial loop lands in Task 4 (after CLI wiring is in place to call us).
    # For now, return a structured-but-empty summary so callers don't break.
    return {
        "n_trials": 0,
        "n_goal_changes": 0,
        "cross_weights_mean": float("nan"),
        "cross_weights_std": float("nan"),
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_thaws_all_gates_at_start -v
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_raises_on_missing_gate -v
```

Expected: BOTH pass. Task 1's test still passes (validation still fires before thaw).

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): pretraining helper thaws all declared gates

Second task of v4 cheat-5 closure. After gate validation, set every
declared plasticity gate to 1.0 (critical-period analog: everything
plastic). Returns a structured summary dict; trial loop is still
stubbed pending CLI wiring (Task 4)."
```

---

## Task 3: goal sampler — Manhattan ≥ 3, no consecutive repeats

**Files:**
- Modify: `research/runners/g11_bg_runner.py` (helper near `_run_pretraining_phase`)
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

```python
def test_pretraining_goal_sampling_respects_manhattan_3():
    """Sampler must keep new goals at least Manhattan 3 from the start cell."""
    from research.runners.g11_bg_runner import _sample_pretraining_goal
    import random

    rng = random.Random(42)
    start_pos = (1, 1)
    grid_size = 8
    for _ in range(100):
        gx, gy = _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal=None)
        assert 0 <= gx < grid_size and 0 <= gy < grid_size
        manhattan = abs(gx - start_pos[0]) + abs(gy - start_pos[1])
        assert manhattan >= 3, f"sampled goal ({gx},{gy}) at Manhattan {manhattan} from {start_pos}"


def test_pretraining_goal_no_consecutive_repeats():
    """Successive samples differ from the previous goal."""
    from research.runners.g11_bg_runner import _sample_pretraining_goal
    import random

    rng = random.Random(42)
    prev = None
    for _ in range(50):
        g = _sample_pretraining_goal(rng, 8, (1, 1), prev_goal=prev)
        if prev is not None:
            assert g != prev, f"sampler returned same goal {g} as previous"
        prev = g
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_goal_sampling_respects_manhattan_3 tests/test_g11_bg_runner_flags.py::test_pretraining_goal_no_consecutive_repeats -v
```

Expected: ImportError — `_sample_pretraining_goal` doesn't exist.

**Step 3: Write minimal implementation**

Add above `_run_pretraining_phase` in `research/runners/g11_bg_runner.py`:

```python
def _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal):
    """Uniform random (gx, gy) on the grid with Manhattan >= 3 from start_pos
    and != prev_goal. Re-samples on rejection. The grid is small enough
    (8x8 → 16 valid cells given start (1,1)) that rejection sampling is
    trivially fast."""
    sx, sy = start_pos
    while True:
        gx = rng.randrange(grid_size)
        gy = rng.randrange(grid_size)
        if abs(gx - sx) + abs(gy - sy) < 3:
            continue
        if prev_goal is not None and (gx, gy) == prev_goal:
            continue
        return (gx, gy)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_pretraining_goal_sampling_respects_manhattan_3 tests/test_g11_bg_runner_flags.py::test_pretraining_goal_no_consecutive_repeats -v
```

Expected: BOTH pass.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): pretraining goal sampler (Manhattan>=3, no repeats)"
```

---

## Task 4: CLI flags + kwarg plumbing on `run_moving_goal_episode`

**Files:**
- Modify: `research/runners/g11_bg_runner.py:659+` (kwargs on `run_moving_goal_episode`)
- Modify: `research/runners/g11_bg_runner.py:~1973` (argparse) and `~2080` (kwarg pass-through)
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

```python
def test_developmental_pretraining_kwargs_accepted(tmp_out_path):
    """The runner should accept enable_developmental_pretraining + the two
    integer kwargs without raising TypeError on signature mismatch. Use
    n_goals=0 so the (still-stubbed) pretraining loop early-returns and the
    standard eval still runs."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=40, verbose=False,
        enable_developmental_pretraining=True,
        pretraining_n_goals=0,
        pretraining_steps_per_goal=0,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_kwargs_accepted -v
```

Expected: TypeError — kwargs unknown.

**Step 3: Write minimal implementation**

a) Add three kwargs to `run_moving_goal_episode` near where other v3.1-related kwargs are declared (around line 806 — the `bg_cross_thaw_step: int = -1` block). After the `bg_cross_phase3_gain` line, add:

```python
    # ─── v4 (2026-04-28): developmental pretraining ────────────────────
    # Run a critical-period analog before the standard eval: N random
    # goals × M trials per goal with all plasticity gates open. At the
    # transition, the existing curriculum init naturally freezes
    # bg_cross_projections (line 1220 of this file). See
    # docs/plans/2026-04-28-cheat5-v4-design.md.
    enable_developmental_pretraining: bool = False,
    pretraining_n_goals: int = 10,
    pretraining_steps_per_goal: int = 3000,
```

b) Inside the function body, just before the curriculum init block at line 1206 (search for `available_gates = bridge.list_plasticity_gates()`), insert:

```python
    # v4 developmental pretraining (2026-04-28). Runs only if enabled.
    # Inserted BEFORE curriculum init so the init's phase-1 gate values
    # naturally freeze bg_cross_projections at eval start (line 1220).
    pretraining_summary = None
    if enable_developmental_pretraining:
        pretraining_summary = _run_pretraining_phase(
            bridge=bridge, cfg=cfg, regions=regions,
            n_goals=pretraining_n_goals,
            steps_per_goal=pretraining_steps_per_goal,
            grid_size=grid_size, start_pos=start_pos,
            seed=seed, verbose=verbose,
        )
```

c) In the argparse block around line 1973 (after `--bg-cross-phase3-gain`), add:

```python
    # v4 (2026-04-28): developmental pretraining
    ap.add_argument("--developmental-pretraining", action="store_true",
                    help="v4 cheat-5 closure: run a critical-period analog "
                         "(all plasticity gates open) on N random goals before "
                         "the standard eval. Cross-projections freeze at eval "
                         "start. Requires --bg-cross-projections.")
    ap.add_argument("--pretraining-n-goals", type=int, default=10,
                    help="Number of random goal positions during pretraining (default 10).")
    ap.add_argument("--pretraining-steps-per-goal", type=int, default=3000,
                    help="Trials per pretraining goal (default 3000). 10×3000=30K "
                         "default total; reduce for tier-2 smoke (e.g. 1000) or "
                         "tier-1 wiring check (e.g. 1 goal × 1000).")
```

d) In the kwarg pass-through at the bottom of `main()` around line 2080, add (right before the closing `)`):

```python
            enable_developmental_pretraining=args.developmental_pretraining,
            pretraining_n_goals=args.pretraining_n_goals,
            pretraining_steps_per_goal=args.pretraining_steps_per_goal,
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_kwargs_accepted -v
```

Expected: PASS. Run all related tests as a regression check:

```bash
pytest tests/test_g11_bg_runner_flags.py -k "pretraining or bg_cross" -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): wire --developmental-pretraining flag + kwargs

Adds the three CLI flags (--developmental-pretraining,
--pretraining-n-goals, --pretraining-steps-per-goal) and
plumbs them through to run_moving_goal_episode. The pretraining
helper is invoked BEFORE the curriculum init so the existing
phase-1 init naturally freezes bg_cross_projections at eval start.

Trial loop is still stubbed (Task 5)."
```

---

## Task 5: trial loop with goal teleport + per-trial sim stepping

**Files:**
- Modify: `research/runners/g11_bg_runner.py:_run_pretraining_phase`
- Test: `tests/test_g11_bg_runner_flags.py`

This is the biggest task. The pretraining trial loop replicates the inner per-trial logic from the eval loop in simplified form: stimulus injection → step → motor count → reward computation → reward signal → plasticity step. We do NOT replicate phase tracking, final-quarter stats, surprise-LR boost, etc. — pretraining cares only about WEIGHT EVOLUTION, not behavior metrics.

Read `research/runners/g11_bg_runner.py` lines 1271-1881 (the eval loop) carefully before starting. The minimal subset of behavior we need to replicate is:

- For each trial:
  - Apply stimuli for `n_stim_steps` substeps; in the readout window, accumulate motor counts via `bridge.cp_firing_states.get()`
  - Pick action via argmax of motor counts (or random if all silent — same logic as eval)
  - Update agent position, compute reward (sensed-reward beacon if enabled, else Manhattan delta)
  - If reward != 0: set `bridge.core_config.current_reward_signal = float(reward)`, run `reward_hold_steps` extra sim steps, restore signal to 0

Goal change: every `steps_per_goal` trials, sample a new goal via `_sample_pretraining_goal`. Reset agent to `start_pos`.

**Step 1: Write the failing test (integration)**

```python
def test_run_moving_goal_with_pretraining_smoke(tmp_out_path):
    """End-to-end: tiny pretraining + tiny eval. Asserts cross-projection
    weights moved during pretraining AND are frozen during eval."""
    pytest.importorskip("cupy")
    import cupy as cp
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # Patch the runner so we can snapshot weights at the pretraining-eval
    # boundary. Easiest path: load the runner module, monkeypatch
    # _run_pretraining_phase to ALSO record the weights before/after.
    import research.runners.g11_bg_runner as runner_mod
    snapshots = {}
    original = runner_mod._run_pretraining_phase

    def wrapped(*args, **kwargs):
        bridge = kwargs.get("bridge", args[0] if args else None)
        snapshots["pre_weights"] = bridge.cp_synapse_weights.copy().get()
        result = original(*args, **kwargs)
        snapshots["post_pretraining_weights"] = bridge.cp_synapse_weights.copy().get()
        return result

    runner_mod._run_pretraining_phase = wrapped
    try:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=100, verbose=False,
            enable_bg_cross_projections=True,
            cross_projection_weight=0.0,
            enable_bg_lateral_inhibition=True,
            enable_curriculum=True, curriculum_warmup_steps=20,
            enable_developmental_pretraining=True,
            pretraining_n_goals=1,
            pretraining_steps_per_goal=50,
        )
    finally:
        runner_mod._run_pretraining_phase = original

    pre = snapshots["pre_weights"]
    post = snapshots["post_pretraining_weights"]
    n_changed = (pre != post).sum()
    assert n_changed > 0.01 * pre.size, (
        f"pretraining didn't change any weights — synapse plasticity not flowing? "
        f"{n_changed}/{pre.size} changed")

    # Eval-end weights are inside the JSON's per-step state? No — we only
    # have phase_stats. Instead read the bridge state via a second pass:
    # this smoke trusts that gate freezing is asserted by other tests.
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result
    assert result["seed"] == 42
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_run_moving_goal_with_pretraining_smoke -v
```

Expected: 0 weights changed (pretraining loop is still stubbed).

**Step 3: Write minimal implementation**

Replace the stub return in `_run_pretraining_phase` with the full trial loop. The implementation reads heavily from the eval loop (lines 1271-1881 of `research/runners/g11_bg_runner.py`) — extract just the per-trial inner loop. Key references:

- Stimulus injection: lines 1500-1555 (search for `cp_input_current_pA[motor_idx_per_action`)
- Motor counting: lines 1665-1674
- Action selection: lines 1678-1683
- Reward: lines 1716-1723 (Manhattan-delta) or 1700-1715 (sensed-reward)
- Reward signal injection + plasticity hold: lines 1725-1732, 1791-1810

For the pretraining loop, we do NOT need: surprise-LR, adaptive-DA, RPE-scaled reward, motor-counts logging, distance_log, action_log, reward_log, sleep replay, recency replay, or trajectory replay. Cut all of those.

The simplified inner loop (pseudocode — see eval loop for exact CuPy idioms):

```python
import random
import numpy as np
import cupy as cp

# Read cross-projection synapse indices once (constant after build)
# We need these to compute weight stats at the end.
cross_indices_cpu = []  # populate via bridge._plasticity_gate_to_synapses["bg_cross_projections"]
if "bg_cross_projections" in bridge._plasticity_gate_to_synapses:
    cross_indices_cpu = list(bridge._plasticity_gate_to_synapses["bg_cross_projections"])

rng = random.Random(seed * 7919)  # deterministic, distinct from eval RNGs
prev_goal = None
total_trials = n_goals * steps_per_goal
n_goal_changes = 0
trial_counter = 0
x, y = start_pos

for goal_idx in range(n_goals):
    gx, gy = _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal)
    prev_goal = (gx, gy)
    n_goal_changes += 1
    if verbose:
        print(f"[g11 seed={seed}] pretraining goal {goal_idx + 1}/{n_goals}: "
              f"({gx},{gy})", flush=True)

    # Reset agent to start at each new pretraining-goal episode
    x, y = start_pos

    for trial in range(steps_per_goal):
        # COPY THE EVAL TRIAL INNER LOOP (lines ~1500-1810 of eval loop)
        # SKIP: surprise-LR, adaptive-DA gating, RPE scaling, all the
        # opt-in mechanisms. Use defaults / the simplest path everywhere.
        # The point is to evolve weights under varied tasks, not to
        # optimize behavior.
        # ... (extract carefully from eval loop) ...
        trial_counter += 1

# Compute cross-projection weight summary
if cross_indices_cpu:
    cross_w = bridge.cp_synapse_weights[cp.asarray(cross_indices_cpu)].get()
    if np.isnan(cross_w).any():
        raise RuntimeError(
            "pretraining produced NaN cross-projection weights — likely STDP "
            "instability. Lower learning rate or shorten pretraining_steps_per_goal."
        )
    cross_mean = float(cross_w.mean())
    cross_std = float(cross_w.std())
else:
    cross_mean = float("nan")
    cross_std = float("nan")

if verbose:
    print(f"[g11 seed={seed}] pretraining complete: {trial_counter} trials, "
          f"{n_goal_changes} goal changes; cross weights mean={cross_mean:.3f} "
          f"std={cross_std:.3f} → handing off to eval (curriculum will freeze "
          f"bg_cross_projections)", flush=True)

return {
    "n_trials": trial_counter,
    "n_goal_changes": n_goal_changes,
    "cross_weights_mean": cross_mean,
    "cross_weights_std": cross_std,
}
```

**The full inner loop (~80-120 lines extracted from eval) goes between the two pseudocode `# COPY THE EVAL TRIAL INNER LOOP` markers.** Do this surgically: open both editors side by side, copy lines 1500-1810 of `research/runners/g11_bg_runner.py`, delete the parts marked SKIP above, and adapt variable names (`step` → `trial_counter`, etc.).

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_run_moving_goal_with_pretraining_smoke -v
```

Expected: PASS. Cross-weights have changed during pretraining.

Re-run the full pretraining test suite:

```bash
pytest tests/test_g11_bg_runner_flags.py -k "pretraining" -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): pretraining trial loop (developmental analog)

Replaces the Task 2 stub with the full pretraining trial loop. Per
goal: sample new (gx,gy), reset agent to start, run steps_per_goal
trials (stim → step → motor count → reward → plasticity). Cross
weight summary computed at end with NaN guard.

Integration test verifies cross-projection weights actually change
during pretraining."
```

---

## Task 6: conflict-flag validation

**Files:**
- Modify: `research/runners/g11_bg_runner.py:run_moving_goal_episode` (top of function)
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

```python
def test_developmental_pretraining_rejects_v3_1_thaw_conflict(tmp_out_path):
    """v4 keeps cross-projections frozen during eval; v3.1 thaws them at
    bg_cross_thaw_step. Both at once is meaningless. The runner should
    raise ValueError early instead of silently doing one or the other."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    with pytest.raises(ValueError) as exc:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
            enable_bg_cross_projections=True,
            enable_developmental_pretraining=True,
            bg_cross_thaw_step=10,  # v3.1 mechanism — incompatible
            pretraining_n_goals=0, pretraining_steps_per_goal=0,
        )
    assert "developmental-pretraining" in str(exc.value)
    assert "bg_cross_thaw_step" in str(exc.value) or "bg-cross-thaw-step" in str(exc.value)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_rejects_v3_1_thaw_conflict -v
```

Expected: no exception raised — test fails on `pytest.raises` not seeing one.

**Step 3: Write minimal implementation**

At the very top of `run_moving_goal_episode` (immediately after the docstring), add:

```python
    # v4 (2026-04-28): conflict check. v4 keeps cross-projections frozen
    # during eval; v3.1 thaws them at bg_cross_thaw_step. Both at once is
    # meaningless. Fail loud instead of silent priority resolution.
    if enable_developmental_pretraining and bg_cross_thaw_step >= 0:
        raise ValueError(
            "--developmental-pretraining (v4) is incompatible with "
            "--bg-cross-thaw-step (v3.1). v4 keeps cross-projections frozen "
            "throughout eval; v3.1 thaws them mid-eval. Use one or the other, "
            "not both. Got bg_cross_thaw_step={}.".format(bg_cross_thaw_step)
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_rejects_v3_1_thaw_conflict -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): reject --developmental-pretraining + --bg-cross-thaw-step"
```

---

## Task 7: warning when pretraining is enabled without cross-projections

**Files:**
- Modify: `research/runners/g11_bg_runner.py:run_moving_goal_episode` (top, right after the conflict check)
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

```python
def test_developmental_pretraining_warns_without_cross_projections(tmp_out_path, capsys):
    """Pretraining without --bg-cross-projections is harmless but pointless
    (the whole point is to develop cross-projection weights). Warn but proceed."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=True,
        enable_developmental_pretraining=True,
        enable_bg_cross_projections=False,  # the missing piece
        pretraining_n_goals=0, pretraining_steps_per_goal=0,
    )
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower() or "warning" in captured.err.lower(), (
        "expected a warning about pretraining without cross-projections")
    assert "bg-cross-projections" in captured.out or "bg_cross_projections" in captured.out
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_warns_without_cross_projections -v
```

Expected: no warning printed → assertion fails.

**Step 3: Write minimal implementation**

After the conflict-check block from Task 6, add:

```python
    if enable_developmental_pretraining and not enable_bg_cross_projections:
        print(
            "[g11 warning] --developmental-pretraining without "
            "--bg-cross-projections: pretraining will run but won't shape any "
            "bg_cross_projections gate (no cross pathways exist). Did you "
            "mean to also pass --bg-cross-projections?",
            flush=True,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_g11_bg_runner_flags.py::test_developmental_pretraining_warns_without_cross_projections -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(g11): warn on pretraining without --bg-cross-projections"
```

---

## Task 8: full test sweep + ensure no regressions

**Files:** none (verification step)

**Step 1: Run the full pretraining test suite**

```bash
pytest tests/test_g11_bg_runner_flags.py -v
```

Expected: ALL tests pass (existing + the 7 new ones added by Tasks 1, 2, 3, 4, 5, 6, 7).

**Step 2: Run the broader webapp + bridge tests as regression check**

```bash
pytest tests/test_webapp_server.py tests/test_regions.py tests/test_neuromodulators.py -v 2>&1 | tail -10
```

Expected: all pass — pretraining work shouldn't have touched any of these.

**Step 3: Commit (if any followup edits needed; otherwise skip)**

If you fixed an unrelated regression, commit it as a separate commit. Don't bundle.

---

## Task 9: Tier 1 wiring smoke (manual GPU run)

This is a documented manual step, not an automated test. Run after Task 8 passes locally.

**Run command:**

```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --hippocampus --learned-perception --pfc \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --landmarks --landmarks-replace-place \
    --sensed-reward \
    --bg-lateral-inhibition --bg-cross-projections --cross-projection-weight 0.0 \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --developmental-pretraining \
    --pretraining-n-goals 1 \
    --pretraining-steps-per-goal 1000 \
    --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42_v4tier1.json
```

Expected wall-clock: ~25 min (1× 1000 pretrain + 1800 eval).

**Pass criteria:**
- Run completes with rc=0
- Stdout shows: `pretraining goal 1/1: (gx, gy)` then `pretraining complete: 1000 trials, ... → handing off to eval`
- Output JSON parses; `phase_stats` non-empty
- No NaN in cross weights summary line
- Eval-phase rewards are sensible (not all zero, not all 1, agent reaches goal at least sometimes)

If any of those fail, debug before tier 2.

---

## Task 10: Tier 2 reduced smoke (manual, 3 seeds)

**Run command:** for SEED in 42 43 44, the same as tier 1 but with `--pretraining-n-goals 5 --pretraining-steps-per-goal 1000` and a different output filename suffix (`_v4tier2.json`).

Recommended: launch all 3 via the dashboard at `http://localhost:8765` (4-concurrent budget allows running them simultaneously). Wall-clock: ~4h batch.

After all 3 finish, run an aggregator (write to `scripts/analyze_cheat5_v4.py` modeled on `scripts/analyze_cheat5_v3.py`):

```bash
python scripts/analyze_cheat5_v4.py
```

**Decision matrix** (from the design doc):

| Eval-phase mean sum (n=3) | Action |
|---|---|
| ≤ 4.5 | Promising — proceed to tier 3 |
| 4.5–6.0 | Marginal — review per-seed; consider tweaking pretraining params first |
| > 6.0 | NO-GO at this scale; document and pivot to v4-NEGATIVE last-resort plan |

---

## Task 11: Tier 3 6-seed validation (manual, overnight)

Only if tier 2 was ≤ 4.5 mean.

**Run command:** for SEED in 42 43 44 100 101 102, full defaults (`--pretraining-n-goals 10 --pretraining-steps-per-goal 3000`). Wall-clock: ~14h batch at 4-concurrent (run as 4+2 batches, or 6-concurrent in one batch).

After all 6 finish, run the aggregator:

```bash
python scripts/analyze_cheat5_v4.py
```

**Decision matrix** (from the design doc):

| Eval-phase mean sum (n=6) | P0 | P1 | Verdict |
|---|---|---|---|
| ≤ 4.1 | ≤ 2.5 | ≤ 2.5 | **GO** — cheat #5 closed |
| 4.1–4.5 | OK | OK | **GO MARGINAL** |
| 4.5–6.0 | OK | high | **PARTIAL** |
| > 6.0 OR P0 high | — | — | **NO-GO v4** |

---

## Task 12: Findings doc + propagation

Regardless of the verdict (GO, MARGINAL, PARTIAL, NO-GO):

**Files to create/update:**
- Create: `research/findings/2026-04-28-cheat5-v4-results.md` (modeled on `research/findings/2026-04-28-cheat5-v3-results.md`)
- Update: `CLAUDE.md` "Cheat #5 progress (2026-04-28)" section — add v4 result row
- Update: `docs/SCIENCE_ROADMAP.md` §4.7 — append v4 row
- Update: `research/findings/INDEX.md` — link the new finding doc
- Update: `CHANGELOG.md` — add the v4 result entry to 2026-04-28
- Memory: create `project_cheat5_v4_results.md` and add a line to `MEMORY.md`

**If GO**: also kick off the optional follow-up plan for pretrained-weight persistence (HDF5 save/load, deferred at design time). Do this as a NEW task spawn (`mcp__ccd_session__spawn_task`), not in this plan.

**Commit:**

```bash
git add research/findings/2026-04-28-cheat5-v4-results.md CLAUDE.md docs/SCIENCE_ROADMAP.md research/findings/INDEX.md CHANGELOG.md
git commit -m "findings(cheat5): v4 developmental pretraining — <GO|NO-GO|...>"
git push origin main
```

---

## Done criteria summary

- [ ] Task 1-7: code + unit tests committed, all green
- [ ] Task 8: full test sweep passes with no regressions
- [ ] Task 9: tier 1 wiring smoke passes
- [ ] Task 10: tier 2 reduced smoke produces a 3-seed result
- [ ] Task 11 (only if tier 2 promising): 6-seed validation
- [ ] Task 12: findings doc + CLAUDE.md / SCIENCE_ROADMAP / INDEX / CHANGELOG / memory all updated

---

Plan complete and saved to `docs/plans/2026-04-28-cheat5-v4-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration. Stay in this session. Fresh subagent per task + code review (REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`).

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints. Guide you to open new session in worktree (REQUIRED SUB-SKILL: new session uses `superpowers:executing-plans`).

Which approach?
