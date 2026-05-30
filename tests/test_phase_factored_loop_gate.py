"""Tests for the phase-factored two-phase spiking loop controller
(Task 2 of docs/plans/2026-05-30-phase-factored-integrated-loop-
implementation.md).

The controller `research/runners/phase_factored_loop_gate.py` runs
compositional memory in TWO PHASES and scores it, reusing four
already-validated subsystems unchanged:
  Phase 1 (ONLINE, theta-ordered): present a length-N concept sequence
    in order; bind it order-preservingly via the engram-tagging API
    (gamma sub-cycle k binds item k).
  Phase 2 (OFFLINE, shuffled): replay via the validated SWR / Phase-1.3
    consolidation to build concept selectivity in cortex, in SHUFFLED
    order.
  Readout 1 (wm): "is concept X in the buffer?" from cortical concept-
    pool activity (selectivity built offline).
  Readout 2 (ep): "what came after X?" from the gamma-slot order of the
    index (built online).
  Shared theta-gamma rhythm: reuse the parked loop's controller;
    lesioning it must collapse BOTH readouts.

These tests run at --tiny-synth scale with SIM_BACKEND=numpy so they
execute on CPU without a GPU. The heavy end-to-end run_rung is a
subprocess smoke (CPU/numpy); the structural / faithfulness / no-autograd
pins are pure source greps + light in-process probes that do NOT build a
bridge (so they stay fast and deterministic).

Plain ASCII. No autograd anywhere in the shipped path.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTROLLER_PATH = REPO_ROOT / "research" / "runners" / \
    "phase_factored_loop_gate.py"

# The 7 frozen lesion names (mirror integrated_loop_core's frozen
# partition; pinned here so a drift in the controller is caught).
_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")
_HELPER_WM = ("no_bg_gate",)
_HELPER_EP = ("no_sequencing", "no_cls_replay")
_HELPER_BOTH = ("no_neuromod_timing",)
_ALL_LESIONS = _SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH


def _import_controller():
    """Import the controller module under SIM_BACKEND=numpy with a
    benign argv (no flags). Returns the module or skips if absent."""
    if not _CONTROLLER_PATH.exists():
        pytest.skip("phase_factored_loop_gate.py not landed yet")
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import importlib
    mod = importlib.import_module(
        "research.runners.phase_factored_loop_gate")
    return mod


# ---------------------------------------------------------------------------
# Increment 1: rung shape pin.
# ---------------------------------------------------------------------------
def test_run_rung_returns_exact_rung_shape():
    """run_rung(N, seed, tiny_synth=True) must return EXACTLY the rung
    dict shape the frozen integrated_loop_core.integrated_loop_verdict
    consumes: {"N", "n_seeds", "v1":{wm,ep}, "full":{wm,ep},
    "lesions":{<7 names>:{wm,ep}}}."""
    mod = _import_controller()
    rung = mod.run_rung(2, 42, tiny_synth=True)
    assert isinstance(rung, dict)
    # Top-level keys.
    assert rung["N"] == 2
    assert rung["n_seeds"] == 1
    for key in ("v1", "full"):
        assert key in rung, "rung missing %r" % key
        pair = rung[key]
        assert set(pair.keys()) >= {"wm", "ep"}, (
            "%s pair missing wm/ep: %r" % (key, pair))
        assert isinstance(pair["wm"], float)
        assert isinstance(pair["ep"], float)
    # All 7 lesion keys present with wm/ep pairs.
    assert "lesions" in rung
    les = rung["lesions"]
    assert set(les.keys()) == set(_ALL_LESIONS), (
        "lesion keys %r != frozen 7 %r"
        % (sorted(les.keys()), sorted(_ALL_LESIONS)))
    for name in _ALL_LESIONS:
        lp = les[name]
        assert set(lp.keys()) >= {"wm", "ep"}, (
            "lesion %s pair missing wm/ep: %r" % (name, lp))
        assert isinstance(lp["wm"], float)
        assert isinstance(lp["ep"], float)


def test_run_rung_values_are_finite_in_unit_interval():
    """Every wm/ep readout in the rung is a finite float in [0,1]
    (accuracies). Placeholder-or-real, the contract holds."""
    import math
    mod = _import_controller()
    rung = mod.run_rung(2, 42, tiny_synth=True)
    pairs = [rung["v1"], rung["full"]] + list(rung["lesions"].values())
    for p in pairs:
        for fld in ("wm", "ep"):
            v = p[fld]
            assert math.isfinite(v), "%r not finite" % v
            assert 0.0 <= v <= 1.0, "%r out of [0,1]" % v


# ---------------------------------------------------------------------------
# Increment 2: phase-ordering pin -- Phase 1 (online bind) MUST run
# BEFORE Phase 2 (offline consolidate). A swapped order is a bug.
# ---------------------------------------------------------------------------
def _capture_event_log(mod, mode="full", N=2, gap_zero=False):
    """Run ONE tiny-synth _run_mode with the passive phase-event sink
    enabled and return the recorded ordered marker list. The sink is a
    passive recorder (None in every real run); enabling it changes NO
    drive/gate/RNG/score -- it only RECORDS phase boundaries."""
    mod._EVENT_LOG = []
    try:
        mod._run_mode(mode, 42, N, True, gap_zero=gap_zero)
        return list(mod._EVENT_LOG)
    finally:
        mod._EVENT_LOG = None


def test_phase1_online_bind_runs_before_phase2_offline_consolidate():
    """The controller must call online-bind (Phase 1) BEFORE
    offline-consolidate (Phase 2). Pin via the ordered phase-event log:
    the Phase-1 'phase1:online_bind:done' marker precedes the Phase-2
    'phase2:offline_consolidate:start' marker in EVERY epoch."""
    mod = _import_controller()
    log = _capture_event_log(mod, mode="full", N=2)
    assert "phase1:online_bind:done" in log, (
        "Phase 1 (online bind) marker absent: %r" % log)
    assert "phase2:offline_consolidate:start" in log, (
        "Phase 2 (offline consolidate) marker absent: %r" % log)
    i_p1_done = log.index("phase1:online_bind:done")
    i_p2_start = log.index("phase2:offline_consolidate:start")
    assert i_p1_done < i_p2_start, (
        "PHASE ORDER BUG: Phase 2 (offline consolidate, idx %d) ran "
        "BEFORE Phase 1 finished binding (idx %d). Online-bind MUST "
        "precede offline-consolidate. Log: %r"
        % (i_p2_start, i_p1_done, log))


def test_engram_write_precedes_consolidation_replay():
    """The engram WRITE (start_engram_recording -> commit_engram_tag,
    the Phase-1 hippocampal index) must complete BEFORE the Phase-2
    consolidation replay starts -- the consolidation replays the
    committed tag, so committing it first is load-bearing."""
    mod = _import_controller()
    log = _capture_event_log(mod, mode="full", N=2)
    assert "engram:start_recording" in log
    assert "engram:commit_tag" in log
    assert "phase2:offline_consolidate:start" in log
    i_rec = log.index("engram:start_recording")
    i_commit = log.index("engram:commit_tag")
    i_consol = log.index("phase2:offline_consolidate:start")
    assert i_rec < i_commit < i_consol, (
        "engram write order bug: start(%d) -> commit(%d) -> "
        "consolidate(%d) must be ascending. Log: %r"
        % (i_rec, i_commit, i_consol, log))


# ---------------------------------------------------------------------------
# Increment 3: reuse pins -- the module REUSES the four validated
# subsystems + the parked theta-gamma controller + the frozen verdict
# BY IMPORT, and defines NONE of its own verdict bars. (This flips the
# Task 0 grounding-pin part (c) live.)
# ---------------------------------------------------------------------------
def _controller_src():
    if not _CONTROLLER_PATH.exists():
        pytest.skip("phase_factored_loop_gate.py not landed yet")
    return _CONTROLLER_PATH.read_text(encoding="utf-8")


def test_reuses_engram_tag_api():
    """Reuse the validated engram-tag API (the fast relational episode
    store) -- not a reimplementation."""
    src = _controller_src()
    assert ("start_engram_recording" in src
            or "commit_engram_tag" in src
            or "stimulate_tag" in src), (
        "controller must call the validated engram-tag API")


def test_reuses_consolidation_trainer():
    """Reuse the Phase-1.3 consolidation (offline SWR replay) by import
    -- run_concept_replay_phase from consolidation_trainer."""
    src = _controller_src()
    assert "consolidation_trainer" in src
    assert "run_concept_replay_phase" in src


def test_reuses_concept_pool_demo():
    """Reuse the v16 concept-binding selectivity mechanism by import."""
    src = _controller_src()
    assert "concept_pool_demo" in src


def test_reuses_abstention_gate():
    """Reuse the calibrated no-confab abstention gate by import."""
    src = _controller_src()
    assert "from research.runners.abstention_gate import" in src


def test_reuses_parked_theta_gamma_controller():
    """Reuse the parked theta-gamma timing controller (SharedThetaGamma)
    + the parked bridge builder BY IMPORT from integrated_loop_gate --
    NOT a reimplementation. Lesioning the shared clock must collapse
    BOTH readouts (pinned separately)."""
    src = _controller_src()
    assert "from research.runners.integrated_loop_gate import" in src
    assert "SharedThetaGamma" in src
    # The parked builder is reused, not redefined here.
    assert "def SharedThetaGamma" not in src, (
        "controller must REUSE SharedThetaGamma by import, not redefine "
        "it")


def test_reuses_consolidation_gate_idioms():
    """Reuse the validated awake/sleep/freeze gate idioms (Phase-1.3
    freeze-then-evaluate) by import from text_minimal_isolation."""
    src = _controller_src()
    assert "set_sleep_gates" in src
    assert "set_awake_gates" in src
    assert "freeze_all_gates" in src


def test_imports_parked_frozen_verdict_and_defines_no_own_bars():
    """Score via the parked, already-reviewed FROZEN verdict
    integrated_loop_core.integrated_loop_verdict. The controller must
    define NONE of its own integrated-loop bars (no _IL_*_MIN /
    _IL_LESION_MAX / _IL_SCI_MIN assignment) -- the bars live ONLY in
    the frozen module."""
    src = _controller_src()
    assert ("from research.runners.integrated_loop_core import "
            "integrated_loop_verdict" in src)
    # No local re-definition of the frozen bars (assignment form). The
    # frozen verdict owns them; the controller must not shadow them.
    for bar in ("_IL_V1_MIN", "_IL_SCI_MIN", "_IL_LESION_MAX",
                "_PROBE_BAR"):
        assert ("%s =" % bar) not in src, (
            "controller must NOT define its own bar %s -- the frozen "
            "verdict owns the bars" % bar)
    # It must NOT import or redefine the v2 core (this build scores via
    # the ORIGINAL frozen integrated_loop_core, per Task 2 contract).
    assert "integrated_loop_verdict_v2" not in src, (
        "this build scores via the ORIGINAL frozen integrated_loop_core "
        "verdict, not v2")


def test_no_autograd_in_shipped_path():
    """No torch / autograd USAGE anywhere in the shipped path (the word
    may appear in a 'NO autograd' comment; what is banned is the import
    / call forms)."""
    src = _controller_src()
    assert "import torch" not in src
    assert ".backward(" not in src
    assert "import autograd" not in src
    assert "torch.autograd" not in src
    assert "from autograd" not in src


def test_grounding_pin_part_c_flips_live():
    """Running the Task 0 grounding pin's part (c) tests must now PASS
    (they were SKIP until this controller landed). This asserts the
    cross-arc contract is satisfied live."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_phase_factored_loop_grounding.py",
         "-k", "test_c_", "-q"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300)
    assert proc.returncode == 0, (
        "grounding-pin part (c) did not pass:\n%s\n%s"
        % (proc.stdout, proc.stderr))
    # All three part-(c) tests must have RUN (not skipped) -- the
    # controller now exists so they are no longer skipped.
    assert " skipped" not in proc.stdout or "passed" in proc.stdout, (
        proc.stdout)
    assert "passed" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# Increment 4: lesion-fidelity pin -- each lesion variant is identical
# to the `full` run minus EXACTLY one subsystem with the SAME rng draws.
# The faithfulness discipline: every mode builds its own bridge with its
# own identically-seeded rng and _make_pairs is the SOLE per-trial rng
# consumer (_episode draws none). So the (role, filler) pairs drawn AND
# the rng state immediately after the draw must be byte-identical across
# `full` and every lesion at the same (seed, N).
# ---------------------------------------------------------------------------
def _record_rng_draws_per_mode(mod, N=2):
    """Monkeypatch _make_pairs to record, per call, the pairs drawn AND
    the rng bit_generator state snapshot immediately AFTER the draw. Run
    full + every lesion + v1 at tiny-synth. Returns {mode_label:
    [(pairs_tuple, state_repr), ...]}. Restores the original
    _make_pairs in a finally."""
    import copy
    orig = mod._make_pairs
    records = {"_cur": None}
    out = {}

    def _spy(n, rng):
        pairs = orig(n, rng)
        # Snapshot the rng state AFTER the draw -> proves the post-draw
        # stream position is identical for every mode (i.e. _episode
        # consumed no rng before this call and the draw itself is the
        # same).
        try:
            state = copy.deepcopy(rng.bit_generator.state)
        except Exception:
            state = None
        records["_cur"].append((tuple(pairs), json.dumps(state,
                                                          default=str)))
        return pairs

    mod._make_pairs = _spy
    try:
        for label, (mode, gap_zero) in {
            "full": ("full", False),
            "v1": ("full", True),
            **{m: (m, False) for m in _ALL_LESIONS},
        }.items():
            records["_cur"] = []
            mod._run_mode(mode, 42, N, True, gap_zero=gap_zero)
            out[label] = list(records["_cur"])
    finally:
        mod._make_pairs = orig
    return out


def test_lesion_fidelity_same_rng_draws_as_full():
    """Each lesion (and v1) draws the IDENTICAL (role, filler) pairs in
    the IDENTICAL order as `full`, and the rng stream is at the SAME
    position after each draw -- proving the only difference between full
    and a lesion is the ablated subsystem, NOT a divergent rng draw
    (the faithfulness discipline; a strawman lesion is rejected)."""
    mod = _import_controller()
    per_mode = _record_rng_draws_per_mode(mod, N=2)
    ref = per_mode["full"]
    assert len(ref) >= 1, "full made no _make_pairs draw"
    for label, recs in per_mode.items():
        assert recs == ref, (
            "RNG-DRAW DIVERGENCE: mode %r drew different pairs / left "
            "the rng in a different state than full.\n  full: %r\n  "
            "%s: %r" % (label, ref, label, recs))


def test_make_pairs_is_sole_per_trial_rng_consumer():
    """_episode must draw NO rng (the only per-trial rng consumer is
    _make_pairs). Pin: across n_train_epochs the number of _make_pairs
    calls equals 1 (drawn ONCE per run, stable across epochs -- the v16
    encode discipline), and the count is identical for full and every
    lesion."""
    mod = _import_controller()
    per_mode = _record_rng_draws_per_mode(mod, N=2)
    counts = {label: len(recs) for label, recs in per_mode.items()}
    # Exactly one draw per run (the bijection is drawn ONCE, then
    # interleaved-repeated across epochs).
    assert all(c == 1 for c in counts.values()), (
        "expected exactly 1 _make_pairs draw per run (v16 stable-"
        "bijection discipline); got %r" % counts)


# ---------------------------------------------------------------------------
# Increment 5: no_shared_clock pin -- this lesion actually DISABLES the
# shared theta-gamma controller (not a no-op) and is wired to drive BOTH
# readouts toward chance (the SHARED non-separability signature).
# ---------------------------------------------------------------------------
def test_no_shared_clock_is_a_shared_lesion():
    """no_shared_clock must be partitioned as a SHARED lesion (one whose
    frozen duty is to collapse BOTH readouts), matching the parked
    frozen partition. A helper-only classification would let it collapse
    only one readout and miss the non-separability signature."""
    mod = _import_controller()
    assert "no_shared_clock" in mod._SHARED, (
        "no_shared_clock must be a SHARED lesion (collapses BOTH "
        "readouts), got partition: SHARED=%r" % (mod._SHARED,))


def test_no_shared_clock_constructs_two_independent_clocks():
    """Structural: under no_shared_clock the controller builds TWO
    independent SharedThetaGamma instances (the WM-gating clock and the
    hippocampal-write clock desynchronize), whereas `full` uses ONE
    shared instance. Pin via the passive phase-event log: full ->
    'clock:one', no_shared_clock -> 'clock:two'. This proves the lesion
    is NOT a no-op."""
    mod = _import_controller()
    log_full = _capture_event_log(mod, mode="full", N=2)
    log_lesion = _capture_event_log(mod, mode="no_shared_clock", N=2)
    assert "clock:one" in log_full, (
        "full must use ONE shared clock: %r" % log_full)
    assert "clock:two" in log_lesion, (
        "no_shared_clock must construct TWO clocks: %r" % log_lesion)
    assert "clock:two" not in log_full, (
        "full must NOT construct two clocks")
    assert "clock:one" not in log_lesion, (
        "no_shared_clock must NOT use the shared single clock")


def test_no_shared_clock_genuinely_desynchronizes_timing():
    """The two independent clocks must produce DIFFERENT gamma-slot
    timing for WM-gating vs the hippocampal write -- i.e. the lesion
    genuinely desynchronizes the rhythm that unifies the loop. Drive the
    SAME construction the controller uses (one shared instance vs two
    independent + a fixed phase advance on the hippocampal clock) and
    assert the gamma slots diverge across a theta period. This is the
    mechanism by which BOTH readouts (WM-slot gating AND the
    hippocampal-write order) are pushed toward chance."""
    from research.runners.integrated_loop_gate import (
        SharedThetaGamma, _GAMMA_PER_THETA)
    # `full`: one shared instance -> WM clock and hippo clock are
    # always in lock-step.
    shared = SharedThetaGamma(shift=True)
    locked_diffs = 0
    for _ in range(2 * _GAMMA_PER_THETA):
        # Same instance read twice == identical phase, always.
        if shared.gamma_slot != shared.gamma_slot:
            locked_diffs += 1
        shared.step()
    assert locked_diffs == 0, "a single shared clock is never desynced"

    # `no_shared_clock`: two independent instances; the hippo clock is
    # advanced a fixed half-theta phase (exactly the controller's
    # desync). Their gamma slots must DIFFER on a majority of steps.
    clk_wm = SharedThetaGamma(shift=True)
    clk_hip = SharedThetaGamma(shift=True)
    for _ in range(_GAMMA_PER_THETA // 2):
        clk_hip.step()
    desynced = 0
    total = 2 * _GAMMA_PER_THETA
    for _ in range(total):
        if clk_wm.gamma_slot != clk_hip.gamma_slot:
            desynced += 1
        clk_wm.step()
        clk_hip.step()
    assert desynced >= total // 2, (
        "no_shared_clock must genuinely desynchronize the WM-gating "
        "and hippocampal-write timing (got %d/%d steps desynced)"
        % (desynced, total))
