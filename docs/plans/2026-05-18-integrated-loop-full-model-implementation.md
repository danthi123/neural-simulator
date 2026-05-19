# Integrated-loop full spiking-model implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Standing autonomy applies: one fresh subagent per task; strict failing-test -> minimal-implementation -> run -> commit; controller verifies every commit leaves the protected set byte-unchanged; honest propagation of every outcome; iterate following the reference catalog on any non-success; no hand-back.

**Goal:** Build the decisive full spiking-network test of the integrated-loop hypothesis: a biology-grounded closed loop that composes the project's already-validated subsystems, unified by one shared timing rhythm, and shows compositional memory that holds as load scales and that every single-system lesion abolishes.

**Architecture:** Reuse the validated subsystems byte-unchanged as the loop's parts (concept-pool representation, hippocampal relational binding, prefrontal working-memory maintenance, basal-ganglia selective gating, replay consolidation, neuromodulatory timing, the trustworthy answer-only-when-grounded output gate). The only net-new code is (1) a pure success-criteria/necessity-verdict module with its own fixed thresholds, and (2) the closed-loop integration runner — a small shared theta-gamma timing controller plus the wiring that composes the reused parts. No new learning mechanism is added anywhere.

**Tech stack:** Python; the project spiking simulator (`sim/bridge.py`, NumPy CPU backend for a deterministic minimal slice); the validated region/pathway framework (`build_biological_brain_regions`); the validated basal-ganglia cascade builder (`build_bg_brain_regions`); the engram-tagging API on the bridge; the neuromodulator subsystem; the native eligibility-trace reward path with the validated temporal-credit rule; the checkpoint module; the trustworthy output gate. Verdict module: standard library + typing only.

---

## Plain-language background (context for the implementing engineer)

The project's earlier work tested one learning mechanism at a time and each was insufficient on its own. The project owner's scientific correction: in real brains, compositional memory does not live in any single mechanism — it emerges from several systems running together as one closed loop. "Each part insufficient alone" is the expected signature of an emergent capability, not evidence the approach is unfit.

A cheap preliminary NumPy simulation of the loop's logic was already run (three transparent fidelity iterations against the project's own reference catalog; fixed success thresholds set in advance and never moved). It robustly confirmed the load-bearing core: the full integrated loop performs compositional memory, and removing any one of three shared systems — the combinatorial binding step, the single shared theta-gamma timing rhythm, or the fast hippocampal store — destroys BOTH the working-memory query AND the episodic-sequence recall together, at every tested load. That is exactly the reference catalog's non-separability prediction (Lisman & Idiart theta-gamma multiplex). The cheap tier honestly could not settle per-helper necessity (a two-item composition is near-trivial; a nearest-match readout over a few items is robust). That is a limitation of the simplification, not a fatal flaw, and is not further cheaply fixable without contriving the toy. The design pre-registered the full spiking-network model as the decisive test. This plan builds that test.

**Reference-catalog grounding (do not re-derive — already established in the design doc):** compositional language is the Memory -> Unification -> Control loop instantiated as the prefrontal-associative reentrant cortex/basal-ganglia/thalamus loop, unified by one shared theta-gamma timing rhythm that simultaneously drives prefrontal working-memory maintenance AND hippocampal episodic sequence-encoding, with theta-paced relational binding, replay-driven consolidation into the cortical store, neuromodulator-timed plasticity windows, and a trustworthy gate at the output. The project built and validated every part individually but never composed them into this one closed loop. That specific integration is the gap this build closes.

**The decisive instrument is a lesion study** — the same necessity-by-ablation method the reference catalog itself uses for the nested-replay evidence. The full loop must show the capability AND every single-system lesion must abolish the capability that system is responsible for. The three shared systems collapsing BOTH readouts together is the decisive emergent-from-integration signature.

## Honest ceiling (state this; never overstate it)

A clean success means: a biology-grounded multi-system loop shows emergent compositional memory that holds or improves as load scales and that every single-system lesion abolishes. It is explicitly NOT fluent open-ended language, NOT a large language model, NOT conversation solved — unless a later, separately pre-registered stage genuinely shows it. The prior validated results and the project's documented boundaries are unaffected. The earlier isolated-mechanism negatives stand and are reinterpreted as predicted by this hypothesis, not refuted.

## Source design

This plan implements `docs/plans/2026-05-18-Q5-integrated-biology-grounded-closed-loop-design.md` (sections 1b, 2, 3, 4 "minimal closed loop", 5, 7, 8, 9). Read it if any spec point here is ambiguous; the design is authoritative.

## Reuse-by-import only (the protected set — byte-unchanged)

The integration runner imports and composes these; it does NOT modify, copy-edit, or re-implement any of them. The controller verifies (per task, and across the whole branch) that every path below is byte-empty in the commit-scoped `git diff`:

- `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (the trustworthy answer-only-when-grounded gate; MUST stay 7/7 green and byte-identical the entire build)
- every existing frozen verdict module: `research/runners/compose_bridge_core.py`, `research/runners/compose_bind_core.py`, `research/runners/td_critic_core.py`, `research/runners/dendritic_fair_core.py`, `research/runners/constrained_decode_core.py`, `research/runners/q2r_core.py`, and every other `*_core.py`
- every existing gate that pairs with the above: `research/runners/constrained_decode_gate.py`, `research/runners/q2r_gate.py`, `research/runners/compose_bridge_gate.py`, `research/runners/engram_bootstrap_gate.py`
- `research/runners/text_minimal_isolation.py` (`build_biological_brain_regions` REUSED UNMODIFIED)
- `research/runners/g11_bg_runner.py` (`build_bg_brain_regions` REUSED UNMODIFIED)
- the validated simulator modules: `sim/bridge.py`, `sim/kernels.py`, `sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/backend.py`, `sim/regions.py`, `sim/text_embeddings.py`, `sim/td_value_critic.py`, `sim/compose_temporal_bind.py`, `sim/dendritic_plasticity.py`
- `research/runners/grounded_decode.py`, `sim/grounded_decode.py`, `research/runners/generator_g_core.py`

The only files this build creates: `research/runners/integrated_loop_core.py`, `research/runners/integrated_loop_gate.py`, `tests/test_integrated_loop_core.py`, `tests/test_integrated_loop_gate.py`. Plus the propagation artifacts in Task 5 (a findings doc, a capability-status pillar edit, a git commit).

## No new automatic differentiation / training anywhere

Every learning update in this build is a REUSED validated local rule (the native eligibility-trace reward path with the validated temporal-credit values, the validated spike-timing plasticity). No `torch`, no `.backward()`, no autograd, no gradient-descent objective is introduced in any shipped path. Task 3 (adversarial review) and Task 4 (no-harm) both explicitly assert this.

## Task naming note

Filenames use plain descriptive names (`integrated_loop_core`, `integrated_loop_gate`) rather than letter-number codenames, per the owner's standing communication requirement. The capability-status pillar for the decisive run is the next sequential pillar after the most recent one (the integration-probe pillar already recorded as PREDICTED); Task 5 flips that pillar's status, it does not append a new number.

---

### Task 0: Grounding pin test

**Files:**
- Create: `tests/test_integrated_loop_gate.py` (this task adds ONLY the grounding-pin test; Task 2 adds the rest)

**Context:** This is a deliberately-failing pin that turns green only after Task 2 lands. It IS the Task-2 gate: it asserts the integration runner exists, exposes the kill-safe entry point, and that its `--tiny-synth` smoke runs end-to-end and produces a verdict object. Committing it now (red) and seeing it go green after Task 2 is the proof Task 2 actually wired the loop.

**Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path


def test_integration_runner_tiny_smoke_produces_verdict(tmp_path):
    """Grounding pin: the integration runner exists, runs a fast
    --tiny-synth smoke end-to-end on the CPU backend, and writes a
    verdict JSON whose classification is the explicitly-not-propagated
    TINY marker (never a real PASS/FAIL/VOID at toy scale)."""
    out = tmp_path / "tiny.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "runner failed: %s\n%s" % (proc.stdout, proc.stderr))
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "tiny-synth verdict must be marked TINY / NOT propagated")
```

**Step 2: Run it to verify it fails**

Run: `pytest tests/test_integrated_loop_gate.py::test_integration_runner_tiny_smoke_produces_verdict -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.runners.integrated_loop_gate'` (or non-zero return code). This is intentional and correct at Task 0.

**Step 3: Commit the red pin**

```bash
git add tests/test_integrated_loop_gate.py
git commit -m "test: grounding pin for the integrated-loop runner (red until the runner lands)"
```

**Controller verification:** the commit-scoped `git diff` touches only `tests/test_integrated_loop_gate.py`. The protected set is byte-empty in the diff. Do NOT mark Task 0 "green" — it is intentionally red and is the Task-2 acceptance gate.

---

### Task 1: The success-criteria / necessity-verdict module (fully specified — transcribe exactly)

**Files:**
- Create: `research/runners/integrated_loop_core.py`
- Test: `tests/test_integrated_loop_core.py`

**Context:** This is a pure, deterministic, fail-closed verdict module. It mirrors the discipline of the existing frozen verdict modules (`q2r_core.py`, `compose_bridge_core.py`) EXACTLY: fixed numeric thresholds set in advance and never moved; instrument-validity checked before any science conclusion; malformed input returns a safe "cannot conclude" rather than crashing; "cannot conclude" (VOID) is kept strictly distinct from "fails" (FAIL); imports only the standard library and typing; does not import or mutate any existing verdict module; introduces no new project-global threshold. It scores the lesion study described in the design's section 5 against the cheap-probe-confirmed structure (the three shared systems must collapse BOTH readouts; each helper must collapse the readout it is responsible for).

The module is LOAD-BEARING and is fully specified below for exact transcription. Transcribe it verbatim. The 16-case adversarial test matrix must pass.

**Pre-registration correction log (transparent; no numeric bar changed; no decisive run had occurred):** the first draft of this verdict applied a blanket "any load below the science bar -> FAIL" check before the scale analysis, which made the pre-registered `WORKS-SMALL-NO-SCALE-CONFIDENCE` classification (the owner's explicitly-defined honest non-success: "works at small load but degrades with scale") unreachable. That is an instrument self-consistency defect, not an outcome-driven change. The classification *precedence* was corrected (check scale-confident-PASS first; then "works at the minimal load but does not scale" -> WORKS-SMALL; then "fails even at the minimal load" -> plain FAIL). Every numeric threshold (`_IL_V1_MIN=0.90`, `_IL_SCI_MIN=0.80`, `_IL_LESION_MAX=0.40`, `_IL_SCALE_TOL=0.10`, ladder `(2,4,8)`, `_IL_MIN_SEEDS=3`) and the shared/helper lesion partition are byte-identical to the first draft. The 16-case test matrix below is unchanged — it always encoded the intended behaviour; the corrected module satisfies all 16.

**Step 1: Write the failing test**

Create `tests/test_integrated_loop_core.py` with the full matrix below. It imports `integrated_loop_verdict` and the frozen constants and exercises 16 cases: full-loop-succeeds-scale-confident; works-small (trend break); works-small (top below floor); each shared lesion failing to collapse one readout -> VOID; each helper lesion failing its responsibility -> VOID; instrument-soundness unmet -> VOID; sound+discriminating but science below bar -> FAIL; ladder mismatch -> VOID; non-numeric/NaN -> VOID-not-raise; too-few-seeds -> VOID; malformed top-level -> VOID-not-raise; and explicit pins that each frozen bar equals its pre-registered value.

```python
import math
import pytest

from research.runners.integrated_loop_core import (
    integrated_loop_verdict,
    _IL_V1_MIN, _IL_SCI_MIN, _IL_LESION_MAX, _IL_SCALE_TOL,
    _IL_LADDER, _IL_MIN_SEEDS,
    _SHARED, _HELPER_WM, _HELPER_EP, _HELPER_BOTH)


def _good_rung(N, full=0.9, n_seeds=5):
    """A rung where the full loop succeeds on BOTH readouts and EVERY
    lesion collapses exactly the readout(s) it is responsible for."""
    lesions = {}
    for name in _SHARED:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    for name in _HELPER_WM:
        lesions[name] = {"wm": 0.2, "ep": 0.9}          # wm collapses
    for name in _HELPER_EP:
        lesions[name] = {"wm": 0.9, "ep": 0.2}          # ep collapses
    for name in _HELPER_BOTH:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    return {"N": N, "n_seeds": n_seeds,
            "v1": {"wm": 0.95, "ep": 0.95},
            "full": {"wm": full, "ep": full},
            "lesions": lesions}


def test_01_scale_confident_pass():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.88), _good_rung(8, 0.90)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "PASS"
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_02_works_small_trend_break():
    # full loop passes every rung but composition drops > tol going up
    rungs = [_good_rung(2, 0.95), _good_rung(4, 0.95), _good_rung(8, 0.81)]
    rungs[2]["full"] = {"wm": 0.81, "ep": 0.66}  # drop 0.95 -> 0.66 > tol
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_03_works_small_top_below_floor():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.84), _good_rung(8, 0.82)]
    # monotone within tol, but largest rung below the science floor
    rungs[2]["full"] = {"wm": 0.78, "ep": 0.78}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_04_shared_lesion_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_SHARED[0]] = {"wm": 0.85, "ep": 0.2}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_05_shared_lesion_collapses_wm_but_not_ep_is_void():
    # the decisive signature: a shared lesion MUST collapse BOTH
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"][_SHARED[1]] = {"wm": 0.2, "ep": 0.88}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_06_helper_wm_lesion_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[2]["lesions"][_HELPER_WM[0]] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_07_helper_ep_lesion_does_not_collapse_ep_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"][_HELPER_EP[0]] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_08_helper_both_lesion_collapses_only_one_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_HELPER_BOTH[0]] = {"wm": 0.2, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_09_v1_unmet_is_void_not_fail():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["v1"] = {"wm": 0.55, "ep": 0.95}  # cannot learn trivial bind
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_10_sound_discriminating_but_science_below_bar_is_fail():
    rungs = [_good_rung(2, 0.70), _good_rung(4, 0.70), _good_rung(8, 0.70)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["instrument_valid"] is True
    assert r["classification"] == "FAIL"


def test_11_ladder_mismatch_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(4)]  # dup, no 8
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_12_non_numeric_and_nan_is_void_not_raise():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["full"] = {"wm": "0.9", "ep": 0.9}          # str junk
    assert integrated_loop_verdict(rungs)["GATE"] == "VOID"
    rungs2 = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs2[2]["full"] = {"wm": float("nan"), "ep": 0.9}  # NaN
    assert integrated_loop_verdict(rungs2)["GATE"] == "VOID"


def test_13_too_few_seeds_is_void():
    rungs = [_good_rung(2, n_seeds=2), _good_rung(4), _good_rung(8)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_14_malformed_top_level_is_void_not_raise():
    assert integrated_loop_verdict(None)["GATE"] == "VOID"
    assert integrated_loop_verdict([])["GATE"] == "VOID"
    assert integrated_loop_verdict("garbage")["GATE"] == "VOID"
    assert integrated_loop_verdict([{"no": "N"}])["GATE"] == "VOID"


def test_15_threshold_tamper_pins():
    # If anyone edits a frozen bar, THIS test fails loudly.
    assert _IL_LADDER == (2, 4, 8)
    assert _IL_V1_MIN == 0.90
    assert _IL_SCI_MIN == 0.80
    assert _IL_LESION_MAX == 0.40
    assert _IL_SCALE_TOL == 0.10
    assert _IL_MIN_SEEDS == 3


def test_16_lesion_set_pins():
    # The shared/helper partition is itself pre-registered structure.
    assert _SHARED == ("no_binding", "no_shared_clock", "no_hippo_store")
    assert _HELPER_WM == ("no_bg_gate",)
    assert _HELPER_EP == ("no_sequencing", "no_cls_replay")
    assert _HELPER_BOTH == ("no_neuromod_timing",)
```

**Step 2: Run it to verify it fails**

Run: `pytest tests/test_integrated_loop_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.runners.integrated_loop_core'`.

**Step 3: Write the minimal implementation — transcribe this verbatim into `research/runners/integrated_loop_core.py`**

```python
"""Pure FIXED-bar success-criteria + necessity-verdict for the full
spiking-network integrated-loop test. Scores the lesion study: the
full closed loop must succeed on BOTH the working-memory query and the
episodic-sequence recall at every load; each of the three SHARED
systems (the combinatorial binding step, the one shared theta-gamma
timing rhythm, the fast hippocampal store) must collapse BOTH readouts
when lesioned (the decisive emergent-from-integration signature,
already robustly confirmed by the cheap preliminary simulation); each
HELPER system must collapse the readout it is responsible for.

Mirrors the adversarial-hardened frozen-verdict discipline EXACTLY:
instrument-validity FIRST; fail-closed; fixed bars pre-registered HERE
and NEVER tuned to a result; "cannot conclude" (VOID) strictly
distinct from "fails" (FAIL); malformed / non-numeric / unorderable
input -> VOID, never an exception. Owns its OWN frozen bars; imports
no other verdict module; standard library + typing only; no torch,
no autograd. ASCII only.

A-priori justification of every frozen value (defensible WITHOUT
reference to any observed run):
- _IL_LADDER = (2, 4, 8): compositional load = number of role-filler
  bindings held and composed simultaneously. Two is the smallest
  non-trivial composition; the ladder doubles to a load where a
  scale-confidence claim actually lives. Same geometric-doubling
  shape as the other scale-confidence ladders in this project.
- _IL_V1_MIN = 0.90: the full loop, on a NO-GAP trivial single bind,
  must nearly perfectly learn the bijection, or the instrument cannot
  even measure composition (this is soundness, not science).
- _IL_SCI_MIN = 0.80: the full loop must clear a clear-majority bar
  on the genuine compositional task on BOTH readouts. Same value
  family as the project's other validated science bars.
- _IL_LESION_MAX = 0.40: a lesioned readout has "collapsed" iff it
  is at/near chance. For a 1-of-N readout chance is <= 0.5 (N=2) and
  lower for larger N; 0.40 is a defensible at/near-chance ceiling and
  is the SAME value the cheap preliminary simulation used for its
  ablation ceiling. A lesion that does NOT collapse its responsibility
  means the capability is not genuinely emergent-from-integration ->
  the instrument cannot discriminate emergence from a wiring artifact
  -> VOID (NOT a science PASS/FAIL), exactly the compose-bridge-core
  "a control learned -> VOID" discipline.
- _IL_SCALE_TOL = 0.10: a stochastic multi-seed accuracy has a noise
  floor; 0.10 is a defensible max permitted DROP between ascending
  rungs, same magnitude family as the other validated tolerances.
- _IL_MIN_SEEDS = 3: below three seeds a multi-seed claim is not
  supportable.
These are pre-registered in this file, BEFORE any full-model run, and
NEVER tuned to an outcome."""
from __future__ import annotations
import math
from typing import Dict

_IL_LADDER = (2, 4, 8)
_IL_V1_MIN = 0.90
_IL_SCI_MIN = 0.80
_IL_LESION_MAX = 0.40
_IL_SCALE_TOL = 0.10
_IL_MIN_SEEDS = 3

# Pre-registered lesion partition. SHARED systems must collapse BOTH
# readouts (non-separability). HELPER systems must collapse the
# readout each is responsible for. This partition is itself frozen.
_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")
_HELPER_WM = ("no_bg_gate",)
_HELPER_EP = ("no_sequencing", "no_cls_replay")
_HELPER_BOTH = ("no_neuromod_timing",)

_ALL_LESIONS = _SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH


def _num(x):
    """Strict finite real or None. bool is NOT a number here; a
    control/metric serialized as a string or bool must force VOID,
    never silently pass."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair(d):
    """Return (wm, ep) as finite floats, or None if malformed."""
    if not isinstance(d, dict):
        return None
    wm = _num(d.get("wm"))
    ep = _num(d.get("ep"))
    if wm is None or ep is None:
        return None
    return (wm, ep)


def integrated_loop_verdict(rungs) -> Dict:
    """Pure, deterministic, fail-closed. Recomputed from the single
    recorded JSON; NEVER raises.

    rungs: list of per-load dicts, each:
      {"N": int, "n_seeds": int,
       "v1":   {"wm": float, "ep": float},   # no-gap trivial bind
       "full": {"wm": float, "ep": float},   # genuine composition
       "lesions": {<name>: {"wm": float, "ep": float}, ...}}

    Precedence (fail-closed, self-consistent): any soundness or
    discrimination defect -> VOID; else if every load meets the
    science bar AND composition is non-decreasing up to tolerance AND
    the largest load holds -> SCALE-CONFIDENT-PASS; else if the
    smallest (minimal-composition) load meets the science bar but
    scale confidence fails (a larger load drops below the bar, or the
    trend breaks, or the top is below the bar) -> GATE FAIL with
    classification WORKS-SMALL-NO-SCALE-CONFIDENCE (an honest
    non-success: works small, does not scale); else (the loop fails
    the science bar even at the smallest load) -> GATE FAIL with
    classification FAIL (it does not perform the capability at all)."""
    bars = {"LADDER": list(_IL_LADDER), "V1_MIN": _IL_V1_MIN,
            "SCI_MIN": _IL_SCI_MIN, "LESION_MAX": _IL_LESION_MAX,
            "SCALE_TOL": _IL_SCALE_TOL, "MIN_SEEDS": _IL_MIN_SEEDS}

    def void(reason):
        return {"GATE": "VOID", "instrument_valid": False,
                "classification": "VOID", "reason": reason,
                "frozen_bars": bars}

    if not isinstance(rungs, list) or not rungs:
        return void("rungs not a non-empty list")
    try:
        ordered = sorted(rungs, key=lambda r: r["N"])
    except (TypeError, KeyError):
        return void("rungs not orderable by N")
    try:
        ladder = tuple(int(r["N"]) for r in ordered)
    except (TypeError, ValueError, KeyError):
        return void("rung N not integer-coercible")
    if ladder != _IL_LADDER:
        return void("ladder %s != pre-registered %s (padding/"
                    "duplicate/missing-rung guard)"
                    % (ladder, _IL_LADDER))

    # ---- instrument validity FIRST (soundness + discrimination) ----
    full_min = []
    for r in ordered:
        ns = _num(r.get("n_seeds"))
        if ns is None or ns < _IL_MIN_SEEDS:
            return void("a rung has fewer than %d seeds"
                        % _IL_MIN_SEEDS)
        v1 = _pair(r.get("v1"))
        fu = _pair(r.get("full"))
        if v1 is None or fu is None:
            return void("v1/full readout pair missing or non-numeric")
        if v1[0] < _IL_V1_MIN or v1[1] < _IL_V1_MIN:
            return void("instrument unsound: the full loop did NOT "
                        "learn the no-gap trivial bind on both "
                        "readouts (>= %.2f) at N=%d"
                        % (_IL_V1_MIN, int(r["N"])))
        les = r.get("lesions")
        if not isinstance(les, dict):
            return void("lesions block missing/not a dict at N=%d"
                        % int(r["N"]))
        for name in _ALL_LESIONS:
            lp = _pair(les.get(name))
            if lp is None:
                return void("lesion '%s' missing/non-numeric at N=%d"
                            % (name, int(r["N"])))
            lw, le = lp
            if name in _SHARED or name in _HELPER_BOTH:
                if not (lw <= _IL_LESION_MAX and le <= _IL_LESION_MAX):
                    return void("non-discriminating: lesion '%s' did "
                                "NOT collapse BOTH readouts at N=%d "
                                "(wm=%.3f ep=%.3f, bar<=%.2f) -- the "
                                "capability is not emergent-from-"
                                "integration here / wiring artifact"
                                % (name, int(r["N"]), lw, le,
                                   _IL_LESION_MAX))
            elif name in _HELPER_WM:
                if not (lw <= _IL_LESION_MAX):
                    return void("non-discriminating: helper lesion "
                                "'%s' did NOT collapse the working-"
                                "memory readout at N=%d (wm=%.3f, "
                                "bar<=%.2f)"
                                % (name, int(r["N"]), lw,
                                   _IL_LESION_MAX))
            else:  # _HELPER_EP
                if not (le <= _IL_LESION_MAX):
                    return void("non-discriminating: helper lesion "
                                "'%s' did NOT collapse the episodic-"
                                "recall readout at N=%d (ep=%.3f, "
                                "bar<=%.2f)"
                                % (name, int(r["N"]), le,
                                   _IL_LESION_MAX))
        full_min.append(min(fu[0], fu[1]))

    # Instrument is sound + discriminating. Now the science verdict.
    # Precedence is ordered so every pre-registered classification is
    # reachable and means exactly what the design defines:
    #   PASS              -> every load meets the bar AND scales
    #   WORKS-SMALL(FAIL) -> minimal load works, but does not scale
    #   FAIL              -> does not even work at the minimal load
    base = {"instrument_valid": True, "frozen_bars": bars,
            "full_min_by_rung": full_min}
    all_science_ok = all(fm >= _IL_SCI_MIN for fm in full_min)
    monotone = all(full_min[i + 1] >= full_min[i] - _IL_SCALE_TOL
                   for i in range(len(full_min) - 1))
    top_ok = full_min[-1] >= _IL_SCI_MIN
    if all_science_ok and monotone and top_ok:
        return {"GATE": "PASS",
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "the full integrated loop succeeds on both "
                          "readouts at every load; every single-system "
                          "lesion collapses the capability it is "
                          "responsible for (the three shared systems "
                          "collapse both readouts together); "
                          "composition is non-decreasing up to "
                          "tolerance across the ascending load ladder "
                          "and holds at the largest load", **base}
    if full_min[0] >= _IL_SCI_MIN:
        why = []
        if not all_science_ok:
            why.append("a larger load falls below the science bar")
        if not monotone:
            why.append("composition drops > tolerance between "
                       "ascending loads")
        if not top_ok:
            why.append("the largest load is below the science bar")
        return {"GATE": "FAIL",
                "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
                "reason": "the loop performs the minimal composition "
                          "(smallest load >= the science bar) but is "
                          "NOT scale-confident: %s -- an honest "
                          "non-success (works small, does not scale)"
                          % "; ".join(why), **base}
    return {"GATE": "FAIL", "classification": "FAIL",
            "reason": "instrument sound+discriminating but the full "
                      "loop is below the science bar even at the "
                      "smallest (minimal-composition) load -- it does "
                      "not perform the capability at all", **base}
```

**Step 4: Run the test to verify it passes**

Run: `pytest tests/test_integrated_loop_core.py -v`
Expected: 16 passed.

**Step 5: Commit**

```bash
git add research/runners/integrated_loop_core.py tests/test_integrated_loop_core.py
git commit -m "feat: pure fixed-bar success-criteria/necessity-verdict for the integrated-loop test"
```

**Controller verification:** the commit-scoped `git diff` touches only the two new files. The protected set is byte-empty. `integrated_loop_core.py` imports only `math` and `typing`. It does not import any `*_core` or any `sim`/runner module. The frozen bars are not derived from any number. `pytest tests/test_abstention_gate.py -q` is still 7/7.

---

### Task 2: The closed-loop integration runner (genuine net-new integration — NOT transcription)

**Files:**
- Create: `research/runners/integrated_loop_gate.py`
- Test: extend `tests/test_integrated_loop_gate.py` (the Task-0 pin must now go green; add a small structural test)

**Context:** This is the decisive instrument. It builds the full biology-grounded closed loop in the real spiking simulator by composing the validated subsystems byte-unchanged, unified by ONE shared theta-gamma timing rhythm, and measures the lesion study the verdict module scores.

This is **genuine net-new integration**, not transcribe-a-reference. The implementer mirrors the proven kill-safe CLI/checkpoint/verdict scaffold of `research/runners/compose_bridge_gate.py` (read it — it is the structural template: backend pin, `_build_bridge` via the reused builder, per-mode episode loop, per-seed checkpoint, `KeyboardInterrupt` -> resumable, `--tiny-synth` smoke whose verdict is never propagated, ASCII prints, verdict from the frozen core). The implementer WRITES the new wiring against the reused validated interfaces. The only genuinely-new pieces are (a) the shared theta-gamma timing controller and (b) the closed-loop wiring; everything else is imported byte-unchanged.

**If the implementer has questions before starting, ask the controller.** Likely questions: exact region composition for the prefrontal working-memory slots; how the basal-ganglia cascade is repurposed from the motor channel to the prefrontal/associative channel; the precise readout definitions. Answers are in this spec and the design doc; ask if still ambiguous.

**Reused interfaces (import byte-unchanged; do NOT modify):**

- `from research.runners.text_minimal_isolation import build_biological_brain_regions` — build the substrate with `enable_hippocampus_consolidation=True` (gives the hippocampal regions `ec/dg/dg_pv_basket/ca3/ca1` + the `ca1 -> motor` / `ca1 -> language_output` consolidation pathways = the fast relational store + the replay-consolidation path) and `enable_dlpfc_verb=True` (gives the prefrontal pool with NMDA bistable maintenance = the working-memory slots). Concept pools (`noun_pool_*`) are the concept/schema layer. Pass NMDA on (`enable_nmda=True` on the config, as `compose_bridge_gate` does).
- `from research.runners.g11_bg_runner import build_bg_brain_regions` — the validated basal-ganglia cascade (per-channel `cortex -> str_D1/str_D2 -> gpi -> thal -> motor` disinhibition gating). Repurpose its disinhibition gate from the MOTOR channel to the PREFRONTAL/associative channel: the cascade's selected output gates WHICH prefrontal working-memory slot updates vs holds (the reference catalog notes the project only ever built the motor channel; composition needs the prefrontal one). The implementer wires the cascade's gate outputs to the dlpfc working-memory slot-update path; it does NOT modify `build_bg_brain_regions`.
- The engram-tagging API on the bridge (already on `sim/bridge.py`, byte-unchanged): `bridge.start_engram_recording(name)`, `bridge.commit_engram_tag(name, top_k=..., region_filter=[...])`, `bridge.stimulate_tag(name, drive_pA=...)`, `bridge.clear_tag_drive(name)` — the one-shot hippocampal relational binding (start before the encoding window, commit over the hippocampal regions, stimulate to recall the bound episode).
- `from sim.neuromodulators import NeuromodulatorConfig, ProductionRule, ModulatorTarget` — construct (do not mutate) the dopamine-from-reward modulator exactly as `compose_bridge_gate._da_modulator_from_delta` does, and an acetylcholine-style plasticity-window modulator gated by the shared clock's theta phase (the neuromodulatory timing of the loop). Construction only; the subsystem code is byte-unchanged.
- `from sim.kernels import fused_eligibility_trace_decay` and the bridge native `cp_eligibility_trace` reward path with the validated temporal-credit values (`_GAMMA = 0.95`, `_LAMBDA = 0.9`, exactly as in `compose_bridge_gate`) — the DA-gated learning inside the loop. No new learning rule.
- `from sim.train_checkpoint import save_checkpoint, load_checkpoint, resume_epoch` — kill-safe per-seed checkpoint/resume (exactly the `compose_bridge_gate` pattern).
- `from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD` — the trustworthy answer-only-when-grounded gate at the output: the working-memory query readout is emitted ONLY when the grounded-evidence gate passes; otherwise the loop abstains (this preserves the no-confabulation moat inside the loop). Byte-unchanged.
- `from sim.text_embeddings import orthogonal_drive_pattern` — the proven orthogonal code idiom for role/filler drives (as `compose_bridge_gate._verb_drive` uses it).

**The net-new pieces (the only new code):**

1. **A small shared theta-gamma timing controller.** A pure helper class/function in this file (no new sim module). It maintains, per simulation step, a theta phase and a gamma sub-cycle index. ONE instance drives BOTH: (i) which prefrontal working-memory slot is gated open for update vs hold on this step, and (ii) which gamma slot the hippocampal episodic encoder writes, and it SHIFTS the role-filler assembly sequence across successive theta cycles (shift, not repeat — the reference-catalog episodic-write rule). This is pure timing/index bookkeeping plus driving existing bridge inputs. It introduces NO learning, NO autograd. Theta:gamma structure: one theta period contains the working-memory buffer span of gamma sub-cycles (set the gamma count >= the largest load on the ladder so 8 bindings fit one buffer; this is a frozen structural choice in this runner, justified in a comment by the catalog ~7+-2 buffer span, not tuned to a result).

2. **The closed-loop wiring.** Compose the reused parts per the design's "minimal closed loop": concept pools <-> hippocampal engram binding <-> prefrontal working-memory slots with basal-ganglia-gated updating (prefrontal/associative channel) <-> replay consolidation into the concept/schema layer, all neuromodulator-timed by the one shared clock, output through the trustworthy gate.

**Behavioral spec — one composition trial at load N:**

- **Encode.** For each of N (role, filler) pairs: on that pair's gamma sub-cycle within the current theta period, drive the role orthogonal code + the filler orthogonal code into the concept pools; the shared clock gates the corresponding prefrontal working-memory slot open so the binding is written into the NMDA-bistable slot; `start_engram_recording` is active so the hippocampal ensemble for the relational episode accumulates. The shared clock SHIFTS the assembly across successive theta cycles so the ordered episode is written. `commit_engram_tag(..., region_filter=[hippocampal regions])` finalizes the relational episode tag.
- **Maintain.** A delay of held steps with no encode drive: the NMDA-bistable prefrontal slots hold the bindings; the shared clock keeps refreshing within theta; no decision drive, reward held at zero (credit strictly delayed past the gap, exactly the `compose_bridge_gate` discipline).
- **Working-memory query readout (`wm`).** The queried role depends on the mode, matching the pre-registered V1-vs-Science distinction in the design (Section 5): in **`v1`** (instrument soundness) the query is the **trivial drilled binding** — query a role that WAS drilled and expect ITS bound filler ("can the loop machinery learn the bijection at all"); in **`full` and every lesion mode** (the genuine science/compositional test) the query includes a **novel composed role-filler combination that was not drilled**, requiring genuine relational generalization, not memorization. In all modes: population-vote the concept pools for the bound filler; emit the answer ONLY if the trustworthy grounded-output gate passes, otherwise abstain (a wrong emission and an abstention on a groundable query both score 0; a correct gated emission scores 1). Accuracy over the N queries = `wm`.

  **Pre-registration conformance log (transparent; no frozen bar changed; caught BEFORE any decisive run):** the first draft of this sentence applied the novel-recombination probe to *all* modes including `v1`, which contradicted the design's authoritative Section 5 (V1 = the no-gap trivial drilled bijection; the novel composed combination is the Science task in `full`/lesions). That made the `v1` soundness baseline embed a by-design-unlearnable probe, structurally capping `v1` wm at chance even when the loop binds correctly. Corrected here to match the pre-registered design. `_IL_V1_MIN=0.90` and every other frozen bar are unchanged; the Science task (novel recombination in `full` + every lesion, which the lesions must collapse) is exactly as hard as before — this strengthens instrument soundness without easing the science.
- **Episodic-sequence recall readout (`ep`).** `stimulate_tag` the committed hippocampal episode tag; read back the ORDER of the bound pairs from the shifted assembly; score the recalled sequence against the true encode order. Accuracy = `ep`.
- **Learn.** Reward (delayed past the maintenance gap) drives the native eligibility-trace path with the temporal-credit delta as the reward signal for one step (exactly `compose_bridge_gate._episode`'s native-path discipline); the acetylcholine-style plasticity-window modulator (clock-gated) times when the update is allowed. Then a short replay/consolidation phase drives the committed tag during a "sleep" gate so the `ca1 -> concept` consolidation pathway transfers the bound structure into the schema layer (the reused Phase-1.3 consolidation path; gates flipped exactly as the consolidation trainer does — awake: encode on, consolidate off; sleep: encode off, replay on).

**The modes (each = the full loop minus EXACTLY one system; identical RNG draws; everything else byte-identical — the `compose_bridge_gate` faithfulness discipline):**

- `full` — the complete closed loop.
- `v1` — the full loop on a NO-GAP trivial single bind (instrument soundness; mirrors `compose_bridge_gate`'s `gap_zero`).
- `no_binding` — suppress the combinatorial binding step (drive role and filler but do NOT form the combined relational assembly). Shared.
- `no_shared_clock` — replace the ONE shared clock with TWO independent clocks (prefrontal and hippocampal timing desynchronized); everything else identical. This is the decisive shared-system lesion. Shared.
- `no_hippo_store` — skip `start/commit/stimulate` engram tagging (no fast relational store). Shared.
- `no_bg_gate` — remove basal-ganglia selective gating (all prefrontal slots always open / never selectively gated). Helper (collapses `wm`).
- `no_sequencing` — the shared clock REPEATS the assembly instead of SHIFTING it across theta cycles (no episodic order written). Helper (collapses `ep`).
- `no_neuromod_timing` — remove the clock-gated plasticity-window modulator (plasticity always on, untimed). Helper (collapses both).
- `no_cls_replay` — skip the replay/consolidation phase (no transfer into the schema layer). Helper (collapses `ep`).

Each lesion must be the full loop minus exactly that one system, consuming the SAME random draws in the SAME order as `full` (only the lesioned system's effect removed) — a strawman crippled elsewhere is a Task-3 reject.

**Scale ladder + scaffold:**

- Full run: load `N` over the frozen ladder `(2, 4, 8)`; `--seeds` default `[42, 43, 44, 45, 46]`; require `>= 3` seeds or print `NOT-RUNNABLE` and return 2 (exactly `compose_bridge_gate`/`q2r_gate`).
- `--tiny-synth`: shrink the ladder to its first rung, shrink pools/steps/epochs so the smoke completes fast on the NumPy CPU backend; the tiny verdict is marked TINY and NEVER propagated (this is what makes the Task-0 pin go green).
- Force `os.environ.setdefault("SIM_BACKEND", "numpy")` BEFORE any sim import (exactly `compose_bridge_gate`) for a deterministic minimal slice.
- Kill-safe: per-seed checkpoint via `save_checkpoint`; on `KeyboardInterrupt` flush the partial and return 130 with a "resumable" message; on resume, skip completed seeds (the `q2r_gate` resume idiom or the `compose_bridge_gate` per-seed-checkpoint idiom — either is acceptable; it MUST be genuinely interruptible/resumable because this in-bridge spiking run is heavier than the NumPy probes).
- The decisive output: assemble per-rung `{"N", "n_seeds", "v1":{wm,ep}, "full":{wm,ep}, "lesions":{...:{wm,ep}}}` (aggregated mean over seeds), call `integrated_loop_verdict(rungs)` from the frozen core, write the JSON, print `GATE=... <honest-ceiling banner>`. The honest-ceiling banner text: emergent compositional memory in a biology-grounded multi-system loop ONLY — NOT fluent open-ended language, NOT a large language model, NOT conversation solved.
- ASCII only. No `torch`, no autograd anywhere.

**Step 1: Make the Task-0 pin executable, add a structural test (failing)**

Add to `tests/test_integrated_loop_gate.py`:

```python
def test_runner_imports_reused_parts_byte_unchanged():
    """The runner composes the validated parts by import; it must not
    declare its own copies of them, and must add no autograd."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "from research.runners.integrated_loop_core import" in src
```

Run: `pytest tests/test_integrated_loop_gate.py -v` -> both tests FAIL (module missing).

**Step 2: Implement `research/runners/integrated_loop_gate.py`**

Mirror the `compose_bridge_gate.py` scaffold structure exactly (backend pin, builder via reused interfaces, per-mode episode loop, per-seed kill-safe checkpoint, `--tiny-synth`, ASCII, verdict from the frozen core). Write the net-new shared theta-gamma timing controller and the closed-loop wiring per the behavioral spec. Reuse-by-import only; modify none of the protected set.

**Step 3: Run the smoke + structural test**

Run: `pytest tests/test_integrated_loop_gate.py -v`
Expected: both tests PASS. The Task-0 pin is now green (the runner exists and the `--tiny-synth` smoke produces a TINY-marked verdict end-to-end). Also run the tiny smoke directly once and read the JSON to confirm a verdict object with a `GATE` field and the TINY marker.

**Step 4: Commit**

```bash
git add research/runners/integrated_loop_gate.py tests/test_integrated_loop_gate.py
git commit -m "feat: closed-loop integration runner composing the validated subsystems under one shared theta-gamma rhythm"
```

**Controller verification (trust-but-verify):** the commit-scoped `git diff` touches only `research/runners/integrated_loop_gate.py` and `tests/test_integrated_loop_gate.py`. Every protected path is byte-empty in the diff AND byte-empty across `git diff <branch-base>..HEAD`. The runner contains no `import torch` / `.backward(` / autograd. `build_biological_brain_regions` and `build_bg_brain_regions` are imported, not redefined. `pytest tests/test_abstention_gate.py -q` is still 7/7. The `--tiny-synth` verdict is marked TINY.

---

### Task 3: Dedicated adversarial review of the load-bearing pieces (BEFORE the no-harm phase)

**Files:** none modified by the reviewer subagent. Strengthen-only fixes (if any) are applied by a follow-up implementer subagent and re-reviewed; frozen bars stay byte-unchanged.

**Context:** Dispatch a fresh subagent as a dedicated adversarial reviewer of `research/runners/integrated_loop_gate.py` and `research/runners/integrated_loop_core.py`. This mirrors the adversarial reviews that found real holes in the temporal-credit and compose-bind work; its job is to find holes, not to bless. It produces a written report; the controller decides on strengthen-only fixes.

**The reviewer must specifically probe and answer, with file:line evidence:**

1. **Is the compositional capability genuinely emergent from the integration, or a wiring artifact?** Does the full loop genuinely succeed AND does each single-system lesion genuinely collapse the capability that system is responsible for? Is the three-shared-systems-collapse-both-readouts signature real in the wiring (not hard-codeable by a config crank)?
2. **Is each lesion faithful?** Is each lesion identical to the full loop minus EXACTLY that one system, consuming the SAME random draws in the SAME order? Specifically: is `no_shared_clock` truly "one shared clock -> two independent clocks" with nothing else changed? Is `no_binding` not secretly also crippling the readout? Are the helper lesions strawmen crippled elsewhere?
3. **Are the validated subsystems genuinely reused unchanged, not copy-edited?** `build_biological_brain_regions`, `build_bg_brain_regions`, the engram API, the neuromodulator subsystem, the native eligibility/temporal-credit path, the checkpoint module, the trustworthy output gate — all imported byte-unchanged?
4. **Can a broken or unsound run be scored a success?** Trace the verdict precedence: a V1-unsound run, a non-discriminating run (a lesion that didn't collapse), a malformed/NaN record — each must be VOID, never PASS/FAIL. Confirm "cannot conclude" is strictly distinct from "fails".
5. **Are the fixed thresholds movable by results?** Confirm every `_IL_*` bar and the `_SHARED/_HELPER_*` partition are pre-registered constants, justified a-priori, never derived from an observed number, and pinned by tests 15-16.
6. **Is any new automatic differentiation/training added?** Must be none — every learning update is the reused validated local rule. Grep both files and the import graph for `torch`, `backward`, autograd, any gradient objective.
7. **Is the shared theta-gamma controller genuinely ONE shared rhythm driving BOTH prefrontal maintenance and hippocampal episodic encoding** (so that `no_shared_clock` is a real, decisive lesion), or are they secretly separate so the lesion is trivial?
8. **Is the basal-ganglia gate genuinely repurposed to the prefrontal/associative channel** (gating which working-memory slot updates), not the motor channel?

**Step:** Dispatch the reviewer subagent (`./code-quality-reviewer-prompt.md` style, but with the eight probes above as the explicit charter). It returns a report. The controller applies strengthen-only fixes via a follow-up implementer subagent (frozen bars byte-unchanged), then re-dispatches the reviewer until the report has no open holes. Commit any strengthen-only fixes with a clear message; controller verifies the protected set stays byte-empty in the diff.

Do NOT proceed to Task 4 until the adversarial review has no open issues.

---

### Task 4: No-harm phase (the full protected set is byte-unchanged; the trustworthy gate still passes 7/7)

**Files:** none created/modified. This task is verification only; it produces a short evidence note appended to the eventual findings doc (Task 5), not a code change.

**Step 1: Prove the protected set is byte-unchanged across the whole branch**

Run `git diff --stat <branch-base>..HEAD` and confirm the ONLY changed paths are: `research/runners/integrated_loop_core.py`, `research/runners/integrated_loop_gate.py`, `tests/test_integrated_loop_core.py`, `tests/test_integrated_loop_gate.py` (plus, only after Task 5, the findings doc + capability-status pillar). Explicitly confirm byte-empty for every protected path listed in "Reuse-by-import only" above:

```bash
git diff --stat <branch-base>..HEAD -- research/runners/abstention_gate.py tests/test_abstention_gate.py research/runners/compose_bridge_core.py research/runners/q2r_core.py research/runners/q2r_gate.py research/runners/constrained_decode_core.py research/runners/constrained_decode_gate.py research/runners/compose_bridge_gate.py research/runners/engram_bootstrap_gate.py research/runners/text_minimal_isolation.py research/runners/g11_bg_runner.py sim/bridge.py sim/kernels.py sim/neuromodulators.py sim/train_checkpoint.py sim/backend.py sim/regions.py sim/text_embeddings.py sim/td_value_critic.py sim/compose_temporal_bind.py sim/dendritic_plasticity.py sim/grounded_decode.py research/runners/grounded_decode.py research/runners/generator_g_core.py
```

Expected: empty output (no protected file changed).

**Step 2: The trustworthy answer-only-when-grounded gate still passes 7/7**

Run: `pytest tests/test_abstention_gate.py -q`
Expected: 7 passed.

**Step 3: Both new verdict/runner test suites still green**

Run: `pytest tests/test_integrated_loop_core.py tests/test_integrated_loop_gate.py -q`
Expected: all passed (16 core + the gate pin + the structural test).

**Step 4: Assert no shipped path imports autograd**

Grep `research/runners/integrated_loop_core.py` and `research/runners/integrated_loop_gate.py` for `torch`, `.backward(`, `autograd`, `grad(` — expected: no matches in any shipped code path (a docstring mention of "no autograd" is acceptable; an actual import/call is a hard stop -> back to Task 2).

Do NOT proceed to Task 5 until all four steps pass. If Step 1 or 2 fails, the build harmed the protected set — stop, revert the offending change, and redo the task that caused it.

---

### Task 5: CONTROLLER-ONLY decisive run (NOT a subagent task — bring this back to the controller)

This task is performed by the controller directly, never delegated to a subagent. It is the decisive arbiter.

**Step 1: Grounding run first (toy numbers NOT reported)**

Run the `--tiny-synth` smoke once more to confirm end-to-end health on the exact machine that will do the decisive run:

```bash
python -m research.runners.integrated_loop_gate --tiny-synth --seeds 42 43 44 --out research/findings/raw/integrated_loop_tiny.json
```

Confirm return code 0 and a TINY-marked verdict. These numbers are a health check ONLY and are explicitly not reported as a result.

**Step 2: The decisive multi-seed run at increasing compositional load**

Fixed pre-registered configuration: the frozen ladder `(2, 4, 8)`, seeds `42 43 44 45 46`, full (non-tiny) scale. Kill-safe and monitored to ACTUAL completion. Use the Bash `run_in_background` mechanism that genuinely notifies on completion, OR run in the foreground — NEVER a detached process with a false "I will be notified" claim. Completion is actively confirmed (poll the output JSON existence + the process state) before any result is stated.

```bash
python -m research.runners.integrated_loop_gate --seeds 42 43 44 45 46 --ckpt research/findings/raw/integrated_loop_ckpt --out research/findings/raw/integrated_loop_decisive.json
```

If interrupted, resume the same command (kill-safe). Do not state any result until the JSON is written and the process has genuinely exited 0.

**Step 3: Mandatory anti-cheat check — scrutinize a nominal success HARDER than a failure**

Recompute the verdict from the single recorded JSON (`research/findings/raw/integrated_loop_decisive.json`) WITHOUT re-running and WITHOUT changing any threshold. Confirm, by hand from the recorded numbers:

- The full loop genuinely clears the science bar on BOTH readouts at every rung.
- Each single-system lesion genuinely collapses the readout it is responsible for.
- The three shared-system lesions (`no_binding`, `no_shared_clock`, `no_hippo_store`) collapse BOTH readouts together at every rung (the decisive emergent-from-integration signature).
- Instrument soundness (`v1`) is met at every rung.
- The composition accuracy is non-decreasing up to tolerance across the ascending load ladder and holds at the largest load (for a SCALE-CONFIDENT-PASS) — or, honestly, where it does not.
- The classification returned by `integrated_loop_verdict` recomputed from the JSON matches what the recorded numbers imply. No re-run, no bar tuning, no overclaim.

A nominal PASS gets MORE scrutiny than a FAIL, not less. If anything is off, the honest classification stands (VOID/FAIL/works-small) and is propagated as such.

**Step 4: Honest propagation of EVERY outcome (plain professional language)**

- Write `research/findings/2026-05-18-integrated-loop-full-model-<outcome>.md` in plain professional language (computational-neuroscientist briefing an informed colleague; no codenames as load-bearing terms; no undefined acronyms; every technical term defined once). State exactly what the run showed, the recomputed verdict, the honest ceiling, and what is and is not claimed. A FAIL or VOID is an honest, valuable finding — write it as such, not as a setback to spin.
- Update the capability-status panel: flip the existing integration pillar (the one recorded as PREDICTED for the integration probe) to the decisive outcome (VALIDATED if SCALE-CONFIDENT-PASS; otherwise the honest classification), in plain language. Do not append a new pillar number; update the existing one in place.
- Run the capability-status schema test green: `pytest tests/test_webapp_server.py -k capability_status -q`.
- Append the Task-4 no-harm evidence (protected set byte-empty; trustworthy gate 7/7) to the findings doc.
- Commit and push to BOTH remotes (origin and gitea).

**Step 5: Continue autonomously per the reference catalog (no hand-back)**

- **Clean SCALE-CONFIDENT-PASS:** proceed to the design's next staged integration step — first add multi-step sequential composition (compose a short grounded sequence, not just one bind), then the fluent-prior variant — each its own pre-registered gate, each built by returning to writing-plans then subagent-driven-development. Do not stop; do not declare victory beyond the honest ceiling.
- **Honest non-success (FAIL / WORKS-SMALL-NO-SCALE-CONFIDENCE / VOID):** do NOT declare the approach unfit and do NOT hand back. Name, with the specific reference-catalog entry, which integration point the spiking wiring is not reproducing faithfully (e.g. wrong theta:gamma buffer span, wrong shift-vs-repeat episodic write rule, wrong prefrontal/associative gating discipline, missing theta-paced compression). Fix THAT biological fidelity in the net-new wiring (the reused validated parts stay byte-unchanged), re-run the SAME pre-registered gate (bars frozen; a bar changes ONLY if required for instrument soundness, transparently logged, never toward an outcome), propagate honestly, and continue. This is bounded only by honest exhaustion of cited biological refinements — and even then the next step is the next catalog-identified integration gap, autonomously.

Bring Task 5 back to the controller. Tasks 0-3 are subagent-driven; Task 4 is controller-verified; Task 5 is controller-only.

---

## Remember

- One fresh subagent per task; strict failing-test -> minimal-implementation -> run -> commit.
- Controller trust-but-verify EVERY commit: the protected set is byte-empty in the commit-scoped diff AND across the whole branch.
- The trustworthy answer-only-when-grounded gate stays byte-identical and 7/7 green throughout.
- `integrated_loop_core.py` is transcribed verbatim; its 16-case adversarial matrix must pass; its frozen bars are never tuned.
- `integrated_loop_gate.py` is genuine net-new integration: the shared theta-gamma timing controller + the wiring are new; everything else is reused by import, byte-unchanged; no new autograd anywhere.
- Plain professional language in every artifact and commit message.
- Honest propagation of every outcome to both remotes; iterate following the reference catalog on any non-success; no hand-back.
- The honest ceiling is stated and never overstated.

## Pre-registered biology-fidelity iteration 1 — per-binding drive symmetry (added 2026-05-19; transparent; no frozen bar changed)

The first full-model build pass produced an honest interim non-success at
the instrument-soundness stage (findings:
`research/findings/2026-05-18-integrated-loop-full-model-instrument-soundness-blocked-per-slot-drive-asymmetry.md`).
After the adversarial review's strengthen-only fixes the loop is
GPU-backed, role-selective (the validated 16-pool concept recipe
transferred), and the trustworthy answer-only-when-grounded gate is
operable. It does NOT yet pass the pre-registered soundness check: at the
minimal two-binding load the closed-loop wiring under-potentiates the
second binding, so the first binding clears the grounding gate and the
second is correctly abstained (soundness one-half, below the bar). The
gate is working as designed; the defect is upstream in the net-new
wiring.

Per the design's iterate-following-biology discipline (Q5 design Section
7), the named, cited next step is its own pre-registered iteration:

- **Named mechanism (cited):** the basal-ganglia gate updates each
  prefrontal working-memory slot *independently and with comparable
  strength*, and multiple slots are maintained simultaneously at
  comparable strength (the prefrontal-basal-ganglia working-memory
  model: independently-gated stripes, each updated equivalently — the
  same reference-catalog grounding the design already cites). The
  current net-new wiring violates this: later bindings receive weaker
  effective drive than the first.
- **Iteration task:** diagnose which net-new wiring element causes the
  asymmetry (candidates to check, not assume: the validated topographic
  -prior pass may boost only the first drilled binding's input edge; the
  basal-ganglia channel / thalamocortical potentiation may favor the
  first-driven channel; per-slot encode order without symmetric
  interleaving may consolidate the first slot more), then fix the
  net-new wiring so every maintained binding is driven symmetrically and
  strongly enough that a genuinely-bound pool clears the byte-unchanged
  no-confabulation gate — for ALL N bindings, not only the first.
- **Acceptance is the SAME pre-registered, frozen gate (NOT changed):**
  on the GPU, the trivial drilled bijection (`v1`) reaches `wm` >= 0.90
  and `ep` >= 0.90 with every drilled binding clearing the
  byte-unchanged grounding gate; the novel-recombination science probe
  in `full` + every lesion is unchanged and exactly as hard. No frozen
  threshold in `integrated_loop_core.py` is touched; the
  no-confabulation gate and every protected/validated module stay
  byte-unchanged; no automatic differentiation; ASCII; GPU for the
  real/decisive path.
- **Discipline:** strengthen-only; controller trust-but-verify each
  diff (protected set byte-empty); then re-enter the dedicated
  adversarial review (confirm the symmetry fix did not introduce a
  hard-feed or break lesion faithfulness, and that the BG-gated
  causal path stays intact), then the no-harm phase, then the
  controller-only decisive multi-seed GPU run. A faithful repeat
  negative is propagated honestly and triggers the next cited
  biology-fidelity iteration — autonomously, no hand-back, no
  declare-unfit, no config-cranking a frozen bar.

## Pre-registered biology-fidelity iteration 2 — active prefrontal-stripe equalization + cold-start-breaking gated learning (added 2026-05-19; supersedes iteration 1; deeper mechanism; needs its own design)

Iteration 1 was run diagnose-before-fix and produced a precise, GPU-
evidenced negative (findings:
`research/findings/2026-05-19-integrated-loop-v1-asymmetry-is-an-emergent-symmetry-break-not-a-wiring-bug.md`).
The `v1` asymmetry is NOT a structural wiring bug — every measured
quantity is symmetric — it is an emergent winner-take-most
symmetry-break in the structurally-symmetric shared-inhibition +
bistable pool network, with the per-stripe basal-ganglia-gated
slot-to-concept efferent dead from spike-timing cold-start. Three
faithful strengthen-only fixes were each falsified and reverted. Minimal
strengthen-only wiring tweaks are therefore demonstrably exhausted for
this defect (the systematic-debugging inflection: stop reactive
patching, take the next step as a deeper, properly-designed iteration).

- **Named mechanism (cited; grounded in the project's own validated,
  protected subsystems):** the prefrontal-basal-ganglia working-memory
  account requires multiple slots maintained simultaneously at
  comparable strength. A symmetric shared-inhibition bistable network
  does not do this alone; biology equalizes competing maintained items
  via active gain control (homeostatic / divisive normalization) and via
  a working dopamine-gated three-factor learning signal that actually
  potentiates the gated slot-to-content pathway. The project already has
  BOTH validated and protected: the homeostatic firing-rate regulation
  mechanism (basis for a per-stripe equalizing drive) and the validated
  temporal-credit eligibility / dopamine-gated substrate (basis for
  breaking the cold-start so the gated efferent genuinely learns).
- **Iteration task (deeper than a tweak — its own design + plan):**
  compose those two already-validated subsystems into the closed loop so
  the prefrontal slots are actively equalized and the gated
  slot-to-concept efferent potentiates for every maintained binding —
  reusing the validated subsystems byte-unchanged; net-new = only the
  composition wiring.
- **Acceptance is the SAME pre-registered frozen gate (unchanged):** on
  the GPU, the trivial drilled bijection (`v1`) reaches `wm` >= 0.90 and
  `ep` >= 0.90 with EVERY drilled binding clearing the byte-unchanged
  no-confabulation gate; the novel-recombination science probe in `full`
  + every lesion is unchanged and exactly as hard. No frozen threshold
  in `integrated_loop_core.py` touched; the no-confabulation gate and
  every protected/validated module byte-unchanged; no automatic
  differentiation; ASCII; GPU for the real/decisive path.
- **Process:** because this is a substantive mechanism, not a
  strengthen-only tweak, it re-enters the full pre-registered pipeline —
  a focused design (which validated homeostasis + temporal-credit
  interfaces, how composed, honest ceiling), then writing-plans, then
  subagent-driven TDD with the dedicated adversarial review before the
  no-harm phase, then the controller-only decisive run. A faithful
  repeat negative is propagated honestly and names the next cited
  biology-fidelity iteration — autonomously, no hand-back, no
  declare-unfit, no config-cranking a frozen bar.

## Pre-registered biology-fidelity iteration 3 — documented non-zero efferent initialization (added 2026-05-19; supersedes iteration 2; single documented config fix; convergent, not thrashing)

Iteration 2 (findings:
`research/findings/2026-05-19-integrated-loop-iter2-homeostasis-works-temporal-credit-blocked-by-zero-init.md`)
produced a precise convergent negative: LEVER 2 (validated homeostatic
per-stripe equalization) transferred and works; LEVER 1 (validated
temporal-credit, faithfully wired, encode-only) is blocked solely
because the net-new `dlpfc_verb -> noun_pool_F*` efferent is zero-
initialized — a truly-zero synapse carries no current, so spike-timing
eligibility never charges and the reward-gated update is identically
zero. This is the project's OWN documented zero-initialization gotcha
(the same root cause as the validated text-input/output "non-zero
readout pathway init" fix; biologically grounded as spontaneous
baseline cortical weights, Barlow 1972).

- **Named mechanism (documented project fix; cited):** initialize the
  net-new slot-to-concept efferent with a small NON-ZERO prior
  (`weight_mean ~= 0.5`, `weight_jitter ~= 0.3`) — exactly the
  precondition the validated reference runners (`compose_bridge_gate`,
  `concept_pool_demo`) already satisfy for their scored pathways — so
  the now-correctly-timed LEVER-1 temporal-credit reward can actually
  charge eligibility, while LEVER-2 homeostasis keeps the stripes
  equalized. NOT a new mechanism; a single configuration change
  (`weight_mean`/`weight_jitter`) on the `RegionPathway` the runner
  already adds.
- **Keep both now-faithful levers** (committed honest-wip `5c27e99`):
  LEVER 1 temporal-credit (encode-only, validated idiom byte-unchanged,
  no query hard-feed, no autograd) and LEVER 2 homeostasis (validated
  kernel byte-unchanged, verified working).
- **Acceptance is the SAME pre-registered frozen gate (unchanged):** on
  the GPU, `v1` wm AND ep >= 0.90 with EVERY drilled binding clearing
  the byte-unchanged no-confabulation gate; `full`+lesions novel probe
  byte-identical. No frozen threshold touched; no-confab moat and every
  protected/validated module byte-unchanged; no autograd; ASCII;
  GPU/CuPy for the real path (numpy only for `--tiny-synth`).
- **Honest hard bound (pre-committed):** if instrument-soundness still
  fails AFTER correctly applying the documented non-zero initialization
  on top of the two now-faithful mechanisms, the evidence then points
  to a deeper architectural limit. That is surfaced as a genuine
  architecture question / a fundamentally different approach — NOT
  another configuration iteration, NOT config-cranking. Until then,
  this single documented fix is the disciplined convergent next step,
  autonomous, no hand-back.
- **Process:** strengthen-only; controller trust-but-verify the diff;
  dedicated adversarial re-review before the no-harm phase (Probe-8
  BG-causal intact; the non-zero init does not itself leak the answer
  / become a static hard-feed; lesions faithful; homeostasis scoped);
  then no-harm; then the controller-only decisive run.

## Pre-registered iteration 4 — the fundamentally-different, project-validated binding approach (added 2026-05-19; supersedes iteration 3; the hard bound's EXPLICITLY-permitted "fundamentally different approach", NOT a config iteration)

Iterations 1-3 (findings:
`research/findings/2026-05-19-integrated-loop-iter3-deeper-architecture-finding-global-scalar-credit-cannot-carry-WM-selectivity.md`)
precisely localized the instrument-soundness gap: with the
basal-ganglia causal wiring intact, per-stripe homeostatic
equalization working, and the documented non-zero initialization
applied, episodic binding is now PERFECT (ep=1.0) and the trustworthy
gate is cleared — but role-selective WORKING-MEMORY binding does not
form under a global scalar three-factor (temporal-credit) signal. This
triangulates with the project's own independent 2026-05-05 verdict:
global scalar feedback fails to produce selective binding at
biological scale; the credit-assignment rule is the bottleneck, the
architecture is sufficient. The project's own documented resolution is
the embodied co-firing plus topographic-prior binding paradigm
(validated: Tier-1 ~6x improvement; the 16-pool concept substrate at
high multi-seed bidirectional binding) — which the design's Section 3
already specifies as the concept-layer substrate.

- **Named mechanism (project-validated; the design's own Section-3
  mandate):** carry the working-memory role-selectivity with the
  validated embodied co-firing + topographic-prior binding mechanism
  (reused byte-unchanged where it exists; net-new = only the in-loop
  wiring that routes it through the prefrontal-slot/concept dimension).
  The temporal-credit signal is relegated to its correct role (credit
  and gating) and is no longer asked to be the binding rule itself.
- **Keep every faithful part already committed (`e02f692`):** the
  basal-ganglia causal wiring (`thal -> dlpfc_verb -> noun_pool`), the
  per-stripe homeostatic equalization (verified working), the
  documented non-zero efferent initialization, and the now-perfect
  episodic binding.
- **Acceptance is the SAME pre-registered frozen gate (unchanged):** on
  the GPU, `v1` wm AND ep >= 0.90 with EVERY drilled binding clearing
  the byte-unchanged no-confabulation gate; `full`+lesions novel probe
  byte-identical. No frozen threshold touched; no-confab moat and every
  protected/validated module byte-unchanged; no autograd; ASCII;
  GPU/CuPy for the real path (numpy only for `--tiny-synth`).
- **Anti-hard-feed control:** selectivity must be LEARNED by co-firing,
  demonstrably absent before training and present only after — not
  pre-wired into the connectivity or fed at query.
- **Pre-committed iteration-4 bound (stated BEFORE the run):** if the
  project's OWN validated co-firing+topographic binding mechanism ALSO
  fails to produce role-selective working-memory binding inside this
  integrated loop at the minimal two-binding load, that is a genuine
  PROGRAM-LEVEL refutation of integrated-loop instrument-soundness at
  this slice — surfaced honestly as a fundamental program decision,
  NOT a further iteration, NOT config-cranking, NOT spin. Stated in
  advance so the outcome cannot be rationalized.
- **Process:** strengthen-only; controller trust-but-verify the diff;
  dedicated adversarial re-review before the no-harm phase (Probe-8
  intact; co-firing selectivity learned not pre-wired; lesions
  faithful; homeostasis scoped; non-zero init not leaking the answer);
  then no-harm; then the controller-only decisive run.

## Execution

Per the owner's standing instruction, execution is same-session subagent-driven. Transition directly to superpowers:subagent-driven-development — do NOT present an execution-choice prompt.
