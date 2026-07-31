---
type: plan
status: live
date: 2026-05-18
---

# Option B — Engram-Bootstrapped Temporal-Credit GENERATIVE Composition (in-bridge) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (standing autonomy pre-selects Subagent-Driven, this session). Task 5 is CONTROLLER-ONLY (not a subagent task) — bring it back to the controller.

**Goal:** Build a kill-safe in-bridge gate that tests whether the validated reward-FREE Tonegawa engram bind BOOTSTRAPS the rewarded episode the compose-bridge VOID lacked (`n_rewarded=0`), so the validated temporal-credit/eligibility mechanism GENERATIVELY refines it — and whether that capability is SCALE-CONFIDENT across a pre-registered local scale ladder `B in {4,8,16}`.

**Architecture:** One net-new module `research/runners/engram_bootstrap_gate.py` (the engram-bootstrap WIRING only). It mirrors the proven `compose_bridge_gate.py` kill-safe scaffold and REUSES byte-UNMODIFIED: `compose_bridge_core.cbr_verdict` (frozen `_CBR_*` inherited, no new movable bar), the Tonegawa engram bridge API, the validated temporal-credit/eligibility path, `build_biological_brain_regions`, `sim.train_checkpoint`, `sim.neuromodulators`. EVERY condition gets the IDENTICAL engram bootstrap; conditions differ ONLY in the temporal-credit refinement on top (the design's mechanism-isolation). A frozen scale ladder + frozen `_SCALE_TOL` give the pre-registered scale-confidence criterion.

**Tech Stack:** Python 3, NumPy CPU backend (`SIM_BACKEND=numpy`), pytest, the in-repo `sim` package. NO autograd/torch anywhere in the shipped path.

**Plan base commit:** `bda6e46` (protected-set empty-diff baseline is `git diff bda6e46..HEAD` on the protected paths; the design+scale-confidence commits `144c6f8/1c18f5f` are docs-only and already on top — use `bda6e46` as the protected-diff base).

---

## Reused interfaces (grounded — do NOT modify these)

- `research/runners/compose_bridge_core.py::cbr_verdict(per_seed: dict) -> dict`
  - Reads per seed: `d["nogap_td"]` (V1), `d["td"]` (science), `d["controls"]` (dict).
  - `_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")` — these EXACT keys must be present in each seed's `controls` dict.
  - Frozen bars: `_CBR_V1_ACC_MIN=0.80`, `_CBR_SCI_ACC_MIN=0.80`, `_CBR_CTRL_ACC_MAX=0.35`, `_CBR_MIN_SEEDS=3`.
  - Returns `{"GATE": "PASS"|"FAIL"|"VOID", "instrument_valid": bool, ...}`. Extra keys in `d` are ignored (safe to add documentation keys).
- `sim.bridge.SimulationBridge` engram API (exact signatures, verified):
  - `start_engram_recording(name: str) -> None`
  - `commit_engram_tag(name, threshold_hz=5.0, top_k=None, region_filter=None) -> dict` (returns `{"name","n_tagged","n_recorded_steps","window_ms","mean_spike_count"}`)
  - `stimulate_tag(name, drive_pA, additive=False) -> int`
  - `clear_tag_drive(name=None) -> None`
- `research/runners/text_minimal_isolation.py::build_biological_brain_regions(...)` — called EXACTLY as in `compose_bridge_gate._build_bridge` (noun_pool_* output pools, language_input verb codes, weak concept-pool dynamics).
- `sim.train_checkpoint.{save_checkpoint, load_checkpoint, resume_epoch}` — per-(seed,rung) atomic checkpoint.
- `sim.kernels.fused_eligibility_trace_decay`, `sim.neuromodulators.{NeuromodulatorConfig,ProductionRule,ModulatorTarget}` — reused exactly as in `compose_bridge_gate`.
- `sim.text_embeddings.orthogonal_drive_pattern(cue_idx, n_cues, n_neurons, drive_max_pA, sparsity)` — the verb drive idiom.

## PROTECTED SET (byte-empty in EVERY commit-scoped diff AND `git diff bda6e46..HEAD`)

```
research/runners/abstention_gate.py
tests/test_abstention_gate.py            # no-confab moat, MUST stay 7/7
sim/td_value_critic.py
sim/compose_temporal_bind.py
sim/kernels.py
sim/bridge.py
sim/neuromodulators.py
sim/train_checkpoint.py
sim/backend.py
sim/dendritic_plasticity.py
research/runners/text_minimal_isolation.py
research/runners/compose_bridge_core.py  # REUSED byte-UNMODIFIED
research/runners/compose_bind_core.py
research/runners/td_critic_core.py
research/runners/dendritic_fair_core.py
research/runners/*_core.py               # every frozen core
```

Controller trust-but-verify EVERY task's `git diff` before marking complete. The genuinely-protected grep is the explicit list above (NOT a bare `*_core` glob — that over-matches the net-new module; the net-new module is `engram_bootstrap_gate.py`, NOT a `*_core`).

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_engram_bootstrap_grounding.py`

**Step 1: Write the grounding test (intentionally references the not-yet-built module)**

```python
"""Task-0 grounding pin for the Option B engram-bootstrap gate.

Intentionally RED until Task 2 ships research/runners/engram_bootstrap_gate.py.
This pins the reused-interface contract so a drift is caught loudly.
"""
import importlib


def test_compose_bridge_core_frozen_bars_unchanged():
    core = importlib.import_module("research.runners.compose_bridge_core")
    # REUSED byte-UNMODIFIED — these frozen bars are INHERITED, never moved.
    assert core._CBR_V1_ACC_MIN == 0.80
    assert core._CBR_SCI_ACC_MIN == 0.80
    assert core._CBR_CTRL_ACC_MAX == 0.35
    assert core._CBR_MIN_SEEDS == 3
    assert core._CONTROLS == ("hebbian_no_trace", "permuted", "wrongsign")


def test_engram_bridge_api_present():
    from sim.bridge import SimulationBridge
    for m in ("start_engram_recording", "commit_engram_tag",
              "stimulate_tag", "clear_tag_drive"):
        assert callable(getattr(SimulationBridge, m, None)), m


def test_engram_bootstrap_gate_exists_and_reuses_core():
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    # Net-new gate REUSES cbr_verdict byte-UNMODIFIED (no new movable bar).
    from research.runners.compose_bridge_core import cbr_verdict
    assert g.cbr_verdict is cbr_verdict
    # Frozen scale ladder + tol pre-registered in the gate (never tuned).
    assert g._SCALE_LADDER == (4, 8, 16)
    assert g._SCALE_TOL == 0.05
    # Scale-confidence aggregator is a pure callable.
    assert callable(g.scale_confidence)
    # NO autograd anywhere in the shipped module source.
    import inspect
    src = inspect.getsource(g)
    assert "autograd" not in src and "torch" not in src
```

**Step 2: Run — verify it FAILS (module not yet built)**

Run: `python -m pytest tests/test_engram_bootstrap_grounding.py -q`
Expected: FAIL (`ModuleNotFoundError: research.runners.engram_bootstrap_gate`) — the pin is intentionally RED; it goes GREEN only after Task 2. This IS the Task-2 gate.

**Step 3: Commit**

```bash
git add tests/test_engram_bootstrap_grounding.py
git commit -m "test: Task-0 grounding pin for Option B engram-bootstrap gate (intentionally RED until Task 2)"
```

**Step 4: Controller verifies** `git diff bda6e46..HEAD -- <PROTECTED SET>` is EMPTY.

---

### Task 1: Pure scale-confidence aggregator (TDD, pure function, no bridge)

The scale-confidence criterion is a pure, deterministic function over per-rung `cbr_verdict` outputs + the two frozen scalars. Build + unit-test it in isolation FIRST (it is recomputable from the recorded JSON; the anti-cheat smell-test depends on it).

**Files:**
- Create: `research/runners/engram_bootstrap_gate.py` (only the constants + `scale_confidence` in this task)
- Test: `tests/test_engram_bootstrap_scale.py`

**Step 1: Write the failing tests**

```python
from research.runners.engram_bootstrap_gate import (
    scale_confidence, _SCALE_LADDER, _SCALE_TOL)


def _rung(B, gate, td, engram_only):
    # Minimal per-rung record shape the aggregator consumes.
    return {"B": B, "verdict": {"GATE": gate}, "td_mean": td,
            "engram_only_mean": engram_only}


def test_all_pass_monotone_is_scale_confident():
    rungs = [_rung(4, "PASS", 0.85, 0.20),
             _rung(8, "PASS", 0.90, 0.22),
             _rung(16, "PASS", 0.92, 0.25)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_works_small_but_plateaus_is_not_confident():
    # PASS at every rung but td DEGRADES beyond tol by B=16.
    rungs = [_rung(4, "PASS", 0.95, 0.20),
             _rung(8, "PASS", 0.88, 0.20),
             _rung(16, "PASS", 0.80, 0.78)]  # (c) fails: margin < tol
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_degradation_beyond_tol_breaks_monotone():
    rungs = [_rung(4, "PASS", 0.95, 0.10),
             _rung(8, "PASS", 0.95, 0.10),
             _rung(16, "PASS", 0.85, 0.10)]  # 0.85 < 0.95 - 0.05 => (b) fails
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_any_void_rung_is_void():
    rungs = [_rung(4, "PASS", 0.9, 0.1),
             _rung(8, "VOID", 0.0, 0.0),
             _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "VOID"


def test_any_fail_rung_is_fail():
    rungs = [_rung(4, "PASS", 0.9, 0.1),
             _rung(8, "FAIL", 0.5, 0.5),
             _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "FAIL"


def test_smallest_rung_must_pass_else_void_or_fail_propagates():
    # Smallest rung VOID dominates (instrument unsound at base scale).
    rungs = [_rung(4, "VOID", 0.0, 0.0),
             _rung(8, "PASS", 0.9, 0.1),
             _rung(16, "PASS", 0.9, 0.1)]
    assert scale_confidence(rungs)["classification"] == "VOID"


def test_frozen_constants_pinned():
    assert _SCALE_LADDER == (4, 8, 16)
    assert _SCALE_TOL == 0.05


def test_non_numeric_or_missing_rung_metric_is_void_not_raise():
    bad = [{"B": 4, "verdict": {"GATE": "PASS"}, "td_mean": "oops",
            "engram_only_mean": 0.1},
           _rung(8, "PASS", 0.9, 0.1), _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(bad)
    assert r["scale_confident"] is False
    assert r["classification"] == "VOID"  # fail-closed, never raise
```

**Step 2: Run — verify FAIL**

Run: `python -m pytest tests/test_engram_bootstrap_scale.py -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`).

**Step 3: Minimal implementation (constants + pure aggregator only)**

Create `research/runners/engram_bootstrap_gate.py` with ONLY this much for Task 1 (the bridge wiring is Task 2):

```python
"""Kill-safe THREE-STATE + SCALE-CONFIDENCE gate: does the validated
reward-FREE Tonegawa engram bind BOOTSTRAP the rewarded episode the
compose-bridge VOID lacked (n_rewarded=0), so the validated temporal-
credit/eligibility mechanism GENERATIVELY refines it -- and is that
capability SCALE-CONFIDENT across a pre-registered local scale ladder?

REUSES byte-UNMODIFIED: compose_bridge_core.cbr_verdict (frozen _CBR_*
INHERITED -- NO new movable bar), the Tonegawa engram bridge API, the
validated temporal-credit/eligibility path, build_biological_brain_
regions, sim.train_checkpoint, sim.neuromodulators. EVERY condition
gets the IDENTICAL engram bootstrap; conditions differ ONLY in the
temporal-credit refinement on top (mechanism isolation). NO automatic
differentiation. ASCII only.

HONEST CEILING (printed, never spun): a SCALE-CONFIDENT PASS = the
generative mechanism works locally at small capacity AND shows no
architectural ceiling across the local ladder (so scale-up is
justified) -- explicitly NOT GPT-class/open-ended fluent composition
on local hardware, NOT an LLM, NOT conversation-solved. A works-small-
but-plateaus result is an honest non-success (NOT a win) that triggers
the autonomous Q2 pivot."""
from __future__ import annotations
import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.compose_bridge_core import cbr_verdict

# Pre-registered, NEVER tuned (mirrors compose_bridge_gate's frozen
# _GAMMA/_LAMBDA pattern). _SCALE_TOL is the substrate's irreducible
# greedy-eval noise floor, justified BEFORE any run.
_SCALE_LADDER = (4, 8, 16)
_SCALE_TOL = 0.05


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def scale_confidence(rungs):
    """Pure, deterministic, fail-closed classification over the ordered
    per-rung records. rungs: list of {"B", "verdict": {"GATE": ...},
    "td_mean", "engram_only_mean"} ordered by ascending B.

    Pre-registered (NEVER tuned):
      (a) every rung GATE == PASS;
      (b) td non-decreasing up to _SCALE_TOL across adjacent rungs;
      (c) at the LARGEST rung td >= _CBR_SCI_ACC_MIN AND
          td - engram_only >= _SCALE_TOL (generative signature holds at
          the hardest scale).
    SCALE-CONFIDENT iff (a)&(b)&(c). Else classify honestly:
      any VOID rung -> VOID; any FAIL rung -> FAIL; all PASS but
      (b)/(c) fails -> WORKS-SMALL-NO-SCALE-CONFIDENCE. Non-numeric/
      missing/unordered -> VOID (never raise)."""
    from research.runners.compose_bridge_core import _CBR_SCI_ACC_MIN
    try:
        ordered = sorted(rungs, key=lambda r: r["B"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by B"}
    if [r.get("B") for r in ordered] != list(_SCALE_LADDER):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder != pre-registered %s"
                          % (_SCALE_LADDER,)}
    gates = []
    for r in ordered:
        v = r.get("verdict")
        g = v.get("GATE") if isinstance(v, dict) else None
        gates.append(g)
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is VOID/missing"}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE is FAIL"}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is not PASS/FAIL/VOID"}
    tds, eos = [], []
    for r in ordered:
        t = _num(r.get("td_mean"))
        e = _num(r.get("engram_only_mean"))
        if t is None or e is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric rung metric"}
        tds.append(t)
        eos.append(e)
    monotone = all(tds[i + 1] >= tds[i] - _SCALE_TOL
                   for i in range(len(tds) - 1))
    top_ok = (tds[-1] >= _CBR_SCI_ACC_MIN
              and (tds[-1] - eos[-1]) >= _SCALE_TOL)
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "all rungs PASS; td monotone up to tol; "
                          "generative signature holds at largest rung",
                "td_by_rung": tds, "engram_only_by_rung": eos}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "all rungs PASS but %s%s"
                      % ("" if monotone else "td degrades beyond tol; ",
                         "" if top_ok else "generative signature absent "
                         "at largest rung"),
            "td_by_rung": tds, "engram_only_by_rung": eos}
```

**Step 4: Run — verify PASS**

Run: `python -m pytest tests/test_engram_bootstrap_scale.py -q`
Expected: PASS (8/8 — the spec test file defines 8 cases; an earlier draft said 9, corrected).

**Step 5: Commit**

```bash
git add research/runners/engram_bootstrap_gate.py tests/test_engram_bootstrap_scale.py
git commit -m "feat: pure pre-registered scale-confidence aggregator (Option B Task 1; reuses cbr_verdict, frozen _SCALE_*)"
```

**Step 6: Controller verifies** `git diff bda6e46..HEAD -- <PROTECTED SET>` EMPTY (the only changed files are the net-new gate module + its test).

---

### Task 2: Engram-bootstrap in-bridge wiring + kill-safe multi-rung CLI

Genuine net-new integration (NOT transcribe-a-reference). Mirror `compose_bridge_gate.py`'s kill-safe scaffold EXACTLY; the ONLY new mechanism is the engram bootstrap and the multi-rung loop. The full reference implementation below IS the spec — write the smoke test first (TDD), then implement to match; the dedicated adversarial reviewer (Task 3) probes mechanism-isolation faithfulness BEFORE Phase B.

**Files:**
- Modify: `research/runners/engram_bootstrap_gate.py` (add the bridge wiring + `main`)
- Test: `tests/test_engram_bootstrap_smoke.py`

**Step 1: Write the failing smoke test**

```python
"""--tiny-synth smoke: the gate RUNS end-to-end in-bridge, emits the
cbr_verdict-shaped per-rung structure with n_rewarded>0 (proves the
engram bootstrap dissolved the compose-bridge n_rewarded=0 cause), and
the toy verdict is explicitly NOT propagated."""
import json
import subprocess
import sys
import tempfile
import os


def test_tiny_synth_smoke_runs_and_bootstraps(tmp_path):
    out = tmp_path / "smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.engram_bootstrap_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "SIM_BACKEND": "numpy"})
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["note"].startswith("TINY-SYNTH")          # NOT propagated
    assert d["tiny_synth"] is True
    # Single shrunk rung in tiny mode; cbr_verdict-shaped.
    assert len(d["ladder"]) == 1
    rung = d["ladder"][0]
    assert rung["verdict"]["GATE"] in ("PASS", "FAIL", "VOID")
    # The decisive bootstrap evidence: n_rewarded>0 for the td condition
    # on at least one seed (the compose-bridge VOID had n_rewarded==0).
    nrew = [s.get("n_rewarded_td", 0)
            for s in rung["verdict"]["per_seed"].values()]
    assert max(nrew) > 0, "engram bootstrap failed to produce a "\
                          "rewarded episode (n_rewarded still 0)"
    assert "scale_confident" in d and d["scale_confident"] in (True, False)
```

**Step 2: Run — verify FAIL**

Run: `python -m pytest tests/test_engram_bootstrap_smoke.py -q`
Expected: FAIL (`main` not implemented / `--tiny-synth` unknown).

**Step 3: Implement the bridge wiring + main (append to `engram_bootstrap_gate.py`)**

Append the following. It mirrors `compose_bridge_gate.py` (`_build_bridge`, `_verb_drive`, `_step`, `_greedy_score`, `_da_modulator_from_delta`, kill-safe `main`) — reuse those verbatim where identical — and ADDS: (i) `_encode_engram` (reward-FREE one-shot bind), (ii) `stimulate_tag` reactivation inside `_episode` so `n_rewarded>0`, (iii) the `engram_only` mode (the faithful storage-only analog, emitted under the inherited control key `hebbian_no_trace` with an explicit `controls_semantics` annotation), (iv) the multi-rung ladder loop.

```python
import numpy as np

from research.runners.text_minimal_isolation import (
    build_biological_brain_regions)
from sim.kernels import fused_eligibility_trace_decay  # noqa: F401 (parity)
from sim.train_checkpoint import save_checkpoint  # kill-safe
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: scale-confidence PoC ONLY -- generative "
           "mechanism local at small capacity + no architectural "
           "ceiling across the local ladder; NOT GPT-class/open-ended "
           "fluent composition on local hardware, NOT an LLM, NOT "
           "conversation-solved. works-small-but-plateaus = honest "
           "non-success -> autonomous Q2 pivot.")
# In THIS build the inherited control key "hebbian_no_trace" carries
# the design's `engram_only` semantics: byte-identical to td (SAME
# engram bootstrap, SAME stimulate_tag, SAME drive/gap/readout/reward/
# RNG consumption) EXCEPT the eligibility trace is suppressed across the
# gap. Documented transparently in controls_semantics (serialized).
_CONTROLS_SEMANTICS = {
    "hebbian_no_trace": "engram_only: identical to td incl. the engram "
                        "bootstrap+stimulate_tag, MINUS EXACTLY the "
                        "eligibility-trace bridging across the gap (the "
                        "faithful storage-only analog; NOT a strawman).",
    "permuted": "identical engram bootstrap; pi(verb->motor) re-"
                "randomized per episode (reward decorrelated).",
    "wrongsign": "identical engram bootstrap; TD delta sign-flipped."}

_GAMMA = 0.95
_LAMBDA = 0.9
_N_BINDINGS_TINY = 4

# Per-rung topology (pre-registered, NEVER tuned): n_lang_input=64*B,
# sparsity=0.5/B  => stride=n_lang_input//B=64 constant, n_active=
# round(sparsity*n_lang_input)=32 < 64 (orthogonal verb codes non-
# overlapping at EVERY rung). n_per_pool/n_fs fixed.
def _params_for(B, tiny):
    if tiny:
        return dict(B=_N_BINDINGS_TINY, n_lang_input=64 * _N_BINDINGS_TINY,
                    sparsity=0.5 / _N_BINDINGS_TINY, n_per_pool=8,
                    n_fs_per_pool=2, stim_steps=4, gap_steps=3,
                    reset_steps=2, readout_steps=4, encode_steps=4,
                    n_train_epochs=2, drive_pA=260.0, teacher_pA=420.0,
                    engram_stim_pA=600.0, engram_top_k=24)
    return dict(B=B, n_lang_input=64 * B, sparsity=0.5 / B,
                n_per_pool=40, n_fs_per_pool=6, stim_steps=24,
                gap_steps=14, reset_steps=10, readout_steps=18,
                encode_steps=20, n_train_epochs=10, drive_pA=260.0,
                teacher_pA=420.0, engram_stim_pA=600.0, engram_top_k=120)


def _da_modulator_from_delta():
    # Catalog C.30 via the REUSED NM subsystem UNMODIFIED (constructed,
    # not mutated) -- identical to compose_bridge_gate.
    return NeuromodulatorConfig(
        name="dopamine_engram_bootstrap", baseline=0.0,
        decay_tau_ms=50.0, concentration_min=-5.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0, threshold=0.0,
                                         window_ms=0.0)])


def _pool_names(B):
    return ["P%d" % i for i in range(B)]


def _build_bridge(seed, P):
    """REUSED build_biological_brain_regions UNMODIFIED -- identical
    construction to compose_bridge_gate._build_bridge (weak concept-pool
    dynamics, native three-factor path ON, NMDA on, hebbian off,
    stdp_w_max=8.0)."""
    from sim.config import (CoreSimConfig, VisualizationConfig,
                            RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    names = _pool_names(P["B"])
    regions, pathways = build_biological_brain_regions(
        n_lang_input=P["n_lang_input"], n_motor_per_action=8,
        enable_motor_fs=False, enable_noun_pools=True,
        noun_pool_names=list(names), n_noun_per_pool=P["n_per_pool"],
        n_noun_fs_per_pool=P["n_fs_per_pool"],
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.05
    cfg.reward_eligibility_tau_ms = 200.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg,
                              viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(),
                              gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _verb_drive(verb_idx, B, n_lang_input, P):
    from sim.text_embeddings import orthogonal_drive_pattern
    return orthogonal_drive_pattern(
        cue_idx=verb_idx, n_cues=B, n_neurons=n_lang_input,
        drive_max_pA=P["drive_pA"], sparsity=P["sparsity"])


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1


def _encode_engram(bridge, tag, verb_idx, target_pool_idx, P,
                   lang_arr, pool_arrs):
    """REWARD-FREE one-shot Tonegawa bind (catalog D.14). Drive verb V_i
    + teacher current on target pool inside an open recording window;
    commit the co-fired ensemble (region_filter spans the target pool so
    stimulate_tag later reactivates it). NO reward here -- this is the
    bootstrap the compose-bridge run LACKED. IDENTICAL for every
    condition."""
    cp = bridge.xp if hasattr(bridge, "xp") else np
    B = P["B"]
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)
    bridge.start_engram_recording(tag)
    drive = cp.asarray(_verb_drive(verb_idx, B, lang_arr.shape[0], P),
                        dtype=cp.float32)
    bridge.cp_external_input_current[lang_arr] = drive
    bridge.cp_external_input_current[pool_arrs[target_pool_idx]] += \
        float(P["teacher_pA"])
    for _ in range(P["encode_steps"]):
        _step(bridge)
    bridge.commit_engram_tag(tag, top_k=int(P["engram_top_k"]))
    bridge.cp_external_input_current[:] = 0.0


def _episode(bridge, mode, verb_idx, target_pool_idx, tag, P,
             lang_arr, pool_arrs, value_table):
    """One refinement episode. IDENTICAL bootstrap-reactivation for
    every mode: stimulate_tag(tag) at t_A so the bound ensemble fires ->
    readout selects the target -> reward r=1 (n_rewarded>0; the native
    eligibility->reward path engages). Conditions differ ONLY in the
    gap: td lets the eligibility trace bridge it; hebbian_no_trace
    (=engram_only) zeroes cp_eligibility_trace EACH gap step (the ONLY
    difference vs td); wrongsign flips delta; permuted re-randomizes pi
    (handled by caller). reward strictly DELAYED past the gap."""
    cp = bridge.xp if hasattr(bridge, "xp") else np
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)
    # t_A: reactivate the bootstrapped ensemble (IDENTICAL all modes) +
    # the verb code (so the lang_input->pool synapses co-fire and the
    # native eligibility trace charges on the path under test).
    drive = cp.asarray(_verb_drive(verb_idx, P["B"], lang_arr.shape[0],
                                   P), dtype=cp.float32)
    bridge.cp_external_input_current[lang_arr] = drive
    bridge.stimulate_tag(tag, drive_pA=float(P["engram_stim_pA"]),
                         additive=True)
    for _ in range(P["stim_steps"]):
        _step(bridge)
    # TEMPORAL GAP: no decision drive; reward strictly 0 here.
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(P["gap_steps"]):
        if mode == "hebbian_no_trace" and \
                bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = 0.0  # engram_only: ONLY diff
        _step(bridge)
    # t_R: population-vote readout.
    counts = np.zeros(P["B"], dtype=np.float64)
    for _ in range(P["readout_steps"]):
        _step(bridge)
        fired = bridge.cp_firing_states
        for j, pa in enumerate(pool_arrs):
            counts[j] += float(fired[pa].sum())
    selected = int(np.argmax(counts))
    reward = 1.0 if selected == target_pool_idx else 0.0
    v = float(value_table[verb_idx])
    delta = reward - v
    value_table[verb_idx] = v + (1.0 - _GAMMA * _LAMBDA) * delta
    if mode == "wrongsign":
        delta = -delta
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = float(delta)
    _step(bridge)
    bridge.core_config.current_reward_signal = 0.0
    return reward, selected


def _bijection(rng, B):
    perm = np.arange(B)
    rng.shuffle(perm)
    return perm


def _greedy_score(bridge, pi, P, lang_arr, pool_arrs):
    """Noise-free greedy accuracy: drive each verb ONLY (NO teacher, NO
    reward, NO stimulate_tag) -> population-vote -> score vs pi. Measures
    what the lang_input->pool synapses LEARNED."""
    cp = bridge.xp if hasattr(bridge, "xp") else np
    B = P["B"]
    correct = 0
    for vi in range(B):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(P["reset_steps"]):
            _step(bridge)
        drive = cp.asarray(_verb_drive(vi, B, lang_arr.shape[0], P),
                            dtype=cp.float32)
        bridge.cp_external_input_current[lang_arr] = drive
        for _ in range(P["stim_steps"] + P["gap_steps"]):
            _step(bridge)
        counts = np.zeros(B, dtype=np.float64)
        for _ in range(P["readout_steps"]):
            _step(bridge)
            fired = bridge.cp_firing_states
            for j, pa in enumerate(pool_arrs):
                counts[j] += float(fired[pa].sum())
        if int(np.argmax(counts)) == int(pi[vi]):
            correct += 1
    bridge.cp_external_input_current[:] = 0.0
    return correct / float(B)


def _run_mode(mode, seed, P, gap_zero=False):
    """Returns (greedy_accuracy, n_rewarded). gap_zero forces G=0 (V1).
    The engram bootstrap (one-shot, reward-FREE) is performed for EVERY
    binding for EVERY mode IDENTICALLY before refinement."""
    Pl = dict(P)
    if gap_zero:
        Pl["gap_steps"] = 0
    B = Pl["B"]
    bridge = _build_bridge(seed, Pl)
    cp = bridge.xp if hasattr(bridge, "xp") else np
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                          dtype=cp.int64)
    names = _pool_names(B)
    pool_arrs = [cp.asarray(list(rm.indices("noun_pool_%s" % nm)),
                            dtype=cp.int64) for nm in names]
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 1.0)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    pi = _bijection(rng, B)
    value_table = np.zeros(B, dtype=np.float64)
    # --- ENGRAM BOOTSTRAP: one reward-FREE bind per (verb,pool) pair,
    # IDENTICAL for every mode (the dissolution of n_rewarded=0). ---
    tags = {}
    for vi in range(B):
        tag = "boot_%d_%d" % (seed, vi)
        _encode_engram(bridge, tag, vi, int(pi[vi]), Pl, lang_arr,
                        pool_arrs)
        tags[vi] = tag
    n_rewarded = 0
    for _ep in range(Pl["n_train_epochs"]):
        if mode == "permuted":
            pi = _bijection(rng, B)  # reward mapping decorrelated
        order = np.arange(B)
        rng.shuffle(order)
        for vi in order:
            r, _sel = _episode(bridge, mode, int(vi), int(pi[vi]),
                               tags[int(vi)], Pl, lang_arr, pool_arrs,
                               value_table)
            n_rewarded += int(r > 0.0)
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 0.0)
    except Exception:
        pass
    return _greedy_score(bridge, pi, Pl, lang_arr, pool_arrs), n_rewarded


def _run_seed(seed, P):
    """Full per-seed row: V1 (td gap=0), science (td), the 3 controls.
    Records n_rewarded for the td condition (decisive smell-test
    evidence: the compose-bridge VOID had n_rewarded==0)."""
    nogap, _ = _run_mode("td", seed, P, gap_zero=True)
    td, nrew_td = _run_mode("td", seed, P, gap_zero=False)
    controls = {}
    for c in _CONTROLS:
        acc, _ = _run_mode(c, seed, P, gap_zero=False)
        controls[c] = acc
    return {"nogap_td": nogap, "td": td, "controls": controls,
            "n_rewarded_td": nrew_td,
            "controls_semantics": _CONTROLS_SEMANTICS}


def _run_rung(B, seeds, tiny, ckpt):
    P = _params_for(B, tiny)
    per_seed = {}
    for s in seeds:
        row = _run_seed(s, P)
        if ckpt:
            save_checkpoint(ckpt, (B * 1000 + s),
                            {"row": [row["nogap_td"], row["td"]]},
                            None, [])
        per_seed[s] = row
    verdict = cbr_verdict(per_seed)
    tds = [per_seed[s]["td"] for s in sorted(per_seed)]
    eos = [per_seed[s]["controls"]["hebbian_no_trace"]
           for s in sorted(per_seed)]
    return {"B": B, "verdict": verdict,
            "td_mean": float(np.mean(tds)),
            "engram_only_mean": float(np.mean(eos))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds for the pre-registered gate")
        return 2
    _ = _da_modulator_from_delta()  # construct (not mutate)
    ladder = (_N_BINDINGS_TINY,) if a.tiny_synth else _SCALE_LADDER
    rungs = []
    try:
        for B in ladder:
            rungs.append(_run_rung(B, a.seeds, a.tiny_synth, a.ckpt))
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    sc = scale_confidence(rungs) if not a.tiny_synth else {
        "scale_confident": False,
        "classification": "TINY-SYNTH (toy; NOT propagated)"}
    out = {"ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"],
           "scale_reason": sc.get("reason", ""), "banner": _BANNER,
           "tiny_synth": bool(a.tiny_synth)}
    if a.tiny_synth:
        out["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    else:
        out["note"] = ("multi-rung scale-confidence verdict -- "
                       "recompute from this JSON; no re-run/no tuning")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("SCALE=%s  classification=%s  %s"
          % (out["scale_confident"], out["scale_classification"],
             _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run the smoke test — verify PASS (and `n_rewarded>0`)**

Run: `python -m pytest tests/test_engram_bootstrap_smoke.py -q`
Expected: PASS. If `max(n_rewarded)==0` the engram bootstrap is not engaging — that is a Task-2 implementation bug to root-cause (systematic-debugging), NOT a science result and NOT a reason to touch any frozen bar.

**Step 5: Run the Task-0 grounding pin — verify it now PASSES**

Run: `python -m pytest tests/test_engram_bootstrap_grounding.py tests/test_engram_bootstrap_scale.py -q`
Expected: PASS (all). Task 0 is now GREEN by design.

**Step 6: Commit**

```bash
git add research/runners/engram_bootstrap_gate.py tests/test_engram_bootstrap_smoke.py
git commit -m "feat: engram-bootstrap in-bridge wiring + kill-safe multi-rung scale ladder (Option B Task 2)"
```

**Step 7: Controller trust-but-verify** `git diff bda6e46..HEAD -- <PROTECTED SET>` is EMPTY. The only changed paths across the whole branch must be: `research/runners/engram_bootstrap_gate.py`, `tests/test_engram_bootstrap_*.py`, and the two design docs. If ANY protected path appears, STOP and root-cause (do NOT proceed to Task 3).

---

### Task 3: DEDICATED ADVERSARIAL REVIEWER (BEFORE Phase B — load-bearing)

Dispatch a fresh adversarial-reviewer subagent against `research/runners/engram_bootstrap_gate.py` (+ its tests). It does NOT rubber-stamp; it tries to break the science integrity. STRENGTHEN-only fixes; frozen bars (`_CBR_*`, `_SCALE_LADDER`, `_SCALE_TOL`) byte-unchanged.

**The reviewer MUST explicitly probe and report on each:**

1. **Mechanism isolation (decisive).** Is the in-bridge discrimination genuinely isolated to the temporal-credit GENERATIVE refinement *on top of an IDENTICAL engram bootstrap*? Verify `hebbian_no_trace` (=`engram_only`) is byte-identical to `td` in EVERY respect — same `_encode_engram`, same `stimulate_tag` call, same drive/gap/readout/reward, same RNG draw count/order — EXCEPT exactly `bridge.cp_eligibility_trace[:]=0.0` each gap step. Trace the RNG consumption per mode. If `engram_only` is crippled anywhere else → strawman → STRENGTHEN-fix.
2. **Bootstrap actually dissolves `n_rewarded=0`.** Confirm the engram `stimulate_tag` reactivation genuinely makes the readout select the target so `n_rewarded>0` for `td` (and that the recorded `n_rewarded_td` is real, not hard-coded). If the bootstrap doesn't engage, the gate must VOID (V1 unmet), NOT fabricate a PASS.
3. **`engram_only` is the FAITHFUL storage-only baseline, not a strawman:** it gets the full bootstrap + stimulate; it can only fail because the delayed reward has no eligibility bridge across the gap. The science signature `td > engram_only` must reflect generative refinement, not a handicapped control. If the engram bootstrap ALONE already saturates greedy accuracy (so `td == engram_only`), the gate must honestly FAIL (temporal-credit adds nothing generative) — verify it does NOT get scored PASS.
4. **Byte-UNMODIFIED reuse:** `cbr_verdict` imported + used unmodified (`g.cbr_verdict is cbr_verdict`); engram API, temporal-credit path, `build_biological_brain_regions`, `train_checkpoint`, NM all reused, not copy-paste-tweaked. Confirm `git diff bda6e46..HEAD -- <PROTECTED SET>` EMPTY.
5. **No movable bar / no PASS-from-broken-instrument:** the inherited `_CBR_*` are not shadowed/overridden; a non-discriminating or V1-broken rung cannot be classified PASS or SCALE-CONFIDENT (test by feeding `scale_confidence` adversarial rung records). `_SCALE_LADDER`/`_SCALE_TOL` not result-movable.
6. **Scale-confidence honesty:** `scale_confidence` cannot return SCALE-CONFIDENT-PASS unless every rung PASS AND monotone-up-to-tol AND generative signature at the largest rung. A works-small-but-plateaus input → `WORKS-SMALL-NO-SCALE-CONFIDENCE` (not spun). Fail-closed on non-numeric/missing.
7. **No autograd/torch** anywhere in the shipped path.

**Loop:** reviewer reports issues → fresh implementer subagent applies STRENGTHEN-only fixes (frozen bars byte-unchanged, transparently logged in the commit message) → reviewer re-reviews → repeat until the reviewer reports NO science-integrity holes. Commit each fix; controller verifies protected-empty each time.

---

### Task 4: LOAD-BEARING no-harm (Phase B)

**Files:**
- Create: `tests/test_engram_bootstrap_no_harm.py`

**Step 1: Write the no-harm test**

```python
"""Phase B no-harm: the net-new gate harms NOTHING protected."""
import subprocess, sys, importlib, inspect


def test_protected_set_byte_empty_diff():
    protected = [
        "research/runners/abstention_gate.py",
        "tests/test_abstention_gate.py",
        "sim/td_value_critic.py", "sim/compose_temporal_bind.py",
        "sim/kernels.py", "sim/bridge.py", "sim/neuromodulators.py",
        "sim/train_checkpoint.py", "sim/backend.py",
        "sim/dendritic_plasticity.py",
        "research/runners/text_minimal_isolation.py",
        "research/runners/compose_bridge_core.py",
        "research/runners/compose_bind_core.py",
        "research/runners/td_critic_core.py",
        "research/runners/dendritic_fair_core.py"]
    d = subprocess.run(["git", "diff", "bda6e46..HEAD", "--", *protected],
                        capture_output=True, text=True)
    assert d.stdout.strip() == "", "PROTECTED set changed:\n" + d.stdout


def test_no_confab_moat_still_7_of_7():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_abstention_gate.py", "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_no_autograd_in_shipped_path():
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    src = inspect.getsource(g)
    assert "autograd" not in src and "import torch" not in src


def test_cbr_verdict_reused_byte_identical():
    from research.runners.compose_bridge_core import cbr_verdict
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    assert g.cbr_verdict is cbr_verdict
```

**Step 2: Run — verify PASS**

Run: `python -m pytest tests/test_engram_bootstrap_no_harm.py tests/test_abstention_gate.py -q`
Expected: PASS (no-harm 4/4, moat 7/7).

**Step 3: Run the full relevant suite**

Run: `python -m pytest tests/test_engram_bootstrap_grounding.py tests/test_engram_bootstrap_scale.py tests/test_engram_bootstrap_smoke.py tests/test_engram_bootstrap_no_harm.py tests/test_abstention_gate.py -q`
Expected: ALL PASS.

**Step 4: Commit**

```bash
git add tests/test_engram_bootstrap_no_harm.py
git commit -m "test: Phase B LOAD-BEARING no-harm — protected byte-empty + moat 7/7 + no autograd (Option B Task 4)"
```

**Step 5: Controller** verifies `git diff bda6e46..HEAD -- <PROTECTED SET>` EMPTY and the moat is 7/7.

---

### Task 5: CONTROLLER-ONLY decisive in-sim run + anti-cheat smell-test + honest propagation

**NOT a subagent task.** The controller performs this directly and brings it back.

**Step 1: Grounding-first tiny run (toy verdict NOT propagated)**

```bash
SIM_BACKEND=numpy python -m research.runners.engram_bootstrap_gate \
    --tiny-synth --seeds 42 43 44 \
    --out research/findings/raw/engram_bootstrap_tiny.json
```
Confirm: exit 0, `note` starts "TINY-SYNTH", `max(n_rewarded_td)>0`. The toy verdict is NEVER propagated.

**Step 2: Decisive kill-safe multi-seed, multi-rung run (FIXED pre-registered config)**

```bash
SIM_BACKEND=numpy python -m research.runners.engram_bootstrap_gate \
    --seeds 42 43 44 45 46 \
    --ckpt research/findings/raw/ebt_ckpt \
    --out research/findings/raw/engram_bootstrap_gate.json
```
Ladder `B in {4,8,16}`, 5 seeds. KILL-SAFE: per-(rung,seed) atomic checkpoint via REUSED `save_checkpoint`; `KeyboardInterrupt` → exit 130, resumable. If the run is interrupted, re-invoke the SAME command (do NOT change config).

**Step 3: MANDATORY anti-cheat smell-test (scrutinize a nominal PASS HARDER than a FAIL).** Recompute from the SINGLE recorded JSON — NO re-run, NO bar-tuning, NO overclaim:
- Per rung: `cbr_verdict` recomputed; V1 (`nogap_td`) genuine + non-degenerate (the degenerate floor is far below 0.80; `engram_only`≈that floor, NOT ≈`td`); `controls` (`hebbian_no_trace`/`permuted`/`wrongsign`) all ≤ 0.35.
- **Decisive bootstrap evidence:** `n_rewarded_td > 0` for ≥3 seeds at every rung (proves the engram bootstrap dissolved the compose-bridge `n_rewarded=0` cause; if `n_rewarded_td==0`, V1 cannot be met → honest VOID, NOT a fabricated PASS).
- **Generative signature genuinely isolated:** `td − engram_only ≥ _SCALE_TOL` at every rung that PASSes; if `td ≈ engram_only`, the honest reading is FAIL (temporal-credit adds nothing generative) — do NOT spin.
- **Scale-confidence recomputed** by `scale_confidence(ladder)` from the recorded JSON; classification ∈ {SCALE-CONFIDENT-PASS, WORKS-SMALL-NO-SCALE-CONFIDENCE, FAIL, VOID}.
- Confirm `git diff bda6e46..HEAD -- <PROTECTED SET>` EMPTY; moat 7/7.

**Step 4: Honest propagation — EVERY outcome.** Write `research/findings/2026-05-18-option-B-engram-bootstrap-temporal-credit-<CLASSIFICATION>.md` (no spin; honest ceiling verbatim; the cheap-precursor-INAPPLICABLE finding restated; `n_rewarded` evidence; per-rung table). Append `webapp/capability_status.json` pillar **n=74** with status:
- `SCALE-CONFIDENT-PASS` → `status: "VALIDATED"`, metric = "in-bridge generative composition; scale-confident across B∈{4,8,16}".
- `WORKS-SMALL-NO-SCALE-CONFIDENCE` → `status: "BOUNDARY"`, metric states works-small / plateaus-with-scale (honest non-success).
- `VOID` → `status: "BOUNDARY"`; `FAIL` → `status: "NEGATIVE"`.
Run `python -m pytest tests/test_webapp_server.py -k capability_status -q` (schema green; fix the JSON not the test if it drifts). Bump `as_of`.

**Step 5: Push BOTH remotes.**
```bash
git add research/findings/2026-05-18-option-B-*.md webapp/capability_status.json research/findings/raw/engram_bootstrap_gate.json
git commit -m "findings: Option B engram-bootstrap temporal-credit in-bridge -- <CLASSIFICATION> (n=74); honest, scale-confidence-gated, NOT spun"
git push origin main && git push gitea main
```

**Step 6: Pivot rule (NON-STOP, no owner-deferral).** If classification is NOT `SCALE-CONFIDENT-PASS`: the finding is propagated (done above) and the arc IMMEDIATELY pivots to **Q2** (two-module constrained decoding) per the design's CORRECTED OPERATING MODE — write the Q2 design doc → writing-plans → subagent-driven → its own pre-registered THREE-STATE + scale-ladder gate → honest propagation. Do NOT stop, do NOT defer to the owner. If `SCALE-CONFIDENT-PASS`: report the validated scale-confident result to the controller/owner (this IS the requested deliverable) and continue the standing arc.

---

## Notes for the executor

- DRY/YAGNI/TDD; frequent commits; exact paths; complete code above (not "add wiring").
- `@superpowers:systematic-debugging` for any `n_rewarded==0` / smoke failure — root-cause, never paper over, NEVER touch a frozen bar to force green.
- `@superpowers:subagent-driven-development` drives Tasks 0–4 (fresh subagent per task; spec-review then code-quality-review each; dedicated adversarial reviewer = Task 3 before Phase B).
- Task 5 is CONTROLLER-ONLY — bring it back.
- The honest ceiling is baked into `_BANNER` and the findings doc and is NEVER spun: scale-confidence PoC, NOT GPT-class/open-ended fluency on local hardware. A works-small-but-plateaus result is an honest non-success that triggers the Q2 pivot.
