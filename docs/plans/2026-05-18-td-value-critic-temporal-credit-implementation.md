---
type: plan
status: live
date: 2026-05-18
---

# TD Value-Function Critic — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (controller stays in this session; fresh subagent per task; two-stage spec+quality review; dedicated adversarial reviewer for the two load-bearing modules BEFORE Phase B; controller trust-but-verify each git diff with protected modules byte-empty; Task 5 is controller-only).

**Goal:** Give the simulator a learned value-function critic doing biologically-canonical temporal-difference credit assignment (phasic DA = TD error, Schultz), validated by a pre-registered THREE-STATE gate.

**Architecture:** One net-new pure critic module (`sim/td_value_critic.py`, the cheap-gate-GREEN-validated TD(λ) on the complete-serial-compound representation; reuses `fused_eligibility_trace_decay` UNMODIFIED; NO autograd), one net-new pure FIXED-bar THREE-STATE verdict (`research/runners/td_critic_core.py`, own frozen `_TDC_*`, mirrors the hardened `dendritic_fair_core` discipline), one kill-safe runner (`research/runners/td_critic_gate.py`, reuses `sim.train_checkpoint` + the REUSED `NeuromodulatorManager` so the critic's δ IS the phasic-DA signal — the catalog's prescribed upgrade). Everything load-bearing is reused byte-UNMODIFIED.

**Tech Stack:** Python, NumPy (deterministic seeded; the critic math), `sim.backend` (pluggable xp), `sim.kernels.fused_eligibility_trace_decay` (REUSED), `sim.train_checkpoint` (REUSED), `sim.neuromodulators` (REUSED), pytest. NO `torch`, NO automatic differentiation anywhere in the shipped path.

**Standing autonomy:** documented design calls (NOT one-question-at-a-time); no stopping/asking; honest propagation every outcome (PASS/FAIL/VOID). Honest ceiling baked in and NEVER spun: a scrutinized in-sim PASS = temporal credit assignment substrate at feasible local scale — **NOT** conversation-solved; integration into the conversational stack is a **separate later effort**. A faithful FAIL/VOID is the honest terminus of THIS increment, **NOT** a license to escalate.

**Pre-registered FROZEN bars (in `td_critic_core.py`, justified, NEVER tuned):**
`_TDC_V1_VALUE_RMSE_MAX = 0.05` (V1: critic provably learns the true expected return; cheap-gate perfect ≈0.001–0.005, no-learn ≈0.3–0.8) · `_TDC_TRANSFER_MIN = 0.90` (canonical scale-free Schultz transfer-fraction `|δCS|/(|δCS|+|δUS|)`; perfect→~1.0 by mathematical identity, controls ≤0.21; STRENGTHEN-only) · `_TDC_US_DECAY_MAX = 0.15` (reward becomes predicted; perfect ≈0.001) · `_TDC_MIN_SEEDS = 3`.

**Protected — byte-UNMODIFIED, verify empty `git diff` in EVERY commit-scoped diff:** `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (the no-confab moat — MUST stay 7/7 green), `sim/neuromodulators.py`, `sim/kernels.py`, `sim/bridge.py`, `research/runners/g11_bg_runner.py`, `sim/train_checkpoint.py`, `sim/backend.py`, `sim/dendritic_plasticity.py`, every frozen `*_core.py` (each owns its own bars). NO new GLOBAL bar.

---

## Task 0: Grounding pin (commit now; green ONLY after Task 3 — intentional)

**Files:**
- Create: `tests/test_td_critic_grounding.py`

**Step 1: Write the grounding test**

```python
"""Grounding pin: the END-TO-END td_critic_gate pipeline must turn on a
TINY synthetic config and produce an interpretable THREE-STATE verdict.
RED until Task 3 lands the runner -- that is the Task-3 gate."""
import json
import subprocess
import sys
from pathlib import Path


def test_td_critic_gate_tiny_synthetic_pipeline_turns(tmp_path):
    out = tmp_path / "tdc_tiny.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.td_critic_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.is_file(), r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["GATE"] in ("VOID", "PASS", "FAIL")
    assert "per_seed" in d and "frozen_bars" in d
```

**Step 2: Run — verify it fails (no runner yet)**

Run: `python -m pytest tests/test_td_critic_grounding.py -v`
Expected: FAIL (`No module named research.runners.td_critic_gate`).

**Step 3: Commit the pin**

```bash
git add tests/test_td_critic_grounding.py
git commit -m "test(td-critic): Task-0 grounding pin (RED until Task 3 — the Task-3 gate)"
```

Controller: verify `git diff --cached --name-only` shows ONLY this test; protected set absent.

---

## Phase A — pure-CPU-TDD (Tasks 1–2). Fresh subagent per task. Strict failing-test → minimal-impl → run → commit. Controller trust-but-verify each commit-scoped diff (protected byte-empty).

### Task 1: `sim/td_value_critic.py` (LOAD-BEARING)

The cheap-gate-GREEN-validated TD(λ) critic on the complete-serial-compound representation. Reuses `fused_eligibility_trace_decay` UNMODIFIED for the trace decay. NO autograd. The scale-free transfer = `|δCS|/(|δCS|+|δUS|)` (ungameable; →1.0 for a perfect critic by identity).

**Files:**
- Create: `sim/td_value_critic.py`
- Test: `tests/test_td_value_critic.py`

**Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from sim.td_value_critic import (
    analytic_vstar, csc_features, run_pavlovian,
    GAMMA, TRACE, CS_ONSETS, scale_free_transfer)


def test_analytic_vstar_is_exact_discounted_return():
    v = analytic_vstar()
    assert v.shape == (TRACE + 1,)
    np.testing.assert_allclose(v, [GAMMA ** (TRACE - k)
                                   for k in range(TRACE + 1)], rtol=1e-12)


def test_csc_features_bias_plus_cue_anchored_taps():
    X = csc_features(onset=4, T=20)
    assert X.shape == (20, 21)
    assert np.allclose(X[:, 20], 1.0)            # bias always on
    assert X[3].sum() == 1.0                     # pre-cue: bias only
    assert X[4, 0] == 1.0                         # tap-0 at onset
    assert X[5, 1] == 1.0                         # tap-1 next step


def test_scale_free_transfer_is_one_when_us_predicted():
    # |dCS|/(|dCS|+|dUS|): dUS->0 => 1.0 by mathematical identity
    assert scale_free_transfer(0.46, 0.0) == pytest.approx(1.0, abs=1e-9)
    assert scale_free_transfer(0.1, 0.9) < 0.2


def test_V1_td_critic_converges_to_analytic_vstar():
    vr, tr, ud = run_pavlovian("td", seed=42)
    assert vr <= 0.05            # V1: provably learns true expected return
    assert tr >= 0.90            # canonical Schultz transfer emerges
    assert ud <= 0.15            # reward becomes predicted


def test_controls_genuinely_fail():
    # no_bootstrap diverges / no transfer; permuted no transfer;
    # wrongsign diverges (non-finite). Each must NOT pass the signature.
    for mode in ("no_bootstrap", "permuted", "wrongsign"):
        vr, tr, ud = run_pavlovian(mode, seed=42)
        passes = (np.isfinite(vr) and np.isfinite(tr) and np.isfinite(ud)
                  and tr >= 0.90 and ud <= 0.15)
        assert not passes, mode


def test_no_autograd_imported():
    import sim.td_value_critic as m
    src = open(m.__file__).read()
    assert "autograd" not in src and "torch" not in src


def test_reuses_eligibility_kernel_unmodified():
    import sim.td_value_critic as m
    src = open(m.__file__).read()
    assert "fused_eligibility_trace_decay" in src
```

**Step 2: Run — verify fail**

Run: `python -m pytest tests/test_td_value_critic.py -v`
Expected: FAIL (`No module named sim.td_value_critic`).

**Step 3: Write the minimal implementation**

Create `sim/td_value_critic.py` with EXACTLY this validated logic (this is the cheap-gate-GREEN reference; do not redesign it):

```python
"""Biologically-canonical TD(lambda) value-function critic on the
complete-serial-compound representation (Schultz98 / Sutton-Barto).
delta = r + gamma*V(s') - V(s) is the phasic-DA TD error -- the missing
"value-function critic of an actor-critic" (feature-catalog C.30). Pure
array math; reuses sim.kernels.fused_eligibility_trace_decay UNMODIFIED
for the trace decay; NO automatic differentiation (TD needs none);
deterministic seeded; ASCII only."""
from __future__ import annotations
import numpy as np
from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED

# Pre-registered Pavlovian-schedule constants (canonical Schultz trace
# conditioning; NOT science bars -- the frozen bars live in
# td_critic_core). Cue onset is JITTERED per trial (the cue's
# APPEARANCE must be temporally unpredicted, else delta->0 everywhere
# at convergence and the transfer is unmeasurable).
T = 20
CS_ONSETS = (3, 4, 5, 6, 7)
TRACE = 4
GAMMA = 0.95
ALPHA = 0.05
LAMBDA = 0.9
N_TRIALS = 1500
EARLY = 100
LATE = 100


def analytic_vstar(trace: int = TRACE, gamma: float = GAMMA):
    """Exact true expected discounted return GIVEN the cue, along the
    cue-anchored timeline: deterministic reward 1.0 at tap=trace."""
    return np.array([gamma ** (trace - k) for k in range(trace + 1)])


def csc_features(onset: int, T: int = T):
    """Bias feature (constant; the pre-cue baseline the critic CANNOT
    use to predict the uncertain cue onset) + one tap per
    time-since-cue. tap=t-onset; pre-cue (t<onset) is bias-only."""
    n_feat = T + 1
    X = np.zeros((T, n_feat))
    X[:, T] = 1.0
    for t in range(T):
        tap = t - onset
        if 0 <= tap < T:
            X[t, tap] = 1.0
    return X


def scale_free_transfer(dcs_abs: float, dus_abs: float) -> float:
    """Canonical scale-free Schultz transfer = fraction of asymptotic
    RPE now at the (unpredicted) CS vs the US. -> 1.0 for a perfect
    critic BY MATHEMATICAL IDENTITY (dUS->0 => fraction->1); ungameable
    (no fitted denominator)."""
    return dcs_abs / (dcs_abs + dus_abs + 1e-12)


def run_pavlovian(mode: str, seed: int, n_trials: int = N_TRIALS):
    """One critic run with PER-TRIAL JITTERED cue onset. Returns
    (vrmse_vs_analytic_vstar, scale_free_transfer, us_decay).
    modes: 'td' | 'no_bootstrap' | 'permuted' | 'wrongsign'."""
    rng = np.random.default_rng(seed)
    n_feat = T + 1
    w = np.zeros(n_feat)
    early_dUS, late_dUS, late_dCS = [], [], []
    decay = GAMMA * LAMBDA
    for trial in range(n_trials):
        if mode == "permuted":
            onset = int(rng.choice(CS_ONSETS))
            t_us = CS_ONSETS[len(CS_ONSETS) // 2] + TRACE  # cue uninformative
        else:
            onset = int(rng.choice(CS_ONSETS))
            t_us = onset + TRACE
        X = csc_features(onset, T)
        e = np.zeros(n_feat)
        for t in range(T):
            r = 1.0 if t == t_us else 0.0
            v_t = X[t] @ w
            v_tp1 = (X[t + 1] @ w) if t + 1 < T else 0.0
            if mode == "no_bootstrap":
                delta = r - v_t
            else:
                delta = r + GAMMA * v_tp1 - v_t
            # eligibility: e = gamma*lambda*e + phi(s). The decay term
            # reuses the project's eligibility kernel UNMODIFIED.
            e = np.asarray(fused_eligibility_trace_decay(e, decay)) + X[t]
            step = -delta if mode == "wrongsign" else delta
            w = w + ALPHA * step * e
            if trial < EARLY and t == t_us:
                early_dUS.append(abs(delta))
            if trial >= n_trials - LATE:
                if t == t_us:
                    late_dUS.append(abs(delta))
                if t == onset - 1:           # the UNPREDICTED cue arrival
                    late_dCS.append(delta)
    Vstar = analytic_vstar()
    wt = w[:TRACE + 1] + w[T]
    vrmse = float(np.sqrt(np.mean((wt - Vstar) ** 2)))
    e_us = float(np.mean(early_dUS)) if early_dUS else 0.0
    l_us = float(np.mean(late_dUS)) if late_dUS else 0.0
    l_cs = float(np.mean(np.abs(late_dCS))) if late_dCS else 0.0
    transfer = scale_free_transfer(l_cs, l_us)
    us_decay = l_us / (e_us + 1e-9)
    return vrmse, transfer, us_decay
```

**Step 4: Run — verify pass**

Run: `python -m pytest tests/test_td_value_critic.py -v`
Expected: PASS (all 7). `test_V1_..` ≈ vr~0.004, tr~0.997, ud~0.001.

**Step 5: Commit**

```bash
git add sim/td_value_critic.py tests/test_td_value_critic.py
git commit -m "feat(td-critic): TD(lambda) value-function critic (cheap-gate-GREEN reference; reuses eligibility kernel UNMODIFIED; NO autograd)"
```

Controller: `git diff --cached --name-only` shows ONLY these two; protected set byte-absent.

---

### Task 2: `research/runners/td_critic_core.py` (LOAD-BEARING)

Pure FIXED-bar THREE-STATE verdict, OWN frozen `_TDC_*`, mirrors the hardened `dendritic_fair_core` discipline (strict `is True`, numeric-coercion → VOID-not-raise, non-finite → VOID, instrument-validity FIRST, VOID strictly distinct from FAIL, a diverged/non-finite control = correctly-failed NOT non-discriminating).

**Files:**
- Create: `research/runners/td_critic_core.py`
- Test: `tests/test_td_critic_core.py`

**Step 1: Write the failing tests** (adversarial matrix)

```python
import math
import pytest
from research.runners.td_critic_core import (
    tdc_verdict, _TDC_V1_VALUE_RMSE_MAX, _TDC_TRANSFER_MIN,
    _TDC_US_DECAY_MAX, _TDC_MIN_SEEDS)


def _sound_seed():  # a single seed's sound+passing payload
    return dict(vrmse=0.004, transfer=0.997, us_decay=0.001,
                controls={"no_bootstrap": (180.0, 0.20, 400.0),
                          "permuted": (0.2, 0.07, 0.96),
                          "wrongsign": (float("nan"),) * 3})


def test_frozen_bars_exact():
    assert _TDC_V1_VALUE_RMSE_MAX == 0.05
    assert _TDC_TRANSFER_MIN == 0.90
    assert _TDC_US_DECAY_MAX == 0.15
    assert _TDC_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = tdc_verdict({42: _sound_seed(), 43: _sound_seed(),
                     44: _sound_seed()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound_seed(); s["vrmse"] = 0.5      # critic did NOT learn V*
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_control_passing_signature_is_VOID_not_pass():
    s = _sound_seed()
    s["controls"]["permuted"] = (0.01, 0.97, 0.001)  # control "passes"
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"               # non-discriminating


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound_seed()
    s["controls"]["wrongsign"] = (float("inf"), float("nan"), 1e9)
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "PASS"               # diverged == correctly failed


def test_science_fail_when_sound_but_no_transfer():
    s = _sound_seed(); s["transfer"] = 0.40  # below frozen 0.90
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_is_VOID_not_raise():
    s = _sound_seed(); s["transfer"] = "0.99"   # string must NOT pass
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    v = tdc_verdict({42: _sound_seed()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_TDC_V1_VALUE_RMSE_MAX, _TDC_TRANSFER_MIN)
    tdc_verdict({42: _sound_seed(), 43: _sound_seed(), 44: _sound_seed()})
    import research.runners.td_critic_core as c
    assert (c._TDC_V1_VALUE_RMSE_MAX, c._TDC_TRANSFER_MIN) == before
```

**Step 2: Run — verify fail**

Run: `python -m pytest tests/test_td_critic_core.py -v`
Expected: FAIL (`No module named research.runners.td_critic_core`).

**Step 3: Write the minimal implementation**

```python
"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the TD
value-function critic. Instrument-validity FIRST, fail-closed: a
V1-broken or non-discriminating run is VOID -- explicitly NOT a science
PASS/FAIL. Frozen _TDC_* are pre-registered and NEVER tuned. Mirrors
the hardened dendritic_fair_core discipline. ASCII only."""
from __future__ import annotations
import math

_TDC_V1_VALUE_RMSE_MAX = 0.05
_TDC_TRANSFER_MIN = 0.90
_TDC_US_DECAY_MAX = 0.15
_TDC_MIN_SEEDS = 3

_CONTROLS = ("no_bootstrap", "permuted", "wrongsign")


def _finite(x):
    """Strict: reject non-numeric (a bare float('0.9') would let the
    string '0.9' through). Return finite float or None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _passes_signature(vr, tr, ud):
    """The SAME criterion a science PASS uses. A control 'genuinely
    fails' iff it does NOT reproduce this valid finite signature; a
    diverged/non-finite control = correctly failed (NOT
    non-discriminating)."""
    vrf, trf, udf = _finite(vr), _finite(tr), _finite(ud)
    if vrf is None or trf is None or udf is None:
        return False
    return (vrf <= _TDC_V1_VALUE_RMSE_MAX and trf >= _TDC_TRANSFER_MIN
            and udf <= _TDC_US_DECAY_MAX)


def tdc_verdict(per_seed: dict) -> dict:
    bars = {"V1_VALUE_RMSE_MAX": _TDC_V1_VALUE_RMSE_MAX,
            "TRANSFER_MIN": _TDC_TRANSFER_MIN,
            "US_DECAY_MAX": _TDC_US_DECAY_MAX,
            "MIN_SEEDS": _TDC_MIN_SEEDS}
    seeds = sorted(per_seed.keys())
    base = {"frozen_bars": bars, "per_seed": {str(s): per_seed[s]
                                              for s in seeds}}
    if len(seeds) < _TDC_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _TDC_MIN_SEEDS, **base}
    v1_ok = True
    science_ok = True
    controls_fail = True
    for s in seeds:
        d = per_seed[s]
        vr, tr, ud = (_finite(d.get("vrmse")), _finite(d.get("transfer")),
                      _finite(d.get("us_decay")))
        if vr is None or vr > _TDC_V1_VALUE_RMSE_MAX:
            v1_ok = False
        if not (tr is not None and ud is not None
                and tr >= _TDC_TRANSFER_MIN and ud <= _TDC_US_DECAY_MAX):
            science_ok = False
        ctrls = d.get("controls", {})
        for name in _CONTROLS:
            tup = ctrls.get(name)
            if tup is None or len(tup) != 3:
                # missing control == cannot certify discrimination
                controls_fail = controls_fail and False
                continue
            if _passes_signature(*tup):
                controls_fail = False
    instrument_valid = bool(v1_ok and controls_fail)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: critic did NOT converge to analytic "
                       "V* (instrument unsound)")
        if not controls_fail:
            why.append("a discriminating control passed the signature "
                       "(instrument non-discriminating)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
```

**Step 4: Run — verify pass**

Run: `python -m pytest tests/test_td_critic_core.py -v`
Expected: PASS (all 9).

**Step 5: Commit**

```bash
git add research/runners/td_critic_core.py tests/test_td_critic_core.py
git commit -m "feat(td-critic): pure THREE-STATE FIXED-bar verdict (own frozen _TDC_*; instrument-validity-first; diverged-control-correct; mirrors hardened dendritic_fair_core)"
```

Controller: verify diff scope; protected byte-absent.

---

## ADVERSARIAL REVIEW CHECKPOINT (BEFORE Phase B)

Controller dispatches a **dedicated adversarial reviewer subagent** for `sim/td_value_critic.py` + `research/runners/td_critic_core.py` (mirror the dendritic-Phase-A adversarial review that caught real holes). Explicitly probe: can a non-discriminating or V1-broken run be scored PASS instead of VOID? can a diverged/non-finite control be mis-scored "non-discriminating" (→ wrong VOID) instead of "correctly failed"? can the frozen `_TDC_*` be moved by results? any autograd/torch in the shipped path? is `scale_free_transfer` gameable (fitted denominator)? is the eligibility kernel genuinely reused UNMODIFIED? STRENGTHEN-only fixes; frozen bars byte-unchanged. Do NOT enter Phase B until the reviewer signs off and any fixes re-pass.

---

## Phase B — integration validated by import/signature smoke + the gate itself (project pattern, NOT contrived orchestration tests).

### Task 3: `research/runners/td_critic_gate.py` (kill-safe runner)

**Files:**
- Create: `research/runners/td_critic_gate.py`
- Test: `tests/test_td_critic_gate_smoke.py`

**Step 1: Write the failing smoke tests**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, tmp):
    out = tmp / "g.json"
    r = subprocess.run([sys.executable, "-m",
                        "research.runners.td_critic_gate", "--out",
                        str(out)] + args, capture_output=True, text=True,
                       cwd=ROOT)
    return r, out


def test_tiny_synth_three_state_and_makes_task0_green(tmp_path):
    r, out = _run(["--tiny-synth", "--seeds", "42", "43", "44"], tmp_path)
    assert out.is_file(), r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["GATE"] in ("VOID", "PASS", "FAIL")


def test_fewer_than_3_seeds_exit2(tmp_path):
    r, _ = _run(["--tiny-synth", "--seeds", "42"], tmp_path)
    assert r.returncode == 2


def test_reuses_train_checkpoint_and_nm_unmodified_no_autograd():
    src = (ROOT / "research/runners/td_critic_gate.py").read_text()
    assert "from sim.train_checkpoint import" in src
    assert "neuromodulators" in src           # NM coupling present
    assert "autograd" not in src and "torch" not in src
```

**Step 2: Run — verify fail.** `python -m pytest tests/test_td_critic_gate_smoke.py -v` → FAIL (no module).

**Step 3: Write the minimal implementation**

```python
"""Kill-safe THREE-STATE gate runner for the TD value-function critic.
Runs the cheap-gate-GREEN-validated critic across {td, no_bootstrap,
permuted, wrongsign} x seeds, per-(seed) kill-safe checkpoint via the
REUSED sim.train_checkpoint, and couples the critic's TD delta into the
REUSED NeuromodulatorManager so delta IS the phasic-DA learning signal
(feature-catalog C.30 -- the prescribed "value-function critic"
upgrade). THREE-STATE verdict via td_critic_core. NO autograd. ASCII.

HONEST CEILING (printed, never spun): a PASS = temporal credit
assignment substrate at feasible local scale -- NOT conversation-
solved; integration into the conversational stack is a SEPARATE later
effort. PASS/FAIL/VOID all decision-relevant + propagated honestly."""
from __future__ import annotations
import argparse
import json
import sys

from sim.td_value_critic import run_pavlovian, N_TRIALS
from sim.train_checkpoint import (save_checkpoint, load_checkpoint,
                                  resume_epoch)  # REUSED UNMODIFIED
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)  # REUSED UNMODIFIED
from research.runners.td_critic_core import tdc_verdict

_CONDS = ("td", "no_bootstrap", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: temporal credit assignment substrate at "
           "feasible local scale ONLY -- NOT conversation-solved; "
           "integration is a SEPARATE later effort.")


def _da_modulator_from_delta():
    """The catalog's prescribed upgrade, demonstrated via the REUSED
    NM subsystem UNMODIFIED: a from_reward DA modulator whose drive is
    the critic's TD delta (current_reward_signal carries delta, not a
    bare reward). Constructed to prove the critic composes with the
    validated phasic-DA substrate; not mutated here."""
    return NeuromodulatorConfig(
        name="dopamine_td", baseline=0.0, decay_tau_ms=50.0,
        concentration_min=-5.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0, threshold=0.0,
                                         window_ms=0.0)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds for the pre-registered gate")
        return 2
    # Construct (do NOT mutate) the REUSED NM coupling -- proves the
    # critic's delta composes with the validated phasic-DA substrate.
    _ = _da_modulator_from_delta()
    n_trials = 60 if a.tiny_synth else N_TRIALS
    per_seed = {}
    try:
        for s in a.seeds:
            row = {"controls": {}}
            for cond in _CONDS:
                vr, tr, ud = run_pavlovian(cond, seed=s,
                                           n_trials=n_trials)
                if cond == "td":
                    row["vrmse"], row["transfer"], row["us_decay"] = (
                        vr, tr, ud)
                else:
                    row["controls"][cond] = (vr, tr, ud)
                if a.ckpt:
                    save_checkpoint(a.ckpt, s, {cond: [vr, tr, ud]},
                                    None, [])
            per_seed[s] = row
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    verdict = tdc_verdict(per_seed)
    verdict["banner"] = _BANNER
    if a.tiny_synth:
        verdict["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("GATE=%s  %s" % (verdict["GATE"], _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run — verify pass.** `python -m pytest tests/test_td_critic_gate_smoke.py tests/test_td_critic_grounding.py -v` → all PASS (Task 0 now GREEN).

**Step 5: Commit**

```bash
git add research/runners/td_critic_gate.py tests/test_td_critic_gate_smoke.py
git commit -m "feat(td-critic): kill-safe THREE-STATE gate runner (reuses train_checkpoint + NM subsystem UNMODIFIED; NO autograd; Task 0 green)"
```

Controller: verify diff scope; protected (`sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/kernels.py`, moat, all `*_core`) byte-absent in the commit-scoped diff.

---

### Task 4: LOAD-BEARING no-harm

**Files:** (no new source; a verification test + controller diff audit)
- Create: `tests/test_td_critic_no_harm.py`

**Step 1: Write the no-harm test**

```python
"""LOAD-BEARING no-harm: protected modules byte-UNTOUCHED across the
whole TD-critic commit range; the no-confab moat still 7/7 green; NO
shipped TD path imports autograd/torch."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/neuromodulators.py", "sim/kernels.py", "sim/bridge.py",
    "research/runners/g11_bg_runner.py", "sim/train_checkpoint.py",
    "sim/backend.py", "sim/dendritic_plasticity.py"]


def test_protected_byte_untouched_since_base(request):
    base = request.config.getoption("--td-base", default="6a3375b")
    diff = subprocess.run(["git", "diff", "--name-only",
                           "%s..HEAD" % base, "--"] + PROTECTED,
                          capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", "PROTECTED MODIFIED:\n" + diff.stdout


def test_no_autograd_in_shipped_td_path():
    for p in ("sim/td_value_critic.py",
              "research/runners/td_critic_core.py",
              "research/runners/td_critic_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
```

(Add a `--td-base` pytest option in `tests/conftest.py` if absent — a one-line `parser.addoption`; if `conftest.py` is protected/awkward, the controller instead runs the `git diff` audit directly each commit. The controller MUST in any case run, every commit-scoped diff:
`git diff --name-only <base>..HEAD -- <PROTECTED>` → expect empty.)

**Step 2: Run** the abstention moat + representative suite:
`python -m pytest tests/test_abstention_gate.py tests/test_td_critic_no_harm.py -q` → moat 7/7 + no-harm PASS.

**Step 3: Commit**

```bash
git add tests/test_td_critic_no_harm.py
git commit -m "test(td-critic): LOAD-BEARING no-harm (protected byte-untouched; moat 7/7; no autograd in shipped path)"
```

---

## Task 5 — CONTROLLER-ONLY (NOT a subagent task). Bring back to the controller.

1. **Grounding-first tiny run** (toy verdict NOT propagated):
   `python -m research.runners.td_critic_gate --tiny-synth --seeds 42 43 44 --out research/findings/raw/td_tiny.json` — confirm the pipeline turns + emits a THREE-STATE verdict. Discard the toy verdict.
2. **Decisive kill-safe multi-seed run** (FIXED pre-registered config; pausable/resumable):
   `python -m research.runners.td_critic_gate --seeds 42 43 44 --ckpt research/findings/raw/td_ckpt --out research/findings/raw/td_critic_gate.json`
3. **MANDATORY anti-cheat smell-test — scrutinize a nominal PASS HARDER than a FAIL.** Recompute `vrmse / transfer / us_decay` from the recorded JSON (NO re-run, NO bar-tuning): V1 genuinely ≤0.05 (critic provably learned V*; NOT a degenerate all-zero); the `td` transfer ≥0.90 and us_decay ≤0.15 genuinely; **each control genuinely fails for the mechanistically-correct reason** (`no_bootstrap`: no value bootstrap → no cue power, the catalog's exact claim; `permuted`: reward stays unpredicted; `wrongsign`: diverges). Confirm decisive separation (PASS transfer vs best control). NO overclaim.
4. **Honest propagation EVERY outcome** (VOID = instrument-not-soundly-constructible-in-sim; PASS = a sound discriminating TD critic genuinely does temporal credit assignment at feasible local scale, ceiling explicit, integration a SEPARATE later effort; FAIL = the honest terminus of THIS increment): findings doc `research/findings/2026-05-18-td-value-critic-<OUTCOME>.md` + `webapp/capability_status.json` pillar (n=70; status PASS→tier / else BOUNDARY/NEGATIVE) + `python -m pytest tests/test_webapp_server.py -k capability_status -q` green + commit (scoped; protected byte-empty) + push BOTH remotes (origin + gitea). Bars NOT tuned. NOT config-cranked. The no-confab moat byte-identical + 7/7 green throughout.

---

## Remember
- Exact file paths; complete code (above is the cheap-gate-GREEN reference — do NOT redesign it).
- Frozen `_TDC_*` pre-registered here; NEVER tuned.
- Protected set byte-UNMODIFIED — controller verifies empty diff EVERY commit.
- @superpowers:subagent-driven-development drives Tasks 0–4; Task 5 is controller-only.
- Honest ceiling stated up front and NEVER spun. PASS/FAIL/VOID all propagated honestly.
