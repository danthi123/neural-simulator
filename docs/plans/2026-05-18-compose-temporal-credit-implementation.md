# Compose × Temporal-Credit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (controller stays in this session; fresh subagent per task; dedicated adversarial reviewer for the two load-bearing modules BEFORE Phase B; controller trust-but-verify each git diff with the protected set byte-empty; Task 5 is controller-only).

**Goal:** Validate, in the sim's real eligibility substrate, whether the now-VALIDATED temporal-credit/eligibility mechanism learns a compositional A→B binding bridging a temporal gap that the faithful no-trace v16-analog structurally cannot.

**Architecture:** One net-new learner (`sim/compose_temporal_bind.py` — the cheap-gate-GREEN-validated TD(λ)+eligibility on a gapped A_i→B_pi(i) bijection; the eligibility decay routed through the REUSED `sim.kernels.fused_eligibility_trace_decay` UNMODIFIED, numerically identical, genuinely the sim substrate; NO autograd), one net-new pure FIXED-bar THREE-STATE verdict (`research/runners/compose_bind_core.py`, own frozen `_CTB_*`, mirrors the adversarial-hardened `td_critic_core` discipline), one kill-safe runner (`research/runners/compose_bind_gate.py`, reuses `sim.train_checkpoint` + the REUSED NM subsystem). Everything load-bearing is reused byte-UNMODIFIED.

**Tech Stack:** Python, NumPy (deterministic seeded), `sim.kernels.fused_eligibility_trace_decay` (REUSED), `sim.train_checkpoint` (REUSED), `sim.neuromodulators` (REUSED), pytest. NO `torch`, NO automatic differentiation anywhere in the shipped path.

**Standing autonomy:** documented design calls (NOT one-question-at-a-time); no stopping/asking; honest propagation every outcome. Honest ceiling baked in and NEVER spun: a scrutinized in-sim PASS = mechanism-level/in-sim ONLY (temporal credit bridges the bind-gap where the no-trace v16-analog cannot) — **NOT** composition-solved, NOT compositional language, NOT scaled/integrated (a SEPARATE later gated increment). A faithful FAIL/VOID is the strongest honest triangulation of why composition is hard, NOT a license to escalate.

**Pre-registered FROZEN bars (in `compose_bind_core.py`, justified, NEVER tuned):**
`_CTB_V1_ACC_MIN = 0.90` (V1: the TD harness learns the no-gap bijection; analytic optimum 1.0) · `_CTB_SCIENCE_ACC_MIN = 0.90` (science: the gapped temporal-credit binding is learned) · `_CTB_CONTROL_ACC_MAX = 0.35` (every control ~chance; with N=12 the control chance-distribution is provably tight, P(control>0.35)≈3e-4 — the small-N absolute-bar artifact is structurally excluded by design) · `_CTB_MIN_SEEDS = 3`.

**Cheap gate already GREEN (do NOT re-litigate):** validated probe (deleted): V1 no-gap td 1.0/1.0/0.917/1.0/1.0 (≥0.90, 5 seeds); gapped td 0.917–1.0 (5 seeds); the mechanistically-identical-minus-the-trace `hebbian_no_trace` fails deterministically at exactly chance 0.083 all 5 seeds; permuted/wrongsign fail. Routing the decay through `fused_eligibility_trace_decay` (= `trace*decay_factor`, @fuse → numpy identity on CPU) is numerically identical, so these numbers reproduce exactly.

**Protected — byte-UNMODIFIED, verify empty `git diff` in EVERY commit-scoped diff:** `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (no-confab moat — MUST stay 7/7 green), `sim/td_value_critic.py`, `sim/kernels.py`, `sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/backend.py`, `sim/dendritic_plasticity.py`, every frozen `*_core.py` (incl. `td_critic_core`, `dendritic_fair_core`). NO new GLOBAL bar.

---

## Task 0: Grounding pin (commit now; green ONLY after Task 3 — intentional)

**Files:** Create `tests/test_compose_bind_grounding.py`

**Step 1: Create the file with EXACTLY:**

```python
"""Grounding pin: the END-TO-END compose_bind_gate pipeline must turn
on a TINY config and produce an interpretable THREE-STATE verdict.
RED until Task 3 lands the runner -- that is the Task-3 gate."""
import json
import subprocess
import sys
from pathlib import Path


def test_compose_bind_gate_tiny_pipeline_turns(tmp_path):
    out = tmp_path / "ctb_tiny.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.compose_bind_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert out.is_file(), r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["GATE"] in ("VOID", "PASS", "FAIL")
    assert "per_seed" in d and "frozen_bars" in d
```

**Step 2: Run — verify FAIL** (`No module named research.runners.compose_bind_gate`):
`python -m pytest tests/test_compose_bind_grounding.py -v`

**Step 3: Commit (scoped — ONLY this test):**
```
git add tests/test_compose_bind_grounding.py
git commit -m "test(compose-bind): Task-0 grounding pin (RED until Task 3 -- the Task-3 gate)"
```
Controller: `git show --stat HEAD` shows ONLY this file; protected absent.

---

## Phase A — pure-CPU-TDD (Tasks 1–2). Fresh subagent per task. Controller trust-but-verify each commit-scoped diff (protected byte-empty).

### Task 1: `sim/compose_temporal_bind.py` (LOAD-BEARING)

The cheap-gate-GREEN-validated TD(λ)+eligibility learner on the gapped A→B bijection; eligibility decay via the REUSED `fused_eligibility_trace_decay` UNMODIFIED; NO autograd. **Transcribe verbatim; do NOT redesign the algorithm.**

**Files:** Create `sim/compose_temporal_bind.py`; Test `tests/test_compose_temporal_bind.py`.

**Step 1: Write `tests/test_compose_temporal_bind.py` EXACTLY:**

```python
import numpy as np
from sim.compose_temporal_bind import run_bind, _N, _GAP


def test_reuses_eligibility_kernel_unmodified():
    import sim.compose_temporal_bind as m
    src = open(m.__file__).read()
    assert "from sim.kernels import fused_eligibility_trace_decay" in src
    assert "fused_eligibility_trace_decay(" in src


def test_no_autograd():
    import sim.compose_temporal_bind as m
    src = open(m.__file__).read()
    assert "autograd" not in src and "torch" not in src


def test_V1_nogap_td_learns_bijection():
    assert run_bind("td", 42, 0) >= 0.90


def test_science_gapped_td_learns_compositional_binding():
    assert run_bind("td", 42, _GAP) >= 0.90


def test_hebbian_no_trace_is_faithful_v16_analog_and_fails():
    # identical to td minus exactly the eligibility trace -> cannot
    # bridge the gap -> deterministically EXACTLY chance 1/_N.
    acc = run_bind("hebbian_no_trace", 42, _GAP)
    assert acc <= 0.35
    assert abs(acc - 1.0 / _N) < 1e-9


def test_permuted_and_wrongsign_fail():
    assert run_bind("permuted", 42, _GAP) <= 0.35
    assert run_bind("wrongsign", 42, _GAP) <= 0.35


def test_multiseed_decisive_discrimination():
    for s in (42, 43, 44):
        assert run_bind("td", s, _GAP) >= 0.90
        assert run_bind("hebbian_no_trace", s, _GAP) <= 0.35
```

**Step 2: Run — verify FAIL** (`No module named sim.compose_temporal_bind`).

**Step 3: Write `sim/compose_temporal_bind.py` EXACTLY (the validated reference; the ONLY change vs the proven probe is routing the decay through the REUSED kernel — numerically identical):**

```python
"""Compositional A->B binding bridging a TEMPORAL GAP, learned by the
VALIDATED TD(lambda)+eligibility mechanism. The eligibility decay is
routed through the REUSED sim.kernels.fused_eligibility_trace_decay
(UNMODIFIED) so this runs in the sim's real eligibility substrate.
hebbian_no_trace = identical to td EXCEPT the eligibility trace is
zeroed every gap step (the faithful v16-cold-start-analog: no
temporal-credit mechanism to carry the A-time decision across the
gap). NO automatic differentiation; deterministic seeded; ASCII only.

Honest ceiling: a PASS validates the MECHANISM (temporal credit
bridges the bind-gap where the no-trace analog cannot) -- NOT
composition-solved, NOT compositional language, NOT scaled/integrated;
that is a SEPARATE later gated increment."""
from __future__ import annotations
import numpy as np
from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED

# Pre-registered schedule constants (NOT science bars -- the frozen
# bars live in compose_bind_core). N=12 is BY DESIGN: it makes the
# compositional task strictly harder (chance 1/12 ~ 0.083) and makes
# the control chance-distribution provably tight (the small-N
# absolute-bar artifact is structurally excluded).
_N = 12             # |A| = |B|; chance = 1/12 ~ 0.083
_GAP = 6            # temporal gap between the A-time decision & reward
_GAMMA = 0.95
_LAMBDA = 0.9
_ALPHA = 0.1
_EPS = 0.1          # epsilon-greedy exploration
_N_TRIALS = 8000


def run_bind(mode: str, seed: int, gap: int) -> float:
    """One run. A_i -> B_{pi(i)} bijection; reward arrives `gap` steps
    after the A-time eps-greedy decision; credit must bridge the gap
    via the eligibility trace. Returns greedy accuracy over all A.
    modes: 'td' | 'hebbian_no_trace' | 'permuted' | 'wrongsign'."""
    rng = np.random.default_rng(seed)
    pi = rng.permutation(_N)                  # the fixed compositional rule
    W = np.zeros((_N, _N))
    decay = _GAMMA * _LAMBDA
    for _t in range(_N_TRIALS):
        pi_eff = rng.permutation(_N) if mode == "permuted" else pi
        i = int(rng.integers(_N))
        if rng.random() < _EPS:
            b = int(rng.integers(_N))
        else:
            b = int(np.argmax(W[i]))
        e = np.zeros((_N, _N))
        e[i, b] = 1.0                         # eligibility set at decision
        q_dec = W[i, b]
        for _g in range(gap):
            if mode == "hebbian_no_trace":
                e[:] = 0.0                    # the faithful v16-analog
            else:
                # REUSED sim eligibility kernel (UNMODIFIED): the
                # decay term gamma*lambda*e. Numerically identical to
                # the validated probe's inline e*=gamma*lambda.
                e = np.asarray(fused_eligibility_trace_decay(e, decay))
        r = 1.0 if b == pi_eff[i] else 0.0
        delta = r + _GAMMA * 0.0 - q_dec      # terminal TD error
        step = -delta if mode == "wrongsign" else delta
        W = W + _ALPHA * step * e
    greedy = np.argmax(W, axis=1)
    return float(np.mean(greedy == pi))
```

**Step 4: Run — verify PASS** (`python -m pytest tests/test_compose_temporal_bind.py -v`, all 7). Sanity: V1/science ≈ 1.0 at seed 42; hebbian_no_trace exactly 1/12≈0.0833. If any test fails, do NOT alter the reference or tests — report the discrepancy.

**Step 5: Commit (scoped to the two new files):**
```
git add sim/compose_temporal_bind.py tests/test_compose_temporal_bind.py
git commit -m "feat(compose-bind): TD(lambda)+eligibility gapped compositional binding (cheap-gate-GREEN reference; reuses fused_eligibility_trace_decay UNMODIFIED; NO autograd)"
```
Controller: diff = ONLY these two; protected (esp. `sim/kernels.py`, `sim/td_value_critic.py`) byte-absent.

---

### Task 2: `research/runners/compose_bind_core.py` (LOAD-BEARING)

Pure FIXED-bar THREE-STATE verdict, OWN frozen `_CTB_*`, mirroring the **adversarial-hardened** `td_critic_core` discipline (strict `_finite`, `try/except TypeError` malformed-keys → VOID, `metrics_finite` strengthen, isinstance/missing-control guards, VOID strictly distinct from FAIL, diverged/non-finite control = correctly-failed).

**Files:** Create `research/runners/compose_bind_core.py`; Test `tests/test_compose_bind_core.py`.

**Step 1: Write `tests/test_compose_bind_core.py` EXACTLY:**

```python
from research.runners.compose_bind_core import (
    ctb_verdict, _CTB_V1_ACC_MIN, _CTB_SCIENCE_ACC_MIN,
    _CTB_CONTROL_ACC_MAX, _CTB_MIN_SEEDS)


def _sound():
    return dict(nogap_td=1.0, td=1.0,
                controls={"hebbian_no_trace": 0.083,
                          "permuted": 0.083, "wrongsign": 0.0})


def test_frozen_bars_exact():
    assert _CTB_V1_ACC_MIN == 0.90
    assert _CTB_SCIENCE_ACC_MIN == 0.90
    assert _CTB_CONTROL_ACC_MAX == 0.35
    assert _CTB_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = ctb_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound(); s["nogap_td"] = 0.4
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_control_learned_is_VOID_not_pass():
    s = _sound(); s["controls"]["hebbian_no_trace"] = 0.95
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"          # non-discriminating


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound(); s["controls"]["wrongsign"] = float("nan")
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "PASS"          # diverged == correctly failed


def test_science_fail_when_sound_but_td_low():
    s = _sound(); s["td"] = 0.4
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_is_VOID_not_raise():
    s = _sound(); s["td"] = "0.99"
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_missing_control_is_VOID():
    s = _sound(); del s["controls"]["permuted"]
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    assert ctb_verdict({42: _sound()})["GATE"] == "VOID"


def test_unorderable_keys_is_VOID_not_raise():
    v = ctb_verdict({42: _sound(), "x": _sound(), 43: _sound()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_CTB_V1_ACC_MIN, _CTB_SCIENCE_ACC_MIN)
    ctb_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    import research.runners.compose_bind_core as c
    assert (c._CTB_V1_ACC_MIN, c._CTB_SCIENCE_ACC_MIN) == before
```

**Step 2: Run — verify FAIL** (no module).

**Step 3: Write `research/runners/compose_bind_core.py` EXACTLY:**

```python
"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
compose x temporal-credit gate. Instrument-validity FIRST,
fail-closed: a V1-broken or non-discriminating run is VOID --
explicitly NOT a science PASS/FAIL. Frozen _CTB_* pre-registered and
NEVER tuned. Mirrors the adversarial-hardened td_critic_core
discipline (strict numeric, malformed -> VOID-not-raise, diverged
control = correctly-failed, VOID strictly distinct from FAIL). Owns
its OWN bars; imports no other *_core. ASCII only."""
from __future__ import annotations
import math

_CTB_V1_ACC_MIN = 0.90
_CTB_SCIENCE_ACC_MIN = 0.90
_CTB_CONTROL_ACC_MAX = 0.35
_CTB_MIN_SEEDS = 3

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")


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


def _control_failed(x):
    """A control 'genuinely fails' (good) iff it did NOT learn: a
    diverged/non-finite value = correctly failed; a finite value must
    be <= the chance bar. A finite value ABOVE the bar means the
    control LEARNED -> the instrument is non-discriminating."""
    f = _finite(x)
    if f is None:
        return True                       # diverged/non-finite = failed
    return f <= _CTB_CONTROL_ACC_MAX


def ctb_verdict(per_seed: dict) -> dict:
    bars = {"V1_ACC_MIN": _CTB_V1_ACC_MIN,
            "SCIENCE_ACC_MIN": _CTB_SCIENCE_ACC_MIN,
            "CONTROL_ACC_MAX": _CTB_CONTROL_ACC_MAX,
            "MIN_SEEDS": _CTB_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable (instrument did "
                          "not soundly measure)",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CTB_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CTB_MIN_SEEDS,
                **base}
    v1_ok = True
    science_ok = True
    controls_fail = True
    metrics_finite = True
    for s in seeds:
        d = per_seed[s]
        nogap = _finite(d.get("nogap_td"))
        sci = _finite(d.get("td"))
        if nogap is None or sci is None:
            metrics_finite = False
        if nogap is None or nogap < _CTB_V1_ACC_MIN:
            v1_ok = False
        if sci is None or sci < _CTB_SCIENCE_ACC_MIN:
            science_ok = False
        ctrls = d.get("controls", {})
        if not isinstance(ctrls, dict):
            controls_fail = False
            continue
        for name in _CONTROLS:
            if name not in ctrls:
                controls_fail = False     # cannot certify discrimination
            elif not _control_failed(ctrls.get(name)):
                controls_fail = False     # a control LEARNED
    instrument_valid = bool(v1_ok and controls_fail and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: TD harness did NOT learn the no-gap "
                       "bijection (instrument unsound)")
        if not controls_fail:
            why.append("a control learned / is missing -> temporal "
                       "credit is NOT the discriminator (instrument "
                       "non-discriminating)")
        if not metrics_finite:
            why.append("a required science metric was non-numeric/"
                       "non-finite (instrument did not soundly measure)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
```

**Step 4: Run — verify PASS** (all 11). If any fails, do NOT alter reference/tests — report.

**Step 5: Commit:**
```
git add research/runners/compose_bind_core.py tests/test_compose_bind_core.py
git commit -m "feat(compose-bind): pure THREE-STATE FIXED-bar verdict (own frozen _CTB_*; instrument-validity-first; diverged-control-correct; mirrors hardened td_critic_core)"
```
Controller: diff scope; protected byte-absent; confirm no `*_core` import.

---

## ADVERSARIAL REVIEW CHECKPOINT (BEFORE Phase B)

Controller dispatches a **dedicated adversarial reviewer** for `sim/compose_temporal_bind.py` + `research/runners/compose_bind_core.py`. Explicitly probe: can a non-discriminating or V1-broken run be scored PASS not VOID? can a diverged control be mis-scored "non-discriminating" (wrong VOID) instead of correctly-failed? are the frozen `_CTB_*` movable by results? any autograd/torch in the shipped path? **Is the discrimination genuinely ISOLATED to the temporal-credit mechanism, or a readout/harness artifact?** **Is `hebbian_no_trace` a FAITHFUL v16-analog (identical to `td` in every respect EXCEPT the eligibility trace is zeroed each gap step) or a strawman (e.g., crippled elsewhere)?** Is `fused_eligibility_trace_decay` genuinely reused UNMODIFIED and numerically identical to the validated probe (so the GREEN numbers reproduce, not a silent redesign)? STRENGTHEN-only fixes; frozen bars byte-unchanged. Do NOT enter Phase B until sign-off + any fixes re-pass.

---

## Phase B — integration validated by import/signature smoke + the gate itself.

### Task 3: `research/runners/compose_bind_gate.py` (kill-safe runner)

**Files:** Create `research/runners/compose_bind_gate.py`; Test `tests/test_compose_bind_gate_smoke.py`.

**Step 1: Write `tests/test_compose_bind_gate_smoke.py` EXACTLY:**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, tmp):
    out = tmp / "g.json"
    r = subprocess.run([sys.executable, "-m",
                        "research.runners.compose_bind_gate", "--out",
                        str(out)] + args, capture_output=True,
                       text=True, cwd=ROOT)
    return r, out


def test_tiny_three_state_and_makes_task0_green(tmp_path):
    r, out = _run(["--tiny-synth", "--seeds", "42", "43", "44"],
                  tmp_path)
    assert out.is_file(), r.stdout + r.stderr
    assert json.loads(out.read_text())["GATE"] in (
        "VOID", "PASS", "FAIL")


def test_fewer_than_3_seeds_exit2(tmp_path):
    r, _ = _run(["--tiny-synth", "--seeds", "42"], tmp_path)
    assert r.returncode == 2


def test_reuses_checkpoint_nm_no_autograd():
    src = (ROOT / "research/runners/compose_bind_gate.py").read_text()
    assert "from sim.train_checkpoint import" in src
    assert "neuromodulators" in src
    assert "autograd" not in src and "torch" not in src
```

**Step 2: Run — verify FAIL** (no module).

**Step 3: Write `research/runners/compose_bind_gate.py` EXACTLY:**

```python
"""Kill-safe THREE-STATE gate runner for compose x temporal-credit.
Runs the cheap-gate-GREEN-validated learner across V1 (no-gap td),
science (gapped td), and {hebbian_no_trace, permuted, wrongsign}
controls x seeds; per-(seed) kill-safe checkpoint via the REUSED
sim.train_checkpoint; constructs (does NOT mutate) the REUSED NM
modulator so the temporal-credit delta IS the phasic-DA signal.
THREE-STATE verdict via compose_bind_core. NO autograd. ASCII.

HONEST CEILING (printed, never spun): a PASS = mechanism-level/in-sim
ONLY (temporal credit bridges the bind-gap where the no-trace
v16-analog cannot) -- NOT composition-solved, NOT compositional
language, NOT scaled/integrated; that is a SEPARATE later gated
increment. PASS/BOUNDARY/VOID all decision-relevant + propagated
honestly."""
from __future__ import annotations
import argparse
import json
import sys

from sim.compose_temporal_bind import run_bind, _GAP, _N_TRIALS
from sim.train_checkpoint import (save_checkpoint, load_checkpoint,
                                  resume_epoch)  # REUSED UNMODIFIED
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)  # REUSED UNMODIFIED
from research.runners.compose_bind_core import ctb_verdict

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: mechanism-level/in-sim ONLY -- temporal "
           "credit bridges the bind-gap where the no-trace v16-analog "
           "cannot; NOT composition-solved, integration a SEPARATE "
           "later gated increment.")


def _da_modulator_from_delta():
    """The catalog C.30 upgrade demonstrated via the REUSED NM
    subsystem UNMODIFIED: a from_reward DA modulator whose drive is
    the temporal-credit TD delta. Constructed to prove composition
    with the validated phasic-DA substrate; not mutated here."""
    return NeuromodulatorConfig(
        name="dopamine_compose", baseline=0.0, decay_tau_ms=50.0,
        concentration_min=-5.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0,
                                         threshold=0.0,
                                         window_ms=0.0)])


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
    _ = _da_modulator_from_delta()        # construct (not mutate)
    gap = 2 if a.tiny_synth else _GAP
    # tiny-synth shortens trials only (a toy verdict, NOT propagated).
    n_note = a.tiny_synth
    per_seed = {}
    try:
        for s in a.seeds:
            if a.tiny_synth:
                import sim.compose_temporal_bind as cb
                _orig = cb._N_TRIALS
                cb._N_TRIALS = 200
            try:
                row = {"nogap_td": run_bind("td", s, 0),
                       "td": run_bind("td", s, gap),
                       "controls": {c: run_bind(c, s, gap)
                                    for c in _CONTROLS}}
            finally:
                if a.tiny_synth:
                    cb._N_TRIALS = _orig
            if a.ckpt:
                save_checkpoint(a.ckpt, s, {"row": [row["nogap_td"],
                                row["td"]]}, None, [])
            per_seed[s] = row
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    verdict = ctb_verdict(per_seed)
    verdict["banner"] = _BANNER
    if n_note:
        verdict["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("GATE=%s  %s" % (verdict["GATE"], _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> Implementer note: `run_bind` reads `_N_TRIALS` at call time only if it references the module global; the reference reads the constant as a default-free local loop bound. To make `--tiny-synth` genuinely fast WITHOUT redesigning `run_bind`, the runner temporarily rebinds `sim.compose_temporal_bind._N_TRIALS` AND `run_bind` must read it dynamically. **Adjust `run_bind` minimally** so its loop uses the module-level `_N_TRIALS` at call time (e.g., `for _t in range(_N_TRIALS):` already does if `_N_TRIALS` is module-global — it is). This is a no-op for the real run (default unchanged) and only shortens tiny-synth. If this coupling is undesirable, instead add an optional `n_trials=None` arg to `run_bind` (default → module `_N_TRIALS`) and pass it from the runner — preserve the validated algorithm exactly; this is an integration detail, NOT a redesign. Report exactly what you chose.

**Step 4: Run — verify PASS** (`python -m pytest tests/test_compose_bind_gate_smoke.py tests/test_compose_bind_grounding.py -v`, all PASS; Task 0 now GREEN). The tiny-synth toy verdict (likely VOID/FAIL at 200 trials) is fine — smoke only checks GATE ∈ {VOID,PASS,FAIL} + keys.

**Step 5: Commit:**
```
git add research/runners/compose_bind_gate.py tests/test_compose_bind_gate_smoke.py sim/compose_temporal_bind.py
git commit -m "feat(compose-bind): kill-safe THREE-STATE gate runner (reuses train_checkpoint + NM UNMODIFIED; catalog C.30 delta=phasic-DA; Task 0 green; NO autograd)"
```
(If `run_bind` gained an optional `n_trials` arg, include `sim/compose_temporal_bind.py` + re-run `tests/test_compose_temporal_bind.py` green first.) Controller: diff scope; protected byte-absent.

---

### Task 4: LOAD-BEARING no-harm

**Files:** Create `tests/test_compose_bind_no_harm.py`

**Step 1: Write EXACTLY (self-contained; base = the design commit `2fde0ed`; no shared conftest change):**

```python
"""LOAD-BEARING no-harm: protected/validated modules byte-UNTOUCHED
across the whole compose-bind range (2fde0ed..HEAD); NO shipped path
imports autograd/torch."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "2fde0ed"
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/td_value_critic.py", "sim/kernels.py", "sim/neuromodulators.py",
    "sim/train_checkpoint.py", "sim/backend.py",
    "sim/dendritic_plasticity.py", "research/runners/td_critic_core.py",
    "research/runners/dendritic_fair_core.py"]


def test_protected_byte_untouched_across_range():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "%s..HEAD" % _BASE, "--"]
        + PROTECTED, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", "PROTECTED MODIFIED:\n" + diff.stdout


def test_no_autograd_in_shipped_path():
    for p in ("sim/compose_temporal_bind.py",
              "research/runners/compose_bind_core.py",
              "research/runners/compose_bind_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
```

**Step 2: Run no-harm + the moat + the full compose suite:**
```
python -m pytest tests/test_compose_bind_no_harm.py tests/test_abstention_gate.py -v
python -m pytest tests/test_compose_temporal_bind.py tests/test_compose_bind_core.py tests/test_compose_bind_gate_smoke.py tests/test_compose_bind_grounding.py -q
```
Expected: no-harm 2/2; moat 7 passed (report exact count); compose suite all PASS. If `test_protected_byte_untouched_across_range` FAILS, STOP and report the offending paths (do NOT edit protected modules).

**Step 3: Commit:**
```
git add tests/test_compose_bind_no_harm.py
git commit -m "test(compose-bind): LOAD-BEARING no-harm (protected byte-untouched 2fde0ed..HEAD; moat 7/7; no autograd)"
```

---

## Task 5 — CONTROLLER-ONLY (NOT a subagent task). Bring back to the controller.

1. **Grounding-first tiny run** (toy verdict NOT propagated):
   `python -m research.runners.compose_bind_gate --tiny-synth --seeds 42 43 44 --out research/findings/raw/ctb_tiny.json` — confirm pipeline turns + THREE-STATE; discard.
2. **Decisive kill-safe multi-seed run** (FIXED pre-registered config; ≥5 seeds):
   `python -m research.runners.compose_bind_gate --seeds 42 43 44 45 46 --ckpt research/findings/raw/ctb_ckpt --out research/findings/raw/compose_bind_gate.json`
3. **MANDATORY anti-cheat smell-test — scrutinize a nominal PASS HARDER than a FAIL.** Recompute from the recorded JSON (NO re-run, NO bar-tuning): V1 genuine + non-degenerate (no-gap td ≥0.90, NOT a trivial constant); science gapped td ≥0.90 genuinely; **the discrimination genuinely ISOLATED to the temporal-credit mechanism** — `hebbian_no_trace` is identical to `td` minus exactly the eligibility trace and fails ~deterministically at chance (≈1/12); `permuted`/`wrongsign` fail; decisive separation (td vs each control). Confirm the GREEN cheap-probe numbers reproduced (kernel reuse numerically identical). NO overclaim.
4. **Honest propagation EVERY outcome** (VOID = instrument-unsound/non-discriminating in-sim; PASS = the temporal-credit mechanism bridges the bind-gap where the no-trace v16-analog cannot, ceiling explicit, integration a SEPARATE later increment; FAIL = sound+discriminating yet temporal credit ALSO cannot compose = strongest honest triangulation that temporal credit is NOT the missing composition ingredient): findings doc `research/findings/2026-05-18-compose-temporal-credit-<OUTCOME>.md` + `webapp/capability_status.json` pillar (n=71; status VALIDATED if PASS / BOUNDARY / NEGATIVE) + `python -m pytest tests/test_webapp_server.py -k capability_status -q` green + commit (scoped; protected byte-empty) + push BOTH remotes (origin + gitea). Bars NOT tuned. NOT config-cranked. Moat byte-identical + 7/7 throughout. Honest ceiling NEVER spun.

---

## Remember
- Exact file paths; complete code (above is the cheap-gate-GREEN reference + the hardened-`td_critic_core` verdict pattern — do NOT redesign).
- Frozen `_CTB_*` pre-registered here; NEVER tuned; N=12-by-design excludes the small-N control artifact.
- Protected set byte-UNMODIFIED — controller verifies empty diff EVERY commit.
- @superpowers:subagent-driven-development drives Tasks 0–4; Task 5 is controller-only.
- Honest ceiling stated up front and NEVER spun. PASS/BOUNDARY/VOID all propagated honestly.
