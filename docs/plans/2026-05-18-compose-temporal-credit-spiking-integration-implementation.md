---
type: plan
status: live
date: 2026-05-18
---

# Temporal-Credit Composition — Spiking-Bridge Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (controller stays in this session; fresh subagent per task; dedicated adversarial reviewer for the two load-bearing modules BEFORE Phase B; controller trust-but-verify each git diff with the protected set byte-empty; Task 5 is controller-only).

**Goal:** Test whether the thrice-validated temporal-credit/eligibility mechanism still bridges the verb→motor compositional bind-gap when wired into a MINIMAL slice of the real spiking `sim.bridge` concept-pool architecture (the actual v16 BOUNDARY setting).

**Architecture:** Net-new pure FIXED-bar THREE-STATE verdict (`research/runners/compose_bridge_core.py`, own frozen `_CBR_*`, an EXACT transcription of the adversarial-hardened `compose_bind_core` discipline). Net-new kill-safe runner (`research/runners/compose_bridge_gate.py`) that builds a minimal verb+motor concept-pool bridge via the REUSED `build_biological_brain_regions`, drives a verb→(gap)→motor+reward episode, and applies the temporal-credit credit rule through the bridge's REAL native eligibility-trace reward-modulation path (`cp_eligibility_trace` + `current_reward_signal` + `reward_learning_rate`); the `hebbian_no_trace` control is identical except the eligibility trace is suppressed across the gap (the faithful v16-cold-start analog). The in-bridge wiring is genuine net-new integration validated by the gate itself + the dedicated adversarial review + the controller smell-test (project pattern — NOT contrived orchestration unit tests, NOT a fabricated "proven reference").

**Tech Stack:** Python, the project `sim.bridge` (CuPy/NumPy via `sim.backend`), REUSED `build_biological_brain_regions`, `sim.kernels.fused_eligibility_trace_decay`, `sim.train_checkpoint`, `sim.neuromodulators`, pytest. NO `torch`, NO automatic differentiation anywhere in the shipped path.

**Honest framing (read this):** Unlike the prior two builds, the in-bridge mechanism is NOT a transcribe-a-proven-numpy-reference job. The cheap pop-transfer probe is the falsify-first PRECURSOR (already GREEN, scrutinized harder than a FAIL: the mechanism transfers to distributed-population + noisy-readout). The in-sim THREE-STATE gate decides the science HONESTLY, every outcome. A PASS = the mechanism transfers into a minimal slice of the real spiking architecture (the first in-architecture mechanistic dent in the composition blocker) — explicitly NOT composition-solved, NOT compositional language, NOT scaled/chat-integrated (a further SEPARATE gated increment). A faithful FAIL/VOID sharply triangulates that the remaining blocker is spiking-dynamics integration, not the thrice-validated temporal-credit principle — NOT a license to escalate.

**Pre-registered FROZEN bars (in `compose_bridge_core.py`; justified BELOW; calibrated to the spiking substrate's irreducible noise floor BEFORE any run; NEVER tuned to a result):**
- `_CBR_V1_ACC_MIN = 0.80` — V1 instrument soundness: in-bridge true-TD on a NO-GAP verb→motor binding must reach ≥0.80. Justification: a spiking concept-pool readout (population vote over noisy LIF rates) has an irreducible misclassification floor higher than the pure-numpy substrates (which hit 1.0); 0.80 (vs the numpy 0.90) is the pre-registered, honest allowance for spiking-readout noise, set from the v16-era concept-pool literature in CLAUDE.md where sound bindings read out at ~0.8–1.0. **A sound true-TD no-gap learner that cannot reach 0.80 is an honest VOID (instrument not soundly constructible at this cheap config), NOT a reason to soften this further.**
- `_CBR_SCI_ACC_MIN = 0.80` — science: in-bridge TD+eligibility on the GAPPED verb→motor bind must reach ≥0.80 (same readout floor as V1; the science is whether the gap is bridged, measured on the same scale as V1).
- `_CBR_CTRL_ACC_MAX = 0.35` — every control (`hebbian_no_trace`, `permuted`, `wrongsign`) must stay ≤0.35. Justification: with ≥8 distinct verb/motor bindings chance ≤0.125; 0.35 (~3× chance) is a generous, unambiguous "did NOT learn" ceiling; identical absolute-bar discipline to the validated sibling, valid because the design mandates ≥8 bindings (the small-N artifact is structurally excluded).
- `_CBR_MIN_SEEDS = 3`.
THREE-STATE, instrument-validity-FIRST, fail-closed: VOID if V1 unmet OR any control learns/is missing OR a metric is non-numeric/malformed; only if sound+discriminating: PASS iff science met, else FAIL. VOID strictly distinct from FAIL.

**Protected — byte-UNMODIFIED, verify empty `git diff e8a99a2..HEAD` in EVERY commit-scoped diff:** `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (no-confab moat — MUST stay 7/7 green), `sim/td_value_critic.py`, `sim/compose_temporal_bind.py`, `sim/kernels.py`, `sim/bridge.py`, `sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/backend.py`, `sim/dendritic_plasticity.py`, `research/runners/text_minimal_isolation.py` (`build_biological_brain_regions` REUSED UNMODIFIED), every frozen `*_core.py` (incl. `compose_bind_core`, `td_critic_core`, `dendritic_fair_core`). NO new GLOBAL bar. NO autograd/torch in the shipped path.

---

## Task 0: Grounding pin (commit now; green ONLY after Task 2 — intentional)

**Files:** Create `tests/test_compose_bridge_grounding.py`

**Step 1: Create with EXACTLY:**

```python
"""Grounding pin: the END-TO-END compose_bridge_gate pipeline must
turn on a TINY config and produce an interpretable THREE-STATE
verdict. RED until the runner task lands -- that is its gate."""
import json
import subprocess
import sys
from pathlib import Path


def test_compose_bridge_gate_tiny_pipeline_turns(tmp_path):
    out = tmp_path / "cbr_tiny.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.compose_bridge_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert out.is_file(), r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["GATE"] in ("VOID", "PASS", "FAIL")
    assert "per_seed" in d and "frozen_bars" in d
```

**Step 2: Run — verify FAIL** (`No module named research.runners.compose_bridge_gate`):
`python -m pytest tests/test_compose_bridge_grounding.py -v`

**Step 3: Commit (scoped — ONLY this test):**
```
git add tests/test_compose_bridge_grounding.py
git commit -m "test(compose-bridge): Task-0 grounding pin (RED until the gate runner -- its gate)"
```
Controller: `git show --stat HEAD` shows ONLY this file; protected absent.

---

## Phase A — Tasks 1–2. Fresh subagent per task. Controller trust-but-verify each commit-scoped diff (protected byte-empty).

### Task 1: `research/runners/compose_bridge_core.py` (LOAD-BEARING) — fully specifiable, full code

EXACT transcription of the adversarial-hardened `compose_bind_core` discipline with its OWN frozen `_CBR_*`. Per-seed payload shape: `{nogap_td: float, td: float, controls: {hebbian_no_trace: float, permuted: float, wrongsign: float}}`.

**Files:** Create `research/runners/compose_bridge_core.py`; Test `tests/test_compose_bridge_core.py`.

**Step 1: Write `tests/test_compose_bridge_core.py` EXACTLY:**

```python
from research.runners.compose_bridge_core import (
    cbr_verdict, _CBR_V1_ACC_MIN, _CBR_SCI_ACC_MIN,
    _CBR_CTRL_ACC_MAX, _CBR_MIN_SEEDS)


def _sound():
    return dict(nogap_td=0.95, td=0.92,
                controls={"hebbian_no_trace": 0.10,
                          "permuted": 0.10, "wrongsign": 0.0})


def test_frozen_bars_exact():
    assert _CBR_V1_ACC_MIN == 0.80
    assert _CBR_SCI_ACC_MIN == 0.80
    assert _CBR_CTRL_ACC_MAX == 0.35
    assert _CBR_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = cbr_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound(); s["nogap_td"] = 0.5
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_control_learned_is_VOID_not_pass():
    s = _sound(); s["controls"]["hebbian_no_trace"] = 0.85
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound(); s["controls"]["wrongsign"] = float("nan")
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "PASS"


def test_non_numeric_junk_control_is_VOID_not_fabricated_pass():
    s = _sound(); s["controls"]["permuted"] = "0.10"
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_science_fail_when_sound_but_td_low():
    s = _sound(); s["td"] = 0.5
    v = cbr_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_science_is_VOID_not_raise():
    s = _sound(); s["td"] = "0.99"
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_missing_control_is_VOID():
    s = _sound(); del s["controls"]["permuted"]
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    assert cbr_verdict({42: _sound()})["GATE"] == "VOID"


def test_unorderable_keys_is_VOID_not_raise():
    v = cbr_verdict({42: _sound(), "x": _sound(), 43: _sound()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_CBR_V1_ACC_MIN, _CBR_SCI_ACC_MIN)
    cbr_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    import research.runners.compose_bridge_core as c
    assert (c._CBR_V1_ACC_MIN, c._CBR_SCI_ACC_MIN) == before
```

**Step 2: Run — verify FAIL** (no module).

**Step 3: Write `research/runners/compose_bridge_core.py` EXACTLY:**

```python
"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
compose temporal-credit SPIKING-BRIDGE integration gate. Instrument-
validity FIRST, fail-closed: a V1-broken or non-discriminating in-sim
run is VOID -- explicitly NOT a science PASS/FAIL. Frozen _CBR_*
pre-registered and NEVER tuned. EXACT mirror of the adversarial-
hardened compose_bind_core discipline (strict numeric, malformed/junk
-> VOID-not-raise, diverged numeric control = correctly-failed, VOID
strictly distinct from FAIL). Owns its OWN bars; imports no other
*_core. ASCII only."""
from __future__ import annotations
import math

_CBR_V1_ACC_MIN = 0.80
_CBR_SCI_ACC_MIN = 0.80
_CBR_CTRL_ACC_MAX = 0.35
_CBR_MIN_SEEDS = 3

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _control_failed(x):
    """Genuinely diverged NUMERIC (nan/inf) = correctly failed (good).
    Non-numeric junk (str/bool/None) is NOT a certified failure (a
    learned control serialized as "0.9"/True must force VOID, not pass
    as good). A finite value above the bar means the control LEARNED
    -> non-discriminating."""
    if x is None:
        return False
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    f = _finite(x)
    if f is None:
        return True
    return f <= _CBR_CTRL_ACC_MAX


def cbr_verdict(per_seed: dict) -> dict:
    bars = {"V1_ACC_MIN": _CBR_V1_ACC_MIN,
            "SCI_ACC_MIN": _CBR_SCI_ACC_MIN,
            "CTRL_ACC_MAX": _CBR_CTRL_ACC_MAX,
            "MIN_SEEDS": _CBR_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable (instrument did "
                          "not soundly measure)",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CBR_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CBR_MIN_SEEDS,
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
        if nogap is None or nogap < _CBR_V1_ACC_MIN:
            v1_ok = False
        if sci is None or sci < _CBR_SCI_ACC_MIN:
            science_ok = False
        ctrls = d.get("controls", {})
        if not isinstance(ctrls, dict):
            controls_fail = False
            continue
        for name in _CONTROLS:
            if name not in ctrls:
                controls_fail = False
            elif not _control_failed(ctrls.get(name)):
                controls_fail = False
    instrument_valid = bool(v1_ok and controls_fail and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: in-bridge TD did NOT learn the "
                       "no-gap verb->motor bind (instrument unsound)")
        if not controls_fail:
            why.append("a control learned / is missing -> temporal "
                       "credit is NOT the in-bridge discriminator "
                       "(non-discriminating)")
        if not metrics_finite:
            why.append("a required metric was non-numeric/non-finite "
                       "(instrument did not soundly measure)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
```

**Step 4: Run — verify PASS** (all 12). If any fails, do NOT alter reference/tests — report.

**Step 5: Commit:**
```
git add research/runners/compose_bridge_core.py tests/test_compose_bridge_core.py
git commit -m "feat(compose-bridge): pure THREE-STATE FIXED-bar verdict (own frozen _CBR_*; instrument-validity-first; junk->VOID; diverged-numeric-control-correct; mirrors hardened compose_bind_core)"
```
Controller: diff scope; protected byte-absent; confirm imports only `math`+`__future__` (no `*_core`).

---

### Task 2: `research/runners/compose_bridge_gate.py` (LOAD-BEARING) — genuine integration; scaffold fully specified, in-bridge mechanism interface-pinned + behaviorally-precise

**This is real integration, not transcription.** The plan pins the scaffold (CLI, kill-safe, verdict wiring — mirror the proven `compose_bind_gate.py` structure exactly) and the BEHAVIORAL SPEC + REUSED interfaces for the in-bridge mechanism. Correctness is established by Task-0 + the smoke + the dedicated adversarial review + the controller decisive run + smell-test (project pattern).

**Files:** Create `research/runners/compose_bridge_gate.py`; Test `tests/test_compose_bridge_gate_smoke.py`.

**REUSED interfaces (byte-UNMODIFIED — import, do not edit):**
- `from research.runners.text_minimal_isolation import build_biological_brain_regions` — builds the minimal verb/motor concept-pool regions+pathways (the v16 setting). Use a MINIMAL config (the design's "minimal slice": the existing default motor pools `motor_{N,E,S,W}` + the verb pools the builder exposes; do NOT scale vocab — YAGNI).
- `from sim.kernels import fused_eligibility_trace_decay` — the eligibility decay primitive (same one the validated mechanism uses).
- `from sim.train_checkpoint import save_checkpoint, load_checkpoint, resume_epoch` — kill-safe per-(seed) atomic checkpoint.
- `from sim.neuromodulators import NeuromodulatorConfig, ProductionRule, ModulatorTarget` — construct (do NOT mutate) a `from_reward`→`plasticity_rate` DA modulator so the TD δ IS the phasic-DA signal (catalog C.30), exactly as `compose_bind_gate._da_modulator_from_delta()` did.
- `from research.runners.compose_bridge_core import cbr_verdict`.
- The bridge's REAL eligibility/reward path: `bridge.cp_eligibility_trace`, `bridge.core_config.current_reward_signal`, `bridge.core_config.reward_learning_rate`, `bridge.set_token_drive(...)`, `bridge.read_language_output(...)` (or the motor-pool rate readout the v16 runners use), and stepping the bridge. (These are READ/CALLED, never modified.)

**BEHAVIORAL SPEC — the in-bridge episode + the four conditions (the implementer writes this against the reused interfaces; the dedicated adversarial reviewer verifies faithfulness):**
- An *episode* binds verb V_i → motor M_{pi(i)} for a fixed bijection `pi` over ≥8 distinct (verb,motor) bindings (≥8 so the `_CBR_CTRL_ACC_MAX=0.35` absolute bar's chance floor ≤0.125 is unambiguous; the small-N artifact is structurally excluded).
- t_A: drive verb pool V_i. Then a TEMPORAL GAP of `G` spiking steps (no decision-relevant drive). t_R: read the motor-pool population vote → selected motor; reward r=1 iff selected == M_{pi(i)} else 0. The reward is delivered a gap after the verb drive — so a static readout cannot shortcut it (the readout-confound is structurally defeated, exactly as in the validated probes).
- **`td`**: the temporal-credit credit rule — the bridge's NATIVE eligibility-trace reward modulation (`cp_eligibility_trace` accumulated at the verb→motor synapses at t_A, decayed across the gap via `fused_eligibility_trace_decay` semantics the bridge already applies, then `current_reward_signal`=TD δ applied at t_R with `reward_learning_rate`). δ = r − V(s) using the validated TD(λ) form (γ=0.95, λ=0.9, α as the bridge `reward_learning_rate`) from the validated mechanism — REUSE the rule, do not reinvent it.
- **`hebbian_no_trace`**: IDENTICAL to `td` in EVERY respect (same drive, same gap, same readout, same reward, same RNG consumption) EXCEPT the eligibility trace is suppressed across the gap (zeroed each gap step) — the faithful v16-cold-start analog (no temporal-credit bridging). It MUST NOT be crippled anywhere else. The adversarial reviewer will verify byte-level that this is the *only* difference.
- **`permuted`**: `pi` re-randomized every episode (no stable verb→motor structure). Must not learn.
- **`wrongsign`**: δ sign-flipped. Must anti-learn.
- V1 = `td` with gap G=0 (no-gap; proves the spiking harness itself learns the verb→motor bijection — instrument soundness, separate from the science).
- Greedy eval (noise-free policy readout) accuracy over all bindings; per-seed `nogap_td`(=V1, gap0 td), `td`(gapped), and the three controls (gapped).

**Scaffold (mirror `compose_bind_gate.py` EXACTLY for CLI/kill-safe/verdict; this part IS specifiable verbatim):**

```python
"""Kill-safe THREE-STATE gate: does the validated temporal-credit/
eligibility mechanism bridge the verb->motor compositional bind-gap
inside a MINIMAL slice of the real spiking sim.bridge concept-pool
architecture (the v16 setting)? Reuses build_biological_brain_regions
+ the bridge's native cp_eligibility_trace reward-modulation path +
the NM subsystem (TD delta = phasic-DA, catalog C.30) +
sim.train_checkpoint, ALL byte-UNMODIFIED. hebbian_no_trace = the
faithful v16-cold-start analog (identical to td minus EXACTLY the
eligibility-trace bridging). NO automatic differentiation. ASCII.

HONEST CEILING (printed, never spun): a PASS = the mechanism
transfers into a MINIMAL slice of the real spiking architecture (the
first in-architecture mechanistic dent in the composition blocker) --
NOT composition-solved, NOT compositional language, NOT scaled/chat-
integrated (a further SEPARATE gated increment). PASS/BOUNDARY/VOID
all decision-relevant + propagated honestly."""
from __future__ import annotations
import argparse
import json
import sys

# REUSED byte-UNMODIFIED:
from research.runners.text_minimal_isolation import (
    build_biological_brain_regions)
from sim.kernels import fused_eligibility_trace_decay
from sim.train_checkpoint import (save_checkpoint, load_checkpoint,
                                  resume_epoch)
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)
from research.runners.compose_bridge_core import cbr_verdict

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: minimal-spiking-slice mechanism-transfer "
           "ONLY -- NOT composition-solved, NOT compositional "
           "language, NOT scaled/chat-integrated (a further SEPARATE "
           "gated increment).")


def _da_modulator_from_delta():
    """Catalog C.30 upgrade via the REUSED NM subsystem UNMODIFIED:
    from_reward DA modulator whose drive is the TD delta. Constructed
    to prove composition with the validated phasic-DA substrate; not
    mutated."""
    return NeuromodulatorConfig(
        name="dopamine_compose_bridge", baseline=0.0,
        decay_tau_ms=50.0, concentration_min=-5.0,
        concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0,
                                         threshold=0.0,
                                         window_ms=0.0)])


def _run_seed(mode, seed, tiny):
    """IMPLEMENTER: build the minimal verb/motor concept-pool bridge
    via build_biological_brain_regions (REUSED UNMODIFIED), run the
    verb->(gap)->motor+reward episodes per the BEHAVIORAL SPEC for
    `mode`, return greedy accuracy (float). `tiny` shrinks
    pools/episodes/steps for the smoke path only (a toy verdict, NOT
    propagated). gap=0 for the nogap V1 caller. Reuse the validated
    TD(lambda)+eligibility credit rule (gamma=0.95, lambda=0.9) via
    the bridge's native cp_eligibility_trace reward path +
    fused_eligibility_trace_decay; hebbian_no_trace suppresses the
    trace across the gap and is otherwise byte-identical to td. NO
    autograd."""
    raise NotImplementedError  # implemented per the BEHAVIORAL SPEC


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
    per_seed = {}
    try:
        for s in a.seeds:
            row = {"nogap_td": _run_seed_nogap(s, a.tiny_synth),
                   "td": _run_seed("td", s, a.tiny_synth),
                   "controls": {c: _run_seed(c, s, a.tiny_synth)
                                for c in _CONTROLS}}
            if a.ckpt:
                save_checkpoint(a.ckpt, s,
                                {"row": [row["nogap_td"], row["td"]]},
                                None, [])
            per_seed[s] = row
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    verdict = cbr_verdict(per_seed)
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
(Implementer adds `_run_seed_nogap(seed, tiny)` = `_run_seed("td", seed, tiny)` with gap forced to 0, plus the `_run_seed` body per the BEHAVIORAL SPEC. Keep the gap, the ≥8 bindings, the faithful-analog constraint, and "NO autograd" exactly. ASCII only.)

**Step 1: Write `tests/test_compose_bridge_gate_smoke.py` EXACTLY:**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, tmp):
    out = tmp / "g.json"
    r = subprocess.run([sys.executable, "-m",
                        "research.runners.compose_bridge_gate",
                        "--out", str(out)] + args,
                       capture_output=True, text=True, cwd=ROOT)
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


def test_reuses_build_checkpoint_nm_no_autograd():
    src = (ROOT / "research/runners/compose_bridge_gate.py").read_text()
    assert "build_biological_brain_regions" in src
    assert "from sim.train_checkpoint import" in src
    assert "neuromodulators" in src
    assert "autograd" not in src and "torch" not in src
```

**Step 2: Run — verify FAIL** (no module).

**Step 3: Implement** the scaffold above + `_run_seed`/`_run_seed_nogap` per the BEHAVIORAL SPEC against the REUSED interfaces. Sigmoid/spiking dynamics come from the bridge; the temporal-credit rule reuses γ=0.95/λ=0.9 + the eligibility kernel + the bridge's native reward path. `--tiny-synth` shrinks pools/episodes so the smoke is fast (toy verdict, likely VOID at tiny scale — fine; smoke only checks GATE membership + keys).

**Step 4: Run — verify GREEN:** `python -m pytest tests/test_compose_bridge_gate_smoke.py tests/test_compose_bridge_grounding.py -v` → all PASS (Task 0 now GREEN).

**Step 5: Commit:**
```
git add research/runners/compose_bridge_gate.py tests/test_compose_bridge_gate_smoke.py
git commit -m "feat(compose-bridge): kill-safe in-bridge THREE-STATE gate runner (reuses build_biological_brain_regions + cp_eligibility_trace reward path + NM + train_checkpoint UNMODIFIED; faithful v16-analog control; Task 0 green; NO autograd)"
```
Controller: diff scope; protected (esp. `text_minimal_isolation.py`, `sim/bridge.py`, `sim/neuromodulators.py`) byte-absent.

---

## ADVERSARIAL REVIEW CHECKPOINT (BEFORE Phase B) — load-bearing pair

Controller dispatches a **dedicated adversarial reviewer** for `compose_bridge_core.py` + `compose_bridge_gate.py`. Read the ACTUAL code; run snippets. Explicitly probe: can a non-discriminating / V1-broken in-bridge run be scored PASS instead of VOID? can a diverged numeric control be mis-scored non-discriminating; is non-numeric junk fail-closed to VOID? are frozen `_CBR_*` movable by results? any autograd/torch in the shipped path? **Is the in-bridge discrimination genuinely ISOLATED to the temporal-credit mechanism, or a spiking-harness artifact (RNG-consumption parity / readout / drive identical between `td` and `hebbian_no_trace` except the trace bridging)?** **Is `hebbian_no_trace` a FAITHFUL in-bridge v16-cold-start analog or a strawman crippled elsewhere?** Are `build_biological_brain_regions` + `cp_eligibility_trace` + the validated TD/eligibility rule genuinely reused byte-UNMODIFIED (no copy-paste-and-tweak of protected code)? STRENGTHEN-only fixes; frozen bars byte-unchanged. Do NOT enter Phase B until sign-off + any fixes re-pass.

---

## Phase B

### Task 3: LOAD-BEARING no-harm

**Files:** Create `tests/test_compose_bridge_no_harm.py` (self-contained; base `e8a99a2`; no shared-conftest change):

```python
"""LOAD-BEARING no-harm: protected/validated modules byte-UNTOUCHED
across the whole compose-bridge range (e8a99a2..HEAD); NO shipped path
imports autograd/torch."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "e8a99a2"
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/td_value_critic.py", "sim/compose_temporal_bind.py",
    "sim/kernels.py", "sim/bridge.py", "sim/neuromodulators.py",
    "sim/train_checkpoint.py", "sim/backend.py",
    "sim/dendritic_plasticity.py",
    "research/runners/text_minimal_isolation.py",
    "research/runners/compose_bind_core.py",
    "research/runners/td_critic_core.py",
    "research/runners/dendritic_fair_core.py"]


def test_protected_byte_untouched_across_range():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "%s..HEAD" % _BASE, "--"]
        + PROTECTED, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", "PROTECTED MODIFIED:\n" + diff.stdout


def test_no_autograd_in_shipped_path():
    for p in ("research/runners/compose_bridge_core.py",
              "research/runners/compose_bridge_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
```

**Step 2:** `python -m pytest tests/test_compose_bridge_no_harm.py tests/test_abstention_gate.py -v` (no-harm 2/2; moat 7/7 — report exact count) + full compose-bridge suite green. If protected-untouched FAILS, STOP and report offending paths.

**Step 3: Commit:**
```
git add tests/test_compose_bridge_no_harm.py
git commit -m "test(compose-bridge): LOAD-BEARING no-harm (protected byte-untouched e8a99a2..HEAD; moat 7/7; no autograd)"
```

---

## Task 5 — CONTROLLER-ONLY (NOT a subagent task). Bring back to the controller.

1. **Grounding-first tiny run** (toy verdict NOT propagated):
   `python -m research.runners.compose_bridge_gate --tiny-synth --seeds 42 43 44 --out research/findings/raw/cbr_tiny.json` — confirm pipeline turns + THREE-STATE; discard.
2. **Decisive kill-safe multi-seed in-sim run** (FIXED pre-registered config; ≥5 seeds; KILL-SAFE is HARD — interruptible/resumable):
   `python -m research.runners.compose_bridge_gate --seeds 42 43 44 45 46 --ckpt research/findings/raw/cbr_ckpt --out research/findings/raw/compose_bridge_gate.json`
3. **MANDATORY anti-cheat smell-test — scrutinize a nominal PASS HARDER than a FAIL.** Recompute from the recorded JSON (NO re-run, NO bar-tuning): V1 genuine + non-degenerate (no-gap td ≥0.80, NOT a trivial constant — confirm the degenerate floor is ≪0.80, e.g. the chance/random-readout level, and that `hebbian_no_trace` sits there); the in-bridge discrimination genuinely ISOLATED to the temporal-credit mechanism (`hebbian_no_trace` identical to `td` minus exactly the trace bridging, ~chance; `permuted`/`wrongsign` fail); decisive separation; confirm the cheap pop-transfer GREEN precursor is consistent with the in-bridge result. NO overclaim.
4. **Honest propagation EVERY outcome** (VOID = in-bridge instrument unsound/non-discriminating; PASS = the mechanism transfers into the minimal real spiking slice, ceiling explicit; FAIL = sound+discriminating yet temporal credit ALSO fails in-bridge = the remaining blocker is spiking-dynamics integration, not the thrice-validated principle): findings doc `research/findings/2026-05-18-compose-temporal-credit-spiking-<OUTCOME>.md` + `webapp/capability_status.json` pillar (n=72; status VALIDATED if PASS / BOUNDARY / NEGATIVE) + `python -m pytest tests/test_webapp_server.py -k capability_status -q` green + commit (scoped; protected byte-empty) + push BOTH remotes. Bars NOT tuned. NOT config-cranked. Moat byte-identical + 7/7 throughout. Honest ceiling NEVER spun.

---

## Remember
- Exact file paths; the verdict core + scaffold are complete verbatim; the in-bridge mechanism is interface-pinned + behaviorally-specified (genuine integration validated by the gate + adversarial review + controller smell-test — NOT a fabricated proven reference).
- Frozen `_CBR_*` pre-registered here with explicit spiking-substrate justification; NEVER tuned; ≥8 bindings by design excludes the small-N control artifact.
- Protected set byte-UNMODIFIED — controller verifies empty `git diff e8a99a2..HEAD` EVERY commit.
- @superpowers:subagent-driven-development drives Tasks 0–3; Task 5 is controller-only.
- Honest ceiling stated up front and NEVER spun. PASS/BOUNDARY/VOID all propagated honestly.
