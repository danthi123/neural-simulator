"""Direction Q verdict module - adversarial test matrix.

These tests pin the Direction Q verdict module's behavior:
- Pre-registered frozen thresholds (set ONCE, never modifiable by results)
- Standard library + typing only (no project imports)
- Instrument-validity check FIRST (fail-closed on malformed input)
- VOID strictly distinct from PASS/NEGATIVE (different tags)

The verdict module is one of the project's load-bearing frozen-verdict
modules following the same discipline pattern as prior moats: thresholds
are codified at module-load time; the function returns a categorical
tag; malformed input never crashes, it returns a VOID branch.

Spec source: docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-design.md
            docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-implementation.md
"""
from __future__ import annotations
import importlib.util
import os
import math
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICT_PATH = os.path.join(
    REPO_ROOT, "research/findings/raw/direction_Q_verdict.py",
)


def _load_verdict_module():
    """Load the verdict module fresh each call (so threshold-tamper
    tests reset state)."""
    spec = importlib.util.spec_from_file_location(
        "direction_Q_verdict", VERDICT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- Test 1: thresholds frozen at design values ----------

def test_thresholds_frozen_at_design_values():
    """Pre-registered design-doc thresholds must be present and equal
    to the values frozen in docs/plans/2026-05-25-direction-Q-...-design.md."""
    mod = _load_verdict_module()
    assert mod._Q_RATE_RATIO_MIN == 2.0, (
        "_Q_RATE_RATIO_MIN tampered: design fixes this at 2.0"
    )
    assert mod._Q_DELAY_MIN_SEC == 3.0, (
        "_Q_DELAY_MIN_SEC tampered: design fixes this at 3.0"
    )
    assert mod._Q_MIN_SEEDS_PASS == 3, (
        "_Q_MIN_SEEDS_PASS tampered: design fixes this at 3"
    )


# ---------- Test 2: PASS - all three seeds clear bar ----------

def test_pass_all_three_seeds_meet_bar():
    """All 3 seeds pass at (ratio>=2.0, sustained>=3.0s); control all fail.
    Expected: Q_BISTABILITY_PASS."""
    mod = _load_verdict_module()
    per_seed = [(2.5, 3.5), (2.5, 3.5), (2.5, 3.5)]
    control = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_BISTABILITY_PASS", (
        "Expected PASS; got " + str(verdict)
    )


# ---------- Test 3: PARTIAL - 2/3 seeds clear bar ----------

def test_partial_two_of_three_pass():
    """2 of 3 seeds pass; 3rd seed below ratio. Expected: PARTIAL."""
    mod = _load_verdict_module()
    per_seed = [(2.5, 3.5), (2.5, 3.5), (1.5, 3.5)]
    control = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_BISTABILITY_PARTIAL", (
        "Expected PARTIAL; got " + str(verdict)
    )


# ---------- Test 4: PARTIAL - short duration on all seeds ----------

def test_partial_short_duration():
    """All 3 seeds pass ratio but sustained < _Q_DELAY_MIN_SEC.
    Each seed individually fails (sustained=2.0 < 3.0). 0 seeds pass.
    Expected: Q_BISTABILITY_NEGATIVE (correction: 0 pass not PARTIAL).
    But spec says 3 seeds duration<3.0s -> PARTIAL. Re-read spec:
    'per_seed=[(2.5,2.0)]*3, control=[1,1,1]; expect PARTIAL (3 seeds
    but duration < 3.0s)' — the rationale is: rate-ratio criterion met
    across all 3 seeds; only the duration sub-criterion failed. Treat
    this as a 'something is happening but not the full bar' = PARTIAL.

    Implementation: a seed 'passes' iff BOTH ratio>=2.0 AND sustained>=3.0.
    Here ratio=2.5 passes but sustained=2.0 fails -> 0 seeds pass.
    To match the spec PARTIAL expectation, the verdict must distinguish
    'ratio-met-but-duration-short' from outright NEGATIVE.

    Adopting interpretation: any seed with rate_ratio>=_Q_RATE_RATIO_MIN
    OR sustained>=_Q_DELAY_MIN_SEC is a 'partial-pass' seed. Full-pass
    requires both. If 0 full-passes but >=1 partial-passes, verdict is
    PARTIAL (not NEGATIVE).
    """
    mod = _load_verdict_module()
    per_seed = [(2.5, 2.0), (2.5, 2.0), (2.5, 2.0)]
    control = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_BISTABILITY_PARTIAL", (
        "Expected PARTIAL (rate met but duration short); got "
        + str(verdict)
    )


# ---------- Test 5: NEGATIVE - all below bar ----------

def test_negative_all_below_bar():
    """All 3 seeds fail both ratio AND sustained. Expected: NEGATIVE."""
    mod = _load_verdict_module()
    per_seed = [(1.5, 1.5), (1.5, 1.5), (1.5, 1.5)]
    control = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_BISTABILITY_NEGATIVE", (
        "Expected NEGATIVE; got " + str(verdict)
    )


# ---------- Test 6: VOID - control also passes all seeds ----------

def test_void_control_also_passes():
    """Test seeds pass AND control seeds pass. Persistence is not
    NMDA-driven -> VOID."""
    mod = _load_verdict_module()
    per_seed = [(2.5, 3.5), (2.5, 3.5), (2.5, 3.5)]
    control = [(2.5, 3.5), (2.5, 3.5), (2.5, 3.5)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_VOID_CONTROL_ALSO_PASSED", (
        "Expected VOID; got " + str(verdict)
    )


# ---------- Test 7: VOID - control passes any single seed ----------

def test_void_control_partial_pass():
    """Pre-registered: ANY control seed passing the bar makes the
    result VOID (persistence not solely NMDA-driven)."""
    mod = _load_verdict_module()
    per_seed = [(2.5, 3.5), (2.5, 3.5), (2.5, 3.5)]
    control = [(2.5, 3.5), (1.0, 0.0), (1.0, 0.0)]
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_VOID_CONTROL_ALSO_PASSED", (
        "Expected VOID (any control pass voids); got " + str(verdict)
    )


# ---------- Test 8: VOID - malformed per_seed (None / empty) ----------

def test_malformed_per_seed_returns_void_none():
    """per_seed=None must return VOID, not crash."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict(None, [(1.0, 0.0)] * 3)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for None per_seed; got " + str(verdict)
    )


def test_malformed_per_seed_returns_void_empty():
    """per_seed=[] must return VOID."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict([], [(1.0, 0.0)] * 3)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for empty per_seed; got " + str(verdict)
    )


# ---------- Test 9: VOID - malformed control ----------

def test_malformed_control_returns_void_none():
    """control=None must return VOID."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict([(2.5, 3.5)] * 3, None)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for None control; got " + str(verdict)
    )


def test_malformed_control_returns_void_empty():
    """control=[] must return VOID."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict([(2.5, 3.5)] * 3, [])
    assert "VOID" in verdict.upper(), (
        "Expected VOID for empty control; got " + str(verdict)
    )


# ---------- Test 10: VOID - non-numeric data ----------

def test_non_numeric_data_returns_void():
    """Non-numeric per_seed entries must return VOID (instrument-validity)."""
    mod = _load_verdict_module()
    per_seed = [("abc", "def"), ("abc", "def"), ("abc", "def")]
    control = [(1.0, 0.0)] * 3
    verdict = mod.compute_verdict(per_seed, control)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for non-numeric; got " + str(verdict)
    )


# ---------- Test 11: VOID - NaN / Inf ----------

def test_nan_returns_void():
    """NaN values must return VOID (numeric instrument-validity)."""
    mod = _load_verdict_module()
    per_seed = [(float("nan"), float("nan"))] * 3
    control = [(1.0, 0.0)] * 3
    verdict = mod.compute_verdict(per_seed, control)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for NaN; got " + str(verdict)
    )


def test_inf_returns_void():
    """Inf values must return VOID (numeric instrument-validity)."""
    mod = _load_verdict_module()
    per_seed = [(float("inf"), float("inf"))] * 3
    control = [(1.0, 0.0)] * 3
    verdict = mod.compute_verdict(per_seed, control)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for Inf; got " + str(verdict)
    )


# ---------- Test 12: threshold tamper detection ----------

def test_threshold_tamper_detection():
    """The thresholds are module-level constants frozen at design-doc
    values. If a future edit silently lowers _Q_RATE_RATIO_MIN below
    2.0 (or _Q_DELAY_MIN_SEC below 3.0, or _Q_MIN_SEEDS_PASS below 3),
    THIS test breaks - explicit pin to the design-doc list-of-constants.

    Variant adopted: hold an EXPECTED_THRESHOLDS table separately and
    compare. Any silent in-source change is flagged at next pytest run.
    """
    mod = _load_verdict_module()
    EXPECTED_THRESHOLDS = {
        "_Q_RATE_RATIO_MIN": 2.0,
        "_Q_DELAY_MIN_SEC": 3.0,
        "_Q_MIN_SEEDS_PASS": 3,
    }
    for name, expected_val in EXPECTED_THRESHOLDS.items():
        actual = getattr(mod, name, None)
        assert actual == expected_val, (
            name + " tampered: expected " + repr(expected_val)
            + " got " + repr(actual)
        )


# ---------- Test 13: PARTIAL - one seed pass, two fail ----------

def test_one_seed_pass_two_fail():
    """1 of 3 seeds passes; 2 fail outright. Expected: PARTIAL."""
    mod = _load_verdict_module()
    per_seed = [(2.5, 3.5), (1.0, 0.5), (1.0, 0.5)]
    control = [(1.0, 0.0)] * 3
    verdict = mod.compute_verdict(per_seed, control)
    assert verdict == "Q_BISTABILITY_PARTIAL", (
        "Expected PARTIAL (1/3 pass); got " + str(verdict)
    )


# ---------- Test 14: stdlib-only imports ----------

def test_imports_only_stdlib():
    """The verdict module must import ONLY standard library / typing
    modules. No project-specific imports (sim.*, research.*) - this
    keeps the verdict module a pure scoring oracle that can be loaded
    without spinning up the simulator."""
    with open(VERDICT_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    forbidden_prefixes = (
        "sim.", "sim ", "research.", "research ", "viz.", "viz ",
        "ui.", "ui ", "experiment.", "experiment ",
        "numpy", "cupy", "scipy", "torch", "h5py", "matplotlib",
    )
    # Inspect every import statement
    allowed_modules = {
        "__future__", "typing", "math", "json", "os", "sys",
        "dataclasses", "enum", "collections", "abc",
    }
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Parse the module name
            if stripped.startswith("from "):
                # "from MODULE import ..."
                parts = stripped.split()
                if len(parts) >= 2:
                    module_name = parts[1].split(".")[0]
            else:
                # "import MODULE" or "import MODULE as X"
                parts = stripped.split()
                if len(parts) >= 2:
                    module_name = parts[1].split(".")[0]
                else:
                    continue
            assert module_name in allowed_modules, (
                "Forbidden import at line " + str(lineno) + ": '"
                + stripped + "' (module '" + module_name + "' is not "
                "stdlib/typing). Allowed: " + str(sorted(allowed_modules))
            )
            for fb in forbidden_prefixes:
                assert not stripped.startswith("import " + fb), (
                    "Forbidden project import at line " + str(lineno)
                    + ": " + stripped
                )
                assert not stripped.startswith("from " + fb), (
                    "Forbidden project import at line " + str(lineno)
                    + ": " + stripped
                )
