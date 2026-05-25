"""Direction 3 V=32 verdict module - adversarial test matrix.

These tests pin the Direction 3 V=32 verdict module's behavior:
- Pre-registered frozen thresholds (set ONCE, never modifiable by results)
- Standard library + typing only (no project imports)
- Instrument-validity check FIRST (fail-closed on malformed input)
- VOID strictly distinct from PASS/PARTIAL/NEGATIVE (different tags)

The verdict module is one of the project's load-bearing frozen-verdict
modules following the same discipline pattern as prior moats: thresholds
are codified at module-load time; the function returns a categorical
tag; malformed input never crashes, it returns a VOID branch.

Spec source: docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_-
             regions-design.md
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICT_PATH = os.path.join(
    REPO_ROOT, "research/findings/raw/direction_3_verdict.py",
)


def _load_verdict_module():
    """Load the verdict module fresh each call (so threshold-tamper
    tests reset state)."""
    spec = importlib.util.spec_from_file_location(
        "direction_3_verdict", VERDICT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_seed_entry(ob_l2=1.0, oi_l2=1.0,
                      ob_l3=1.0, oi_l3=1.0,
                      ob_l5=1.0, oi_l5=1.0):
    """Build a valid per-seed dict with overrideable per-cell values."""
    return {
        "L=2": {"OB": ob_l2, "OI": oi_l2},
        "L=3": {"OB": ob_l3, "OI": oi_l3},
        "L=5": {"OB": ob_l5, "OI": oi_l5},
    }


# ---------- Test 1: thresholds frozen at design values ----------

def test_thresholds_frozen_at_design_values():
    """Pre-registered design-doc thresholds must be present and equal
    to the values frozen in the design doc."""
    mod = _load_verdict_module()
    assert mod._DIRECTION_3_V32_OB_MIN == 0.80, (
        "_DIRECTION_3_V32_OB_MIN tampered: design fixes this at 0.80"
    )
    assert mod._DIRECTION_3_V32_OI_MIN == 0.80, (
        "_DIRECTION_3_V32_OI_MIN tampered: design fixes this at 0.80"
    )
    assert tuple(mod._DIRECTION_3_V32_LOADS) == (2, 3, 5), (
        "_DIRECTION_3_V32_LOADS tampered: design fixes this at (2, 3, 5)"
    )
    assert mod._DIRECTION_3_V32_MIN_SEEDS == 3, (
        "_DIRECTION_3_V32_MIN_SEEDS tampered: design fixes this at 3"
    )


# ---------- Test 2: PASS - all cells multi-seed-mean above bar ----------

def test_pass_all_cells_above_bar():
    """All 3 seeds at 1.0/1.0 across all 3 loads -> mean 1.0/1.0 each cell.
    Expected: DIRECTION_3_V32_PASS."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry() for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_PASS", (
        "Expected PASS; got " + str(verdict)
    )


# ---------- Test 3: PASS - exactly at the bar ----------

def test_pass_exactly_at_bar():
    """All seeds exactly at 0.80 multi-seed-mean. The bar is inclusive
    (>= 0.80). Expected: PASS."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry(
        ob_l2=0.80, oi_l2=0.80,
        ob_l3=0.80, oi_l3=0.80,
        ob_l5=0.80, oi_l5=0.80,
    ) for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_PASS", (
        "Expected PASS at-bar; got " + str(verdict)
    )


# ---------- Test 4: PARTIAL - OI fails at L=5 only ----------

def test_partial_oi_fails_at_l5():
    """OB perfect at all loads; OI passes L=2 + L=3 but fails L=5.
    Expected: DIRECTION_3_V32_PARTIAL."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry(
        ob_l2=1.0, oi_l2=1.0,
        ob_l3=1.0, oi_l3=1.0,
        ob_l5=1.0, oi_l5=0.50,
    ) for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_PARTIAL", (
        "Expected PARTIAL (1 cell below); got " + str(verdict)
    )


# ---------- Test 5: PARTIAL - OB fails at L=3 only ----------

def test_partial_ob_fails_at_l3():
    """OI perfect; OB fails at L=3 only. Expected: PARTIAL."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry(
        ob_l2=1.0, oi_l2=1.0,
        ob_l3=0.60, oi_l3=1.0,
        ob_l5=1.0, oi_l5=1.0,
    ) for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_PARTIAL", (
        "Expected PARTIAL (OB L=3 below); got " + str(verdict)
    )


# ---------- Test 6: NEGATIVE - all cells below bar ----------

def test_negative_all_cells_below_bar():
    """Every (load, readout) cell below 0.80; multi-seed-mean is 0.10
    each cell. Expected: DIRECTION_3_V32_NEGATIVE (no cells pass)."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry(
        ob_l2=0.10, oi_l2=0.10,
        ob_l3=0.10, oi_l3=0.10,
        ob_l5=0.10, oi_l5=0.10,
    ) for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_NEGATIVE", (
        "Expected NEGATIVE; got " + str(verdict)
    )


# ---------- Test 7: PARTIAL - only one cell above bar ----------

def test_partial_only_one_cell_above():
    """Only OB L=2 above bar; all other cells below. Expected: PARTIAL
    (because the count of passing cells is >0 but not all)."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry(
        ob_l2=1.0, oi_l2=0.30,
        ob_l3=0.30, oi_l3=0.30,
        ob_l5=0.30, oi_l5=0.30,
    ) for _ in range(3)]
    verdict = mod.compute_verdict(per_seed)
    assert verdict == "DIRECTION_3_V32_PARTIAL", (
        "Expected PARTIAL (1/6 cells pass); got " + str(verdict)
    )


# ---------- Test 8: VOID - None input ----------

def test_void_none_input():
    """per_seed=None must return VOID, not crash."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict(None)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for None per_seed; got " + str(verdict)
    )


# ---------- Test 9: VOID - empty list ----------

def test_void_empty_list():
    """per_seed=[] must return VOID (zero seeds is malformed)."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict([])
    assert "VOID" in verdict.upper(), (
        "Expected VOID for empty per_seed; got " + str(verdict)
    )


# ---------- Test 10: VOID - below MIN_SEEDS ----------

def test_void_too_few_seeds():
    """2 seeds (< _DIRECTION_3_V32_MIN_SEEDS=3) must return VOID."""
    mod = _load_verdict_module()
    per_seed = [_make_seed_entry() for _ in range(2)]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for 2 seeds (below MIN_SEEDS=3); got "
        + str(verdict)
    )


# ---------- Test 11: VOID - missing load key ----------

def test_void_missing_load_key():
    """A seed entry missing the L=5 key must return VOID."""
    mod = _load_verdict_module()
    bad_entry = {
        "L=2": {"OB": 1.0, "OI": 1.0},
        "L=3": {"OB": 1.0, "OI": 1.0},
        # missing L=5
    }
    per_seed = [bad_entry, _make_seed_entry(), _make_seed_entry()]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for missing L=5 key; got " + str(verdict)
    )


# ---------- Test 12: VOID - missing OB key inside a load ----------

def test_void_missing_ob_key_inside_load():
    """A seed missing the 'OB' key inside L=3 must return VOID."""
    mod = _load_verdict_module()
    bad_entry = {
        "L=2": {"OB": 1.0, "OI": 1.0},
        "L=3": {"OI": 1.0},  # missing OB
        "L=5": {"OB": 1.0, "OI": 1.0},
    }
    per_seed = [bad_entry, _make_seed_entry(), _make_seed_entry()]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for missing OB inside L=3; got " + str(verdict)
    )


# ---------- Test 13: VOID - NaN value ----------

def test_void_nan_value():
    """NaN OB value must return VOID."""
    mod = _load_verdict_module()
    bad_entry = {
        "L=2": {"OB": float("nan"), "OI": 1.0},
        "L=3": {"OB": 1.0, "OI": 1.0},
        "L=5": {"OB": 1.0, "OI": 1.0},
    }
    per_seed = [bad_entry, _make_seed_entry(), _make_seed_entry()]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for NaN; got " + str(verdict)
    )


# ---------- Test 14: VOID - Inf value ----------

def test_void_inf_value():
    """Inf OI value must return VOID."""
    mod = _load_verdict_module()
    bad_entry = {
        "L=2": {"OB": 1.0, "OI": float("inf")},
        "L=3": {"OB": 1.0, "OI": 1.0},
        "L=5": {"OB": 1.0, "OI": 1.0},
    }
    per_seed = [bad_entry, _make_seed_entry(), _make_seed_entry()]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for Inf; got " + str(verdict)
    )


# ---------- Test 15: VOID - string value (non-numeric) ----------

def test_void_string_value():
    """Non-numeric (string) value must return VOID."""
    mod = _load_verdict_module()
    bad_entry = {
        "L=2": {"OB": "abc", "OI": 1.0},
        "L=3": {"OB": 1.0, "OI": 1.0},
        "L=5": {"OB": 1.0, "OI": 1.0},
    }
    per_seed = [bad_entry, _make_seed_entry(), _make_seed_entry()]
    verdict = mod.compute_verdict(per_seed)
    assert "VOID" in verdict.upper(), (
        "Expected VOID for string OB; got " + str(verdict)
    )


# ---------- Test 16: VOID - wrong container type ----------

def test_void_wrong_container_type():
    """per_seed=dict (not list/tuple) must return VOID."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict({"seed_42": _make_seed_entry()})
    assert "VOID" in verdict.upper(), (
        "Expected VOID for dict-typed per_seed; got " + str(verdict)
    )


# ---------- Test 17: threshold tamper detection ----------

def test_threshold_tamper_detection():
    """The thresholds are module-level constants frozen at design-doc
    values. If a future edit silently lowers either bar (or shrinks the
    load ladder, or drops MIN_SEEDS), THIS test breaks."""
    mod = _load_verdict_module()
    EXPECTED_THRESHOLDS = {
        "_DIRECTION_3_V32_OB_MIN": 0.80,
        "_DIRECTION_3_V32_OI_MIN": 0.80,
        "_DIRECTION_3_V32_MIN_SEEDS": 3,
    }
    for name, expected_val in EXPECTED_THRESHOLDS.items():
        actual = getattr(mod, name, None)
        assert actual == expected_val, (
            name + " tampered: expected " + repr(expected_val)
            + " got " + repr(actual)
        )
    # Load ladder is separately tuple-checked.
    assert tuple(mod._DIRECTION_3_V32_LOADS) == (2, 3, 5), (
        "_DIRECTION_3_V32_LOADS tampered: expected (2, 3, 5) got "
        + repr(tuple(mod._DIRECTION_3_V32_LOADS))
    )


# ---------- Test 18: stdlib-only imports ----------

def test_imports_only_stdlib():
    """The verdict module must import ONLY standard library / typing
    modules. No project-specific imports (sim.*, research.*) - this
    keeps the verdict module a pure scoring oracle that can be loaded
    without spinning up the simulator."""
    with open(VERDICT_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    allowed_modules = {
        "__future__", "typing", "math", "json", "os", "sys",
        "dataclasses", "enum", "collections", "abc",
    }
    forbidden_prefixes = (
        "sim.", "sim ", "research.", "research ", "viz.", "viz ",
        "ui.", "ui ", "experiment.", "experiment ",
        "numpy", "cupy", "scipy", "torch", "h5py", "matplotlib",
    )
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module_name = parts[1].split(".")[0]
            else:
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


# ---------- Test 19: VOID - completely wrong shape (list of int) ----------

def test_void_completely_wrong_shape():
    """per_seed=[1, 2, 3] must return VOID."""
    mod = _load_verdict_module()
    verdict = mod.compute_verdict([1, 2, 3])
    assert "VOID" in verdict.upper(), (
        "Expected VOID for list-of-int per_seed; got " + str(verdict)
    )


# ---------- Test 20: tag constants are well-formed strings ----------

def test_tag_constants_are_strings():
    """Verdict tag constants must be unique non-empty strings (so external
    callers can compare with == without typos)."""
    mod = _load_verdict_module()
    tags = {
        "PASS": mod.DIRECTION_3_V32_PASS,
        "PARTIAL": mod.DIRECTION_3_V32_PARTIAL,
        "NEGATIVE": mod.DIRECTION_3_V32_NEGATIVE,
        "VOID": mod.DIRECTION_3_V32_VOID_MALFORMED,
    }
    for label, tag in tags.items():
        assert isinstance(tag, str), (
            label + " tag is not a string: " + repr(tag)
        )
        assert len(tag) > 0, label + " tag is empty"
    # All 4 must be distinct
    assert len(set(tags.values())) == 4, (
        "verdict tag constants must be unique; got " + repr(tags)
    )
