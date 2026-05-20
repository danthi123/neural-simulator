"""TDD tests for the net-new theta-gamma mode-unification runner (Task 2).

Written FIRST (red before the runner lands). The runner implements
CUE SUPPRESSION DURING RETRIEVE: a three-phase theta cycle
(encode/gap/retrieve) where the FULL arm suppresses the cued-noun's
diffuse lang_input drive during the retrieve window so the engram
tag's selective bound-adj drive can dominate; the UNIFORM_CTRL arm
keeps the cue ON during retrieve as the decisive built-in control.

This is the load-bearing experimental contrast the architecture
introduces. It is grounded in the localisation finding (commit
110f7cd): the cued-noun's diffuse lang_input drive dominates the
engram tag's selective bound-adj drive at deployment.

The decisive multi-seed CuPy run is a later controller-only task;
this suite screens only that:

  (a) ``run_theta_gamma_mode_unification(seeds=[42,43,44],
      tiny_synth=True)`` runs end-to-end, returns a dict with
      ``rungs`` + ``verdict`` whose ``gate`` is one of the four
      frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the six required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a
      structural reason (it may legitimately FAIL on toy numbers --
      fine);
  (c) no shipped module text imports torch.autograd / .backward;
  (d) STRUCTURAL-EFFECT PROBE: the runner's actual code path
      produces NON-byte-identical bridge state between
      suppress_cue_during_retrieve=True (FULL arm) and
      suppress_cue_during_retrieve=False (UNIFORM_CTRL arm) -- the
      mechanism is structurally active (mirrors Pirazzini d462bf0
      lesson: must work via the runner's ACTUAL code path);
  (e) per-cell raw_cells emit BOTH full_acc and uniform_ctrl_acc and
      at least one cell exhibits the mechanism's structural effect
      (full_acc != uniform_ctrl_acc on at least one (seed, N)
      combination at tiny-synth scale -- if every cell shows
      equality, the mechanism is inert).

tiny_synth shrinks pools / events / phase-block lengths so this is
a fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.runners.theta_gamma_mode_unification_runner as tgr
from research.runners.theta_gamma_mode_unification_core import (
    REQUIRED_KEYS,
    theta_gamma_mode_unification_verdict,
)


_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    """(a): the runner exposes the documented public entry point AND a
    main() CLI dispatcher. Task 0's grounding pin asserts main is
    importable; this also asserts the orchestrating function."""
    assert hasattr(tgr, "run_theta_gamma_mode_unification")
    assert callable(tgr.run_theta_gamma_mode_unification)
    assert hasattr(tgr, "main")
    assert callable(tgr.main)


def test_tiny_synth_smoke_outputs_expected_json_shape(tmp_path):
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts. Every rung must carry EXACTLY the six
    required keys with correct types/ranges so the frozen verdict does
    not VOID structurally. The smoke must also write the JSON output
    when out_path is provided and disclaim its toy numbers."""
    out = tmp_path / "tg_smoke.json"
    result = tgr.run_theta_gamma_mode_unification(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
        out_path=str(out),
    )
    assert out.exists()
    assert isinstance(result, dict)
    assert result.get("mode") == "evaluation"
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

    for r in result["rungs"]:
        assert isinstance(r, dict)
        for k in REQUIRED_KEYS:
            assert k in r, "rung missing required key %s" % k
        assert isinstance(r["N"], int) and not isinstance(r["N"], bool)
        assert isinstance(r["n_seeds"], int) and not isinstance(
            r["n_seeds"], bool
        )
        for ak in (
            "full_acc",
            "uniform_ctrl_acc",
            "direct_retain_acc",
            "abstain_correct",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    recomputed = theta_gamma_mode_unification_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )
    assert result.get("tiny_synth") is True
    assert "note" in result and "NOT a result" in result["note"]


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(tgr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_structural_effect_probe_runs_via_runner_actual_code_path():
    """(d): the structural-effect probe MUST be exposed by the runner
    AND must exercise the runner's actual code path (the per-step
    theta-cycle helper) -- NOT a synthetic-bypass probe.

    The probe's contract: build the same substrate twice from the
    SAME seed; run the runner's actual theta-cycle helper once with
    suppress_cue_during_retrieve=True and once with
    suppress_cue_during_retrieve=False at the SAME initial state.
    Compare bridge.cp_membrane_potential_v. Must differ by > 1 mV.

    Mirrors the Pirazzini d462bf0 lesson: the structural-effect
    probe must work via the runner's ACTUAL code path, not a
    synthetic per-step loop. If the probe fails the runner must
    raise (no decisive numbers are reported when the mechanism is
    structurally inert).
    """
    assert hasattr(tgr, "_structural_effect_probe"), (
        "the runner must expose a `_structural_effect_probe` helper "
        "(mirrors Pirazzini d462bf0 lesson)"
    )
    # The probe must not raise -- if it raises, the mechanism is
    # structurally inert and the runner must abort.
    diff_mv = tgr._structural_effect_probe(seed=42, tiny_synth=True)
    assert isinstance(diff_mv, float) and diff_mv > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between suppress_cue_during_retrieve=True and "
        "=False at the SAME initial state; got %.6g mV. This is the "
        "inert-mechanism failure mode the Pirazzini d462bf0 lesson "
        "guards against."
        % diff_mv
    )


def test_full_vs_uniform_arms_differ_at_least_on_some_query():
    """(e): the FULL arm (cue suppressed during retrieve) and the
    UNIFORM_CTRL arm (cue stays ON during retrieve) must produce a
    DIFFERENT result on at least one (seed, N) cell at tiny-synth
    scale. If full_acc == uniform_ctrl_acc on EVERY cell, the
    mechanism is structurally inert.

    Decisive built-in control: per the frozen verdict,
    uniform_ctrl_max=0.10 is the bar the FULL arm must beat; this
    test checks only that the mechanism PRODUCES a difference (not
    that the difference is in the right direction at tiny-synth)."""
    result = tgr.run_theta_gamma_mode_unification(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
    )
    cells = result.get("raw_cells", [])
    assert isinstance(cells, list) and len(cells) >= 1
    has_a_diff = False
    for c in cells:
        full = float(c.get("full_acc", 0.0))
        uniform = float(c.get("uniform_ctrl_acc", 0.0))
        if abs(full - uniform) > 1e-9:
            has_a_diff = True
            break
    assert has_a_diff, (
        "the cue-suppression-during-retrieve mechanism produced "
        "ZERO difference between FULL and UNIFORM_CTRL on every "
        "(seed, N) cell at tiny-synth. The mechanism is structurally "
        "inert -- fix and re-run BEFORE decisive. raw_cells=%r"
        % cells
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
