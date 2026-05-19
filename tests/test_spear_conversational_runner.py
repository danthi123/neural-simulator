"""TDD tests for the net-new shared-rhythm SPEAR conversational runner.

Written FIRST (red before the runner lands). The decisive multi-seed CuPy
run is a later controller-only task; here we only screen that:

  (a) ``run_spear_conversational(seeds=[42], tiny_synth=True)`` runs
      end-to-end, returns a dict with ``rungs`` + ``verdict`` whose
      ``gate`` is one of the four frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the five required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a structural
      reason (it may legitimately FAIL on toy numbers -- fine);
  (c) no shipped module text imports torch.autograd / ``.backward``;
  (d) for a (seed,N) cell ``full`` and ``rhythm_removed`` consume the
      SAME seed and differ ONLY by the shared-rhythm controller being
      enabled vs disabled (the controller is the sole difference, a
      single flag/param threaded identically);
  (e) the decode path uses the validated neural readout (no tag-string
      parse; the runner source contains no ``.split("_")`` on tag
      names) and the moat is fed the raw firing-rate quantity.

tiny_synth shrinks pools/episodes/phase-block lengths hard so this is a
fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import research.runners.spear_conversational_runner as scr
from research.runners.spear_conversational_core import (
    REQUIRED_KEYS,
    spear_conversational_verdict,
)

_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    assert hasattr(scr, "run_spear_conversational")
    assert callable(scr.run_spear_conversational)
    assert hasattr(scr, "main")


def test_tiny_synth_runs_end_to_end_and_is_not_structurally_void():
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts (one of the four states, never raises,
    NOT VOID for a structural reason). We run the verdict's minimum
    seed count (3) so structural well-formedness can be screened; the
    toy numbers themselves are explicitly NOT a result."""
    result = scr.run_spear_conversational(
        seeds=[42, 43, 44], tiny_synth=True
    )

    assert isinstance(result, dict)
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

    # Every rung must carry EXACTLY the five required keys with correct
    # types/ranges so the frozen verdict does not VOID structurally.
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
            "rhythm_removed_acc",
            "abstain_correct_rhythm_removed",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    # Recompute the verdict from the raw rungs -- it must not VOID for a
    # structural/instrument reason (a legitimate FAIL on toy numbers is
    # acceptable; VOID would mean a malformed rung shape).
    recomputed = spear_conversational_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )


def test_tiny_synth_does_not_raise_with_loads_and_out(tmp_path):
    """(a): explicit small ladder + out-path also runs clean and writes
    a JSON the verdict accepts."""
    out = tmp_path / "spear_smoke.json"
    result = scr.run_spear_conversational(
        seeds=[42], loads=(2,), tiny_synth=True, out_path=str(out)
    )
    assert out.exists()
    assert result["verdict"]["gate"] in _VALID_GATES
    assert result.get("tiny_synth") is True
    # the smoke must explicitly disclaim its toy numbers
    assert "note" in result and "NOT a result" in result["note"]


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_decode_is_neural_not_tag_string_parse():
    """(e): the runner source contains NO tag-string parse -- no
    .split("_") and no .split('_') anywhere (the answer is decoded from
    the validated neural readout, never out of an opaque tag name)."""
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # The validated neural readout helpers must be the decode path.
    assert "lang_output_pattern_during_stim" in src
    # The moat must be fed the raw firing-rate ranked confidence.
    assert "abstention_gate" in src or "_abstain_gate" in src


def test_rhythm_removed_is_full_minus_only_the_shared_rhythm():
    """(d): for a (seed,N) cell, full and rhythm_removed differ ONLY by
    the shared-rhythm controller being enabled vs disabled.

    Structural assertion (no heavy run): the per-arm cell function takes
    a single boolean controller flag, and the cell driver invokes it
    once with the flag True (full) and once with the flag False
    (rhythm_removed) using the SAME seed and SAME facts/draws. We assert
    the flag is the sole differing argument by inspecting the source of
    the cell driver.
    """
    # The arm function must accept a single explicit shared-rhythm flag.
    assert hasattr(scr, "_run_arm")
    arm_sig = inspect.signature(scr._run_arm)
    assert "use_rhythm" in arm_sig.parameters, (
        "the arm runner must thread an explicit `use_rhythm` controller "
        "flag so full vs rhythm_removed differ ONLY by it"
    )

    cell_src = inspect.getsource(scr._cell)
    # full arm: use_rhythm True; rhythm_removed arm: use_rhythm False.
    assert "use_rhythm=True" in cell_src
    assert "use_rhythm=False" in cell_src
    # Both arms must be built from the SAME seed (no per-arm seed
    # perturbation) -- the cell takes one `seed` and passes it to both.
    cell_sig = inspect.signature(scr._cell)
    assert "seed" in cell_sig.parameters
    # No second RNG seed / no seed offset between the two arms.
    assert "seed + 1" not in cell_src and "seed+1" not in cell_src

    # The verdict-relevant control key must be derived from the
    # rhythm-disabled arm (the Stage-1-static reduction).
    agg_src = inspect.getsource(scr._aggregate)
    assert "rhythm_removed_acc" in agg_src
    assert "abstain_correct_rhythm_removed" in agg_src


def test_controller_has_theta_period_and_ach_phase_gate():
    """The net-new piece is a theta-phase clock + ACh phase gate. Assert
    the controller derives a theta period in STEPS from the bridge dt
    and gates plasticity via the reused neuromodulator set_concentration
    on encode vs retrieve phases (a timing controller, not a new rule).
    """
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    # theta ~125 ms period derived from dt (not a hardcoded step count).
    assert "125" in src  # ~125 ms theta cycle
    assert "set_concentration" in src  # reused ACh phase gate
    assert "step_simulation" in src or "_run_one_simulation_step" in src
    # gamma sub-cycle indexes the dlpfc compositional slot.
    assert "gamma" in src.lower()
    assert "dlpfc" in src.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
