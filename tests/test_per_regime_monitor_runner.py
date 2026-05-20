"""TDD tests for the net-new per-regime metacognitive-monitor runner.

Written FIRST (red before the runner lands). The decisive multi-seed CuPy
run is a later controller-only task (Task 6); this suite screens only:

  (a) ``run_per_regime_monitor(seeds=[42,43,44], tiny_synth=True,
      calibrate=False)`` runs end-to-end, returns a dict with ``rungs`` +
      ``verdict`` whose ``gate`` is one of the four frozen states, AND
      NEVER raises; ``mode == "evaluation"`` is recorded;
  (b) ``run_per_regime_monitor(seeds=[42,43,44], tiny_synth=True,
      calibrate=True)`` returns ``mode == "calibration"`` + a
      well-formed calibration payload (per-seed calibrated thresholds,
      committed_threshold echoing COMPOSITIONAL_THRESHOLD = 0.0,
      calibration_status in {MATCH, PENDING, MISMATCH}, non-empty
      method docstring);
  (c) no shipped module text imports torch.autograd / ``.backward``;
  (d) OPAQUE tag names: the runner source contains no ``.split("_")``
      on tag names (Stage-1 / SPEAR / Pirazzini lesson);
  (e) ``direct_retain`` is a separate measurement from ``full_acc`` --
      the runner records direct-query accuracy as its own rung key (and
      ``direct_retain_acc != full_acc`` is structurally possible);
  (f) ``uniform_ctrl`` plumbing: the runner source uses ``MOAT_DIRECT``
      (the existing 650 threshold) in BOTH gate calls for the
      uniform_ctrl arm (the SOLE difference vs ``full`` is the
      threshold routing decision);
  (g) source-file integrity: a tiny-synth calibration run does NOT
      mutate ``research/runners/abstention_gate_compositional.py`` --
      calibration only writes the JSON output; updating the source-file
      constant is the controller's job (a separate commit).

tiny_synth shrinks pools/episodes/queries hard so the smoke is fast.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

import research.runners.per_regime_monitor_runner as prr
from research.runners.per_regime_monitor_core import (
    REQUIRED_KEYS,
    per_regime_monitor_verdict,
)
from research.runners.abstention_gate_compositional import (
    COMPOSITIONAL_THRESHOLD,
)
from research.runners.abstention_gate import (
    DEFAULT_THRESHOLD as _MOAT_DIRECT,
)

_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}

_COMPOSITIONAL_GATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "research"
    / "runners"
    / "abstention_gate_compositional.py"
)


def test_runner_module_exposes_entry_point():
    assert hasattr(prr, "run_per_regime_monitor")
    assert callable(prr.run_per_regime_monitor)
    assert hasattr(prr, "main")


def test_tiny_synth_evaluation_runs_end_to_end_and_is_not_structurally_void():
    """(a): a tiny-synth multi-seed evaluation run returns a well-formed
    dict the frozen verdict accepts (one of the four states, never
    raises, NOT VOID for a structural reason). Toy numbers are
    explicitly NOT a result."""
    result = prr.run_per_regime_monitor(
        seeds=[42, 43, 44], tiny_synth=True, calibrate=False
    )

    assert isinstance(result, dict)
    assert result.get("mode") == "evaluation"
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

    # Every rung must carry EXACTLY the six required keys with correct
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
            "uniform_ctrl_acc",
            "direct_retain_acc",
            "abstain_correct",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    recomputed = per_regime_monitor_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )


def test_tiny_synth_evaluation_writes_json_with_out_path(tmp_path):
    """(a): explicit small ladder + out-path also runs clean and writes
    a JSON the verdict accepts."""
    out = tmp_path / "per_regime_eval_smoke.json"
    result = prr.run_per_regime_monitor(
        seeds=[42],
        loads=(2,),
        tiny_synth=True,
        calibrate=False,
        out_path=str(out),
    )
    assert out.exists()
    assert result["verdict"]["gate"] in _VALID_GATES
    assert result.get("tiny_synth") is True
    assert result.get("mode") == "evaluation"
    # the smoke must explicitly disclaim its toy numbers
    assert "note" in result and "NOT a result" in result["note"]


def test_tiny_synth_calibration_runs_and_returns_well_formed_payload(tmp_path):
    """(b): a tiny-synth multi-seed calibration run returns
    mode=='calibration' + a well-formed calibration payload. The
    calibration does NOT produce a decisive verdict."""
    out = tmp_path / "per_regime_calib_smoke.json"
    result = prr.run_per_regime_monitor(
        seeds=[42, 43, 44],
        tiny_synth=True,
        calibrate=True,
        out_path=str(out),
    )

    assert isinstance(result, dict)
    assert result.get("mode") == "calibration"
    assert "per_seed_calibrated_thresholds" in result
    assert isinstance(result["per_seed_calibrated_thresholds"], list)
    assert len(result["per_seed_calibrated_thresholds"]) == 3
    for v in result["per_seed_calibrated_thresholds"]:
        assert isinstance(v, float)

    # committed_threshold echoes the placeholder constant for this Task.
    assert result.get("committed_threshold") == float(COMPOSITIONAL_THRESHOLD)
    assert result.get("calibration_status") in {"MATCH", "PENDING", "MISMATCH"}
    method = result.get("method")
    assert isinstance(method, str) and len(method.strip()) > 0
    assert result.get("tiny_synth") is True
    # The runner makes the toy nature explicit on calibration too.
    assert "note" in result and "NOT a result" in result["note"]
    # Calibration does NOT carry a decisive verdict / rungs payload.
    assert "verdict" not in result or result.get("verdict") is None
    assert out.exists()


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_decode_is_neural_not_tag_string_parse():
    """(d): the runner source contains NO tag-string parse on tag names
    (.split("_") / .split('_')). Stage-1 / SPEAR / Pirazzini lesson:
    the answer is decoded from the validated neural readout, never out
    of an opaque tag name."""
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # The validated neural readout: the runner reuses _ranked_from_pattern
    # (the calibrated raw firing-rate confidence formula).
    assert "_ranked_from_pattern" in src
    # Tag names must be opaque: e.g. "ep_%d" / f"ep_{i}".
    assert ("ep_%d" in src) or ("ep_{i}" in src) or ('"ep_"' in src)


def test_direct_retain_is_a_separate_measurement_from_full_acc():
    """(e): the runner records direct-query accuracy as its OWN rung
    field, independent of full_acc. Structural test on a tiny-synth
    evaluation run: rung dicts must carry both keys, and the runner
    source must compute direct_retain_acc separately (no aliasing)."""
    result = prr.run_per_regime_monitor(
        seeds=[42, 43, 44], tiny_synth=True, calibrate=False
    )
    assert len(result["rungs"]) >= 1
    r = result["rungs"][0]
    # Both keys must exist on every rung (the frozen verdict requires
    # them).
    assert "full_acc" in r and "direct_retain_acc" in r
    # The runner source must NOT collapse direct_retain_acc to full_acc.
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "direct_retain_acc" in src
    # The runner must measure direct-query accuracy under the per-regime
    # arm (the SAME run as `full`, not a separate draw). Source must
    # contain a per-arm direct-only accumulator (the simplest reliable
    # marker is a named direct-counter inside the same arm function).
    assert "n_direct" in src or "direct_correct" in src or "direct_total" in src


def test_uniform_ctrl_uses_moat_direct_in_both_gates():
    """(f): uniform_ctrl differs from `full` ONLY by the threshold-
    routing decision. The runner source must use the existing
    MOAT_DIRECT constant (650.0) for BOTH gate calls in the
    uniform_ctrl arm (otherwise it is not 'full minus only the
    threshold decision'). The full arm uses MOAT_DIRECT for direct
    queries and COMPOSITIONAL_THRESHOLD for compositional queries.
    """
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The runner imports both moat thresholds.
    assert "MOAT_DIRECT" in src
    assert "COMPOSITIONAL_THRESHOLD" in src
    # The uniform_ctrl arm must be present as a named concept (so the
    # control thread is auditable).
    assert "uniform_ctrl" in src
    # Both gates wired in (the direct moat AND the compositional gate).
    assert (
        "abstention_gate" in src
        or "gate_direct" in src
        or "_abstain_gate" in src
    )
    assert (
        "abstention_gate_compositional" in src
        or "gate_compositional" in src
        or "_gate_compositional" in src
    )


def test_calibration_does_not_mutate_compositional_gate_source():
    """(g): the runner's calibration mode writes its output to the
    --out JSON ONLY. It must NOT modify the source file
    `research/runners/abstention_gate_compositional.py` (updating the
    committed constant is a separate controller commit, NOT the
    runner's job). We assert that a tiny-synth calibration leaves the
    file's SHA-256 unchanged.
    """
    assert _COMPOSITIONAL_GATE_PATH.exists()
    before = hashlib.sha256(_COMPOSITIONAL_GATE_PATH.read_bytes()).hexdigest()
    prr.run_per_regime_monitor(
        seeds=[42, 43, 44], tiny_synth=True, calibrate=True
    )
    after = hashlib.sha256(_COMPOSITIONAL_GATE_PATH.read_bytes()).hexdigest()
    assert before == after, (
        "The runner's calibration mode wrote to "
        "abstention_gate_compositional.py. The committed-constant update "
        "is a separate controller commit; the runner must record the "
        "calibrated value in JSON only."
    )


def test_run_signature_threads_calibrate_flag_and_per_query_routing():
    """Structural: the entry point accepts a single boolean
    `calibrate` flag (the calibration / evaluation switch). The runner
    has separate handling for direct vs compositional queries (the
    per-query-type routing layer)."""
    sig = inspect.signature(prr.run_per_regime_monitor)
    assert "seeds" in sig.parameters
    assert "calibrate" in sig.parameters
    # Per-query routing: the source must explicitly handle the two
    # query-type strings.
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "direct" in src and "compositional" in src


def test_main_entry_accepts_calibrate_and_tiny_synth_flags():
    """The CLI must expose --calibrate and --tiny-synth so the
    controller can drive both modes at full scale (Task 6) without
    changing the runner."""
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '"--calibrate"' in src
    assert '"--tiny-synth"' in src
    assert '"--seeds"' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
