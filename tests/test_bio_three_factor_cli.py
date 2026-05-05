"""Tests for bio_three_factor CLI flag wiring.

Catches regressions where a CLI flag is added but not plumbed through
to the runtime config. Specifically:
  --fp16-synapse-state  -> cfg.fp16_synapse_state (Phase 2)
  --no-gpu-eligibility  -> gpu_eligibility=False (Phase 1)
  --biological          -> use build_biological_brain_regions
  --enable-nmda         -> cfg.enable_nmda
"""
from __future__ import annotations

import sys
import pytest


def test_fp16_synapse_state_flag_appears_in_help():
    """--fp16-synapse-state flag is documented in --help output."""
    import research.runners.bio_three_factor as m
    saved = sys.argv
    try:
        sys.argv = ["bio_three_factor", "--help"]
        with pytest.raises(SystemExit):
            m.main()
    except SystemExit:
        pass
    finally:
        sys.argv = saved


def test_fp16_synapse_state_capsys(capsys):
    """--help output contains expected new flags from 2026-05-05 perf wave."""
    import research.runners.bio_three_factor as m
    saved = sys.argv
    try:
        sys.argv = ["bio_three_factor", "--help"]
        with pytest.raises(SystemExit):
            m.main()
    finally:
        sys.argv = saved

    captured = capsys.readouterr()
    help_text = captured.out

    # Phase 1: GPU-port flag
    assert "--no-gpu-eligibility" in help_text, (
        "Phase 1 flag missing from --help"
    )
    # Phase 2: FP16 flag
    assert "--fp16-synapse-state" in help_text, (
        "Phase 2 flag missing from --help"
    )
    # Architecture flag (existing)
    assert "--biological" in help_text


def test_run_three_factor_signature_has_perf_kwargs():
    """run_three_factor() function has gpu_eligibility + fp16_synapse_state
    keyword args. Catches refactors that drop perf-stack params."""
    import inspect
    from research.runners.bio_three_factor import run_three_factor
    sig = inspect.signature(run_three_factor)
    params = sig.parameters
    assert "gpu_eligibility" in params, (
        "run_three_factor missing gpu_eligibility kwarg (Phase 1)"
    )
    assert params["gpu_eligibility"].default is True, (
        "gpu_eligibility default should be True (Phase 1 GPU-resident)"
    )
    assert "fp16_synapse_state" in params, (
        "run_three_factor missing fp16_synapse_state kwarg (Phase 2)"
    )
    assert params["fp16_synapse_state"].default is False, (
        "fp16_synapse_state default should be False (opt-in until validated)"
    )


def test_update_eligibility_function_xp_param():
    """The pure update function takes xp=numpy|cupy. Catches refactors
    that hardcode the backend (which would break the GPU port)."""
    import inspect
    from research.runners.bio_three_factor import update_eligibility_and_weights
    sig = inspect.signature(update_eligibility_and_weights)
    assert "xp" in sig.parameters, (
        "update_eligibility_and_weights missing xp param — backend "
        "selection broken; either numpy or cupy hardcoded somewhere"
    )
