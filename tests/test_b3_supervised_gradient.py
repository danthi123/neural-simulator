"""Tests for research.runners.b3_supervised_gradient.

The runner replaces STDP with a delta-rule supervised gradient update
on language_input -> motor_X weights. Tests here verify the runner
imports cleanly and the argparse CLI has the expected flags. Full GPU
verification happens at runtime under the autonomous chain or
experiments/b3_supervised_gradient.yaml batch.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_module_imports_cleanly():
    """No missing imports, no syntax errors."""
    import research.runners.b3_supervised_gradient as m
    # Functions exist
    assert callable(m.build_supervised_gradient_step)
    assert callable(m.run_supervised_gradient)
    assert callable(m.main)


def test_argparse_has_expected_flags():
    """The CLI exposes the documented flags."""
    import sys
    import research.runners.b3_supervised_gradient as m

    saved_argv = sys.argv
    try:
        sys.argv = ["b3_supervised_gradient", "--help"]
        with pytest.raises(SystemExit) as ei:
            m.main()
        assert ei.value.code == 0
    finally:
        sys.argv = saved_argv


def test_argparse_required_flags_documented(capsys):
    """The expected flags are present in --help output."""
    import sys
    import research.runners.b3_supervised_gradient as m

    saved_argv = sys.argv
    try:
        sys.argv = ["b3_supervised_gradient", "--help"]
        with pytest.raises(SystemExit):
            m.main()
        captured = capsys.readouterr()
        help_text = captured.out
        # Core flags from the spec
        assert "--seed" in help_text
        assert "--n-events-per-direction" in help_text
        assert "--learning-rate" in help_text
        assert "--stim-steps-per-event" in help_text
        assert "--reset-steps" in help_text
        assert "--token-sparsity" in help_text
        assert "--apply-topographic-bias" in help_text
        assert "--enable-motor-fs" in help_text
        assert "--out-stats" in help_text
    finally:
        sys.argv = saved_argv


def test_b3_yaml_parseable():
    """The B3 follow-up YAML loads and matches the experiment_runner schema."""
    from research.experiment_runner import ExperimentConfig

    yaml_path = Path("experiments/b3_supervised_gradient.yaml")
    assert yaml_path.exists()
    cfg = ExperimentConfig.from_yaml(yaml_path)
    assert cfg.name == "b3-supervised-gradient"
    assert cfg.runner == "research.runners.b3_supervised_gradient"
    # 3 conditions: vanilla / with_topo / with_topo_fs
    assert len(cfg.conditions) == 3
    cond_names = {c.name for c in cfg.conditions}
    assert cond_names == {"vanilla", "with_topo", "with_topo_fs"}
    # 3 seeds (gradient training is slower)
    assert len(cfg.seeds) == 3
    assert set(cfg.seeds) == {42, 43, 44}
    # parallelism 3 (one per seed in parallel for one condition at a time)
    assert cfg.parallelism == 3


def test_b3_aggregator_config_present():
    """The b3 follow-up is wired into result_aggregator."""
    from research.result_aggregator import BUILTIN_CONFIGS
    assert "b3_supervised_gradient" in BUILTIN_CONFIGS
    b3 = BUILTIN_CONFIGS["b3_supervised_gradient"]["conditions"]
    # Three conditions matching the YAML
    assert len(b3) == 3
    # Filenames must match out_stats_template in the YAML
    for label, pattern in b3.items():
        assert pattern.startswith("text_eval_b3_")
        assert "seed{seed}" in pattern
    # b3 uses 3 seeds (matches its own YAML / runner cost)
    assert BUILTIN_CONFIGS["b3_supervised_gradient"].get("seeds") == [42, 43, 44]
