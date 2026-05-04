"""Tests for research.runners.eval_sanity_check.

The runner builds the minimal arch + hand-built perfect weights and
runs evaluate_word_to_action. Tests here verify the runner imports
cleanly and the argparse CLI has the expected flags. Full GPU
verification happens at runtime (under wait_biology_then_decide.ps1
or via the experiments/eval_sanity_check.yaml batch).
"""
from __future__ import annotations

import pytest


def test_module_imports_cleanly():
    """No missing imports, no syntax errors."""
    import research.runners.eval_sanity_check as m
    # Functions exist
    assert callable(m.hand_build_perfect_weights)
    assert callable(m.run_sanity_check)
    assert callable(m.main)


def test_argparse_has_expected_flags():
    """The CLI exposes the documented flags."""
    import argparse
    import research.runners.eval_sanity_check as m

    # Build an argparse from the module's main signature by inspecting
    # the source. Simpler: just call main with --help and check stdout
    # contains expected flag names. But --help calls sys.exit(0), so
    # capture through subprocess instead. Cleaner: parse manually.
    import sys
    saved_argv = sys.argv
    try:
        sys.argv = ["eval_sanity_check", "--help"]
        with pytest.raises(SystemExit) as ei:
            m.main()
        # argparse --help exits with code 0
        assert ei.value.code == 0
    finally:
        sys.argv = saved_argv


def test_argparse_required_flags_documented(capsys):
    """The expected flags are present in --help output."""
    import sys
    import research.runners.eval_sanity_check as m

    saved_argv = sys.argv
    try:
        sys.argv = ["eval_sanity_check", "--help"]
        with pytest.raises(SystemExit):
            m.main()
        captured = capsys.readouterr()
        help_text = captured.out
        # Expected flags
        assert "--seed" in help_text
        assert "--n-lang-input" in help_text
        assert "--n-motor-per-action" in help_text
        assert "--token-sparsity" in help_text
        assert "--target-weight" in help_text
        assert "--text-input-to-motor-density" in help_text
        assert "--n-eval-per-word" in help_text
        assert "--out-stats" in help_text
    finally:
        sys.argv = saved_argv


def test_min_biology_yaml_parseable():
    """The A1 follow-up YAML loads and matches the experiment_runner schema."""
    import yaml
    from pathlib import Path
    from research.experiment_runner import ExperimentConfig

    yaml_path = Path("experiments/minimum_biology.yaml")
    assert yaml_path.exists()
    cfg = ExperimentConfig.from_yaml(yaml_path)
    assert cfg.name == "minimum-biology"
    assert cfg.runner == "research.runners.text_minimal_isolation"
    assert cfg.parallelism == 3
    assert len(cfg.seeds) == 6
    assert len(cfg.conditions) == 4
    cond_names = {c.name for c in cfg.conditions}
    assert cond_names == {"topo_weak", "fs_minimal", "topo_strong", "combo_weak"}


def test_sanity_check_yaml_parseable():
    """The B1 follow-up YAML loads and matches the experiment_runner schema."""
    from pathlib import Path
    from research.experiment_runner import ExperimentConfig

    yaml_path = Path("experiments/eval_sanity_check.yaml")
    assert yaml_path.exists()
    cfg = ExperimentConfig.from_yaml(yaml_path)
    assert cfg.name == "eval-sanity-check"
    assert cfg.runner == "research.runners.eval_sanity_check"
    assert len(cfg.conditions) == 2
    cond_names = {c.name for c in cfg.conditions}
    assert cond_names == {"density030", "density100"}
