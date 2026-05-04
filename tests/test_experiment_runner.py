"""Tests for research.experiment_runner — universal experiment config."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def test_load_experiment_config_from_yaml(tmp_path: Path):
    """A valid YAML config produces an ExperimentConfig with the right
    structure (conditions, seeds, parallelism, base_args, etc.)."""
    from research.experiment_runner import ExperimentConfig

    yaml_text = textwrap.dedent("""
        name: test-sweep
        runner: research.runners.text_minimal_isolation
        output_dir: /tmp/test-out
        parallelism: 2
        seeds: [42, 43]
        base_args:
          n-events-per-direction: 100
          dt-ms: 1.0
        conditions:
          - name: baseline
            args: {}
          - name: variant_a
            args:
              enable-motor-fs: true
        out_stats_template: "text_eval_{name}_seed{seed}.json"
    """).strip()
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_text)

    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.name == "test-sweep"
    assert cfg.runner == "research.runners.text_minimal_isolation"
    assert cfg.parallelism == 2
    assert cfg.seeds == [42, 43]
    assert cfg.base_args == {"n-events-per-direction": 100, "dt-ms": 1.0}
    assert len(cfg.conditions) == 2
    assert cfg.conditions[0].name == "baseline"
    assert cfg.conditions[0].args == {}
    assert cfg.conditions[1].name == "variant_a"
    assert cfg.conditions[1].args == {"enable-motor-fs": True}


def test_build_cli_args_dash_normalization():
    """Underscore keys become dash CLI flags. Booleans are flag-only."""
    from research.experiment_runner import _build_cli_args

    out = _build_cli_args({
        "n-events-per-direction": 100,
        "dt-ms": 1.0,
        "enable-motor-fs": True,
        "freeze-stdp": False,            # False = omitted
        "missing-arg": None,             # None = omitted
        "underscored_arg": "value",      # underscore -> dash
    })
    # bool-True is flag only (no value follows)
    assert "--enable-motor-fs" in out
    fs_idx = out.index("--enable-motor-fs")
    # Next item should NOT be the value of the bool
    if fs_idx + 1 < len(out):
        assert out[fs_idx + 1].startswith("--") or fs_idx == len(out) - 1
    # False and None are omitted entirely
    assert "--freeze-stdp" not in out
    assert "--missing-arg" not in out
    # int/float values follow the flag
    assert out[out.index("--n-events-per-direction") + 1] == "100"
    assert out[out.index("--dt-ms") + 1] == "1.0"
    # underscores converted
    assert "--underscored-arg" in out
    assert "--underscored_arg" not in out


def test_biology_sweep_yaml_is_valid():
    """The shipped biology_sweep.yaml parses correctly and has all
    4 expected conditions."""
    from research.experiment_runner import ExperimentConfig
    cfg_path = Path(__file__).resolve().parents[1] / "experiments" / "biology_sweep.yaml"
    if not cfg_path.exists():
        pytest.skip(f"biology_sweep.yaml not present at {cfg_path}")

    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.name == "biology-sweep"
    assert len(cfg.seeds) == 6
    cond_names = [c.name for c in cfg.conditions]
    assert set(cond_names) == {"baseline", "fs_only", "topo_only", "topo_fs"}

    # Anti-cheat: topographic factors should be biology-mid (1.5/0.7
    # gives ratio ~2.1x, within Pulvermuller 2001-2003 range of 2-3x).
    topo_only = [c for c in cfg.conditions if c.name == "topo_only"][0]
    assert topo_only.args["topographic-bias-factor"] == 1.5
    assert topo_only.args["off-target-bias-factor"] == 0.7

    # Anti-cheat: FS pool is enabled via boolean (handled in _build_cli_args)
    fs_only = [c for c in cfg.conditions if c.name == "fs_only"][0]
    assert fs_only.args["enable-motor-fs"] is True
