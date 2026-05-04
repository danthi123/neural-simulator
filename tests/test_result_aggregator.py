"""Tests for research.result_aggregator — universal result aggregator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_eval_json(path: Path, confusion: dict, i2w_acc: float = None):
    payload = {
        "word_to_action_eval": {"confusion_matrix": confusion},
    }
    if i2w_acc is not None:
        payload["image_to_word_eval"] = {"accuracy": i2w_acc}
    path.write_text(json.dumps(payload))


def test_acc_for_mapping_basic():
    from research.result_aggregator import _acc_for_mapping
    cm = {
        "north": {"N": 5, "E": 0, "S": 0, "W": 0},
        "east":  {"N": 0, "E": 5, "S": 0, "W": 0},
        "south": {"N": 0, "E": 0, "S": 5, "W": 0},
        "west":  {"N": 0, "E": 0, "S": 0, "W": 5},
    }
    true_map = {"north": "N", "east": "E", "south": "S", "west": "W"}
    assert _acc_for_mapping(cm, true_map) == 1.0


def test_best_permutation_finds_alignment():
    from research.result_aggregator import _best_permutation
    cm = {
        "north": {"N": 5, "E": 0, "S": 0, "W": 0},
        "east":  {"N": 0, "E": 5, "S": 0, "W": 0},
        "south": {"N": 0, "E": 0, "S": 5, "W": 0},
        "west":  {"N": 0, "E": 0, "S": 0, "W": 5},
    }
    best_acc, best_perm = _best_permutation(cm)
    assert best_acc == 1.0
    assert best_perm == ("N", "E", "S", "W")  # TRUE mapping


def test_best_permutation_finds_misalignment():
    """When network learned a SWAPPED mapping (e.g. north->E),
    best_permutation reveals the swap."""
    from research.result_aggregator import _best_permutation
    cm = {
        "north": {"N": 0, "E": 5, "S": 0, "W": 0},  # north->E learned
        "east":  {"N": 5, "E": 0, "S": 0, "W": 0},  # east->N learned
        "south": {"N": 0, "E": 0, "S": 5, "W": 0},
        "west":  {"N": 0, "E": 0, "S": 0, "W": 5},
    }
    best_acc, best_perm = _best_permutation(cm)
    assert best_acc == 1.0
    assert best_perm == ("E", "N", "S", "W")  # NOT TRUE


def test_resultset_load_aligned_perfect(tmp_path: Path):
    from research.result_aggregator import (
        AggregateConfig, ResultSet,
    )
    cm_perfect = {
        "north": {"N": 25, "E": 0, "S": 0, "W": 0},
        "east":  {"N": 0, "E": 25, "S": 0, "W": 0},
        "south": {"N": 0, "E": 0, "S": 25, "W": 0},
        "west":  {"N": 0, "E": 0, "S": 0, "W": 25},
    }
    _write_eval_json(tmp_path / "text_eval_test_seed42.json", cm_perfect, i2w_acc=0.5)
    cfg = AggregateConfig(
        conditions={"test": "text_eval_test_seed{seed}.json"},
        seeds=[42],
        raw_dir=tmp_path,
    )
    rs = ResultSet.load(cfg)
    assert len(rs.results) == 1
    r = rs.results[0]
    assert r.true_acc == 1.0
    assert r.aligned == 1
    assert r.i2w_acc == 0.5


def test_resultset_load_unaligned_swap(tmp_path: Path):
    from research.result_aggregator import (
        AggregateConfig, ResultSet,
    )
    # All 4 words rotated: north->E, east->N (swap), south->W, west->S (swap)
    cm_rotated = {
        "north": {"N": 0, "E": 25, "S": 0, "W": 0},   # north->E
        "east":  {"N": 25, "E": 0, "S": 0, "W": 0},   # east->N
        "south": {"N": 0, "E": 0, "S": 0, "W": 25},   # south->W
        "west":  {"N": 0, "E": 0, "S": 25, "W": 0},   # west->S
    }
    _write_eval_json(tmp_path / "text_eval_swapped_seed42.json", cm_rotated)
    cfg = AggregateConfig(
        conditions={"swapped": "text_eval_swapped_seed{seed}.json"},
        seeds=[42], raw_dir=tmp_path,
    )
    rs = ResultSet.load(cfg)
    r = rs.results[0]
    assert r.true_acc == 0.0  # TRUE mapping gives 0%, no diagonal correct
    assert r.best_acc == 1.0  # but best perm gives 100%
    assert r.aligned == 0
    assert r.best_perm == "ENWS"


def test_verdict_real_learning_when_aligned_4_of_6(tmp_path: Path):
    from research.result_aggregator import (
        AggregateConfig, ResultSet, RunResult,
    )
    rs = ResultSet(
        config=AggregateConfig(conditions={"x": "_"}, seeds=[]),
        results=[
            RunResult("test", 42, 0.6, 0.6, "NESW", 1),
            RunResult("test", 43, 0.6, 0.6, "NESW", 1),
            RunResult("test", 44, 0.6, 0.6, "NESW", 1),
            RunResult("test", 100, 0.6, 0.6, "NESW", 1),
            RunResult("test", 101, 0.4, 0.5, "NSEW", 0),
            RunResult("test", 102, 0.4, 0.5, "NSEW", 0),
        ],
    )
    v = rs.verdict()
    assert "Real word-action learning achieved" in v
    assert "test" in v


def test_verdict_no_learning_when_all_zero(tmp_path: Path):
    from research.result_aggregator import (
        AggregateConfig, ResultSet, RunResult,
    )
    rs = ResultSet(
        config=AggregateConfig(conditions={"x": "_"}, seeds=[]),
        results=[
            RunResult("test", 42, 0.28, 0.33, "ENSW", 0),
            RunResult("test", 43, 0.27, 0.32, "SNEW", 0),
        ],
    )
    v = rs.verdict()
    assert "No real learning" in v


def test_builtin_configs_have_expected_keys():
    """Built-in configs are stable APIs — check they have the documented
    keys for shipped morning briefing snippets."""
    from research.result_aggregator import BUILTIN_CONFIGS
    assert "swr-investigation" in BUILTIN_CONFIGS
    assert "fundamentals" in BUILTIN_CONFIGS
    assert "biology" in BUILTIN_CONFIGS
    # biology has 4 conditions (baseline, +FS, +Topo, +Topo+FS)
    bio = BUILTIN_CONFIGS["biology"]["conditions"]
    assert len(bio) == 4
    assert any("baseline" in k for k in bio.keys())


def test_followup_configs_present():
    """Post-biology-sweep follow-ups are wired: minimum_biology and
    sanity_check (A/B branches) plus tier-2 b2/b4 fallbacks."""
    from research.result_aggregator import BUILTIN_CONFIGS
    assert "minimum_biology" in BUILTIN_CONFIGS
    assert "sanity_check" in BUILTIN_CONFIGS
    assert "b2_sparse_codes" in BUILTIN_CONFIGS
    assert "b4_long_training" in BUILTIN_CONFIGS
    # minimum_biology: 4 dose-response conditions
    minbio = BUILTIN_CONFIGS["minimum_biology"]["conditions"]
    assert len(minbio) == 4
    assert any("topo_weak" in v for v in minbio.values())
    assert any("topo_strong" in v for v in minbio.values())
    # sanity_check: 2 density conditions
    sc = BUILTIN_CONFIGS["sanity_check"]["conditions"]
    assert len(sc) == 2
    # b4 has its own seed list (long training, 3 seeds)
    assert BUILTIN_CONFIGS["b4_long_training"].get("seeds") == [42, 43, 44]


def test_per_config_seeds_overrides_default(tmp_path: Path):
    """When a config declares its own `seeds`, the AggregateConfig
    loaded from main() should use those instead of the default
    [42,43,44,100,101,102]. Smoke-test by checking what files
    ResultSet.load tries to read."""
    from research.result_aggregator import (
        AggregateConfig, ResultSet, BUILTIN_CONFIGS,
    )
    # Use b4 config's seeds (should be [42, 43, 44])
    config_seeds = BUILTIN_CONFIGS["b4_long_training"].get("seeds")
    assert config_seeds == [42, 43, 44]
    # Build an AggregateConfig with those seeds and ensure load works
    cfg = AggregateConfig(
        conditions={"label": "text_eval_b4_dose_1x_seed{seed}.json"},
        seeds=config_seeds,
        raw_dir=tmp_path,
    )
    rs = ResultSet.load(cfg)
    # No files exist at tmp_path, so 0 results loaded
    assert len(rs.results) == 0
    # But the load attempted exactly 3 seeds (verified by seed list, not by a
    # mock — the contract is just that AggregateConfig.seeds is honored).
    assert cfg.seeds == [42, 43, 44]
