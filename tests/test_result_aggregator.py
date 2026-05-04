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
