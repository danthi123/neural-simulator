"""CPU tests for research/runners/c1_flip_dev_check.py (the C1 flip dev bar). No brain is built."""
import json
import os

import pytest

from research.runners import c1_flip_dev_check as F


def test_selftest_passes():
    assert F.selftest(verbose=False)


def _write(root, seed, on, off):
    for side, cells in (("on", on), ("off", off)):
        for g, shard in cells.items():
            p = F.shard_path(root, seed, side, g)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(shard, f)


def test_score_reads_files_and_passes_a_perfect_brain(tmp_path):
    for s in F.DEV_SEEDS:
        _write(str(tmp_path), s, *F.synthetic_seed("perfect"))
    out = F.score(str(tmp_path))
    assert out["verdict"] == "PASS" and out["n_pass"] == 8
    assert out["bridge_summary"]["n_defined"] == 8          # the bridge group is in the bar since the hop fix


def test_one_moat_leak_on_one_seed_fails(tmp_path):
    for s in F.DEV_SEEDS:
        _write(str(tmp_path), s, *F.synthetic_seed("leak_moat" if s == 4 else "perfect"))
    out = F.score(str(tmp_path))
    assert out["verdict"] == "FAIL"
    assert any("seed 4: word/c_otter_chase" in w for w in out["why"])


def test_missing_seed_is_undefined_not_fail(tmp_path):
    for s in F.DEV_SEEDS[:-1]:
        _write(str(tmp_path), s, *F.synthetic_seed("perfect"))
    out = F.score(str(tmp_path))
    assert out["verdict"] == "UNDEFINED" and out["undefined_seeds"] == [8]


def test_answer_text_noise_alone_is_not_a_violation():
    on, off = F.synthetic_seed("perfect")
    on["word"]["responses"]["c_dog_chase"]["answer"] = "On dog, then -- x"
    assert F.seed_verdict(on, off)["pass"] is True


def test_gate_seeds_are_refused():
    with pytest.raises(SystemExit):
        F.pool_job_lines("0" * 40, seeds=(1, 42))
    with pytest.raises(SystemExit):
        F.run_shard(100, "on", "word", "/nonexistent/x.json")


def test_bridge_turns_extend_the_c1_context_and_the_gate_table_is_untouched():
    labels = [l for (l, _m, _r) in F.CAT_BRIDGE_TURNS]
    assert "cat_bridge" not in F.SG.GROUPS                  # registered only inside a worker process
    assert labels[:10] == [l for (l, _m, _r) in F.SG._CAT_CONTEXT]
    assert {"t_bat_eat", "t_bat_fly", "c_canary_swim"} <= set(labels)


def test_the_bridge_is_in_the_bar_and_every_line_runs_the_hop_fix(tmp_path):
    """RULING c1-hop-first: the flip dev check runs WITH the hop fix; lake for the canary (two hops) fails a seed,
    a flagged lake for the bat (one hop) is allowed, and a cell without the hop env is UNDEFINED."""
    for s in F.DEV_SEEDS:
        _write(str(tmp_path), s, *F.synthetic_seed("bridge_lake" if s == 2 else "perfect_bat_lake"))
    out = F.score(str(tmp_path))
    assert out["verdict"] == "FAIL" and out["n_pass"] == 7
    assert any("seed 2: cat_bridge/c_canary_swim" in w for w in out["why"]), out["why"]
    assert "cat_bridge" in F.BAR_GROUPS and F.ALLOWED[("cat_bridge", "c_bat_swim")] == {"lake"}
    assert F.seed_verdict(*F.synthetic_seed("hop_off"))["defined"] is False
    lines = F.pool_job_lines("a" * 40)
    assert len(lines) == 96 and all("--group" in c for c, _w in lines)
    assert all("BRAIN_CATEGORY_INHERIT_HOP=1" in w for _c, w in lines)


def test_run_shard_sets_the_hop_env_on_both_sides(monkeypatch, tmp_path):
    seen = []

    class _P:
        returncode = 1

    def _fake_run(cmd, env=None, cwd=None):
        seen.append((env.get(F.FLAG), env.get(F.HOP_FLAG)))
        return _P()

    monkeypatch.setattr(F.subprocess, "run", _fake_run)
    for side in ("on", "off"):
        F.run_shard(3, side, "word", str(tmp_path / ("%s.json" % side)))
    assert seen == [("1", "1"), ("0", "1")]
