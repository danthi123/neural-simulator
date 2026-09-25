"""tools/lb_shard.py expected env (B2c, research/findings/2026-09-25-production-default-battery-B2c-paired-flip-
PREREGISTRATION.md). A flip-candidate arm runs with declared BRAIN_* flags in every process's env; under the plain
B2b Amendment 1.2 pin rule each of its cells reads "stray BRAIN_* env key" and is excluded, so such an arm could never
be aggregated verified (gate LBP). `--expect-env` / the EXPECT_ENV.txt that `jobs --pin` writes make those keys
REQUIRED instead of forbidden.

Both directions: a cell whose sidecars carry exactly the expected flags is VALID (test 1); a cell missing one, carrying
a wrong value, or carrying an extra BRAIN_* key is EXCLUDED and named (tests 2-4); with no expected env the rule is
unchanged, so a flag-carrying cell stays invalid (test 5); `jobs --pin --extra-env` records the contract and
`aggregate` reads it back by default, and a base-arm `jobs --pin` records none (tests 6-7); malformed entries raise
(test 8)."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from tools import lb_shard

PIN = "b" * 40
PAIR = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn", "BRAIN_SLEEP_REPLAY_CAPTURE": "1"}


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _prov(extra_env=None):
    env = {"SIM_BACKEND": "numpy"}
    env.update(extra_env or {})
    return {"git_sha": PIN, "git_dirty": False, "source_kind": "git_archive",
            "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True, "env": env}


def _cell(base, tag, seed, fac, lb_env, arm_env):
    cell = os.path.join(base, tag, "s%d" % seed, fac)
    _write(os.path.join(cell, "lb.json"), {"per_faculty": [{"faculty": fac, "kind": "neural-lesion",
                                                             "verdict": "regressed", "load_bearing": True,
                                                             "null_control_clean": True}], "UNRELIABLE": False})
    _write(os.path.join(cell, "lb.json.prov.json"), _prov(lb_env))
    arm = dict(arm_env)
    arm["BRAIN_CHAT_SEED"] = str(seed)
    _write(os.path.join(cell, "intact_a_x_s%d.json" % seed), {"answer": "ok"})
    _write(os.path.join(cell, "intact_a_x_s%d.json.prov.json" % seed), _prov(arm))


def _agg(tmp_path, tag, seeds, expect_env=None, pin=PIN):
    out = str(tmp_path / tag / "aggregate.json")
    a = argparse.Namespace(tag=tag, seeds=seeds, pin=pin, base=str(tmp_path), out=out, expect_env=expect_env)
    lb_shard.cmd_aggregate(a)
    return json.load(open(out))


def _expect_list(d):
    return ["%s=%s" % kv for kv in sorted(d.items())]


def test_flipcand_cell_with_exactly_the_expected_flags_is_valid(tmp_path):
    _cell(str(tmp_path), "flip", 42, "fac-a", PAIR, PAIR)
    r = _agg(tmp_path, "flip", [42], expect_env=_expect_list(PAIR))
    prov = r["provenance"]
    assert prov["status"] == "verified"
    assert (prov["n_valid"], prov["n_invalid"]) == (1, 0)
    assert prov["expect_env"] == PAIR
    assert prov["expect_env_source"] == "--expect-env"
    assert r["per_faculty"]["fac-a"]["n_load_bearing"] == 1


@pytest.mark.parametrize("where", ["lb", "arm"])
def test_missing_expected_flag_is_excluded_and_named(tmp_path, where):
    partial = {k: v for k, v in PAIR.items() if k != "BRAIN_SLEEP_REPLAY_CAPTURE"}
    _cell(str(tmp_path), "flip", 42, "fac-a", partial if where == "lb" else PAIR, partial if where == "arm" else PAIR)
    r = _agg(tmp_path, "flip", [42], expect_env=_expect_list(PAIR))
    fails = r["provenance"]["invalid_cells"]["s42/fac-a"]
    assert any("BRAIN_SLEEP_REPLAY_CAPTURE" in f and "missing" in f for fs in fails.values() for f in fs)
    assert "fac-a" not in r["per_faculty"]  # excluded, never scored 0


def test_wrong_value_is_excluded(tmp_path):
    wall = dict(PAIR, BRAIN_DA_TAG_CAPTURE_CLOCK="wall")
    _cell(str(tmp_path), "flip", 42, "fac-a", wall, wall)
    r = _agg(tmp_path, "flip", [42], expect_env=_expect_list(PAIR))
    fails = r["provenance"]["invalid_cells"]["s42/fac-a"]["lb.json.prov.json"]
    assert any("BRAIN_DA_TAG_CAPTURE_CLOCK" in f and "'wall'" in f for f in fails)


def test_extra_brain_key_beyond_the_expected_set_is_still_stray(tmp_path):
    more = dict(PAIR, BRAIN_AWAKE_REPLAY_CAPTURE="1")   # a separate, still-off route must not ride along unnoticed
    _cell(str(tmp_path), "flip", 42, "fac-a", more, more)
    r = _agg(tmp_path, "flip", [42], expect_env=_expect_list(PAIR))
    fails = r["provenance"]["invalid_cells"]["s42/fac-a"]["lb.json.prov.json"]
    assert any("stray" in f and "BRAIN_AWAKE_REPLAY_CAPTURE" in f for f in fails)


def test_no_expected_env_keeps_the_b2b_rule(tmp_path):
    _cell(str(tmp_path), "flip", 42, "fac-a", PAIR, PAIR)
    _cell(str(tmp_path), "flip", 42, "fac-b", {}, {})
    r = _agg(tmp_path, "flip", [42], expect_env=None)
    prov = r["provenance"]
    assert prov["n_invalid"] == 1 and "s42/fac-a" in prov["invalid_cells"]
    assert "expect_env" not in prov and prov["rule"] == "B2b Amendment 1.2"
    assert "fac-b" in r["per_faculty"]


def test_jobs_pin_records_expected_env_and_aggregate_reads_it(tmp_path, monkeypatch):
    monkeypatch.setattr(lb_shard, "OUT_BASE", str(tmp_path))
    j = argparse.Namespace(seeds=[42], tag="flip", root=None, repeats=2, faculties=["fac-a"],
                           extra_env=_expect_list(PAIR), probe_set="adequate", no_fixes=True, pin=PIN)
    lb_shard.cmd_jobs(j)
    ef = os.path.join(str(tmp_path), "flip", lb_shard.EXPECT_ENV_FILENAME)
    assert open(ef).read() == "".join("%s\n" % s for s in _expect_list(PAIR))
    _cell(str(tmp_path), "flip", 42, "fac-a", PAIR, PAIR)
    r = _agg(tmp_path, "flip", [42], expect_env=None, pin=None)  # no flags: PIN.txt + EXPECT_ENV.txt from jobs
    prov = r["provenance"]
    assert prov["status"] == "verified" and prov["n_invalid"] == 0
    assert prov["expect_env"] == PAIR and prov["expect_env_source"] == "file:%s" % ef


def test_base_arm_jobs_pin_records_no_expected_env_and_clears_a_stale_one(tmp_path, monkeypatch):
    monkeypatch.setattr(lb_shard, "OUT_BASE", str(tmp_path))
    ef = os.path.join(str(tmp_path), "base", lb_shard.EXPECT_ENV_FILENAME)
    os.makedirs(os.path.dirname(ef))
    open(ef, "w").write("BRAIN_DA_TAG_CAPTURE=1\n")
    j = argparse.Namespace(seeds=[42], tag="base", root=None, repeats=2, faculties=["fac-a"], extra_env=None,
                           probe_set="adequate", no_fixes=True, pin=PIN)
    lb_shard.cmd_jobs(j)
    assert not os.path.exists(ef)


@pytest.mark.parametrize("bad", ["LB_PMEM_DRIVE_PROBE=1", "BRAIN_X", "BRAIN_X=", "=1"])
def test_malformed_expected_env_raises(bad):
    with pytest.raises(ValueError):
        lb_shard.parse_expect_env([bad])
