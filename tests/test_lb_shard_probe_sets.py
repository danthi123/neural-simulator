"""tools/lb_shard.py job env: the historical default is unchanged, and --probe-set thin / --no-fixes drop exactly the
flags they name (the 2026-09-23 production-default validation needs a battery with NO fix and NO probe flags)."""
from __future__ import annotations

import argparse
import contextlib
import io

from tools import lb_shard


def _jobs(**kw):
    a = argparse.Namespace(seeds=[42], tag="t_unit", root=None, repeats=2, faculties=["episodic-memory"],
                           extra_env=None, probe_set="adequate", no_fixes=False)
    for k, v in kw.items():
        setattr(a, k, v)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lb_shard.cmd_jobs(a)
    return buf.getvalue().strip().splitlines()


def test_default_env_is_the_historical_allfixes_env():
    assert lb_shard.job_env() == lb_shard.ENV
    assert set(lb_shard.FIX_ENV) <= set(lb_shard.ENV)
    assert set(lb_shard.PROBE_SETS["adequate"]) <= set(lb_shard.ENV)
    (line,) = _jobs()
    for k in lb_shard.ENV:
        assert f"{k}=1" in line, k


def test_no_fixes_drops_every_fix_flag_keeps_probes():
    envd = lb_shard.job_env(fixes=False)
    assert not set(envd) & set(lb_shard.FIX_ENV)
    assert envd == lb_shard.PROBE_SETS["adequate"]
    (line,) = _jobs(no_fixes=True)
    for k in lb_shard.FIX_ENV:
        assert k not in line, k
    assert "LB_EPISODIC_DRIVE_PROBE=1" in line


def test_thin_no_fixes_passes_no_brain_or_probe_flag():
    envd = lb_shard.job_env("thin", fixes=False)
    assert envd == {}
    (line,) = _jobs(no_fixes=True, probe_set="thin")
    assert "BRAIN_" not in line and "LB_" not in line, line
    assert "--only episodic-memory --seed 42" in line
    assert "research/findings/raw/_load_bearing/_shards/t_unit/s42/episodic-memory/lb.json" in line


def test_extra_env_still_layers_on_top():
    envd = lb_shard.job_env("thin", fixes=False, extra_env=["BRAIN_X=0"])
    assert envd == {"BRAIN_X": "0"}
