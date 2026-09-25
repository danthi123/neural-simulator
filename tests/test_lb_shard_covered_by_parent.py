"""tools/lb_shard.py's 'covered-by-parent' allowance (2026-09-25, research/oed-provenance-coverage).

THE GAP. research/runners/load_bearing_fraction.py:1190 writes `oed_distributional<seed>.json` into a cell's
out_dir directly from the PARENT process, at a point the provenance hook's argv scan never sees (no --out/--
output/--json names that path). Before research.runners.declare_output existed, that file got no `.prov.json`
sidecar at all -- and `cell_prov_fails` (merged 0d2642ee6) required EVERY non-lb.json file in a cell directory to
carry its own clean sidecar, so a missing one failed the WHOLE cell even though `lb.json.prov.json` itself was a
clean measurement of the pin. Every open-ended-generation cell, on every seed, was excluded for exactly this
reason.

Going forward, `declare_output` closes this at the source (test_runner_provenance_declare_output.py). This file
covers the OTHER half: existing shards predate that fix and their producing code is pinned, so they cannot be
cheaply re-run. `_covered_by_parent_reason` lets a file with no sidecar of its own still count as a clean
measurement of `pin` when it is provably the SAME process's output -- reported as 'covered-by-parent', never
silently folded into a plain pass.

Both directions, per the build spec: a covered file inside the parent's run window passes (tests 1-2); a file
outside the window, not on the allow-list, or riding on an invalid parent sidecar stays invalid (tests 3-6).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from tools import lb_shard

PIN = "b" * 40


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _clean_prov(pin, started):
    return {"git_sha": pin, "git_dirty": False, "source_kind": "git_archive",
            "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True,
            "env": {"SIM_BACKEND": "numpy"}, "started": started}


def _dirty_prov(started):
    return {"git_sha": "short123", "git_dirty": True, "source_kind": None,
            "source_manifest_verified_at_start": None, "source_manifest_verified_at_exit": None,
            "env": {"SIM_BACKEND": "numpy"}, "started": started}


def _lb_report(fac):
    return {"per_faculty": [{"faculty": fac, "kind": "neural-lesion", "verdict": "pass",
                             "load_bearing": True, "null_control_clean": True}], "UNRELIABLE": False}


def _touch(path, mtime):
    open(path, "w").write("{}")
    os.utime(path, (mtime, mtime))


def _agg(tmp_path, tag, seeds, pin, base):
    out = str(tmp_path / "out" / "aggregate.json")
    a = argparse.Namespace(tag=tag, seeds=seeds, pin=pin, base=base, out=out)
    lb_shard.cmd_aggregate(a)
    return json.load(open(out)), out


# --- direction 1: a covered file inside the window is admitted, reported, never silent -----------------------

def test_oed_file_with_no_sidecar_inside_parent_window_is_covered(tmp_path, capsys):
    cell = os.path.join(str(tmp_path), "t", "s42", "open-ended-generation")
    start = time.time() - 100.0
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
    _write(os.path.join(cell, "lb.json"), _lb_report("open-ended-generation"))
    _write(os.path.join(cell, "lb.json.prov.json"), _clean_prov(PIN, started_str))
    # lb.json.prov.json's own mtime (written by _stamp_outputs at the parent's exit) is "now" by default
    os.utime(os.path.join(cell, "lb.json.prov.json"), (start + 50.0, start + 50.0))
    # oed_distributional.json: written mid-run, no sidecar of its own
    _touch(os.path.join(cell, "oed_distributional.json"), start + 10.0)

    result, _ = _agg(tmp_path, "t", [42], pin=PIN, base=str(tmp_path))

    prov = result["provenance"]
    assert prov["status"] == "verified"
    assert prov["n_invalid"] == 0
    assert prov["n_covered_by_parent"] == 1
    assert "s42/open-ended-generation" in prov["covered_by_parent_cells"]
    assert "oed_distributional.json" in prov["covered_by_parent_cells"]["s42/open-ended-generation"]
    # covered means VALID, not excluded: the faculty is present and counted
    assert "open-ended-generation" in result["per_faculty"]
    assert result["per_faculty"]["open-ended-generation"]["n_load_bearing"] == 1

    printed = capsys.readouterr().out
    assert "n_covered_by_parent=1" in printed
    assert "COVERED-BY-PARENT s42/open-ended-generation" in printed
    assert "oed_distributional.json" in printed


def test_seed_suffixed_oed_filename_matches_the_allowlist(tmp_path):
    cell = os.path.join(str(tmp_path), "t", "s43", "open-ended-generation")
    start = time.time() - 100.0
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
    _write(os.path.join(cell, "lb.json"), _lb_report("open-ended-generation"))
    _write(os.path.join(cell, "lb.json.prov.json"), _clean_prov(PIN, started_str))
    os.utime(os.path.join(cell, "lb.json.prov.json"), (start + 50.0, start + 50.0))
    _touch(os.path.join(cell, "oed_distributional_s43.json"), start + 10.0)  # _seed_suffix(43) == "_s43"

    result, _ = _agg(tmp_path, "t", [43], pin=PIN, base=str(tmp_path))

    assert result["provenance"]["n_invalid"] == 0
    assert result["provenance"]["n_covered_by_parent"] == 1


# --- direction 2: outside the window / off the allow-list / unclean parent all stay invalid ---------------------

def test_oed_file_written_after_parent_exit_is_not_covered(tmp_path):
    """A file that postdates lb.json.prov.json's own write cannot have been written by that same run -- an
    unrelated LATER process reusing the directory must not be laundered through 'covered-by-parent'."""
    cell = os.path.join(str(tmp_path), "t", "s42", "open-ended-generation")
    start = time.time() - 100.0
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
    _write(os.path.join(cell, "lb.json"), _lb_report("open-ended-generation"))
    _write(os.path.join(cell, "lb.json.prov.json"), _clean_prov(PIN, started_str))
    sidecar_mtime = start + 50.0
    os.utime(os.path.join(cell, "lb.json.prov.json"), (sidecar_mtime, sidecar_mtime))
    _touch(os.path.join(cell, "oed_distributional.json"), sidecar_mtime + 30.0)  # well after, outside slack

    result, _ = _agg(tmp_path, "t", [42], pin=PIN, base=str(tmp_path))

    prov = result["provenance"]
    assert prov["n_covered_by_parent"] == 0
    assert prov["n_invalid"] == 1
    assert "open-ended-generation" not in result["per_faculty"]


def test_filename_off_the_allowlist_is_not_covered(tmp_path):
    cell = os.path.join(str(tmp_path), "t", "s42", "some-other-faculty")
    start = time.time() - 100.0
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
    _write(os.path.join(cell, "lb.json"), _lb_report("some-other-faculty"))
    _write(os.path.join(cell, "lb.json.prov.json"), _clean_prov(PIN, started_str))
    os.utime(os.path.join(cell, "lb.json.prov.json"), (start + 50.0, start + 50.0))
    _touch(os.path.join(cell, "unexpected_side_output.json"), start + 10.0)  # not on the allow-list

    result, _ = _agg(tmp_path, "t", [42], pin=PIN, base=str(tmp_path))

    assert result["provenance"]["n_covered_by_parent"] == 0
    assert result["provenance"]["n_invalid"] == 1
    assert "some-other-faculty" not in result["per_faculty"]


def test_oed_file_riding_on_a_dirty_parent_sidecar_is_not_covered(tmp_path):
    """lb.json.prov.json itself fails the pin rule -- an unclean parent proves nothing about what it wrote, so its
    oed_distributional.json (also missing its own sidecar) must NOT be laundered through covered-by-parent."""
    cell = os.path.join(str(tmp_path), "t", "s102", "open-ended-generation")
    start = time.time() - 100.0
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
    _write(os.path.join(cell, "lb.json"), _lb_report("open-ended-generation"))
    _write(os.path.join(cell, "lb.json.prov.json"), _dirty_prov(started_str))
    os.utime(os.path.join(cell, "lb.json.prov.json"), (start + 50.0, start + 50.0))
    _touch(os.path.join(cell, "oed_distributional_s102.json"), start + 10.0)  # otherwise-clean window + allow-list

    result, _ = _agg(tmp_path, "t", [102], pin=PIN, base=str(tmp_path))

    prov = result["provenance"]
    assert prov["n_covered_by_parent"] == 0
    assert prov["n_invalid"] == 1
    fails = prov["invalid_cells"]["s102/open-ended-generation"]
    assert "lb.json.prov.json" in fails  # the dirty parent itself is named, not silently absorbed
    assert "open-ended-generation" not in result["per_faculty"]


def test_pool_worker_started_in_a_different_timezone_is_still_covered(tmp_path, monkeypatch):
    """Regression for the real B2a data (2026-09-25): a pool/cloud worker records `started` in ITS OWN local
    timezone, which need not match the machine running `aggregate`. Re-parsing "started" with ONLY
    `time.mktime()` (this machine's local zone) on a host several hours off from the worker's zone previously
    computed a start AFTER the sidecar's own exit-time mtime, so `_parent_run_window`'s exit-before-start sanity
    check rejected every real cell outright, before the window-membership check ever ran (observed: a pool
    worker's UTC "started" re-read as this machine's America/New_York EDT, a 4h swing).

    Deterministic regardless of the TEST HOST's own timezone: `time.mktime` is monkeypatched to return a
    deliberately too-late value (simulating "this machine's local interpretation disagrees with the recording
    host's"), while `calendar.timegm` is left real. `_parent_run_window` must still recover a valid window by
    taking the min() of the two candidate interpretations."""
    cell = os.path.join(str(tmp_path), "t", "s42", "open-ended-generation")
    real_exit = time.time() - 50.0     # the true instant lb.json.prov.json was written (this filesystem's clock)
    real_start = real_exit - 55.0      # ~55s run, truly BEFORE the exit
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(real_start))  # the recording host's own rendering

    real_mktime = time.mktime
    def _wrong_local_interpretation(struct):
        return real_mktime(struct) + 4 * 3600.0  # simulate a 4h TZ disagreement, always LATER than the true start
    monkeypatch.setattr(lb_shard.time, "mktime", _wrong_local_interpretation)

    _write(os.path.join(cell, "lb.json"), _lb_report("open-ended-generation"))
    _write(os.path.join(cell, "lb.json.prov.json"), _clean_prov(PIN, started_str))
    os.utime(os.path.join(cell, "lb.json.prov.json"), (real_exit, real_exit))
    _touch(os.path.join(cell, "oed_distributional.json"), real_exit - 1.0)  # written just before the sidecar

    result, _ = _agg(tmp_path, "t", [42], pin=PIN, base=str(tmp_path))

    prov = result["provenance"]
    assert prov["n_invalid"] == 0, prov.get("invalid_cells")
    assert prov["n_covered_by_parent"] == 1
    assert "open-ended-generation" in result["per_faculty"]


def test_direct_unit_no_lb_prov_at_all_yields_no_reason(tmp_path):
    """Unit-level check on _covered_by_parent_reason itself: no lb.json.prov.json to read at all (e.g. it was
    deleted, or the glob raced a write) must return None, not raise."""
    cell = os.path.join(str(tmp_path), "cell")
    os.makedirs(cell)
    _touch(os.path.join(cell, "oed_distributional.json"), time.time())
    missing_prov = os.path.join(cell, "lb.json.prov.json")

    reason = lb_shard._covered_by_parent_reason(cell, "oed_distributional.json", missing_prov, lb_fails={})

    assert reason is None
