"""tools/lb_shard.py aggregate provenance gate (2026-09-25). B2b Amendment 1.2's per-cell validity rule
(research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md) says a (faculty, seed) cell is a
measurement of a battery's registered pin only if EVERY sidecar in its shard directory records `git_sha` == the
pin IN FULL, `source_kind` "git_archive", both manifest-verified flags true, `env.SIM_BACKEND` "numpy", and no
stray `BRAIN_*` key. `lb_shard.py aggregate` never checked it: research/findings/2026-09-25-production-default-
battery-B2a-FAIL.md found 23/28 coverable faculties' seed-102 cell had run off a dirty local worktree (short
git_sha, source_kind null) and the aggregate folded them in as clean because it never looked
(research/FAILURE_LOG.md's B2a row declared this "NOT-GATEABLE as a repo-wide pre-commit check" — the fix belongs
in the tool that reads per-cell provenance, not a generic diff scanner).

Both directions: an off-pin/dirty cell is EXCLUDED and reported with its failing fields, never counted, never
scored 0 (test 1); a clean tree pinned aggregates BYTE-IDENTICAL to the unpinned aggregate apart from the new
`provenance` block (test 2). Tests 3-5 cover the unverified-by-default path, the PIN.txt file `jobs --pin` writes
and `aggregate` reads back, and a missing (not just mismatched) sidecar."""
from __future__ import annotations

import argparse
import json
import os

from tools import lb_shard

PIN = "a" * 40  # syntactically a full git SHA; the fixtures never touch a real repo


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _clean_prov(pin):
    return {"git_sha": pin, "git_dirty": False, "source_kind": "git_archive",
            "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True,
            "env": {"SIM_BACKEND": "numpy"}}


def _dirty_prov():
    # the exact B2a shape: short SHA, no source_kind, no manifest verification, dirty worktree
    return {"git_sha": "9db761329", "git_dirty": True, "source_kind": None,
            "source_manifest_verified_at_start": None, "source_manifest_verified_at_exit": None,
            "env": {"SIM_BACKEND": "numpy"}}


def _lb_report(fac, verdict="pass", load_bearing=True, kind="neural-lesion"):
    return {"per_faculty": [{"faculty": fac, "kind": kind, "verdict": verdict, "load_bearing": load_bearing,
                             "null_control_clean": True}], "UNRELIABLE": False}


def _build_cell(base, tag, seed, fac, prov_factory, arm_prov_factory=None):
    """One (faculty, seed) shard directory: lb.json + its sidecar, plus one intact-arm file + its own sidecar."""
    cell = os.path.join(base, tag, "s%d" % seed, fac)
    _write(os.path.join(cell, "lb.json"), _lb_report(fac))
    _write(os.path.join(cell, "lb.json.prov.json"), prov_factory())
    arm = (arm_prov_factory or prov_factory)()
    arm.setdefault("env", {})["BRAIN_CHAT_SEED"] = str(seed)  # allowed on an ARM sidecar, never on lb.json's own
    _write(os.path.join(cell, "intact_a_x_s%d.json" % seed), {"answer": "ok"})
    _write(os.path.join(cell, "intact_a_x_s%d.json.prov.json" % seed), arm)
    return cell


def _agg(tmp_path, tag, seeds, pin=None, base=None):
    out = str(tmp_path / tag / "aggregate.json")
    a = argparse.Namespace(tag=tag, seeds=seeds, pin=pin, base=base or str(tmp_path), out=out)
    lb_shard.cmd_aggregate(a)
    return json.load(open(out)), out


# --- direction 1: an off-pin / dirty cell is EXCLUDED and reported -----------------------------------------

def test_off_pin_dirty_cell_is_excluded_and_reported(tmp_path, capsys):
    tag = "mixed"
    _build_cell(str(tmp_path), tag, 42, "clean-faculty", lambda: _clean_prov(PIN))
    _build_cell(str(tmp_path), tag, 42, "dirty-faculty", _dirty_prov)

    result, _ = _agg(tmp_path, tag, [42], pin=PIN)

    prov = result["provenance"]
    assert prov["status"] == "verified"
    assert prov["n_cells_checked"] == 2
    assert prov["n_valid"] == 1
    assert prov["n_invalid"] == 1
    assert "s42/dirty-faculty" in prov["invalid_cells"]
    fails = prov["invalid_cells"]["s42/dirty-faculty"]["lb.json.prov.json"]
    assert any("git_sha" in f for f in fails)
    assert any("source_kind" in f for f in fails)

    # never counted, never scored 0: the dirty faculty is ABSENT, not present with n_load_bearing == 0
    assert "dirty-faculty" not in result["per_faculty"]
    assert "dirty-faculty" not in result["robust_core"]
    assert result["per_faculty"]["clean-faculty"]["n_load_bearing"] == 1
    assert result["per_seed"]["42"]["n_coverable_present"] == 1  # only the clean faculty counted
    assert result["union_n"] == 1

    printed = capsys.readouterr().out
    assert "n_valid=1 n_invalid=1" in printed
    assert "dirty-faculty" in printed  # per-cell validity summary is printed, not just written to disk


def test_missing_sidecar_is_reported_as_missing_not_silently_skipped(tmp_path):
    tag = "missingsidecar"
    cell = os.path.join(str(tmp_path), tag, "s42", "fac-a")
    _write(os.path.join(cell, "lb.json"), _lb_report("fac-a"))
    # deliberately NO lb.json.prov.json at all

    result, _ = _agg(tmp_path, tag, [42], pin=PIN)

    assert result["provenance"]["n_invalid"] == 1
    assert result["provenance"]["invalid_cells"]["s42/fac-a"] == {"lb.json.prov.json": ["missing"]}
    assert "fac-a" not in result["per_faculty"]


# --- direction 2: a clean tree aggregates exactly as before -------------------------------------------------

def test_clean_tree_aggregates_byte_identical_apart_from_provenance_block(tmp_path):
    tag = "clean"
    _build_cell(str(tmp_path), tag, 42, "fac-a", lambda: _clean_prov(PIN))
    _build_cell(str(tmp_path), tag, 43, "fac-a", lambda: _clean_prov(PIN))

    no_pin, _ = _agg(tmp_path, tag, [42, 43], pin=None)
    with_pin, _ = _agg(tmp_path, tag, [42, 43], pin=PIN)

    assert with_pin["provenance"]["status"] == "verified"
    assert with_pin["provenance"]["n_invalid"] == 0
    assert no_pin["provenance"]["status"] == "unverified"

    no_pin.pop("provenance")
    with_pin.pop("provenance")
    assert no_pin == with_pin
    # and the run actually measured something, so this isn't a vacuous "two empty dicts are equal" pass
    assert with_pin["robust_core_n"] == 1
    assert with_pin["per_faculty"]["fac-a"]["n_load_bearing"] == 2


def test_no_pin_and_none_recorded_marks_unverified_loudly(tmp_path, capsys):
    tag = "unpinned"
    _build_cell(str(tmp_path), tag, 42, "fac-a", lambda: _clean_prov(PIN))

    result, _ = _agg(tmp_path, tag, [42], pin=None)

    assert result["provenance"] == {
        "status": "unverified", "pin": None, "pin_source": None,
        "warning": result["provenance"]["warning"],  # message text checked below, not pinned verbatim here
    }
    assert "not checked" in result["provenance"]["warning"].lower()
    err = capsys.readouterr().err
    assert "unverified" in err
    assert "PIN.txt" in err
    # unverified is not silent about the numbers it did NOT check, either
    assert result["robust_core_n"] == 1


def test_pin_recorded_by_jobs_is_read_as_default_by_aggregate(tmp_path, monkeypatch):
    tag = "recorded"
    monkeypatch.setattr(lb_shard, "OUT_BASE", str(tmp_path))
    j = argparse.Namespace(seeds=[42], tag=tag, root=None, repeats=2, faculties=["fac-a"], extra_env=None,
                           probe_set="thin", no_fixes=True, pin=PIN)
    lb_shard.cmd_jobs(j)

    pin_path = os.path.join(str(tmp_path), tag, lb_shard.PIN_FILENAME)
    assert os.path.exists(pin_path)
    assert open(pin_path).read().strip() == PIN

    _build_cell(str(tmp_path), tag, 42, "fac-a", lambda: _clean_prov(PIN))
    result, _ = _agg(tmp_path, tag, [42], pin=None, base=str(tmp_path))  # no --pin: must fall back to the file

    assert result["provenance"]["status"] == "verified"
    assert result["provenance"]["pin"] == PIN
    assert result["provenance"]["pin_source"] == "file:%s" % pin_path


def test_jobs_without_pin_writes_no_pin_file(tmp_path, monkeypatch):
    tag = "nopin"
    monkeypatch.setattr(lb_shard, "OUT_BASE", str(tmp_path))
    j = argparse.Namespace(seeds=[42], tag=tag, root=None, repeats=2, faculties=["fac-a"], extra_env=None,
                           probe_set="thin", no_fixes=True, pin=None)
    lb_shard.cmd_jobs(j)
    assert not os.path.exists(os.path.join(str(tmp_path), tag, lb_shard.PIN_FILENAME))
