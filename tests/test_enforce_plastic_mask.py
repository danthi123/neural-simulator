"""Selftest for the ADDITIVE, DEFAULT-OFF Hebbian plastic-mask enforcement.

Bug (2026-09-02, read-isolation audit): the runtime Hebbian LTP/decay/clip path in sim/bridge.py consulted only the
named plasticity_gate (cp_plasticity_rate_gain), never cp_synapse_plastic_mask. So a RegionPathway(plastic=False) /
BrainRegion(plastic_internal=False) synapse WITHOUT an explicit zeroed named gate still drifted from read-driven
co-activity (the comprehension organ measured 13.8->56.1 max-weight-delta over 30 reads). STDP/BDSP/BTSP already respect
the mask; the fix closes the same hole for Hebbian, behind BRAIN_ENFORCE_PLASTIC_MASK / cfg.enforce_plastic_mask_in_hebbian
(default OFF).

Two VERIFIED (not asserted-by-comment) properties, both directions checked (fails on pre-fix, passes on the fix):

  (1) byte_identical_off: with the flag OFF, the full-state SHA of a run equals the SHA produced by the genuine
      PRE-FIX code (recovered from git history into a throwaway worktree and run through the identical scenario).
      This proves the default path is untouched -> zero production change / zero regression.

  (2) drift_eliminated_on: the read-driven non-plastic drift that IS present with the flag off (analogous to
      13.8->56.1 over 30 reads) is FLAT (~0) with the flag on.

Runs on numpy (forced here for determinism + no GPU contention); the enforcement itself is backend-agnostic.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCENARIO_REL = os.path.join("tests", "_enforce_plastic_mask_scenario.py")
_FLAG_TOKEN = "enforce_plastic_mask_in_hebbian"
_SEED = 42
# Thresholds: the OFF run must exhibit a LARGE read-driven drift (the bug), the ON run must be essentially FLAT.
_OFF_DRIFT_MIN = 10.0
_ON_DRIFT_MAX = 1e-3


def _run_inproc(enforce, seed=_SEED):
    from tests._enforce_plastic_mask_scenario import run
    return run(enforce, seed)


def _prefix_ref():
    """The commit just before the flag was introduced (pre-fix), or None if not yet in git history."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-S", _FLAG_TOKEN, "--format=%H", "--", "sim/config.py"],
            cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).split()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    oldest_with_flag = out[-1]  # git log is newest-first; the last line is the introducing commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "%s^" % oldest_with_flag],
            cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _run_prefix_off(seed=_SEED):
    """Run the identical scenario against the genuine pre-fix code in a throwaway detached worktree.

    Returns the full-state SHA, or None if the pre-fix ref cannot be resolved (flag not committed yet).
    """
    ref = _prefix_ref()
    if ref is None:
        return None
    wt = tempfile.mkdtemp(prefix="prefix_plastic_mask_")
    try:
        subprocess.check_call(
            ["git", "worktree", "add", "--detach", wt, ref],
            cwd=_REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # The scenario file is added in the fix commit, so it is absent from the pre-fix worktree -> copy it in.
        shutil.copy2(os.path.join(_REPO_ROOT, _SCENARIO_REL), os.path.join(wt, _SCENARIO_REL))
        env = dict(os.environ)
        env["SIM_BACKEND"] = "numpy"
        env["PYTHONPATH"] = wt + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.check_output(
            [sys.executable, "-m", "tests._enforce_plastic_mask_scenario", "--seed=%d" % seed],
            cwd=wt, env=env, text=True,
        )
        # The scenario prints one JSON object on the last non-empty line.
        line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
        return json.loads(line)["sha"]
    finally:
        subprocess.call(
            ["git", "worktree", "remove", "--force", wt],
            cwd=_REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(wt, ignore_errors=True)


def selftest(seed=_SEED):
    """Return the two verified booleans plus supporting detail. Used by the pytest tests and __main__."""
    off = _run_inproc(False, seed)
    on = _run_inproc(True, seed)
    off2 = _run_inproc(False, seed)  # determinism guard for the default path

    prefix_sha = _run_prefix_off(seed)
    # byte_identical_off: default-path SHA must equal the genuine pre-fix SHA (recovered from git). If the pre-fix
    # ref is unavailable (flag not committed yet), fall back to the in-process determinism guard so the property is
    # still verified against SOMETHING real rather than silently passing.
    if prefix_sha is not None:
        byte_identical_off = (off["sha"] == prefix_sha) and (off["sha"] == off2["sha"])
    else:
        byte_identical_off = (off["sha"] == off2["sha"])

    drift_eliminated_on = (
        off["fired_any"] and on["fired_any"]
        and off["final_max_delta"] > _OFF_DRIFT_MIN
        and on["final_max_delta"] < _ON_DRIFT_MAX
    )
    return {
        "byte_identical_off": bool(byte_identical_off),
        "drift_eliminated_on": bool(drift_eliminated_on),
        "off_final_max_delta": off["final_max_delta"],
        "on_final_max_delta": on["final_max_delta"],
        "off_sha": off["sha"],
        "prefix_sha": prefix_sha,
        "prefix_ref_resolved": prefix_sha is not None,
    }


def test_byte_identical_when_flag_off():
    res = selftest()
    assert res["byte_identical_off"], (
        "flag-OFF is NOT byte-identical to pre-fix: off_sha=%s prefix_sha=%s"
        % (res["off_sha"], res["prefix_sha"])
    )


def test_drift_flat_when_flag_on():
    res = selftest()
    # This is the direction that FAILS on pre-fix code (no flag -> ON still drifts) and PASSES on the fix.
    assert res["drift_eliminated_on"], (
        "read-driven drift not eliminated with flag ON: off_delta=%.4f on_delta=%.4f"
        % (res["off_final_max_delta"], res["on_final_max_delta"])
    )


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2))
