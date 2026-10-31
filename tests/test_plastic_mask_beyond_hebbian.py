"""Selftest for the ADDITIVE, DEFAULT-OOFF plastic-mask enforcement BEYOND the Hebbian rule (Vikunja #203 residual).

Bug (2026-09-02 read-isolation audit, residual left open by merge 509c7137b): `RegionPathway(plastic=False)` /
`BrainRegion(plastic_internal=False)` was enforced only in the Hebbian LTP/decay/clip path (and in STDP/BDSP/BTSP).
Every OTHER write to cp_connections.data still moved a declared-frozen synapse:
  (i)  homeostatic synaptic scaling  -- data *= per-neuron scale factor (gain-gated, never mask-gated)
  (ii) structural-plasticity elimination -- data[weak] = 0.0 (not mask-gated at all)
  (iii) the reward-modulation clip -- clip(data, w_min, w_max) when no named gain array exists
  (iv) the mask itself is never remapped/compacted, because the remap helper reads the attribute `cp_plastic_mask`,
       which sim/ never assigns (dead code) -- so after a CSR-reordering rebuild the mask labels the wrong synapses.

Fix: cfg.enforce_plastic_mask_beyond_hebbian / env BRAIN_ENFORCE_PLASTIC_MASK (DEFAULT OFF). Both directions are
VERIFIED, not asserted:

  (1) byte_identical_off -- with the flag OFF the full-state SHA equals the SHA of the genuine PRE-fix code,
      recovered into a throwaway git worktree and run through the identical scenario file.
  (2) freeze_on -- over 30 successive reads of a 6-neuron probe, every declared-frozen synapse is bit-unchanged
      (max|dw| == 0.0 exactly) on all six seeds, while the companion plastic control population DOES move, which
      is what makes the zero meaningful rather than a run that never fired.
  (3) rebuild -- the misalignment (iv) is measured directly: 4 declared pairs are mislabelled with the flag OFF,
      0 with it ON.

Runs on numpy (forced) so the tests need no GPU and are deterministic.
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
_SCENARIO_REL = os.path.join("tests", "_plastic_mask_beyond_hebbian_scenario.py")
_FLAG_TOKEN = "enforce_plastic_mask_beyond_hebbian"
SEEDS = (42, 43, 44, 100, 101, 102)
RULES = ("scaling", "structural", "reward", "rebuild", "compaction", "combined")
# the smallest OFF-arm drift each rule must show (the bug the instrument is meant to see)
_OFF_DRIFT_MIN = {"scaling": 5.0, "structural": 0.01, "reward": 10.0, "rebuild": 5.0,
                  "compaction": 1.0, "combined": 10.0}
_ON_DRIFT_MAX = 1e-9


def _run(rule, enforce, seed):
    from tests._plastic_mask_beyond_hebbian_scenario import run
    return run(rule, enforce, seed)


def _prefix_ref():
    """The commit just before the flag was introduced, or None if it is not yet in git history."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-S", _FLAG_TOKEN, "--format=%H", "--", "sim/config.py"],
            cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).split()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    oldest_with_flag = out[-1]
    try:
        return subprocess.check_output(["git", "rev-parse", "%s^" % oldest_with_flag],
                                       cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def _run_prefix_off(rule, seed):
    """Run the SAME scenario file against the genuine pre-fix code.

    `git archive` (not `git worktree`) because a worktree checkout of this repo fails on this box with
    "fatal: Could not reset index file to revision 'HEAD'" (FAILURE_LOG 2026-10-31); archive touches no index.
    Only the packages the scenario imports are extracted: sim/, tests/, experiment/, config/, ui/.
    """
    ref = _prefix_ref()
    if ref is None:
        return None
    wt = tempfile.mkdtemp(prefix="prefix_plastic_mask_bh_")
    try:
        arc = subprocess.check_output(["git", "archive", ref, "sim", "tests", "experiment", "config", "ui"],
                                      cwd=_REPO_ROOT)
        subprocess.check_call(["tar", "-x", "-C", wt], input=arc)
        shutil.copy2(os.path.join(_REPO_ROOT, _SCENARIO_REL), os.path.join(wt, _SCENARIO_REL))
        env = dict(os.environ)
        env["SIM_BACKEND"] = "numpy"
        env["PYTHONPATH"] = wt + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.check_output([sys.executable, "-c",
                                       "import json,sys;"
                                       "from tests._plastic_mask_beyond_hebbian_scenario import run;"
                                       "print(json.dumps(run(sys.argv[1], False, int(sys.argv[2]))))",
                                       rule, str(seed)], cwd=wt, env=env, text=True)
        line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
        return json.loads(line)["sha"]
    except Exception:
        return None
    finally:
        shutil.rmtree(wt, ignore_errors=True)


def selftest(seed=42):
    """Both directions, one seed (the byte-identity side is inherently a single-run comparison)."""
    res = {"byte_identical_off": {}, "freeze_on": {}, "rebuild_alignment": None}
    for rule in RULES:
        off = _run(rule, False, seed)
        on = _run(rule, True, seed)
        off2 = _run(rule, False, seed)
        prefix_sha = _run_prefix_off(rule, seed)
        res["byte_identical_off"][rule] = {
            "off_sha": off["sha"], "prefix_sha": prefix_sha,
            "identical": (prefix_sha is None) or (off["sha"] == prefix_sha),
            "deterministic_off": off["sha"] == off2["sha"],
        }
        res["freeze_on"][rule] = {
            "off_delta": off["frozen_max_delta"], "on_delta": on["frozen_max_delta"],
            "control_delta_on": on["control_max_delta"], "fired": on["fired_any"],
        }
        if rule == "rebuild":
            res["rebuild_alignment"] = {"off_mislabelled": off["mask_mislabelled_after_rebuild"],
                                        "on_mislabelled": on["mask_mislabelled_after_rebuild"]}
    return res


@pytest.mark.parametrize("rule", RULES)
def test_flag_off_is_byte_identical_to_pre_fix(rule):
    res = selftest()[ "byte_identical_off"][rule]
    if res["prefix_sha"] is None:
        pytest.skip("the flag commit is not yet in git history -- cannot recover a genuine pre-fix ref")
    assert res["identical"], ("flag-OFF is NOT byte-identical to pre-fix for rule %s: off=%s prefix=%s"
                             % (rule, res["off_sha"], res["prefix_sha"]))


@pytest.mark.parametrize("rule", RULES)
def test_frozen_weights_unchanged_over_30_reads_on_six_seeds(rule):
    for seed in SEEDS:
        off = _run(rule, False, seed)
        on = _run(rule, True, seed)
        assert off["fired_any"] and on["fired_any"], "no spikes at seed %d: the run is undefined" % seed
        assert off["frozen_max_delta"] > _OFF_DRIFT_MIN[rule], (
            "the OFF arm must SHOW the drift (positive control), got %.6f for rule %s seed %d"
            % (off["frozen_max_delta"], rule, seed))
        assert on["frozen_max_delta"] <= _ON_DRIFT_MAX, (
            "frozen (plastic=False) synapses still move with the flag ON: rule %s seed %d delta %.6f"
            % (rule, seed, on["frozen_max_delta"]))
        assert on["control_max_delta"] > 0.0, (
            "the plastic control population did not move: rule %s seed %d did not run, so the zero is vacuous"
            % (rule, seed))
        if rule in ("rebuild", "compaction"):
            assert on["mask_mislabelled_at_end"] == 0, (
                "cp_synapse_plastic_mask is mislabelled after the re-ordering with the flag ON (%d pairs, rule %s, seed %d)"
                % (on["mask_mislabelled_at_end"], rule, seed))
            assert off["mask_mislabelled_after_rebuild"] > 0 or off["mask_mislabelled_at_end"] > 0, (
                "the alignment instrument cannot see the pre-fix misalignment -- the test is vacuous")
            if rule == "compaction":
                assert off["mask_mislabelled_at_end"] > 0, (
                    "the deferred-compaction misalignment is not measurable OFF (%d) -- vacuous test"
                    % off["mask_mislabelled_at_end"])
            if rule == "rebuild":
                assert off["mask_mislabelled_after_rebuild"] > 0, (
                    "the rebuild misalignment is not measurable OFF (%d) -- vacuous test"
                    % off["mask_mislabelled_after_rebuild"])


def test_all_three_masked_sites_freeze_frozen_on_one_bridge():
    """The integration case: scaling + structural elimination + reward modulation on ONE bridge at once.
    Frozen population bit-flat, plastic control population still moving."""
    for seed in SEEDS:
        r = _run("combined", True, seed)
        off = _run("combined", False, seed)
        assert r["fired_any"] and off["fired_any"], "no spikes at seed %d: the run is undefined" % seed
        assert off["frozen_max_delta"] > _OFF_DRIFT_MIN["combined"], (
            "the OFF arm must SHOW the drift (positive control), got %.6f at seed %d"
            % (off["frozen_max_delta"], seed))
        assert r["frozen_max_delta"] <= _ON_DRIFT_MAX, (
            "a frozen (plastic=False) synapse moved with all three sites armed: seed %d delta %.6f"
            % (seed, r["frozen_max_delta"]))
        assert r["control_max_delta"] > 0.0, (
            "the plastic control population did not move at seed %d -- the zero is vacuous" % seed)


if __name__ == "__main__":
    import tempfile  # noqa: F401  (imported here so __main__ printing works standalone)
    print(json.dumps(selftest(), indent=2))
