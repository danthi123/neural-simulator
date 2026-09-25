"""Tests for the LOCAL per-pathway plasticity freeze (BRAIN_WORLDMODEL_LOCAL_FREEZE / BRAIN_SURPRISE_LOCAL_FREEZE,
both default-OFF) that replaces `cfg.enable_hebbian_learning = False` on the SHARED wave3-pool cfg object.

Background (research/findings/2026-09-24-chat-time-plasticity-audit-*.md): `WorldModelProductionOrgan._build_one`
and `SurpriseProductionOrgan._build_one` each train their own pathway then set `cfg.enable_hebbian_learning =
False` on the cfg object EIGHT co-resident production organs share (`onebrain_wave3_pool_production.
get_merged_cortical_pool`) -- a bridge-WIDE kill switch, not a per-organ freeze. No production faculty exploits
ongoing chat-time bridge-Hebbian learning TODAY (every wave3-pool organ is itself a documented "build once,
frozen circuit" oracle -- see the finding's per-faculty table), so this is a LATENT hazard fix, not a behavior
change for any existing faculty: the new flags are default-OFF and, off, must reproduce the exact prior
mechanism (global kill). On, they must (a) still freeze each organ's OWN trained pathway and (b) leave
`cfg.enable_hebbian_learning` genuinely True so an UNGATED pathway elsewhere on the shared bridge is no longer
silently killed -- proven here with a synthetic probe pathway standing in for "whatever faculty wires an
ongoing chat-time Hebbian pathway onto this pool next."

Runs in a subprocess with `SIM_BACKEND=numpy` set before `sim.bridge`'s first import in that process (mirrors
tests/test_plastic_mask.py's own documented reason: the backend resolves once at module import and a sibling
test file earlier in the same pytest session can otherwise pin the wrong one)."""
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_scenario(mode: str) -> dict:
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    out = subprocess.check_output(
        [sys.executable, "-m", "tests._chat_time_plasticity_local_freeze_scenario", "--mode", mode],
        cwd=_REPO_ROOT, env=env, text=True)
    line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def test_off_is_byte_identical_to_the_old_global_kill():
    """Default (flag unset/off): no new gate is even DECLARED, and the shared cfg ends up with
    `enable_hebbian_learning=False` exactly as `main` does today -- the mechanism this test guards is entirely
    unchanged when the flag is off."""
    r = _run_scenario("off")
    assert r["gates_after_organ_build"] == r["gates_before_organ_build"], (
        "flag OFF must not declare worldmodel_frozen/surprise_frozen at all -- got a gate-set change: "
        f"before={r['gates_before_organ_build']!r} after={r['gates_after_organ_build']!r}")
    assert r["worldmodel_gate_value"] is None and r["surprise_gate_value"] is None
    assert r["enable_hebbian_learning_after_both_organs_built"] is False, (
        "flag OFF must reproduce the pre-existing bridge-wide kill: enable_hebbian_learning should read False")
    # The old mechanism's side effect: with the whole bridge frozen, even the UNGATED probe pathway (a stand-in
    # for some other faculty's chat-time-plastic pathway) cannot move -- this is exactly the blast radius the
    # finding measured, reproduced here as the OFF-arm control.
    assert r["probe_pathway"]["n_synapses"] > 0, "the probe pathway's synapses were not found on the merged bridge"
    assert r["probe_pathway"]["max_dw"] == 0.0, (
        "OFF: the global kill should have frozen the UNGATED probe pathway too (max_dw should be 0.0), got "
        f"{r['probe_pathway']['max_dw']!r}")


def test_on_freezes_own_pathway_but_frees_the_rest_of_the_bridge():
    """Flag ON: world-model's and surprise's OWN trained pathways stay frozen (max|dw|==0, by the NAMED gate
    now held at 0.0) AND `enable_hebbian_learning` reads True AND a co-resident UNGATED pathway (the probe,
    standing in for a faculty that DOES want ongoing chat-time learning) genuinely moves -- the causal proof
    that the fix removes the blast radius rather than just relabeling it."""
    r = _run_scenario("on")
    assert "worldmodel_frozen" in r["gates_after_organ_build"]
    assert "surprise_frozen" in r["gates_after_organ_build"]
    assert r["worldmodel_gate_value"] == 0.0
    assert r["surprise_gate_value"] == 0.0
    assert r["enable_hebbian_learning_after_both_organs_built"] is True, (
        "ON: the shared cfg's global switch must stay True -- world-model/surprise now freeze only their own "
        "pathway via the named gate, never the bridge-wide switch")

    assert r["worldmodel_own_pathway"]["n_synapses"] > 0
    assert r["worldmodel_own_pathway"]["max_dw"] == 0.0, (
        "ON: world-model's own state->pred_{pos,neg} transition must still read frozen (its own local gate "
        f"holds it at 0), got max_dw={r['worldmodel_own_pathway']['max_dw']!r}")
    assert r["surprise_own_pathway"]["n_synapses"] > 0
    assert r["surprise_own_pathway"]["max_dw"] == 0.0, (
        "ON: surprise's own cue->patient_expected association must still read frozen, got "
        f"max_dw={r['surprise_own_pathway']['max_dw']!r}")

    assert r["probe_pathway"]["n_synapses"] > 0
    assert r["probe_pathway"]["max_dw"] > 0.0, (
        "ON: the UNGATED probe pathway (a stand-in for a co-resident faculty's own chat-time-plastic pathway) "
        "must be genuinely free to learn now that the global switch is no longer held False -- got "
        f"max_dw={r['probe_pathway']['max_dw']!r} (expected > 0)")


def test_on_vs_off_only_differ_in_the_freeze_mechanism_not_in_which_pathways_exist():
    """Sanity cross-check: both arms see the SAME three pathways (same synapse counts) -- the flag changes HOW
    the freeze is applied, not the wiring topology itself outside of the two newly-tagged pathways."""
    off, on = _run_scenario("off"), _run_scenario("on")
    assert off["probe_pathway"]["n_synapses"] == on["probe_pathway"]["n_synapses"]
    assert off["worldmodel_own_pathway"]["n_synapses"] == on["worldmodel_own_pathway"]["n_synapses"]
    assert off["surprise_own_pathway"]["n_synapses"] == on["surprise_own_pathway"]["n_synapses"]
    # NO-REGRESSION (the 6-seed pre-registration's answer-preservation arm): both organs' own functional
    # reads are IDENTICAL whichever mechanism froze their pathway -- the fix is invisible to every caller.
    assert off["surprise_answer"] == on["surprise_answer"], (
        "surprise's own judge() answer changed between the old global-kill and the new local-gate mechanism")
    assert off["worldmodel_answer"] == on["worldmodel_answer"], (
        "world-model's own expectation()/read_surprise() answers changed between the old global-kill and the "
        "new local-gate mechanism")
