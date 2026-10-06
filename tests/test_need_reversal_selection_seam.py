"""Need-reversal selection seam: the harness race IS the production race, and the host path carries no candidate
identity before the neural commit. Dev seed 7, numpy, a few trials (a CI guard, not a result)."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _need_reversal_selection_seam as NR
from research.runners._vocal_action_selector_gate import (
    DIRECT_PATH_GATE, _set_equal_tonic_current, _step, build_selector_bridge, selector_config,
)
from research.runners.bg_action_selection_production_organ import _run_biased_trial


def _warm(seed):
    cfg = selector_config("v2")
    b = build_selector_bridge(seed, cfg)
    b.set_transmission_gate(DIRECT_PATH_GATE, 1.0)
    _set_equal_tonic_current(b, cfg)
    _step(b, cfg.warmup_steps)
    return b, cfg


def test_trial_is_the_production_race():
    """With no body afferent and no clamps, run_need_trial reproduces the production _run_biased_trial spike for
    spike (same winners, motor counts and decision steps) on identically built bridges. The two bridges run one
    after the other: a build reseeds the noise stream, and interleaving two live bridges would mix their draws."""
    sals = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    b1, cfg = _warm(7)
    prod = [_run_biased_trial(b1, cfg, s[0], s[1]) for s in sals]
    b2, _ = _warm(7)
    ours = [NR.run_need_trial(b2, cfg, NR.NeedParams(), None, host_salience=s) for s in sals]
    for a, b in zip(prod, ours):
        assert (a["winner"], a["motor_spikes"], a["decision_step"]) == (b["winner"], b["motor_spikes"],
                                                                         b["decision_step"])


def test_static_audit_clean():
    audit = NR.static_audit()
    assert audit["clean"], audit


def test_touch_audit_sees_host_rule_and_not_the_synaptic_arm():
    host = NR.run_block(7, 1.0, 2, arm="host_salience")
    syn = NR.run_block(7, 1.0, 2, arm="synaptic_need")
    assert host["pre_commit_reads"] > 0      # the host value rule reads labels before the race: detected
    assert syn["pre_commit_reads"] == 0      # the synaptic arm never does


def test_after_arm_ignores_host_variables():
    """Swapping the world's labels and inverting the host's belief about the need leaves the synaptic arm's spike
    record byte-identical: no host decision variable reaches the race."""
    base = NR.run_block(7, 1.0, 3, arm="synaptic_need")
    flipped = NR.run_block(7, 1.0, 3, arm="synaptic_need",
                           cands=NR.WorldCandidates({0: "forage", 1: "rest"}), host_belief_h=0.0)
    assert base["digest"] == flipped["digest"]


def test_before_arm_follows_host_variables():
    """Positive control for the instrument: in the host-salience arm the same flip changes the record."""
    base = NR.run_block(7, 1.0, 3, arm="host_salience")
    flipped = NR.run_block(7, 1.0, 3, arm="host_salience", host_belief_h=0.0)
    assert base["digest"] != flipped["digest"]


def _synthetic(R_intact=0.9, R_shuffled=0.0, commit=1.0, lesion_commit=1.0):
    """A minimal per-seed result in the shape run_seed returns (only the fields seed_verdict reads)."""
    def blk(commit_rate, p_forage, need=1.0, direct=1.0, reads=0):
        return {"commit_rate": commit_rate, "p_forage": p_forage, "pre_commit_reads": reads,
                "gates_at_end": {"need_value": need, "vocal_selector_direct_path": direct}}
    hi, lo = 0.5 + R_intact / 2, 0.5 - R_intact / 2
    S = {}
    for h, p in (("1.0", hi), ("0.0", lo)):
        S[f"after/intact/h{h}"] = blk(commit, p)
        S[f"before/intact/h{h}"] = blk(1.0, p, need=0.0, reads=2)
        S[f"before/host_intervened/h{h}"] = blk(1.0, 1 - p, need=0.0, reads=2)
        S[f"before/neural_intervened/h{h}"] = blk(1.0, p, need=0.0, reads=2)
        S[f"after/need_lesion/h{h}"] = blk(lesion_commit, 0.5, need=0.0)
        S[f"after/shuffled_synapses/h{h}"] = blk(1.0, 0.5 + (R_shuffled / 2 if h == "1.0" else -R_shuffled / 2))
        S[f"after/swapped_interoception/h{h}"] = blk(commit, 1 - p)
        S[f"after/direct_path_lesion/h{h}"] = blk(0.0, None, direct=0.0)
    R = {"after/intact": R_intact if commit >= 0.5 else None, "after/need_lesion": 0.0,
         "after/shuffled_synapses": R_shuffled, "after/swapped_interoception": -R_intact}
    return {"seed": 1, "blocks": S, "reversal": R, "static_audit": {"clean": True},
            "after_host_intervention_byte_identical": True,
            "host_share": {"before": [1.0, 1.0], "after": [0.0, 0.0]},
            "neural_share": {"before": [0.0, 0.0], "after": [0.9, 0.9]}}


def test_verdict_can_read_go_nogo_and_undefined():
    assert NR.seed_verdict(_synthetic())["decision"] == "GO"
    # a measured, failing criterion is a NO-GO, never an UNDEFINED
    assert NR.seed_verdict(_synthetic(R_shuffled=0.6))["decision"] == "NO-GO"
    assert NR.seed_verdict(_synthetic(R_intact=0.3))["decision"] == "NO-GO"
    # a control arm that does not choose cannot be read: UNDEFINED
    assert NR.seed_verdict(_synthetic(lesion_commit=0.2))["decision"] == "UNDEFINED"


def test_overall_block_only_asserts_over_the_six_gate_seeds():
    sv = [dict(NR.seed_verdict(_synthetic()), seed=s) for s in NR.GATE_SEEDS]
    full = NR.overall_block(sv, list(NR.GATE_SEEDS))
    assert full["verdict"] == "GO" and full["status"] == "GO" and full["preconditions"]
    dev = NR.overall_block(sv[:3], list(NR.GATE_SEEDS[:3]))
    assert "verdict" not in dev and dev["dev_readout"].startswith("DEV")
    # three UNDEFINED seeds -> the preregistered verdict is UNDEFINED and its precondition failed
    bad = [dict(NR.seed_verdict(_synthetic(lesion_commit=0.2)), seed=s) for s in NR.GATE_SEEDS[:3]] + sv[3:]
    out = NR.overall_block(bad, list(NR.GATE_SEEDS))
    assert out["verdict"] == "UNDEFINED" and out["status"] == "UNDEFINED"
