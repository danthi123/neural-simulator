"""research/runners/stable_preset_compare.py: the mechanism classifier reproduces the stable-integration finding's
mechanism (c) cell (b < 0, no conductance hazard, no refractory state, resting branch below v_low), separates the other
classes, and the comparison runs end to end on two synthetic census records. Reads JSON only."""
import json
import math

from research.runners import stable_preset_compare as SPC

# the first LearnedAssocGraph offender of research/findings/raw/_stable_integration/census/on/s7/inner__intact_a.json
ROW_C = {"b": -1.9349, "k": 0.25, "vr": -65.0, "vt": -45.0, "u_before": 1095.39, "extreme_low_mV": -166.53,
         "extreme_high_mV": None, "n_steps_high": 0, "refractory": False, "refractory_post": False,
         "gtot_dt_over_C": 0.012, "rf_masked": None}


def test_rest_branch_matches_the_findings_hand_value():
    # "for k 0.25, vr -65, vt -45 and u near 1100 pA it is near -122 mV"
    assert abs(SPC.rest_branch_mV(0.25, -65.0, -45.0, 1100.0) - (-122.1)) < 0.2
    assert math.isclose(SPC.rest_branch_mV(0.25, -65.0, -45.0, 0.0), -65.0)          # u = 0: the resting root is vr
    assert SPC.rest_branch_mV(0.25, -65.0, -45.0, -1000.0) is None                    # no real root below the parabola


def test_classify_row_separates_the_mechanisms():
    c = SPC.classify_row
    assert c(ROW_C, -110.0) == "u_held_b_neg_c"
    assert c({**ROW_C, "rf_masked": True}, -110.0) == "phasor"
    assert c({**ROW_C, "gtot_dt_over_C": 2.5}, -110.0) == "conductance_a"
    assert c({**ROW_C, "extreme_high_mV": 120.0, "extreme_low_mV": None}, -110.0) == "high"
    assert c({**ROW_C, "b": 3.0}, -110.0) == "low_b_nonneg"
    assert c({**ROW_C, "refractory": True}, -110.0) == "low_b_neg_other"
    assert c({**ROW_C, "u_before": 50.0}, -110.0) == "low_b_neg_other"                # branch above v_low: not held by u


def _rec(group, oor_rows, b_mean, n_b_neg, preset_centred, answer):
    bridge = {"organ": "x.y:Org", "n_oor_neurons": len(oor_rows), "oor_neurons": oor_rows, "oor_neurons_truncated": False,
              "first_reading_classes": {"low_other": len(oor_rows)},
              "preset_centred_het": {"requested": preset_centred, "applied": preset_centred, "organ_value": False,
                                     "readback": {"preset_centred": preset_centred, "n_with_region_preset": 4,
                                                  "n_preset_off_centre": 4 if preset_centred else 0}},
              "param_summary": {"r": {"n": 10, "a_mean": 0.03, "b_mean": b_mean, "d_increment_mean": 100.0, "C_mean": 90.0,
                                      "n_b_negative": n_b_neg, "preset_counts": {"X": 10}}}}
    organ = {"organ": "x.y:Org", "n_oor_neurons": len(oor_rows), "n_low_neurons": len(oor_rows)}
    return {"group": group, "git_sha": "s", "stable_integration_env": "1",
            "preset_centred_het_env": "1" if preset_centred else None, "wall_s": 1.0,
            "census": {"bridges": [bridge], "organs": [organ], "thresholds": {"v_low_mV": -110.0},
                       "observer_errors": 0, "observe_s": 0.0, "total_bridge_steps": 1},
            "workload": {"rc": 0, "wall_s": 1.0, "turn_replies": {
                "t1": {"sha256": "a" if not preset_centred else "b", "answer": answer, "abstained": answer is None}}}}


def test_validity_refuses_a_run_without_its_corpus():
    ok = {"workload": {"rc": 0, "corpus_present": True, "turn_errors": {}, "pythonhashseed_ok": True,
                       "shard": {"responses": {"t": {"comprehension": None}}}}, "census": {"observer_errors": 0}}
    assert SPC.validity(ok) == []
    bad = json.loads(json.dumps(ok))
    bad["workload"]["corpus_present"] = False
    bad["workload"]["shard"]["responses"]["t"]["comprehension"] = {"on": True, "error": "FileNotFoundError: x"}
    why = SPC.validity(bad)
    assert any("corpus_present" in w for w in why) and any("comprehension error" in w for w in why)


def test_compare_pair_runs_and_counts(tmp_path):
    base = _rec("emo", [ROW_C, ROW_C], -2.0, 10, False, "yes")
    alt = _rec("emo", [], 5.0, 0, True, None)
    res = SPC.compare_pair("emo", ("b.json", base), ("a.json", alt))
    m = res["mechanism"]
    assert (m["u_held_b_neg_c_base"], m["u_held_b_neg_c_alt"]) == (2, 0)
    assert (m["n_b_negative_base"], m["n_b_negative_alt"]) == (10, 0) and m["n_preset_off_centre_alt"] == 4
    assert m["regions_moved"][0]["b_mean_base"] == -2.0 and m["regions_moved"][0]["b_mean_alt"] == 5.0
    assert res["answer_rate"]["transitions"] == {"answered->abstained": 1}
    assert res["replies"]["n_changed"] == 1
    json.dumps(res, default=str)
