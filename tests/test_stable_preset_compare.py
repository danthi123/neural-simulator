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


def test_leaf_diff_reports_every_changed_leaf_with_its_largest_relative_change():
    def rec(g, mag, label="p_a"):
        return {"workload": {"shard": {"responses": {label: {"answer": "x", "da_encoding": {"g": g},
                                                           "activity": {"rf": {"mean_magnitude": mag}},
                                                           "list": [1.0, 2.0]}}, "timings_s": {label: 3.0}}}}
    d = SPC.leaf_diff(rec(1.0, 100.0), rec(1.1, 100.0))
    assert d["keys_equal"] and d["n_changed"] == 1 and d["changed_types"] == {"float": 1}
    r = d["readouts"]["responses/da_encoding/g"]
    assert r["n_changed"] == 1 and abs(r["max_rel"] - 0.1) < 1e-12 and r["values"] == [1.0, 1.1]
    assert d["n_non_numeric_changed"] == 0
    # a changed answer string is reported as non-numeric, never folded into a number
    a, b = rec(1.0, 100.0), rec(1.0, 100.0)
    b["workload"]["shard"]["responses"]["p_a"]["answer"] = "y"
    d2 = SPC.leaf_diff(a, b)
    assert d2["n_non_numeric_changed"] == 1 and d2["readouts"]["responses/answer"]["non_numeric"] == 1
    # timings are excluded, and a different key set is flagged
    c = rec(1.0, 100.0)
    c["workload"]["shard"]["timings_s"]["p_a"] = 99.0
    assert SPC.leaf_diff(rec(1.0, 100.0), c)["n_changed"] == 0
    e = rec(1.0, 100.0)
    del e["workload"]["shard"]["responses"]["p_a"]["list"]
    assert SPC.leaf_diff(rec(1.0, 100.0), e)["keys_equal"] is False
    tot = SPC.merge_leaf_diffs({"conv": d, "emo": d2})
    assert tot["n_changed"] == 2 and tot["readouts"]["responses/da_encoding/g"]["group"] == "conv"


def test_trait_read_back_and_type_split_flow_into_the_comparison():
    row = {**ROW_C, "type": "Izh2007_THALAMIC_RETICULAR"}
    base = _rec("inner", [row, row], -2.0, 10, False, "yes")
    alt = _rec("inner", [], 2.0, 0, False, "yes")
    base["census"]["bridges"][0]["trait_centred_het"] = {"requested": False, "applied": False, "organ_value": None,
                                                         "readback": {"trait_centred": False}}
    alt["census"]["bridges"][0]["trait_centred_het"] = {"requested": True, "applied": True, "organ_value": False,
                                                        "readback": {"trait_centred": True, "n_trait_neurons": 10,
                                                                     "n_trait_off_centre": 8}}
    alt["trait_centred_het_env"] = "1"
    res = SPC.compare_pair("inner", ("b.json", base), ("a.json", alt))
    m = res["mechanism"]
    assert m["n_trait_off_centre_alt"] == 8 and m["u_held_by_type_base"] == {"Izh2007_THALAMIC_RETICULAR": 2}
    assert m["u_held_by_type_alt"] == {}
    pa = res["mechanism"]["organs"][0]["params_alt"]
    assert pa["trait_switch_applied_seen"] == ["True"] and pa["trait_draw_centred_seen"] == ["True"]
    assert res["env"]["alt"] == ["1", None, "1"] and res["env"]["base"] == ["1", None, None]
    assert "leaf_diff" in res
    json.dumps(res, default=str)
