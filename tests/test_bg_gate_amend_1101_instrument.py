"""RULING bg-gate-1101 instrument (report-only; no bar, no default, no decision changes).

Pins (numpy, CPU):
  * the organ's trial record is untouched: tests/test_bg_commit.py::test_bg_flag_off_neural_seam_trials_byte_identical
    still hashes it to the cc9b4bf3e golden (re-asserted here so this file alone proves it);
  * the new per-trial diagnostics (`NeuralSeamSelector.last_diag`, `decide_action_neural.last_diag`): per-region
    sub-reversal step counts (v below E_inh - 5 mV) over every v-checked step, their minima agree with the trial's
    own aggregate v_min, and GPi / thalamus / motor_k / GPe / STN rates over the drive and hold windows;
  * the scorer: pooled values and n are unchanged (the 10-02 gate shards score as committed), the G3 numerator split
    sums to G3's numerator and reproduces ns, and the relabelled attribution carries a per-turn-class breakdown.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

GATE_DIR = os.path.join(REPO, "research", "findings", "raw", "_bg_live_seam", "gate")
GATE = (42, 43, 44, 100, 101, 102)
POOLED_KEY = "value-route lesion attribution (held rate), pooled"
CLASS_KEY = "value-route lesion attribution (held rate), by turn class"
SPLIT_KEY = "G3 numerator split (report-only; bar unchanged)"


# ── scorer ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_g3_split_unit():
    import research.runners.host_share_bg_seam as H
    evs = [{"class": "empty", "intact_held": 1, "lesion_held": 0, "ns": 1},     # hold loss
           {"class": "empty", "intact_held": 1, "lesion_held": 1, "ns": 0},
           {"class": "content", "intact_held": 0, "lesion_held": 1, "ns": 1},   # wrong hold
           {"class": "content", "intact_held": 0, "lesion_held": 0, "ns": 1},   # reply-only change
           {"class": "one_word", "intact_held": 0, "lesion_held": 1, "ns": 1}]
    s = H.g3_numerator_split(evs)
    assert (s["denominator"], s["numerator"]) == (5, 4)
    assert (s["empty_turn_hold_loss"], s["content_turn_wrong_hold"], s["other"]) == (1, 1, 2)
    assert s["other_by_kind"] == {"content:spoke->spoke": 1, "one_word:spoke->held": 1}
    assert s["by_class"]["empty"] == {"denominator": 2, "numerator": 1}
    assert s["ns"] == 0.8
    assert H.g3_numerator_split([])["ns"] is None


def test_scorer_pooled_unchanged_and_split_matches_g3():
    import research.runners.host_share_bg_seam as H
    sc = H.score(GATE_DIR, GATE)
    ref = json.load(open(os.path.join(GATE_DIR, "score.json")))
    assert sc["pooled"] == ref["pooled"] and sc["n"] == ref["n"]        # no bar input moved
    d = sc["diagnostics"]
    assert "seam_held_attribution" not in d                              # relabelled
    assert d[POOLED_KEY] == ref["diagnostics"]["seam_held_attribution"]  # same number, new label
    sp = d[SPLIT_KEY]
    assert sp["denominator"] == sc["n"]["ns"]
    assert sp["ns"] == sc["pooled"]["ns"]
    assert sp["empty_turn_hold_loss"] + sp["content_turn_wrong_hold"] + sp["other"] == sp["numerator"]
    assert sum(v["denominator"] for v in sp["by_class"].values()) == sp["denominator"]
    assert set(d[CLASS_KEY]) == {"empty", "content", "one_word"}
    assert sum(v["n"] for v in d[CLASS_KEY].values()) == sc["n"]["seam_lesion_held"]
    vd = H.verdict(sc)
    assert vd["verdict"] == ref["verdict"] and vd["gates"] == ref["gates"]


# ── organ instrument ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def organ_trials():
    from research.runners import bg_action_selection_production_organ as BG
    org = BG.NeuralSeamSelector(seed=1)
    out = []
    for m in ("...", "what does the dog chase"):
        r = org.select_once(m)
        out.append((r, org.last_diag, org))
    return BG, out


def test_trial_record_still_matches_the_golden():
    import importlib
    T = importlib.import_module("tests.test_bg_commit")
    T.test_bg_flag_off_neural_seam_trials_byte_identical()


def test_trial_record_carries_no_new_key(organ_trials):
    _BG, out = organ_trials
    for r, _dg, _org in out:
        assert "sub_reversal" not in r and "bg_rates_hz" not in r


def test_sub_reversal_counts(organ_trials):
    BG, out = organ_trials
    for r, dg, org in out:
        sr = dg["sub_reversal"]
        assert sr["threshold_mv"] == -80.0 and sr["e_inh_mv"] == -75.0
        cfg = org.config
        assert sr["steps_checked"] == BG.T_DRIVE + BG.T_HOLD + int(cfg.reset_steps) + int(cfg.washout_steps)
        names = [x.name for x in org._bridge.region_manager.regions()]
        assert set(sr["steps_below"]) == set(names) and "gpi_0" in names and "selector_stn" in names
        vmins = [v for v in sr["v_min"].values() if v is not None]
        assert min(vmins) == pytest.approx(r["v_min"], abs=1e-3)        # same reads as the membrane precondition
        for n, k in sr["steps_below"].items():
            assert 0 <= k <= sr["steps_checked"]
            assert (k > 0) == (sr["v_min"][n] is not None and sr["v_min"][n] < -80.0)
        assert sr["regions_below"] == sorted(n for n, k in sr["steps_below"].items() if k)


def test_rates_recorded(organ_trials):
    _BG, out = organ_trials
    for r, dg, _org in out:
        rates = dg["bg_rates_hz"]
        for w in ("drive", "hold"):
            for n in ("gpi_0", "gpi_1", "thal_0", "thal_1", "motor_0", "motor_1", "gpe_0", "gpe_1", "selector_stn"):
                assert n in rates[w] and rates[w][n] >= 0.0
        # motor_k over drive + hold agrees with the trial's own motor_total (n_motor 30, dt 1 ms)
        for c in (0, 1):
            tot = (rates["drive"]["motor_%d" % c] * 300 + rates["hold"]["motor_%d" % c] * 200) * 30 / 1000.0
            assert tot == pytest.approx(r["motor_total"][c], abs=0.5)


def test_decide_action_neural_exposes_the_diag(monkeypatch):
    from research.runners import bg_action_selection_production_organ as BG
    for k in ("BRAIN_BG_COMMIT", "BRAIN_BG_NEURAL_SEAM_LESION"):
        monkeypatch.delenv(k, raising=False)
    BG._NEURAL_ORGANS.clear()
    BG.decide_action_neural("...", seed=1)
    dg = BG.decide_action_neural.last_diag
    assert dg is not None and dg is BG._NEURAL_ORGANS[(1, None)].last_diag
    BG._NEURAL_ORGANS.clear()
