"""Review fixes for the gap#4 transport-ceiling runner (prereg AMENDMENT 5), each pinned in its failing direction.

1. The evaluation-seed guard can FAIL: the existing PREREGISTRATION file (which only carries dev-seed amendments), an
   untracked path, a section that registers another fingerprint or omits a seed, and the phrase used only in prose are
   all refused; only a committed 'AMENDMENT ... EVALUATION CONFIG' section with this run's fingerprint and seeds passes.
2. The apical lesion is a MATCHED cut: with the interneuron rate zeroed, the hidden apical stays at rest during
   training (0 mV), while the parent's unmatched lesion and the intact arm drive it (the check can fail).
3. The calibration tie-break uses the registered cost, epochs x steps per example.
"""
import json
import os
import types

os.environ.setdefault("SIM_BACKEND", "numpy")


def _mod():
    import research.runners._gap4_transport_ceiling_readout_derisk as m
    return m


def test_eval_seed_guard_refuses_the_existing_prereg_and_fakes():
    m = _mod()
    fp = "0123456789abcdef"
    ok, why = m.check_eval_amendment([42], fp, m.PREREG)
    assert not ok, why                                            # the prereg itself no longer unlocks seed 42
    ok, _ = m.check_eval_amendment([42], fp, "research/findings/_definitely_untracked_amendment.md")
    assert not ok
    ok, _ = m.check_eval_amendment([42], fp, None)
    assert not ok
    good = "## AMENDMENT 9 -- EVALUATION CONFIG\n\nevaluation-config-fingerprint: `%s`\nevaluation-seeds: 42 43\n" % fp
    ok, _ = m.check_eval_amendment([42], "f" * 16, "x.md", read_committed=lambda _p: (good, "fake"))
    assert not ok                                                 # another config's fingerprint
    ok, _ = m.check_eval_amendment([42, 100], fp, "x.md", read_committed=lambda _p: (good, "fake"))
    assert not ok                                                 # seed 100 not registered
    prose = "We will write an EVALUATION CONFIG later.\nevaluation-config-fingerprint: %s\nevaluation-seeds: 42\n" % fp
    ok, _ = m.check_eval_amendment([42], fp, "x.md", read_committed=lambda _p: (prose, "fake"))
    assert not ok                                                 # keys outside an amendment heading
    ok, _ = m.check_eval_amendment([42], fp, "x.md", read_committed=lambda _p: (None, "not in HEAD"))
    assert not ok                                                 # uncommitted
    ok, why = m.check_eval_amendment([42, 43], fp, "x.md", read_committed=lambda _p: (good, "fake"))
    assert ok, why                                                # the guard can pass
    ok, _ = m.check_eval_amendment([7], fp, None)
    assert ok                                                     # dev seed needs no amendment


def test_matched_lesion_keeps_hidden_apical_at_rest(tmp_path):
    m = _mod()
    out = tmp_path / "lesion.json"
    rc = m.lesion_selftest(types.SimpleNamespace(out=str(out)))
    d = json.loads(out.read_text())
    assert rc == 0, d
    by = {c["case"].split(" (")[0]: c["hidden_apical_max_abs_dev_mV"] for c in d["cases"]}
    assert by["matched lesion"] == 0.0
    assert by["UNMATCHED parent lesion"] > 1.0                    # the pre-fix lesion drove the apical
    assert by["intact micro_inengine"] > 1.0


def test_calibration_tie_break_uses_epochs_times_steps(tmp_path):
    m = _mod()

    def art(label, epochs, settle, headroom):
        cfg = {"label": label, "settle_steps": settle, "credit_steps": 25, "isi_steps": 0, "read_window": 30,
               "read_gain": 20, "epochs": epochs}
        rep = {"inherit_heldout": {"transport_ceiling": 0.4, "frozen": 0.4 - headroom},
               "train_acc": {}, "decode_h2_heldout": {}, "ceiling_binom_p": 0.001, "headroom": headroom,
               "chance": 0.167, "oracle": 0.9}
        p = tmp_path / ("%s.json" % label)
        p.write_text(json.dumps({"config": cfg, "per_seed": {"7": {"replicates": [rep]}}}))
        return str(p)
    # A: fewer steps per example (65) but 30 epochs -> cost 1950; B: more steps (85) but 10 epochs -> cost 850
    a = art("A", 30, 40, 0.20)
    b = art("B", 10, 60, 0.19)
    res = m.select_calibration([a, b], str(tmp_path / "sel.json"))
    assert res["chosen"]["label"] == "B", res["chosen"]
