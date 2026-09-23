"""Fix-round pins for research/runners/_lbf_open_ended_production_turn_probe.py (review 2026-09-23).

Each test FAILS on the pre-fix runner (b34be3f2f) and passes after:
  * the seed, not the ask, is the statistical unit: one seed can never be significant, and no within-session
    permutation p-value is carried as evidence (the old scorer reported p = 1e-4 for ONE deterministic session);
  * the lesioned edge is declared as a HOST weight vector, and the host shortcut list names P / w;
  * the host_oracle arm exists (what the spiking WTA itself contributes);
  * the true open-ended configuration (teach AND ask under BRAIN_OPEN_ENDED=1, both routes) exists, and
    oe_routed_taught is flagged NOT independent.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.runners._lbf_open_ended_production_turn_probe as P  # noqa: E402


def _mk(outs, ablated, n):
    return {"seed": 42, "stored_facts": [["dog", "chase", "cat"]],
            "replies": [{"hypothesis_svo": ["dog", "chase", o]} for o in outs],
            "draw_counter": {"n_calls": n, "n_ablated_calls": ablated, "n_sample_calls": n},
            "likelihood_weight": {"deer": 3.0, "rabbit": 2.0, "cat": 2.0, "beetle": 1.0, "minnow": 1.0}}


def test_single_seed_is_one_observation():
    # the seed-42 shape: deer 39/40 intact, rabbit 39/40 lesion -- the old scorer called this p = 1e-4.
    I = ["rabbit"] + ["deer"] * 39
    L = ["deer"] + ["rabbit"] * 39
    s = P.score_seed(_mk(I, 0, 41), _mk(I, 0, 41), _mk(L, 432, 432))
    assert "p_perm" not in s
    assert s["verdict"] == "DEFINED" and s["label"] == "CHANGED-TOWARD-LIKELIHOOD"
    assert P.seed_level_signflip([s["direction_D"]]) == 0.5          # one seed can never be significant
    assert abs(P.seed_level_signflip([s["direction_D"]] * 6) - 1 / 64.) < 1e-12
    assert P.seed_level_signflip([s["direction_D"], None]) is None   # an UNDEFINED seed is never a pass


def test_chance_base_rate_excludes_stored_fact():
    s = P.score_seed(_mk(["deer"] * 40, 0, 40), _mk(["deer"] * 40, 0, 40), _mk(["rabbit"] * 40, 400, 400))
    assert "cat" not in s["admissible_approx"]                       # (dog, chase, cat) is stored -> not novel
    assert s["host_argmax_prediction"] == ["deer"]
    assert abs(s["chance_p_modal_differs"] - 0.75) < 1e-12


def test_lesioned_edge_declared_host():
    assert "HOST weight vector" in P.LESIONED_EDGE
    assert any("co-occurrence matrix P" in h for h in P.HOST_SHORTCUTS)
    s = P.score_seed(_mk(["deer"] * 40, 0, 40), _mk(["deer"] * 40, 0, 40), _mk(["rabbit"] * 40, 400, 400))
    assert s["lesioned_edge"] == P.LESIONED_EDGE


def test_host_oracle_arm_and_sharpening():
    assert P.ARMS["host_oracle"]["BRAIN_SPIKING_DRAW"] == "0"
    assert "host_oracle" not in P.DEFAULT_ARMS
    H = ["deer"] * 17 + ["rabbit"] * 11 + ["beetle"] * 6 + ["minnow"] * 6
    s = P.score_seed(_mk(["deer"] * 40, 0, 40), _mk(["deer"] * 40, 0, 40), _mk(["rabbit"] * 40, 400, 400),
                     host_oracle=_mk(H, 0, 0))
    assert abs(s["sharpening_modal_frac_spiking_minus_host"] - (1.0 - 17 / 40.)) < 1e-12


def test_true_open_ended_configuration_mode():
    m = P.MODES["oe_routed_full"]
    assert m["BRAIN_OPEN_ENDED"] == "1" and m["BRAIN_OPEN_ENDED_GENERATE_ROUTE"] == "1"
    assert m["BRAIN_OPEN_ENDED_ACQUIRE_ROUTE"] == "1"
    assert "oe_routed_full" not in P.TEACH_ENV                     # teach phase ALSO in open-ended mode
    assert "oe_routed_taught" in P.NOT_INDEPENDENT


def test_runner_selftest_passes():
    assert P.selftest() == 0


def _verdicts(d, seeds, verdict_of, d_of):
    import json
    d.mkdir()
    out = []
    for s in seeds:
        p = d / ("default_s%s_verdict.json" % s)
        p.write_text(json.dumps({"mode": "default", "seed": s,
                                 "score": {"verdict": verdict_of(s), "direction_D": d_of(s), "label": "x",
                                           "modal_changed": True}}))
        out.append(str(p))
    return out


def test_aggregate_gate_can_fail(tmp_path):
    seeds = [42, 43, 44, 100, 101, 102]
    ok = _verdicts(tmp_path / "a", seeds, lambda s: "DEFINED", lambda s: 0.9)
    r = P.aggregate(ok, str(tmp_path / "agg.json"))["summary"]["default"]
    assert r["GO"] is True and abs(r["p_signflip_over_seeds"] - 1 / 64.) < 1e-12
    assert abs(r["p_signflip_heldout_excl_seed42"] - 1 / 32.) < 1e-12 and r["GO_heldout_only"] is True
    one_undef = _verdicts(tmp_path / "b", seeds, lambda s: "UNDEFINED" if s == 101 else "DEFINED",
                          lambda s: None if s == 101 else 0.9)
    assert P.aggregate(one_undef, str(tmp_path / "agg2.json"))["summary"]["default"]["GO"] is False  # never a pass
    five = _verdicts(tmp_path / "c", seeds[:5], lambda s: "DEFINED", lambda s: 0.9)
    assert P.aggregate(five, str(tmp_path / "agg3.json"))["summary"]["default"]["GO"] is False
    mixed = _verdicts(tmp_path / "d", seeds, lambda s: "DEFINED", lambda s: -0.9 if s == 44 else 0.9)
    r = P.aggregate(mixed, str(tmp_path / "agg4.json"))["summary"]["default"]
    assert r["GO"] is False and abs(r["p_signflip_over_seeds"] - 7 / 64.) < 1e-12
