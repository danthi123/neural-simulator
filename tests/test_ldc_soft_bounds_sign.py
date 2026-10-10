"""A2 sign+size unit test (RULING a2-test2-diagnose, part 1).

One D1 and one D2 synapse each at floor, middle and ceiling; assert the SIGN of the
soft-bounded update after the D2 sign flip and its SIZE against the soft-bound factors
(exponents frozen at 1.0 per a02-conditions (5)).

Method: two identical seed-7 credit-selector bridges (soft_bounds 0 vs 1), DA primed
with learning off, then weights and eligibility set directly and ONE measured step
taken on each. Ratios between the two bridges cancel learning rate, DA magnitude,
eligibility and every shared multiplier, so a swapped up/down branch or a sign slip
fails loudly. Two cases (D2 at floor, D1 at ceiling) are unmeasurable on the sb0 side
because the unbounded rule clips at the bound, so those are asserted as exact zeros
on the sb1 side instead - the unbounded rule would move there, the bounded rule must
not.
"""
import numpy as np

from sim.backend import to_host
from research.runners._ldc_credit_omission import CreditParams, build_credit_selector
from research.runners._ldc_dip_observability import Run, calibrate_da, Synapses

E0 = 1.0
PRIME = 60


def _build(sb):
    p = CreditParams(soft_bounds=sb)
    bridge, config = build_credit_selector(7, p, elig_tau_ms=p.elig_tau_ms, tag_gain=1.0)
    calibrate_da(bridge, config, p)
    run = Run(bridge, config, p, learn=True)
    cfg = bridge.core_config
    lr_on = float(cfg.reward_learning_rate)
    cfg.reward_learning_rate = 0.0
    drive = dict(arousal=True, extra=(("ctx_o", p.ctx_pA), ("reward_us", p.rus_pA)))
    for _ in range(PRIME):
        run.steps(1, **drive)
    da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    return {"b": bridge, "cfg": cfg, "run": run, "lr_on": lr_on, "drive": drive, "da": da}


def _measure(ctx, d1, d2, w_set):
    b, cfg, run = ctx["b"], ctx["cfg"], ctx["run"]
    for i in d1 + d2:
        b.cp_connections.data[i] = w_set
        b.cp_eligibility_trace[i] = E0
    before = np.asarray(to_host(b.cp_connections.data)).copy()
    cfg.reward_learning_rate = ctx["lr_on"]
    run.steps(1, **ctx["drive"])
    after = np.asarray(to_host(b.cp_connections.data))
    return {"d1": float(np.mean(after[d1] - before[d1])),
            "d2": float(np.mean(after[d2] - before[d2]))}


def test_sign_and_size_after_d2_flip():
    a = _build(0)   # pre-A2 rule
    c = _build(1)   # soft-bounded rule
    cfg = c["cfg"]
    assert int(cfg.reward_soft_bounds) == 1, "flag must be on for the bound arm"
    assert float(cfg.reward_soft_bound_n_up) == 1.0 and float(cfg.reward_soft_bound_n_down) == 1.0, \
        "exponents are frozen at 1.0 (a02-conditions 5)"
    assert not bool(cfg.enable_stdp), "this test assumes the hebbian-bound clip path"
    assert c["da"] > 1e-6, "DA must deviate (appetitive) for the update path to fire"

    nnz = int(c["b"].cp_connections.nnz)
    syn1, syn0 = Synapses(c["b"]), Synapses(a["b"])
    d1 = [int(x) for x in syn1.mask("ctx_a", "str_d1_0")][:3]
    d2 = [int(x) for x in syn1.mask("ctx_a", "str_d2_0")][:3]
    assert len(d1) == 3 and len(d2) == 3, (d1, d2)
    assert [int(x) for x in syn0.mask("ctx_a", "str_d1_0")][:3] == d1, "builds must be identical"
    sgn = np.asarray(to_host(c["b"].cp_d1_d2_sign))[:nnz]
    assert np.all(sgn[d1] == 1.0), "build sign array must be +1 on D1"
    assert np.all(sgn[d2] == -1.0), "build sign array must be -1 on D2"

    wmin, wmax = float(cfg.hebbian_min_weight), float(cfg.hebbian_max_weight)
    res = {state: (_measure(a, d1, d2, w), _measure(c, d1, d2, w))
           for state, w in (("floor", wmin), ("mid", (wmin + wmax) / 2.0), ("ceiling", wmax))}

    # --- SIGN (unclipped state, sb0 reference rule): positive DA -> D1 with, D2 against.
    assert res["mid"][0]["d1"] > 0.0, res["mid"][0]
    assert res["mid"][0]["d2"] < 0.0, "D2 must move opposite to reward after the sign flip"
    for state, (sb0, sb1) in res.items():
        assert sb1["d1"] >= 0.0 and sb1["d2"] <= 0.0, (state, sb1)

    # --- SIZE: sb1/sb0 ratio must equal the soft-bound factor (n=1).
    def ratio(state, key):
        sb0, sb1 = res[state]
        assert sb0[key] != 0.0, (state, key, sb0)
        return sb1[key] / sb0[key]

    assert abs(ratio("mid", "d1") - 0.5) < 1e-3, ratio("mid", "d1")     # up factor (wmax-mid)/R = 0.5
    assert abs(ratio("mid", "d2") - 0.5) < 1e-3, ratio("mid", "d2")     # down factor (mid-wmin)/R = 0.5
    assert abs(ratio("ceiling", "d2") - 1.0) < 1e-3, ratio("ceiling", "d2")  # down free at ceiling
    assert abs(ratio("floor", "d1") - 1.0) < 1e-3, ratio("floor", "d1")      # up free at floor

    # --- EXACT endpoints (sb1 side; sb0 clips at the bound by construction).
    assert res["floor"][1]["d2"] == 0.0, \
        "D2 at floor: depression must be fully braked, got %r" % res["floor"][1]["d2"]
    assert res["ceiling"][1]["d1"] == 0.0, \
        "D1 at ceiling: potentiation must be fully braked, got %r" % res["ceiling"][1]["d1"]