"""Instrument tests for the long-delay temporal-credit benchmark on the production BG organ. They check the schedule,
the trace manipulations, the event decomposition (exact by construction), lesion persistence, the absence of a host
reward scalar and the verdict reader. They are not evidence for the capability."""
import math
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

import research.runners.long_delay_credit_bg as B
import research.runners.long_delay_credit_bg_verdict as V


def test_schedule_delays_decoys_and_pairing():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        tr = B.make_trial(rng, {})
        seen.add(tr["delay"])
        assert tr["decoy"] == "ctx_n"
        assert all(0 < on < tr["delay"] for on, _ in tr["distractors"])
        assert B.LATE_OUTCOME_GAP_MS[0] <= tr["late_gap"] <= B.LATE_OUTCOME_GAP_MS[1]
    assert seen == {125, 250, 500, 1000, 2000}
    tr = B.make_trial(np.random.default_rng(1), {"misleading": True})
    assert tr["decoy"] == ("ctx_b" if tr["cue"] == 0 else "ctx_a")
    # every arm sees the same trial sequence for a seed (paired design): the misleading flag draws nothing extra
    a = [B.make_trial(np.random.default_rng(5), {}) for _ in range(1)]
    b = [B.make_trial(np.random.default_rng(5), {"misleading": True}) for _ in range(1)]
    assert a[0]["cue"] == b[0]["cue"] and a[0]["delay"] == b[0]["delay"]


def test_trace_manipulations():
    full = B.arm_config("full")
    integ = B.arm_config("trace_ablated_integral")
    peak = B.arm_config("trace_ablated_peak")
    assert "tau" not in full
    assert integ["tau"] * integ["gain"] == pytest.approx(1000.0)          # equal kernel integral
    assert peak["gain"] == 1.0
    d = B.arm_config("dose_m8.0_tau300.0")
    assert d["fixed_mult"] == 8.0 and d["tau"] * d["gain"] == pytest.approx(1000.0)


class _FakeRun:
    def __init__(self, marks, lr):
        self.marks = marks

        class _C:
            reward_learning_rate = lr

        class _B:
            core_config = _C()
        self.b = _B()


def test_decomposition_is_exact():
    """Simulate a decaying eligibility with tags added in windows; the per-window contributions must sum to e(t0)."""
    tau, d = 50.0, math.exp(-1 / 50.0)
    rng = np.random.default_rng(2)
    e = rng.random(8)
    marks, t = [], 0
    for label, n in (("carry", 30), ("early", 20), ("delay", 40), ("interv", 15), ("decoy", 10),
                     ("terminal", 12), ("post", 5), ("outcome", 0)):
        marks.append((label, t, e.copy()))
        for _ in range(n):
            e = e * d + (rng.random(8) if label != "delay" else 0.0) * 0.01
            t += 1
    das = [0.6, 0.55, 0.5, 0.45]

    class P:
        da_baseline = 0.5
    dw = B.decompose(_FakeRun(marks, 2.0), das, P, tau, np.ones(8))
    A = 2.0 * sum((x - 0.5) * d ** i for i, x in enumerate(das))
    total = sum(dw.values())
    np.testing.assert_allclose(total, A * marks[-1][2], rtol=1e-9)
    assert set(dw) == {"carry", "iti", "early", "delay", "interv", "decoy", "terminal", "post"}


def _cells(over=None):
    good = {("credit", "full"): 0.9, ("credit", "trace_ablated_integral"): 0.5, ("credit", "trace_ablated_peak"): 0.5,
            ("credit", "shuffled_yoked"): 0.5, ("credit", "misleading_final_cue"): 0.85,
            ("credit", "memory_lesioned"): 0.9, ("credit", "critic_lesioned"): 0.8,
            ("credit", "reward_to_da_lesioned"): 0.5,
            ("credit", "dose_m0.5_tau300.0"): 0.85, ("credit", "dose_m8.0_tau300.0"): 0.5,
            ("credit", "dose_m8.0_tau3000.0"): 0.8,
            ("credit_immediate", "full"): 0.95, ("credit_immediate", "trace_ablated_integral"): 0.9,
            ("late_delayed", "full"): 0.85, ("late_delayed", "trace_ablated_integral"): 0.5,
            ("late_delayed", "misleading_final_cue"): 0.8, ("late_delayed", "memory_lesioned"): 0.5,
            ("late_delayed", "shuffled_yoked"): 0.5,
            ("late_immediate", "full"): 0.9, ("late_immediate", "trace_ablated_integral"): 0.9,
            ("late_immediate", "memory_lesioned"): 0.5}
    good.update(over or {})
    cells = {}
    for (s, a), v in good.items():
        for sd in V.GATE_SEEDS:
            cells[(s, a, sd)] = {"schedule": s, "arm": a, "seed": sd, "success": v, "response_rate": 1.0,
                                 "mem_hold_acc": 1.0, "void_floor_ceiling": False, "early_share": 0.7,
                                 "M_early": 1.0, "M_decoy": 0.1,
                                 "test": {"terminal": {"response_rate": 1.0, "acc_among_responders": 0.5}}}
    return cells


def test_verdict_reader():
    v = V.verdict(_cells())
    assert v["verdict"] == "GO", {k: x["state"] for k, x in v["gates"].items()}
    assert v["mechanisms"]["credit | slow eligibility tag (equal integral)"]["class"] == "NECESSARY"
    # a slower tag that buys no reach at matched magnitude -> NO-GO (the R2b outcome)
    assert V.verdict(_cells({("credit", "dose_m8.0_tau3000.0"): 0.5}))["verdict"] == "NO-GO"
    # a silent ablated arm cannot count as "abolished": it is VOID, so C1 cannot pass on it
    c = _cells()
    for sd in V.GATE_SEEDS:
        c[("credit", "trace_ablated_integral", sd)]["response_rate"] = 0.1
    assert V.verdict(c)["construct_validity"]["C1_trace_ablation_abolishes_credit"]["state"] != "PASS"
    # memory that does not hold leaves the late gates undefined, never passed
    c = _cells()
    for sd in V.GATE_SEEDS:
        c[("late_immediate", "full", sd)]["mem_hold_acc"] = 0.5
    v = V.verdict(c)
    assert v["gates"]["G5_memory_positive_control"]["state"] == "UNDEFINED" and v["verdict"] != "GO"


def test_one_trial_on_the_organ():
    os.environ["BRAIN_BG_CREDIT"] = "1"
    from research.runners.bg_corticostriatal_credit import CreditParams, Synapses, build_credit_selector, calibrate_da
    p = CreditParams()
    bridge, config = build_credit_selector(9, p)
    calibrate_da(bridge, config, p, n_steps=2000)
    syn = Synapses(bridge)
    fixed = syn.weights("proposal_0", "str_d1_0").copy()
    assert fixed.mean() > p.w_max                       # a fixed weight above the plastic clip bound
    n = syn.zero("mem_a", "mem_a")
    assert n > 0
    run = B.Run(bridge, config, p, learn=True)
    info, das = B.run_trial(run, B.make_trial(np.random.default_rng(9), {}), "credit", {},
                            np.random.default_rng(1))
    assert info["early"] in (-1, 0, 1) and info["terminal"] in (-1, 0, 1)
    assert len(das) == B.OUTCOME_MS
    np.testing.assert_array_equal(syn.weights("proposal_0", "str_d1_0"), fixed)   # the clip never touched it
    assert float(np.abs(syn.weights("mem_a", "mem_a")).max()) == 0.0              # the lesion holds
    assert float(bridge.core_config.current_reward_signal) == 0.0
