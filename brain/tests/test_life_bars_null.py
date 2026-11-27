# Capability: Lifelong learning
# Record: brain/eval/life_bars_null.py (permutation null for the life_bars composite verdicts)
# Expected-fail: none (synthetic lives only; no model is run)
import numpy as np

from brain.eval import life_bars as LB
from brain.eval import life_bars_null as N

SEEDS = (1, 2, 3, 4, 5, 6)


def _transfer_life(seed, rng, effect):
    probes = []
    for night, k in ((1, 1), (5, 2), (9, 4)):
        per = {"sysadmin": list(rng.normal(effect * (k / 4), 0.5, 8)), "home repair": list(rng.normal(0, 0.5, 8))}
        tr = N._retransfer({"T0": 0.0, "variant": {"sysadmin": "X", "home repair": "Y"}, "k": {"sysadmin": k, "home repair": k}},
                           per)
        probes.append({"night": night, "arm": "intact", "line": "trunk", "rows": [], "span": [], "transfer": tr})
    return {"seed": seed, "probes": probes, "ledger": {}, "schedule": []}


def _savings_life(seed, rng, effect):
    ids = [f"ab{i:02d}" for i in range(16)]
    a, b = ids[:8], ids[8:]
    base = {i: float(rng.normal(0, 3)) for i in ids}       # item difficulty shared by both copies
    off = {i: base[i] + float(rng.normal(0, 1)) + (effect if i in a else 0) for i in ids}
    nul = {i: base[i] + float(rng.normal(0, 1)) for i in ids}

    def probe(arm, lo, ok):
        return {"night": 5, "arm": arm, "line": "prunedA", "summary": {},
                "span": [{"item": i, "cls": "AB", "logodds": lo[i], "span_nll": 1.0} for i in ids],
                "rows": [{"kind": "ab", "item": i, "ok": bool(ok[i])} for i in ids]}
    ok_off = {i: rng.random() < (0.9 if (i in a and effect) else 0.5) for i in ids}
    ok_nul = {i: False for i in ids}
    return {"seed": seed, "ledger": {}, "schedule": [], "pruned_a": {"day": 5, "A": a, "B": b, "sameday_items": [], "report": {}},
            "probes": [probe("hippocampus-off", off, ok_off), probe("both-off", nul, ok_nul)]}


def _cue_life(seed, rng, effect):
    rows = []
    for arm, dl in (("intact", 1), ("hippocampus-off", 3)):
        for j in range(16):
            for rung in ("C0", "C1", "C2", "C3", "C4"):
                p = 0.5 if not effect else {"C0": 0.95, "C2": 0.9}.get(rung, 0.15)
                rows.append({"item": f"e{j:02d}", "track": "E", "delay": dl, "rung": rung, "ok": bool(rng.random() < p)})
    led = {f"e{j:02d}": {"cls": "E", "told_days": [1], "keys": ["k"], "telling": "t"} for j in range(16)}
    return {"seed": seed, "ledger": led, "schedule": [], "probes": [],
            "cue_probes": [{"night": 4, "arm": "intact", "rows": [r for r in rows if r["delay"] == 1]},
                           {"night": 4, "arm": "hippocampus-off", "rows": [r for r in rows if r["delay"] == 3]}]}


def _lives(make, effect, seed0):
    rng = np.random.default_rng(seed0)
    return [make(s, rng, effect) for s in SEEDS]


def test_no_signal_false_pass_near_nominal():
    for make, sec, bid in ((_transfer_life, "transfer", "TRF-T4"), (_savings_life, "savings", "SAV-G"),
                           (_cue_life, "cue", "CUE-R")):
        R = N.run_null(_lives(make, 0.0, 11), [sec], k=60, inner=200, seed=5)
        assert R["bars"][bid]["false_pass_rate"] <= 0.15, (bid, R["bars"][bid])
        assert R["composite"][sec]["false_pass_rate"] <= 0.15
        assert R["bars"][bid]["frac_ge"] > 0.02     # a no-signal observation is not at the top of its own null
    assert LB.N_PERM == 10000                       # the monkeypatch is restored


def test_strong_signal_ranked_top():
    for make, sec, bid in ((_transfer_life, "transfer", "TRF-T4"), (_savings_life, "savings", "SAV-G"),
                           (_cue_life, "cue", "CUE-R")):
        R = N.run_null(_lives(make, 3.0, 12), [sec], k=60, inner=200, seed=6)
        b = R["bars"][bid]
        assert b["observed_verdict"] == "PASS", (bid, b)
        assert b["frac_ge"] <= 1 / 30 and b["beyond_p99"] and b["rank"] == 1, (bid, b)
        assert R["bars"][bid]["false_pass_rate"] >= 0.0 and R["composite"][sec]["false_pass_rate"] <= 0.05


def test_shuffles_do_not_mutate_and_recompute_from_items():
    rng = np.random.default_rng(0)
    r = _transfer_life(1, rng, 2.0)
    before = [p["transfer"]["T"] for p in r["probes"]]
    r2 = N.shuffle_transfer(r, np.random.default_rng(1))
    assert [p["transfer"]["T"] for p in r["probes"]] == before
    t = r2["probes"][0]["transfer"]
    assert abs(t["T"] - (np.mean(t["per"]["sysadmin"]) - np.mean(t["per"]["home repair"]))) < 1e-3
    s = _savings_life(1, rng, 2.0)
    s2 = N.shuffle_savings(s, np.random.default_rng(2))
    assert sorted(s2["pruned_a"]["A"] + s2["pruned_a"]["B"]) == sorted(s["pruned_a"]["A"] + s["pruned_a"]["B"])
    assert s["pruned_a"]["A"] == [f"ab{i:02d}" for i in range(8)]


def test_p0_null_runs_on_v1_life():
    led = {f"f{i:02d}": {"cls": "F", "told_days": [1 + i % 2], "keys": ["k"], "telling": "t"} for i in range(6)}
    lives = []
    for s in SEEDS:
        probes = []
        for night in (1, 10):
            rows = [{"kind": "rotating", "item": "f00", "ok": night == 1} for _ in range(3)]
            span = [{"item": "f01", "cls": "F", "age": night - 1, "n_told": 1, "logodds": 2.0 if night == 1 else 0.0}]
            probes.append({"night": night, "arm": "hippocampus-off", "line": "trunk", "rows": rows, "span": span,
                           "summary": {"H_right": 0, "H_n": 0, "never_false": 0, "never_n": 3}})
        lives.append({"seed": s, "ledger": led, "schedule": [], "probes": probes})
    R = N.run_null(lives, ["p0"], k=5, inner=50, seed=1)
    assert "P0" in R["bars"] and R["bars"]["P0"]["null_stat"]["n"] > 0


def test_synthetic_seeds_labelled_and_run():
    lives = _lives(_cue_life, 3.0, 13)[:3]
    syn = N.synthetic_lives(lives, 6, 7)
    assert len(syn) == 6 and len({s["seed"] for s in syn}) == 6
    R = N.run_null(syn, ["cue", "transfer"], k=5, inner=50, seed=2)
    R["synthetic"] = "SIMULATED"
    assert "SIMULATED" in N.render(R)


def _ladder_life(seed, rng):
    rows = []
    for i in range(24):
        old = i % 2
        rows.append({"kind": "ladder" if i % 4 < 2 else "control", "item": f"l{i:02d}", "age_nights": 12 if old else 1,
                     "ok": bool(rng.random() < (0.1 if old else 0.9))})
    probe = {"night": 14, "arm": "hippocampus-off", "line": "trunk", "rows": [], "span": [], "age_ladder": rows,
             "summary": {"H_right": 0, "H_n": 0, "never_false": 0, "never_n": 3}}
    return {"seed": seed, "ledger": {}, "schedule": [], "probes": [probe], "age_ladder": {"plan": 1}}


def test_ladder_shuffle_p0_permutes_age_nights():
    rng = np.random.default_rng(3)
    r = _ladder_life(1, rng)
    snap = [dict(x) for x in r["probes"][0]["age_ladder"]]
    r2 = N.shuffle_p0(r, np.random.default_rng(4))
    new = r2["probes"][0]["age_ladder"]
    assert r["probes"][0]["age_ladder"] == snap                                        # input untouched
    assert sorted(x["age_nights"] for x in new) == sorted(x["age_nights"] for x in snap)   # a permutation of age_nights
    assert [x["age_nights"] for x in new] != [x["age_nights"] for x in snap]
    assert [(x["item"], x["ok"], x["kind"]) for x in new] == [(x["item"], x["ok"], x["kind"]) for x in snap]
    lives = [_ladder_life(s, rng) for s in SEEDS]
    R = N.run_null(lives, ["p0"], k=40, inner=50, seed=3)
    b = R["bars"]["P0"]
    assert b["observed_stat"] > 0.5                       # the real ladder has a large drop
    assert b["null_stat"]["n"] == 40 and abs(b["null_stat"]["mean"]) < 0.2      # label-shuffled drop is ~0


def test_shuffled_savings_recomputed_by_savings_table():
    from brain.eval import life as L
    rng = np.random.default_rng(9)
    life = _savings_life(1, rng, 6.0)
    base = L.savings_table(life)["5"]["prunedA"]["hippocampus-off"]["G_null_sub"]
    gs = [L.savings_table(N.shuffle_savings(life, np.random.default_rng(s)))["5"]["prunedA"]["hippocampus-off"]["G_null_sub"]
          for s in range(30)]
    assert base is not None and all(g is not None for g in gs)
    assert max(gs) < base or np.mean(gs) < base - 1e-6      # shuffling A/B destroys the real A>B savings
    assert any(abs(g - base) > 1e-9 for g in gs)


def test_observed_matches_life_bars_analyse():
    lives = _lives(_transfer_life, 3.0, 21)
    for l, c in zip(lives, _lives(_cue_life, 3.0, 22)):
        l["cue_probes"] = c["cue_probes"]
        l["ledger"] = c["ledger"]
        l["nights"] = []
        for pr in l["probes"]:
            pr["summary"] = {"collapse": 0.0, "H_right": 0, "H_n": 0, "never_false": 0, "never_n": 3}
    A = LB.analyse(lives)
    want = {b["id"]: b["verdict_core"] for b in A["bars"]}
    got = N._score(["savings", "cue", "transfer"], sorted(lives, key=lambda r: r["seed"]), LB.RNG_SEED)
    for sec in got.values():
        for bid, v in sec.items():
            assert v["verdict"] == want[bid], (bid, v, want[bid])
