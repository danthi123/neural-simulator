# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md (sections 5, 10, 13)
# Expected-fail: none (pins the bar logic of brain/eval/life_bars.py on synthetic records)
import numpy as np

from brain.eval import life_bars as LBARS


def _probe(night, arm, rows=(), span=(), summary=None, transfer=None, line="trunk"):
    s = {"H_right": 0, "H_n": 0, "never_false": 0, "never_n": 3, "collapse": 0.1}
    s.update(summary or {})
    p = {"night": night, "arm": arm, "line": line, "rows": list(rows), "span": list(span), "summary": s, "store": 1,
         "adapter_norm": 1.0}
    if transfer is not None:
        p["transfer"] = transfer
    return p


def _life(seed, probes, ledger=None, **kw):
    r = {"seed": seed, "ledger": ledger or {}, "schedule": [], "probes": probes, "nights": [], "lived": []}
    r.update(kw)
    return r


LED = {f"f{i:02d}": {"cls": "F", "told_days": [1], "keys": ["k"], "telling": "t"} for i in range(4)}


def _p0_life(seed, rec_old, rec_young, lo_old, lo_young, never_false):
    probes = []
    # age 1 night = harness age 0 = asked on night 1 (told day 1); age >=10 nights = harness age >=9 = night 10+
    for night, rec, lo in ((1, rec_young, lo_young), (10, rec_old, lo_old)):
        rows = [{"kind": "rotating", "item": "f00", "ok": ok_, "brain": "", "you": ""} for ok_ in rec]
        span = [{"item": "f01", "cls": "F", "age": night - 1, "n_told": 1, "logodds": lo}]
        probes.append(_probe(night, "hippocampus-off", rows, span, {"never_false": never_false}))
    return _life(seed, probes, LED)


def test_need_and_count_rule():
    assert LBARS._need(6) == 5 and LBARS._need(3) == 3 and LBARS._need(1) == 1
    r = LBARS.count_rule({1: True, 2: True, 3: True, 4: True, 5: True, 6: False})
    assert r["ok"] and r["count"] == 5
    r = LBARS.count_rule({1: True, 2: True, 3: True, 4: True, 5: None, 6: False})   # undefined counts as not passing
    assert not r["ok"] and r["defined"] == 5
    assert LBARS.count_rule({1: True, 2: False}, need=2)["ok"] is False


def test_combine_and_label():
    assert LBARS.combine(["PASS", "PASS"]) == "PASS"
    assert LBARS.combine(["PASS", "UNDEFINED"]) == "UNDEFINED"
    assert LBARS.combine(["FAIL", "UNDEFINED"]) == "FAIL"
    assert LBARS._label("PASS", 6) == "PASS" and LBARS._label("FAIL", 3) == "DEV (3 seeds)"
    assert LBARS._label("UNDEFINED", 3) == "DEV (3 seeds) UNDEFINED"


def test_permutation_deterministic_and_directional():
    g = [(np.array([5, 6, 7, 5.5, 0, 0.5, -1, 0.2]), 4) for _ in range(3)]
    o1, p1 = LBARS.perm_label_p(g, np.random.default_rng(LBARS.RNG_SEED))
    o2, p2 = LBARS.perm_label_p(g, np.random.default_rng(LBARS.RNG_SEED))
    assert (o1, p1) == (o2, p2)
    assert p1 < 0.05
    flat = [(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 3)]
    _, pf = LBARS.perm_label_p(flat, np.random.default_rng(1))
    assert pf > 0.9
    assert LBARS.perm_label_p([], np.random.default_rng(1)) == (None, None)


def test_p0_threshold_boundaries():
    ok = LBARS.p0_seed(_p0_life(1, (True, False), (True, True), 1.0, 3.0, 0.0))     # 0.5 vs 1.0: drop 0.5, floor 0.5
    assert ok["pass"] is True and ok["drop"] == 0.5
    # drop exactly 0.15 would pass, but with one obs per bin recall is 0/1: use the clauses to pin each leg
    no_drop = LBARS.p0_seed(_p0_life(1, (True, True), (True, True), 1.0, 3.0, 0.0))
    assert no_drop["pass"] is False and no_drop["clauses"]["drop>=0.15"] is False
    no_lo = LBARS.p0_seed(_p0_life(1, (True, False), (True, True), 4.0, 3.0, 0.0))
    assert no_lo["pass"] is False and no_lo["clauses"]["logodds_lower"] is False
    # floor: recall(age>=10) must sit 0.10 above the never-told false rate; recall there is 1.0 or 0.0 here
    floor = LBARS.p0_seed(_p0_life(1, (True, False), (True, True), 1.0, 3.0, 3))     # false rate 3/3 (never_n 3 per probe)
    assert floor["clauses"]["above_false+0.10"] is False


def test_p0_drop_threshold_exact():
    young = (True,) * 20
    old = (True,) * 17 + (False,) * 3             # 0.85 vs 1.00: drop exactly 0.15 passes
    assert LBARS.p0_seed(_p0_life(1, old, young, 1.0, 3.0, 0.0))["clauses"]["drop>=0.15"] is True
    old = (True,) * 18 + (False,) * 2             # drop 0.10 fails
    assert LBARS.p0_seed(_p0_life(1, old, young, 1.0, 3.0, 0.0))["clauses"]["drop>=0.15"] is False


def test_p0_empty_bin_is_undefined_and_not_passing():
    life = _life(2, [_probe(1, "hippocampus-off", [{"kind": "rotating", "item": "f00", "ok": True, "brain": "", "you": ""}], [])], LED)
    r = LBARS.p0_seed(life)
    assert r["pass"] is None
    res = LBARS.p0([life])
    assert res["valid"] is False and res["bar"]["verdict_core"] == "UNDEFINED"


def test_p0_valid_needs_5_of_6():
    good = [_p0_life(s, (True, False), (True, True), 1.0, 3.0, 0.0) for s in range(5)]
    bad = _p0_life(9, (True, True), (True, True), 1.0, 3.0, 0.0)
    assert LBARS.p0(good + [bad])["valid"] is True
    assert LBARS.p0(good[:4] + [bad, bad])["valid"] is False


def _sav_life(seed, shift):
    a = {f"a{i}": 5.0 + shift + 0.1 * i for i in range(8)}
    b = {f"b{i}": 1.0 + 0.1 * i for i in range(8)}
    null_a = {k: 4.0 for k in a}
    null_b = {k: 1.0 for k in b}
    row = {"11": {"prunedA": {
        "hippocampus-off": {"logodds": {"A": a, "B": b}, "G": 1.0, "G_null_sub": 2.0 + shift, "dR": 3, "dR_null_sub": 2},
        "both-off": {"logodds": {"A": null_a, "B": null_b}, "G": 3.0, "dR": 1}}}}
    return _life(seed, [], pruned_a={"day": 11, "A": list(a), "B": list(b), "savings": row})


def test_savings_gated_by_p0_and_computed_when_valid():
    lives = [_sav_life(s, 1.0) for s in range(6)]
    rng = np.random.default_rng(LBARS.RNG_SEED)
    out = LBARS.savings(lives, True, "VALID", rng)
    g = next(b for b in out["bars"] if b["id"] == "SAV-G")
    assert g["verdict_core"] == "PASS" and g["p"] < 0.05
    d = next(b for b in out["bars"] if b["id"] == "SAV-dR")
    assert d["verdict_core"] == "PASS"            # dR_null_sub 2 >= 1.5
    out2 = LBARS.savings(lives, False, "FAIL", np.random.default_rng(LBARS.RNG_SEED))
    g2 = next(b for b in out2["bars"] if b["id"] == "SAV-G")
    assert g2["verdict_core"] == "UNDEFINED" and g2["threshold_verdict"] == "PASS"      # numbers stay beside


def test_savings_missing_section_is_undefined_not_crash():
    out = LBARS.savings([_life(1, [])], True, "VALID", np.random.default_rng(1))
    assert all(b["verdict_core"] == "UNDEFINED" for b in out["bars"])


def test_savings_sign_rule_counts_missing_seed_as_failing():
    lives = [_sav_life(s, 1.0) for s in range(4)] + [_life(8, []), _life(9, [])]
    out = LBARS.savings(lives, True, "VALID", np.random.default_rng(LBARS.RNG_SEED))
    g = next(b for b in out["bars"] if b["id"] == "SAV-G")
    assert g["pooled"]["seeds_positive"] == 4 and g["verdict_core"] == "FAIL"      # 4 of 6 < 5


def test_h_wrong_counts():
    rows = [{"kind": "H", "item": "dan", "ok": False, "wrong_fetch": False, "brain": "dunno", "you": "name?"},
            {"kind": "H", "item": "nova", "ok": False, "wrong_fetch": True, "brain": "x", "you": "q"}]
    res = _life(1, [_probe(1, "intact", rows)], {"dan": {"cls": "H", "told_days": [1], "keys": ["dan"], "telling": None}})
    h = LBARS._h_wrong(res)
    assert h["wrong_fetch"] == 1 and h["not_right"] == 2
    clean = _life(1, [_probe(1, "intact", rows[:1])], res["ledger"])
    assert LBARS._h_wrong(clean)["wrong_fetch"] == 0


def test_cue_h_verdict_is_intact_not_right_not_wrong_fetch():
    led = {"dan": {"cls": "H", "told_days": [1], "keys": ["dan"], "telling": None}}
    right_wf = {"kind": "H", "item": "dan", "ok": True, "wrong_fetch": True, "brain": "Dan", "you": "name?"}
    miss_off = {"kind": "H", "item": "dan", "ok": False, "wrong_fetch": False, "brain": "dunno", "you": "name?"}
    # a right reply with a wrong fetch, and a miss on the null arm only: no wrong H answer on the intact arm
    res = _life(1, [_probe(1, "intact", [right_wf]), _probe(1, "both-off", [miss_off])], led)
    b = next(x for x in LBARS.cue([res], False, np.random.default_rng(0))["bars"] if x["id"] == "CUE-H")
    assert b["verdict_core"] == "PASS"
    res2 = _life(1, [_probe(1, "intact", [right_wf, miss_off])], led)
    b2 = next(x for x in LBARS.cue([res2], False, np.random.default_rng(0))["bars"] if x["id"] == "CUE-H")
    assert b2["verdict_core"] == "FAIL"


def test_b_set_is_not_low():
    led = {"a1": {"cls": "AB", "told_days": [2, 11]}, "b1": {"cls": "AB", "told_days": [11]}, "f1": {"cls": "F", "told_days": [3]}}
    assert LBARS._b_ids(led) == {"b1"}
    assert LBARS._told_once(led, "b1", 12) is None and LBARS._told_once(led, "a1", 10) == 2
    assert LBARS._told_once(led, "a1", 11) is None and LBARS._told_once(led, "f1", 5) == 3


def test_consolidated_off_compared_on_same_night():
    k4 = {"sysadmin": 4, "home repair": 4}
    var = {"sysadmin": "X", "home repair": "Y"}
    tr = lambda T: {"T": T, "T0": 0.0, "dT": T, "per": {"sysadmin": [T], "home repair": [0.0]}, "k": k4, "variant": var}  # noqa: E731
    lives = []
    for s in range(6):
        lives.append(_life(s, [_probe(13, "intact", transfer=tr(5.0)), _probe(15, "intact", transfer=tr(0.1)),
                               _probe(15, "consolidated-off", transfer=tr(0.2))]))
    b = next(x for x in LBARS.transfer(lives, np.random.default_rng(0))["bars"] if x["id"] == "TRF-C")
    assert b["verdict_core"] == "FAIL"      # night-15 intact 0.1 < consolidated-off 0.2, although night-13 intact is 5.0


def test_kill_needs_two_consecutive_probe_nights():
    def life(shares):
        return _life(1, [_probe(i + 1, "intact", summary={"collapse": c}) for i, c in enumerate(shares)])
    assert LBARS.kill_check(life([0.1, 0.4, 0.1, 0.4]))["triggered"] is False
    k = LBARS.kill_check(life([0.1, 0.4, 0.5, 0.1]))
    assert k["triggered"] and k["night"] == 3 and k["first_night"] == 2
    assert LBARS.kill_check(life([0.39, 0.39]))["triggered"] is False


def test_language_guard_missing_record_is_undefined():
    life = _life(1, [_probe(1, "intact"), _probe(1, "hippocampus-off")], nights=[{"night": 1, "line": "trunk", "adapter_norm": 1.0}])
    out = LBARS.language([life])
    kl = next(b for b in out["bars"] if b["id"] == "LG-KL")
    assert kl["verdict_core"] == "UNDEFINED"


def _tr(k, T, per=None):
    return {"T": T, "T0": 0.0, "dT": T, "k": {"sysadmin": k, "home repair": k}, "per": per or {"sysadmin": [T] * 4, "home repair": [0.0] * 4},
            "variant": {"sysadmin": "X", "home repair": "Y"}}


def test_transfer_bars_and_nosleep_undefined():
    lives = []
    for s in range(6):
        pr = []
        for night, k, T in ((3, 1, 0.1), (6, 2, 0.3), (13, 4, 0.8)):
            pr.append(_probe(night, "intact", transfer=_tr(k, T)))
        pr.append(_probe(15, "consolidated-off", transfer=_tr(4, 0.1)))
        pr.append(_probe(15, "both-off", transfer=_tr(4, 0.0)))
        lives.append(_life(s, pr))
    # intact has k=4 only at night 13: the arms probed on full nights are read at 15 (still k=4 there)
    for r in lives:
        r["probes"].append(_probe(15, "intact", transfer=_tr(4, 0.8)))
    out = LBARS.transfer(lives, np.random.default_rng(LBARS.RNG_SEED))
    by = {b["id"]: b for b in out["bars"]}
    assert by["TRF-T4"]["verdict_core"] == "PASS"
    assert by["TRF-K"]["verdict_core"] == "PASS"
    assert by["TRF-C"]["verdict_core"] == "PASS"
    assert by["TRF-N0"]["verdict_core"] == "UNDEFINED"
    assert by["TRF-B0"]["verdict_core"] in ("PASS", "UNDEFINED", "FAIL")


def test_transfer_negative_direction_fails():
    lives = [_life(s, [_probe(13, "intact", transfer=_tr(4, -0.5))]) for s in range(6)]
    out = LBARS.transfer(lives, np.random.default_rng(LBARS.RNG_SEED))
    assert next(b for b in out["bars"] if b["id"] == "TRF-T4")["verdict_core"] == "FAIL"


def test_analyse_runs_on_empty_sections():
    A = LBARS.analyse([_p0_life(1, (True, False), (True, True), 1.0, 3.0, 0.0)])
    assert {b["id"] for b in A["bars"]} >= {"P0", "SAV-G", "CUE-R", "TRF-T4", "LG-KL", "KILL-10"}
    assert all(b["verdict"].startswith("DEV (1 seeds)") for b in A["bars"])
    assert isinstance(LBARS.render(A), str)
