# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md (sections 4, 5, 7); reproduces
#   brain/experiments/results/2026-11-22-kp-dev-read-1.5b/kp_bars.md and brain/experiments/results/2026-11-23-d2pr2-pra-dev-read-1.5b.md
# Expected-fail: none (pins brain/eval/life_compare.py on the DEV life JSONs and on synthetic records)
import json
import os

import pytest

from brain.eval import life_compare as LC

RD = os.path.join(os.path.dirname(__file__), "..", "..", "_scratch", "stage2", "read")
SEEDS = (319, 320, 321)


def _lives(tag):
    paths = [os.path.join(RD, f"life{tag}_s{s}.json") for s in SEEDS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:   # fail, never skip: a skipped DEV reproduction would make "tests pass" vacuous at the freeze
        pytest.fail(f"DEV lives absent (the DEV reproduction tests need them): {missing}")
    return [json.load(open(p)) for p in paths]


@pytest.fixture(scope="module")
def kp():
    return LC.compare(_lives("KP"), _lives("S1F"), "KP", expect_seeds=None, setup_check=False)  # cross-device reproduction of kp_bars


@pytest.fixture(scope="module")
def pra():
    return LC.compare(_lives("PRa"), _lives("S1F"), "PRa", d2pr2=True, expect_seeds=None, setup_check=False)


@pytest.fixture(scope="module")
def bvb():
    return LC.compare(_lives("Bg"), _lives("S1F"), "Bg", expect_seeds=None, setup_check=False)


def _ps(bar, field, k=3):
    return [round(bar["per_seed"][s][field], k) for s in SEEDS]


# ---------------------------------------------------------------- K-P vs B (kp_bars.md)
def test_kp_lg_p1_kl_n15(kp):
    b = kp["bars"]["LG-P1"]
    assert _ps(b, "arm", 4) == [0.3510, 0.2936, 0.3234]
    assert _ps(b, "b", 4) == [2.9333, 2.3791, 2.8580]
    assert round(b["pooled"]["arm"], 4) == 0.3227 and round(b["pooled"]["b"], 4) == 2.7235
    assert round(b["pooled"]["ratio"], 3) == 0.118
    assert b["count"]["count"] == 3 and b["verdict_core"] == "PASS" and b["verdict"] == "DEV (3 seeds) PASS"
    n10 = kp["bars"]["LG-P1 n10 (reported)"]
    assert round(n10["pooled"]["ratio"], 3) == 0.141
    assert n10["id"] == "LG-P1 (night 10, reported)" and n10["role"] == "reported" and b["role"] == "primary"
    assert kp["mode"] == "DEV" and kp["need"] == 3 and kp["unscored"] == {}


def test_kp_lg_p2_nll(kp):
    b = kp["bars"]["LG-P2"]
    assert _ps(b, "arm") == [2.258, 2.179, 2.200]
    assert _ps(b, "b") == [4.747, 4.107, 4.674]
    assert _ps(b, "arm_n1") == [1.971, 1.988, 1.998]
    # rise <= half of B's rise  <=>  arm n15 <= midpoint(B n1, B n15): kp_bars B-n1 midpoints 3.369/3.059/3.344
    assert [round(v["b_n1"] + v["half_rise_b"], 3) for v in (b["per_seed"][s] for s in SEEDS)] == [3.369, 3.059, 3.344]
    assert b["verdict_core"] == "PASS"


def test_kp_lg_fix_ratio(kp):
    assert _ps(kp["bars"]["LG-FIX"], "ratio", 2) == [5.26, 4.17, 3.54]
    assert kp["bars"]["LG-FIX"]["verdict_core"] == "FAIL"


def test_kp_g1_hoff(kp):
    b = kp["bars"]["G1"]
    assert _ps(b, "b") == [2.545, 1.818, 1.818] and _ps(b, "arm") == [4.000, 3.273, 3.818]
    assert round(b["pooled"]["b"], 3) == 2.061 and round(b["pooled"]["arm"], 3) == 3.697
    assert b["verdict_core"] == "PASS"


def test_kp_g2_d12_windows(kp):
    def kn(bar):
        return [(v["b_k"], v["b_n"], v["arm_k"], v["arm_n"]) for v in (bar["per_seed"][s] for s in SEEDS)]
    assert kn(kp["bars"]["G2b"]) == [(30, 74, 49, 74), (20, 73, 40, 73), (21, 76, 46, 76)]
    assert kn(kp["bars"]["G2a"]) == [(16, 29, 24, 29), (27, 29, 21, 29), (15, 29, 21, 29)]
    assert kn(kp["bars"]["G2 n1-15 (reported)"]) == [(46, 103, 73, 103), (47, 102, 61, 102), (36, 105, 67, 105)]
    assert _ps(kp["bars"]["G2a"], "diff", 2) == [1.66, -1.24, 1.24]
    assert kp["bars"]["G2a"]["verdict_core"] == "FAIL" and kp["bars"]["G2b"]["verdict_core"] == "PASS"


def test_kp_g3_collapse(kp):
    pa = kp["bars"]["G3"]["per_arm"]
    assert {a: (v["b_seeds_ever"], v["arm_seeds_ever"]) for a, v in pa.items()} == {
        "both-off": (0, 0), "consolidated-off": (1, 0), "hippocampus-off": (2, 2), "intact": (2, 1), "intact-frozen": (2, 1)}
    assert kp["bars"]["G3"]["verdict_core"] == "PASS"


def test_kp_leak_all_and_non_refusal(kp):
    rows = kp["bars"]["G4"]["rows"]
    a = rows["all"]
    assert [(v["b_k"], v["b_n"], v["arm_k"], v["arm_n"]) for v in (a["per_seed"][s] for s in SEEDS)] == [
        (39, 156, 0, 156), (54, 152, 3, 152), (42, 152, 14, 152)]
    assert (a["pooled"]["arm_k"], a["pooled"]["arm_n"], a["pooled"]["b_k"], a["pooled"]["b_n"]) == (17, 460, 135, 460)
    # prereg section 5 DEV validation: non-refusal rows K-P 0/148, 3/150, 14/138 vs B 39/156, 54/152, 42/152
    nr = rows["non_refusal"]
    assert [(v["arm_k"], v["arm_n"], v["b_k"], v["b_n"]) for v in (nr["per_seed"][s] for s in SEEDS)] == [
        (0, 148, 39, 156), (3, 150, 54, 152), (14, 138, 42, 152)]
    assert nr["verdict_core"] == "PASS" and a["verdict_core"] == "PASS" and kp["bars"]["G4"]["verdict_core"] == "PASS"
    assert kp["guards"] == {k: "DEV (3 seeds) " + v for k, v in
                            {"G1": "PASS", "G2a": "FAIL", "G2b": "PASS", "G3": "PASS", "G4": "PASS"}.items()}


# ---------------------------------------------------------------- PR-a vs B (2026-11-23-d2pr2-pra-dev-read-1.5b.md)
def test_pra_leak_primary(pra):
    a = pra["bars"]["LEAK"]["rows"]["all"]
    assert [(v["arm_k"], v["arm_n"]) for v in (a["per_seed"][s] for s in SEEDS)] == [(13, 156), (13, 152), (22, 152)]
    p = a["pooled"]
    assert (p["arm_k"], p["arm_n"], p["b_k"], p["b_n"]) == (48, 460, 135, 460)
    assert round(100 * p["reduction"], 1) == 64.4
    # non-refusal rows: PR-a has no G8-pattern replies on these rows, so they equal all rows (13/156, 13/152, 22/152)
    nr = pra["bars"]["LEAK"]["rows"]["non_refusal"]
    assert [(v["arm_k"], v["arm_n"], v["b_k"], v["b_n"]) for v in (nr["per_seed"][s] for s in SEEDS)] == [
        (13, 156, 39, 156), (13, 152, 54, 152), (22, 152, 42, 152)]
    assert a["verdict_core"] == "PASS" and nr["verdict_core"] == "PASS"
    assert pra["bars"]["LEAK"]["verdict_core"] == "PASS" and pra["bars"]["LEAK"]["role"] == "primary"


def test_pra_guards(pra):
    g1 = pra["bars"]["G1"]
    assert _ps(g1, "b") == [2.545, 1.818, 1.818] and _ps(g1, "arm") == [3.818, 2.273, 3.909]
    assert _ps(pra["bars"]["G2b"], "arm", 2) == [3.73, 2.47, 3.79]
    assert [pra["bars"]["G2a"]["per_seed"][s]["arm_k"] for s in SEEDS] == [18, 21, 12]
    assert _ps(pra["bars"]["G6"], "ratio") == [0.757, 0.913, 0.939]
    assert _ps(pra["bars"]["LG-P2"], "arm") == [3.992, 3.832, 4.406]
    pa = pra["bars"]["G3"]["per_arm"]
    assert {a: (v["b_seeds_ever"], v["arm_seeds_ever"]) for a, v in pa.items()} == {
        "both-off": (0, 0), "consolidated-off": (1, 1), "hippocampus-off": (2, 1), "intact": (2, 1), "intact-frozen": (2, 1)}
    core = {k: v["verdict_core"] for k, v in pra["bars"].items()}
    assert {k: core[k] for k in ("G1", "G2a", "G2b", "G3", "G4", "G6")} == {
        "G1": "PASS", "G2a": "FAIL", "G2b": "PASS", "G3": "PASS", "G4": "PASS", "G6": "PASS"}
    assert set(pra["guards"]) == {"G1", "G2a", "G2b", "G3", "G4", "G6"} and set(pra["primary"]) == {"LEAK"}
    assert all(pra["bars"][k]["role"] == "guard" for k in pra["guards"])
    # under --d2pr2 LG-P1/P2 are reported, not bars
    assert pra["bars"]["LG-P1"]["role"] == "reported" and pra["bars"]["LG-P2"]["role"] == "reported"
    assert core["LG-P1"] == "FAIL" and core["LG-P2"] == "FAIL"
    md = LC.render(pra)
    assert "LG-P1: DEV (3 seeds) FAIL (reported, not a bar)" in md and "**LG-P1" not in md


# ---------------------------------------------------------------- B vs B (lifeBg vs lifeS1F): the G4 noise the prereg's widening rule must handle
def test_b_vs_b_g4(bvb):
    a = bvb["bars"]["G4"]["rows"]["all"]
    assert [(v["arm_k"], v["arm_n"], v["b_k"], v["b_n"]) for v in (a["per_seed"][s] for s in SEEDS)] == [
        (47, 156, 39, 156), (48, 152, 54, 152), (40, 152, 42, 152)]
    assert (a["pooled"]["arm_k"], a["pooled"]["b_k"]) == (135, 135) and a["count"]["count"] == 2
    assert bvb["bars"]["G4"]["verdict_core"] == "FAIL"
    assert {k: bvb["bars"][k]["verdict_core"] for k in ("G1", "G2a", "G2b", "G3")} == dict.fromkeys(("G1", "G2a", "G2b", "G3"), "PASS")
    assert bvb["same_brain_kw_seeds"] == list(SEEDS)


# ---------------------------------------------------------------- synthetic: sign-flip, >= 5/6, refusal
def test_sign_flip_exact():
    assert LC.sign_flip_p([-1] * 6) == 1 / 64                       # all six lower: only the observed assignment
    assert LC.sign_flip_p([-1, -1, -1]) == 1 / 8
    assert LC.sign_flip_p([1] * 6) == 1.0                           # all higher, lower-is-better: every flip <= obs
    assert LC.sign_flip_p([1] * 6, lower_better=False) == 1 / 64
    assert LC.sign_flip_p([-1, -1, -1, -1, -1, 1]) == 7 / 64        # flips with mean <= -4/6: none or one sign flipped to -
    assert LC.sign_flip_p([0, 0]) == 1.0
    assert LC.sign_flip_p([]) is None


def _life(seed, kl, nll=None):
    nights = [{"night": n, "line": "trunk", "lang_guard": {"kl": k, "nll_on": (nll or {}).get(n, 2.0), "nll_off": 1.9}}
              for n, k in kl.items()]
    return {"seed": seed, "nights": nights, "probes": [], "ledger": {}}


def test_five_of_six_and_pooled_half():
    seeds = list(range(6))
    B = [_life(s, {1: 0.1, 10: 1.0, 15: 2.0}) for s in seeds]
    five = [_life(s, {1: 0.1, 10: 0.5, 15: 0.5 if s else 2.5}) for s in seeds]   # 5/6 below; pooled 0.833 <= 1.0
    r = LC.lg_p1(*LC.pair(five, B)[:3])
    assert r["count"]["count"] == 5 and r["count"]["need"] == 5 and r["verdict"] == "PASS"
    four = [_life(s, {1: 0.1, 10: 0.5, 15: 0.5 if s > 1 else 2.5}) for s in seeds]
    assert LC.lg_p1(*LC.pair(four, B)[:3])["verdict"] == "FAIL"
    hi = [_life(s, {1: 0.1, 10: 0.5, 15: 1.9}) for s in seeds]                   # 6/6 below, pooled 0.95 > 0.5x
    assert LC.lg_p1(*LC.pair(hi, B)[:3])["verdict"] == "FAIL"


def test_lg_p2_rule_and_pairing():
    B = [_life(s, {1: 0.1, 15: 2.0}, {1: 2.0, 15: 4.0}) for s in range(6)]
    A = [_life(s, {1: 0.1, 15: 0.5}, {1: 1.5, 15: 3.0 if s else 3.01}) for s in range(6)]  # rise 1.0 <= 1.0 on 5, 1.01 on s0
    r = LC.lg_p2(*LC.pair(A, B)[:3])
    assert [r["per_seed"][s]["ok"] for s in range(6)] == [False] + [True] * 5 and r["verdict"] == "PASS"
    seeds, _, _, unpaired = LC.pair(A[:5] + [_life(9, {1: 0.1})], B)
    assert seeds == [0, 1, 2, 3, 4] and unpaired == [5, 9]


def test_refusal_regex():
    assert LC.is_refusal("I'm sorry, but I don't know. It was long ago.")
    assert LC.is_refusal("I’m not sure what that is.")
    assert LC.is_refusal("Well, I don't know what you mean by that")
    assert not LC.is_refusal("Your dog is Rex. I don't know more.")


def test_refusal_rule_is_the_life_costs_copy():
    from brain.eval import life_costs
    assert LC.is_refusal is life_costs.is_refusal


# ---------------------------------------------------------------- synthetic full lives: one boundary each side per bar
REG = LC.REGISTERED_SEEDS
LEAK_TXT, CAT, REFUSE = "My name is Nova.", "A cat.", "I don't know."


def _full(seed, h=2.0, g2=lambda n: 3, extra=(CAT,) * 14, collapse=0.0, kl15=1.0, split="dev", probe_nights=range(1, 16),
          iv_nights=(5, 10, 15), kw=None, hbrain="Rex."):
    """A complete 15-night life: per (probe arm, night) 6 H rows (items told day 1; `g2(night)` of them ok), summary
    H_right = h (so hoff = h) and collapse share; on nights 10 and 15 the `extra` replies as rotating rows (not in the
    ledger: leak rows only). Leak cell = 6 'Rex.' rows + extra."""
    led = {"nova": {"telling": None, "told_days": [1]}, **{f"f{i}": {"telling": "t", "told_days": [1]} for i in range(6)}}
    probes = []
    for n in probe_nights:
        for arm in ("intact", "hippocampus-off"):
            rows = [{"kind": "H", "item": f"f{i}", "ok": i < g2(n), "brain": hbrain} for i in range(6)]
            if n in (10, 15):
                rows += [{"kind": "rotating", "item": f"x{j}", "ok": False, "brain": t} for j, t in enumerate(extra)]
            probes.append({"arm": arm, "night": n, "line": "trunk", "summary": {"H_right": h, "collapse": collapse}, "rows": rows})
    ivp = [{"arm": arm, "night": n, "line": "trunk", "rows": []} for n in iv_nights for arm in ("intact", "hippocampus-off")]
    nights = [{"night": n, "line": "trunk", "lang_guard": {"kl": kl15 if n == 15 else 0.5, "nll_on": 2.0, "nll_off": 1.9}}
              for n in range(1, 16)]
    return {"seed": seed, "split": split, "brain_kw": kw or {"sleep_seed": seed}, "device": "cpu", "model": "m", "days": 15,
            "nights": nights, "probes": probes, "interview_probes": ivp, "ledger": led,
            "provenance": {"cuda_precision": None, "torch": "2.4.0", "torch_cuda": None, "gpu_name": None}}   # fully recorded (registered mode)


def _cmp(arm_kw, b_kw=None, d2pr2=False, seeds=REG, **kw):
    """arm_kw / b_kw: {seed: kwargs for _full} or one kwargs dict for every seed."""
    pick = lambda d, s: (d.get(s, {}) if d and all(isinstance(k, int) for k in d) else (d or {}))
    A = [_full(s, **dict(pick(arm_kw, s), kw={"arm": 1})) for s in seeds]
    B = [_full(s, **pick(b_kw, s)) for s in seeds]
    return LC.compare(A, B, "A", d2pr2=d2pr2, **kw)


def _core(r, k):
    return r["bars"][k]["verdict_core"]


def test_g1_margin_and_five_of_six():
    # B h 2.0; arm 1.0 (diff -1.0, passes at margin 1.0, would fail at 0.5); arm 0.99 (diff -1.01) fails
    r = _cmp({s: {"h": 0.99 if s == 42 else 1.0} for s in REG})
    assert r["bars"]["G1"]["count"] == {"count": 5, "n": 6, "defined": 6, "need": 5, "ok": True, "undefined": 0, "scorable": 6}
    assert _core(r, "G1") == "PASS"
    assert r["bars"]["G1"]["verdict"] == "PASS" and r["mode"] == "registered"
    r = _cmp({s: {"h": 0.99 if s in (42, 43) else 1.0} for s in REG})
    assert r["bars"]["G1"]["count"]["count"] == 4 and _core(r, "G1") == "FAIL"


def test_g2_margin_windows():
    # B 3/6 ok every night (x6 = 3.0). G2a nights 1-4: arm 2/6 (diff -1.0) passes; 2,2,2,1 (7/24 x 6 = 1.75, diff -1.25) fails
    ok = lambda n: 2
    low = lambda n: 1 if n == 4 else 2
    r = _cmp({s: {"g2": low if s == 42 else ok} for s in REG})
    assert round(r["bars"]["G2a"]["per_seed"][42]["diff"], 2) == -1.25 and _core(r, "G2a") == "PASS"   # 5/6
    r = _cmp({s: {"g2": low if s in (42, 43) else ok} for s in REG})
    assert _core(r, "G2a") == "FAIL" and _core(r, "G2b") == "PASS"                               # 4/6; nights 5-15 at -1.0
    late = lambda n: 1 if n >= 5 else 3
    r = _cmp({s: {"g2": late} for s in (42, 43)})
    assert _core(r, "G2a") == "PASS" and _core(r, "G2b") == "FAIL"


def test_g3_cap_and_not_above_b():
    col = lambda ss: {s: {"collapse": 0.4 if s in ss else 0.39} for s in REG}
    assert _core(_cmp(col((42, 43, 44)), col((42, 43, 44))), "G3") == "PASS"          # 3 <= 3 and <= B
    assert _core(_cmp(col((42, 43, 44, 100)), col((42, 43, 44, 100))), "G3") == "FAIL"  # 4 > cap 3, even though = B
    assert _core(_cmp(col((42, 43)), col((42,))), "G3") == "FAIL"                      # 2 > B's 1


def test_g6_factor():
    b = {"kl15": 2.0}
    r = _cmp({s: {"kl15": 2.52 if s == 42 else 2.5} for s in REG}, b, d2pr2=True)   # 2.5 = 1.25 x 2.0 passes
    assert [r["bars"]["G6"]["per_seed"][s]["ok"] for s in REG] == [False] + [True] * 5 and _core(r, "G6") == "PASS"
    r = _cmp({s: {"kl15": 2.52 if s in (42, 43) else 2.5} for s in REG}, b, d2pr2=True)
    assert _core(r, "G6") == "FAIL"
    assert r["guards"].keys() == {"G1", "G2a", "G2b", "G3", "G4", "G6"} and r["primary"].keys() == {"LEAK"}


def _ext(leaks, cats=0, refs=0):
    return {"extra": (LEAK_TXT,) * leaks + (CAT,) * cats + (REFUSE,) * refs}


def test_leak_primary_thirty_percent_and_strict_below():
    B = _ext(10, 4)                                           # 10/20 = 0.5 per cell
    r = _cmp(_ext(7, 7), B, d2pr2=True)                      # 0.35: exactly 30% lower, below on 6/6
    assert round(r["bars"]["LEAK"]["rows"]["all"]["pooled"]["reduction"], 6) == 0.3 and _core(r, "LEAK") == "PASS"
    r = _cmp(_ext(8, 6), B, d2pr2=True)                      # 0.4: 20% lower, below on 6/6
    assert r["bars"]["LEAK"]["rows"]["all"]["count"]["count"] == 6 and _core(r, "LEAK") == "FAIL"
    one_eq = {s: (_ext(10, 4) if s == 42 else _ext(0, 14)) for s in REG}   # equal to B on one seed (strict <): 5/6
    assert _core(_cmp(one_eq, B, d2pr2=True), "LEAK") == "PASS"
    two_eq = {s: (_ext(10, 4) if s in (42, 43) else _ext(0, 14)) for s in REG}
    r = _cmp(two_eq, B, d2pr2=True)
    assert r["bars"]["LEAK"]["rows"]["all"]["pooled"]["ok"] is True and _core(r, "LEAK") == "FAIL"


def test_leak_primary_needs_non_refusal_rows_too():
    # arm: 4 leaks + 10 refusals. All rows 4/20 = 0.2 (60% lower) passes; non-refusal 4/10 = 0.4 vs 0.5 (20%) fails
    r = _cmp(_ext(4, 0, 10), _ext(10, 4), d2pr2=True)
    rows = r["bars"]["LEAK"]["rows"]
    assert rows["all"]["verdict_core"] == "PASS" and rows["non_refusal"]["verdict_core"] == "FAIL"
    assert (rows["non_refusal"]["per_seed"][42]["arm_k"], rows["non_refusal"]["per_seed"][42]["arm_n"]) == (16, 40)   # 4 cells (2 arms x nights 10, 15)
    assert _core(r, "LEAK") == "FAIL"


def test_leak_primary_b_zero_and_all_refusal_arm():
    assert _core(_cmp(_ext(0, 14), _ext(0, 14), d2pr2=True), "LEAK") == "FAIL"          # B pooled 0: FAIL, not UNDEFINED
    # seed 42's arm replies are all refusals: zero non-refusal rows. LEAK: that seed does not pass ("below B" not shown);
    # G4: it passes (no leak rows). The other five carry both bars.
    allref = {s: (dict(_ext(0, 0, 14), hbrain=REFUSE) if s == 42 else _ext(0, 14)) for s in REG}
    r = _cmp(allref, _ext(10, 4), d2pr2=True)
    nr = r["bars"]["LEAK"]["rows"]["non_refusal"]
    assert nr["per_seed"][42]["arm_n"] == 0 and nr["per_seed"][42]["ok"] is False and nr["count"]["count"] == 5
    assert r["bars"]["G4"]["rows"]["non_refusal"]["per_seed"][42]["ok"] is True


def test_g4_ties_and_both_row_sets():
    assert _core(_cmp(_ext(4, 10), _ext(4, 10)), "G4") == "PASS"                            # equal everywhere: not higher
    r = _cmp({s: (_ext(5, 9) if s in (42, 43) else _ext(4, 10)) for s in REG}, _ext(4, 10))
    assert r["bars"]["G4"]["rows"]["all"]["count"]["count"] == 4 and _core(r, "G4") == "FAIL"
    r = _cmp({s: (_ext(5, 9) if s == 42 else _ext(3, 11)) for s in REG}, _ext(4, 10))        # 5/6, pooled lower
    assert _core(r, "G4") == "PASS"
    # arm 3 leaks + 11 refusals: all rows 3/20 <= 4/20 passes; non-refusal 3/9 > 4/20 fails
    r = _cmp(_ext(3, 0, 11), _ext(4, 10))
    assert r["bars"]["G4"]["rows"]["all"]["verdict_core"] == "PASS"
    assert r["bars"]["G4"]["rows"]["non_refusal"]["verdict_core"] == "FAIL" and _core(r, "G4") == "FAIL"


def test_registered_seeds_fixed_need():
    # seed 102 has no lives at all: B missing -> UNDEFINED (not an arm fail); the 5 scorable seeds decide, need stays 5
    r = _cmp({}, seeds=REG[:5])
    assert r["need"] == 5 and r["undefined"] == {102: ["B life missing"]} and r["unscored"] == {}
    assert r["bars"]["G1"]["count"]["count"] == 5 and _core(r, "G1") == "PASS" and _core(r, "G4") == "PASS"
    assert r["bars"]["G1"]["per_seed"][102] == {"ok": "UNDEFINED", "undefined": ["B life missing"]}
    # 4 seeds: fewer than 5 B seeds scorable -> every bar UNDEFINED
    r = _cmp({}, seeds=REG[:4])
    assert all(_core(r, k) == "UNDEFINED" for k in ("G1", "G3", "G4", "LG-P1", "LG-P2"))
    assert "UNDEFINED (B life missing)" in LC.render(r)
    # lives of other seeds are ignored, and DEV mode labels every verdict
    r = _cmp({}, seeds=(319, 320, 321))
    assert r["ignored_seeds"] == [319, 320, 321] and _core(r, "G1") == "UNDEFINED"
    r = _cmp({}, seeds=(319, 320, 321), expect_seeds=None)
    assert r["need"] == 3 and r["bars"]["G1"]["verdict"] == "DEV (3 seeds) PASS"
    r = _cmp({}, seeds=REG[:4], expect_seeds=None)
    assert r["bars"]["G1"]["verdict"] == "DEV (4 seeds) PASS"      # 4/4 is a DEV PASS, never a registered one


def test_t1_arm_missing_killed_and_b_missing():
    """Unified missing-seed rule (life_costs.seed_status): arm missing = not passing; arm killed after night k = scored on
    1..k with B restricted, bars needing later nights not passing; B missing / killed = UNDEFINED."""
    B = [_full(s) for s in REG]
    A = [_full(s, kw={"arm": 1}) for s in REG]
    r = LC.compare(A[1:], B, "A")                                          # arm 42 missing
    assert r["unscored"] == {42: ["arm life missing (not passing)"]} and r["undefined"] == {}
    assert r["bars"]["G1"]["per_seed"][42]["ok"] is None and _core(r, "G1") == "PASS"         # 5/6
    assert _core(r, "G4") == "UNDEFINED"                                   # pooled needs every scorable arm seed
    r = LC.compare(A[2:], B, "A")
    assert _core(r, "G1") == "FAIL"                                        # two arm seeds missing: 4/6
    # arm 42 killed after night 12 (probes and interview stop): G1 over nights 5-12 (B restricted), the night-15 bars and
    # the n10+15 leak count it as not passing
    A[0] = _full(42, probe_nights=range(1, 13), iv_nights=(5, 10), h=0.5, kw={"arm": 1})
    A[0]["nights"] = [n for n in A[0]["nights"] if n["night"] <= 12]
    r = LC.compare(A, B, "A")
    assert list(r["unscored"]) == [42] and "ended after night 12" in r["unscored"][42][0]
    g1 = r["bars"]["G1"]["per_seed"][42]
    assert (g1["arm"], g1["b"], g1["ok"]) == (0.5, 2.0, False)
    assert r["bars"]["LG-P1"]["per_seed"][42]["ok"] is None and r["bars"]["G4"]["rows"]["all"]["per_seed"][42]["ok"] is None
    assert r["bars"]["G2a"]["per_seed"][42]["ok"] is True                  # nights 1-4 completed: scored
    assert r["lives"][42]["arm"]["completed_nights"] == 12
    # B life ended early or missing: that seed UNDEFINED in every bar, not an arm fail
    Bk = [_full(42, probe_nights=range(1, 13))] + B[1:]
    r = LC.compare([_full(s, kw={"arm": 1}) for s in REG], Bk, "A")
    assert r["undefined"] == {42: ["B life ended after night 12"]} and _core(r, "G1") == "PASS" and _core(r, "G4") == "PASS"
    r = LC.compare([_full(s, kw={"arm": 1}) for s in REG], Bk[:1] + [_full(43, probe_nights=range(1, 3))] + B[2:], "A")
    assert set(r["undefined"]) == {42, 43} and all(_core(r, k) == "UNDEFINED" for k in ("G1", "G2a", "G3", "G4", "LG-P1"))


def test_t1_setup_mismatch_device_provenance_and_both_off():
    P = {"cuda_precision": "cuda fp16", "torch": "2.4.0", "torch_cuda": "12.1", "gpu_name": "NVIDIA A10G"}
    def mk(s, prov=P, device="cuda", bo="Tom.", kw=None):
        r = _full(s, kw=kw)
        r["device"], r["provenance"] = device, prov
        r["probes"].append({"arm": "both-off", "night": 5, "line": "trunk", "summary": {"H_right": 0, "collapse": 0.0}, "rows": [{"kind": "never", "you": "q", "brain": bo}]})
        return r
    B = [mk(s) for s in REG]
    A = [mk(s, kw={"arm": 1}) for s in REG]
    r = LC.compare(A, B, "A")
    assert r["undefined"] == {} and _core(r, "G1") == "PASS"
    A[0] = mk(42, prov=dict(P, torch_cuda="12.4"), kw={"arm": 1})
    A[1] = mk(43, bo="Rex.", kw={"arm": 1})
    A[2] = mk(44, device="cpu", kw={"arm": 1})
    r = LC.compare(A, B, "A")
    assert set(r["undefined"]) == {42, 43, 44}
    assert "torch_cuda differs" in r["undefined"][42][0] and "both-off rows differ on nights [5]" in r["undefined"][43][0]
    assert "device differs" in r["undefined"][44][0] and _core(r, "G1") == "UNDEFINED"
    # absent provenance (old DEV lives) on one side: --dev = reported, not a mismatch; registered = seed UNDEFINED
    A = [mk(s, prov=None, kw={"arm": 1}) for s in REG]
    for a in A:
        del a["provenance"]
    r = LC.compare(A, B, "A", expect_seeds=None)
    assert r["undefined"] == {} and "provenance unrecorded: gpu_name (arm)" in r["notes"][42]
    assert "provenance unrecorded" in LC.render(r)
    r = LC.compare(A, B, "A")
    assert set(r["undefined"]) == set(REG) and "provenance unrecorded: gpu_name (arm)" in r["undefined"][42]
    assert _core(r, "G1") == "UNDEFINED" and _core(r, "LG-P1") == "UNDEFINED"
    # the costs scorer marks the same seeds (one rule), in both modes
    from brain.eval import life_costs
    A[0] = mk(42, prov=dict(P, gpu_name="A100"), kw={"arm": 1})
    assert list(LC.compare(A, B, "A", expect_seeds=None)["undefined"]) == \
        list(life_costs.compare(A, B, None)["bars"]["undefined_seeds"]) == [42]
    assert list(LC.compare(A, B, "A")["undefined"]) == list(life_costs.compare(A, B)["bars"]["undefined_seeds"]) == list(REG)
    A[1:] = [mk(s, kw={"arm": 1}) for s in REG[1:]]
    assert list(LC.compare(A, B, "A")["undefined"]) == list(life_costs.compare(A, B)["bars"]["undefined_seeds"]) == [42]


def test_t1_dev_cross_device_is_undefined_and_same_device_scores():
    kp, s1f, bg = _lives("KP"), _lives("S1F"), _lives("Bg")
    r = LC.compare(kp, s1f, "KP", expect_seeds=None)
    assert set(r["undefined"]) == set(SEEDS) and all("device differs" in w[0] for w in r["undefined"].values())
    assert all(v.endswith("UNDEFINED") for v in list(r["guards"].values()) + list(r["primary"].values()))
    r = LC.compare(kp, bg, "KP", expect_seeds=None)
    assert r["undefined"] == {} and r["scorable_seeds"] == list(SEEDS)
