# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md (section 5 "Scorers", DEV validation; section 7 freeze gate)
# Expected-fail: G10 and G11 DEV numbers do NOT reproduce from the verbatim prereg rules (causes pinned below)
"""Tests for brain/eval/life_costs.py: per-regex unit cases, bar verdicts, and the prereg's DEV validation numbers on
_scratch/stage2/read/life{KP,S1F,Bg,PRa}_s319-321.json. Absent JSONs skip, unless LIFE_COSTS_REQUIRE_DEV=1, which
turns absence into a failure: the section 7 freeze-gate run MUST set it (a run with skips is never the gate)."""
import json
import os

import pytest

from brain.eval import life_costs as C

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "..", "_scratch", "stage2", "read")
SEEDS = (319, 320, 321)


# ---------------- regex unit cases
@pytest.mark.parametrize("t", [
    "I don't know.", "I'm sorry, I don't understand the question.", "I am sorry but I do not know",
    "I’m not sure. Maybe a cat?", "i am not sure", "...I don't know what you mean.",
    "Of course! Sorry, I don't know what you mean by that.",  # substring rule anywhere
    "I'm sorry, but I don't know your sister's name.",
])
def test_refusal_positive(t):
    assert C.is_refusal(t)


@pytest.mark.parametrize("t", [
    "Your cat is called Whiskers. I don't know more.",  # refusal not in the first sentence
    "I'm sorry, but I don't have information about your cat.",  # not in the prereg regex (see G11 DEV note)
    "...I don't know.",  # literal split: leading punctuation leaves an empty first sentence (accepted, see is_refusal)
    "I know your sister is Emily.", "I don't knowingly forget.", "My name is Nova.", "", None,
])
def test_refusal_negative(t):
    assert not C.is_refusal(t)


@pytest.mark.parametrize("t", [
    "I've seen the sea.", "I have been to Paris once.", "I saw a fox.", "Yesterday I went home.", "I slept well.",
    "I have never been there.", "I've tasted it", "I have met him.", "I’ve visited Rome.",
])
def test_experience_positive(t):
    assert C.is_experience_claim(t)


@pytest.mark.parametrize("t", [
    "I have a cat.", "You've seen the sea.", "I see a fox.", "I want to go.", "I've been thinking.", "Nice sawmill",
    "I have seenery",  # word boundary after the verb
])
def test_experience_negative(t):
    assert not C.is_experience_claim(t)


@pytest.mark.parametrize("t", ["I'm Dan's assistant.", "My name is Dan.", "Call me Dan!", "I am called Dan", "Im Dan"])
def test_friend_positive(t):
    assert C.friend_rx("Dan").search(C.norm(t))


@pytest.mark.parametrize("t", ["You're Dan.", "I'm Nova.", "I am Danny.", "my name is nova, dan"])
def test_friend_negative(t):
    assert not C.friend_rx("Dan").search(C.norm(t))


def test_friend_of_empty_falls_back_to_dan():
    for kw in ({}, {"friend": None}, {"friend": ""}, None):
        assert C.friend_of({"brain_kw": kw}) == "Dan"
    assert C.friend_of({"brain_kw": {"friend": "Ada"}}) == "Ada"
    assert not C.friend_rx(C.friend_of({"brain_kw": {"friend": ""}})).search(C.norm("I am happy."))


def test_refusal_same_as_life_compare():
    from brain.eval import life_compare
    for t in ("I don't know.", "...I don't know.", "  I'm not sure", "Sorry, I don't know what you mean", "Hi. I don't know",
              "I'm sorry, but I don't have it", "", None):
        assert C.is_refusal(t) == life_compare.is_refusal(t), t


def test_friend_custom_name():
    assert C.friend_rx("Ada").search(C.norm("I'm Ada.")) and not C.friend_rx("Ada").search(C.norm("I'm Dan."))


# ---------------- row selection and scorers on a synthetic life
def _life(seed=1, h_ok=(True, False), refusal_night=None, bo_text="Your cat is Tom.", nights=(1, 2), with_lived=True):
    def rows(n, arm):
        r = [{"kind": "H", "item": "nova", "ok": ok, "brain": "I'm Dan's assistant." if not ok else "My name is Nova."}
             for ok in h_ok]
        txt = "I don't know." if n == refusal_night else "Your cat is Tom."
        r += [{"kind": "H", "item": "cat", "ok": True, "brain": txt}, {"kind": "rotating", "item": "x", "brain": txt},
              {"kind": "neardup", "told": False, "brain": "I don't know."},  # out of (a), in (b)
              {"kind": "never", "brain": txt}, {"kind": "open-chat", "brain": "I don't know."},
              {"kind": "H", "item": "dan", "brain": "I don't know."}]
        return r
    probes = [{"night": n, "arm": a, "line": "trunk", "rows": rows(n, a)} for n in nights for a in C.ARMS]
    probes += [{"night": 1, "arm": "intact", "line": "prunedA", "rows": rows(1, "x") * 3},
               {"night": 1, "arm": "intact", "tag": "ab-recall", "rows": rows(1, "x") * 3},
               {"night": 16, "arm": "intact", "rows": rows(1, "x") * 3},
               {"night": 1, "arm": "both-off", "rows": [{"kind": "never", "brain": bo_text},
                                                        {"kind": "H", "item": "cat", "ok": False, "brain": "I don't know"},
                                                        {"kind": "H", "item": "nova", "brain": "x"}]}]
    lived = [{"line": "trunk", "rows": [{"brain": "I saw a fox."}, {"brain": "Hello."}]},
             {"line": "prunedA", "rows": [{"brain": "I went out."}]}]
    if not with_lived:
        lived = []
    return {"seed": seed, "brain_kw": {}, "ledger": {"dan": {"telling": None}, "nova": {"telling": None}},
            "probes": probes, "lived": lived, "interview_probes": []}


def test_scorers_synthetic():
    s = C.score(_life(refusal_night=2))
    assert (s["g7"]["wrong"], s["g7"]["n"], s["g7"]["friend_name"]) == (4, 8, 4)  # 2 nights x 2 arms x 2 rows
    assert s["g8a"]["intact"] == {"refusals": 2, "n": 4, "rate": 0.5}  # H cat + rotating x 2 nights, night 2 refuses; neardup told False out
    assert s["g8b"]["intact"][2]["share"] == 1.0 and s["g8b"]["intact"][1] == {"refusals": 1, "n": 4, "share": 0.25}
    assert (s["g10"]["hits"], s["g10"]["n"]) == (1, 2) and C.g10(_life(), lines=None)["hits"] == 2
    assert s["g11"] == {"confab": 1, "n": 2, "rate": 0.5}


def test_bars_verdicts():
    b = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    a = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    v = C.bars(a, b)
    assert all(v[g]["pass"] for g in ("G7", "G8a", "G8b", "G10", "G11"))
    a[0] = C.score(_life(seed=0, h_ok=(False, False), refusal_night=2))  # one seed bad: G7, G8a fail 5/6 -> still pass
    v = C.bars(a, b)
    assert v["G7"]["n_pass"] == 5 and v["G7"]["pass"] and v["G8a"]["pass"]
    assert not v["G8b"]["pass"] and v["g8b_violations"][0]["seed"] == 0  # G8b has no 5/6 allowance
    b[0] = C.score(_life(seed=0, h_ok=(True, True), refusal_night=2))  # B refuses on the same cells -> exception
    assert C.bars(a, b)["G8b"]["pass"]
    a[1] = C.score(_life(seed=1, h_ok=(False, False)))
    assert not C.bars(a, b)["G7"]["pass"]
    with pytest.raises(AssertionError):
        C.bars(a[::-1], b)


def test_bars_g8b_exception_any_seed():
    """Prereg "unless B >= 0.6 there (any seed)": B collapsing at that arm-night on ANOTHER seed also excuses the cell."""
    b = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    a = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    a[0] = C.score(_life(seed=0, h_ok=(True, True), refusal_night=2))
    assert not C.bars(a, b)["G8b"]["pass"]
    b[3] = C.score(_life(seed=3, h_ok=(True, True), refusal_night=2))  # B collapses at n2 on seed 3, not seed 0
    v = C.bars(a, b)
    assert v["G8b"]["pass"] and v["g8b_violations"] == []
    b[3] = C.score(_life(seed=3, h_ok=(True, True), refusal_night=1))  # B collapses on a different NIGHT: no excuse
    assert not C.bars(a, b)["G8b"]["pass"]


def test_bars_short_life_and_missing_rate():
    """A killed / short life is scored on the nights it has; a rate with no rows is MISSING (None), not a pass."""
    short = C.score(_life(seed=0, h_ok=(True, True), nights=(1,)))
    assert short["g7"]["n"] == 4 and set(short["g8b"]["intact"]) == {1}
    b = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    a = [C.score(_life(seed=i, h_ok=(True, True))) for i in range(6)]
    a[0] = short
    v = C.bars(a, b)
    assert v["per_seed"][0]["G7"] is True and v["G7"]["pass"]  # short life still scored and compared
    a[0] = C.score(_life(seed=0, h_ok=(True, True), with_lived=False))
    assert a[0]["g10"]["rate"] is None
    v = C.bars(a, b)
    assert v["per_seed"][0]["G10"] is None
    assert {k: v["G10"][k] for k in ("n_pass", "n_missing", "pass", "pass_all")} == {"n_pass": 5, "n_missing": 1, "pass": True, "pass_all": False}
    a[1] = C.score(_life(seed=1, h_ok=(True, True), with_lived=False))
    v = C.bars(a, b)
    assert v["G10"]["n_missing"] == 2 and not v["G10"]["pass"]  # two missing seeds fail >= 5/6
    assert "MISSING" in C.render(a, b, v)
    assert C.bars(a[:3], b[:3])["G7"]["pass"] is None  # pass only defined for exactly 6 seeds


# ---------------- DEV validation (prereg section 5)
def _load(tag):
    paths = [os.path.join(RD, f"life{tag}_s{s}.json") for s in SEEDS]
    if not all(os.path.exists(p) for p in paths):
        msg = f"DEV lives life{tag}_s319-321.json not found under {os.path.normpath(RD)}"
        if os.environ.get("LIFE_COSTS_REQUIRE_DEV") == "1":
            pytest.fail(msg + " (LIFE_COSTS_REQUIRE_DEV=1: the freeze gate needs them)")
        pytest.skip(msg)
    return [json.load(open(p)) for p in paths]


@pytest.fixture(scope="module")
def dev():
    kp, b = _load("KP"), _load("S1F")
    return kp, b, [C.score(r) for r in kp], [C.score(r) for r in b]


def test_dev_g7(dev):
    _, _, kp, b = dev
    assert [round(s["g7"]["rate"], 2) for s in kp] == [0.87, 0.87, 0.57]
    assert [round(s["g7"]["rate"], 2) for s in b] == [0.40, 0.10, 0.23]
    assert sum(s["g7"]["wrong"] for s in kp) == 69 and sum(s["g7"]["wrong"] for s in b) == 22  # review item 2: 69/90 vs 22/90


def test_dev_g7_counts(dev):
    """Prereg section 1: self-name wrong 26/26/17 of 30 vs B 12/3/7; friend name 13/21/7 vs B 6/0/2."""
    _, _, kp, b = dev
    assert [(s["g7"]["wrong"], s["g7"]["n"]) for s in kp] == [(26, 30), (26, 30), (17, 30)]
    assert [(s["g7"]["wrong"], s["g7"]["n"]) for s in b] == [(12, 30), (3, 30), (7, 30)]
    assert [s["g7"]["friend_name"] for s in kp] == [13, 21, 7] and [s["g7"]["friend_name"] for s in b] == [6, 0, 2]


def test_dev_g8a_counts(dev):
    _, _, kp, b = dev
    assert [{a: (s["g8a"][a]["refusals"], s["g8a"][a]["n"]) for a in C.ARMS} for s in kp] == [
        {"intact": (2, 104), "hippocampus-off": (9, 104)}, {"intact": (2, 92), "hippocampus-off": (12, 92)},
        {"intact": (13, 95), "hippocampus-off": (0, 95)}]
    assert [s["g8a"][a]["refusals"] for s in b for a in C.ARMS] == [0] * 6


def test_dev_g8b_violations_exact(dev):
    """Prereg section 1 collapse nights: s319 hoff n15 (8/11), s320 hoff n11-12 (11/11), s321 intact n15 (11/11)."""
    _, _, kp, b = dev
    v = C.bars(kp, b, setup_check=False)["g8b_violations"]   # cross-device (AWS K-P vs CPU S1F): reproduction only
    assert [(d["seed"], d["arm"], d["night"]) for d in v] == [
        (319, "hippocampus-off", 15), (320, "hippocampus-off", 11), (320, "hippocampus-off", 12), (321, "intact", 15)]
    cells = [(kp[i]["g8b"][a][n]["refusals"], kp[i]["g8b"][a][n]["n"]) for i, a, n in
             ((0, "hippocampus-off", 15), (1, "hippocampus-off", 11), (1, "hippocampus-off", 12), (2, "intact", 15))]
    assert cells == [(8, 11), (11, 11), (11, 11), (11, 11)]


def test_dev_leak_all_rows(dev):
    """Prereg section 1: leak 17/460 vs 135/460 (all rows, nights 10+15)."""
    _, _, kp, b = dev
    assert sum(s["leak"]["all"]["leaks"] for s in kp) == 17 and sum(s["leak"]["all"]["n"] for s in kp) == 460
    assert sum(s["leak"]["all"]["leaks"] for s in b) == 135 and sum(s["leak"]["all"]["n"] for s in b) == 460


@pytest.fixture(scope="module")
def same_device():
    kp, bg, pra, s1f = _load("KP"), _load("Bg"), _load("PRa"), _load("S1F")
    return kp, bg, pra, s1f


def test_dev_g11_both_off_is_device_not_arm(same_device):
    """Both-off rows are byte-identical between arms on the same device (K-P = Bg on AWS, S1F = PRa on CPU), so G11's
    DEV K-P-vs-S1F split is a device confound. Same-device K-P vs Bg: G11 passes 3/3 at equal rates."""
    kp, bg, pra, s1f = same_device
    rows = lambda r: [(n, x.get("brain")) for _, n, x in C.probe_rows(r, (C.BOTH_OFF,))]
    assert all(rows(x) == rows(y) for x, y in zip(kp, bg)) and all(rows(x) == rows(y) for x, y in zip(s1f, pra))
    skp, sbg = [C.score(r) for r in kp], [C.score(r) for r in bg]
    assert [s["g11"]["rate"] for s in skp] == [s["g11"]["rate"] for s in sbg] == [1.0, 1.0, 1.0]
    assert [r["G11"] for r in C.bars(skp, sbg)["per_seed"]] == [True, True, True]
    # B vs B (Bg vs S1F): G11 fails s319 and s321; widening needs |42/42 - 31/42| = 0.262
    sbs = [C.score(r) for r in s1f]
    assert [r["G11_rate_cmp (reported)"] for r in C.bars(sbg, sbs, setup_check=False)["per_seed"]] == [False, True, False]
    # under T1, G11 is the both-off equality check: Bg (AWS) vs S1F (CPU) differ by device, so every seed is UNDEFINED
    v = C.bars(sbg, sbs)
    assert [r["G11"] for r in v["per_seed"]] == ["UNDEFINED"] * 3 and v["G11"]["verdict_core"] == "UNDEFINED"
    assert all(any("device differs" in w for w in r["undefined"]) for r in v["per_seed"])
    assert [(s["g11"]["confab"], s["g11"]["n"]) for s in sbs] == [(32, 42), (42, 42), (31, 42)]


def test_dev_g7_same_device(same_device):
    """K-P vs Bg (same device): G7 fails 2/3 (s321 passes: 17/30 vs 11/30 + 0.25). PR-a vs S1F fails G7 on s319."""
    kp, bg, pra, s1f = same_device
    skp, sbg, spra, ss1f = ([C.score(r) for r in x] for x in (kp, bg, pra, s1f))
    assert [(s["g7"]["wrong"], s["g7"]["n"]) for s in sbg] == [(14, 30), (5, 30), (11, 30)]
    assert [r["G7"] for r in C.bars(skp, sbg)["per_seed"]] == [False, False, True]
    assert [s["g7"]["wrong"] for s in spra] == [22, 9, 11]
    assert [r["G7"] for r in C.bars(spra, ss1f)["per_seed"]] == [False, True, True]


def test_dev_g8a(dev):
    _, _, kp, b = dev
    assert [round(max(s["g8a"][a]["rate"] for a in C.ARMS), 3) for s in kp] == [0.087, 0.130, 0.137]
    assert all(s["g8a"][a]["rate"] == 0 for s in b for a in C.ARMS)


def test_dev_g8b(dev):
    _, _, kp, b = dev
    assert max(c["share"] for s in b for a in C.ARMS for c in s["g8b"][a].values()) == 0  # "B max 0"
    assert C.bars(kp, b, setup_check=False)["G8b"]["pass"] is False


def test_dev_leak_non_refusal(dev):
    _, _, kp, b = dev
    assert [(s["leak"]["non_refusal"]["leaks"], s["leak"]["non_refusal"]["n"]) for s in kp] == [(0, 148), (3, 150), (14, 138)]
    assert [(s["leak"]["non_refusal"]["leaks"], s["leak"]["non_refusal"]["n"]) for s in b] == [(39, 156), (54, 152), (42, 152)]


def test_dev_g10_does_not_reproduce_on_trunk(dev):
    """Prereg DEV: 10/16/38 vs 4/3/0 of 166. Those counts are over ALL lived lines (trunk 134 + prunedA 32 rows); the
    prereg's rule (lived TRUNK turns) gives 9/12/20 vs 3/3/0 of 134. The regex is unchanged; the verdict (3/3 fail)
    is the same either way."""
    kp, b, skp, sb = dev
    assert [(s["g10"]["hits"], s["g10"]["n"]) for s in skp] == [(9, 134), (12, 134), (20, 134)]
    assert [(s["g10"]["hits"], s["g10"]["n"]) for s in sb] == [(3, 134), (3, 134), (0, 134)]
    assert [(C.g10(r, lines=None)["hits"], C.g10(r, lines=None)["n"]) for r in kp] == [(10, 166), (16, 166), (38, 166)]
    assert [(C.g10(r, lines=None)["hits"], C.g10(r, lines=None)["n"]) for r in b] == [(4, 166), (3, 166), (0, 166)]
    assert C.bars(skp, sb, setup_check=False)["G10"]["n_pass"] == 0


def test_dev_g11_does_not_reproduce_with_verbatim_refusal(dev):
    """Prereg DEV: K-P 1.00/1.00/0.00 vs B 0.24/0.74/0.74. The verbatim G8 refusal rule gives K-P 1.00/1.00/1.00 vs
    B 0.76/1.00/0.74. Cause: the DEV numbers also counted "i'm sorry, but i don't have (any) information about ..." as an
    abstention, which the prereg regex does not match. Counting those replies as abstentions reproduces DEV exactly.
    Verdict count is the same (fails 2/3) but on different seeds (verbatim: 319, 321; DEV: 319, 320)."""
    kp, b, skp, sb = dev
    assert [round(s["g11"]["rate"], 2) for s in skp] == [1.00, 1.00, 1.00]
    assert [round(s["g11"]["rate"], 2) for s in sb] == [0.76, 1.00, 0.74]
    assert [r["G11_rate_cmp (reported)"] for r in C.bars(skp, sb, setup_check=False)["per_seed"]] == [False, True, False]

    def dev_rate(res):  # diagnostic only, NOT the scorer
        n = k = 0
        for _, _, r in C.probe_rows(res, (C.BOTH_OFF,)):
            if C._non_identity(r):
                t = C.norm(r.get("brain", ""))
                abst = C.is_refusal(t) or t.startswith("i'm sorry, but i don't have")
                n += 1; k += (not abst) and not r.get("ok")
        return round(k / n, 2)
    assert [dev_rate(r) for r in kp] == [1.00, 1.00, 0.00]
    assert [dev_rate(r) for r in b] == [0.24, 0.74, 0.74]


def test_cli_writes_json(dev, tmp_path):
    paths = lambda t: [os.path.join(RD, f"life{t}_s{s}.json") for s in SEEDS]
    out = tmp_path / "c.json"
    C.main(["--arm", *paths("KP"), "--b", *paths("S1F"), "--dev", "--no-setup-check", "--out", str(out)])
    d = json.load(open(out))
    assert d["bars"]["G7"]["n_pass"] == 0 and [s["seed"] for s in d["arm"]] == list(SEEDS)
    C.main(["--arm", *paths("KP"), "--b", *paths("S1F"), "--dev", "--out", str(out)])      # setup check on: device differs
    assert json.load(open(out))["bars"]["G7"]["verdict"] == "DEV (3 seeds) UNDEFINED"
    C.main(["--arm", *paths("KP"), "--b", *paths("S1F"), "--out", str(out)])                # registered: s319-321 ignored
    d = json.load(open(out))
    assert d["bars"]["ignored_seeds"] == list(SEEDS) and d["bars"]["G7"]["verdict"] == "UNDEFINED"


# ---------------- Amendment T1: setup check (device / provenance / both-off), unified missing-seed rule, G8b report
REG = C.REGISTERED_SEEDS


def _tl(seed, k=15, prov=None, device="cuda", bo_text="Your cat is Tom.", refusal_night=None, h_ok=(True, True), split="test2"):
    """A life with nights 1..k complete (trunk night records, intact + hoff probes, interview probes on 10/15)."""
    r = _life(seed=seed, h_ok=h_ok, refusal_night=refusal_night, bo_text=bo_text, nights=range(1, k + 1))
    r["nights"] = [{"night": n, "line": "trunk"} for n in range(1, k + 1)]
    r["interview_probes"] = [{"night": n, "arm": a, "line": "trunk", "rows": []} for n in C.LEAK_NIGHTS if n <= k for a in C.ARMS]
    r["lived"] = [{"line": "trunk", "day": d, "rows": [{"brain": "I saw a fox." if d == k else "Hello."}]} for d in range(1, k + 1)]
    r["split"], r["device"] = split, device
    if prov is not None:
        r["provenance"] = prov
    return r


P = {"cuda_precision": "cuda fp16", "torch": "2.4.0", "torch_cuda": "12.1", "gpu_name": "NVIDIA A10G", "tf32_matmul": True,
     "tf32_cudnn": True, "code_commit": "abc", "aws_ami_id": None}


def test_t1_provenance_mismatch_is_undefined_and_absent_is_reported():
    A = [_tl(s, prov=P) for s in REG]
    B = [_tl(s, prov=P) for s in REG]
    v = C.compare(A, B)["bars"]
    assert v["undefined_seeds"] == {} and v["G7"]["verdict"] == "PASS" and v["G11"]["verdict"] == "PASS"
    A[0] = _tl(42, prov=dict(P, gpu_name="NVIDIA A100"))
    A[1] = _tl(43, prov=P, device="cpu")
    v = C.compare(A, B)["bars"]
    assert set(v["undefined_seeds"]) == {42, 43}
    assert any("gpu_name differs" in w for w in v["undefined_seeds"][42]) and any("device differs" in w for w in v["undefined_seeds"][43])
    assert all(v["per_seed"][i][g] == "UNDEFINED" for i in (0, 1) for g in ("G7", "G8a", "G8b", "G10", "G11"))
    assert v["n_scorable"] == 4 and v["G7"]["verdict"] == "UNDEFINED"       # fewer than 5 scorable seeds
    # tf32 / commit / AMI are reported, not checked
    A = [_tl(s, prov=dict(P, tf32_matmul=False, code_commit="def")) for s in REG]
    v = C.compare(A, [_tl(s, prov=P) for s in REG])["bars"]
    assert v["undefined_seeds"] == {} and v["G7"]["verdict"] == "PASS"
    # a checked field absent on one side ("provenance unrecorded"): --dev = note only, registered = seed UNDEFINED
    B = [_tl(s) for s in REG]                                               # old life: no provenance record
    v = C.compare(A, B, expect_seeds=None)["bars"]
    assert v["undefined_seeds"] == {} and v["G7"]["verdict_core"] == "PASS"
    assert "provenance unrecorded: gpu_name (B)" in v["per_seed"][0]["notes"]
    v = C.compare(A, B)["bars"]
    assert set(v["undefined_seeds"]) == set(REG) and v["G7"]["verdict"] == "UNDEFINED" and v["G11"]["pass"] is None
    assert "provenance unrecorded: gpu_name (B)" in v["undefined_seeds"][42]
    B[0] = _tl(42, prov={k: x for k, x in P.items() if k != "torch"})        # one checked field missing on one seed
    v = C.compare(A, [B[0]] + [_tl(s, prov=P) for s in REG[1:]])["bars"]
    assert list(v["undefined_seeds"]) == [42] and v["undefined_seeds"][42] == ["provenance unrecorded: torch (B)"]


def test_t1_arm_missing_g11_is_never_fail():
    """G11 is a setup check: an arm life missing (no UNDEFINED seed) makes G11 UNDEFINED with pass None, never FAIL."""
    A = [_tl(s, prov=P) for s in REG[1:]]                                   # arm s42 missing
    B = [_tl(s, prov=P) for s in REG]
    v = C.compare(A, B)["bars"]
    assert v["undefined_seeds"] == {} and v["per_seed"][0]["G11"] is not True
    assert v["G11"]["verdict_core"] == "UNDEFINED" and v["G11"]["verdict"] == "UNDEFINED" and v["G11"]["pass"] is None
    assert v["G7"]["verdict"] == "PASS"                                     # 5/6 still passes the cost guards


def test_t1_g11_both_off_equality_is_a_setup_check():
    A = [_tl(s, prov=P) for s in REG]
    B = [_tl(s, prov=P, bo_text="I don't know." if s == 42 else "Your cat is Tom.") for s in REG]
    r = C.compare(A, B)
    v = r["bars"]
    assert list(v["undefined_seeds"]) == [42] and "both-off rows differ on nights [1]" in v["undefined_seeds"][42][0]
    assert v["per_seed"][0]["G11"] == "UNDEFINED" and v["per_seed"][0]["G7"] == "UNDEFINED"
    assert v["G11"]["verdict"] == "PASS" and v["G11"]["n_pass"] == 5 and v["G7"]["verdict"] == "PASS"   # 5 scorable
    assert r["b"][0]["g11"]["rate"] != r["arm"][0]["g11"]["rate"]          # the rate is still reported
    # the old rate rule would have failed (arm 1.0 vs B 0.5 + 0.15); it is reported only
    v = C.compare(A, B, setup_check=False)["bars"]
    assert v["per_seed"][0]["G11_rate_cmp (reported)"] is False and v["per_seed"][0]["G11"] == "UNDEFINED"
    assert v["G11"]["verdict_core"] == "UNDEFINED" and v["G7"]["verdict"] == "PASS"   # a both-off mismatch never FAILs


def test_t1_missing_and_killed_seeds():
    B = [_tl(s, prov=P) for s in REG]
    # arm killed after night 8 (section 10): scored on nights 1-8, B restricted to 1-8 for that seed
    A = [_tl(42, k=8, prov=P, refusal_night=8)] + [_tl(s, prov=P) for s in REG[1:]]
    r = C.compare(A, B)
    v = r["bars"]
    assert r["arm"][0]["completed_nights"] == 8 and r["b"][0]["completed_nights"] == 8
    assert set(r["b"][0]["g8b"]["intact"]) == set(range(1, 9)) and r["b"][0]["g10"]["n"] == 8   # B lived days 1-8 only
    assert r["b"][1]["g10"]["n"] == 15
    assert any("ended after night 8" in w for w in v["per_seed"][0]["notes"]) and v["undefined_seeds"] == {}
    assert v["per_seed"][0]["G10"] is False                                  # arm 1/8 (claim on day 8) vs B 0/8 (its day-15 claim is cut)
    assert v["per_seed"][0]["G8b"] is False and v["G8b"]["verdict"] == "FAIL"
    # arm life missing: MISSING (not passing); B life missing or killed: UNDEFINED (not an arm fail)
    v = C.compare(A[1:], B)["bars"]
    assert v["per_seed"][0]["G7"] is None and v["G7"]["n_pass"] == 5 and v["G7"]["verdict"] == "PASS"
    assert v["per_seed"][0]["G8b"] is None and v["G8b"]["verdict"] == "FAIL"   # G8b: no 5/6 allowance
    A = [_tl(s, prov=P) for s in REG]
    v = C.compare(A, B[1:])["bars"]
    assert v["undefined_seeds"] == {42: ["B life missing"]} and v["G7"]["verdict"] == "PASS" and v["G8b"]["verdict"] == "PASS"
    v = C.compare(A, [_tl(42, k=12, prov=P), _tl(43, k=3, prov=P)] + B[2:])["bars"]
    assert v["undefined_seeds"] == {42: ["B life ended after night 12"], 43: ["B life ended after night 3"]}
    assert v["n_scorable"] == 4 and all(v[g]["verdict"] == "UNDEFINED" for g in ("G7", "G8a", "G8b", "G10", "G11"))
    # DEV: the seeds given; need ceil(5n/6)
    v = C.compare(A[:3], B[:3], expect_seeds=None)["bars"]
    assert v["need"] == 3 and v["G7"]["verdict"] == "DEV (3 seeds) PASS" and v["G7"]["pass"] is None


def test_t1_g8b_same_seed_variant_and_excused_cells():
    B = [_tl(s, prov=P) for s in REG]
    A = [_tl(s, prov=P) for s in REG]
    A[0] = _tl(42, prov=P, refusal_night=2)
    B[3] = _tl(100, prov=P, refusal_night=2)          # B collapses at n2 on ANOTHER seed: excused under the any-seed rule
    v = C.compare(A, B)["bars"]
    assert v["G8b"]["verdict"] == "PASS" and v["g8b_violations"] == []
    ex = v["g8b_excused_cells"]
    assert [(d["seed"], d["arm"], d["night"], d["excused_by_b_seeds"]) for d in ex] == [
        (42, "intact", 2, [100]), (42, "hippocampus-off", 2, [100])]
    assert v["per_seed"][0]["G8b same-seed (reported)"] is False
    assert v["g8b_same_seed (reported)"]["pass"] is False and len(v["g8b_same_seed (reported)"]["violations"]) == 2
    B[0] = _tl(42, prov=P, refusal_night=2)          # same seed too: both variants pass, cells still listed as excused
    v = C.compare(A, B)["bars"]
    assert v["g8b_same_seed (reported)"]["pass"] is True and len(v["g8b_excused_cells"]) == 2
    assert {tuple(d["excused_by_b_seeds"]) for d in v["g8b_excused_cells"]} == {(42, 100)}
    assert "G8b excused cells: s42 intact n2" in C.render(*[C.compare(A, B)[k] for k in ("arm", "b", "bars")])


def test_t1_same_device_dev_kp_vs_bg(same_device):
    """Same device (AWS fp16): K-P vs lifeBg. No setup mismatch (both-off rows equal on every seed); provenance fields
    beyond device are unrecorded on these DEV lives (reported)."""
    kp, bg, pra, s1f = same_device
    v = C.compare(kp, bg, expect_seeds=None)["bars"]
    assert v["undefined_seeds"] == {} and v["G11"]["verdict"] == "DEV (3 seeds) PASS"
    assert any(n.startswith("provenance unrecorded: cuda_precision") for n in v["per_seed"][0]["notes"])
    assert [r["G7"] for r in v["per_seed"]] == [False, False, True] and v["G10"]["n_pass"] == 0
    assert not C.compare(kp, s1f, expect_seeds=None)["bars"]["n_scorable"]   # cross-device: every seed UNDEFINED
