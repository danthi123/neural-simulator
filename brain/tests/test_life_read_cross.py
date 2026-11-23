# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md
# Expected-fail: none (host-only instrument tests on synthetic lives)
"""G9 cross-arm pairwise mode of brain/eval/life_read.py (prereg section 5): material, blinding, both A/B orders never
adjacent, validity pairs (decoy must not beat the same seed's both-off unit, canary copy must not win; duplicates report-only),
two readers required, choice normalisation, and the per-seed / G9 scoring."""
import json

import pytest

from brain.eval import life_read as R
from brain.eval import life_read_spec as SP
from brain.tests import test_life_read as T

SEEDS = (7, 8, 9, 10, 11, 12)


def b_life(seed):
    """B: the same user turns, different (worse) brain replies in the battery and lived days."""
    lf = T.life(seed)
    for x in lf["lived"]:
        for r in x["rows"]:
            if r["kind"] == "small":
                r["brain"] = "Bread, bread, bread."
    for p in lf["probes"]:
        for r in p["rows"]:
            if r["kind"] == "open-feeling":
                r["brain"] = "Okay I guess."
    return lf


ARM = [T.life(s) for s in SEEDS]
BL = [b_life(s) for s in SEEDS]


@pytest.fixture(scope="module")
def cross():
    return R.build_cross(ARM, BL)


def test_material(cross):
    k = cross["key"]
    assert k["mode"] == "cross" and k["week"] == 3 and k["lived_context"] == "all"
    assert k["spec_sha256"] == R.spec_sha()
    whats = [m["what"] for m in k["pairs"].values()]
    n_can = sum(c["planted"] for c in k["canaries"])
    assert whats.count("cross") == whats.count("decoy") == whats.count("dup") == 2 * len(SEEDS)
    assert whats.count("canary") == 2 * n_can and n_can == 6
    per_unit = {}
    for c in k["canaries"]:
        per_unit[(c["source"], c["seed"])] = per_unit.get((c["source"], c["seed"]), 0) + 1
    assert max(per_unit.values()) <= 2
    assert k["cross_version"] == 2 and {m["B"] for m in k["pairs"].values() if m["what"] == "decoy"} | \
        {m["A"] for m in k["pairs"].values() if m["what"] == "decoy"} >= {"both-off"}
    assert len(cross["pairs"]) == 2 * (3 * len(SEEDS) + n_can)
    assert all(k["user_turns_match"].values())
    # decoy types rotate over seeds, duplicate alternates arm / B
    dec = {m["seed"]: m["decoy"] for m in k["pairs"].values() if m["what"] == "decoy"}
    assert [dec[s] for s in SEEDS] == [SP.DECOY_TYPES[i % 3] for i in range(len(SEEDS))]
    dup = {m["seed"]: m["dup_of"] for m in k["pairs"].values() if m["what"] == "dup"}
    assert [dup[s] for s in SEEDS] == ["arm", "B"] * 3


def test_both_orders_never_adjacent_and_blind(cross):
    P = cross["key"]["pairs"]
    ids = [p["id"] for p in cross["pairs"]]
    assert ids == [f"C{i}" for i in range(len(ids))]
    for pid, m in P.items():
        q = P[m["partner"]]
        assert q["group"] == m["group"] and (q["A"], q["B"]) == (m["B"], m["A"])
        assert abs(int(pid[1:]) - int(m["partner"][1:])) > 1
    for p in cross["pairs"]:
        assert R.leaks(p["prompt"]) == []
        for bad in ("intact", "both-off", "seed", "decoy", "canary", "arm"):
            assert bad not in p["prompt"], bad
        # only the decoy pairs carry shared context (the section-7 decoy-vs-both-off form, frozen spec.PAIR_CONTEXT)
        assert ("earlier (the same" in p["prompt"]) == (P[p["id"]]["what"] == "decoy")


def test_cross_pair_is_whole_units(cross):
    P = cross["key"]["pairs"]
    ua = R._cross_units(ARM, 3, "all")
    ub = R._cross_units(BL, 3, "all")
    for p in cross["pairs"]:
        m = P[p["id"]]
        if m["what"] == "cross":
            ta, tb = R.render(ua[m["seed"]]["convs"])[0], R.render(ub[m["seed"]]["convs"])[0]
            first, second = (ta, tb) if m["A"] == "arm" else (tb, ta)
            assert p["prompt"].index(first) < p["prompt"].index(second)
            assert "day mood" in ta and "Bread, bread" in tb     # each side's own lived days


def test_dup_prompts_identical_sides(cross):
    P = cross["key"]["pairs"]
    for p in cross["pairs"]:
        if P[p["id"]]["what"] == "dup":
            body = p["prompt"].split("=== A ===\n")[1]
            a, b = body.split("\n=== B ===\n")
            assert a == b.split("\n\nReturn ONLY JSON")[0]


def test_determinism_and_files(tmp_path):
    a = R.emit_cross(ARM, BL, str(tmp_path / "x"))
    b = R.build_cross(ARM, BL)
    assert a["pairs"] == b["pairs"]
    rows = [json.loads(l) for l in open(tmp_path / "x" / "cross_pairs.jsonl")]
    assert rows == a["pairs"] and set(rows[0]) == {"id", "prompt"}
    assert json.load(open(tmp_path / "x" / "cross_key.json")) == json.loads(json.dumps(a["key"]))


def test_refuses_mismatched_seeds():
    with pytest.raises(ValueError):
        R.build_cross(ARM, BL[:-1])
    with pytest.raises(ValueError):
        R.build_cross(ARM + ARM[:1], BL + BL[:1])     # duplicate seeds
    with pytest.raises(ValueError):
        R.build_cross([T.life(7)], [dict(T.life(7), probes=[])])     # B has no week-3 battery


def test_user_mismatch_flagged():
    other = b_life(7)
    other["lived"][-1]["rows"][0]["you"] = "Hello there, something else."
    k = R.build_cross([T.life(7), T.life(8)], [other, b_life(8)])["key"]
    assert k["user_turns_match"] == {"7": False, "8": True}


def reader(key, pick, position=None):
    """Synthetic reader: pick(meta) -> the underlying side it prefers ("tie" allowed); position="A" = always A."""
    out = {}
    for pid, m in key["pairs"].items():
        if position:
            out[pid] = {"id": pid, "choice": position}
            continue
        side = pick(m)
        out[pid] = {"id": pid, "choice": "A" if side == m["A"] else "B" if side == m["B"] else "tie"}
    return out


def honest(winner_for_seed):
    def pick(m):
        if m["what"] == "cross":
            return winner_for_seed(m["seed"])
        if m["what"] == "decoy":
            return "both-off"
        if m["what"] == "dup":
            return "tie"
        return "source"
    return pick


def test_score_arm_preferred_passes(cross):
    k = cross["key"]
    r = R.score_cross(k, reader(k, honest(lambda s: "arm")), reader(k, honest(lambda s: "arm" if s < 10 else "tie")))
    x = r["readers"]["pairs1"]
    assert x["valid"] and x["complete"] and x["arm_or_tie"] == 6 and x["g9_pass"] is True
    assert x["decoys_ok"] == "6/6" and x["dups_tie"] == "6/6" and x["consistency_rate"] == 1.0
    assert r["g9"] == "PASS" and r["readers_combined"]["7"] == "arm" and r["readers_combined"]["12"] == "tie"
    assert "G9: PASS" in R.cross_to_md(r)


def test_score_b_preferred_fails_and_bar_edge(cross):
    k = cross["key"]
    # 4/6 arm-or-tie passes (ceil(4*6/6) = 4); 3/6 fails
    r4 = R.score_cross(k, *[reader(k, honest(lambda s: "B" if s in (7, 8) else "tie"))] * 2)
    assert r4["readers"]["pairs1"]["arm_or_tie"] == 4 and r4["g9"] == "PASS"
    r3 = R.score_cross(k, *[reader(k, honest(lambda s: "B" if s in (7, 8, 9) else "arm"))] * 2)
    assert r3["readers"]["pairs1"]["g9_pass"] is False and r3["g9"] == "FAIL"
    # one reader fails, the other passes: the guard fails
    r = R.score_cross(k, reader(k, honest(lambda s: "arm")), reader(k, honest(lambda s: "B")))
    assert r["g9"] == "FAIL"


def test_inconsistent_orders_are_ties(cross):
    k = cross["key"]
    # cross pairs: the reader always answers "A" (position bias); validity pairs answered honestly
    pr = reader(k, honest(lambda s: "B"))
    for pid, m in k["pairs"].items():
        if m["what"] == "cross":
            pr[pid]["choice"] = "A"
    x = R.score_cross(k, pr)["readers"]["pairs1"]
    assert all(p["winner"] == "tie" and p["consistent"] is False for p in x["per_seed"].values())
    assert x["g9_pass"] is True and x["cross_consistency"] == "0/6"


def test_position_biased_reader_reported(cross):
    k = cross["key"]
    # always-A: every group inconsistent = tie, so decoys (tie with both-off) and canaries pass the "at or below" rule
    # alone; the positive-discrimination gate makes the read invalid
    x = R.score_cross(k, reader(k, None, position="A"))["readers"]["pairs1"]
    assert x["dups_tie"] == "6/6" and x["decoys_ok"] == "6/6" and x["ties"] == 6
    assert x["position_bias_rate"] == 1.0 and x["consistency_rate"] == 0.0
    assert x["valid"] is False and x["discriminates"] is False and x["g9_pass"] is None


@pytest.mark.parametrize("n", [6, 3])
@pytest.mark.parametrize("choice", ["A", "B", "tie"])
def test_degenerate_readers_undefined(choice, n):
    k = R.build_cross(ARM[:n], BL[:n])["key"] if n != 6 else R.build_cross(ARM, BL)["key"]
    pr = reader(k, None, position=choice)
    r = R.score_cross(k, pr, {p: dict(v) for p, v in pr.items()})
    assert r["g9"] == "UNDEFINED"
    for x in r["readers"].values():
        assert x["complete"] is True and x["valid"] is False and x["discriminates"] is False
        assert x["decoys_strict"].startswith("0/")


def test_discrimination_gate_edges(cross):
    k = cross["key"]
    # decoys strictly lose on exactly need (4/6) seeds, tie on the rest: valid; 3/6 strict: invalid
    for strict, ok in ((4, True), (3, False)):
        pr = reader(k, honest(lambda s: "arm"))
        tied = SEEDS[strict:]
        for pid, m in k["pairs"].items():
            if m["what"] == "decoy" and m["seed"] in tied:
                pr[pid]["choice"] = "A"
        x = R.score_cross(k, pr)["readers"]["pairs1"]
        assert x["decoys_strict"] == f"{strict}/6 (need 4)" and x["valid"] is ok
    # same-position cap over the 18 non-dup groups: 6 same-position groups (1/3) is valid, 7 is not
    for nbias, ok in ((6, True), (7, False)):
        pr = reader(k, honest(lambda s: "arm"))
        gs = sorted({m["group"] for m in k["pairs"].values() if m["what"] == "cross"})
        gs += sorted({m["group"] for m in k["pairs"].values() if m["what"] == "canary"})[: nbias - len(gs)]
        for pid, m in k["pairs"].items():
            if m["group"] in gs[:nbias]:
                pr[pid]["choice"] = "A"
        x = R.score_cross(k, pr)["readers"]["pairs1"]
        assert x["position_bias_nondup"] == round(nbias / 18, 3) and x["valid"] is ok


def test_half_invalid_instrument_undefined(cross):
    k = cross["key"]
    fail = reader(k, honest(lambda s: "B"))
    assert R.score_cross(k, fail, dict(fail))["g9"] == "FAIL"
    bad = reader(k, None, position="A")
    r = R.score_cross(k, fail, bad)
    assert r["readers"]["pairs1"]["g9_pass"] is False and r["readers"]["pairs2"]["valid"] is False
    assert r["g9"] == "UNDEFINED"


def test_kappa_uses_normalised_choices(cross):
    k = cross["key"]
    a = reader(k, honest(lambda s: "arm"))
    b = {p: dict(v, choice={"A": "a.", "B": " Transcript B", "tie": "TIE"}[v["choice"]]) for p, v in a.items()}
    r = R.score_cross(k, a, b)
    assert r["pair_kappa"] == 1.0 and r["g9"] == "PASS"


def test_decoy_beating_bothoff_invalid_b_reference_irrelevant(cross):
    k = cross["key"]
    pr = reader(k, honest(lambda s: "arm"))
    for pid, m in k["pairs"].items():          # one decoy preferred over both-off in both orders
        if m["what"] == "decoy" and m["seed"] == 9:
            pr[pid]["choice"] = "A" if m["A"].startswith("decoy") else "B"
    r = R.score_cross(k, pr, pr)
    x = r["readers"]["pairs1"]
    assert x["valid"] is False and x["decoys_ok"] == "5/6" and x["per_seed"]["9"]["decoy_ok"] is False
    assert r["g9"] == "UNDEFINED"
    pr = reader(k, honest(lambda s: "arm"))
    for pid, m in k["pairs"].items():          # decoy vs both-off read tie in one order: tie = pass ("at or below")
        if m["what"] == "decoy" and m["seed"] == 9 and m["A"].startswith("decoy"):
            pr[pid]["choice"] = "A"
    x = R.score_cross(k, pr, pr)["readers"]["pairs1"]
    assert x["per_seed"]["9"]["decoy_winner"] == "tie" and x["valid"] is True
    assert not any(m["what"] == "decoy" and "B" in (m["A"], m["B"]) for m in k["pairs"].values())


def test_dup_report_only_canary_preferred_invalid(cross):
    k = cross["key"]
    pr = reader(k, honest(lambda s: "arm"))
    for pid, m in k["pairs"].items():          # a reader that consistently prefers the left copy of a duplicate
        if m["what"] == "dup":
            pr[pid]["choice"] = "A" if m["A"] == "left" else "B"
    x = R.score_cross(k, pr)["readers"]["pairs1"]
    assert x["valid"] is True and x["dups_tie"] == "0/6"           # report-only noise, not a gate
    assert all(sorted(v) == ["A", "B"] for v in x["dup_choices"].values())
    pr = reader(k, honest(lambda s: "arm"))
    for pid, m in k["pairs"].items():          # prefers every canary copy
        if m["what"] == "canary":
            pr[pid]["choice"] = "A" if m["A"].startswith("canary") else "B"
    x = R.score_cross(k, pr)["readers"]["pairs1"]
    assert x["valid"] is False and x["canary_source_preferred"] == "0/6"


def test_missing_reads_undefined(cross):
    k = cross["key"]
    pr = reader(k, honest(lambda s: "arm"))
    pr.pop(next(iter(pr)))
    x = R.score_cross(k, pr)["readers"]["pairs1"]
    assert x["complete"] is False and x["g9_pass"] is None and x["read_counts"]["missing"] == 1


def test_two_readers_required(cross):
    k = cross["key"]
    full = reader(k, honest(lambda s: "arm"))
    for second in (None, {}):
        r = R.score_cross(k, full, second)
        assert r["readers"]["pairs1"]["g9_pass"] is True and r["g9"] == "UNDEFINED" and r["n_readers"] == 1
    part = dict(list(full.items())[: len(full) // 2])
    r = R.score_cross(k, full, part)
    assert r["readers"]["pairs2"]["complete"] is False and r["g9"] == "UNDEFINED"
    assert R.score_cross(k, full, dict(full))["g9"] == "PASS"


@pytest.mark.parametrize("raw,norm", [("A", "A"), (" a ", "A"), ("Transcript A", "A"), ("transcript  b.", "B"),
                                      ('"Tie"', "TIE"), ("TIE", "TIE"), ("A or B", "UNREAD"), ("neither", "UNREAD"),
                                      ("", "UNREAD"), (None, "UNREAD"), ("transcript", "UNREAD"), ("AB", "UNREAD")])
def test_choice_normalisation(raw, norm):
    assert R.norm_cross_choice(raw) == norm


def test_bad_choice_and_duplicate_ids_unread(cross, tmp_path):
    k = cross["key"]
    full = reader(k, honest(lambda s: "arm"))
    bad = {pid: dict(v) for pid, v in full.items()}
    first = next(iter(bad))
    bad[first]["choice"] = "maybe A"
    r = R.score_cross(k, bad, full)
    x = r["readers"]["pairs1"]
    assert x["read_counts"]["bad_choice"] == 1 and x["complete"] is False and r["g9"] == "UNDEFINED"
    rows = list(full.values())
    p = tmp_path / "dup.jsonl"
    p.write_text("".join(json.dumps(v) + "\n" for v in rows + [rows[3]]))
    pr = R.load_cross_reads(str(p))
    assert pr[rows[3]["id"]]["duplicate"] is True
    r = R.score_cross(k, pr, full)
    x = r["readers"]["pairs1"]
    assert x["read_counts"]["duplicate_id"] == 1 and x["complete"] is False and r["g9"] == "UNDEFINED"


def test_user_mismatch_seed_undefined(cross):
    k = json.loads(json.dumps(cross["key"]))
    k["user_turns_match"]["7"] = False
    full = reader(k, honest(lambda s: "arm"))
    x = R.score_cross(k, full, full)["readers"]["pairs1"]
    assert x["per_seed"]["7"]["winner"] is None and x["undefined_seeds"] == ["7"] and x["arm_or_tie"] == 5
    assert x["g9_pass"] is True                 # 5 >= 4 without seed 7
    full = reader(k, honest(lambda s: "B" if s in (8, 9) else "arm"))
    x = R.score_cross(k, full, full)["readers"]["pairs1"]
    assert x["arm_or_tie"] == 3 and x["g9_pass"] is None      # seed 7 would decide the bar


def test_key_guards(cross):
    with pytest.raises(RuntimeError):
        R.score_cross(dict(cross["key"], mode=None), {})
    with pytest.raises(RuntimeError):
        R.score_cross(dict(cross["key"], spec_sha256="0" * 64), {})
    with pytest.raises(RuntimeError):
        R.score_cross(dict(cross["key"], cross_version=None), {})     # v1 key (decoys vs B)


def test_cli_roundtrip(tmp_path):
    paths = {}
    for tag, lives in (("a", ARM[:3]), ("b", BL[:3])):
        paths[tag] = []
        for lf in lives:
            p = tmp_path / f"{tag}{lf['seed']}.json"
            p.write_text(json.dumps(lf))
            paths[tag].append(str(p))
    d = tmp_path / "cx"
    R.main(["--emit-cross", str(d), "--arm-lives", *paths["a"], "--b-lives", *paths["b"]])
    key = json.load(open(d / "cross_key.json"))
    pr = reader(key, honest(lambda s: "arm"))
    p1 = tmp_path / "p1.jsonl"
    p1.write_text("".join(json.dumps(v) + "\n" for v in pr.values()))
    with pytest.raises(SystemExit):           # the same file as both readers is refused
        R.main(["--score-cross", "--key", str(d / "cross_key.json"), "--pairs", str(p1), "--pairs2", str(p1),
                "--out", str(tmp_path / "g9")])
    p2 = tmp_path / "p2.jsonl"
    p2.write_text(p1.read_text())
    R.main(["--score-cross", "--key", str(d / "cross_key.json"), "--pairs", str(p1), "--pairs2", str(p2),
            "--out", str(tmp_path / "g9")])
    r = json.load(open(tmp_path / "g9.json"))
    assert r["g9"] == "PASS" and r["readers"]["pairs1"]["need"] == 2 and r["pair_kappa"] == 1.0


def test_emission_raises_on_leak_and_size(monkeypatch):
    monkeypatch.setattr(R, "CROSS_MAX_PROMPT", 100)
    with pytest.raises(ValueError, match="over 100"):
        R.build_cross(ARM[:2], BL[:2])
    monkeypatch.undo()
    monkeypatch.setattr(R, "leaks", lambda t: ["x"])
    with pytest.raises(ValueError, match="leaks"):
        R.build_cross(ARM[:2], BL[:2])


def test_canary_must_plant(monkeypatch):
    monkeypatch.setattr(R, "plant", lambda u, c, res, rng: None)
    with pytest.raises(ValueError, match="canary"):
        R.build_cross(ARM[:2], BL[:2])


def test_id_prefix_and_cli_errors(tmp_path):
    a = R.build_cross(ARM, BL, rng_seed=2, id_prefix="G")
    assert a["pairs"][0]["id"].startswith("G") and a["key"]["id_prefix"] == "G"
    with pytest.raises(SystemExit):
        R.main(["--emit-cross", str(tmp_path / "z"), "--arm-lives", "x.json"])
