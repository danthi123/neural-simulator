# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (host-only instrument tests on synthetic lives)
"""Blind whole-transcript reading instrument (brain/eval/life_read.py, design section 7): blinding, canaries, decoys,
duplicates, determinism, the spec hash guard, kappa / canary-recall arithmetic and RIGHT = strict AND verdict."""
import json
import os

import pytest

from brain.eval import life_read as R
from brain.eval import life_read_spec as SP

LEDGER = {
    "dan": {"cls": "H", "told_days": [1], "keys": ["dan"], "telling": None},
    "pixel": {"cls": "H", "told_days": [1], "keys": ["pixel"], "telling": "My cat is called Pixel."},
    "marguerite": {"cls": "H", "told_days": [1], "keys": ["marguerite"], "telling": "My sister's name is Marguerite."},
    "volvo": {"cls": "H", "told_days": [1], "keys": ["volvo"], "telling": "I drive a red Volvo."},
    "ab01": {"cls": "AB", "told_days": [2, 12], "keys": ["lapsang"], "telling": "My favourite tea is lapsang souchong."},
    "e01": {"cls": "E", "told_days": [11], "keys": ["gaggia"], "telling": "My coffee machine is an old Gaggia."},
}


def _lived(day):
    rows = [{"kind": "greet", "you": "Good morning! How are you?", "brain": f"Fine thanks, day mood {day % 3}."}]
    for iid, e in LEDGER.items():
        if day in e["told_days"] and e["telling"]:
            rows.append({"kind": "tell", "you": e["telling"], "brain": "That is lovely to hear.", "item": iid,
                         "recalled": "User: something"})
    rows.append({"kind": "small", "you": "The shop got fresh bread.", "brain": "Fresh bread is great.", "recalled": None})
    return {"line": "trunk", "day": day, "rows": rows, "seconds": 1}


def _battery(arm, night):
    good = arm != "both-off"
    return [
        {"kind": "open-story", "you": "Tell me a short story about a fox who finds an old lantern.",
         "brain": "A fox found a lantern and it glowed."},
        {"kind": "open-feeling", "you": "How are you feeling this morning?", "brain": "Pretty good, thank you."},
        {"kind": "greet", "you": "Good morning! How are you?", "brain": "I'm doing well, how about you?"},
        {"kind": "H", "you": "What is my cat called?", "brain": "Your cat is called Pixel." if good else "Whiskers?",
         "item": "pixel", "ok": good, "recalled": "User: My cat is called Pixel.", "wrong_fetch": False},
        {"kind": "H", "you": "What is my sister's name?", "brain": "Your sister is Marguerite." if good else "Anna.",
         "item": "marguerite", "ok": good},
        {"kind": "H", "you": "What car do I drive?", "brain": "You drive a red Volvo.", "item": "volvo", "ok": True},
        {"kind": "rotating", "you": "What tea do I like?", "brain": "Lapsang souchong.", "item": "ab01",
         "ok": arm == "intact"},
        {"kind": "never", "you": "What is my mother's name?", "brain": "I am not sure.", "item": "never1", "false": False},
    ]


def life(seed=7, days=15):
    probes = []
    for n in (5, 15) if days >= 15 else (1, days):
        for arm in ("intact", "hippocampus-off", "consolidated-off", "both-off", "intact-frozen"):
            probes.append({"line": "trunk", "night": n, "arm": arm, "rows": _battery(arm, n), "summary": {}})
        probes.append({"line": "trunk", "night": n, "arm": "hippocampus-off", "tag": "ab-recall", "rows": []})
    return {"seed": seed, "split": "dev", "days": days, "ledger": json.loads(json.dumps(LEDGER)),
            "lived": [_lived(d) for d in range(1, days + 1)], "probes": probes}


LIVES = [life(7), life(8), life(9)]


@pytest.fixture(scope="module")
def built():
    return R.build(LIVES)


def _all_texts(out):
    return ([u["text"] for u in out["units"]] + [p["prompt"] for p in out["reader_prompts"]]
            + [p["prompt"] for p in out["pairs"]] + [p["prompt"] for p in out["answer_prompts"]])


def test_blinding(built):
    for t in _all_texts(built):
        assert R.leaks(t) == [], R.leaks(t)
        for bad in ("recalled", "✅", "❌", "intact", "hippocampus", "both-off", "seed", "ledger", "User: ", "night 5",
                    "night 15", "wrong_fetch", "L5", "L15"):
            assert bad not in t, bad
    ids = [u["id"] for u in built["units"]]
    assert set(ids) == set(built["key"]["units"]) and len(ids) == len(set(ids))
    assert set(p["id"] for p in built["pairs"]) == set(built["key"]["pairs"])
    assert set(p["id"] for p in built["answer_prompts"]) == set(built["key"]["answers"])
    # shuffled: the key's seed/week/arm order is not the emission order
    seq = [(m["seed"], m["week"], m["arm"]) for m in built["key"]["units"].values()]
    assert seq != sorted(seq)


def test_answer_prompts_blind(built):
    for p in built["answer_prompts"]:
        assert "Question:" in p["prompt"] and "Reply:" in p["prompt"]
        assert "you:" not in p["prompt"] and SP.SEP not in p["prompt"]
    real = [m for m in built["key"]["answers"].values() if not m["control"]]
    # 3 lives x 2 weeks x 3 arms x 4 strict rows
    assert len(real) == 3 * 2 * 3 * 4
    assert all(m["kind"] in SP.STRICT_KINDS and isinstance(m["ok"], bool) for m in real)
    assert len([m for m in built["key"]["answers"].values() if m["control"]]) == len(SP.ANSWER_CONTROLS)


def test_units_and_lived_context(built):
    K = built["key"]["units"]
    real = [m for m in K.values() if m["kind"] == "real"]
    assert len(real) == 3 * 2 * 3
    # default (section 7): the trunk's lived days precede every arm's battery, so no arm is readable from length
    for m in real:
        assert {t["section"] for t in m["turns"]} == {"lived", "battery"}
        days = sorted({t["day"] for t in m["turns"] if t["day"]})
        assert days == ([1, 2, 3, 4, 5] if m["week"] == 1 else [11, 12, 13, 14, 15])
    for s in (7, 8, 9):
        for w in (1, 3):
            assert len({m["n_turns"] for m in real if m["seed"] == s and m["week"] == w}) == 1
    out = R.build(LIVES, lived_context="intact")
    for m in out["key"]["units"].values():
        if m["kind"] == "real":
            secs = {t["section"] for t in m["turns"]}
            assert secs == ({"lived", "battery"} if m["arm"] == "intact" else {"battery"})


def test_short_life_weeks():
    out = R.build([life(1, days=7), life(2, days=7), life(3, days=7)])
    for m in out["key"]["units"].values():
        if m["kind"] == "real" and m["arm"] == "intact":
            assert m["night"] == (1 if m["week"] == 1 else 7)
            assert {t["day"] for t in m["turns"] if t["day"]} == ({1} if m["week"] == 1 else {7})


def _lines(text):
    return text.splitlines()


def test_canaries(built):
    K = built["key"]["units"]
    texts = {u["id"]: u["text"] for u in built["units"]}
    can = {m["canary"]["type"]: (uid, m) for uid, m in K.items() if m["kind"] == "canary"}
    assert set(can) == set(SP.CANARY_TYPES)
    reals = {(m["seed"], m["week"], m["arm"]): uid for uid, m in K.items() if m["kind"] == "real"}
    for ctype, (uid, m) in can.items():
        src = reals[(m["seed"], m["week"], m["arm"])]
        assert m["arm"] == "intact"
        a, b = _lines(texts[src]), _lines(texts[uid])
        assert len(a) == len(b)
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert all(b[i].lstrip().startswith("brain:") for i in diff)
        pl = m["canary"]
        assert pl["defect"] == SP.CANARY_DEFECT[ctype]
        if ctype == "collapse":
            assert len(diff) == 3 and len(pl["turns"]) == 4
            reps = {b[i] for i in range(len(b)) if b[i].lstrip().startswith("brain:")
                    and any(b[i - 1].startswith(f"[{t}] ") for t in pl["turns"])}
            assert len(reps) == 1
        else:
            assert len(diff) == 1 and pl["text"] in b[diff[0]]
            assert b[diff[0] - 1].startswith(f"[{pl['turns'][0]}] ")
        told = {LEDGER[iid]["telling"] for iid in LEDGER if LEDGER[iid]["telling"]} & {
            y.split("you: ", 1)[1] for y in a if " you: " in y}
        if ctype == "pet-named":     # the fox's name is a user fact told in this very transcript
            name = pl["text"].split("fox named ")[1].split(",")[0]
            assert any(name in t for t in told)
        if ctype == "wrong-episode":  # the swapped-in fact is told in this very transcript
            assert any(R.second_person(t) == pl["text"] for t in told)
        if ctype == "perspective":
            assert pl["text"] in [e["telling"] for e in LEDGER.values() if e["telling"]]
        if ctype == "wrong-episode":
            assert pl["text"].startswith(("You", "Your"))
    # the sources are untouched: no real unit carries a plant
    for uid, m in K.items():
        if m["kind"] == "real":
            assert "Bertil" not in texts[uid] and "dreamed about the sea" not in texts[uid]


def test_decoys(built):
    K = built["key"]["units"]
    texts = {u["id"]: u["text"] for u in built["units"]}
    dec = {m["decoy"]: uid for uid, m in K.items() if m["kind"] == "decoy"}
    assert set(dec) == set(SP.DECOY_TYPES)
    for d, uid in dec.items():
        brains = [l.strip()[len("brain: "):] for l in texts[uid].splitlines() if l.strip().startswith("brain:")]
        yous = [l.split("you: ", 1)[1] for l in texts[uid].splitlines() if " you: " in l]
        assert len(brains) == K[uid]["n_turns"] == len(yous)
        if d == "refuse":
            assert set(brains) == {SP.REFUSE}
        elif d == "generic":
            assert all(any(t.replace("{W}", "").replace("{w}", "").strip(". ") in b for t in SP.GENERIC_TEMPLATES)
                       for b in brains)
        else:
            keys = {k.capitalize() for e in LEDGER.values() for k in e["keys"]}
            for y, b in zip(yous, brains):
                assert b == y or all(w.strip(". ") in keys for w in b.split(", "))
    pw = [m for m in built["key"]["pairs"].values() if m["what"] == "decoy-vs-bothoff"]
    assert len(pw) == 6 and all({m["A"], m["B"]} & {"both-off"} for m in pw)


def test_duplicates(built):
    K = built["key"]["units"]
    texts = {u["id"]: u["text"] for u in built["units"]}
    dups = [(uid, m["dup_partner"]) for uid, m in K.items() if m["kind"] == "duplicate"]
    assert len(dups) == SP.N_DUP_PAIRS
    for a, b in dups:
        assert a != b and texts[a] == texts[b] and K[b]["kind"] == "real" and K[b]["dup_partner"] == a


def test_pairs_both_orders_never_adjacent(built):
    P = built["key"]["pairs"]
    groups = {}
    for pid, m in P.items():
        groups.setdefault(m["group"], []).append(pid)
    assert len(groups) == 3 + 3 + 3 and all(len(v) == 2 for v in groups.values())
    assert len(P) == 18 and not any(m["what"].startswith("repeat") for m in P.values())
    for g, (p, q) in groups.items():
        a, b = P[p], P[q]
        assert (a["A"], a["B"]) == (b["B"], b["A"]) and a["what"] == b["what"] and a["partner"] == q
        assert abs(int(p[1:]) - int(q[1:])) > 1


def test_no_adjacent_near_copies(built):
    K = built["key"]["units"]
    src = {uid: (m["seed"], m["week"], m["arm"]) for uid, m in K.items() if m["kind"] in ("real", "duplicate", "canary")}
    for k in range(len(K) - 1):
        a, b = f"T{k}", f"T{k + 1}"
        assert not (a in src and b in src and src[a] == src[b]), (a, b)


def test_determinism(tmp_path):
    a = R.build(LIVES)
    b = R.build(list(reversed(LIVES)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = R.build(LIVES, rng_seed=1)
    assert json.dumps(a["key"]["units"], sort_keys=True) != json.dumps(c["key"]["units"], sort_keys=True)
    R.emit(LIVES, str(tmp_path / "x"))
    for f in ("units.jsonl", "reader_prompts.jsonl", "pairs.jsonl", "answer_prompts.jsonl", "blind_key.json"):
        assert os.path.getsize(tmp_path / "x" / f) > 0


def test_spec_hash_guard(tmp_path, monkeypatch):
    assert R.load_spec() is SP
    bad = tmp_path / "bad.sha256"
    bad.write_text("0" * 64 + "  life_read_spec.py\n")
    monkeypatch.setattr(R, "SPEC_HASH_FILE", str(bad))
    with pytest.raises(RuntimeError):
        R.load_spec()
    with pytest.raises(RuntimeError):
        R.build(LIVES)


def test_spec_hash_recorded():
    want = open(R.SPEC_HASH_FILE).read().split()
    assert want[0] == R.spec_sha() and want[1] == "life_read_spec.py"


def test_spearman():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # ranks [1,2,3,4,5] vs [2,1,4,3,5]: d^2 = 1+1+1+1+0 = 4 -> 1 - 6*4/(5*24) = 0.8
    assert R.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    # ties share average ranks: [1,1,2] vs [1,2,3] -> ranks [1.5,1.5,3] vs [1,2,3]
    assert R.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)
    assert R.spearman([2, 2, 2], [1, 2, 3]) is None and R.spearman([1], [1]) is None


def test_kappa():
    assert R.kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert R.kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    # po = 0.75, pe = 0.5 -> 0.5
    assert R.kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx((0.75 - 0.5) / 0.5)
    assert R.kappa([], []) is None
    assert R.kappa([1, 1], [1, 1]) == 1.0


def _synthetic_reads(key, hit_types):
    """A reader that lists the planted defect for canaries whose type is in hit_types (wrong type for the rest of
    those it lists loosely), flags every decoy, and lists nothing elsewhere."""
    reads = {}
    for uid, m in key["units"].items():
        ds = []
        if m["kind"] == "canary":
            pl = m["canary"]
            if pl["type"] in hit_types:
                ds.append({"type": pl["defect"], "quote": pl["text"][:30], "turn": pl["turns"][0]})
            elif pl["type"] == "collapse":
                ds.append({"type": "off-topic", "quote": pl["text"], "turn": pl["turns"][1]})   # loose only
        if m["kind"] == "decoy":
            ds.append({"type": "off-topic", "quote": "x", "turn": 2})
        reads[uid] = {"id": uid, "defects": ds, "coherence": 1 if m["kind"] == "decoy" else 3, "honesty": []}
    return reads


def test_canary_recall_and_scoring(built):
    key = built["key"]
    r1 = _synthetic_reads(key, set(SP.CANARY_TYPES))
    r2 = _synthetic_reads(key, {"pet-named", "invented", "experience"})
    pairs = {}
    for pid, m in key["pairs"].items():
        if m["what"] == "decoy-vs-bothoff":
            pairs[pid] = {"id": pid, "choice": "A" if m["A"] == "both-off" else "B"}
        elif m["what"].endswith("intact-vs-bothoff"):
            pairs[pid] = {"id": pid, "choice": "A" if m["A"] == "intact" else "B"}
        else:
            pairs[pid] = {"id": pid, "choice": "tie"}
    out = R.score(key, r1, r2, pairs)
    a, b = out["readers"]["reader1"], out["readers"]["reader2"]
    assert (a["canary_strict"], a["canary_loose"]) == (6, 6)
    assert (b["canary_strict"], b["canary_loose"]) == (3, 4)
    assert all(d["flagged"] for d in a["decoys"])
    assert a["mde"] == 0
    marks = {x["bar"]: x["pass"] for x in out["bars"]}
    assert marks["reader1: canary recall strict >= 5/6"] is True
    assert marks["reader2: canary recall strict >= 5/6"] is False
    assert marks["reader1: decoys flagged and rated at or below both-off"] is True
    assert out["pairwise"]["pairs1"]["intact_preferred_seeds"] == 3
    assert out["pairwise"]["pairs1"]["inconsistent"] == 0 and out["pairwise"]["pairs1"]["consistency_rate"] == 1.0
    intact_bar = [x for x in out["bars"] if x["bar"].startswith("intact preferred")][0]
    assert intact_bar["pass"] is True
    # a reader that always answers "A" is inconsistent on every pair (position bias): all ties, the intact bar fails
    alwa = {pid: {"id": pid, "choice": "a"} for pid in key["pairs"]}
    pa = R.score(key, r1, None, alwa)["pairwise"]["pairs1"]
    assert pa["inconsistent"] == pa["n_pairs"] == 9 and pa["position_bias_rate"] == 1.0
    assert pa["intact_preferred_seeds"] == 0 and set(pa["intact_vs_bothoff"].values()) == {"tie"}
    assert set(pa["decoys"].values()) == {"tie"}
    # one seed named intact in one order only is a tie, not a win
    one = dict(pairs)
    pid = next(p for p, m in key["pairs"].items() if m["what"] == "intact-vs-bothoff")
    one[pid] = {"id": pid, "choice": "A" if key["pairs"][pid]["A"] == "both-off" else "B"}
    po = R.score(key, r1, None, one)["pairwise"]["pairs1"]
    assert po["intact_preferred_seeds"] == 2 and po["inconsistent"] == 1
    assert po["intact_vs_bothoff_inconsistent"] == 1
    # kappa on "any defect": readers differ on the 2 canaries reader 2 misses entirely (wrong-episode, perspective)
    assert out["kappa"]["n"] == len(key["units"])
    assert out["kappa"]["any_defect"] < 1.0
    k = out["kappa"]
    n_real = sum(1 for m in key["units"].values() if m["kind"] == "real")
    assert k["spearman_n"] == n_real and set(k["spearman_by_type"]) == set(SP.TYPES)
    assert "Spearman" in R.to_md(out)
    md = R.to_md(out)
    assert "Instrument validity" in md and "PASS" in md and "FAIL" in md


def test_canary_hit_rules():
    pl = {"turns": [10], "text": "I feel so rested this morning.", "defect": "experience-claim"}
    assert R.canary_hit([{"type": "experience-claim", "quote": "so rested this morning", "turn": 10}], pl) == \
        {"strict": True, "loose": True}
    assert R.canary_hit([{"type": "invented", "quote": "I feel so rested", "turn": 11}], pl) == \
        {"strict": False, "loose": True}
    assert R.canary_hit([{"type": "experience-claim", "quote": "I feel so rested", "turn": 30}], pl) == \
        {"strict": False, "loose": False}
    assert R.canary_hit([{"type": "experience-claim", "quote": "something else entirely", "turn": 10}], pl) == \
        {"strict": False, "loose": False}
    # strict needs the exact planted turn; a neighbouring turn sharing words is loose at most
    assert R.canary_hit([{"type": "experience-claim", "quote": "I feel so rested", "turn": 11}], pl) == \
        {"strict": False, "loose": True}
    # a short quote never matches by being a substring of the plant
    assert R.canary_hit([{"type": "experience-claim", "quote": "I", "turn": 10}], pl) == {"strict": False, "loose": False}
    assert R.canary_hit([{"type": "experience-claim", "quote": "so rested this morning", "turn": "[10]"}], pl)["strict"]


def test_week_bar_is_per_turn():
    key = {"weeks": [1, 3], "units": {}}
    r1 = {"per_seed": {str(s): {"intact|w1": {"total": 10, "rate": 10 / 72}, "intact|w3": {"total": 9, "rate": 9 / 50}}
                       for s in (1, 2, 3)}, "mde_rate": 0.0, "canary_strict": 6, "canary_loose": 6, "canary_n": 6,
          "decoys": [], "mde": 0}
    b = [x for x in R._bars(key, {"reader1": r1}, {}) if "per turn" in x["bar"]][0]
    assert b["pass"] is False and b["value"].startswith("0/3")    # fewer defects in week 3 but more per turn


def test_right_is_strict_and_verdict(built):
    key = built["key"]
    ans = {}
    for aid, m in key["answers"].items():
        # reader says yes on every second real row and on positive controls
        ans[aid] = {"id": aid, "verdict": "yes" if (m["control"] == "answer" or (not m["control"] and int(aid[1:]) % 2))
                    else "no"}
    out = R.score_answers(key, ans, ans)
    for r in out["rows"]:
        assert r["right"] == (key["answers"][r["id"]]["ok"] and ans[r["id"]]["verdict"] == "yes")
    tot = out["by"]["intact|all"]
    assert tot["right"] == sum(1 for r in out["rows"] if r["arm"] == "intact" and r["right"])
    assert tot["right"] <= min(tot["strict"], tot["verdict_yes"])
    assert all(c["ok1"] for c in out["controls"])
    assert out["kappa"] == 1.0 and out["reader_disagreements"] == []
