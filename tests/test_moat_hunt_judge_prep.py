"""Pins for the blind-round judge prep and the mechanical adjudication (research/findings/raw/_moat_allowlist_hunt4/judge_prep.py,
adjudicate.py). Pure functions plus each script's main() on a synthetic round in tmp_path; no brain, no checker.

  judge_prep: items carry exactly {jid, gathered, sentence_as_written, sentence_exact_escaped} (no id/expect/class/why); a reply case adds
  one item per proper prefix, joined as the composer joins (run_hunt.reply_join + the composer's _terminate); jids are deterministic per
  seed and unique; the items are shuffled and split into N chunks; the private key maps jid -> [case id, part]; a malformed reply case
  stops the script.
  adjudicate: the rounds 4-7 rule (U -> B if arguable, either judge F, or both B; F -> B if either judge U or both B; --f-rule stays-F
  keeps F), prefixes U only if both judges U, F only if both F, else B; prefix_labels end in the case's final label and run_hunt reads
  them; the output keeps rounds 4-7's keys; an incomplete or inconsistent judge output stops the script.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_hunt_judge_prep.py -q
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import pytest

from research.findings.raw._moat_allowlist_hunt4 import adjudicate as A
from research.findings.raw._moat_allowlist_hunt4 import judge_prep as J
from research.findings.raw._moat_allowlist_hunt4 import run_hunt as H

PAIR = [["aug", "follows", "jul"], ["jul", "follows", "june"]]
TRIO = PAIR + [["june", "follows", "may"]]


def _cases():
    return [
        {"id": "C1", "gathered": [PAIR[0]], "sentence": "Aug follows Jul .", "expect": "U", "class": "k", "why": "secret"},
        {"id": "C2", "gathered": PAIR, "sentence": "Aug follows Jul.", "expect": "F", "class": "k", "why": "secret"},
        {"id": "R1", "gathered": TRIO, "sentences": ["Aug follows Jul", "Jul follows June.", "June follows May."],
         "sentence": "Aug follows Jul Jul follows June. June follows May.", "expect": "U", "class": "k", "why": "secret", "reply": True},
        {"id": "R2", "gathered": PAIR, "sentences": ["Aug follows Jul.", "Jul follows June."],
         "sentence": "Aug follows Jul. Jul follows June.", "expect": "F", "class": "k", "why": "secret", "reply": True, "arguable": True},
    ]


# ---- judge_prep -----------------------------------------------------------------------------------------------------------------
def test_escape_exact():
    assert J.escape_exact("Solomon is\n\r\\ ok’s \U0001f600\x07") == \
        "Solomon\\u00a0is\\n\\r\\\\ ok\\u2019s \\U0001f600\\u0007"
    assert J.escape_exact("plain ASCII, kept.") == "plain ASCII, kept."


def test_items_are_anonymised_with_one_item_per_proper_prefix():
    items, key = J.build_items(_cases(), seed=8)
    assert all(tuple(sorted(it)) == tuple(sorted(J.ITEM_KEYS)) for it in items)            # no id, expect, class, why, arguable, reply
    blob = json.dumps(items)
    assert "secret" not in blob and "C1" not in blob and "R1" not in blob
    assert len(items) == 4 + 2 + 1                                                           # 4 cases, R1 prefixes 1-2, R2 prefix 1
    assert sorted(map(tuple, key.values()), key=str) == sorted([("C1", "full"), ("C2", "full"), ("R1", "full"), ("R1", 1), ("R1", 2),
                                                                ("R2", "full"), ("R2", 1)], key=str)
    by = {tuple(key[it["jid"]]): it for it in items}
    assert by[("R1", 1)]["sentence_as_written"] == "Aug follows Jul."                       # the composer's join: "." added
    assert by[("R1", 2)]["sentence_as_written"] == "Aug follows Jul. Jul follows June."
    assert by[("R1", "full")]["sentence_as_written"] == _cases()[2]["sentence"]             # the case's sentence as written
    assert by[("R1", 1)]["gathered"] == TRIO                                                # a prefix is judged against the full union
    assert by[("C1", "full")]["sentence_exact_escaped"] == "Aug follows Jul\\u00a0."
    assert by[("R1", 2)]["sentence_as_written"] == H.reply_join(_cases()[2]["sentences"], [1, 2], H.composer_terminate())


def test_jids_are_deterministic_per_seed_unique_and_shuffled():
    a, ka = J.build_items(_cases(), seed=8)
    b, kb = J.build_items(_cases(), seed=8)
    c, kc = J.build_items(_cases(), seed=9)
    assert a == b and ka == kb and set(ka) != set(kc)
    assert len(set(ka)) == len(a) and all(j.startswith("j") and len(j) == 11 for j in ka)
    assert [tuple(ka[it["jid"]]) for it in a] != sorted((tuple(ka[it["jid"]]) for it in a), key=str)   # not in case order
    assert J.make_jid(8, "R1", 1) == J.make_jid(8, "R1", 1) != J.make_jid(8, "R1", 2)


def test_subsequence_items_are_opt_in():
    assert J.proper_selections(3) == [1, 2]
    assert J.proper_selections(3, subsequences=True) == [1, 2, "2", "3", "1+3", "2+3"]
    items, key = J.build_items(_cases(), seed=8, subsequences=True)
    by = {tuple(key[it["jid"]]): it["sentence_as_written"] for it in items}
    assert by[("R1", "1+3")] == "Aug follows Jul. June follows May." and by[("R2", "2")] == "Jul follows June."


def test_split_chunks():
    assert [len(c) for c in J.split_chunks(list(range(7)), 3)] == [3, 2, 2]
    assert sum(J.split_chunks(list(range(7)), 3), []) == list(range(7))
    with pytest.raises(ValueError):
        J.split_chunks([1], 0)


@pytest.mark.parametrize("mutate,why", [
    (lambda cs: cs[2].update(sentence="something else"), "not its 'sentences' joined"),
    (lambda cs: cs[3].update(sentences=["Aug follows Jul."]), "1 sentences for 2"),
    (lambda cs: cs[1].update(id="C1"), "repeated case id")])
def test_a_malformed_round_stops_judge_prep(tmp_path, mutate, why):
    cs = _cases()
    mutate(cs)
    with pytest.raises(ValueError, match=why.replace("(", r"\(")):
        J.build_items(cs, seed=8)
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(cs))
    assert J.main([str(p), str(tmp_path / "out"), "--chunks", "2"]) == 2 and not (tmp_path / "out").exists()


def test_judge_prep_main_writes_chunks_and_a_private_key(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(_cases()))
    assert J.main([str(p), str(tmp_path / "j"), "--chunks", "3", "--seed", "8"]) == 0
    chunks = [json.loads((tmp_path / "j" / ("chunk_%02d.json" % i)).read_text()) for i in (1, 2, 3)]
    key = json.loads((tmp_path / "j" / "key.json").read_text())
    assert sum(len(c) for c in chunks) == key["n_items"] == 7 and key["seed"] == 8 and key["n_reply_cases"] == 2
    assert {it["jid"] for c in chunks for it in c} == set(key["key"])
    assert "Solomon" not in json.dumps(key) and "\\u00a0" in json.dumps(chunks)


# ---- adjudicate -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("expect,a,b,arguable,want", [
    ("U", "U", "U", False, ("U", None)), ("U", "U", "B", False, ("U", None)), ("U", "B", "U", False, ("U", None)),
    ("U", "B", "B", False, ("B", "both blind judges: either way")), ("U", "F", "U", False, ("B", "a blind judge read it as supported")),
    ("U", "U", "F", False, ("B", "a blind judge read it as supported")), ("U", "U", "U", True, ("B", "skeptic flagged it arguable")),
    ("F", "F", "F", False, ("F", None)), ("F", "F", "B", False, ("F", None)),
    ("F", "U", "F", False, ("B", "a blind judge read it as unsupported")), ("F", "B", "B", False, ("B", "both blind judges: either way")),
    ("B", "U", "U", False, ("B", None))])
def test_the_rounds_4_to_7_case_rule(expect, a, b, arguable, want):
    assert A.case_label(expect, a, b, arguable=arguable) == want


def test_stays_f_keeps_every_expect_f_case_f():
    for a in A.LABELS:
        for b in A.LABELS:
            assert A.case_label("F", a, b, f_rule="stays-F") == ("F", None)
    with pytest.raises(ValueError):
        A.case_label("F", "F", "F", f_rule="other")


def test_the_prefix_rule():
    assert [A.part_label(a, b) for a, b in (("U", "U"), ("F", "F"), ("U", "F"), ("U", "B"), ("B", "B"), ("F", "B"))] == \
        ["U", "F", "B", "B", "B", "B"]


def _judged(key, labels):
    """labels: {(case id, part): (a, b)} -> two judges' item lists."""
    inv = {tuple(v): j for j, v in key.items()}
    a = [{"jid": inv[k], "label": la, "reason": "a"} for k, (la, _lb) in labels.items()]
    b = [{"jid": inv[k], "label": lb, "reason": "b"} for k, (_la, lb) in labels.items()]
    return a, b


LAB = {("C1", "full"): ("U", "B"), ("C2", "full"): ("F", "U"), ("R1", "full"): ("U", "U"), ("R1", 1): ("F", "F"),
       ("R1", 2): ("U", "B"), ("R2", "full"): ("F", "F"), ("R2", 1): ("U", "U")}


def test_adjudicate_writes_rounds_4_to_7_keys_and_prefix_labels_run_hunt_reads():
    _items, key = J.build_items(_cases(), seed=8)
    a, b = _judged(key, LAB)
    out = A.adjudicate(_cases(), key, {"literal": A.collect([a], key), "ordinary": A.collect([b], key)})
    for k in ("_doc", "relabel_to_B", "reasons", "final_counts", "label_crosstab", "judge_labels", "backend", "device",
              "final_labels", "prefix_labels", "prefix_judge_labels", "prefix_counts"):
        assert k in out, k
    assert out["final_labels"] == {"C1": "U", "C2": "B", "R1": "U", "R2": "F"}
    assert out["relabel_to_B"] == {"C2": "F->B"} and out["reasons"] == {"C2": "a blind judge read it as unsupported"}
    assert out["prefix_labels"] == {"R1": ["F", "B", "U"], "R2": ["U", "F"]}                  # the last = the case's final label
    assert out["judge_labels"]["C1"] == {"literal": ["U", "a"], "ordinary": ["B", "b"]}
    assert out["prefix_judge_labels"]["R1"]["2"] == {"literal": ["U", "a"], "ordinary": ["B", "b"]}
    assert out["label_crosstab"]["expect U, literal U, ordinary B"] == 1 and out["prefix_counts"] == {"F": 1, "B": 1, "U": 1}
    assert "subset_labels" not in out
    cases = _cases()
    label = {c["id"]: ("B" if c["id"] in out["relabel_to_B"] else c["expect"]) for c in cases}
    got = H.load_reply_labels(out, cases, label)
    assert got == {"R1": {"prefix": ["F", "B", "U"], "subset": None, "why": None}, "R2": {"prefix": ["U", "F"], "subset": None, "why": None}}


def test_an_arguable_reply_case_relabels_and_its_prefix_labels_follow():
    cases = _cases()
    cases[3]["expect"] = "U"                                                                 # R2: expect U, arguable
    _items, key = J.build_items(cases, seed=8)
    a, b = _judged(key, {**LAB, ("R2", "full"): ("U", "U")})
    out = A.adjudicate(cases, key, {"literal": A.collect([a], key), "ordinary": A.collect([b], key)})
    assert out["final_labels"]["R2"] == "B" and out["reasons"]["R2"] == "skeptic flagged it arguable"
    assert out["prefix_labels"]["R2"] == ["U", "B"]


def test_adjudicate_with_subsequence_items():
    _items, key = J.build_items(_cases(), seed=8, subsequences=True)
    labs = dict(LAB)
    for v in key.values():
        if tuple(v) not in labs:
            labs[tuple(v)] = ("F", "F")
    a, b = _judged(key, labs)
    out = A.adjudicate(_cases(), key, {"literal": A.collect([a], key), "ordinary": A.collect([b], key)})
    assert out["subset_labels"] == {"R1": {"2": "F", "3": "F", "1+3": "F", "2+3": "F"}, "R2": {"2": "F"}}
    assert H.check_subset_labels(out["subset_labels"]["R1"], 3) is None


def test_an_incomplete_or_inconsistent_judge_output_stops_adjudication():
    _items, key = J.build_items(_cases(), seed=8)
    a, b = _judged(key, LAB)
    with pytest.raises(ValueError, match="unlabelled"):
        A.collect([a[1:]], key)
    with pytest.raises(ValueError, match="twice"):
        A.collect([a, a[:1]], key)
    with pytest.raises(ValueError, match="unknown jid"):
        A.collect([a + [{"jid": "jnope", "label": "U"}]], key)
    with pytest.raises(ValueError, match="not U, F or B"):
        A.collect([[dict(a[0], label="X")] + a[1:]], key)
    judges = {"literal": A.collect([a], key), "ordinary": A.collect([b], key)}
    with pytest.raises(ValueError, match="exactly two"):
        A.adjudicate(_cases(), key, {"literal": judges["literal"]})
    short = {j: v for j, v in key.items() if v != ["R1", 2]}
    with pytest.raises(ValueError, match="no prefix 2"):
        A.adjudicate(_cases(), short, judges)


def test_adjudicate_main_end_to_end(tmp_path):
    cp = tmp_path / "cases.json"
    cp.write_text(json.dumps(_cases()))
    assert J.main([str(cp), str(tmp_path / "j"), "--chunks", "2"]) == 0
    key = json.loads((tmp_path / "j" / "key.json").read_text())["key"]
    a, b = _judged(key, LAB)
    (tmp_path / "a1.json").write_text(json.dumps(a[:3]))
    (tmp_path / "a2.json").write_text(json.dumps(a[3:]))
    (tmp_path / "b.json").write_text(json.dumps(b))
    out = tmp_path / "r8" / "adjudication.json"
    assert A.main([str(cp), str(tmp_path / "j" / "key.json"), str(out), "--judge", "literal", str(tmp_path / "a1.json"),
                   str(tmp_path / "a2.json"), "--judge", "ordinary", str(tmp_path / "b.json")]) == 0
    d = json.loads(out.read_text())
    assert d["prefix_labels"]["R1"] == ["F", "B", "U"] and d["relabel_to_B"] == {"C2": "F->B"} and "no checker was run" in d["provenance"]
    assert A.main([str(cp), str(tmp_path / "j" / "key.json"), str(out), "--judge", "literal", str(tmp_path / "a1.json"),
                   "--judge", "ordinary", str(tmp_path / "b.json")]) == 2                     # literal is missing items: refused
    assert A.main([str(cp), str(tmp_path / "j" / "key.json"), str(tmp_path / "s.json"), "--f-rule", "stays-F", "--judge", "literal",
                   str(tmp_path / "a1.json"), str(tmp_path / "a2.json"), "--judge", "ordinary", str(tmp_path / "b.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["final_labels"]["C2"] == "F"
    # run_hunt reads the round the same way: relabel_to_B for the label, prefix_labels for the reply entry
    (tmp_path / "r8" / "cases.json").write_text(json.dumps(_cases()))
    cases, label = H.load_cases(str(tmp_path / "r8"))
    assert label == {"C1": "U", "C2": "B", "R1": "U", "R2": "F"}
    assert H.load_reply_labels(H.load_adjudication(str(tmp_path / "r8")), cases, label)["R1"]["why"] is None
