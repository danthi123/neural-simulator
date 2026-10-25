"""Pins for the blind-round judge prep and the mechanical adjudication (research/findings/raw/_moat_allowlist_hunt4/judge_prep.py,
adjudicate.py). Pure functions plus each script's main() on a synthetic round in tmp_path; no brain, no checker.

  judge_prep: items carry exactly {jid, gathered, sentence_as_written, sentence_exact_escaped} (no id/expect/class/why); a reply case
  (2 <= k <= 4) adds one item per PROPER in-order non-empty selection of its sentences, prefixes and other subsequences (the default,
  Claude ruling 2026-10-21; --prefixes-only keeps the old prefix-only set), joined as the composer joins (run_hunt.reply_join + the
  composer's _terminate); jids are deterministic per seed and unique; the items are shuffled and split into N chunks; the private key
  maps jid -> [case id, "full" or the selection key]; a malformed reply case stops the script.
  adjudicate: the rounds 4-7 case rule, the only behaviour (U -> B if arguable, either judge F, or both B; F -> B if either judge U or
  both B); a selection is U if neither judge reads F and at least one reads U, F if both read F, else B; subset_labels (one key family,
  every selection plus the full reply = the case's final label) and the derived prefix_labels are what run_hunt reads; the output keeps
  rounds 4-7's keys; an incomplete or inconsistent judge output or key stops the script.

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
R1_PARTS = ["1", "1+2", "2", "3", "1+3", "2+3"]


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


def test_proper_selections_are_every_in_order_selection_by_default():
    assert J.proper_selections(2) == ["1", "2"]
    assert J.proper_selections(3) == R1_PARTS
    assert len(J.proper_selections(4)) == 14                                                 # + the full item = 15 per case at k = 4
    assert J.proper_selections(3, prefixes_only=True) == ["1", "1+2"]
    assert all(H.parse_selection(p, 4) is not None for p in J.proper_selections(4))         # run_hunt's key family


def test_items_are_anonymised_with_one_item_per_proper_selection():
    items, key = J.build_items(_cases(), seed=8)
    assert all(tuple(sorted(it)) == tuple(sorted(J.ITEM_KEYS)) for it in items)            # no id, expect, class, why, arguable, reply
    blob = json.dumps(items)
    assert "secret" not in blob and "C1" not in blob and "R1" not in blob
    assert len(items) == 2 + (1 + 6) + (1 + 2)
    want = [("C1", "full"), ("C2", "full"), ("R1", "full"), ("R2", "full"), ("R2", "1"), ("R2", "2")] + [("R1", p) for p in R1_PARTS]
    assert sorted(map(tuple, key.values())) == sorted(want)
    by = {tuple(key[it["jid"]]): it for it in items}
    assert by[("R1", "1")]["sentence_as_written"] == "Aug follows Jul."                     # the composer's join: "." added
    assert by[("R1", "1+2")]["sentence_as_written"] == "Aug follows Jul. Jul follows June."
    assert by[("R1", "1+3")]["sentence_as_written"] == "Aug follows Jul. June follows May."
    assert by[("R2", "2")]["sentence_as_written"] == "Jul follows June."
    assert by[("R1", "full")]["sentence_as_written"] == _cases()[2]["sentence"]             # the case's sentence as written
    assert by[("R1", "2+3")]["gathered"] == TRIO                                            # a selection is judged against the full union
    assert by[("C1", "full")]["sentence_exact_escaped"] == "Aug follows Jul\\u00a0."
    assert by[("R1", "1+2")]["sentence_as_written"] == H.reply_join(_cases()[2]["sentences"], [1, 2], H.composer_terminate())


def test_prefixes_only_keeps_the_old_item_set():
    items, key = J.build_items(_cases(), seed=8, prefixes_only=True)
    assert sorted(map(tuple, key.values())) == sorted([("C1", "full"), ("C2", "full"), ("R1", "full"), ("R1", "1"), ("R1", "1+2"),
                                                       ("R2", "full"), ("R2", "1")])


def test_jids_are_deterministic_per_seed_unique_and_shuffled():
    a, ka = J.build_items(_cases(), seed=8)
    b, kb = J.build_items(_cases(), seed=8)
    c, kc = J.build_items(_cases(), seed=9)
    assert a == b and ka == kb and set(ka) != set(kc)
    assert len(set(ka)) == len(a) and all(j.startswith("j") and len(j) == 11 for j in ka)
    assert [tuple(ka[it["jid"]]) for it in a] != sorted(tuple(ka[it["jid"]]) for it in a)   # not in case order
    assert J.make_jid(8, "R1", "1") == J.make_jid(8, "R1", "1") != J.make_jid(8, "R1", "1+2")


def test_split_chunks():
    assert [len(c) for c in J.split_chunks(list(range(7)), 3)] == [3, 2, 2]
    assert sum(J.split_chunks(list(range(7)), 3), []) == list(range(7))
    with pytest.raises(ValueError):
        J.split_chunks([1], 0)


@pytest.mark.parametrize("mutate,why", [
    (lambda cs: cs[2].update(sentence="something else"), "not its 'sentences' joined"),
    (lambda cs: cs[3].update(sentences=["Aug follows Jul."]), "1 sentences for 2"),
    (lambda cs: cs[2].update(gathered=TRIO + [["may", "follows", "april"], ["april", "follows", "march"]],
                             sentences=cs[2]["sentences"] + ["May follows April.", "April follows March."],
                             sentence=cs[2]["sentence"] + " May follows April. April follows March."), "2 <= k <= 4"),
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
    assert sum(len(c) for c in chunks) == key["n_items"] == 12 and key["seed"] == 8 and key["n_reply_cases"] == 2
    assert key["selections"] == "all" and {it["jid"] for c in chunks for it in c} == set(key["key"])
    assert "Solomon" not in json.dumps(key) and "\\u00a0" in json.dumps(chunks)
    assert J.main([str(p), str(tmp_path / "jp"), "--chunks", "1", "--prefixes-only"]) == 0
    kp = json.loads((tmp_path / "jp" / "key.json").read_text())
    assert kp["selections"] == "prefixes" and kp["n_items"] == 7


# ---- adjudicate -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("expect,a,b,arguable,want", [
    ("U", "U", "U", False, ("U", None)), ("U", "U", "B", False, ("U", None)), ("U", "B", "U", False, ("U", None)),
    ("U", "B", "B", False, ("B", "both blind judges: either way")), ("U", "F", "U", False, ("B", "a blind judge read it as supported")),
    ("U", "U", "F", False, ("B", "a blind judge read it as supported")), ("U", "U", "U", True, ("B", "skeptic flagged it arguable")),
    ("F", "F", "F", False, ("F", None)), ("F", "F", "B", False, ("F", None)),
    ("F", "U", "F", False, ("B", "a blind judge read it as unsupported")), ("F", "B", "B", False, ("B", "both blind judges: either way")),
    ("B", "U", "U", False, ("B", None))])
def test_the_rounds_4_to_7_case_rule_is_the_only_rule(expect, a, b, arguable, want):
    assert A.case_label(expect, a, b, arguable=arguable) == want


def test_the_f_rule_option_is_gone():
    import inspect
    assert "f_rule" not in inspect.signature(A.case_label).parameters and "f_rule" not in inspect.signature(A.adjudicate).parameters
    with pytest.raises(SystemExit):
        A.main(["c", "k", "o", "--f-rule", "stays-F", "--judge", "a", "x", "--judge", "b", "y"])


@pytest.mark.parametrize("a,b,want", [
    ("U", "U", "U"), ("U", "B", "U"), ("B", "U", "U"),                  # a U/B split counts as U (the conservative direction)
    ("F", "F", "F"),
    ("U", "F", "B"), ("F", "U", "B"), ("B", "B", "B"), ("F", "B", "B"), ("B", "F", "B")])
def test_the_selection_rule(a, b, want):
    assert A.part_label(a, b) == want


def _judged(key, labels, default=("F", "F")):
    """labels: {(case id, part): (a, b)}, `default` for every other key item -> two judges' item lists."""
    a, b = [], []
    for jid, (cid, part) in key.items():
        la, lb = labels.get((cid, part), default)
        a.append({"jid": jid, "label": la, "reason": "a"})
        b.append({"jid": jid, "label": lb, "reason": "b"})
    return a, b


LAB = {("C1", "full"): ("U", "B"), ("C2", "full"): ("F", "U"), ("R1", "full"): ("U", "U"), ("R1", "1"): ("F", "F"),
       ("R1", "1+2"): ("U", "B"), ("R1", "2"): ("U", "F"), ("R1", "1+3"): ("B", "B"), ("R2", "full"): ("F", "F"),
       ("R2", "1"): ("U", "U"), ("R2", "2"): ("B", "U")}


def _adj(cases=None, labels=LAB, **kw):
    cases = cases or _cases()
    _items, key = J.build_items(cases, seed=8, **kw)
    a, b = _judged(key, labels)
    return A.adjudicate(cases, key, {"literal": A.collect([a], key), "ordinary": A.collect([b], key)},
                        prefixes_only=kw.get("prefixes_only", False))


def test_adjudicate_writes_rounds_4_to_7_keys_and_one_selection_key_family_run_hunt_reads():
    out = _adj()
    for k in ("_doc", "relabel_to_B", "reasons", "final_counts", "label_crosstab", "judge_labels", "backend", "device",
              "final_labels", "subset_labels", "subset_judge_labels", "subset_counts", "prefix_labels"):
        assert k in out, k
    assert out["final_labels"] == {"C1": "U", "C2": "B", "R1": "U", "R2": "F"}             # R2: expect F, both judges F ("arguable"
    assert out["relabel_to_B"] == {"C2": "F->B"}                                          # applies to an expect-U case only)
    assert out["reasons"]["C2"] == "a blind judge read it as unsupported"
    assert out["subset_labels"]["R1"] == {"1": "F", "1+2": "U", "2": "B", "3": "F", "1+3": "B", "2+3": "F", "1+2+3": "U"}
    assert out["subset_labels"]["R2"] == {"1": "U", "2": "U", "1+2": "F"}                # the full reply = the case's final label
    assert out["prefix_labels"] == {"R1": ["F", "U", "U"], "R2": ["U", "F"]}              # derived, for compatibility
    assert out["judge_labels"]["C1"] == {"literal": ["U", "a"], "ordinary": ["B", "b"]}
    assert out["subset_judge_labels"]["R1"]["1+2"] == {"literal": ["U", "a"], "ordinary": ["B", "b"]}
    assert out["label_crosstab"]["expect U, literal U, ordinary B"] == 1 and sum(out["subset_counts"].values()) == 8
    assert "stays F" not in out["_doc"] and "at least one read it U" in out["_doc"]
    cases = _cases()
    label = {c["id"]: ("B" if c["id"] in out["relabel_to_B"] else c["expect"]) for c in cases}
    got = H.load_reply_labels(out, cases, label)
    assert all(v["why"] is None for v in got.values()), got
    m, why = H.reply_label_map(got["R1"]["prefix"], got["R1"]["subset"], 3, "U")
    assert why is None and m == out["subset_labels"]["R1"]


def test_an_f_reply_case_becomes_b_when_a_judge_reads_it_u():
    out = _adj(labels={**LAB, ("R2", "full"): ("U", "F")})
    assert out["final_labels"]["R2"] == "B" and out["reasons"]["R2"] == "a blind judge read it as unsupported"


def test_an_arguable_u_reply_case_relabels_and_its_full_selection_follows():
    cases = _cases()
    cases[3]["expect"] = "U"                                                                 # R2: expect U, arguable
    out = _adj(cases=cases, labels={**LAB, ("R2", "full"): ("U", "U")})
    assert out["final_labels"]["R2"] == "B" and out["reasons"]["R2"] == "skeptic flagged it arguable"
    assert out["subset_labels"]["R2"]["1+2"] == "B" and out["prefix_labels"]["R2"] == ["U", "B"]


def test_prefixes_only_adjudication_labels_the_prefixes():
    out = _adj(prefixes_only=True)
    assert out["subset_labels"]["R1"] == {"1": "F", "1+2": "U", "1+2+3": "U"} and out["selections"] == "prefixes"


def test_an_incomplete_or_inconsistent_judge_output_or_key_stops_adjudication():
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
    short = {j: v for j, v in key.items() if v != ["R1", "1+3"]}
    with pytest.raises(ValueError, match="no '1\\+3' item for R1"):
        A.adjudicate(_cases(), short, judges)                                               # every selection is required by default
    bad = dict(key, jextra=["C1", "1"])
    with pytest.raises(ValueError, match="not 'full' or a proper"):
        A.adjudicate(_cases(), bad, judges)


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
    assert d["subset_labels"]["R1"]["2"] == "B" and d["relabel_to_B"] == {"C2": "F->B"}
    assert "no checker was run" in d["provenance"]
    assert A.main([str(cp), str(tmp_path / "j" / "key.json"), str(out), "--judge", "literal", str(tmp_path / "a1.json"),
                   "--judge", "ordinary", str(tmp_path / "b.json")]) == 2                     # literal is missing items: refused
    # run_hunt reads the round the same way: relabel_to_B for the label, subset_labels for the reply entry
    (tmp_path / "r8" / "cases.json").write_text(json.dumps(_cases()))
    cases, label = H.load_cases(str(tmp_path / "r8"))
    assert label == {"C1": "U", "C2": "B", "R1": "U", "R2": "F"}
    got = H.load_reply_labels(H.load_adjudication(str(tmp_path / "r8")), cases, label)
    assert got["R1"]["why"] is None and got["R1"]["subset"]["1+3"] == "B"
