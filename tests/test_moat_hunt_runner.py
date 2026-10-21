"""Pins for the general moat leak-hunt runner (research/findings/raw/_moat_allowlist_hunt4/run_hunt.py).

Prereg 2026-10-20-moat-allowlist-claim-checker, AMENDMENT 2: an arm that records an error or a non-boolean verdict on a case is
UNDEFINED for that case, never "kept 0" (round 3's stock arm stored a raw None from the claim gate and an AssertionError from the
composer entry, and its summary counted both as kept 0). These tests pin that rule with a MOCKED verifier (no brain build):
  - a None gate verdict under the allowlist flag, or any exception, is UNDEFINED (never "not kept");
  - for stock / B2h a None gate verdict is the production signal to fall back to the single-triple `_verify`, and the record is that
    fallback's bool, labelled src="fallback";
  - the composer entry's bool() coercion of a None allowlist verdict is seen through (UNDEFINED);
  - the summary counts kept / not kept / UNDEFINED per label and sets VALID only when nothing is UNDEFINED;
  - main() exits nonzero when the allowlist arm has any UNDEFINED case.
The composer entry is the REAL `RichAnswerComposer._verify_rendered` on a bare object whose `.chat` is the mock.
The REPLY entry (round 8): the prefix mapping `map_reply` (the composer's declared normalisation only; every non-prefix text UNDEFINED,
an in-order selection a `subsequence`), the reply-case validation, the stand-in mouth that answers by fact, the prefix-labelled
adjudication reader, the reply summary counting, the reply controls; the real `render_paragraph` on a mocked chat (coercion, exception,
wrong path); and end to end on the REAL composer (BridgeParser + ChatBrain + RichAnswerComposer over two synthetic store triples) the
reply controls RP1 / RN1 / RN0 / RNJ at seeds 42 and 43 on both paths, flag ON, plus stock (flag OFF) emitting the full reply.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_hunt_runner.py -q
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import pytest

from research.findings.raw._moat_allowlist_hunt4 import run_hunt as H


GATHERED = [["dog", "eat", "bird"], ["cat", "eat", "fish"]]
SENT = "The cat eats the fish."


class FakeChat:
    """Mock of the three things the runner reads from a ChatBrain: the claim gate, the single-triple `_verify` and the vocabulary."""

    def __init__(self, gate=None, verify_ok=("cat", "eat", "fish"), raises=None):
        self.gate = gate or {}                     # arm -> (acc, res) or an Exception instance
        self.verify_ok = tuple(verify_ok) if verify_ok else None
        self.raises = raises
        self.agents_set, self.actions_set, self.patients_set = {"cat", "dog"}, {"eat"}, {"fish", "bird"}
        self.inflect = {"eat": "eat", "eats": "eat", "eating": "eat", "eated": "eat"}
        self.single_calls = []

    @staticmethod
    def _arm():
        if os.environ.get("BRAIN_MOAT_ALLOWLIST"):
            return "allowlist"
        return "b2h" if os.environ.get("BRAIN_MOAT_SLUG_NORM") else "stock"

    def _verify_claim_set(self, surface, gated):
        r = self.gate.get(self._arm(), (True, {}))
        if isinstance(r, Exception):
            raise r
        return r

    def _verify(self, surface, asserted, svo):
        self.single_calls.append((asserted, tuple(svo)))
        if self.raises is not None:
            raise self.raises
        return tuple(svo) == self.verify_ok


def composer_over(chat):
    from research.runners.rich_answer_composer import RichAnswerComposer
    rc = RichAnswerComposer.__new__(RichAnswerComposer)
    rc.chat = chat
    return rc


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    for k in H.FLAGS:
        monkeypatch.delenv(k, raising=False)


def set_arm(monkeypatch, arm):
    for k, v in H.ARMS[arm].items():
        monkeypatch.setenv(k, v)


# ---- the verdict rule ---------------------------------------------------------------------------------------------------------------
def test_none_gate_verdict_under_the_allowlist_is_undefined_never_not_kept(monkeypatch):
    set_arm(monkeypatch, "allowlist")
    chat = FakeChat(gate={"allowlist": (None, None)})
    g = H.gate_entry(chat, "allowlist", SENT, GATHERED, alt_vocab=False)
    assert g["v"] is None and not H.is_defined(g)
    assert H.case_status([g]) == "UNDEFINED"
    assert not chat.single_calls                                   # the strict arm never falls back to the single-triple re-parse
    trace = H.Trace(chat)
    c = H.composer_entry(composer_over(chat), chat, trace, "allowlist", SENT, GATHERED, alt_vocab=False)
    assert c["v"] is None and "coerced" in c["why"]                # the composer's bool(None) == False is seen through
    assert H.case_status([c]) == "UNDEFINED"


@pytest.mark.parametrize("arm", ["stock", "b2h", "allowlist"])
def test_an_exception_is_undefined_in_both_entries(monkeypatch, arm):
    set_arm(monkeypatch, arm)
    chat = FakeChat(gate={arm: AssertionError("boom")})
    g = H.gate_entry(chat, arm, SENT, GATHERED, alt_vocab=False)
    assert g["src"] == "error" and g["v"] is None and "AssertionError" in g["why"]
    c = H.composer_entry(composer_over(chat), chat, H.Trace(chat), arm, SENT, GATHERED, alt_vocab=False)
    assert c["src"] == "error" and c["v"] is None
    assert H.case_status([g]) == H.case_status([c]) == "UNDEFINED"


def test_a_non_bool_verdict_is_undefined(monkeypatch):
    set_arm(monkeypatch, "allowlist")
    g = H.gate_entry(FakeChat(gate={"allowlist": ("yes", {})}), "allowlist", SENT, GATHERED, alt_vocab=False)
    assert not H.is_defined(g)
    assert not H.is_defined({"v": 1, "src": "claim"})              # an int is not a bool
    assert H.case_status([{"v": True, "src": "claim"}, {"v": None, "src": "error"}]) == "UNDEFINED"
    assert H.case_status([{"v": False}, {"v": False}]) == "not_kept"
    assert H.case_status([{"v": False}, {"v": True}]) == "kept"


def test_stock_none_gate_verdict_is_the_labelled_single_triple_fallback_not_none(monkeypatch):
    set_arm(monkeypatch, "stock")
    chat = FakeChat(gate={"stock": (None, None)})                  # claim verifier unbuildable -> production falls back to `_verify`
    g = H.gate_entry(chat, "stock", SENT, GATHERED, alt_vocab=False)
    assert g["v"] is True and g["src"] == "fallback" and H.is_defined(g)
    assert g["extracted"] == ["cat", "eat", "fish"] and g["targets"] == [["cat", "eat", "fish"]]
    # production passes the renderer's SVO (stub) or None (free prose), never the prose itself (the round-3 AssertionError)
    assert chat.single_calls and all(a is None for a, _ in chat.single_calls)
    # the composer entry reaches the same fallback through the REAL `_verify_rendered`, with the first gathered fact as its target:
    # the sentence is about the second fact, and the any-target policy still finds it
    chat2 = FakeChat(gate={"stock": (None, None)})
    c = H.composer_entry(composer_over(chat2), chat2, H.Trace(chat2), "stock", SENT, GATHERED, alt_vocab=False)
    assert c["v"] is True and c["src"] == "fallback"
    # a sentence about nothing gathered is a defined False
    chat3 = FakeChat(gate={"stock": (None, None)})
    g3 = H.gate_entry(chat3, "stock", "Birds sing.", GATHERED, alt_vocab=False)
    assert g3["v"] is False and g3["src"] == "fallback" and H.is_defined(g3)


def test_a_bool_claim_gate_verdict_is_taken_as_is(monkeypatch):
    for arm, accepted in (("stock", True), ("b2h", False), ("allowlist", False)):
        set_arm(monkeypatch, arm)
        chat = FakeChat(gate={arm: (accepted, {"accepted": accepted})})
        assert H.gate_entry(chat, arm, SENT, GATHERED, alt_vocab=False) == {"v": accepted, "src": "claim"}
        c = H.composer_entry(composer_over(chat), chat, H.Trace(chat), arm, SENT, GATHERED, alt_vocab=False)
        assert c == {"v": accepted, "src": "claim"}
        assert not chat.single_calls
        for k in H.FLAGS:
            monkeypatch.delenv(k, raising=False)


# ---- the summary --------------------------------------------------------------------------------------------------------------------
def _row(i, label, per_arm):
    """per_arm: arm -> list of verdict records (one per seed) used for BOTH entries unless a (gate, composer) pair is given."""
    row = {"id": i, "label": label, "sentence": "s", "n_gathered": 2, "gate": {}, "composer": {}}
    for arm, recs in per_arm.items():
        gate, comp = recs if isinstance(recs, tuple) else (recs, recs)
        row["gate"][arm], row["composer"][arm] = gate, comp
    return row


T, F, N = {"v": True, "src": "claim"}, {"v": False, "src": "claim"}, {"v": None, "src": "error", "why": "AssertionError: x"}


def test_summary_counts_kept_not_kept_and_undefined_per_label_and_the_valid_flag():
    rows = [
        _row("U1", "U", {"stock": [T, F], "b2h": [F, F], "allowlist": [F, F]}),
        _row("U2", "U", {"stock": [N, F], "b2h": [F, F], "allowlist": [F, F]}),      # one UNDEFINED seed -> UNDEFINED, not not_kept
        _row("U3", "U", {"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]}),
        _row("F1", "F", {"stock": [T, T], "b2h": [T, T], "allowlist": [T, F]}),
        _row("B1", "B", {"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]}),
    ]
    s = H.summarise(rows)
    su = s["stock"]["entries"]["gate"]["U"]
    assert (su["n"], su["kept"], su["not_kept"], su["UNDEFINED"]) == (3, 1, 1, 1)
    assert su["kept_ids"] == ["U1"] and su["undefined_ids"] == ["U2"]
    assert s["stock"]["entries"]["gate"]["F"]["kept"] == 1 and s["stock"]["entries"]["gate"]["B"]["not_kept"] == 1
    assert s["stock"]["VALID"] is False and s["stock"]["n_undefined_case_entries"] == 2          # U2, in the gate and composer entries
    assert s["b2h"]["VALID"] is True and s["allowlist"]["VALID"] is True
    assert s["stock"]["reasons"]["gate"] == {"AssertionError: x": 1}


def test_valid_is_per_entry_not_just_per_arm():
    rows = [_row("U1", "U", {"stock": ([F], [N]), "b2h": [F], "allowlist": [F]})]            # only the composer entry errs
    s = H.summarise(rows)
    assert s["stock"]["entries"]["gate"]["U"]["UNDEFINED"] == 0 and s["stock"]["entries"]["composer"]["U"]["UNDEFINED"] == 1
    assert s["stock"]["VALID"] is False


def test_gathered_vocab_sensitivity_reads_alt_v_and_defaults_to_the_primary_verdict():
    r = {"v": False, "src": "fallback", "alt_v": True}
    s = H.summarise([_row("U1", "U", {"stock": [r], "b2h": [F], "allowlist": [F]})])
    e = s["stock"]["entries"]["gate"]["U"]
    assert e["kept"] == 0 and e["gathered_vocab_sensitivity"]["kept"] == 1 and e["gathered_vocab_sensitivity"]["kept_ids"] == ["U1"]
    assert s["b2h"]["entries"]["gate"]["U"]["gathered_vocab_sensitivity"]["kept"] == 0


def test_exit_code_is_nonzero_when_the_allowlist_arm_has_any_undefined_case():
    ok = [_row("U1", "U", {"stock": [F], "b2h": [F], "allowlist": [F]})]
    bad_allow = [_row("U1", "U", {"stock": [F], "b2h": [F], "allowlist": [N]})]
    bad_stock = [_row("U1", "U", {"stock": [N], "b2h": [F], "allowlist": [F]})]
    assert H.exit_code(H.summarise(ok)) == 0
    assert H.exit_code(H.summarise(bad_allow)) == H.EXIT_ALLOWLIST_UNDEFINED != 0
    assert H.exit_code(H.summarise(bad_stock)) == H.EXIT_COMPARISON_VOID != 0                # void comparison columns are loud too
    assert H.exit_code(H.summarise(ok), equiv_ok=False) == H.EXIT_EQUIV_FAILED != 0
    assert H.exit_code(H.summarise(bad_allow), equiv_ok=False) == H.EXIT_ALLOWLIST_UNDEFINED     # the allowlist verdict outranks
    assert H.exit_code(H.summarise(ok), controls_ok=False) == H.EXIT_CONTROL_FAILED != 0


# ---- controls: a column that can only say "not kept" would also read "kept 0" -------------------------------------------------------
def test_control_report_needs_supported_controls_kept_and_the_reversed_one_rejected():
    p = lambda arm_recs: _row("P1", "P", arm_recs)
    n = lambda arm_recs: _row("N1", "N", arm_recs)
    good = [p({"stock": [T, T], "b2h": [T, T], "allowlist": [T, T]}), n({"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]})]
    assert H.control_report(good)["ALL_OK"] is True
    dead_stock = [p({"stock": [F, F], "b2h": [T, T], "allowlist": [T, T]}), n({"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]})]
    r = H.control_report(dead_stock)
    assert r["ALL_OK"] is False and r["ok"]["stock"] == {"gate": False, "composer": False} and r["ok"]["b2h"]["gate"] is True
    leaky = [p({"stock": [T, T], "b2h": [T, T], "allowlist": [T, T]}), n({"stock": [F, F], "b2h": [T, F], "allowlist": [F, F]})]
    assert H.control_report(leaky)["ok"]["b2h"]["composer"] is False
    undefined = [p({"stock": [T, N], "b2h": [T, T], "allowlist": [T, T]}), n({"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]})]
    assert H.control_report(undefined)["ok"]["stock"]["gate"] is False                       # an UNDEFINED control is not a pass
    assert [c["id"] for c in H.CONTROL_CASES] == ["P1", "P2", "N1"]


# ---- main() end to end with mocked objects ----------------------------------------------------------------------------------------
def _cases_dir(tmp_path):
    cases = [{"id": "X1", "gathered": GATHERED, "sentence": SENT, "expect": "U", "class": "c"},
             {"id": "X2", "gathered": GATHERED, "sentence": SENT, "expect": "F", "class": "c"},
             {"id": "X3", "gathered": GATHERED, "sentence": SENT, "expect": "U", "class": "c"}]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "adjudication.json").write_text(json.dumps({"relabel_to_B": {"X3": "U->B"}}))
    return str(tmp_path)


def _objects(gate):
    def make(seed, kb):
        chat = FakeChat(gate=gate)
        return chat, composer_over(chat)
    return make


def test_main_exits_nonzero_and_writes_the_undefined_when_the_allowlist_gate_returns_none(tmp_path):
    out = tmp_path / "out" / "o.json"
    code = H.main([_cases_dir(tmp_path), str(out), "--seeds", "42,43", "--equiv-n", "0"], control_cases=[],
                  objects=_objects({"stock": (None, None), "b2h": (False, {}), "allowlist": (None, None)}))
    assert code == H.EXIT_ALLOWLIST_UNDEFINED
    d = json.loads(out.read_text())
    assert d["exit_code"] == code and d["summary"]["allowlist"]["VALID"] is False
    assert d["summary"]["allowlist"]["entries"]["gate"]["U"]["UNDEFINED"] == 1 and d["summary"]["allowlist"]["entries"]["gate"]["B"]["n"] == 1
    assert d["summary"]["stock"]["VALID"] is True                           # stock's Nones were fallbacks, which are bools
    assert d["summary"]["stock"]["fallback_case_seeds"]["gate"] == {"fallback": 6}
    for k in ("host", "argv", "backend", "device", "sim_backend", "seeds"):
        assert k in d, k
    assert d["backend"] == "numpy" and d["device"].startswith("cpu")


def test_main_exits_zero_when_every_arm_is_valid(tmp_path):
    out = tmp_path / "o.json"
    code = H.main([_cases_dir(tmp_path), str(out), "--seeds", "42", "--equiv-n", "0"], control_cases=[],
                  objects=_objects({"stock": (False, {}), "b2h": (False, {}), "allowlist": (True, {})}))
    assert code == 0
    d = json.loads(out.read_text())
    assert all(d["summary"][a]["VALID"] for a in H.ARMS)
    assert d["summary"]["allowlist"]["entries"]["composer"]["U"]["kept"] == 1                 # X1 kept; X3 was relabelled B
    assert d["summary"]["allowlist"]["entries"]["composer"]["B"]["kept"] == 1


# ---- the allowlist reject reason (instrument only: no verdict or existing summary field changes) ----------------------------------
WHY = "allowlist: entity span 'Ján' is not an exact store surface of j_n (fold to find, exact to accept: accepted 'j n') [sentence 1 of 1]"


def test_the_allowlist_arm_records_the_gates_reject_reason_in_both_entries(monkeypatch):
    set_arm(monkeypatch, "allowlist")
    chat = FakeChat(gate={"allowlist": (False, {"accepted": False, "reject_reason": WHY})})
    assert H.gate_entry(chat, "allowlist", SENT, GATHERED, alt_vocab=False) == {"v": False, "src": "claim", "reason": WHY}
    c = H.composer_entry(composer_over(chat), chat, H.Trace(chat), "allowlist", SENT, GATHERED, alt_vocab=False)
    assert c == {"v": False, "src": "claim", "reason": WHY}
    keep = FakeChat(gate={"allowlist": (True, {"accepted": True, "reject_reason": None})})
    assert H.gate_entry(keep, "allowlist", SENT, GATHERED, alt_vocab=False) == {"v": True, "src": "claim"}     # a keep carries none
    for k in H.FLAGS:
        monkeypatch.delenv(k, raising=False)
    set_arm(monkeypatch, "b2h")
    other = FakeChat(gate={"b2h": (False, {"accepted": False, "reject_reason": "b2h: x"})})
    assert H.gate_entry(other, "b2h", SENT, GATHERED, alt_vocab=False) == {"v": False, "src": "claim"}         # allowlist arm only


@pytest.mark.parametrize("reason,tag", [
    (WHY, "entity span"), ("allowlist: no shape: at token 2 'x' expected y", "no shape"),
    ("allowlist: not gathered: (a, b, c)", "not gathered"), ("allowlist: all-caps token 'WHO' in the SUBJ/OBJ span", "all-caps token"),
    ("allowlist: apostrophe (possessive or contraction) in 'x'", "apostrophe"),
    ("allowlist: non-ASCII character 'é' (U+00E9) at 3 outside", "non-ASCII character"),
    ("allowlist: nothing gathered", "nothing gathered"), ("something else", "other"), (None, "other")])
def test_reason_tag_is_the_leading_words(reason, tag):
    assert H.reason_tag(reason) == tag


def test_the_summary_counts_allowlist_reasons_by_tag_and_nothing_else_changes():
    r1 = {"v": False, "src": "claim", "reason": WHY}
    r2 = {"v": False, "src": "claim", "reason": "allowlist: no shape: at token 1"}
    rows = [_row("U1", "U", {"stock": [F, F], "b2h": [F, F], "allowlist": [r1, r2]}),
            _row("F1", "F", {"stock": [T, T], "b2h": [T, T], "allowlist": [T, r1]})]
    bare = [_row("U1", "U", {"stock": [F, F], "b2h": [F, F], "allowlist": [F, F]}),
            _row("F1", "F", {"stock": [T, T], "b2h": [T, T], "allowlist": [T, F]})]
    s, b = H.summarise(rows), H.summarise(bare)
    assert s["allowlist"]["reject_reason_tags"] == {"gate": {"entity span": 2, "no shape": 1}, "composer": {"entity span": 2, "no shape": 1}}
    assert b["allowlist"]["reject_reason_tags"] == {"gate": {}, "composer": {}}
    for arm in s:
        assert {k: v for k, v in s[arm].items() if k != "reject_reason_tags"} == {k: v for k, v in b[arm].items() if k != "reject_reason_tags"}
    assert "reject_reason_tags" not in s["stock"] and "reject_reason_tags" not in s["b2h"]


# =====================================================================================================================================
# the reply entry (round 8): the composer's paragraph path, sequential and batched, with the build-6 joined check
# =====================================================================================================================================
def _term():
    return H.composer_terminate()


S2 = ["Aug follows Jul.", "Jul follows June."]


# ---- the prefix mapping (pure) ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,j", [("Aug follows Jul. Jul follows June.", 2), ("Aug follows Jul.", 1), ("", 0)])
def test_map_reply_reads_the_prefix_index(text, j):
    for term in (None, _term()):                                    # terminated sentences: both joins coincide
        m = H.map_reply(text, S2, terminate=term)
        assert m == {"status": "prefix", "j": j, "subset": list(range(1, j + 1))}, (term, m)
        assert H.outcome_key(m) == str(j)


def test_map_reply_applies_only_the_composers_declared_normalisation():
    raw = ["Aug follows Jul", "Jul follows June  "]                   # unterminated; a trailing U+0020 run
    on, off = "Aug follows Jul. Jul follows June.", "Aug follows Jul Jul follows June  "
    assert H.map_reply(on, raw, terminate=_term())["j"] == 2          # flag ON: "." added, trailing spaces stripped, one U+0020
    assert H.map_reply(off, raw, terminate=None)["j"] == 2            # flag OFF: the sentences as rendered, one U+0020
    assert H.map_reply("Aug follows Jul.", raw, terminate=_term())["j"] == 1
    assert H.map_reply(on, raw, terminate=None)["status"] == "UNDEFINED"          # the other arm's join is not this arm's
    assert H.map_reply(off, raw, terminate=_term())["status"] == "UNDEFINED"
    # no other normalisation: a U+00A0 is not a space, case and whitespace runs are not folded
    nb = ["Solomon is located in Oceania.", "Jul follows June."]
    assert H.map_reply("Solomon is located in Oceania. Jul follows June.", nb, terminate=_term())["status"] == "UNDEFINED"
    assert H.map_reply("Solomon is located in Oceania. Jul follows June.", nb, terminate=_term())["j"] == 2
    assert H.map_reply("Aug follows Jul.  Jul follows June.", S2, terminate=_term())["status"] == "UNDEFINED"
    assert H.map_reply("aug follows jul.", S2, terminate=_term())["status"] == "UNDEFINED"


@pytest.mark.parametrize("text,why", [
    ("Jul follows June. Aug follows Jul.", "no in-order"),           # reordered
    ("Aug follows", "no in-order"),                                   # a partial sentence
    ("Aug follows Jul. Jul follows June. And more.", "no in-order"),  # extra text
    ("Aug follows Jul!", "no in-order"),                              # the mood marker is not this mouth's text
    ("aug follow jul", "no in-order"),                                # another mouth (the raw triple)
    (None, "not a string"), (["Aug follows Jul."], "not a string")])
def test_map_reply_is_undefined_never_not_kept_for_text_that_matches_no_prefix(text, why):
    m = H.map_reply(text, S2, terminate=_term())
    assert m["status"] == "UNDEFINED" and m["j"] is None and why in m["why"], m
    assert H.outcome_key(m) is None


def test_map_reply_reads_a_non_prefix_selection_as_a_subsequence_never_as_a_prefix():
    s3 = ["A is b.", "C is d.", "E is f."]
    assert H.map_reply("C is d.", s3) == {"status": "subsequence", "j": None, "subset": [2]}
    assert H.map_reply("A is b. E is f.", s3) == {"status": "subsequence", "j": None, "subset": [1, 3]}
    assert H.outcome_key({"status": "subsequence", "subset": [2]}) == "sub:2" != H.outcome_key({"status": "prefix", "j": 2})
    assert H.map_reply("A is b. C is d.", s3)["j"] == 2


def test_map_reply_uses_the_composers_kept_facts_and_never_guesses():
    rep = ["Aug follows Jul.", "Aug follows Jul."]                    # the same text for both facts
    assert "ambiguous" in H.map_reply("Aug follows Jul.", rep)["why"]
    assert H.map_reply("Aug follows Jul.", rep, kept_idx=[1]) == {"status": "prefix", "j": 1, "subset": [1]}
    assert H.map_reply("Aug follows Jul.", rep, kept_idx=[2]) == {"status": "subsequence", "j": None, "subset": [2]}
    assert "do not give" in H.map_reply("Aug follows Jul.", rep, kept_idx=[1, 2])["why"]
    assert "do not give" in H.map_reply("Aug follows Jul.", S2, kept_idx=[2])["why"]
    assert "kept facts" in H.map_reply("", S2, kept_idx=[1])["why"]                    # nothing spoken, yet a fact kept
    assert "in-order" in H.map_reply("Jul follows June. Aug follows Jul.", S2, kept_idx=[2, 1])["why"]
    assert H.map_reply("", S2, kept_idx=[]) == {"status": "prefix", "j": 0, "subset": []}


def _reply_case(cid="R1", sentences=S2, gathered=None, expect="U", **kw):
    g = gathered if gathered is not None else [list(f) for f in H.CONTROL_PAIR]
    c = {"id": cid, "gathered": g, "sentences": list(sentences), "sentence": " ".join(sentences), "expect": expect, "class": "r",
         "reply": True}
    c.update(kw)
    return c


def test_validate_reply_case():
    assert H.validate_reply_case(_reply_case()) is None
    unterminated = _reply_case(sentences=["Aug follows Jul", "Jul follows June"])
    assert H.validate_reply_case(unterminated) is None                                    # sentence = the flag-off join
    unterminated["sentence"] = "Aug follows Jul. Jul follows June."
    assert H.validate_reply_case(unterminated) is None                                    # or the flag-on join
    bad = [(_reply_case(sentences=S2[:1], gathered=[list(H.CONTROL_PAIR[0])]), "2 <= k"),
           (_reply_case(sentences=S2 + ["X is y."]), "3 sentences for 2"),
           (_reply_case(gathered=[list(H.CONTROL_PAIR[0])] * 2), "gathered twice"),
           (_reply_case(sentence="Aug follows Jul. Jul follows June!"), "not its 'sentences' joined"),
           (_reply_case(sentences=["Aug follows Jul.", "  "]), "non-blank"),
           (_reply_case(gathered=[["aug", "follows"], ["jul", "follows", "june"]]), "three-string")]
    for c, why in bad:
        assert why in (H.validate_reply_case(c) or ""), (c, H.validate_reply_case(c))
    c = _reply_case()
    del c["sentences"]
    assert "sentences" in H.validate_reply_case(c)
    # k <= 4 (Claude ruling 2026-10-21: every selection is judged, at most 15 items per case)
    g5 = [["m%d" % i, "follows", "m%d" % (i + 1)] for i in range(5)]
    s5 = ["M%d follows M%d." % (i, i + 1) for i in range(5)]
    assert H.REPLY_MAX_K == 4
    assert H.validate_reply_case(_reply_case(sentences=s5[:4], gathered=g5[:4])) is None
    assert "2 <= k <= 4" in H.validate_reply_case(_reply_case(sentences=s5, gathered=g5))


def test_the_stand_in_renders_by_fact_never_by_call_index():
    g = [list(f) for f in H.CONTROL_PAIR]
    ren = H.ReplyRenderer(g, S2, batch=True)
    assert ren.render_svo(*g[1]) == ("Jul follows June.", None)                           # fact 2 asked first -> sentence 2
    assert ren.render_svo_batch([tuple(g[1]), tuple(g[0])]) == ([("Jul follows June.", None), ("Aug follows Jul.", None)], 0.0)
    assert ren.asked == [("render_svo", 2), ("render_svo_batch", 2), ("render_svo_batch", 1)]
    with pytest.raises(KeyError):
        ren.render_svo("x", "y", "z")
    assert ren.errors and "not a gathered fact" in ren.errors[0]
    assert not hasattr(H.ReplyRenderer(g, S2), "render_svo_batch")


# ---- the adjudication reader -------------------------------------------------------------------------------------------------------
def test_load_reply_labels_with_and_without_prefix_labels():
    cases = [_reply_case("R1"), _reply_case("R2"), _reply_case("R3", expect="U"), {"id": "X1", "expect": "U"}]
    label = {"R1": "U", "R2": "U", "R3": "B", "X1": "U"}                                  # R3 relabelled U->B
    assert H.load_reply_labels({}, cases, label) == {i: {"prefix": None, "subset": None, "why": H.NO_LABELS}
                                                     for i in ("R1", "R2", "R3")}         # non-reply X1 is not listed
    adj = {"prefix_labels": {"R1": ["F", "U"], "R2": ["U"], "R3": ["U", "U"]}, "subset_labels": {"R1": {"2": "F"}}}
    got = H.load_reply_labels(adj, cases, label)
    assert got["R1"] == {"prefix": ["F", "U"], "subset": {"2": "F"}, "why": None}
    assert "not a list of 2" in got["R2"]["why"] and got["R2"]["prefix"] is None
    assert "not the case's own label 'B'" in got["R3"]["why"]                            # the full reply's label must be the case's
    assert "not one of" in H.check_prefix_labels(["X", "U"], 2, "U")
    # subset_labels: every in-order selection, prefixes and the full reply included (one key family)
    assert H.check_subset_labels({"1": "U", "2": "F", "1+2": "U"}, 2, "U") is None
    assert H.check_subset_labels({"1+3": "U", "2": "B", "1+2+3": "B"}, 3, "B") is None
    for bad, why in (({"2+1": "U"}, "not the key"), ({"01": "U"}, "not the key"), ({"3": "U"}, "not the key"),
                     ({"2": "Q"}, "not one of"), ({"1+2": "F"}, "not the case's own label")):
        assert why in (H.check_subset_labels(bad, 2, "U") or ""), bad


def test_subset_labels_alone_label_every_selection_and_the_two_families_must_agree():
    cases, label = [_reply_case("R1")], {"R1": "U"}
    got = H.load_reply_labels({"subset_labels": {"R1": {"1": "F", "2": "U", "1+2": "U"}}}, cases, label)
    assert got["R1"] == {"prefix": None, "subset": {"1": "F", "2": "U", "1+2": "U"}, "why": None}
    m, why = H.reply_label_map(None, {"1": "F", "2": "U"}, 2, "U")
    assert why is None and m == {"1": "F", "2": "U", "1+2": "U"}                         # the full reply is the case's own label
    m, why = H.reply_label_map(["F", "U"], {"2": "B"}, 2, "U")
    assert why is None and m == {"1": "F", "1+2": "U", "2": "B"}                         # prefix_labels still read (compatibility)
    assert "disagree on selection 1" in H.reply_label_map(["F", "U"], {"1": "U"}, 2, "U")[1]
    assert H.reply_label_map(None, None, 2, "U") == (None, H.NO_LABELS)


# ---- the summary counting (pure) ---------------------------------------------------------------------------------------------------
def _jr(j, path="sequential", seed=42):
    return {"seed": seed, "path": path, "status": "prefix", "j": j, "subset": list(range(1, j + 1))}


def _sr(sub, path="sequential", seed=42):
    return {"seed": seed, "path": path, "status": "subsequence", "j": None, "subset": sub}


UR = {"seed": 42, "path": "batched", "status": "UNDEFINED", "j": None, "subset": None, "why": "the text matches no in-order selection"}


def _rrow(i, label, k, allow, pl, sl=None, other=None):
    """A reply row: the gate/composer entries all F, the reply entry `allow` in the allowlist arm and `other` (default: all j = 0)
    in stock and B2h."""
    row = _row(i, label, {"stock": [F], "b2h": [F], "allowlist": [F]})
    other = other if other is not None else [_jr(0), _jr(0, "batched")]
    row.update(reply={"stock": other, "b2h": other, "allowlist": allow}, k=k, prefix_labels=pl, subset_labels=sl, labels_why=None)
    return row


def test_reply_summary_counts_kept_labels_by_prefix_label_at_any_seed_through_either_path():
    rows = [
        _rrow("R1", "U", 2, [_jr(1), _jr(2, "batched"), _jr(1, seed=43), _jr(1, "batched", 43)], ["F", "U"]),   # full reply via batched
        _rrow("R2", "U", 2, [_jr(0), _jr(0, "batched")], ["U", "U"]),                                         # nothing emitted
        _rrow("R3", "B", 3, [_jr(2), _jr(2, "batched")], ["F", "B", "B"]),                                    # cut to 2 of 3
        _rrow("R4", "U", 2, [_sr([2]), _jr(0, "batched")], ["U", "U"], sl={"2": "F"}),                        # labelled selection
        _row("X1", "U", {"stock": [F], "b2h": [F], "allowlist": [F]}),                                        # not a reply case
    ]
    s = H.summarise(rows)
    rp = s["allowlist"]["reply"]
    assert (rp["n"], rp["n_na"], rp["UNDEFINED"]) == (4, 1, 0)
    assert rp["kept"] == {"U": {"n": 1, "ids": ["R1"]}, "B": {"n": 1, "ids": ["R3"]}, "F": {"n": 2, "ids": ["R1", "R4"]}}
    assert rp["kept_by_path"]["batched"]["U"] == ["R1"] and rp["kept_by_path"]["sequential"]["U"] == []
    assert rp["j_distribution"]["all"] == {"0": 3, "1": 3, "2": 3, "sub:2": 1}
    assert rp["j_distribution"]["sequential"] == {"0": 1, "1": 2, "2": 1, "sub:2": 1}
    assert rp["case_emitted"] == {"R1": ["1", "2"], "R2": ["0"], "R3": ["2"], "R4": ["0", "sub:2"]}
    assert (rp["full_reply_ids"], rp["cut_ids"], rp["nothing_emitted_ids"], rp["subsequence_ids"]) == (["R1"], ["R1", "R3"], ["R2"], ["R4"])
    assert s["allowlist"]["VALID"] is True and s["stock"]["reply"]["kept"]["U"]["n"] == 0
    assert "reject_reason_tags" in rp and "reject_reason_tags" not in s["stock"]["reply"]


@pytest.mark.parametrize("row,why", [
    (_rrow("R1", "U", 2, [_jr(2), _jr(2, "batched")], None), "no subset_labels"),                         # labels missing
    (_rrow("R1", "U", 3, [_jr(1), _jr(2, "batched")], None, sl={"1": "U"}), "no adjudicated label"),      # an emitted prefix unlabelled
    (_rrow("R1", "U", 2, [_jr(2), _jr(2, "batched")], ["U", "F"]), "not the case's own label"),            # full label != case label
    (_rrow("R1", "U", 2, [_jr(2), UR], ["F", "U"]), "no in-order"),                                       # one UNDEFINED record
    (_rrow("R1", "U", 2, [_sr([2]), _jr(0, "batched")], ["U", "U"]), "no adjudicated label"),             # unlabelled selection
    (_rrow("R1", "U", 2, [], ["U", "U"]), "no reply records")])
def test_a_reply_case_without_usable_labels_or_with_an_undefined_record_is_undefined_and_the_arm_not_valid(row, why):
    s = H.summarise([row])
    rp = s["allowlist"]["reply"]
    assert rp["UNDEFINED"] == 1 and rp["undefined_ids"] == ["R1"] and why in rp["undefined_why"]["R1"], rp["undefined_why"]
    assert all(rp["kept"][lab]["n"] == 0 for lab in H.LABELS)                              # never counted
    assert s["allowlist"]["VALID"] is False and s["allowlist"]["n_undefined_case_entries"] == 1
    assert H.exit_code(s) == H.EXIT_ALLOWLIST_UNDEFINED


def test_without_reply_cases_the_summary_and_exit_code_are_as_before():
    rows = [_row("U1", "U", {"stock": [F], "b2h": [F], "allowlist": [F]})]
    s = H.summarise(rows)
    assert s["allowlist"]["reply"]["n"] == 0 and s["allowlist"]["reply"]["n_na"] == 1 and s["allowlist"]["VALID"] is True
    assert H.exit_code(s) == 0 and H.exit_code(s, reply_controls_ok=False) == H.EXIT_CONTROL_FAILED


# ---- the reply controls (pure) -------------------------------------------------------------------------------------------------------
def test_reply_control_ok_rules():
    full = [_jr(2), _jr(2, "batched")]
    assert H.reply_control_ok(full, "full", 2) and not H.reply_control_ok([_jr(2), _jr(1, "batched")], "full", 2)
    assert H.reply_control_ok([_jr(1), _jr(1, "batched")], "first", 2) and not H.reply_control_ok(full, "first", 2)
    assert H.reply_control_ok([_sr([2]), _jr(0, "batched")], "not_first", 2)
    assert not H.reply_control_ok([_jr(1), _jr(0, "batched")], "not_first", 2)             # a prefix holds sentence 1
    assert not H.reply_control_ok([_jr(2), UR], "full", 2) and not H.reply_control_ok([], "full", 2)   # UNDEFINED / nothing: fail
    assert [c["id"] for c in H.REPLY_CONTROL_CASES] == ["RP1", "RN1", "RN0", "RNJ"]
    assert all(H.validate_reply_case(c) is None and c["gathered"] == H.CONTROL_PAIR for c in H.REPLY_CONTROL_CASES)


def test_reply_control_report_fails_only_on_the_allowlist_arm():
    cases = [c for c in H.REPLY_CONTROL_CASES if c["id"] in ("RP1", "RNJ")]
    good = {"RP1": [_jr(2), _jr(2, "batched")], "RNJ": [_jr(1), _jr(1, "batched")]}
    rows = [{"id": i, "k": 2, "reply": {"stock": [_jr(2), _jr(2, "batched")], "b2h": [_jr(2), _jr(2, "batched")], "allowlist": recs}}
            for i, recs in good.items()]
    rep = H.reply_control_report(rows, cases)
    assert rep["STRICT_OK"] is True and rep["arm_ok"]["stock"] is False                   # stock RNJ keeps the repeat: reported only
    assert rep["outcomes"]["stock"]["RNJ"] == ["2"] and rep["outcomes"]["allowlist"]["RNJ"] == ["1"]
    rows[1]["reply"]["allowlist"] = [_jr(2), _jr(1, "batched")]
    assert H.reply_control_report(rows, cases)["STRICT_OK"] is False


# ---- the reply entry through the REAL composer on a mocked chat --------------------------------------------------------------------
class FakeReplyChat(FakeChat):
    """FakeChat plus what `render_paragraph` reads from a ChatBrain: raw_mode, renderer and the (absent) spiking recall surface."""
    raw_mode = False
    renderer = None

    def spiking_recall_surface(self, a, v, p):
        return None

    def spiking_recall_parts(self, a, v, p):
        return None

    def _raw(self, svo):
        return " ".join(str(x) for x in svo)                               # ChatBrain._raw: the brain's raw triple


@pytest.mark.parametrize("path", H.REPLY_PATHS)
def test_reply_one_runs_the_real_paragraph_path_and_restores_the_chat_and_the_environment(monkeypatch, path):
    set_arm(monkeypatch, "allowlist")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "1")
    monkeypatch.delenv("BRAIN_RICH_BATCH_RENDER", raising=False)
    chat = FakeReplyChat(gate={"allowlist": (True, {})})
    rc = composer_over(chat)
    rec = H.reply_one(rc, chat, H.Trace(chat), _reply_case(), 42, path)
    assert rec["status"] == "prefix" and rec["j"] == 2 and rec["text"] == "Aug follows Jul. Jul follows June.", rec
    assert rec["asked"] == [1, 2] and rec["kept_idx"] == [1, 2] and rec["n_gate_calls"] == 4        # 2 candidates + 2 joined checks
    assert chat.renderer is None and os.environ["BRAIN_SPIKING_MOUTH_RECALL"] == "1" and "BRAIN_RICH_BATCH_RENDER" not in os.environ


def test_reply_one_sees_through_the_composers_coercion_of_a_none_allowlist_verdict(monkeypatch):
    set_arm(monkeypatch, "allowlist")
    chat = FakeReplyChat(gate={"allowlist": (None, None)})                 # the composer reads None as a reject (bool / `if not ok`)
    rec = H.reply_one(composer_over(chat), chat, H.Trace(chat), _reply_case(), 42, "sequential")
    assert rec["status"] == "UNDEFINED" and "coerces" in rec["why"] and rec["text"] == ""        # it emitted nothing: still UNDEFINED


def test_reply_one_undefined_on_an_exception_the_wrong_path_or_another_mouths_text(monkeypatch):
    set_arm(monkeypatch, "allowlist")
    chat = FakeReplyChat(gate={"allowlist": AssertionError("boom")})
    rec = H.reply_one(composer_over(chat), chat, H.Trace(chat), _reply_case(), 42, "sequential")
    assert rec["status"] == "UNDEFINED" and rec["src"] == "error" and "AssertionError" in rec["why"]
    raw = FakeReplyChat(gate={"allowlist": (True, {})})
    raw.raw_mode = True                                                    # the composer speaks its raw triples, not the mouth
    rec = H.reply_one(composer_over(raw), raw, H.Trace(raw), _reply_case(), 42, "batched")
    assert rec["status"] == "UNDEFINED" and "batched path was not taken" in rec["why"]
    rec = H.reply_one(composer_over(raw), raw, H.Trace(raw), _reply_case(), 42, "sequential")
    assert rec["status"] == "UNDEFINED" and "no in-order" in rec["why"]


def test_an_invalid_reply_case_is_undefined_in_every_arm_and_a_non_reply_case_is_na():
    chats = {42: FakeReplyChat()}
    rows = H.run_rows([_reply_case(sentence="nope"), {"id": "X1", "gathered": GATHERED, "sentence": SENT, "expect": "U"}],
                      {"R1": "U", "X1": "U"}, (42,), chats, {42: composer_over(chats[42])}, {42: H.Trace(chats[42])}, alt_vocab=False)
    assert rows[1]["reply"] == "n/a" and set(rows[0]["gate"]) == set(H.ARMS)              # the gate/composer entries still run
    assert all(len(r) == 1 and r[0]["status"] == "UNDEFINED" and "not its 'sentences'" in r[0]["why"] for r in rows[0]["reply"].values())


def _reply_cases_dir(tmp_path, prefix_labels=True):
    cases = [{"id": "X1", "gathered": GATHERED, "sentence": SENT, "expect": "U", "class": "c"}, _reply_case("R1", expect="U")]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    adj = {"relabel_to_B": {}}
    if prefix_labels:
        adj["prefix_labels"] = {"R1": ["F", "U"]}
    (tmp_path / "adjudication.json").write_text(json.dumps(adj))
    return str(tmp_path)


def _reply_objects(gate):
    def make(seed, kb):
        chat = FakeReplyChat(gate=gate)
        return chat, composer_over(chat)
    return make


def test_main_counts_the_reply_entry_from_prefix_labels_and_exits_nonzero_without_them(tmp_path):
    gate = {"stock": (True, {}), "b2h": (True, {}), "allowlist": (True, {})}
    out = tmp_path / "o.json"
    code = H.main([_reply_cases_dir(tmp_path), str(out), "--seeds", "42,43", "--equiv-n", "0"], control_cases=[],
                  objects=_reply_objects(gate))
    d = json.loads(out.read_text())
    assert code == 0, d["summary"]["allowlist"]["reply"]
    rp = d["summary"]["allowlist"]["reply"]
    assert rp["kept"]["U"] == {"n": 1, "ids": ["R1"]} and rp["j_distribution"]["all"] == {"2": 4}       # 2 seeds x 2 paths
    assert d["rows"][0]["reply"] == "n/a" and len(d["rows"][1]["reply"]["allowlist"]) == 4
    assert d["reply_controls"]["cases"] == [] and d["summary"]["allowlist"]["entries"]["gate"]["U"]["n"] == 2
    sub = tmp_path / "nolabels"
    sub.mkdir()
    code = H.main([_reply_cases_dir(sub, prefix_labels=False), str(sub / "o.json"), "--seeds", "42", "--equiv-n", "0"],
                  control_cases=[], objects=_reply_objects(gate))
    assert code == H.EXIT_ALLOWLIST_UNDEFINED


# ---- end to end on the REAL composer (BridgeParser + ChatBrain + RichAnswerComposer, synthetic store triples) ----------------------
E2E_SEEDS = (42, 43)


@pytest.fixture(scope="module")
def real_objects():
    kb = H.kb_of([tuple(f) for f in H.CONTROL_PAIR])
    chats, comps, traces = {}, {}, {}
    for sd in E2E_SEEDS:
        chats[sd], comps[sd] = H.build_objects(sd, kb)
        traces[sd] = H.Trace(chats[sd])
    return chats, comps, traces


def test_reply_controls_on_the_real_composer_both_paths_two_seeds(real_objects):
    chats, comps, traces = real_objects
    before = {k: os.environ.get(k) for k in ("BRAIN_SPIKING_MOUTH_RECALL", "BRAIN_RICH_BATCH_RENDER", "BRAIN_TOUCHPOINT_A_FACT_CLAUSE")}
    cc = H.REPLY_CONTROL_CASES
    rows = H.run_rows(cc, {c["id"]: c["expect"] for c in cc}, E2E_SEEDS, chats, comps, traces, entries=())
    rep = H.reply_control_report(rows, cc)
    assert rep["STRICT_OK"] is True, rep["outcomes"]
    allow = rep["outcomes"]["allowlist"]
    assert allow == {"RP1": ["2"], "RN1": ["1"], "RN0": ["sub:2"], "RNJ": ["1"]}, allow
    for r in rows:                                                          # every seed x path, both paths really taken
        recs = r["reply"]["allowlist"]
        assert [(x["seed"], x["path"]) for x in recs] == [(sd, p) for sd in E2E_SEEDS for p in H.REPLY_PATHS]
    rnj = [x for x in rows[3]["reply"]["allowlist"]]
    assert all("both state" in x["cut"][0] for x in rnj)                    # RNJ is cut by the JOINED check, not a lone check
    # flag OFF: stock emits the full faithful reply on both paths, and keeps the repeat (it has no joined check)
    assert rep["outcomes"]["stock"]["RP1"] == ["2"] and rep["ok"]["stock"]["RP1"] is True
    assert rep["outcomes"]["stock"]["RNJ"] == ["2"] and rep["arm_ok"]["stock"] is False
    assert all(x["text"] == "Aug follows Jul. Jul follows June." for x in rows[0]["reply"]["stock"])
    assert all(ch.renderer is None for ch in chats.values())
    assert {k: os.environ.get(k) for k in before} == before


def test_the_reply_summary_tags_reject_reasons_with_sentence_numbers_read_as_n():
    cut1 = "allowlist: sentences 1 and 2 both state (aug, follows, jul): a reply that states one fact twice"
    cut2 = "allowlist: sentences 2 and 3 both state (jul, follows, june): a reply that states one fact twice"
    recs = [dict(_jr(1), reasons=[cut1, "allowlist: not gathered: (x, y, z)"], cut=[cut1]), dict(_jr(2, "batched"), reasons=[cut2], cut=[cut2])]
    rp = H.summarise([_rrow("R1", "U", 3, recs, ["U", "U", "U"])])["allowlist"]["reply"]
    assert rp["reject_reason_tags"] == {"every_check": {"sentences N and N both state": 2, "not gathered": 1},
                                        "joined_cut": {"sentences N and N both state": 2}}


def test_the_reply_summary_reads_every_selection_prefixes_included_from_subset_labels():
    sl = {"1": "F", "2": "U", "1+2": "U"}
    rows = [_rrow("R1", "U", 2, [_jr(1), _sr([2], "batched")], None, sl=sl),
            _rrow("R2", "U", 2, [_jr(2), _jr(2, "batched")], None, sl=sl)]
    rp = H.summarise(rows)["allowlist"]["reply"]
    assert rp["UNDEFINED"] == 0 and rp["kept"]["F"]["ids"] == ["R1"] and rp["kept"]["U"]["ids"] == ["R1", "R2"]
    assert rp["kept_by_path"]["sequential"]["F"] == ["R1"] and rp["kept_by_path"]["batched"]["U"] == ["R1", "R2"]
