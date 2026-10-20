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
