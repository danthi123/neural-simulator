"""Honesty-boundary guard for the multistep chase's never-told terminal (Stage-2 HANDOFF item 18b, 2026-10-13).

`webapp.gnw_multistep_deliberation.deliberate_multistep` used to return a >=2-hop chase terminal as a plain
`[agent, action, terminal]` list, so the reply spoke a conclusion the brain was never told ("the lynx chases the
mite") as a recalled assertion with `derived` false. A >=2-hop terminal is now a `ChainedSVO` carrying the hop facts
the workspace committed, which the handler flags `derived` and frames with `frame_derived_answer`; a one-hop terminal
(the told fact itself) stays plain; `BRAIN_GNW_MULTISTEP_DERIVED=0` restores the pre-fix plain list.

LOGIC-LEVEL: a synthetic composer + a synthetic chase trace shaped like `confidence_gated_chase`'s real return
(the same shape tests/test_reasoning_transitive_chat_wirein.py uses). No brain is built; the keystone workspace
carries its own GO and is not re-verified here.
"""
import types

import pytest

import webapp.gnw_multistep_deliberation as G
from research.runners.compositional_chain_route import ChainedSVO, frame_derived_answer

CHAIN = {("lynx", "chase"): "vole", ("vole", "chase"): "beetle", ("beetle", "chase"): "mite"}


class _FakeComposer:
    kb = []

    def query_patient(self, agent, action):
        return CHAIN.get((agent, action))


def _chat():
    return types.SimpleNamespace(inner=types.SimpleNamespace(composer=_FakeComposer()))


def _fake_chase(chat, agent, action, *, seed=42, lesion=False):
    trace, x, cyc = [], agent, 0
    if lesion:
        return None, {"trace": [{"cycle": 1, "x": x, "target": CHAIN.get((x, action)), "committed": None,
                                 "n_ignited": 0, "conf": 0.0, "action": "ABSTAIN"}],
                      "resolved_hops": 0, "cycles": 1, "abstained": True}
    while True:
        cyc += 1
        tgt = CHAIN.get((x, action))
        if tgt is None:
            trace.append({"cycle": cyc, "x": x, "target": None, "committed": None, "n_ignited": 0, "conf": 0.0,
                          "action": "COMMIT"})
            break
        trace.append({"cycle": cyc, "x": x, "target": tgt, "committed": tgt, "n_ignited": 1, "conf": 0.94,
                      "action": "ADVANCE"})
        x = tgt
    hops = sum(1 for e in trace if e["action"] == "ADVANCE")
    terminal = x if hops else None
    return terminal, {"trace": trace, "resolved_hops": hops, "cycles": cyc, "halted_at_cap": False,
                      "abstained": False}


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    monkeypatch.setattr(G, "multistep_chase", _fake_chase)
    monkeypatch.setattr(G, "_all_concepts", lambda composer: ["beetle", "lynx", "mite", "vole"])
    monkeypatch.delenv("BRAIN_GNW_MULTISTEP_DERIVED", raising=False)
    monkeypatch.delenv("BRAIN_GNW_MULTISTEP_LESION", raising=False)


def test_three_hop_terminal_is_derived_with_committed_hops():
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert isinstance(out, ChainedSVO)
    assert list(out) == ["lynx", "chase", "mite"]
    assert out.derived_from == [["lynx", "chase", "vole"], ["vole", "chase", "beetle"], ["beetle", "chase", "mite"]]
    assert info["derived"] is True and info["derived_from"] == out.derived_from
    assert info["reason"] == "chain_terminal" and info["terminal"] == "mite" and info["resolved_hops"] == 3


def test_two_hop_terminal_is_derived():
    out, info = G.deliberate_multistep(_chat(), ["vole", "chase", "beetle"], "what does the vole chase?")
    assert isinstance(out, ChainedSVO) and list(out) == ["vole", "chase", "mite"]
    assert out.derived_from == [["vole", "chase", "beetle"], ["beetle", "chase", "mite"]]
    assert info["derived"] is True


def test_one_hop_terminal_is_the_told_fact_and_stays_plain():
    out, info = G.deliberate_multistep(_chat(), ["beetle", "chase", "mite"], "what does the beetle chase?")
    assert type(out) is list and out == ["beetle", "chase", "mite"]
    assert not hasattr(out, "derived_from")
    assert info["derived"] is False and info["derived_from"] is None


def test_escape_restores_the_plain_assertion(monkeypatch):
    monkeypatch.setenv("BRAIN_GNW_MULTISTEP_DERIVED", "0")
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert type(out) is list and out == ["lynx", "chase", "mite"]
    assert info["derived"] is False and info["derived_from"] is None
    assert info["reason"] == "chain_terminal" and info["terminal"] == "mite"


def test_abstain_and_moat_paths_unchanged(monkeypatch):
    out, info = G.deliberate_multistep(_chat(), None, "what does the mite chase?")
    assert out is None and info["reason"] == "already_abstained"
    monkeypatch.setenv("BRAIN_GNW_MULTISTEP_LESION", "1")
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert out is None and info["abstained"] is True and info["derived"] is False


def test_trace_terminal_disagreement_keeps_derived_with_generic_lead(monkeypatch):
    def _bad(chat, agent, action, *, seed=42, lesion=False):
        return "mite", {"trace": [], "resolved_hops": 3, "cycles": 4}
    monkeypatch.setattr(G, "multistep_chase", _bad)
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert isinstance(out, ChainedSVO) and out.derived_from == [] and info["derived"] is True
    assert frame_derived_answer("the lynx chases the mite", out.derived_from).startswith(
        "I reasoned this myself rather than being told it directly.")


def test_committed_path_reads_the_trace_only():
    _t, meta = _fake_chase(None, "lynx", "chase")
    assert G.committed_path("lynx", meta) == ["lynx", "vole", "beetle", "mite"]
    assert G.committed_path("lynx", {}) == ["lynx"]


def test_reply_framing_is_the_c2_infer_lead():
    out, _info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    framed = frame_derived_answer("the lynx chases the mite", out.derived_from)
    assert framed == ("I derived this from: lynx chase vole; vole chase beetle; beetle chase mite. "
                      "the lynx chases the mite")


def test_installed_gate_returns_derived_on_chase_form_only(monkeypatch):
    chat = _chat()
    chat.gate = lambda q: (["lynx", "chase", "vole"] if "lynx" in q else None)
    assert G.install_multistep_gate(chat) is True
    out = chat.gate("what does the lynx chase all the way?")
    assert isinstance(out, ChainedSVO) and list(out) == ["lynx", "chase", "mite"]
    assert chat._last_gnw_multistep["derived"] is True
    plain = chat.gate("what does the lynx chase?")                 # no chase marker: pure pass-through
    assert type(plain) is list and plain == ["lynx", "chase", "vole"]


class _FakeRich:
    def __init__(self):
        self._said, self._conversation_said = [], set()

    def render_paragraph(self, facts):
        return " ".join("the %s %ss the %s." % (a, v, p) for a, v, p in facts), [list(f) for f in facts], []


def test_confidence_cap_truncation_keeps_the_derived_lead():
    from webapp.confidence_forthcoming_chat import apply_cap
    hops = [["lynx", "chase", "vole"], ["vole", "chase", "beetle"], ["beetle", "chase", "mite"]]
    facts = [["lynx", "chase", "mite"]] + hops
    r = {"answer": frame_derived_answer("x", hops), "facts": facts, "derived": True, "derived_from": hops,
         "abstained": False, "dropped": []}
    out, trace = apply_cap(_FakeRich(), r, 2, False)
    assert trace["reason"] == "low_confidence_capped"
    assert out["answer"].startswith("I derived this from: lynx chase vole; vole chase beetle; beetle chase mite. ")
    assert out["derived"] is True
    plain = dict(r, derived=False, derived_from=[])
    out2, _ = apply_cap(_FakeRich(), plain, 2, False)
    assert not out2["answer"].startswith("I derived")
