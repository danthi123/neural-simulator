"""Honesty-boundary guard for the multistep chase's never-told terminal (Stage-2 HANDOFF item 18b, 2026-10-13).

`webapp.gnw_multistep_deliberation.deliberate_multistep` used to return a >=2-hop chase terminal as a plain
`[agent, action, terminal]` list, so the reply spoke a conclusion the brain was never told ("the lynx chases the
mite") as a recalled assertion with `derived` false. A >=2-hop terminal that is not itself stored is now a
`ChainedSVO` carrying the hop facts the workspace committed (none, with the generic lead, when a hop committed a
non-target concept); a one-hop terminal or a directly-stored conclusion stays plain; `BRAIN_GNW_MULTISTEP_DERIVED=0`
restores the pre-fix plain list. Downstream: the single-fact path (`webapp.server._single_fact_turn`) flags any
derived gate result, the rich path reports the hops (not the conclusion) as `supporting_facts` plus `derived_svo`
(`rich_answer_composer.derived_reply_fields`), and the confidence cap keeps the lead.

LOGIC-LEVEL: a synthetic composer + a synthetic chase trace shaped like `confidence_gated_chase`'s real return
(the same shape tests/test_reasoning_transitive_chat_wirein.py uses). No brain is built; the keystone workspace
carries its own GO and is not re-verified here.
"""
import types

import pytest

import webapp.gnw_multistep_deliberation as G
from research.runners.compositional_chain_route import ChainedSVO, frame_derived_answer

CHAIN = {("lynx", "chase"): "vole", ("vole", "chase"): "beetle", ("beetle", "chase"): "mite"}
HOPS3 = [["lynx", "chase", "vole"], ["vole", "chase", "beetle"], ["beetle", "chase", "mite"]]
LEAD3 = "I derived this from: lynx chase vole; vole chase beetle; beetle chase mite. "


class _FakeComposer:
    def __init__(self, extra=()):
        self.kb = [({"agent": a, "action": v, "patient": p}, None) for (a, v), p in CHAIN.items()]
        self.kb += [({"agent": a, "action": v, "patient": p}, None) for a, v, p in extra]

    def query_patient(self, agent, action):
        return CHAIN.get((agent, action))

    def elaborate(self, topic):
        return None

    def _assoc_graph(self):
        return {}


def _chat(extra=()):
    return types.SimpleNamespace(inner=types.SimpleNamespace(composer=_FakeComposer(extra), seed=7))


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
    for k in ("BRAIN_GNW_MULTISTEP_DERIVED", "BRAIN_GNW_MULTISTEP_LESION", "BRAIN_GNW_MULTISTEP",
              "BRAIN_D6_ENGRAM_READTIME", "BRAIN_TRANSITIVE_CHAT", "BRAIN_TRANSITIVE_SPIKING_REACH"):
        monkeypatch.delenv(k, raising=False)


# ── deliberate_multistep ─────────────────────────────────────────────────────────────────────────────────────────
def test_three_hop_terminal_is_derived_with_committed_hops():
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert isinstance(out, ChainedSVO)
    assert list(out) == ["lynx", "chase", "mite"]
    assert out.derived_from == HOPS3
    assert info["derived"] is True and info["derived_from"] == out.derived_from
    assert info["reason"] == "chain_terminal" and info["terminal"] == "mite" and info["resolved_hops"] == 3
    assert not info.get("hop_mismatch") and not info.get("conclusion_stored")


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


def test_conclusion_that_is_itself_stored_stays_plain():
    # lynx->vole->beetle->mite AND "the lynx chases the mite" told directly: the reply states a stored fact
    out, info = G.deliberate_multistep(_chat(extra=[("lynx", "chase", "mite")]), ["lynx", "chase", "vole"],
                                       "what does the lynx chase?")
    assert type(out) is list and out == ["lynx", "chase", "mite"]
    assert info["derived"] is False and info.get("conclusion_stored") is True


def test_conclusion_stored_only_in_the_ltm_shard_stays_plain():
    chat = _chat()
    shard = types.SimpleNamespace(kb=[({"agent": "lynx", "action": "chase", "patient": "mite"}, None)])
    chat.inner.composer.ltm = types.SimpleNamespace(shard_for=lambda concept: shard)
    out, info = G.deliberate_multistep(chat, ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert type(out) is list and info.get("conclusion_stored") is True


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
    assert info.get("hop_mismatch") is True
    assert frame_derived_answer("the lynx chases the mite", out.derived_from).startswith(
        "I reasoned this myself rather than being told it directly.")


def test_a_distractor_commit_never_names_an_untold_premise(monkeypatch):
    # the workspace WTA lets a distractor ('owl') win hop 2 although the composer proposed 'beetle'
    def _distractor(chat, agent, action, *, seed=42, lesion=False):
        trace = [{"x": "lynx", "target": "vole", "committed": "vole", "action": "ADVANCE"},
                 {"x": "vole", "target": "beetle", "committed": "owl", "action": "ADVANCE"},
                 {"x": "owl", "target": None, "committed": None, "action": "COMMIT"}]
        return "owl", {"trace": trace, "resolved_hops": 2, "cycles": 3}
    monkeypatch.setattr(G, "multistep_chase", _distractor)
    out, info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert isinstance(out, ChainedSVO) and list(out) == ["lynx", "chase", "owl"]
    assert out.derived_from == [] and info["hop_mismatch"] is True and info["derived_from"] == []
    assert "vole chase owl" not in frame_derived_answer("the lynx chases the owl", out.derived_from)


def test_committed_hops_reads_the_trace_only():
    _t, meta = _fake_chase(None, "lynx", "chase")
    assert G.committed_hops("lynx", "chase", meta) == (HOPS3, False)
    assert G.committed_hops("lynx", "chase", {}) == ([], False)


def test_reply_framing_is_the_c2_infer_lead():
    out, _info = G.deliberate_multistep(_chat(), ["lynx", "chase", "vole"], "what does the lynx chase?")
    assert frame_derived_answer("the lynx chases the mite", out.derived_from) == LEAD3 + "the lynx chases the mite"


def _install(chat):
    chat.gate = lambda q: (["lynx", "chase", "vole"] if "lynx" in q else None)
    assert G.install_multistep_gate(chat) is True
    return chat


def test_installed_gate_returns_derived_on_chase_form_only():
    chat = _install(_chat())
    out = chat.gate("what does the lynx chase all the way?")
    assert isinstance(out, ChainedSVO) and list(out) == ["lynx", "chase", "mite"]
    assert chat._last_gnw_multistep["derived"] is True
    plain = chat.gate("what does the lynx chase?")                 # no chase marker: pure pass-through
    assert type(plain) is list and plain == ["lynx", "chase", "vole"]


# ── the single-fact reply path (webapp.server._single_fact_turn, the branch brain_reply runs on rich=False) ──────
def _render(svo):
    a, v, p = svo
    return "the %s %ss the %s." % (a, v, p)


def test_single_fact_path_flags_and_frames_a_derived_gate_result():
    from webapp.server import _single_fact_turn
    chat = _install(_chat())
    chat.render = _render
    sf = _single_fact_turn(chat, "what does the lynx chase all the way?")
    assert sf["is_chain_route"] is True                          # fails if the gate-result derived check is removed
    assert sf["answer"] == LEAD3 + "the lynx chases the mite."
    assert sf["chain_raw_answer"] == "the lynx chases the mite."
    assert sf["abstained"] is False and sf["transitive_matched"] is False
    assert sf["gate_svo"].derived_from == HOPS3


def test_single_fact_path_leaves_a_plain_recall_and_an_abstain_alone():
    from webapp.server import _single_fact_turn
    chat = _install(_chat())
    chat.render = _render
    sf = _single_fact_turn(chat, "what does the lynx chase?")
    assert sf["is_chain_route"] is False and sf["answer"] == "the lynx chases the vole."
    sf = _single_fact_turn(chat, "what does the zorblat chase?")
    assert sf["gate_svo"] is None and sf["abstained"] is True and sf["is_chain_route"] is False


# ── the rich reply path (RichAnswerComposer + the confidence cap + the reply-level fields) ──────────────────────
def _rich_chat():
    chat = _install(_chat())
    chat.is_multiturn = False
    chat.raw_mode = True                                          # the raw triple is the verified sentence
    chat.renderer = None
    chat._raw = _render
    return chat


def test_rich_turn_reports_hops_not_the_conclusion_and_keeps_the_lead_after_capping():
    from research.runners.rich_answer_composer import RichAnswerComposer, derived_reply_fields
    from webapp.confidence_forthcoming_chat import apply_cap
    rich = RichAnswerComposer(_rich_chat(), max_sentences=4)
    r = rich.answer("what does the lynx chase all the way?")
    assert r["derived"] is True and r["derived_from"] == HOPS3
    assert r["facts"][0] == ["lynx", "chase", "mite"]
    assert r["answer"].startswith(LEAD3 + "the lynx chases the mite.")
    capped, trace = apply_cap(rich, r, 2, False)                 # low confidence: truncate to the 2-fact floor
    assert trace["reason"] == "low_confidence_capped" and len(capped["facts"]) == 2
    assert capped["answer"].startswith(LEAD3)                    # the inference lead survives the re-render
    supporting, derived_svo = derived_reply_fields(capped["facts"], capped["derived"], capped["derived_from"])
    assert derived_svo == ["lynx", "chase", "mite"]
    assert ["lynx", "chase", "mite"] not in supporting           # the never-told conclusion is not a supporting fact
    assert supporting == HOPS3


def test_reply_fields_keep_c2_and_plain_turns_unchanged():
    from research.runners.rich_answer_composer import derived_reply_fields
    c2_hops = [["heron", "hunt", "frog"], ["frog", "eat", "gnat"]]
    c2_facts = [["frog", "eat", "gnat"], ["gnat", "use", "wings"]]
    assert derived_reply_fields(c2_facts, True, c2_hops) == (c2_facts, ["frog", "eat", "gnat"])
    plain = [["dog", "chase", "cat"]]
    assert derived_reply_fields(plain, False, None) == (plain, None)
    assert derived_reply_fields([], True, c2_hops) == ([], None)


def test_confidence_cap_truncation_keeps_the_derived_lead():
    from webapp.confidence_forthcoming_chat import apply_cap

    class _FakeRich:
        _said, _conversation_said = [], set()

        def render_paragraph(self, facts):
            return " ".join(_render(f) for f in facts), [list(f) for f in facts], []
    facts = [["lynx", "chase", "mite"]] + HOPS3
    r = {"answer": frame_derived_answer("x", HOPS3), "facts": facts, "derived": True, "derived_from": HOPS3,
         "abstained": False, "dropped": []}
    out, trace = apply_cap(_FakeRich(), r, 2, False)
    assert trace["reason"] == "low_confidence_capped"
    assert out["answer"].startswith(LEAD3) and out["derived"] is True
    out2, _ = apply_cap(_FakeRich(), dict(r, derived=False, derived_from=[]), 2, False)
    assert not out2["answer"].startswith("I derived")
