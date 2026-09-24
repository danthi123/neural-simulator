"""CI guards for A6 / reasoning-transitive-chat (webapp/reasoning_transitive_chat.py, default-OFF
BRAIN_TRANSITIVE_CHAT / BRAIN_TRANSITIVE_LESION). See
research/findings/2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md.

These are LOGIC-LEVEL guards against a synthetic composer + a synthetic chase trace shaped exactly like
`research.runners._gnw_reentrant_metacog_gated_deliberation_derisk.confidence_gated_chase`'s real return value
(cycle/x/target/committed/n_ignited/conf/action dicts). They exercise `resolve_transitive_query`'s own dispatch,
direct-fact short-circuit, trace-walking and lesion-collapse logic in isolation from the real keystone GNW
workspace (which already carries its own 6-seed GO + anti-cheat battery and is not re-verified here) -- a cheap,
RAM-free CI guard, NOT a substitute for the production-handler seed-7/6-seed gate
(`research/runners/reasoning_transitive_chat_gate.py`), which needs a real tiny-demo brain build and is staged
separately (this box read `avail=2-3G` against `tools/mem_ok.sh` at authoring time -- several other concurrent
brain-building processes were live).
"""
import os
import sys
import types

os.environ.setdefault("BRAIN_TRANSITIVE_CHAT", "")

import webapp.reasoning_transitive_chat as T  # noqa: E402

CHAIN = ["e0", "e1", "e2", "e3", "e4"]


class _FakeComposer:
    def __init__(self, facts):
        self.facts = facts

    def query_patient(self, agent, action):
        return self.facts.get((agent, action))


class _FakeChat:
    def __init__(self, facts):
        self.inner = types.SimpleNamespace(composer=_FakeComposer(facts))


def _facts_for(chain):
    return {(chain[i], "precede"): chain[i + 1] for i in range(len(chain) - 1)}


def _fake_trace(facts, agent, lesion):
    """Mirror confidence_gated_chase's real trace-entry shape for a chase over `facts` along 'precede'."""
    trace = []
    x = agent
    cyc = 0
    if lesion:
        trace.append({"cycle": 1, "x": x, "target": facts.get((x, "precede")), "committed": None,
                      "n_ignited": 0, "conf": 0.0, "action": "ABSTAIN"})
        return trace
    while True:
        cyc += 1
        tgt = facts.get((x, "precede"))
        if tgt is None:
            trace.append({"cycle": cyc, "x": x, "target": None, "committed": None,
                          "n_ignited": 0, "conf": 0.0, "action": "COMMIT"})
            break
        trace.append({"cycle": cyc, "x": x, "target": tgt, "committed": tgt,
                      "n_ignited": 1, "conf": 0.9, "action": "ADVANCE"})
        x = tgt
    return trace


def _install_fake_chase(facts):
    """Inject a fake `webapp.gnw_multistep_deliberation` module into sys.modules so
    `resolve_transitive_query`'s lazy `from webapp.gnw_multistep_deliberation import multistep_chase` never
    imports the real (numpy/GNW-bridge-touching) module -- these tests are about THIS module's own dispatch
    logic, not a re-verification of the already-GO'd keystone chase."""
    def _fake_multistep_chase(chat, agent, action, *, seed=42, lesion=False):
        trace = _fake_trace(facts, agent, lesion)
        terminal = next((e["committed"] for e in reversed(trace) if e["action"] == "ADVANCE"), None)
        return terminal, {"trace": trace}

    mod = types.ModuleType("webapp.gnw_multistep_deliberation")
    mod.multistep_chase = _fake_multistep_chase
    sys.modules["webapp.gnw_multistep_deliberation"] = mod


def setup_module(_module):
    _install_fake_chase(_facts_for(CHAIN))


def teardown_module(_module):
    sys.modules.pop("webapp.gnw_multistep_deliberation", None)


# ── parsing ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_parse_matches_does_shape():
    assert T.parse_transitive_question("does e0 precede e3?") == ("e0", "precede", "e3")
    assert T.parse_transitive_question("Does E0 Precede E3") == ("e0", "precede", "e3")


def test_parse_rejects_other_shapes():
    assert T.parse_transitive_question("what does the wolf bite") is None
    assert T.parse_transitive_question("does e0 precede") is None
    assert T.parse_transitive_question(None) is None


# ── byte-identity off (the flag's first-line short-circuit) ────────────────────────────────────────────────────────
def test_flag_off_is_a_pure_noop(monkeypatch):
    monkeypatch.delenv("BRAIN_TRANSITIVE_CHAT", raising=False)
    # a bare `object()` (no .inner/.composer at all) proves the composer is NEVER touched when the flag is off.
    assert T.resolve_transitive_query(object(), "does e0 precede e3?") is None


def test_flag_explicit_zero_is_a_pure_noop(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "0")
    assert T.resolve_transitive_query(object(), "does e0 precede e3?") is None


def test_flag_on_nonmatching_question_falls_through(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    assert T.resolve_transitive_query(object(), "what does the wolf bite") is None


# ── G1: held-out non-adjacent pairs derive correctly ────────────────────────────────────────────────────────────
def test_nonadjacent_pairs_derive_with_correct_chain(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.delenv("BRAIN_TRANSITIVE_LESION", raising=False)
    chat = _FakeChat(_facts_for(CHAIN))
    pairs = [(CHAIN[i], CHAIN[j]) for i in range(len(CHAIN)) for j in range(i + 2, len(CHAIN))]
    assert len(pairs) == 6
    for a, b in pairs:
        matched, svo = T.resolve_transitive_query(chat, f"does {a} precede {b}?")
        assert matched is True
        assert svo is not None and hasattr(svo, "derived_from")
        assert list(svo) == [a, "precede", b]
        expect = [[CHAIN[i], "precede", CHAIN[i + 1]] for i in range(CHAIN.index(a), CHAIN.index(b))]
        assert svo.derived_from == expect


# ── G2: an adjacent pair is a plain recall, never mislabeled as derived ─────────────────────────────────────────
def test_adjacent_pairs_are_plain_recall_not_derived(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.delenv("BRAIN_TRANSITIVE_LESION", raising=False)
    chat = _FakeChat(_facts_for(CHAIN))
    for i in range(len(CHAIN) - 1):
        a, b = CHAIN[i], CHAIN[i + 1]
        matched, svo = T.resolve_transitive_query(chat, f"does {a} precede {b}?")
        assert matched is True
        assert svo == [a, "precede", b]
        assert not hasattr(svo, "derived_from")


# ── G3: reverse-direction / unrelated-entity negatives -> honest abstain, never a false "yes" ───────────────────
def test_negative_controls_abstain_honestly(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.delenv("BRAIN_TRANSITIVE_LESION", raising=False)
    chat = _FakeChat(_facts_for(CHAIN))
    for a, b in [("e4", "e0"), ("e2", "e1"), ("e0", "zzz")]:
        matched, svo = T.resolve_transitive_query(chat, f"does {a} precede {b}?")
        assert matched is True
        assert svo is None


# ── G5: the lesion collapses every non-adjacent derivation but leaves the adjacent short-circuit UNCHANGED ─────
def test_lesion_collapses_nonadjacent_but_spares_adjacent(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_LESION", "1")
    chat = _FakeChat(_facts_for(CHAIN))
    matched, svo = T.resolve_transitive_query(chat, "does e0 precede e3?")
    assert matched is True and svo is None                          # a genuinely-derivable pair now abstains

    matched, svo = T.resolve_transitive_query(chat, "does e0 precede e1?")
    assert matched is True and svo == ["e0", "precede", "e1"]        # the direct-fact short-circuit is UNAFFECTED


# ── G4: a freshly-scrambled premise mapping (never hardcoded) still resolves correctly ──────────────────────────
def test_scrambled_premise_mapping_resolves_via_live_traversal(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.delenv("BRAIN_TRANSITIVE_LESION", raising=False)
    scrambled = ["zeta", "mu", "kappa", "rho", "iota"]
    facts = _facts_for(scrambled)
    _install_fake_chase(facts)          # re-point the fake chase at the NEW mapping
    chat = _FakeChat(facts)
    matched, svo = T.resolve_transitive_query(chat, f"does {scrambled[0]} precede {scrambled[3]}?")
    assert matched is True
    assert svo is not None and hasattr(svo, "derived_from")
    assert svo.derived_from == [[scrambled[i], "precede", scrambled[i + 1]] for i in range(3)]
    _install_fake_chase(_facts_for(CHAIN))   # restore for any test running after this one in-process
