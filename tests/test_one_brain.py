"""M2 of the one-brain shared-memory plan: the Brain / Context split behind BRAIN_ONE_BRAIN (webapp/one_brain.py).

docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md sections 1, 2, 5 and the owner rulings of section 9. CPU-only,
no brain build: the real `brain_chat` handler runs with the brain build and `brain_reply` stubbed (as in
tests/test_brain_turn_lock.py), and the Brain's bind / source tags / deferred mouth / zero / freeze run on fake
chats. These pin BEHAVIOUR. Reply byte-identity at OFF, and the M0 probe arms at ON, are measured on the real brain
(docs/drafts/one-brain-m2/WORKLOG.md).
"""
import json
import re
import threading
import time
from pathlib import Path

import pytest

from webapp import brain_turn_lock as L
from webapp import one_brain as OB

REPO = Path(__file__).resolve().parents[1]


# ── the flag ────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,expected", [(None, False), ("", False), ("1", True), ("on", True), (" YES ", True),
                                          ("0", False), ("off", False), ("maybe", False)])
def test_flag_parse_default_off(val, expected):
    env = {} if val is None else {OB.ONE_BRAIN_ENV: val}
    assert OB.one_brain_on(env) is expected


def test_freeze_needs_the_one_brain():
    assert not OB.freeze_learning_on({OB.FREEZE_ENV: "1"})
    assert OB.freeze_learning_on({OB.FREEZE_ENV: "1", OB.ONE_BRAIN_ENV: "1"})
    assert not OB.freeze_learning_on({OB.ONE_BRAIN_ENV: "1"})


def test_default_anchor_is_off():
    assert OB._ONE_BRAIN_DEFAULT_ON is False


def test_brain_key():
    k = ("s", "tiny-demo", "stub")
    assert OB.brain_key(k) == ("__one_brain__", "tiny-demo", "stub")
    assert OB.is_brain_key(OB.brain_key(k)) and not OB.is_brain_key(k)


# ── Context registry: the server's dicts are views with exact dict semantics ───────────────────────────────────────

def _ops():
    """A fixed op sequence over keys a..c exercising set / re-set / pop / setdefault / pop-then-set order."""
    return [("set", "a", 1), ("set", "b", 2), ("set", "a", 3), ("setdefault", "c", 4), ("setdefault", "a", 9),
            ("pop", "a", None), ("set", "a", 5), ("pop", "zz", None), ("set", "b", None), ("del", "c", None)]


def _apply(d, ops):
    out = []
    for op, k, v in ops:
        if op == "set":
            d[k] = v
        elif op == "setdefault":
            out.append(d.setdefault(k, v))
        elif op == "pop":
            out.append(d.pop(k, "missing"))
        elif op == "del":
            del d[k]
        out.append((list(d), [d[x] for x in d], k in d, d.get(k, "dflt"), len(d)))
    return out


def test_field_view_matches_a_plain_dict():
    reg = OB.ContextRegistry()
    view = reg.view("mood")
    assert _apply(view, _ops()) == _apply({}, _ops())
    assert reg.get("a").mood == 5 and reg.get("b").mood is None and "b" in view   # a stored None stays present


def test_view_keyerror_and_values_live_on_the_context():
    reg = OB.ContextRegistry()
    v1, v2 = reg.view("mood"), reg.view("pmem")
    with pytest.raises(KeyError):
        v1["nope"]
    v1["k"] = {"valence": 1.0}
    assert reg.get("k").mood == {"valence": 1.0} and "k" not in v2   # another field of the same Context: absent
    reg.remove("k")
    assert "k" not in v1 and list(v1) == [] and reg.get("k") is None


@pytest.fixture
def fresh(monkeypatch):
    """A fresh Context registry + Brain wired into the server's names (every view re-pointed at it)."""
    from webapp import server as S
    from webapp import continuous_engine as CE
    reg = OB.ContextRegistry()
    monkeypatch.setattr(S, "_CONTEXTS", reg)
    for name, field in (("_BRAIN_CHATS", "chat"), ("_BRAIN_RICH", "rich"), ("_SESSION_MOOD", "mood"),
                        ("_SESSION_WORLDVIEW", "worldview"), ("_SESSION_MULTIREF", "multiref"),
                        ("_SESSION_SILENT_WM", "silent_wm"), ("_SESSION_DISCOURSE", "discourse"),
                        ("_SESSION_PMEM", "pmem"), ("_SESSION_SELFINIT", "selfinit")):
        monkeypatch.setattr(S, name, reg.view(field))
    brain = OB.Brain()
    monkeypatch.setattr(S, "_ONE_BRAIN", brain)
    monkeypatch.setattr(OB, "TICK_ADAPTER", OB.TickAdapter(brain))
    for name in ("_LAST_REQUEST", "_LAST_TURN_END", "_REQ_SEQ", "_WANDER_BUDGET", "_INNER_LIFE", "_LAST_WANDERED",
                 "_RESET_SEQ", "_CONSOLIDATED_TOPICS", "_SLEEP_REPLAYED", "_LAST_SLEEP_KB", "_LAST_HOMEO_KB"):
        monkeypatch.setattr(CE, name, {})
    return S, reg, brain


def test_server_registries_are_fields_of_one_context(fresh):
    S, reg, _ = fresh
    key = ("s", "tiny-demo", "stub")
    S._SESSION_MOOD[key] = {"valence": 0.1}
    S._SESSION_PMEM[key] = "latch"
    S._BRAIN_RICH[key] = "thread"
    ctx = reg.get(key)
    assert (ctx.mood, ctx.pmem, ctx.rich) == ({"valence": 0.1}, "latch", "thread")
    S._evict_brain_chat_session(key)
    assert reg.get(key) is None and key not in S._SESSION_MOOD and key not in S._BRAIN_RICH


# ── the attribute classification (fails loudly on an unclassified per-turn write) ───────────────────────────────────

def _chat_attr_writes():
    names = set()
    pat = re.compile(r"(?:\bchat|\b_chat\d*|\bc)\.([A-Za-z_]\w*)\s*=(?!=)")
    for p in sorted((REPO / "webapp").glob("*.py")):
        names |= set(pat.findall(p.read_text(errors="replace")))
    import ast
    tree = ast.parse((REPO / "research" / "runners" / "brain_chat_tui.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ChatBrain":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name != "__init__":
                    for n in ast.walk(fn):
                        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) \
                                and isinstance(n.value, ast.Name) and n.value.id == "self":
                            names.add(n.attr)
    return names


def test_every_per_turn_chat_attribute_is_classified():
    writes = _chat_attr_writes()
    assert len(writes) > 30                                          # the scan sees the real writes
    classified = set(OB.CHAT_CONTEXT_ATTRS) | set(OB.CHAT_BRAIN_ATTRS)
    missing = sorted(writes - classified)
    assert not missing, ("chat attributes written by a turn but classified neither conversation-level nor "
                         "brain-level in webapp/one_brain.py: %s" % missing)
    assert not (set(OB.CHAT_CONTEXT_ATTRS) & set(OB.CHAT_BRAIN_ATTRS))


# ── fake chats ──────────────────────────────────────────────────────────────────────────────────────────────────────

class _Reg:
    def __init__(self):
        self.ident = 0
        self.slots = [0, 0, 0, 0]
        self._boundary = False


class _Comp:
    def __init__(self, facts):
        self.D = 2
        self.kb = [({"agent": a, "action": v, "patient": p, "polarity": "AFFIRM"}, None) for a, v, p in facts]
        self.store_conns = [("c", i) for i in range(len(facts) * self.D)]
        self._store_dirty = self._persistent_dirty = False
        self._csr_cache = {"x": 1}

    def hear(self, sentence, voice="active", polarity=None):
        a, v, p = sentence.split()
        self.kb.append(({"agent": a, "action": v, "patient": p, "polarity": polarity or "AFFIRM"}, None))
        self.store_conns += [("c", 0)] * self.D
        return {"agent": a, "action": v, "patient": p}


class _Inner:
    def __init__(self, comp):
        self.composer = comp

    def hear(self, sentence, voice="active", polarity=None):
        return self.composer.hear(sentence, voice, polarity=polarity)


class _Agent:
    def __init__(self, inner):
        self.agent = inner
        self.wm = None
        self.bcw = None
        self._referent_history = []
        self._event_register = _Reg()


class _Rend:
    name = "fake-template"


class _FakeChat:
    def __init__(self, facts=(("dog", "chase", "cat"),), renderer=None):
        self.inner = _Inner(_Comp(facts))
        self.agent = _Agent(self.inner)
        self.renderer = renderer if renderer is not None else _Rend()
        self._boundary_seen = False
        self._heard_any_clause = False
        self.last_word_role_meta = None
        self._refresh_facts()

    def _refresh_facts(self):
        self.stored_facts = OB.learned_triples(self)


def test_bind_swaps_conversation_state_and_shares_brain_state():
    brain = OB.Brain()
    bk = ("__one_brain__", "tiny-demo", "stub")
    chat = brain.adopt(bk, _FakeChat(), "fake")
    a, b = OB.Context(("A", "tiny-demo", "stub")), OB.Context(("B", "tiny-demo", "stub"))
    assert brain.bind(bk, a) is chat
    wm_a, ws_a = object(), object()
    chat._boundary_seen = True
    chat._affect_drives_workspace = ws_a
    chat.agent.wm = wm_a
    chat.agent._referent_history.append("dog")
    chat.agent._event_register.slots = [1, 2, 3, 0]
    chat._da_tag_capture = "brain ledger"                            # brain-level: shared
    brain.bind(bk, b)                                               # B starts from the pristine conversation state
    assert chat._boundary_seen is False and not hasattr(chat, "_affect_drives_workspace")
    assert chat.agent.wm is None and chat.agent._referent_history == [] and chat.agent._event_register.slots == [0] * 4
    assert chat._da_tag_capture == "brain ledger"
    chat.agent._referent_history.append("cat")
    brain.bind(bk, a)                                               # A's own state comes back exactly
    assert chat._boundary_seen is True and chat._affect_drives_workspace is ws_a and chat.agent.wm is wm_a
    assert chat.agent._referent_history == ["dog"] and chat.agent._event_register.slots == [1, 2, 3, 0]
    brain.bind(bk, b)
    assert chat.agent._referent_history == ["cat"]                   # and B's
    assert brain.bind(bk, b) is chat                                 # re-binding the resident one is a no-op


def test_an_evicted_resident_conversation_starts_clean_when_it_returns():
    brain = OB.Brain()
    bk = ("__one_brain__", "tiny-demo", "stub")
    chat = brain.adopt(bk, _FakeChat(), "fake")
    a = OB.Context(("A", "tiny-demo", "stub"))
    brain.bind(bk, a)
    chat._boundary_seen = True
    brain.release(a)                                                 # reset of A while A is resident
    a2 = OB.Context(("A", "tiny-demo", "stub"))                      # the fresh conversation under the same key
    brain.bind(bk, a2)
    assert chat._boundary_seen is False


# ── host source tags + the "someone told me" frame ──────────────────────────────────────────────────────────────────

def test_taught_elsewhere_reads_the_host_tag():
    brain = OB.Brain()
    bk = ("__one_brain__", "b", "r")
    brain.note_taught(bk, [("dog", "chase", "cat")], [("dog", "chase", "cat"), ("wug", "eat", "dax")], "A")
    assert brain.taught_elsewhere(bk, ["wug", "eat", "dax"], "B")
    assert not brain.taught_elsewhere(bk, ["wug", "eat", "dax"], "A")       # the teller's own conversation
    assert not brain.taught_elsewhere(bk, ["dog", "chase", "cat"], "B")     # a build fact has no tag
    assert not brain.taught_elsewhere(bk, None, "B")


def test_source_frame_is_anonymous_and_only_for_another_conversations_fact(fresh):
    from fastapi.responses import JSONResponse
    S, _, brain = fresh
    bk = ("__one_brain__", "tiny-demo", "stub")
    brain.fact_sources[bk] = {("wug", "eat", "dax"): ("A", "tiny-demo", "stub")}
    B = ("B", "tiny-demo", "stub")

    def frame(body, key=B):
        return json.loads(S._one_brain_source_frame(JSONResponse(body), bk, key).body)

    r = frame({"answer": "the wug eats the dax", "abstained": False, "recalled_svo": ["wug", "eat", "dax"]})
    assert r["answer"] == "Someone told me: the wug eats the dax"
    assert r["one_brain_source"]["reason"] == "host_tag" and r["one_brain_source"]["placement"] == "lead"
    r = frame({"answer": "the dog chases the cat", "abstained": False, "recalled_svo": ["dog", "chase", "cat"],
               "supporting_facts": [["dog", "chase", "cat"], ["wug", "eat", "dax"]]})
    assert r["answer"] == "the dog chases the cat (Someone told me part of that.)"
    # the M2 s7 one_brain arm's B_ask shape: the told fact recalled first, then build facts the brain knew itself ->
    # the part note, never a lead that attributes the build facts to "someone"
    r = frame({"answer": "the wug eats the dax the dog chases the cat", "abstained": False,
               "recalled_svo": ["wug", "eat", "dax"], "supporting_facts": [["wug", "eat", "dax"], ["dog", "chase", "cat"]]})
    assert r["answer"] == "the wug eats the dax the dog chases the cat (Someone told me part of that.)"
    assert r["one_brain_source"]["facts"] == [["wug", "eat", "dax"]] and r["one_brain_source"]["n_facts_stated"] == 2
    r = frame({"answer": "the dog chases the cat", "abstained": False, "recalled_svo": None,
               "derived": True, "derived_from": [["dog", "chase", "cat"], ["wug", "eat", "dax"]]})
    assert r["one_brain_source"]["placement"] == "part_note"                # a derived answer with a told hop
    same = {"answer": "the wug eats the dax", "abstained": False, "recalled_svo": ["wug", "eat", "dax"]}
    assert frame(same, ("A", "tiny-demo", "stub")) == same                 # the teller's conversation: untouched
    ab = {"answer": "I don't know about that.", "abstained": True, "recalled_svo": ["wug", "eat", "dax"]}
    assert frame(ab) == ab


# ── the tonic mood baseline ─────────────────────────────────────────────────────────────────────────────────────────

def test_tonic_seeds_a_new_conversation_and_leaves_a_single_one_unchanged(fresh, monkeypatch):
    S, _, brain = fresh
    appr = {"n_hits": 2, "valence": -0.8, "arousal": 0.6}
    A, B = ("A", "tiny-demo", "stub"), ("B", "tiny-demo", "stub")
    monkeypatch.delenv(OB.ONE_BRAIN_ENV, raising=False)
    off = S._update_session_mood(A, appr)
    S._evict_brain_chat_session(A)
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    on = S._update_session_mood(A, appr)
    assert on == off                                                 # the first conversation: unchanged
    assert brain.tonic["valence"] == pytest.approx(0.1 * -0.8) and brain.n_tonic_updates == 1
    seeded = S._update_session_mood(B, {"n_hits": 0})                # B's first, neutral, turn
    assert seeded == brain.tonic_seed() and seeded["valence"] < 0    # B starts from the brain's tonic baseline


# ── the handler at ON (brain build + brain_reply stubbed) ───────────────────────────────────────────────────────────

@pytest.fixture
def on_handler(fresh, monkeypatch):
    from fastapi.responses import JSONResponse
    S, reg, brain = fresh
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    monkeypatch.setenv("BRAIN_CHAT_RENDERER", "stub")
    monkeypatch.delenv(L.TIMEOUT_ENV, raising=False)
    monkeypatch.setattr(S, "_pin_bridge_backend", lambda: None)
    st = {"builds": 0, "chats": [], "reloads": []}

    def build(b, r):
        st["builds"] += 1
        return st.get("chat_factory", _FakeChat)(), "stub-build"

    monkeypatch.setattr(S, "_build_chat_brain", build)
    monkeypatch.setattr(S, "_reload_persisted_learning", lambda key, chat: st["reloads"].append(key))

    def fake_reply(chat, req, source, key):
        st["chats"].append(id(chat))
        st.setdefault("boundary_seen_at_start", []).append(chat._boundary_seen)
        if req.message.startswith("teach "):
            chat.inner.hear(req.message[len("teach "):])
            chat._refresh_facts()
        chat._boundary_seen = True                                   # a conversation-level write
        svo = req.message.split()[-3:] if req.message.startswith("ask ") else None
        return JSONResponse({"answer": " ".join(svo) if svo else "ok", "abstained": False, "recalled_svo": svo,
                             "key": list(key), "held": L.held_by_current_thread()})

    monkeypatch.setattr(S, "brain_reply", fake_reply)
    return S, reg, brain, st


def _turn(S, sess, msg, **kw):
    return json.loads(S.brain_chat(S.BrainChatRequest(session=sess, message=msg, brain="tiny-demo",
                                                      renderer="stub", **kw)).body)


def test_on_conversations_share_one_brain_and_keep_their_own_state(on_handler):
    S, reg, brain, st = on_handler
    _turn(S, "A", "teach wug eat dax")
    r = _turn(S, "B", "ask wug eat dax")
    assert st["builds"] == 1 and len(set(st["chats"])) == 1         # one shared ChatBrain for both conversations
    assert st["reloads"] == [("__one_brain__", "tiny-demo", "stub")]  # the brain-level persistence identity
    assert st["boundary_seen_at_start"] == [False, False]            # B did not inherit A's conversation state
    assert r["answer"] == "Someone told me: wug eat dax" and r["one_brain_source"]["frame"] == "someone_told_me"
    assert len(S._BRAIN_CHATS) == 0 and set(reg.keys()) == {("A", "tiny-demo", "stub"), ("B", "tiny-demo", "stub")}
    a = _turn(S, "A", "ask wug eat dax")
    assert a["answer"] == "wug eat dax" and "one_brain_source" not in a     # the teller hears no frame
    assert st["boundary_seen_at_start"][-1] is True                  # A's own state came back


def test_a_reset_conversation_is_a_new_conversation_for_the_source_frame(on_handler):
    """The conversation id dies with the conversation: after a reset under the same session key, a fact the earlier
    conversation taught is framed like any other conversation's (the brain keeps the fact, not who told it)."""
    S, reg, brain, st = on_handler
    _turn(S, "A", "teach wug eat dax")
    first = reg.get(("A", "tiny-demo", "stub")).conversation_id
    r = _turn(S, "A", "ask wug eat dax", reset=True)
    assert reg.get(("A", "tiny-demo", "stub")).conversation_id != first
    assert r["answer"] == "Someone told me: wug eat dax"


def test_interlocutor_stays_on_the_context(on_handler):
    S, reg, brain, st = on_handler
    _turn(S, "A", "teach wug eat dax", interlocutor="Alice")
    r = _turn(S, "B", "ask wug eat dax", user="Bob")
    assert reg.get(("A", "tiny-demo", "stub")).interlocutor == "Alice"
    assert reg.get(("B", "tiny-demo", "stub")).interlocutor == "Bob"
    assert "Alice" not in json.dumps(r)
    assert "Alice" not in repr(brain.fact_sources)


def test_off_handler_keeps_a_chat_per_conversation(on_handler, monkeypatch):
    S, reg, brain, st = on_handler
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "0")
    _turn(S, "A", "teach wug eat dax")
    r = _turn(S, "B", "ask wug eat dax")
    assert st["builds"] == 2 and len(set(st["chats"])) == 2 and "one_brain_source" not in r
    assert set(S._BRAIN_CHATS) == {("A", "tiny-demo", "stub"), ("B", "tiny-demo", "stub")}
    assert brain.chats == {}


def test_reset_route_at_on_takes_the_lock_and_forgets_the_wander_state(on_handler, monkeypatch):
    S, reg, brain, st = on_handler
    from webapp import continuous_engine as CE
    key = ("A", "tiny-demo", "stub")
    _turn(S, "A", "teach wug eat dax")
    CE._INNER_LIFE[key] = [{"wandered": "cat"}]
    seen = []
    orig = S._evict_brain_chat_session
    monkeypatch.setattr(S, "_evict_brain_chat_session", lambda k: (seen.append(L.held_by_current_thread()), orig(k)))
    out = json.loads(S.brain_chat_reset(S.BrainChatResetRequest(session="A", brain="tiny-demo",
                                                                  renderer="stub")).body)
    assert out["reset"] is True and seen == [True]
    assert key not in CE._INNER_LIFE and reg.get(key) is None
    assert brain.chat(OB.brain_key(key)) is not None                 # the brain (and what it learned) stays
    assert brain.taught_elsewhere(OB.brain_key(key), ["wug", "eat", "dax"], ("B", "tiny-demo", "stub"))


# ── the deferred mouth: generation after the lock, VERIFY under it ──────────────────────────────────────────────────

class _Fac:
    CONSTRAIN_TEMPLATE = "Say: {a} {v} {p}."

    def __init__(self, log):
        self.log = log

    def _generate(self, prompt):
        self.log.append(("gen", prompt, L.held_by_current_thread()))
        return "a fluent %s" % prompt, prompt, 0.0


class _GenRend:
    name = "fake-generative"

    def __init__(self, log, slow=0.0):
        self._fac = _Fac(log)
        self.log, self.slow = log, slow

    def render_svo(self, a, v, p):
        self.log.append(("render", (a, v, p), L.held_by_current_thread()))
        time.sleep(self.slow)
        return "%s really %s the %s" % (a, v, p), None


def test_deferred_mouth_generates_outside_the_lock_and_verifies_under_it(on_handler, monkeypatch):
    S, reg, brain, st = on_handler
    log = []
    st["chat_factory"] = lambda: _FakeChat(renderer=_GenRend(log))

    def reply(chat, req, source, key):
        surface, _ = chat.renderer.render_svo("dog", "chase", "cat")   # in the lock: the template articulation
        from fastapi.responses import JSONResponse
        return JSONResponse({"answer": "Well, " + surface, "abstained": False, "recalled_svo": ["dog", "chase", "cat"]})

    monkeypatch.setattr(S, "brain_reply", reply)
    verified = []
    monkeypatch.setattr(_FakeChat, "_verify", lambda self, s, a, svo, _v=verified: (
        _v.append(L.held_by_current_thread()), "really" in s)[1], raising=False)
    r = _turn(S, "A", "what does the dog chase")
    assert r["answer"] == "Well, dog really chase the cat"
    assert [e[2] for e in log if e[0] == "render"] == [False]         # the generative mouth ran without the turn lock
    assert verified == [True]                                         # its VERIFY re-took the lock
    assert r["one_brain_mouth"]["n_verified"] == 1


def test_deferred_mouth_keeps_the_verified_template_when_prose_fails_verify(on_handler, monkeypatch):
    S, reg, brain, st = on_handler
    st["chat_factory"] = lambda: _FakeChat(renderer=_GenRend([]))

    def reply(chat, req, source, key):
        surface, _ = chat.renderer.render_svo("dog", "chase", "cat")
        from fastapi.responses import JSONResponse
        return JSONResponse({"answer": surface, "abstained": False, "recalled_svo": ["dog", "chase", "cat"]})

    monkeypatch.setattr(S, "brain_reply", reply)
    monkeypatch.setattr(_FakeChat, "_verify", lambda self, s, a, svo: False, raising=False)
    r = _turn(S, "A", "what does the dog chase")
    assert r["answer"] == "The dog chases cat." and r["one_brain_mouth"]["n_kept_template"] == 1


def test_a_second_conversation_gets_its_turn_while_the_first_one_speaks(on_handler, monkeypatch):
    S, reg, brain, st = on_handler
    log = []
    st["chat_factory"] = lambda: _FakeChat(renderer=_GenRend(log, slow=0.6))
    order = []

    def reply(chat, req, source, key):
        order.append(("brain", req.session, time.perf_counter()))
        from fastapi.responses import JSONResponse
        if req.session == "A":
            surface, _ = chat.renderer.render_svo("dog", "chase", "cat")
            return JSONResponse({"answer": surface, "abstained": False, "recalled_svo": ["dog", "chase", "cat"]})
        return JSONResponse({"answer": "ok", "abstained": False, "recalled_svo": None})

    monkeypatch.setattr(S, "brain_reply", reply)
    monkeypatch.setattr(_FakeChat, "_verify", lambda self, s, a, svo: True, raising=False)
    out = {}
    ta = threading.Thread(target=lambda: out.setdefault("A", _turn(S, "A", "q")))
    ta.start()
    deadline = time.time() + 5
    while not any(e[0] == "render" for e in log) and time.time() < deadline:
        time.sleep(0.01)
    t0 = time.perf_counter()
    out["B"] = _turn(S, "B", "hi")                                   # admitted while A's mouth is generating
    took = time.perf_counter() - t0
    ta.join(5)
    assert took < 0.5 and out["B"]["answer"] == "ok"
    assert out["A"]["answer"] == "dog really chase the cat"


def test_mood_wrapper_hands_the_manner_to_the_deferred_mouth():
    from research.runners.affect_production_organ import MoodConditionedRenderer
    log = []
    mouth = OB.DeferredMouth(_GenRend(log))
    w = MoodConditionedRenderer(mouth)
    w.manner = "warmly, about {a}"
    surface, asserted = w.render_svo("dog", "chase", "cat")
    assert surface == "The dog chases cat." and asserted == ["dog", "chase", "cat"]
    calls = mouth.take_calls()
    assert calls[0]["manner"] == "warmly, about dog" and calls[0]["template"] == surface
    prose = mouth.generate(calls[0])
    assert prose == "a fluent Say: dog chase cat. warmly, about dog" and log[-1][0] == "gen"


def test_a_template_renderer_gets_no_deferred_mouth():
    brain = OB.Brain()
    chat = brain.adopt(("__one_brain__", "b", "r"), _FakeChat(), "fake")
    assert isinstance(chat.renderer, _Rend)
    gen = brain.adopt(("__one_brain__", "b", "q"), _FakeChat(renderer=_GenRend([])), "fake")
    assert isinstance(gen.renderer, OB.DeferredMouth)


# ── the experimenter instruments: freeze + zero ─────────────────────────────────────────────────────────────────────

def test_freeze_gate_blocks_the_shared_store_write(monkeypatch):
    brain = OB.Brain()
    chat = brain.adopt(("__one_brain__", "b", "r"), _FakeChat(), "fake")
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    monkeypatch.setenv(OB.FREEZE_ENV, "1")
    with pytest.raises(OB.LearningFrozen):                          # a refused write, not a silent success
        chat.inner.hear("wug eat dax")
    assert len(chat.inner.composer.kb) == 1
    monkeypatch.setenv(OB.FREEZE_ENV, "0")
    chat.inner.hear("wug eat dax")
    assert len(chat.inner.composer.kb) == 2


def test_frozen_acquisition_reports_no_acquisition_so_no_learner_observes_it(monkeypatch):
    """The real `ChatBrain._maybe_acquire` under the gate: the refused write makes it report no
    acquisition (None), so the category-inheritance organ, which feeds on the acquisition verdict, sees nothing."""
    from research.runners.brain_chat_tui import ChatBrain
    brain = OB.Brain()
    chat = brain.adopt(("__one_brain__", "b", "r"), _FakeChat(), "fake")
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    monkeypatch.setenv(OB.FREEZE_ENV, "1")
    acquire = ChatBrain._maybe_acquire.__get__(chat)                 # the production (B3 polar-assertion) path
    assert acquire("the wug eats the dax") is None and len(chat.inner.composer.kb) == 1
    monkeypatch.setenv(OB.FREEZE_ENV, "0")
    assert acquire("the wug eats the dax") == ["wug", "eat", "dax"] and len(chat.inner.composer.kb) == 2


def test_zero_drops_only_what_was_learned_since_the_build():
    brain = OB.Brain()
    bk = ("__one_brain__", "b", "r")
    chat = brain.adopt(bk, _FakeChat(facts=(("dog", "chase", "cat"), ("cat", "eat", "fish"))), "fake")
    chat.inner.hear("wug eat dax")
    chat.inner.hear("crow chase wren")
    brain.note_taught(bk, [], [("wug", "eat", "dax")], "A")
    chat._category_inherit_organ = "member->frame pathway taught wug->eat dax"   # built after the build: dropped
    chat._gen_proposer, chat._gen_nfacts = "proposer over 4 facts", 4
    comp = chat.inner.composer
    rec = brain.zero_learned(bk)
    assert rec["zeroed"] == 2 and rec["n_build_facts"] == 2
    assert rec["organs_dropped"] == ["_category_inherit_organ"] and chat._category_inherit_organ is None
    assert chat._gen_proposer is None
    assert [f["agent"] for f, _ in comp.kb] == ["dog", "cat"] and len(comp.store_conns) == 2 * comp.D
    assert comp._store_dirty and comp._persistent_dirty and comp._csr_cache == {}
    assert brain.fact_sources[bk] == {} and chat.stored_facts == [("dog", "chase", "cat"), ("cat", "eat", "fish")]


def test_zero_keeps_a_learned_organ_that_existed_at_the_build():
    brain = OB.Brain()
    bk = ("__one_brain__", "b", "r")
    fc = _FakeChat()
    fc.inner._source_provenance_monitor = "built with the brain"
    chat = brain.adopt(bk, fc, "fake")
    assert brain.zero_learned(bk)["organs_dropped"] == [] and chat.inner._source_provenance_monitor == "built with the brain"


def test_zero_store_function_is_refused_off_and_in_production(fresh, monkeypatch):
    S, _, _ = fresh
    monkeypatch.delenv(OB.ONE_BRAIN_ENV, raising=False)
    with pytest.raises(RuntimeError):
        S.one_brain_zero_learned_store()
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    monkeypatch.setenv("BRAIN_PRODUCTION", "1")
    with pytest.raises(RuntimeError):
        S.one_brain_zero_learned_store()


# ── sleep is a BRAIN event at ON ────────────────────────────────────────────────────────────────────────────────────

class _Adapter:
    def __init__(self, bks):
        self.bks, self.log = bks, []

    def brain_key_of(self, key):
        return OB.brain_key(key) if not OB.is_brain_key(key) else None

    def brain_keys(self):
        return list(self.bks)

    def chat(self, bk):
        return "chat:%s" % bk[1]

    def episodic(self, bk):
        return "organ:%s" % bk[1]


def test_sleep_replay_and_persist_are_brain_events(fresh, monkeypatch):
    from webapp import continuous_engine as CE
    monkeypatch.setenv(OB.ONE_BRAIN_ENV, "1")
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    monkeypatch.delenv("BRAIN_DA_ENCODING_SLEEP_TRIGGER", raising=False)
    calls = []
    monkeypatch.setattr(CE, "consolidate_sleep_replay", lambda k, o: calls.append(("sleep", k, o)))
    monkeypatch.setattr(CE, "consolidate_substrate_homeostasis",
                        lambda k, c, trigger: calls.append(("homeo", k, trigger)))
    import research.runners.cross_session_persistence as CSP
    monkeypatch.setattr(CSP, "save_session_learning", lambda k, c, o: calls.append(("save", k, c, o)))
    bk = ("__one_brain__", "tiny-demo", "stub")
    ad = _Adapter([bk])
    A, B = ("A", "tiny-demo", "stub"), ("B", "tiny-demo", "stub")
    now = 10_000.0
    CE._LAST_REQUEST.update({A: now - 400.0, B: now - 30.0})         # B spoke 30 s ago: the brain is awake
    CE.tick_idle_sessions({}, lambda: None, now=now, one_brain=ad)
    assert [c[0] for c in calls] == ["homeo"]                         # idle >= IDLE_SEC across all: the light pass only
    calls.clear()
    CE._LAST_REQUEST[B] = now - 301.0                                 # nobody for > 300 s: the brain sleeps, once
    CE.tick_idle_sessions({}, lambda: None, now=now, one_brain=ad)
    assert calls == [("sleep", bk, "organ:tiny-demo"), ("save", bk, "chat:tiny-demo", "organ:tiny-demo"),
                     ("homeo", bk, "idle_tick")]
    calls.clear()
    CE._LAST_REQUEST[B] = now - 5.0                                   # a turn just now: no brain event at all
    CE.tick_idle_sessions({}, lambda: None, now=now, one_brain=ad)
    assert calls == []


def test_off_tick_keeps_the_per_session_steps(fresh, monkeypatch):
    from webapp import continuous_engine as CE
    monkeypatch.delenv(OB.ONE_BRAIN_ENV, raising=False)
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    calls = []
    import research.runners.cross_session_persistence as CSP
    monkeypatch.setattr(CSP, "save_session_learning", lambda k, c, o: calls.append(("save", k)))
    monkeypatch.setattr(CE, "consolidate_sleep_replay", lambda k, o: calls.append(("sleep", k)))
    monkeypatch.setattr(CE, "consolidate_substrate_homeostasis", lambda k, c, trigger: calls.append(("homeo", k)))
    A = ("A", "tiny-demo", "stub")
    now = 10_000.0
    CE._LAST_REQUEST[A] = now - 400.0
    CE.tick_idle_sessions({A: {"valence": 0.0, "arousal": 0.0}}, lambda: type("Aff", (), {
        "read_differential": staticmethod(lambda v, lesion=False: {"differential": 0.0})})(), now=now,
        episodic_getter=lambda k: "organ", chat_getter=lambda k: "chat")
    assert ("sleep", A) in calls and ("save", A) in calls and ("homeo", A) in calls


# ── the OpenAI shim carries the interlocutor (request `user`, else the last user message's `name`) ──────────────────

def test_openai_shim_passes_the_interlocutor(monkeypatch):
    from fastapi.responses import JSONResponse
    from webapp import server as S
    from webapp import openai_shim as OAS
    seen = []
    monkeypatch.setattr(S, "brain_chat", lambda r: (seen.append(r), JSONResponse({"answer": "ok"}))[1])
    msgs = [{"role": "user", "content": "hi", "name": "Old"}, {"role": "assistant", "content": "x"},
            {"role": "user", "content": "what does the wug eat", "name": " Carol "}]
    S.openai_chat_completions(S.OpenAIChatRequest(messages=msgs))
    S.openai_chat_completions(S.OpenAIChatRequest(messages=msgs, user="dave"))
    S.openai_chat_completions(S.OpenAIChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert [r.interlocutor for r in seen] == ["Carol", "dave", None]
    assert OAS._last_user_name([{"role": "user", "content": "x", "name": "  "}]) is None


# ── PACKET-m2-fixes item 1 (major, honesty): the deferred mouth verifies MULTI-FACT prose against the whole set ─────

class _FakeClaimChat:
    """A ChatBrain stand-in with the production split: `_verify` = the old single-triple re-parse (accepts any
    surface that re-parses carrying this svo - the hole), `_verify_claim_set` = the claim moat over the gathered
    set (rejects a surface asserting ANY clause outside it; (None, None) when the moat is disabled)."""

    def __init__(self, moat=True):
        self.moat = moat
        self.verify_seen = []
        self.claim_seen = []

    def _verify(self, surface, asserted, svo):
        # the old single-triple hole, faithfully: a surface PASSES if it carries this triple ANYWHERE - which lets
        # a multi-clause sentence with an unsupported extra clause through (the packet's repro).
        self.verify_seen.append((surface, list(svo)))
        phrase = " ".join(str(x).lower() for x in svo)
        return phrase in " ".join(surface.lower().split())

    def _verify_claim_set(self, surface, gated):
        self.claim_seen.append((surface, [list(f) for f in gated]))
        if not self.moat:
            return None, None
        allowed = {str(x).lower() for f in gated for x in f[:3]}
        stop = {"the", "a", "an", "and", "too", "eats", "eat", "in", "of", "to", "is", "are"}
        unknown = [t for t in surface.lower().replace(",", " ").replace(".", " ").split()
                   if t not in stop and t not in allowed]
        return (not unknown, {"unknown": unknown})


class _FakeMouth:
    def __init__(self, prose):
        self._prose = prose

    def generate(self, call):
        return self._prose


def _ob_resp(answer, supporting_facts):
    from types import SimpleNamespace
    return SimpleNamespace(body=json.dumps(
        {"answer": answer, "supporting_facts": supporting_facts}).encode())


def test_m2_item1_multi_fact_prose_never_substitutes_an_unsupported_clause():
    """PACKET-m2-fixes item 1. Repro pinned: the single-fact `_verify` alone ACCEPTED
    'the wug eats the dax, and the cat eats the moon too' (n_verified=1) - a sentence carrying a clause no
    gathered fact supports. At ON, a reply with MORE THAN ONE supporting fact must verify each prose surface
    against the whole set (claim moat), so the unsupported clause rejects the whole surface and the template
    the brain already verified stays in place."""
    from webapp import server as S
    tmpl = "the wug eats the dax"
    answer = f"Right: {tmpl}."
    chat = _FakeClaimChat()
    mouth = _FakeMouth("the wug eats the dax, and the cat eats the moon too")
    calls = [{"method": "render_svo", "svo": ["the wug", "eats", "the dax"], "manner": "", "template": tmpl}]
    facts = [["the wug", "eats", "the dax"], ["the cat", "eats", "the fish"]]
    resp = S._one_brain_articulate(_ob_resp(answer, facts), (mouth, calls, chat))
    body = json.loads(bytes(resp.body))
    assert body["answer"] == answer, "unsupported clause reached the answer as verified prose"
    assert body["one_brain_mouth"]["n_verified"] == 0
    assert body["one_brain_mouth"]["n_kept_template"] == 1
    assert chat.claim_seen, "multi-fact reply was never verified against its supporting set"


def test_m2_item1_single_fact_reply_keeps_the_single_triple_path():
    """The single-fact reply keeps the old behaviour: `_verify` substitutes it, the claim moat is not consulted."""
    from webapp import server as S
    tmpl = "a wug dax meal happens"
    answer = f"Note: {tmpl}."
    chat = _FakeClaimChat()
    mouth = _FakeMouth("the wug eats the dax")
    calls = [{"method": "render_svo", "svo": ["the wug", "eats", "the dax"], "manner": "", "template": tmpl}]
    resp = S._one_brain_articulate(_ob_resp(answer, [["the wug", "eats", "the dax"]]), (mouth, calls, chat))
    body = json.loads(bytes(resp.body))
    assert body["answer"] == "Note: the wug eats the dax."
    assert body["one_brain_mouth"]["n_verified"] == 1
    assert chat.claim_seen == []


# ── PACKET-m2-fixes item 2 (major, honesty): the D5 referential disclosure is source-neutral at ON ──────────────────

def test_m2_item2_d5_recall_is_source_neutral_at_one_brain_on(monkeypatch):
    """PACKET-m2-fixes item 2. The D5 episodic store is brain-level with NO source tag: a topic conversation A
    stored can be recalled while conversation B is asking, and the old disclosure asserted
    'Earlier you brought up {topic}' - a false first-person claim about THIS conversation. At ON the reply must
    disclose source-neutrally ('Someone mentioned ...') or abstain.

    Drives `recall_disclosure` (a pure function of the recall record) with a record in `EpisodicDapMemory.recall`'s
    exact shape rather than a REAL store: a genuine numpy BTSP write is ~510s/topic (server.py _episodic_store_ok),
    i.e. pool-class, not unit-testable here. The scenario's precondition is pinned instead: BRAIN_EPISODIC_STORE=1
    forces that write path ON on numpy (it is deferred there by default), so cross-conversation storage CAN happen
    once forced - which is exactly why the disclosure needs a source gate."""
    monkeypatch.setenv("BRAIN_EPISODIC_STORE", "1")
    import research.runners.d5_episodic_production_organ as EP
    from webapp import server as S
    assert S._episodic_store_ok(), "BRAIN_EPISODIC_STORE=1 must force the numpy write path ON (scenario precondition)"
    graded = {r: 0.0 for r in EP.GRADED_READS}
    rec = {"topic": "wug", "slot": 0, "formed": True, "in_memory": True,
           "apical_cue": 0.85, "apical_perm": 0.0, "apical_nocue": 0.0,
           "graded_cue": dict(graded), "graded_perm": dict(graded), "graded_nocue": dict(graded),
           "reason": "completion"}
    # at ON - no source tag until M3 - the disclosure must not claim THIS conversation:
    out = EP.recall_disclosure(rec, None, source_attributed=False)
    assert "you brought up" not in out, out
    assert "Someone mentioned" in out, out
    assert "my hippocampal readout completes its assembly for it" in out, out   # the spiking completion still discloses
    # the source-attributed default (every existing non-ON caller) stays byte-identical to HEAD:
    head = EP.recall_disclosure(rec, None)
    assert head.startswith("Earlier you brought up wug"), head
    # the abstain branch is already source-honest (a non-recall claims nothing):
    fail = dict(rec, in_memory=False, formed=False)
    abstain = EP.recall_disclosure(fail, None, source_attributed=False)
    assert "I don't recall" in abstain, abstain


def test_m2_item2_wiring_gates_on_one_brain_on():
    """The gate is wired at the call site: the server passes source_attributed=not _one_brain_on()."""
    import inspect
    from webapp import server as S
    assert "source_attributed=not _one_brain_on()" in inspect.getsource(S)
