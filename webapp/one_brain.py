"""ONE BRAIN, MANY CONVERSATIONS -- step M2: the `Brain` / `Context` split (docs/plans/2026-10-13-one-brain-shared-
memory-DESIGN.md sections 1, 2, 5; owner rulings section 9). DEFAULT OFF behind `BRAIN_ONE_BRAIN`.

`Context` (one per live conversation) owns the conversation-level (C) rows of the design's section-1 inventory: the
phasic mood EMA, the affective world-model prediction, the multi-referent WM, the activity-silent WM, the discourse
fold state, the prospective-memory latch, the self-init organ (a declared C residual: design row "B target, C
today"), the rich discourse thread, the typed interlocutor name, and -- while the conversation is not bound to the
brain -- the conversation-level fields of the shared ChatBrain (discourse WM, event-register slots, per-turn traces,
the #84 / DA-mode / swap workspaces). The server's `_SESSION_*` dicts, `_BRAIN_CHATS` and `_BRAIN_RICH` are
`ContextFieldView`s over one `ContextRegistry` (`_CONTEXTS`): every entry IS a Context field.

`Brain` (one per process) owns the brain-level (B) rows when the flag is ON: one shared ChatBrain per (brain, renderer)
-- its composer fact buffer (the learned store), the agent, the LTM prior (BRAIN_SHARED_LTM is forced on), the gate
wrappers and lazily built organs -- plus the tonic mood baseline, the host source tags of taught facts, and the
pristine conversation-level state a new conversation starts from. The D5 episodic organ, the idle-replay record, the
causal organ, the sleep-replay / consolidation / homeostasis bookkeeping and the persistence identity are keyed by the
brain key (`brain_key`) instead of the conversation key. The turn lock (webapp/brain_turn_lock.py, M0) is the Brain's
`turn_lock`.

FLAG OFF (the default): today's per-session behaviour, routed through the Context fields; nothing in this module
changes a reply (asserted by hash on data: docs/drafts/one-brain-m2/WORKLOG.md).

FLAG ON: a conversation is BOUND to the shared ChatBrain for its turn (`Brain.bind`, under the turn lock): the
outgoing conversation's C fields are saved into its Context and the incoming one's restored. Host code here is
bookkeeping only (which conversation's state is resident); every recall, store and read is the brain's.

DECLARED HOST SCAFFOLDS (ON only; each is named, none decides a recall):
  * `fact_sources`: the conversation that taught each runtime-acquired fact (design section 3 "host bookkeeping");
    a recall of a fact another conversation taught is framed with the anonymous "Someone told me: " lead (owner
    ruling section 9, interim policy B). Replacement: the M3 spiking source-monitor organ.
  * the tonic mood baseline update (a slow EMA over every conversation's affective appraisals) and the seeding of a
    new conversation's phasic mood from it (design section 1 "Affect", owner call 2).
  * `DeferredMouth`: with a generative (Qwen) renderer the brain decides on the template articulation inside the
    turn lock, and the generative mouth runs after the lock (design section 2); its re-parse VERIFY re-takes the lock.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import MutableMapping

ONE_BRAIN_ENV = "BRAIN_ONE_BRAIN"
FREEZE_ENV = "BRAIN_ONE_BRAIN_FREEZE_LEARNING"
# The production-integration anchor: M2 lands DEFAULT OFF (design section 5: default-off, then flip after the
# six-seed validation). Flipping this to True turns the one brain on by default.
_ONE_BRAIN_DEFAULT_ON = False
BRAIN_KEY_TAG = "__one_brain__"
TONIC_DECAY = 0.9          # the tonic baseline is a SLOW EMA (phasic is _MOOD_EMA_DECAY = 0.4 in server.py)
SOURCE_LEAD = "Someone told me: "
SOURCE_PART_NOTE = "(Someone told me part of that.)"


def _truthy(v) -> bool:
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


def one_brain_on(env=None) -> bool:
    """`BRAIN_ONE_BRAIN` unset/blank -> `_ONE_BRAIN_DEFAULT_ON` (off); 1/true/on/yes -> on; anything else -> off."""
    env = os.environ if env is None else env
    v = env.get(ONE_BRAIN_ENV)
    if v is None or not v.strip():
        return _ONE_BRAIN_DEFAULT_ON
    return _truthy(v)


def freeze_learning_on(env=None) -> bool:
    """`BRAIN_ONE_BRAIN_FREEZE_LEARNING=1` (ON only): the shared store's plasticity gate is shut -- an assertion is
    heard but not written to the composer buffer, and no D5 episode is stored. A lesion arm for the M0 probe."""
    env = os.environ if env is None else env
    return one_brain_on(env) and _truthy(env.get(FREEZE_ENV))


def brain_key(cache_key) -> tuple:
    """The brain-level key for a conversation key (session, brain, renderer): one brain per (brain, renderer)."""
    return (BRAIN_KEY_TAG, cache_key[1], cache_key[2])


def is_brain_key(key) -> bool:
    return isinstance(key, tuple) and len(key) == 3 and key[0] == BRAIN_KEY_TAG


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Context + the registry the server's per-conversation dicts are views of
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class Context:
    """One live conversation. `key` is the conversation id: (session, brain, renderer), the key every per-
    conversation registry in the server and its organs already uses. Never checkpointed; dies with the
    conversation (`ContextRegistry.remove`)."""

    FIELDS = ("chat", "rich", "mood", "worldview", "multiref", "silent_wm", "discourse", "pmem", "selfinit")

    def __init__(self, key):
        self.key = key
        self.conversation_id = key
        self.interlocutor = None     # the typed name (request `interlocutor` / OpenAI `user`); never leaves this object
        for f in self.FIELDS:
            setattr(self, f, None)
        self.chat_state = None       # ON: this conversation's ChatBrain fields while it is not bound to the brain
        self.created_at = time.time()

    def __repr__(self):
        return "Context(%r)" % (self.key,)


class ContextRegistry:
    """conversation key -> Context. Owns the field views; removing a Context drops it from every view."""

    def __init__(self):
        self._ctx = {}
        self._views = []

    def view(self, field) -> "ContextFieldView":
        v = ContextFieldView(self, field)
        self._views.append(v)
        return v

    def get(self, key):
        return self._ctx.get(key)

    def ensure(self, key) -> Context:
        ctx = self._ctx.get(key)
        if ctx is None:
            ctx = self._ctx[key] = Context(key)
        return ctx

    def remove(self, key):
        for v in self._views:
            v._present.pop(key, None)
        return self._ctx.pop(key, None)

    def __contains__(self, key):
        return key in self._ctx

    def __len__(self):
        return len(self._ctx)

    def keys(self):
        return list(self._ctx)

    def items(self):
        return list(self._ctx.items())


class ContextFieldView(MutableMapping):
    """A dict-shaped view of ONE Context field across conversations, so the server's `_SESSION_MOOD[key]`,
    `.get`, `.pop`, `.setdefault`, iteration and `in` keep their exact dict semantics while the value lives on the
    Context. Presence and order are tracked like a dict's (insertion order; a re-set keeps its place; a pop then a
    set moves the key to the end), so code that iterates a registry (the idle tick) sees the same order as before."""

    def __init__(self, registry: ContextRegistry, field: str):
        self._reg = registry
        self._field = field
        self._present = {}           # key -> None, in insertion order (an ordered set)

    def __getitem__(self, key):
        if key not in self._present:
            raise KeyError(key)
        return getattr(self._reg.get(key), self._field)

    def __setitem__(self, key, value):
        setattr(self._reg.ensure(key), self._field, value)
        self._present.setdefault(key, None)

    def __delitem__(self, key):
        if key not in self._present:
            raise KeyError(key)
        del self._present[key]
        ctx = self._reg.get(key)
        if ctx is not None:
            setattr(ctx, self._field, None)

    def __iter__(self):
        return iter(list(self._present))

    def __len__(self):
        return len(self._present)

    def __contains__(self, key):
        return key in self._present

    def __repr__(self):
        return "ContextFieldView(%s, %d)" % (self._field, len(self._present))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The conversation-level fields of the shared ChatBrain (ON). Every attribute a turn writes on a ChatBrain is
# classified here; tests/test_one_brain.py scans webapp/*.py for `chat.<attr> = ...` writes and fails on any name in
# neither tuple, so a new per-turn field cannot silently become brain-level.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
_ABSENT = object()

# conversation-level (C): saved into the outgoing Context and restored for the incoming one at every bind
CHAT_CONTEXT_ATTRS = (
    "_boundary_seen", "_heard_any_clause",                 # D3 discourse boundary state (ChatBrain)
    "last_word_role_meta", "_mood_tone_level", "_multiref_referent_override",
    "_affect_drives_workspace",                            # #84 affect->tone EMA (the phasic felt state of THIS talk)
    "_da_drives_workspace",                                # DA-mode engagement EMA of THIS conversation
    "_gnw_swap_workspace",                                 # the held topic (swap drives)
    "_last_affect_drives", "_last_da_drives", "_last_cg_drives", "_last_swap_drives", "_last_gnw_swap",
    "_last_gnw_bus", "_last_gnw_delib", "_last_gnw_multistep", "_last_gnw_stop", "_last_two_organ",
    "_last_three_organ", "_last_category_inherit", "_last_open_ended_gated", "_ci_last_observe", "_ci_last_extract",
    "_oeg_calls", "_oeg_rng_state", "_oeg_seed",          # open-ended gated turn record + its RNG (default-off route)
)
# brain-level (B): one copy shared by every conversation (installed once, or learned)
CHAT_BRAIN_ATTRS = (
    "gate", "gate_extract", "renderer", "raw_mode", "_brain_chat_source", "_substrate_recall",
    "_gnw_bus_installed", "_gnw_delib_installed", "_gnw_multistep_installed", "_gnw_orig_gate",
    "_gnw_predelib_gate", "_gnw_premultistep_gate", "_two_organ_installed", "_two_organ_orig_gate",
    "_three_organ_installed", "_three_organ_orig_gate", "_category_inherit_installed", "_category_inherit_organ",
    "_da_tag_capture",                                     # synaptic tag / PRP ledger of the shared store
    "_da_encoding_mu",                                     # running mean of the shared store's encoding strength
)
# conversation-level fields of the agent (MultiTurnAgent): the discourse WM loop, the biased-competition buffer and
# the referent registry. A fresh conversation gets None (the agent builds its own lazily, `_ensure_wm`).
AGENT_CONTEXT_ATTRS = ("wm", "bcw", "_referent_history")


def _fresh_value(attr, value):
    """The value a NEW conversation starts from, given the pristine (just-built) value."""
    if attr in ("wm", "bcw"):
        return None
    if value is _ABSENT or value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, dict, set, tuple)):
        return copy.copy(value)
    return _ABSENT                                         # any other object: absent, so its owner builds a fresh one


def capture(chat) -> dict:
    """The conversation-level state currently resident on the shared chat."""
    st = {"chat": {a: getattr(chat, a, _ABSENT) for a in CHAT_CONTEXT_ATTRS}}
    r = getattr(chat, "renderer", None)
    st["manner"] = getattr(r, "manner", _ABSENT) if r is not None else _ABSENT
    agent = getattr(chat, "agent", None)
    st["agent"] = {a: getattr(agent, a, _ABSENT) for a in AGENT_CONTEXT_ATTRS} if agent is not None else {}
    reg = getattr(agent, "_event_register", None) if agent is not None else None
    st["register"] = ((list(reg.slots), bool(getattr(reg, "_boundary", False)))
                      if reg is not None and hasattr(reg, "slots") else None)
    return st


def _put(obj, attr, value):
    if value is _ABSENT:
        if attr in getattr(obj, "__dict__", {}):
            delattr(obj, attr)
    else:
        setattr(obj, attr, value)


def restore(chat, st) -> None:
    for a, v in st["chat"].items():
        _put(chat, a, v)
    r = getattr(chat, "renderer", None)
    if r is not None and st.get("manner", _ABSENT) is not _ABSENT:
        r.manner = st["manner"]
    agent = getattr(chat, "agent", None)
    if agent is not None:
        for a, v in st["agent"].items():
            _put(agent, a, v)
        reg = getattr(agent, "_event_register", None)
        if reg is not None and st.get("register") is not None:
            reg.slots = list(st["register"][0])
            reg._boundary = st["register"][1]


def fresh_from(pristine) -> dict:
    return {"chat": {a: _fresh_value(a, v) for a, v in pristine["chat"].items()},
            "manner": pristine.get("manner", _ABSENT),
            "agent": {a: _fresh_value(a, v) for a, v in pristine["agent"].items()},
            "register": (None if pristine.get("register") is None
                         else (list(pristine["register"][0]), pristine["register"][1]))}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The deferred mouth (ON, generative renderer only)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def is_generative(renderer) -> bool:
    """A renderer whose surface comes from a generative model (the Qwen faculty), not a template."""
    fac = getattr(renderer, "_fac", None)
    return fac is not None and hasattr(fac, "_generate") and hasattr(fac, "CONSTRAIN_TEMPLATE")


_MOUTH_LOCK = threading.Lock()     # one generative mouth at a time (the warm model is shared); never the turn lock


class DeferredMouth:
    """Stands in for a generative renderer on the shared ChatBrain at ON. Inside the turn lock every render call
    returns the template articulation (the brain decides, and VERIFYs, on it) and is recorded; after the lock,
    `articulate` runs the generative model on each recorded call whose template surface reached the answer, re-takes
    the lock only to VERIFY each prose surface (the on-substrate re-parse, `ChatBrain._verify`), and substitutes the
    verified prose. A prose surface that fails VERIFY leaves the template surface in place (never unverified prose).
    No batch methods: the rich composer then renders per fact through `render_svo`."""

    def __init__(self, base):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self.base = base
        self.name = getattr(base, "name", "renderer")
        self._tmpl = TemplateStubFaculty()
        self.calls = []

    def _record(self, method, a, v, p, manner):
        surface, asserted = self._tmpl.render_svo(a, v, p)
        self.calls.append({"method": method, "svo": [a, v, p], "manner": manner, "template": surface})
        return surface, asserted

    def render_svo(self, a, v, p):
        return self._record("render_svo", a, v, p, "")

    def render_svo_manner(self, a, v, p, manner):
        """The mood-conditioned call (MoodConditionedRenderer routes a non-empty manner here when its base is a
        DeferredMouth), so the manner reaches the generative prompt after the lock."""
        return self._record("render_svo", a, v, p, manner or "")

    def render_svo_regen(self, a, v, p):
        return self._record("render_svo_regen", a, v, p, "")

    def take_calls(self) -> list:
        out, self.calls = self.calls, []
        return out

    def generate(self, call) -> str | None:
        """The generative surface for one recorded call (no lock held but the mouth lock)."""
        a, v, p = call["svo"]
        with _MOUTH_LOCK:
            if call.get("manner") and is_generative(self.base):
                fac = self.base._fac
                prompt = fac.CONSTRAIN_TEMPLATE.format(a=a, v=v, p=p) + " " + call["manner"]
                surface = fac._generate(prompt)[0]
            elif call["method"] == "render_svo_regen" and hasattr(self.base, "render_svo_regen"):
                surface = self.base.render_svo_regen(a, v, p)[0]
            else:
                surface = self.base.render_svo(a, v, p)[0]
        return surface if isinstance(surface, str) else None


def articulate(answer, calls, mouth, verify, relock):
    """Run the deferred mouth after the turn lock. `verify(surface, svo) -> bool` is the brain's re-parse, run inside
    `relock()` (a context manager that takes the turn lock). Returns (answer, record)."""
    rec = {"n_calls": len(calls), "n_spoken": 0, "n_verified": 0, "n_kept_template": 0, "seconds": 0.0}
    if not isinstance(answer, str) or not calls:
        return answer, rec
    t0 = time.perf_counter()
    todo = []
    for c in calls:
        t = c.get("template")
        if t and t in answer:
            rec["n_spoken"] += 1
            todo.append((c, mouth.generate(c)))
    if todo:
        with relock():
            for c, prose in todo:
                if prose and prose != c["template"] and verify(prose, c["svo"]) and c["template"] in answer:
                    answer = answer.replace(c["template"], prose, 1)
                    rec["n_verified"] += 1
                else:
                    rec["n_kept_template"] += 1
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    return answer, rec


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The Brain
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class Brain:
    """The one brain of this process. At OFF it holds nothing (each conversation's Context owns its own ChatBrain,
    as before M2). At ON it owns the brain-level rows: one shared ChatBrain per (brain, renderer) and what is listed
    in the module docstring. Every method that touches a shared chat is called with the turn lock held."""

    def __init__(self):
        from webapp import brain_turn_lock as _L
        self.turn_lock = _L._TURN_LOCK          # the M0 FIFO lock (webapp/brain_turn_lock.py) is the Brain's lock
        self.chats = {}                          # brain key -> the shared ChatBrain
        self.sources = {}                        # brain key -> its build source string
        self.n_build_facts = {}                  # brain key -> composer kb length right after the build
        self.pristine = {}                       # brain key -> conversation-level state of the just-built chat
        self.bound = {}                          # brain key -> the Context whose state is resident on the chat
        self.fact_sources = {}                   # brain key -> {(a, v, p): conversation key that taught it}
        self.tonic = {"valence": 0.0, "arousal": 0.0}
        self.n_tonic_updates = 0

    # -- the shared chat --
    def chat(self, bkey):
        return self.chats.get(bkey)

    def adopt(self, bkey, chat, source):
        """Register a freshly built ChatBrain as the shared chat for `bkey`: record its pristine conversation-level
        state, its build fact count, install the plasticity gate (freeze lesion) and, for a generative renderer, the
        deferred mouth."""
        if is_generative(getattr(chat, "renderer", None)):
            chat.renderer = DeferredMouth(chat.renderer)
        _install_freeze_gate(chat)
        self.pristine[bkey] = capture(chat)
        self.n_build_facts[bkey] = len(_buffer_kb(chat) or ())
        self.chats[bkey] = chat
        self.sources[bkey] = source
        self.bound[bkey] = None
        self.fact_sources.setdefault(bkey, {})
        return chat

    def bind(self, bkey, ctx):
        """Make `ctx`'s conversation-level state the one resident on the shared chat (turn lock held)."""
        chat = self.chats[bkey]
        cur = self.bound.get(bkey)
        if cur is ctx:
            return chat
        if cur is not None:
            cur.chat_state = capture(chat)
        restore(chat, ctx.chat_state if ctx.chat_state is not None else fresh_from(self.pristine[bkey]))
        ctx.chat_state = None
        self.bound[bkey] = ctx
        return chat

    def release(self, ctx) -> None:
        """A conversation ends (reset / eviction): if it is the bound one, the next bind starts clean from it."""
        for bkey, cur in list(self.bound.items()):
            if cur is ctx:
                self.bound[bkey] = _Evicted()

    # -- tonic mood (brain-level) --
    def tonic_seed(self) -> dict:
        return {"valence": float(self.tonic["valence"]), "arousal": float(self.tonic["arousal"])}

    def update_tonic(self, appr) -> None:
        d = TONIC_DECAY
        self.tonic = {"valence": d * self.tonic["valence"] + (1.0 - d) * float(appr["valence"]),
                      "arousal": d * self.tonic["arousal"] + (1.0 - d) * float(appr["arousal"])}
        self.n_tonic_updates += 1

    # -- host source tags (declared scaffold) --
    def note_taught(self, bkey, before, after, ckey) -> list:
        """Tag every fact that entered the shared store during this conversation's turn with that conversation."""
        new = [f for f in after if f not in before]
        tags = self.fact_sources.setdefault(bkey, {})
        for f in new:
            tags[tuple(f)] = ckey
        return new

    def taught_elsewhere(self, bkey, svo, ckey) -> bool:
        if not isinstance(svo, (list, tuple)) or len(svo) < 3:
            return False
        src = self.fact_sources.get(bkey, {}).get(tuple(str(x) for x in svo[:3]))
        return src is not None and src != ckey

    # -- zero (experimenter instrument) --
    def zero_learned(self, bkey) -> dict:
        """Drop every composer block learned since the build (the build KB and the LTM prior are untouched)."""
        chat = self.chats.get(bkey)
        if chat is None:
            return {"brain_key": list(bkey), "zeroed": 0, "reason": "no shared chat"}
        n0 = self.n_build_facts.get(bkey, 0)
        n = _truncate_buffer(chat, n0)
        try:
            chat._refresh_facts()
        except Exception:
            pass
        self.fact_sources[bkey] = {}
        return {"brain_key": list(bkey), "zeroed": n, "n_build_facts": n0}


class TickAdapter:
    """What the idle tick (webapp/continuous_engine.tick_idle_sessions) needs from the Brain at ON: which brain a
    conversation belongs to, the brains, and each brain's shared chat and D5 organ (never built here)."""

    def __init__(self, brain: Brain):
        self.brain = brain

    def brain_key_of(self, key):
        if isinstance(key, tuple) and len(key) == 3 and not is_brain_key(key):
            return brain_key(key)
        return None

    def brain_keys(self):
        return list(self.brain.chats)

    def chat(self, bkey):
        return self.brain.chat(bkey)

    def episodic(self, bkey):
        try:
            import research.runners.d5_episodic_production_organ as _EP
            return _EP._ORGANS.get(bkey)
        except Exception:
            return None


TICK_ADAPTER = None   # set by webapp/server.py to TickAdapter(_ONE_BRAIN); read by the idle tick at ON


class _Evicted:
    """Placeholder for a bound conversation that was evicted: never `is` any live Context, so the next bind saves the
    chat's state into this throwaway object and restores the incoming conversation's."""
    chat_state = None


def _buffer(chat):
    """The composer that holds the learned facts (the TieredFactStore buffer, or the composer itself)."""
    inner = getattr(chat, "inner", None)
    comp = getattr(inner, "composer", None)
    return getattr(comp, "buffer", comp)


def _buffer_kb(chat):
    return getattr(_buffer(chat), "kb", None)


def learned_triples(chat) -> list:
    """The (agent, action, patient) string triples in the shared composer buffer right now (read from its kb, so a
    reconsolidation rewrite in place shows as a new triple)."""
    out = []
    for item in (_buffer_kb(chat) or ()):
        f = item[0] if isinstance(item, tuple) else item
        if isinstance(f, dict) and all(isinstance(f.get(r), str) for r in ("agent", "action", "patient")):
            out.append((f["agent"], f["action"], f["patient"]))
    return out


def _truncate_buffer(chat, n0) -> int:
    """Remove composer blocks after the first `n0` (the build KB): the onebrain composer's kb + store synapses, with
    every cache that indexes them marked stale. Returns the number removed. Raises on a composer it cannot edit."""
    comp = _buffer(chat)
    kb = getattr(comp, "kb", None)
    if kb is None:
        raise NotImplementedError("composer %s has no kb" % type(comp).__name__)
    n = max(0, len(kb) - int(n0))
    if n == 0:
        return 0
    del kb[int(n0):]
    if hasattr(comp, "store_conns") and hasattr(comp, "D"):
        del comp.store_conns[int(n0) * int(comp.D):]
        for flag in ("_store_dirty", "_persistent_dirty"):
            if hasattr(comp, flag):
                setattr(comp, flag, True)
        if getattr(comp, "integrated_loop", False):
            comp._seq_dirty = True
            if getattr(comp, "_fused", False):
                comp._fused_dirty = True
        cache = getattr(comp, "_csr_cache", None)
        if cache is not None:
            cache.clear()
        if hasattr(comp, "_dg_index"):
            comp._dg_index = None
            comp._dg_built_V = -1
        if hasattr(comp, "_fact_shard"):
            comp._fact_shard = None
    else:
        raise NotImplementedError("composer %s: no block store to truncate" % type(comp).__name__)
    return n


def _install_freeze_gate(chat) -> None:
    """The shared store's plasticity gate (ON): with `BRAIN_ONE_BRAIN_FREEZE_LEARNING=1` the agent's `hear` comprehends
    nothing into the store (the assertion is heard by the turn; no block is written). Read per call, so the flag can
    move between turns. Installed on the shared inner agent only (an instance attribute; the class is untouched)."""
    inner = getattr(chat, "inner", None)
    if inner is None or getattr(inner, "_one_brain_freeze_gate", False):
        return
    orig = inner.hear

    def hear(sentence, voice="active", polarity=None):
        if freeze_learning_on():
            return None
        return orig(sentence, voice, polarity=polarity)

    inner.hear = hear
    inner._one_brain_freeze_gate = True
