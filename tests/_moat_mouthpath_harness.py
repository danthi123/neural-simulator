"""Flag-OFF golden harness for the build-6 mouth-path edits of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`).

Build 6 (docs/drafts/moat-allowlist/SPEC.md) makes the flag own the whole mouth path: `ChatBrain._verify`, the spiking recall surface,
the hypothesis path, Touchpoint A and the composer's joined reply. Every edit is behind the flag. This harness drives those paths with
the flag unset and records (output, call sequence); `capture()` was run at the build-5 verdict tip fde5e739e BEFORE any build-6 edit
and written to tests/fixtures/moat_mouthpath_flagoff_golden.json. tests/test_moat_allowlist_build6.py re-runs `run_all()` with the flag
unset / '0' / 'off' / garbage and requires equality with that golden.

Two halves:
  - COMPOSER: `RichAnswerComposer.render_paragraph`, `_render_one_verified` and `_render_paragraph_batched` on a scripted fake chat and
    renderer (stub-like OK, extra-clause, role-swapped, unterminated, spiking hit / miss, Touchpoint-A hit / miss via a fake
    `webapp.wkv_mouth_generator`, raw mode, no renderer, 1..4 facts, sequential and batched);
  - CHATBRAIN: the real `ChatBrain` methods (`render`, `render_hypothesis_verified`, `spiking_recall_surface`, `_apply_mouth_mood_tone`,
    `_verify`) on a ChatBrain built without its heavy __init__ (real BridgeParser seed 42, the tiny-demo facts, the real StubRenderer,
    the real spiking producers), mood levels -3..3 with BRAIN_SPIKING_MOUTH_MOOD on and off.

Run (capture only at the base commit): CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m tests._moat_mouthpath_harness <out.json>
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import types

os.environ.setdefault("SIM_BACKEND", "numpy")

ENV_KEYS = ("BRAIN_MOAT_ALLOWLIST", "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT", "BRAIN_RICH_BATCH_RENDER",
            "BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "BRAIN_SPIKING_MOUTH", "BRAIN_SPIKING_MOUTH_RECALL", "BRAIN_SPIKING_MOUTH_MOOD")

TINY = [["brain", "use", "spikes"], ["brain", "learn", "words"], ["brain", "store", "memory"], ["dog", "chase", "cat"],
        ["cat", "eat", "fish"]]


@contextlib.contextmanager
def env(**kw):
    """Set exactly these env keys (None = unset) for the block; every other ENV_KEYS key is unset; restore afterwards."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in kw.items():
            if v is not None:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def fake_touchpoint_a(table):
    """A fake `webapp.wkv_mouth_generator` whose `render_fact_sentence([svo])` returns table.get(tuple(svo)) (None = a miss)."""
    name = "webapp.wkv_mouth_generator"
    saved = sys.modules.get(name)
    mod = types.ModuleType(name)
    mod._RNG = types.SimpleNamespace(run=lambda seed, fn: fn())
    mod.render_fact_sentence = lambda svos, seed=42: table.get(tuple(svos[0]))
    sys.modules[name] = mod
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


# ============================================================================================================== composer half
def surface_of(kind, svo):
    a, v, p = svo
    return {"good": "The %s %ss %s." % (a, v, p),
            "extra": "The %s %ss %s and is famous for its vodka." % (a, v, p),
            "swap": "The %s %ss %s." % (p, v, a),
            "unterm": "The %s %ss %s" % (a, v, p),
            "bang": "The %s %ss %s!" % (a, v, p),
            "none": None}[kind]


class FakeChat:
    """What the composer touches. `_verify` / `_verify_claim_set` accept a surface iff it is the 'good' or 'unterm' surface of the
    svo (claim set: of SOME gathered svo) -- a scripted stand-in, recorded call by call."""

    def __init__(self, rec, renderer=None, spiking=None, raw_mode=False):
        self.rec, self.renderer, self.raw_mode = rec, renderer, raw_mode
        self._spiking = spiking or {}
        self.inner = types.SimpleNamespace(seed=42)

    def spiking_recall_surface(self, a, v, p):
        self.rec.append(["spiking_recall_surface", a, v, p])
        return self._spiking.get((a, v, p))

    def _ok(self, surface, svo):
        return isinstance(surface, str) and surface.rstrip(".") == surface_of("good", svo).rstrip(".")

    def _verify(self, surface, asserted, svo):
        self.rec.append(["_verify", surface, list(svo)])
        return self._ok(surface, svo)

    def _verify_claim_set(self, surface, gated):
        self.rec.append(["_verify_claim_set", surface, [list(g) for g in gated]])
        return any(self._ok(surface, g) for g in gated), {"reject_reason": None}

    def _raw(self, svo):
        return " ".join(str(x) for x in svo)


def fake_renderer(rec, first, regen=None, batch=False, regen_batch=False):
    """first / regen: svo-tuple -> kind. Only the methods asked for exist (the composer probes with hasattr)."""
    r = types.SimpleNamespace()

    def render_svo(a, v, p):
        rec.append(["render_svo", a, v, p])
        return surface_of(first.get((a, v, p), "good"), (a, v, p)), None
    r.render_svo = render_svo
    if regen is not None:
        def render_svo_regen(a, v, p):
            rec.append(["render_svo_regen", a, v, p])
            return surface_of(regen.get((a, v, p), "good"), (a, v, p)), None
        r.render_svo_regen = render_svo_regen
    if batch:
        def render_svo_batch(svos):
            rec.append(["render_svo_batch", [list(s) for s in svos]])
            return [(surface_of(first.get(tuple(s), "good"), s), None) for s in svos], 0.0
        r.render_svo_batch = render_svo_batch
    if regen_batch:
        def render_svo_regen_batch(svos):
            rec.append(["render_svo_regen_batch", [list(s) for s in svos]])
            return [(surface_of((regen or {}).get(tuple(s), "good"), s), None) for s in svos], 0.0
        r.render_svo_regen_batch = render_svo_regen_batch
    return r


F1, F2, F3, F4 = ("dog", "chase", "cat"), ("cat", "eat", "fish"), ("brain", "use", "spikes"), ("brain", "learn", "words")

COMPOSER_SCENARIOS = {
    "one_good": dict(facts=[F1], first={}),
    "one_extra_regen_good": dict(facts=[F1], first={F1: "extra"}, regen={}),
    "one_extra_regen_extra": dict(facts=[F1], first={F1: "extra"}, regen={F1: "extra"}),
    "one_extra_no_regen": dict(facts=[F1], first={F1: "extra"}),
    "one_swap": dict(facts=[F1], first={F1: "swap"}, regen={F1: "swap"}),
    "one_unterm": dict(facts=[F1], first={F1: "unterm"}),
    "one_bang": dict(facts=[F1], first={F1: "bang"}),
    "two_good": dict(facts=[F1, F2], first={}),
    "two_unterm": dict(facts=[F1, F2], first={F1: "unterm", F2: "unterm"}),
    "three_mixed": dict(facts=[F1, F2, F3], first={F2: "extra"}, regen={F2: "swap"}),
    "four_good": dict(facts=[F1, F2, F3, F4], first={}),
    "spiking_hit": dict(facts=[F1, F2], first={}, spiking={F1: "the dog chases the cat", F2: "the cat eats the fish"}),
    "spiking_one_miss": dict(facts=[F1, F3], first={F3: "extra"}, regen={}, spiking={F1: "the dog chases the cat"}),
    "tpa_hit": dict(facts=[F1], first={F1: "extra"}, tpa={F1: "the dog is located in the cat."}),
    "tpa_miss": dict(facts=[F1, F2], first={}, tpa={}),
    "raw_mode": dict(facts=[F1, F2], first={}, raw=True),
    "no_renderer": dict(facts=[F1, F2], first={}, no_renderer=True),
    "batched_good": dict(facts=[F1, F2, F3], first={}, batch=True, regen_batch=True, regen={}),
    "batched_regen": dict(facts=[F1, F2, F3], first={F2: "extra", F3: "swap"}, regen={F3: "extra"}, batch=True, regen_batch=True),
    "batched_no_regen_batch": dict(facts=[F1, F2], first={F2: "extra"}, regen={}, batch=True),
    "batched_spiking": dict(facts=[F1, F2, F3], first={F3: "extra"}, regen={F3: "unterm"}, batch=True, regen_batch=True,
                            spiking={F1: "the dog chases the cat"}),
}


def run_composer_scenario(name):
    from research.runners.rich_answer_composer import RichAnswerComposer
    sc = COMPOSER_SCENARIOS[name]
    out = {}
    for method in ("render_paragraph", "_render_one_verified"):
        rec = []
        ren = None if sc.get("no_renderer") else fake_renderer(rec, sc["first"], sc.get("regen"), sc.get("batch", False),
                                                              sc.get("regen_batch", False))
        chat = FakeChat(rec, ren, sc.get("spiking"), raw_mode=sc.get("raw", False))
        comp = RichAnswerComposer.__new__(RichAnswerComposer)
        comp.chat = chat
        facts = [list(f) for f in sc["facts"]]
        extra = {"BRAIN_RICH_BATCH_RENDER": "1"} if sc.get("batch") else {}
        if sc.get("tpa") is not None:
            extra["BRAIN_TOUCHPOINT_A_FACT_CLAUSE"] = "1"
        cur = {k: os.environ.get(k) for k in ("BRAIN_MOAT_ALLOWLIST",)}
        with env(**cur, **extra), fake_touchpoint_a(sc.get("tpa") or {}):
            if method == "render_paragraph":
                res = list(comp.render_paragraph(facts))
            else:
                res = [list(comp._render_one_verified(f, gated=facts)) for f in facts]
        out[method] = {"result": res, "calls": rec}
    return out


# ============================================================================================================== ChatBrain half
def tiny_chat(renderer="stub"):
    from research.runners._grounded_lang_integration_derisk import _build_inflection_map
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer
    from research.runners.brain_conversational_agent import BridgeParser
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=42)
    chat.stored_facts = [tuple(f) for f in TINY]
    chat.agents_set = {a for a, _, _ in chat.stored_facts}
    chat.actions_set = {v for _, v, _ in chat.stored_facts}
    chat.patients_set = {p for _, _, p in chat.stored_facts}
    chat.inflect = _build_inflection_map(sorted(chat.actions_set))
    chat.raw_mode = False
    chat.renderer = StubRenderer() if renderer == "stub" else renderer
    return chat


class ProseRenderer:
    """A Qwen-like mouth (asserted None: the prose is re-parsed) returning scripted prose per svo."""

    name = "scripted-prose"

    def __init__(self, table, regen=None):
        self.table, self.regen = table, regen

    def render_svo(self, a, v, p):
        return self.table.get((a, v, p), "The %s %ss the %s." % (a, v, p)), None


class ProseRegenRenderer(ProseRenderer):
    def render_svo_regen(self, a, v, p):
        return (self.regen or {}).get((a, v, p), "The %s %ss the %s." % (a, v, p)), None


_SHARED = {}


def _shared_chat(kind):
    """One ChatBrain per renderer kind for the whole run (the spiking producers are built once per chat, ~0.4 s each)."""
    if kind not in _SHARED:
        if kind == "stub":
            _SHARED[kind] = tiny_chat("stub")
        elif kind == "prose":
            _SHARED[kind] = tiny_chat(ProseRegenRenderer(
                {("dog", "chase", "cat"): "The dog chases the cat and is famous for its vodka.",
                 ("cat", "eat", "fish"): "The fish eats the cat.",
                 ("brain", "use", "spikes"): "The brain uses spikes."},
                regen={("dog", "chase", "cat"): "The dog chases the cat.", ("cat", "eat", "fish"): "The cat eats the fish!"}))
        elif kind == "prose_noregen":
            _SHARED[kind] = tiny_chat(ProseRenderer({("dog", "chase", "cat"): "The dog chases the cat and the bird."}))
    return _SHARED[kind]


def run_chatbrain():
    out = {}
    for kind in ("stub", "prose", "prose_noregen"):
        chat = _shared_chat(kind)
        for recall in (None, "0"):
            for svo in TINY + [["dog", "eat", "fish"], ["big_dog", "chase", "cat"]]:
                with env(BRAIN_MOAT_ALLOWLIST=os.environ.get("BRAIN_MOAT_ALLOWLIST"), BRAIN_SPIKING_MOUTH_RECALL=recall):
                    chat._mood_tone_level = 0
                    out["render|%s|recall=%s|%s" % (kind, recall, "/".join(svo))] = chat.render(list(svo))
        for sm in (None, "0"):
            for hyp in (["dog", "eat", "fish"], ["cat", "chase", "bird"], ["big_dog", "eat", "fish"], ["dog", "chase", "cat"]):
                with env(BRAIN_MOAT_ALLOWLIST=os.environ.get("BRAIN_MOAT_ALLOWLIST"), BRAIN_SPIKING_MOUTH=sm):
                    out["hyp|%s|sm=%s|%s" % (kind, sm, "/".join(hyp))] = list(chat.render_hypothesis_verified(list(hyp)))
    chat = _shared_chat("stub")
    for mood in (None, "0"):
        for level in (-3, -2, -1, 0, 1, 2, 3):
            for svo in (["dog", "chase", "cat"], ["cat", "eat", "fish"], ["brain", "use", "spikes"]):
                with env(BRAIN_MOAT_ALLOWLIST=os.environ.get("BRAIN_MOAT_ALLOWLIST"), BRAIN_SPIKING_MOUTH_MOOD=mood):
                    chat._mood_tone_level = level
                    out["recall|mood=%s|level=%d|%s" % (mood, level, "/".join(svo))] = chat.spiking_recall_surface(*svo)
                    out["tone|mood=%s|level=%d|%s" % (mood, level, "/".join(svo))] = chat._apply_mouth_mood_tone(
                        "the %s %ss the %s" % tuple(svo))
    chat._mood_tone_level = 0
    pairs = []
    for svo in TINY + [["dog", "eat", "fish"]]:
        a, v, p = svo
        for s, asserted in (("The %s %ss %s." % (a, v, p), [a, v, p]), ("The %s %ss %s." % (a, v, p), None),
                            ("The %s %ss %s." % (p, v, a), None), ("The %s %ss %s." % (p, v, a), [a, v, p]),
                            ("The %s %ss %s and is famous for its vodka." % (a, v, p), None),
                            ("the %s %ss the %s" % (a, v, p), None), ("perhaps the %s %ss the %s" % (a, v, p), None),
                            ("The %s %ss %s." % (a, v, p), [p, v, a])):
            pairs.append([s, asserted, svo])
    with env(BRAIN_MOAT_ALLOWLIST=os.environ.get("BRAIN_MOAT_ALLOWLIST")):
        out["_verify"] = [[s, asserted, svo, bool(chat._verify(s, asserted, svo))] for s, asserted, svo in pairs]
    return out


def run_all():
    """Everything, under the CURRENT value of BRAIN_MOAT_ALLOWLIST (every other mouth flag set per case)."""
    return {"composer": {name: run_composer_scenario(name) for name in sorted(COMPOSER_SCENARIOS)},
            "chatbrain": run_chatbrain()}


def capture(path):
    with env():
        doc = run_all()
    doc = json.loads(json.dumps(doc))
    doc["_doc"] = ("flag-OFF golden of the build-6 mouth paths, captured by tests/_moat_mouthpath_harness.py at the build-5 verdict tip "
                   "fde5e739e before any build-6 edit")
    doc["backend"], doc["device"] = os.environ.get("SIM_BACKEND"), "cpu"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    capture(sys.argv[1])
