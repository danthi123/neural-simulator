"""INSTRUMENT — the AI-teacher isolation guard: every brain STORE-WRITING entry point, wrapped.

The AI teacher (research/runners/ai_teacher.py) may reach the brain only through the chat handler. This module wraps
every function that writes a brain store (fact blocks, the LTM, the D6 Hebbian encode, the acquisition hook, the
agent's `hear*` family) so that a call can be attributed to WHO reached it:
  * "brain"   -- the call came through a registered CHANNEL boundary (by default webapp/server.py `brain_chat` /
                 `brain_reply`): the brain's own pipeline decided to write, having heard a sentence. Legitimate.
  * "teacher" -- a frame of the teacher module is on the stack and NO channel boundary sits between it and the
                 write: the teacher reached a store directly. A VIOLATION.
  * "other"   -- neither (brain build, experimenter lesions). Recorded, not a violation.
Two modes: `mode="raise"` makes EVERY wrapped call raise `StoreWriteForbidden` (the unit test's strict form: a
teacher session run against a text-only fake brain must complete without touching any of them); `mode="attribute"`
lets calls through, counts them by attribution, and raises only on a "teacher" violation (the real-session form).

It is an instrument, not part of the teacher and not part of the brain. Honest limits: a module-level function that
some caller imported by name (`from x import f`) BEFORE `install()` keeps its unwrapped reference; the teacher is
separately required (by an AST test) to import no brain module at all, so it holds no such reference.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
from collections import Counter

TEACHER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_teacher.py")

# (module, [method or function names]) -- every store-writing entry point in the brain code paths the chat handler can
# build. Classes are discovered inside each module (any class DEFINED there whose __dict__ has the name).
ENTRY_POINTS = [
    ("research.runners.one_brain_composer", ["store", "store_fact", "_store_composite", "_write_block",
                                             "update_on_mismatch", "hear", "apply_homeostatic_scaling"]),
    ("research.runners.rf_phasor_composer", ["store", "update_on_mismatch", "hear"]),
    ("research.runners.routed_composer", ["store", "update_on_mismatch"]),
    ("research.runners.sharded_phasor_store", ["store"]),
    ("research.runners.tiered_fact_store", ["store", "encode_fast", "build_ltm_from_facts"]),
    ("research.runners.slotbinder_composer", ["store"]),
    ("research.runners.argstructure_composer", ["store_fact"]),
    ("research.runners.core_sim_composition", ["store"]),
    ("research.runners.unified_brain_bridge", ["store"]),
    ("research.runners.cross_session_persistence", ["store", "save_session_learning"]),
    ("research.runners.brain_conversational_agent", ["hear", "hear_multicue", "hear_case", "hear_clause_fact",
                                                     "hear_attributed", "hear_multiframe"]),
    ("research.runners.brain_chat_tui", ["_maybe_acquire"]),
    ("research.runners.d6_hebbian_store", ["hebbian_encode"]),
    ("sim.bridge_memory", ["store"]),
]

DEFAULT_BOUNDARIES = {(os.path.join("webapp", "server.py"), "brain_chat"),
                      (os.path.join("webapp", "server.py"), "brain_reply")}


class StoreWriteForbidden(RuntimeError):
    pass


class TeacherIsolationGuard:
    def __init__(self, mode="attribute", boundaries=None, entry_points=None):
        assert mode in ("raise", "attribute")
        self.mode = mode
        self.boundaries = set(DEFAULT_BOUNDARIES if boundaries is None else boundaries)
        self.entry_points = list(ENTRY_POINTS if entry_points is None else entry_points)
        self.counts = Counter()          # (attribution, qualified name) -> calls
        self.violations = []             # [{"name", "stack"}]
        self.patched = []                # [(owner, attr, original)]
        self.missing = []                # modules that failed to import (reported, not silently skipped)
        self._local = threading.local()

    # ── attribution ──
    def _attribute(self, start_frame):
        f = start_frame
        while f is not None:
            fn = f.f_code.co_filename
            for suffix, name in self.boundaries:
                if f.f_code.co_name == name and fn.endswith(suffix):
                    return "brain"
            if os.path.abspath(fn) == TEACHER_FILE:
                return "teacher"
            f = f.f_back
        return "other"

    def _wrap(self, qualname, orig):
        guard = self

        def wrapped(*a, **k):
            if getattr(guard._local, "inside", False):      # a wrapped writer calling another: count once
                return orig(*a, **k)
            who = guard._attribute(sys._getframe(1))
            guard.counts[(who, qualname)] += 1
            if guard.mode == "raise":
                raise StoreWriteForbidden("%s called during a teacher session (attributed: %s)" % (qualname, who))
            if who == "teacher":
                guard.violations.append({"name": qualname,
                                         "stack": [(fr.filename, fr.function) for fr in inspect.stack()[1:8]]})
                raise StoreWriteForbidden("the AI teacher reached %s without the chat channel" % qualname)
            guard._local.inside = True
            try:
                return orig(*a, **k)
            finally:
                guard._local.inside = False
        wrapped.__wrapped__ = orig
        wrapped.__name__ = getattr(orig, "__name__", qualname)
        wrapped._ai_teacher_guard = True
        return wrapped

    # ── install / uninstall ──
    def install(self):
        for modname, names in self.entry_points:
            try:
                mod = importlib.import_module(modname)
            except Exception as e:  # reported: an entry point that cannot be imported cannot be guarded
                self.missing.append({"module": modname, "error": "%s: %s" % (type(e).__name__, e)})
                continue
            for name in names:
                obj = mod.__dict__.get(name)
                if callable(obj) and not inspect.isclass(obj) and not getattr(obj, "_ai_teacher_guard", False):
                    self.patched.append((mod, name, obj))
                    setattr(mod, name, self._wrap("%s.%s" % (modname, name), obj))
            for cname, cls in list(mod.__dict__.items()):
                if not inspect.isclass(cls) or cls.__module__ != mod.__name__:
                    continue
                for name in names:
                    if name in cls.__dict__:
                        orig = cls.__dict__[name]
                        if isinstance(orig, (staticmethod, classmethod)) or getattr(orig, "_ai_teacher_guard", False):
                            continue
                        self.patched.append((cls, name, orig))
                        setattr(cls, name, self._wrap("%s.%s.%s" % (modname, cname, name), orig))
        self.n_patched_at_install = len(self.patched)
        self.patched_names = sorted({"%s.%s" % (getattr(o, "__name__", str(o)), n) for o, n, _ in self.patched})
        return self

    def uninstall(self):
        for owner, name, orig in reversed(self.patched):
            setattr(owner, name, orig)
        self.patched = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    def report(self):
        by = Counter()
        for (who, _q), n in self.counts.items():
            by[who] += n
        return {"mode": self.mode, "n_patched": getattr(self, "n_patched_at_install", len(self.patched)),
                "patched_names": list(getattr(self, "patched_names", [])), "missing_modules": list(self.missing),
                "calls_by_attribution": dict(by),
                "calls": {"%s|%s" % k: v for k, v in sorted(self.counts.items())},
                "violations": list(self.violations)}
