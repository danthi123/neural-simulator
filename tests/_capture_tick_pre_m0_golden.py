"""Capture the PRE-M0 idle tick's behaviour on a stub scenario, as the golden for
tests/test_tick_pre_m0_equivalence.py (the one-brain M0 re-review, 2026-10-13, item 6).

M0 split `continuous_engine.tick_idle_sessions` into locked cheap steps with the wander outside the turn lock. With
a caller-supplied `now` (the regression battery's single-threaded world steps) nothing can preempt it, so it must
run exactly the pre-M0 sequence: the same faculty calls in the same order, the same inner-life records, the same
wander and D5 budgets and the same IOR adaptation. `scenario(mod)` drives one copy of the module with stub organs
and every gated idle faculty armed; the golden is that output for the pre-M0 source (commit 2e7392f12).

Regenerate (only if the scenario itself changes):
    git show 2e7392f12:webapp/continuous_engine.py > <scratch>/ce_pre_m0.py
    .venv/bin/python -m tests._capture_tick_pre_m0_golden <scratch>/ce_pre_m0.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

PRE_M0_COMMIT = "2e7392f12"
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tick_pre_m0_golden.json")
# every gated idle faculty armed; the wander budget allows an ideation tick (every 3rd wander by default)
ENV = {"BRAIN_CONTINUOUS": "1", "BRAIN_D5_CONSOLIDATE": "1", "BRAIN_SLEEP_REPLAY": "1",
       "BRAIN_CONTINUOUS_AFFECT_RELAX": "1", "BRAIN_CONTINUOUS_DA_RELAX": "1", "BRAIN_WANDER_BUDGET": "3",
       "BRAIN_CONTINUOUS_IDEATE": "1", "BRAIN_WANDER_IOR": "1", "BRAIN_CONTINUOUS_WANDER_MULL": "1"}
UNSET = ("BRAIN_PERSIST_LEARNING", "BRAIN_CONTINUOUS_IDEATE_EVERY", "BRAIN_TURN_LOCK")
NOWS = (1030.0, 1400.0, 1500.0, 1600.0, 1700.0)


def load(path, name):
    """A fresh, isolated copy of a continuous_engine source file (its own module-level state)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scenario(mod) -> dict:
    """Three idle sessions, five battery-clock ticks. Returns the JSON-normalised trace."""
    log = []

    class Aff:
        def read_differential(self, v, lesion=False):
            log.append(("affect", round(v, 6)))
            return {"differential": v * 0.5}

    class SI:
        def __init__(self, k):
            self.k, self.n = k, 0
            self.agents, self.gains_on, self.novelties = ["cat", "dog", "bird"], [1.0, 1.1, 1.2], [0.1, 0.2, 0.3]

        def _ensure_mouth(self):
            log.append(("mouth", self.k[0]))

        def speak(self, lesion=False):
            self.n += 1
            c = self.agents[(self.n + len(self.k[0])) % 3]
            log.append(("speak", self.k[0], c, [round(g, 6) for g in self.gains_on]))
            return {"concept": c}

    sis = {}
    mod.consolidate_used_memory = lambda k, o, **kw: log.append(("d5", k[0])) or {"ok": 1}
    mod.consolidate_sleep_replay = lambda k, o: log.append(("sleep", k[0]))
    mod.consolidate_substrate_homeostasis = lambda k, c, trigger="": log.append(("homeo", k[0], trigger))
    mood = {("s%d" % i, "tiny-demo", "stub"): {"valence": 0.5 - 0.2 * i, "arousal": 0.3} for i in range(3)}
    for k in mood:
        mod._LAST_REQUEST[k] = 1000.0
        mod._WANDER_BUDGET[k] = 3
        mod._D5_BUDGET[k] = 3          # so D5 and the sleep replay share ticks (their order is pinned)
        mod._RECALLED_TOPIC[k] = "cat"
    mod.mark_mull(("s1", "tiny-demo", "stub"), "dog")
    ns = [mod.tick_idle_sessions(mood, lambda: Aff(), now=now,
                                 selfinit_getter=lambda k: sis.setdefault(k, SI(k)),
                                 episodic_getter=lambda k: object(), chat_getter=lambda k: object())
          for now in NOWS]
    out = {"n": ns, "log": log, "inner": {k[0]: v for k, v in mod._INNER_LIFE.items()},
           "budget": {k[0]: v for k, v in mod._WANDER_BUDGET.items()},
           "d5_budget": {k[0]: v for k, v in mod._D5_BUDGET.items()},
           "ior": {k[0]: v for k, v in mod._WANDER_ADAPT.items()},
           "ideate_tick": {k[0]: v for k, v in mod._IDEATE_TICK.items()},
           "mood": {k[0]: v for k, v in mood.items()}}
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def _set_env():
    for k, v in ENV.items():
        os.environ[k] = v
    for k in UNSET:
        os.environ.pop(k, None)


if __name__ == "__main__":
    _set_env()
    trace = scenario(load(sys.argv[1], "ce_pre_m0"))
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        json.dump({"source": "webapp/continuous_engine.py @ %s" % PRE_M0_COMMIT, "trace": trace,
                   "backend": "none (stub organs, pure-Python tick; no substrate)"}, fh, indent=1,
                  sort_keys=True)
    print("wrote", GOLDEN, "| calls:", len(trace["log"]), "| records:", sum(len(v) for v in trace["inner"].values()))
