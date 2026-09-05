"""RANK-13 PRODUCTION-FLIP: CORRECTED DIAGNOSIS instrument (2026-09-05).

The flip-verify NO-GO (research/findings/2026-09-05-rank13-selfid-anaphora-PRODUCTION-FLIP-NO-GO-stale-referent-
selfalias-misroute.md) attributed the seed=44 "it eat" regression to a "stale-referent self-alias misroute": the
theory was that `_resolve_anaphora` FAILS to substitute a decayed-referent pronoun (returns the question
UNCHANGED, `held_referent()[0] is None`), leaving a literal "it" to survive into `_extract_route`, where the
BRAIN_NEURAL_SELFID extension's unconditional `self.router._resolve_self(t)` then wrongly treats it as
self-referential.

Directly instrumenting `ChatBrain._resolve_anaphora` / `_extract_route` / `_gate_router_combine` /
`QuestionRouter.match_fact` at the CLASS level (so the REAL, unmodified `_rank13_selfid_anaphora_prodflip_verify`
functions are traced with zero reimplementation) shows this theory does not match what the code actually does:

  - seed=44 "what does it eat?": `_resolve_anaphora` SUBSTITUTES successfully -> "what does ball eat?"
    (`held_referent()` returns a CONFIDENT-but-WRONG referent, 'ball', not None). `_extract_route("what does ball
    eat?")` never sees a self-alias token at all ('ball' isn't one) -- the regression is 100% inside `gate()`'s
    BRAIN_NEURAL_ANAPHORA_ABSTAIN branch: the substrate honestly abstains on the (wrong) query "ball eat" (no such
    fact), and the flag converts that into a hard `None` instead of falling through to the host router, whose
    forgiving verb-only keyword match (`QuestionRouter.match_fact`) would have found `('cat','eat','fish')` via
    "eat" alone, ignoring the wrong "ball" keyword, recovering the CORRECT answer despite the bad referent.
  - seed=43 "what does it fly?": `_resolve_anaphora` ALSO substitutes successfully, but to 'brain' this time --
    a THIRD distinct wrong value (seed=44 got 'ball'), confirming this is a general referent-misidentification
    property of the discourse-WM read, not a fixed failure mode. Because the wrongly-substituted agent happens to
    equal the literal string 'brain', `_substrate_recall`'s SELF/IDENTITY candidate-relation retry -- documented
    as "reached ONLY for a bare identity query on the self" but (before this session's fix) missing the `v ==
    "isa"` check its own docstring claims and its `gnw_bus_shadow.gate_via_bus` mirror already enforces -- fires
    on ANY agent=='brain' miss and fabricates ['brain','use','spikes'] for a question that was never about the
    brain.

BOTH are downstream of the SAME root phenomenon: the discourse-referent WM's cleanup-memory read
(`MultiTurnAgent.held_referent`) can return a CONFIDENT (above its own `self._spec` threshold) but WRONG referent
under noise, well after the correct referent was established. This is a pre-existing WM-fidelity property,
orthogonal to rank-13's own scope; the rank-13 flags merely determine how the SUBSEQUENT flag-gated logic reacts
once a wrong referent has already been substituted.

This runner reproduces both traces and saves them as a citable artifact. numpy-CPU only; no sim/ edit.

Run: SIM_BACKEND=numpy python -m research.runners._rank13_prodflip_corrected_diagnosis
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.brain_chat_tui as bct  # noqa: E402

TARGET = {"what does it eat?", "what does it fly?"}


def _trace_seed(seed, questions_of_interest):
    trace = []
    orig_ra = bct.ChatBrain._resolve_anaphora
    orig_er = bct.ChatBrain._extract_route
    orig_rc = bct.ChatBrain._gate_router_combine
    orig_mf = bct.QuestionRouter.match_fact

    def ra(self, question):
        out = orig_ra(self, question)
        if question in TARGET:
            trace.append({"call": "_resolve_anaphora", "in": question, "out": out,
                          "substituted": out != question})
        return out

    def er(self, question):
        out = orig_er(self, question)
        if question in questions_of_interest:
            trace.append({"call": "_extract_route", "in": question, "out": out})
        return out

    def rc(self, q):
        out = orig_rc(self, q)
        if q in questions_of_interest:
            trace.append({"call": "_gate_router_combine", "in": q, "out": out})
        return out

    def mf(self, question, stored_facts):
        out = orig_mf(self, question, stored_facts)
        if question in questions_of_interest:
            trace.append({"call": "match_fact", "in": question, "out": list(out) if out[0] else [None, out[1]]})
        return out

    bct.ChatBrain._resolve_anaphora = ra
    bct.ChatBrain._extract_route = er
    bct.ChatBrain._gate_router_combine = rc
    bct.QuestionRouter.match_fact = mf
    try:
        import research.runners._rank13_selfid_anaphora_prodflip_verify as v
        chat_off = v._build(seed, False, False, "rf")
        off_batch = v._batch(chat_off, v._ORDERED_PANEL)
        chat_on = v._build(seed, True, False, "rf")
        on_batch = v._batch(chat_on, v._ORDERED_PANEL)
    finally:
        bct.ChatBrain._resolve_anaphora = orig_ra
        bct.ChatBrain._extract_route = orig_er
        bct.ChatBrain._gate_router_combine = orig_rc
        bct.QuestionRouter.match_fact = orig_mf
    return {
        "seed": seed,
        "trace": trace,
        "final_off": {q: off_batch[q][0] for q in questions_of_interest},
        "final_on": {q: on_batch[q][0] for q in questions_of_interest},
    }


def main():
    out = {
        "runner": "_rank13_prodflip_corrected_diagnosis",
        "seed44_it_eat": _trace_seed(44, {"what does it eat?"}),
        "seed43_it_fly": _trace_seed(43, {"what does it fly?"}),
    }
    op = "research/findings/raw/_rank13_selfid_anaphora_prodflip/corrected_diagnosis_trace.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
