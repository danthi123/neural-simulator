"""Kill-safe THREE-STATE + SCALE-CONFIDENCE gate: does the validated
reward-FREE Tonegawa engram bind BOOTSTRAP the rewarded episode the
compose-bridge VOID lacked (n_rewarded=0), so the validated temporal-
credit/eligibility mechanism GENERATIVELY refines it -- and is that
capability SCALE-CONFIDENT across a pre-registered local scale ladder?

REUSES byte-UNMODIFIED: compose_bridge_core.cbr_verdict (frozen _CBR_*
INHERITED -- NO new movable bar), the Tonegawa engram bridge API, the
validated temporal-credit/eligibility path, build_biological_brain_
regions, sim.train_checkpoint, sim.neuromodulators. EVERY condition
gets the IDENTICAL engram bootstrap; conditions differ ONLY in the
temporal-credit refinement on top (mechanism isolation). NO automatic
differentiation. ASCII only.

HONEST CEILING (printed, never spun): a SCALE-CONFIDENT PASS = the
generative mechanism works locally at small capacity AND shows no
architectural ceiling across the local ladder (so scale-up is
justified) -- explicitly NOT GPT-class/open-ended fluent composition
on local hardware, NOT an LLM, NOT conversation-solved. A works-small-
but-plateaus result is an honest non-success (NOT a win) that triggers
the autonomous Q2 pivot."""
from __future__ import annotations
import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.compose_bridge_core import cbr_verdict

# Pre-registered, NEVER tuned (mirrors compose_bridge_gate's frozen
# _GAMMA/_LAMBDA pattern). _SCALE_TOL is the substrate's irreducible
# greedy-eval noise floor, justified BEFORE any run.
_SCALE_LADDER = (4, 8, 16)
_SCALE_TOL = 0.05


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def scale_confidence(rungs):
    """Pure, deterministic, fail-closed classification over the ordered
    per-rung records. rungs: list of {"B", "verdict": {"GATE": ...},
    "td_mean", "engram_only_mean"} ordered by ascending B.

    Pre-registered (NEVER tuned):
      (a) every rung GATE == PASS;
      (b) td non-decreasing up to _SCALE_TOL across adjacent rungs;
      (c) at the LARGEST rung td >= _CBR_SCI_ACC_MIN AND
          td - engram_only >= _SCALE_TOL (generative signature holds at
          the hardest scale).
    SCALE-CONFIDENT iff (a)&(b)&(c). Else classify honestly:
      any VOID rung -> VOID; any FAIL rung -> FAIL; all PASS but
      (b)/(c) fails -> WORKS-SMALL-NO-SCALE-CONFIDENCE. Non-numeric/
      missing/unordered -> VOID (never raise)."""
    from research.runners.compose_bridge_core import _CBR_SCI_ACC_MIN
    try:
        ordered = sorted(rungs, key=lambda r: r["B"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by B"}
    if [r.get("B") for r in ordered] != list(_SCALE_LADDER):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder != pre-registered %s"
                          % (_SCALE_LADDER,)}
    gates = []
    for r in ordered:
        v = r.get("verdict")
        g = v.get("GATE") if isinstance(v, dict) else None
        gates.append(g)
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is VOID/missing"}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE is FAIL"}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is not PASS/FAIL/VOID"}
    tds, eos = [], []
    for r in ordered:
        t = _num(r.get("td_mean"))
        e = _num(r.get("engram_only_mean"))
        if t is None or e is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric rung metric"}
        tds.append(t)
        eos.append(e)
    monotone = all(tds[i + 1] >= tds[i] - _SCALE_TOL
                   for i in range(len(tds) - 1))
    top_ok = (tds[-1] >= _CBR_SCI_ACC_MIN
              and (tds[-1] - eos[-1]) >= _SCALE_TOL)
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "all rungs PASS; td monotone up to tol; "
                          "generative signature holds at largest rung",
                "td_by_rung": tds, "engram_only_by_rung": eos}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "all rungs PASS but %s%s"
                      % ("" if monotone else "td degrades beyond tol; ",
                         "" if top_ok else "generative signature absent "
                         "at largest rung"),
            "td_by_rung": tds, "engram_only_by_rung": eos}
