"""Pure FIXED-bar THREE-STATE + SCALE-CONFIDENCE verdict for Q2
two-module per-token grounded constrained decoding. Mirrors the
adversarial-hardened generator_g_core / compose_bridge_core DISCIPLINE
(fixed bars NEVER tuned, instrument-validity FIRST, fail-closed, VOID
strictly distinct from FAIL, malformed/junk -> VOID-not-raise). Holds
its OWN frozen _CDC_*; does NOT import/mutate any *_core. REUSES the
validated generator_g_core metric PRIMITIVES (FUNCTION_WORDS,
is_answered) by import, byte-UNMODIFIED. Pure stdlib+typing; NO torch,
NO autograd. ASCII only.

KEY (cheap-probe-surfaced, recorded in the design): a per-token veto
makes ungrounded-entity-rate ~0 BY CONSTRUCTION -- MECHANICAL, NOT the
discriminating result. The DISCRIMINATING Q2 signature is "constrained
stays NON-VACUOUS" via a STRENGTHENED grounded-CONTENT-word bar
(>= _CDC_MIN_GROUNDED_CONTENT distinct on-proposition content words),
NOT bare is_answered>=1 (proven too weak by the cheap probe)."""
from __future__ import annotations
import math
import re
from typing import Dict

from research.runners.generator_g_core import (
    FUNCTION_WORDS, is_answered)  # reused byte-UNMODIFIED

_CDC_FAITHFUL_MAX = 0.20
_CDC_MIN_GROUNDED_CONTENT = 2
_CDC_MIN_GROUNDED_ANSWER_RATE = 0.5
_CDC_MIN_SEEDS = 3
_CDC_SCALE_LADDER = (6, 12, 24)
_CDC_SCALE_TOL = 0.10
# ADDITIVE instrument-validity floor (pre-registered NOW, before the
# decisive run; a NEW bar, never loosens an existing one). The Q2
# premise is "the per-token grounded veto makes ungrounded content ~0
# BY CONSTRUCTION while staying NON-VACUOUS". A BPE (subword) tokenizer
# means grounded content words are multi-subword (e.g. max ->
# ['ma','x</w>']). If the faithful veto cannot even EXPRESS the
# grounded content (the model literally cannot emit "max"), then a
# FAIL is ambiguous -- it conflates "composition tested & vacuous"
# with "instrument structurally cannot test the premise". This floor
# makes the subword-defeated regime an HONEST VOID (cannot-test), NOT
# an ambiguous FAIL. >= half of the KB propositions must be fully
# emittable under the constructed faithful mask for the instrument to
# be able to see the tested effect at all.
_CDC_MIN_MULTITOKEN_EMITTABLE = 0.5


def _norm(s):
    out = []
    for w in str(s).split():
        t = re.sub(r"[^\w]", "", w.lower())
        if t:
            out.append(t)
    return out


def grounded_content_count(response_text, retrieved_text,
                           function_words=FUNCTION_WORDS) -> int:
    ret = set(_norm(retrieved_text))
    seen = set()
    for w in _norm(response_text):
        if w not in function_words and w in ret:
            seen.add(w)
    return len(seen)


def nonvacuous_answered(response_text, retrieved_text) -> bool:
    if not is_answered(response_text):
        return False
    return (grounded_content_count(response_text, retrieved_text)
            >= _CDC_MIN_GROUNDED_CONTENT)


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


_REQUIRED = ("unconstrained_uer", "constrained_uer",
             "constrained_nonvac_rate", "shuffled_uer",
             "shuffled_nonvac_rate", "bare_moat_abstain_rate",
             "abstain_on_ungrounded_rate",
             "constrained_multitoken_emittable_rate")


def cdc_verdict(per_seed: dict) -> dict:
    bars = {"FAITHFUL_MAX": _CDC_FAITHFUL_MAX,
            "MIN_GROUNDED_CONTENT": _CDC_MIN_GROUNDED_CONTENT,
            "MIN_GROUNDED_ANSWER_RATE": _CDC_MIN_GROUNDED_ANSWER_RATE,
            "MIN_SEEDS": _CDC_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CDC_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CDC_MIN_SEEDS, **base}
    v1_ok = science_ok = controls_fail = no_confab_ok = True
    multitoken_ok = True
    metrics_finite = True
    for s in seeds:
        d = per_seed[s]
        if not isinstance(d, dict):
            metrics_finite = False
            continue
        vals = {k: _finite(d.get(k)) for k in _REQUIRED}
        if any(v is None for v in vals.values()):
            metrics_finite = False
            continue
        if not (vals["unconstrained_uer"] > _CDC_FAITHFUL_MAX
                and vals["bare_moat_abstain_rate"] > 0.0):
            v1_ok = False
        if (vals["constrained_multitoken_emittable_rate"]
                < _CDC_MIN_MULTITOKEN_EMITTABLE):
            multitoken_ok = False
        unconstrained_fails = vals["unconstrained_uer"] > _CDC_FAITHFUL_MAX
        shuffled_fails = (vals["shuffled_uer"] > _CDC_FAITHFUL_MAX
                          or vals["shuffled_nonvac_rate"]
                          < _CDC_MIN_GROUNDED_ANSWER_RATE)
        if not (unconstrained_fails and shuffled_fails):
            controls_fail = False
        if not (vals["constrained_uer"] <= _CDC_FAITHFUL_MAX
                and vals["constrained_nonvac_rate"]
                >= _CDC_MIN_GROUNDED_ANSWER_RATE):
            science_ok = False
        if vals["abstain_on_ungrounded_rate"] < \
                vals["bare_moat_abstain_rate"] - 1e-9:
            no_confab_ok = False
    instrument_valid = bool(v1_ok and controls_fail and no_confab_ok
                            and multitoken_ok and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: unconstrained did NOT drift above "
                       "the faithful bar (instrument cannot see drift)")
        if not controls_fail:
            why.append("a control did not fail -> veto NOT the "
                       "discriminator (non-discriminating)")
        if not no_confab_ok:
            why.append("no-confab NOT preserved (abstain_on_ungrounded "
                       "< bare moat)")
        if not multitoken_ok:
            why.append("veto structurally cannot emit grounded "
                       "multi-token content (subword-defeated) -- "
                       "instrument cannot test the premise")
        if not metrics_finite:
            why.append("a required metric non-numeric/non-finite/"
                       "malformed")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            "note": ("constrained ungrounded-entity-rate ~0 is "
                     "MECHANICAL (per-token veto, by construction) -- "
                     "NOT the discriminating result; the discriminator "
                     "is constrained NON-VACUITY vs the failing "
                     "controls"), **base}


def cdc_scale_confidence(rungs):
    try:
        ordered = sorted(rungs, key=lambda r: r["K"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by K"}
    if [r.get("K") for r in ordered] != list(_CDC_SCALE_LADDER):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder != %s" % (_CDC_SCALE_LADDER,)}
    gates = [r.get("verdict", {}).get("GATE")
             if isinstance(r.get("verdict"), dict) else None
             for r in ordered]
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE VOID/missing"}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE FAIL"}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE not PASS/FAIL/VOID"}
    nv = []
    for r in ordered:
        f = _finite(r.get("constrained_nonvac_rate_mean"))
        if f is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric rung non-vacuity"}
        nv.append(f)
    monotone = all(nv[i + 1] >= nv[i] - _CDC_SCALE_TOL
                   for i in range(len(nv) - 1))
    top_ok = nv[-1] >= _CDC_MIN_GROUNDED_ANSWER_RATE
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "all rungs PASS; non-vacuity non-decreasing "
                          "up to tol; holds at largest rung",
                "nonvac_by_rung": nv}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "all rungs PASS but %s%s"
                      % ("" if monotone else "non-vacuity degrades "
                         "beyond tol; ",
                         "" if top_ok else "non-vacuity below bar at "
                         "largest rung"),
            "nonvac_by_rung": nv}
