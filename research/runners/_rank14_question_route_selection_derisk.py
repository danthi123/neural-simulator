"""Rank-14 scaffold retirement: NL question-ROUTE selection as a spiking N-way lateral-inhibition WTA
(`research/coordination/scaffold_retirement_backlog.md` rank-14, "NL question-routing host comprehension" --
MED-HIGH/fresh, not previously attempted). Biology binding: `research/biology/question-route-selection-wta.md`.

WHAT THIS RETIRES. `research/runners/brain_chat_tui.py`'s `ChatBrain._extract_route` decides, in host Python,
WHICH of four comprehension routes handles an incoming question by testing three regex-shaped special cases in a
FIXED PRIORITY ORDER and falling through only when none matched:

    _relf = self._relation_fronted_route(question)          # 'what country is chelsea fc from?'
    if _relf is not None: return _relf
    _kbrel = self._kb_relation_question_route(question)     # 'where was X born?', 29-relation curated table
    if _kbrel is not None: return _kbrel
    if len(content) <= 1:
        _defo = self._definitional_copula_route(question)   # 'what is X?'
        if _defo is not None: return _defo
    ... falls through to the ALREADY-NEURAL generic SVO BridgeParser (the CHOOSE branch, `_neural_question_parse`)

Each individual regex TEST is a legitimate matched-filter read of the surface string -- the same honesty class as
any other host-side sensory feature extraction already accepted in this codebase (e.g. the ACC/BG STOP-trigger
de-risk's `n_ignited`/`mm_peak` afferents, themselves host-computed reads of other organs). What was NOT neural is
the COMBINATION: which construction wins WHEN MORE THAN ONE COULD APPLY, decided unconditionally by a fixed
textual `if`/`elif` order rather than by the STRENGTH of the recognized cue. This module retires exactly that
combination step -- not the three regex feature-extractors themselves (a separate, larger residual: teaching the
substrate to recognize these constructions from corpus statistics instead of a curated table; out of scope here,
named honestly in the biology binding's "Honesty boundary" section).

THE MECHANISM (generalizes an ALREADY-VALIDATED primitive; see the biology binding for the full citation chain).
Four small excitatory assemblies (GENERIC, DEFCOP, RELFRONT, KBREL), each with its own dedicated fast-spiking
cross-inhibition sub-pool, compete under mutual/reciprocal lateral inhibition -- `_build_bridge`/`_pool_rates`
REUSED VERBATIM BY IMPORT from `_affect_marker_wta_derisk.py` (the already-GO'd, production-flipped N=6
affect-expression-marker selector), with `n_pools=4` instead of 6. GENERIC additionally carries a constant
BASELINE "elsewhere" current (Pinker-Ullman words-and-rules: the default rule proceeds unopposed unless a
specific, genuinely-recognized exception is supra-threshold) so it wins whenever none of the three construction-
specific regexes match; each exception assembly's current is driven ON only when its OWN host regex genuinely
matches the question, OFF otherwise. The assembly whose rate clears the runner-up by a dead margin IS the route
decision, read off `cp_firing_states` -- never a host `if`/`elif`.

BATTERY. A real question battery is generated programmatically from the SAME tables `_kb_relation_question_route`
dispatches from (`_KB_UNDERSCORED_RELATIONS`, `_kb_relation_phrase`, `_KB_RELATION_IDIOMS`) plus hand-written
GENERIC/DEFCOP/RELFRONT examples, INCLUDING one deliberately AMBIGUOUS item ("what country is entity_x a citizen
of?") that genuinely matches BOTH `_REL_FRONTED_RE` and a KBREL idiom -- host priority (relf checked first) must
be reproduced, not merely the easy non-overlapping cases (the discriminating-power, non-ceiling arm). Ground
truth (`host_route_type`) reproduces `_extract_route`'s own priority order for the PARITY comparison only;
`_extract_route` itself is never imported for its decision, only its regex objects/tables for feature extraction.

GO GATE (6 seeds 42/43/44/100/101/102, >=5/6 seeds, pre-registered BEFORE the decisive run):
  (1) PARITY -- the intact circuit's argmax route matches `host_route_type` on >=95% of the full battery.
  (2) AMBIGUOUS-CASE PARITY -- the deliberately overlapping RELFRONT/KBREL item resolves to RELFRONT (matching
      host priority) via genuinely stronger RELFRONT drive, not a host tie-break.
  (3) FULL LESION -- zeroing all three exception pathways' evidence-driven gain (mirrors `_relation_fronted_
      enabled()==False` + `_kb_relation_questions_enabled()==False` + the defcop guard off) routes EVERY battery
      item to GENERIC (100%) -- the substrate's own "elsewhere" case, matching the host's own all-flags-off
      fallthrough.
  (4) PER-PATHWAY INDEPENDENCE -- lesioning the OTHER TWO exception pathways leaves EACH pathway's own battery
      subset still resolving correctly (>=90%) -- no pathway depends on another to function.
  (5) GRADED SWEEP (discriminating, non-ceiling) -- scaling a genuine RELFRONT item's evidence-driven current from
      0 to full drive crosses GENERIC's rate at some intermediate point (a real flip, not an always-GENERIC or
      always-RELFRONT ceiling).
  (6) DETERMINISM -- build-twice-at-one-seed identical seed-derived Izhikevich-parameter hash (cfg.seed).

ANTI-CHEATS: `tools.lab.attributable_to` on parity restricted to the exception-labeled subset (intact vs full-
lesion) must be >=0.5; the battery health-check asserts all 4 route labels are represented AND the ambiguous item
is genuinely dual-matching (not just asserted); `tools.verdict.Verdict` carries every precondition into the
artifact so a partial pass cannot silently read as GO.

HONEST RESIDUALS (named, not claimed closed):
  1. The regex feature EXTRACTION stays host code (see biology binding) -- this de-risk retires the DISPATCH only.
  2. The evidence->current scale (OFF_PA/GENERIC_BASELINE_PA/EXCEPTION_ON_PA) and RELFRONT's small priority-tilt
     over KBREL are UNTUNED de-risk operating-point knobs (reusing the affect-marker circuit's own calibrated
     150/1350 pA regime verbatim where possible), not biology-REQUIRED constants.
  3. This is a DE-RISK (research/runners only) -- NOT wired to replace `_extract_route` in production by this
     change (RAM-discipline: no integrated brain-chat verify was run).

Usage (CPU cheap-first; tiny nets, <200 neurons total):
  SIM_BACKEND=numpy python -u -m research.runners._rank14_question_route_selection_derisk --smoke --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._rank14_question_route_selection_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_rank14_question_route_wta_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: the N-pool cross-inhibition WTA primitive -- NOT reimplemented. See the biology binding's
# second LOCAL source (`_build_bridge` generalizes the N=6 affect-marker selector to N=4 route candidates here).
from research.runners._affect_marker_wta_derisk import _build_bridge, _pool_rates

# reuse-by-import: the determinism-hash helper (cfg.seed, not actual_seed_used -- CLAUDE.md's seeded-substrate
# trap).
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash

# reuse-by-import: the EXACT host regex evidence objects/tables `_extract_route` dispatches from -- legitimate
# host-side feature EXTRACTION (see the biology binding's Honesty boundary), NOT the decision. Importing the
# module is a static/cheap operation (only compiles regexes at import time; no bridge is built).
from research.runners.brain_chat_tui import (
    _REL_FRONTED_RE, _KB_RELATION_PATTERNS, _KB_UNDERSCORED_RELATIONS, _kb_relation_phrase, _KB_RELATION_IDIOMS,
)

from sim.backend import get_backend

ROUTES = ("GENERIC", "DEFCOP", "RELFRONT", "KBREL")
N_ROUTES = len(ROUTES)
_IDX = {r: i for i, r in enumerate(ROUTES)}

# ── `_extract_route`'s OWN stopword set (brain_chat_tui.py, a LOCAL variable inside the method body -- copied
# verbatim since it cannot be imported; DATA describing the host's own tokenization convention, not a decision).
_STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
         "to", "it", "that", "this", "they", "them", "of", "about"}

# `_definitional_copula_route`'s OWN regex, copied verbatim (written inline in that method -- not a module-level
# compiled pattern, so it cannot be imported directly; reused for feature-extraction fidelity only).
_DEFCOP_RE = re.compile(r"^\s*(?:what(?:'s|s| is| are)|who(?:'s|s| is| are)|define)\s+"
                        r"(?:an?\s+|the\s+)?(.+?)\s*\??\s*$")


def _content_len(question: str) -> int:
    toks = [t.lower().strip(".,!?") for t in question.split()]
    content = [t for t in toks if t and t not in _STOP]
    return len(content)


def _defcop_shape_matches(question: str) -> bool:
    m = _DEFCOP_RE.match(question.lower())
    if not m:
        return False
    subj = m.group(1).strip().strip(".,!?").strip()
    if not subj or " of " in (" %s " % subj):
        return False
    return True


def host_evidence(question: str) -> dict:
    """The THREE regex tests `_extract_route` runs, as independent boolean evidence -- the legitimate host-side
    feature EXTRACTION this de-risk does not retire. GENERIC's 'evidence' is always 1.0 (the elsewhere case is
    always structurally available -- it is not something a flag disables)."""
    q = question.strip()
    relf = 1.0 if _REL_FRONTED_RE.match(q) else 0.0
    kbrel = 1.0 if any(rx.match(q) for rx, _ in _KB_RELATION_PATTERNS) else 0.0
    clen = _content_len(question)
    defcop = 1.0 if (clen <= 1 and _defcop_shape_matches(question)) else 0.0
    return {"GENERIC": 1.0, "DEFCOP": defcop, "RELFRONT": relf, "KBREL": kbrel, "content_len": clen}


def host_route_type(question: str) -> str:
    """The EXACT priority order `_extract_route` uses (relf > kbrel > defcop[if content_len<=1] > generic),
    reproduced ONLY for the parity comparison -- `_extract_route` itself is untouched by this de-risk."""
    ev = host_evidence(question)
    if ev["RELFRONT"] > 0:
        return "RELFRONT"
    if ev["KBREL"] > 0:
        return "KBREL"
    if ev["DEFCOP"] > 0:
        return "DEFCOP"
    return "GENERIC"


# ── the battery: real questions, labels computed from the REAL imported host evidence (never hand-asserted) ──────
_GENERIC_QS = [
    "what does the wolf hunt", "what did the cat eat", "who does the dog chase", "what does the bird eat",
    "what does the fox hunt", "who did the boy see", "what does the girl want", "what does the teacher teach",
]
_DEFCOP_QS = [
    "what is canada", "who is einstein", "what is a mammal", "who is the president", "what is oxygen",
    "who is shakespeare",
]
_RELFRONT_QS = [
    "what country is chelsea fc from", "what sport is wimbledon in", "what language is quebec in",
    "what continent is egypt in", "what color is the sky", "what genre is jazz",
]
# One idiomatic literal example per relation that HAS a curated idiom (hand-written to match its own regex).
_KBREL_IDIOM_QS = {
    "place_of_birth": "where was entity_x born",
    "place_of_death": "where did entity_x die",
    "educated_at": "where was entity_x educated",
    "work_location": "where does entity_x work",
    "headquarters_location": "where is entity_x headquartered",
    "employer": "who does entity_x work for",
    "award_received": "what award did entity_x receive",
    "member_of_political_party": "what political party is entity_x a member of",
    "country_of_citizenship": "what country is entity_x a citizen of",  # DELIBERATE: also matches _REL_FRONTED_RE
    "followed_by": "what is entity_x followed by",
    "subclass_of": "what is entity_x a subclass of",
    "part_of": "what is entity_x part of",
    "member_of": "what is entity_x a member of",
    "participant_of": "what did entity_x participate in",
    "shares_border_with": "what does entity_x share a border with",
    "record_label": "what record label is entity_x signed to",
    "languages_spoken_written_or_signed": "what languages does entity_x speak",
}
_AMBIGUOUS_QUESTION = "what country is entity_x a citizen of"


def build_battery():
    """Programmatically assembled real-question battery. Labels come from `host_route_type` (the real regex
    objects), never hand-asserted -- see `battery_health_check`."""
    qs = list(_GENERIC_QS) + list(_DEFCOP_QS) + list(_RELFRONT_QS)
    for relation in _KB_UNDERSCORED_RELATIONS:               # the 29 GENERIC kb-relation templates
        phrase = _kb_relation_phrase(relation)
        qs.append(f"what is entity_x's {phrase}")
    for relation, q in _KBREL_IDIOM_QS.items():               # the 17 idiom examples (incl. the ambiguous one)
        assert relation in _KB_RELATION_IDIOMS, relation
        qs.append(q)
    # de-duplicate while preserving order (the ambiguous item is written once, in the idiom dict).
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def battery_health_check(battery, verbose=True):
    """Assert the battery is genuinely diverse (all 4 labels represented) and that the ambiguous item really is
    dual-matching under the REAL regexes (not just asserted) -- printed once, not per-seed."""
    labels = {q: host_route_type(q) for q in battery}
    counts = {r: sum(1 for v in labels.values() if v == r) for r in ROUTES}
    amb_ev = host_evidence(_AMBIGUOUS_QUESTION)
    dual_matching = bool(amb_ev["RELFRONT"] > 0 and amb_ev["KBREL"] > 0)
    if verbose:
        print(f"[battery] n={len(battery)} label counts={counts}", flush=True)
        print(f"[battery] ambiguous item evidence={amb_ev} dual_matching={dual_matching}", flush=True)
    ok = all(c > 0 for c in counts.values()) and dual_matching
    return {"counts": counts, "ambiguous_evidence": amb_ev, "dual_matching": dual_matching,
            "n_battery": len(battery), "ok": bool(ok)}


# ── evidence -> current (pA). Reuses the affect-marker circuit's OWN calibrated OFF/ON regime (150 / 1350 pA)
# verbatim where possible (see HONEST RESIDUAL #2: an untuned de-risk operating-point knob, not a biology-
# required constant). GENERIC gets a fixed BASELINE between the two so it wins only when unopposed.
OFF_PA = 150.0
GENERIC_BASELINE_PA = 700.0
EXCEPTION_ON_PA = 1350.0
RELFRONT_PRIORITY_TILT = 1.10          # RELFRONT's own ON drive is checked FIRST by the host; a small, honestly-
                                        # named tilt lets it win the one deliberately ambiguous overlap case via a
                                        # real drive-strength difference instead of a hidden host tie-break.

WARMUP_STEPS = 60
WASHOUT_STEPS = 40
RUN_STEPS = 60
DEAD_MARGIN = 0.05                      # same convention/units as `_affect_marker_wta_derisk.DEAD_MARGIN`


def evidence_to_currents(ev: dict, *, lesion_relf: bool = False, lesion_kbrel: bool = False,
                         lesion_defcop: bool = False, relf_gain: float = 1.0) -> np.ndarray:
    """ROUTES-ordered drive array (pA). `lesion_*` forces that exception's evidence-driven gain to OFF regardless
    of the real regex evidence (the load-bearing lesion probes); `relf_gain` in [0,1] scales RELFRONT's ON drive
    for the graded-sweep anti-cheat (0 = fully lesioned, 1 = full drive)."""
    cur = np.zeros(N_ROUTES, dtype=np.float64)
    cur[_IDX["GENERIC"]] = GENERIC_BASELINE_PA
    defcop_on = (ev["DEFCOP"] > 0) and not lesion_defcop
    cur[_IDX["DEFCOP"]] = EXCEPTION_ON_PA if defcop_on else OFF_PA
    kbrel_on = (ev["KBREL"] > 0) and not lesion_kbrel
    cur[_IDX["KBREL"]] = EXCEPTION_ON_PA if kbrel_on else OFF_PA
    relf_on = (ev["RELFRONT"] > 0) and not lesion_relf
    relf_full = EXCEPTION_ON_PA * RELFRONT_PRIORITY_TILT
    cur[_IDX["RELFRONT"]] = (OFF_PA + max(0.0, min(1.0, relf_gain)) * (relf_full - OFF_PA)) if relf_on else OFF_PA
    return cur


def run_trial(bridge, marker_idx, question: str, **lesion_kwargs) -> dict:
    ev = host_evidence(question)
    drive = evidence_to_currents(ev, **lesion_kwargs)
    rates = _pool_rates(bridge, marker_idx, drive, warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
    order = np.argsort(rates)[::-1]
    top, second = int(order[0]), int(order[1])
    margin = float(rates[top] - rates[second])
    winner = ROUTES[top] if margin > DEAD_MARGIN else None
    return {"question": question, "evidence": ev, "drive": drive.tolist(), "rates": rates.tolist(),
           "winner": winner, "margin": margin, "host": host_route_type(question),
           "match": bool(winner == host_route_type(question))}


def evaluate_seed(seed: int, battery, verbose: bool = True) -> dict:
    xp, _ = get_backend()
    bridge, marker_idx, fsi_idx = _build_bridge(seed, N_ROUTES, "rt")

    # ── (1)+(2) INTACT PARITY over the full battery (incl. the ambiguous item) ──────────────────────────────────
    intact = [run_trial(bridge, marker_idx, q) for q in battery]
    n_match = sum(1 for r in intact if r["match"])
    parity_rate = n_match / len(intact)
    amb = next(r for r in intact if r["question"] == _AMBIGUOUS_QUESTION)
    ambiguous_ok = bool(amb["winner"] == "RELFRONT" and amb["host"] == "RELFRONT")

    exception_qs = [q for q in battery if host_route_type(q) != "GENERIC"]
    generic_qs = [q for q in battery if host_route_type(q) == "GENERIC"]

    # ── (3) FULL EXCEPTION LESION -- every item must fall to GENERIC ────────────────────────────────────────────
    full_lesioned = [run_trial(bridge, marker_idx, q, lesion_relf=True, lesion_kbrel=True, lesion_defcop=True)
                     for q in battery]
    full_lesion_all_generic = all(r["winner"] == "GENERIC" for r in full_lesioned)
    full_lesion_rate = sum(1 for r in full_lesioned if r["match"]) / len(full_lesioned)

    # attribution: how much of PARITY on the exception-labeled subset is owned by the exception pathways
    # (intact vs fully-lesioned), on the SAME subset -- the anti-cheat this repo's gap#5 lesson requires.
    def _subset_parity(results, qs):
        s = {r["question"]: r for r in results}
        return sum(1 for q in qs if s[q]["match"]) / max(1, len(qs))

    parity_exc_intact = _subset_parity(intact, exception_qs)
    parity_exc_lesioned = _subset_parity(full_lesioned, exception_qs)
    attribution = attributable_to("exception-subset parity: intact vs full-exception-lesion",
                                  parity_exc_intact, parity_exc_lesioned, warn_below=0.5)

    # ── (4) PER-PATHWAY INDEPENDENCE -- lesion the OTHER TWO, this pathway's own subset must still resolve ───────
    def _subset_for(route):
        return [q for q in battery if host_route_type(q) == route]

    per_pathway = {}
    for route, lesion_kwargs in (
        ("RELFRONT", dict(lesion_kbrel=True, lesion_defcop=True)),
        ("KBREL", dict(lesion_relf=True, lesion_defcop=True)),
        ("DEFCOP", dict(lesion_relf=True, lesion_kbrel=True)),
    ):
        subset = _subset_for(route)
        res = [run_trial(bridge, marker_idx, q, **lesion_kwargs) for q in subset]
        rate = sum(1 for r in res if r["match"]) / max(1, len(res))
        per_pathway[route] = {"n": len(subset), "rate": rate, "ok": bool(rate >= 0.90)}
    per_pathway_independent = all(v["ok"] for v in per_pathway.values())

    # ── (5) GRADED SWEEP (discriminating, non-ceiling): a real RELFRONT item, evidence gain 0..1 ────────────────
    sweep_q = "what country is chelsea fc from"
    ev_sweep = host_evidence(sweep_q)
    assert ev_sweep["RELFRONT"] > 0 and ev_sweep["KBREL"] == 0.0, "sweep item must be a clean single-route match"
    sweep = []
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        drive = evidence_to_currents(ev_sweep, relf_gain=g)
        rates = _pool_rates(bridge, marker_idx, drive, warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
        sweep.append({"gain": g, "rate_generic": float(rates[_IDX["GENERIC"]]),
                     "rate_relfront": float(rates[_IDX["RELFRONT"]])})
    sweep_starts_generic = bool(sweep[0]["rate_relfront"] <= sweep[0]["rate_generic"])
    sweep_ends_relfront = bool(sweep[-1]["rate_relfront"] > sweep[-1]["rate_generic"])
    sweep_flips = bool(sweep_starts_generic and sweep_ends_relfront)
    sweep_monotone = bool(all(sweep[i + 1]["rate_relfront"] >= sweep[i]["rate_relfront"] - 1e-9
                              for i in range(len(sweep) - 1)))

    # ── (6) DETERMINISM: build-twice-at-one-seed identical seed-derived hash ───────────────────────────────────
    h1 = _threshold_hash(bridge, xp)
    bridge2, marker_idx2, _ = _build_bridge(seed, N_ROUTES, "rt")
    h2 = _threshold_hash(bridge2, xp)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    seed_go = bool(parity_rate >= 0.95 and ambiguous_ok and full_lesion_all_generic and per_pathway_independent
                  and sweep_flips and seed_deterministic and (attribution is None or attribution >= 0.5))

    v = Verdict("question-route-selection WTA @ frozen operating point (seed %d)" % seed)
    v.require("real generic-labeled subset non-empty", len(generic_qs) > 0, expect=True)
    v.require("real exception-labeled subset non-empty", len(exception_qs) > 0, expect=True)
    v.require("PARITY with host_route_type >= 0.95 over the full battery", parity_rate >= 0.95, expect=True)
    v.require("ambiguous RELFRONT/KBREL overlap item resolves to RELFRONT (host priority)", ambiguous_ok,
              expect=True)
    v.require("FULL exception lesion routes EVERY item to GENERIC (100%)", full_lesion_all_generic, expect=True)
    v.require("EACH exception pathway independently sufficient (>=90% on its own subset, others lesioned)",
              per_pathway_independent, expect=True)
    v.require("graded RELFRONT-evidence sweep FLIPS the winner from GENERIC to RELFRONT", sweep_flips, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.knob("GENERIC_BASELINE_PA / EXCEPTION_ON_PA / OFF_PA", requested=(GENERIC_BASELINE_PA, EXCEPTION_ON_PA, OFF_PA),
           applied=(GENERIC_BASELINE_PA, EXCEPTION_ON_PA, OFF_PA))
    v.disabled("homeostasis/stdp/hebbian/short-term-plasticity/structural-plasticity",
              why="frozen cross-inhibition weights; a pure feedforward-driven competitive read, no learning")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "n_battery": len(battery), "n_exception_qs": len(exception_qs), "n_generic_qs": len(generic_qs),
        "parity_rate": parity_rate, "n_match": n_match,
        "ambiguous": {"winner": amb["winner"], "host": amb["host"], "ok": ambiguous_ok, "rates": amb["rates"]},
        "full_lesion": {"all_generic": full_lesion_all_generic, "parity_rate": full_lesion_rate},
        "attribution_exception_subset_intact_vs_lesioned": (None if attribution is None else float(attribution)),
        "per_pathway": per_pathway, "per_pathway_independent": per_pathway_independent,
        "sweep": sweep, "sweep_flips": sweep_flips, "sweep_monotone": sweep_monotone,
        "seed_deterministic": seed_deterministic, "threshold_hash": h1,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
        "mismatches": [{"question": r["question"], "host": r["host"], "winner": r["winner"], "margin": r["margin"]}
                       for r in intact if not r["match"]],
    }
    if verbose:
        print(f"[rank14-route-wta seed={seed}] verdict={vd['status']} seed_go={result['seed_go']} "
              f"parity={parity_rate:.3f} ({n_match}/{len(intact)}) ambiguous_ok={ambiguous_ok} "
              f"full_lesion_all_generic={full_lesion_all_generic} per_pathway_ok={per_pathway_independent} "
              f"sweep_flips={sweep_flips} attribution={attribution} det={seed_deterministic}", flush=True)
        if result["mismatches"]:
            print(f"    mismatches: {result['mismatches']}", flush=True)
    return result


def run_smoke(seed: int, args):
    battery = build_battery()
    health = battery_health_check(battery)
    print(f"[rank14-route-wta smoke] seed={seed} battery health={health['ok']}", flush=True)
    r = evaluate_seed(seed, battery, verbose=True)
    out = {"runner": "_rank14_question_route_selection_derisk", "mode": "smoke", "battery_health": health,
          "result": r}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  wrote {args.json}", flush=True)
    return 0 if r["seed_go"] else 1


def main():
    ap = argparse.ArgumentParser(description="Rank-14 question-route-selection spiking WTA de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (smoke)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_rank14_question_route_wta_6seed.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        return run_smoke(args.seed, args)

    battery = build_battery()
    health = battery_health_check(battery)
    print(f"[rank14-route-wta] battery n={health['n_battery']} health_ok={health['ok']} "
          f"seeds={args.seeds} backend={args.backend}\n", flush=True)
    if not health["ok"]:
        print("  ABORT: battery health check failed (see counts/dual_matching above)", flush=True)
        return 2

    results = [evaluate_seed(s, battery, verbose=True) for s in args.seeds]
    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)
    all_go = bool(n_go >= 5 and not any_undefined)

    summary = {
        "runner": "_rank14_question_route_selection_derisk", "mode": "6seed", "seeds": list(args.seeds),
        "backend": args.backend, "battery_health": health, "all_go": all_go, "n_go": n_go,
        "n_seeds": len(results), "any_undefined": any_undefined, "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("UNDEFINED" if any_undefined else "NO-GO")
    print(f"\n{'=' * 100}", flush=True)
    print(f"  RANK-14 QUESTION-ROUTE-SELECTION WTA VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    for r in results:
        print(f"    seed {r['seed']}: {r['verdict']:9s} parity={r['parity_rate']:.3f} "
              f"ambiguous_ok={r['ambiguous']['ok']} full_lesion={r['full_lesion']['all_generic']} "
              f"per_pathway={r['per_pathway_independent']} sweep_flips={r['sweep_flips']} "
              f"det={r['seed_deterministic']}", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
