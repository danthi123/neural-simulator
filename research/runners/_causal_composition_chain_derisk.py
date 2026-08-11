"""EMERGENT CAUSAL COMPOSITION de-risk -- the INTEGRATION #5 follow-on (named per THE LAW).

INTEGRATION #5 (`2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md`) made turn 4 of the live
14-turn chat ("why did the dog go east?") HONESTLY DISCLOSE that the brain has no causal faculty: it CONFIRMS the
stored fact via the no-confab moat (`comp.query_patient("dog","go") -> east`) and declines to invent a reason. The
finding NAMED the follow-on: the truly-emergent answer would COMPOSE stored facts into a grounded causal CHAIN --
  "dog goes east"  +  "dog looks at river"  +  [river is east]  =>  "to reach the river".

THIS de-risk asks: given the brain's stored SVO facts (the #6/#7 corpus-learned kb), can it CHAIN a grounded
goal-directed causal answer WITHOUT confabulating -- and does the no-confab moat correctly ABSTAIN (falling back to
the #5 disclaimer) whenever the facts do NOT support a chain?

THE COMPOSITION (a goal-directed JOIN over the shared entity, EVERY edge a `query_patient` moat read):
    HOP 1  dir      = comp.query_patient(agent, motion_verb)     # where the agent moved  (a stored direction)
    HOP 2  obj      = comp.query_patient(agent, goal_verb)       # the agent's OWN goal-object  (SHARED-ENTITY join)
    HOP 3  obj_dir  = comp.query_patient(obj,   locative_verb)   # where that object is located
    COMPOSE  iff  dir == obj_dir   ->   "the agent moves <dir> to reach the <obj>"
    else  ABSTAIN  ->  fall back to the #5 honest disclaimer  (`_honest_causal_answer`).  NEVER invent a link.

Because every link is a `query_patient` read (abstain -> None), the composed answer asserts ONLY moat-confirmed
facts: 0 confabulation BY CONSTRUCTION. The de-risk's teeth are the moat's DISCRIMINATION -- it must compose only
when the chain genuinely grounds, and abstain on the two confab traps a "why" invites:
  * GOAL-SHORTCUT trap: the agent HAS a known goal, but moved in a DIFFERENT direction than the goal's location
    ("why did the dog run north?" -- river is the dog's goal but river is EAST, the dog ran NORTH). A naive
    reasoner answers "to reach the river"; the moat must ABSTAIN (dir_mismatch).
  * SPATIAL-SHORTCUT trap: an object IS located in the motion direction, but the moving agent never looked at it
    ("why did the fish go east?" -- the river is east, but the fish has no stored goal). A naive spatial lookup
    answers "to reach the river"; the moat must ABSTAIN (no_goal).

HONEST SCOPE (per THE LAW + docs/TERMS.md). The chaining reliability itself is ALREADY de-risked GO
(`2026-06-17-multihop-query-chain-GO.md`: role-structured `query_chain`, moat at every hop; `2026-06-27-tier2.2-
chain-of-thought-GO.md`). This de-risk does NOT re-derive that. Its DELTA is (a) the CAUSAL / goal-directed
composition SHAPE (a 2-fact join on the shared agent + a spatial-location grounding, not a linear patient->agent
chain), (b) the moat's DISCRIMINATION on the two "why"-specific confab traps, and (c) graduating the #5 turn-4
disclaimer to a moat-verified composed reason when the grounding exists. The composition POLICY (the motion+goal+
spatial join) is a DECLARED HOST SCAFFOLD -- same status as `query_chain`'s caller-supplied action list and the #5
`why`+known-cue trigger. The toy substrate stores FLAT `(agent,action)->patient` associations with NO relational /
causal / spatial graph; the join is host-orchestrated. The named neural successor (the honest negative that launches
the next arc): a LEARNED relational/spatial code (a factorised relation binder, TEM-style; or the co-occurrence
stream cortex) so the causal chain EMERGES from the substrate rather than host orchestration -- at which point the
#5 disclaimer graduates from "I have not learned causes" to "the dog goes east to reach the river".

What IS substrate/mechanism: every FACT in every chain is a spiking RF-VSA unbind + cleanup (`query_patient`), and
the moat's abstain-vs-compose DECISION is driven entirely by those reads.

DISCIPLINE: SIM_BACKEND=numpy substrate, reuse-by-import (RFPhasorComposer + build_one_brain + the #5
`_honest_causal_answer`), NO `sim/` edit, cfg.seed (build_one_brain), additive (a NEW runner).

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._causal_composition_chain_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/lanes/stageA/causal/causal_composition_chain_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from research.runners.rf_phasor_composer import RFPhasorComposer, DEFAULT_VOCAB  # noqa: E402
from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners._conversation_turing_test_derisk import _honest_causal_answer, _PRESENT3  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE TOY CAUSAL WORLD -- flat SVO facts, all words in-vocab. Three fact TYPES compose a goal-directed reason:
#   motion  (agent, motion_verb) -> direction        goal (agent, look) -> object        spatial (object, at) -> dir
# The extra vocab {at, bird, hill, fish} supplies the spatial relation + two subjects that exercise the abstain
# reasons no_spatial (bird's goal 'hill' has no stored location) and no_goal (fish has no stored goal).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
BASE_VOCAB = sorted(set(DEFAULT_VOCAB) | {"at", "bird", "hill", "fish"})
DIRECTIONS = {"north", "south", "east", "west"}
GOAL_VERBS = ("look",)
LOCATIVE_VERBS = ("at",)

WORLD_FACTS = [
    # motion (agent, motion_verb) -> direction
    ("dog", "go", "east"), ("dog", "run", "north"),
    ("cat", "go", "west"), ("cat", "run", "south"),
    ("bird", "go", "north"),
    ("fish", "go", "east"),
    # goal (agent, look) -> object   (the SHARED-ENTITY hop)
    ("dog", "look", "river"), ("cat", "look", "apple"), ("bird", "look", "hill"),
    # spatial (object, at) -> direction   (the grounding hop; NO (hill,at), NO (fish,look) on purpose)
    ("river", "at", "east"), ("apple", "at", "west"),
]

# (agent, motion, expected_supported, expected_obj, expected_reason). The 6 abstain rows span all four reasons +
# the two confab traps (dir_mismatch = goal-shortcut trap; no_goal on 'fish go' = spatial-shortcut trap).
GRID = [
    ("dog", "go",   True,  "river", "grounded"),
    ("cat", "go",   True,  "apple", "grounded"),
    ("dog", "run",  False, None,    "dir_mismatch"),    # goal-shortcut trap: river is dog's goal but @east, dog ran north
    ("cat", "run",  False, None,    "dir_mismatch"),    # goal-shortcut trap
    ("fish", "go",  False, None,    "no_goal"),         # spatial-shortcut trap: river@east but fish has no goal
    ("bird", "go",  False, None,    "no_spatial"),      # bird's goal 'hill' has no stored location
    ("dog", "come", False, None,    "unstored_motion"),
    ("cat", "stop", False, None,    "unstored_motion"),
]

# Untaught cues that MUST abstain (raw query_patient -> None): the 0-false-accept moat battery.
MOAT_BATTERY = [("dog", "stop"), ("cat", "come"), ("fish", "look"), ("hill", "at"),
                ("river", "go"), ("bird", "run"), ("apple", "go"), ("dog", "at")]


def _present(verb):
    return _PRESENT3.get(verb, verb + "s")


def _compose_answer(agent, motion_verb, dir_, goal_verb, obj, loc_verb):
    """The composed causal chain, asserting ONLY the three moat-confirmed facts + their entailed goal-directed
    conclusion. Same read-out status as the #5 disclaimer (a functional read-out, never a phenomenal claim)."""
    return (
        "I know the %s %s %s -- that fact is stored, and my no-confab moat confirms it ((%s, %s) -> %s). This "
        "time I can say WHY, because two more stored facts COMPOSE into a grounded reason: (%s, %s) -> %s, and "
        "(%s, %s) -> %s. So the %s %s %s to reach the %s. Every link in that chain is moat-confirmed -- I "
        "composed it from what I stored, I did not invent it."
        % (agent, _present(motion_verb), dir_, agent, motion_verb, dir_,
           agent, goal_verb, obj, obj, loc_verb, dir_,
           agent, _present(motion_verb), dir_, obj)
    )


def _every_edge_moat_read(comp, chain):
    """Structural anti-cheat: EVERY triple in a composed chain must read back via query_patient (the moat). A
    composed answer whose links are not all moat reads would be a confabulation -- this proves it is not."""
    return all(comp.query_patient(a, v) == p for (a, v, p) in chain)


def compose_causal_reason(comp, agent, motion_verb):
    """Compose a goal-directed causal reason for 'why did AGENT MOTION?' ENTIRELY from query_patient moat reads.
    Returns a dict; supported=True only when all three hops ground and dir==obj_dir, else supported=False with the
    most-specific abstain reason. NEVER invents a link (any None / mismatch -> abstain)."""
    dir_ = comp.query_patient(agent, motion_verb)                       # HOP 1
    if dir_ is None:
        return {"supported": False, "reason": "unstored_motion", "dir": None, "obj": None,
                "obj_dir": None, "chain": None, "answer": None}
    if dir_ not in DIRECTIONS:
        return {"supported": False, "reason": "nondirectional_motion", "dir": dir_, "obj": None,
                "obj_dir": None, "chain": None, "answer": None}
    best_reason, best_obj, best_obj_dir = "no_goal", None, None
    for gv in GOAL_VERBS:
        obj = comp.query_patient(agent, gv)                             # HOP 2 (shared-entity goal)
        if obj is None:
            continue
        for lv in LOCATIVE_VERBS:
            obj_dir = comp.query_patient(obj, lv)                       # HOP 3 (grounding)
            if obj_dir is None:
                if best_reason == "no_goal":
                    best_reason, best_obj, best_obj_dir = "no_spatial", obj, None
                continue
            if obj_dir == dir_:                                         # the chain grounds -> COMPOSE
                chain = [(agent, motion_verb, dir_), (agent, gv, obj), (obj, lv, obj_dir)]
                return {"supported": True, "reason": "grounded", "dir": dir_, "obj": obj, "obj_dir": obj_dir,
                        "chain": chain, "answer": _compose_answer(agent, motion_verb, dir_, gv, obj, lv)}
            best_reason, best_obj, best_obj_dir = "dir_mismatch", obj, obj_dir
    return {"supported": False, "reason": best_reason, "dir": dir_, "obj": best_obj,
            "obj_dir": best_obj_dir, "chain": None, "answer": None}


def _fresh_composer(seed, facts, vocab=BASE_VOCAB):
    comp = RFPhasorComposer(seed=int(seed), D=128, vocab=sorted(set(vocab)))
    for (a, v, p) in facts:
        comp.store(a, v, p)
    return comp


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 0 -- the core de-risk on the faithful standalone composer (the same RF-VSA the live chat's comp wraps).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def tier0_grid(seed):
    comp = _fresh_composer(seed, WORLD_FACTS)
    rows, supported_correct, abstain_correct = [], 0, 0
    false_accepts, every_edge_ok, confab = 0, 0, 0
    goal_shortcut_fa, spatial_shortcut_fa = 0, 0
    n_supported_expected = sum(1 for r in GRID if r[2])
    for (agent, motion, exp_sup, exp_obj, exp_reason) in GRID:
        res = compose_causal_reason(comp, agent, motion)
        got_sup = bool(res["supported"])
        # abstain answer = the #5 honest disclaimer (byte-identical reuse) over the moat-read motion patient
        answer = res["answer"] if got_sup else _honest_causal_answer(agent, motion, res["dir"])
        if exp_sup:
            if got_sup and res["obj"] == exp_obj:
                supported_correct += 1
            if got_sup and _every_edge_moat_read(comp, res["chain"]):
                every_edge_ok += 1
            # confab check on a SUPPORTED answer: every asserted triple must be a moat read
            if got_sup and not _every_edge_moat_read(comp, res["chain"]):
                confab += 1
        else:
            if not got_sup:
                abstain_correct += 1
            else:                                    # a FALSE-ACCEPT: composed a chain the facts do not support
                false_accepts += 1
                confab += 1
                if exp_reason == "dir_mismatch":
                    goal_shortcut_fa += 1
                if exp_reason == "no_goal":
                    spatial_shortcut_fa += 1
        rows.append({"agent": agent, "motion": motion, "expected_supported": exp_sup, "expected_obj": exp_obj,
                     "expected_reason": exp_reason, "got_supported": got_sup, "got_reason": res["reason"],
                     "got_obj": res["obj"], "dir": res["dir"], "obj_dir": res["obj_dir"],
                     "chain": res["chain"], "answer": answer})
    # ---- moat battery: untaught cues must abstain (raw query_patient -> None) ----
    battery_false_accepts = sum(1 for (a, v) in MOAT_BATTERY if comp.query_patient(a, v) is not None)
    # ---- permuted-spatial control: derange the (object,at,dir) grounding -> the originally-grounded chains must
    #      collapse to abstain (the causal chain READS the stored grounding; it is not a hardcoded link). ----
    perm_facts = [(a, v, p) for (a, v, p) in WORLD_FACTS if v not in LOCATIVE_VERBS]
    perm_facts += [("river", "at", "west"), ("apple", "at", "east")]     # derangement of {river:east, apple:west}
    comp_perm = _fresh_composer(seed, perm_facts)
    perm_still_supported = sum(1 for (a, m, es, eo, er) in GRID
                               if es and compose_causal_reason(comp_perm, a, m)["supported"])
    # ---- permuted-spatial POSITIVE control: a derangement that GROUNDS a different query proves the chain follows
    #      the data both directions (river@north makes "why dog run north" -> supported "to reach the river"). ----
    perm2 = [(a, v, p) for (a, v, p) in WORLD_FACTS if v not in LOCATIVE_VERBS]
    perm2 += [("river", "at", "north"), ("apple", "at", "west")]
    comp_perm2 = _fresh_composer(seed, perm2)
    perm2_dogrun = compose_causal_reason(comp_perm2, "dog", "run")
    perm2_doggo = compose_causal_reason(comp_perm2, "dog", "go")
    perm2_positive_ok = bool(perm2_dogrun["supported"] and perm2_dogrun["obj"] == "river"
                             and not perm2_doggo["supported"])

    go = (supported_correct == n_supported_expected and abstain_correct == (len(GRID) - n_supported_expected)
          and false_accepts == 0 and every_edge_ok == n_supported_expected and confab == 0
          and battery_false_accepts == 0 and perm_still_supported == 0 and perm2_positive_ok)
    return {
        "seed": int(seed), "n_grid": len(GRID), "n_supported_expected": n_supported_expected,
        "supported_correct": supported_correct, "abstain_correct": abstain_correct,
        "false_accepts": false_accepts, "every_edge_moat_ok": every_edge_ok, "confab_count": confab,
        "goal_shortcut_false_accepts": goal_shortcut_fa, "spatial_shortcut_false_accepts": spatial_shortcut_fa,
        "moat_battery_false_accepts": battery_false_accepts, "moat_battery_n": len(MOAT_BATTERY),
        "permuted_spatial_still_supported": perm_still_supported, "permuted_positive_ok": perm2_positive_ok,
        "GO": bool(go), "rows": rows,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 1 -- graduate the #5 turn-4 disclaimer on the LIVE co-resident one-brain composer. WITH the spatial facts
# the composed chain fires ("to reach the river"); WITHOUT them the #5 honest disclaimer is the correct fallback.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
CURATED = [("dog", "run", "north"), ("cat", "run", "south"), ("dog", "go", "east"),
           ("cat", "go", "west"), ("dog", "look", "river"), ("cat", "look", "apple")]
SPATIAL = [("river", "at", "east"), ("apple", "at", "west")]


def tier1_live_graduation(seed):
    def _build_and_store(extra):
        bridge, comp, idx, snap = SA.build_one_brain(int(seed), with_faculties=True,
                                                     co_resident_affect_ladder=True, vocab=BASE_VOCAB)
        for (a, v, p) in CURATED + list(extra):
            comp.store(a, v, p)
        return comp

    # WITH spatial grounding -> the #5 turn-4 query graduates to a composed causal chain
    comp_g = _build_and_store(SPATIAL)
    res_g = compose_causal_reason(comp_g, "dog", "go")
    with_reply = res_g["answer"] if res_g["supported"] else _honest_causal_answer("dog", "go", res_g["dir"])
    with_confab = 0 if (res_g["supported"] and _every_edge_moat_read(comp_g, res_g["chain"])) else \
        (0 if not res_g["supported"] else 1)

    # WITHOUT spatial grounding (the #5 world) -> abstain -> the #5 honest disclaimer, byte-identical
    comp_a = _build_and_store([])
    res_a = compose_causal_reason(comp_a, "dog", "go")
    without_reply = res_a["answer"] if res_a["supported"] else _honest_causal_answer("dog", "go", res_a["dir"])
    baseline_disclaimer = _honest_causal_answer("dog", "go", "east")

    graduated = bool(res_g["supported"] and res_g["obj"] == "river"
                     and not res_a["supported"] and without_reply == baseline_disclaimer
                     and with_confab == 0)
    return {
        "seed": int(seed),
        "with_spatial": {"supported": res_g["supported"], "obj": res_g["obj"], "chain": res_g["chain"],
                         "reply": with_reply, "confab": with_confab},
        "without_spatial": {"supported": res_a["supported"], "reason": res_a["reason"], "reply": without_reply,
                            "matches_5_disclaimer": bool(without_reply == baseline_disclaimer)},
        "graduated_when_supported_else_disclaimer": graduated,
    }


def run_seed(seed, do_tier1=True):
    t0 = time.time()
    t0res = tier0_grid(seed)
    t1res = tier1_live_graduation(seed) if do_tier1 else None
    return {"seed": int(seed), "tier0": t0res, "tier1": t1res,
            "GO": bool(t0res["GO"] and (t1res is None or t1res["graduated_when_supported_else_disclaimer"])),
            "elapsed_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-tier1", action="store_true", help="Tier-0 only (skip the slow build_one_brain live check)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    per_seed = [run_seed(s, do_tier1=not args.no_tier1) for s in seeds]
    n_go = sum(1 for r in per_seed if r["GO"])

    # ---- verdict with the preconditions that EARN it (tools.verdict.Verdict -> a `preconditions` block the
    #      verdict-preconditions gate enforces travels with the GO). Every check is measured over ALL seeds. ----
    t0 = [r["tier0"] for r in per_seed]
    tier1_ran = any(r["tier1"] is not None for r in per_seed)
    v = Verdict("emergent causal composition chain (6-seed)")
    v.require("tier0 GO on every seed", all(x["GO"] for x in t0), expect=True)
    v.require("supported chains correct (2/2) every seed",
              all(x["supported_correct"] == x["n_supported_expected"] for x in t0), expect=True)
    v.require("abstains correct (6/6) every seed",
              all(x["abstain_correct"] == (x["n_grid"] - x["n_supported_expected"]) for x in t0), expect=True)
    v.require("moat false-accepts == 0 every seed", all(x["false_accepts"] == 0 for x in t0), expect=True)
    v.require("confabulations == 0 every seed", all(x["confab_count"] == 0 for x in t0), expect=True)
    v.require("moat-battery false-accepts == 0 every seed",
              all(x["moat_battery_false_accepts"] == 0 for x in t0), expect=True)
    # anti-cheat control: permuting the (object,at,dir) grounding must CHANGE the outcome (collapse the chains).
    treat = sum(x["supported_correct"] for x in t0)
    ctrl = sum(x["permuted_spatial_still_supported"] for x in t0)
    # ATTRIBUTION (tools.lab): what fraction of the composed chains is owed to the STORED spatial grounding? The
    # permuted control deranges (object,at)->direction; (treat-ctrl)/treat == 1.0 means the chains are 100%
    # attributable to the grounding, not to a hardcoded link (whose the difference was, not just both arms measured).
    grounding_attribution = attributable_to("composed chains attributable to the stored spatial grounding",
                                            treatment_value=float(treat), control_value=float(ctrl))
    v.control("permuted-spatial collapses the chain", treatment=treat, control=ctrl,
              note="derange river/apple locations -> originally-grounded chains must abstain")
    v.require("permuted-positive moves the supported set with the data (every seed)",
              all(x["permuted_positive_ok"] for x in t0), expect=True)
    if tier1_ran:
        v.require("tier1 graduates the #5 turn-4 disclaimer (composed when supported, else #5 fallback)",
                  all(r["tier1"]["graduated_when_supported_else_disclaimer"]
                      for r in per_seed if r["tier1"] is not None), expect=True)
    v.disabled("spiking generator mouth",
               why="CPU numpy run; the grounded CONTENT is the RF-VSA query_patient read the mouth would render")
    v.disabled("emergent relational/spatial code",
               why="the motion+goal+spatial JOIN policy + the (object,at)->direction grounding are DECLARED HOST "
                   "SCAFFOLDS; the named neural successor is a learned relational/spatial code")
    verdict = v.decide(go=(n_go == len(seeds)), verbose=False)

    agg = {
        "seeds": seeds, "n_seeds": len(seeds), "n_GO": n_go, "GO": bool(n_go == len(seeds)),
        "status": verdict["status"], "preconditions": verdict["preconditions"],
        "disabled_processes": verdict["disabled_processes"], "undefined_reasons": verdict["undefined_reasons"],
        "grounding_attribution": grounding_attribution,
        "tier0_all_go": all(r["tier0"]["GO"] for r in per_seed),
        "tier0_supported_correct": [r["tier0"]["supported_correct"] for r in per_seed],
        "tier0_abstain_correct": [r["tier0"]["abstain_correct"] for r in per_seed],
        "tier0_false_accepts": [r["tier0"]["false_accepts"] for r in per_seed],
        "tier0_confab": [r["tier0"]["confab_count"] for r in per_seed],
        "tier0_moat_battery_fa": [r["tier0"]["moat_battery_false_accepts"] for r in per_seed],
        "tier0_permuted_still_supported": [r["tier0"]["permuted_spatial_still_supported"] for r in per_seed],
        "tier1_graduated": [None if r["tier1"] is None else r["tier1"]["graduated_when_supported_else_disclaimer"]
                            for r in per_seed],
        "per_seed": per_seed,
    }
    _verbose_keys = {"per_seed", "preconditions", "disabled_processes", "undefined_reasons"}
    print(json.dumps({k: v for k, v in agg.items() if k not in _verbose_keys}, indent=2))
    for r in per_seed:
        t0 = r["tier0"]
        print("  seed %d: GO=%s sup=%d/%d abstain=%d/%d fa=%d confab=%d battery_fa=%d perm_still=%d tier1=%s (%.1fs)"
              % (r["seed"], r["GO"], t0["supported_correct"], t0["n_supported_expected"], t0["abstain_correct"],
                 t0["n_grid"] - t0["n_supported_expected"], t0["false_accepts"], t0["confab_count"],
                 t0["moat_battery_false_accepts"], t0["permuted_spatial_still_supported"],
                 None if r["tier1"] is None else r["tier1"]["graduated_when_supported_else_disclaimer"],
                 r["elapsed_s"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2)
        print("wrote", args.out)
    return 0 if agg["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
