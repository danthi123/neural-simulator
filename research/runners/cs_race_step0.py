#!/usr/bin/env python3
"""cs-race step 0 (PACKET.md line 64): the default-regime probe. No build.

QUESTION: at the production default (BRAIN_RECONSOLIDATION on, flag-off
world) does a second patient taught for one cue APPEND a new store slot or
REWRITE in place? The record conflicts:
  * host_share.py:36 and load_bearing_fraction.py:1392-1396 say
    reconsolidation rewrites in place, so a second patient cannot be taught
    through chat (BRAIN_RECONSOLIDATION=0 departs from default);
  * research/findings/2026-10-12-validation-battery-prereg.md:394-397 and
    the dev-7 re-tell smoke (tools/c5_battery_score.py:135) show BOTH
    re-tells APPENDED (lynx chase frog at kb_idx 10, lynx chase vole kept
    at 5; no in-slot rewrite on this route).

METHOD (one dev seed, report only): drive the battery's RE-TELL session
turn-for-turn through the production brain_chat handler at default env
(mirrors onebrain_regression_battery.py C5B_RETELL: F1..F4 taught, t5 re-
tells F2 verbatim, t6 teaches the second patient for F1's cue), then read
the idle turn's candidate record (the same read c5_battery_score.retell_read
uses: candidates carry fact, kb_idx, t_enc). Verdict for cue (lynx, chase):
APPEND = both vole and frog slots present; REWRITE = only frog, at vole's
old slot; UNRESOLVED = the idle record did not land. The answer decides
whether the cs-race compete reversal counts at the default or only under
the declared BRAIN_RECONSOLIDATION=0 departure. This changes nothing.
"""
import argparse
import json
import os
import sys
import time

REPO = "/home/dant123/Projects/neural-simulator"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

C5RE_TURNS = [
    "the lynx chases the vole",      # F1 taught
    "the heron hunts the frog",      # F2 taught
    "the yak eats the grass",        # F3 taught
    "the crow chases the wren",      # F4 taught
    "the heron hunts the frog",      # t5: F2 re-told verbatim
    "the lynx chases the frog",      # t6: F1's cue re-told with a SECOND patient
]
CUE = ("lynx", "chase")


def _si(body):
    """self_initiated record (mirrors tools/c5_battery_score._si, inlined:
    a research probe must not depend on a scorer module)."""
    si = (body or {}).get("self_initiated")
    return si if isinstance(si, dict) else {}


def _ir(body):
    ir = _si(body).get("idle_replay")
    return ir if isinstance(ir, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--session", default="cs0")
    ap.add_argument("--out", default="research/findings/raw/_cs_race/step0/seed7.json")
    args = ap.parse_args()

    # Default regime: BRAIN_RECONSOLIDATION must be UNSET (the shipped
    # default is on). Record what we ran under; refuse a caller override.
    recons = os.environ.get("BRAIN_RECONSOLIDATION")
    if recons is not None:
        print("REFUSE: BRAIN_RECONSOLIDATION=%r set; step 0 measures the"
              " production default (unset)" % recons)
        return 2
    cs_flag = os.environ.get("BRAIN_CS_RACE")   # no such flag exists yet (step 2); record it
    os.environ["BRAIN_CHAT_SEED"] = str(args.seed)   # BEFORE webapp import (brain builds lazily)
    os.environ.setdefault("SIM_BACKEND", "numpy")

    from tools.lab import assert_backend
    from tools.verdict import Verdict
    assert_backend("numpy", note="cs-race step 0 default-regime probe")
    from webapp.server import BrainChatRequest, brain_chat

    t0 = time.time()
    responses = []
    for i, msg in enumerate(C5RE_TURNS + [""]):     # final "" = the idle turn
        req = BrainChatRequest(session=args.session, message=msg, brain="tiny-demo",
                               renderer="stub", rich=False, reset=(i == 0))
        r = brain_chat(req)
        responses.append(json.loads(r.body))

    idle_body = responses[-1]
    ir = _ir(idle_body)
    cands = [c for c in (ir.get("candidates") or []) if isinstance(c, dict)]
    slots = [{"kb_idx": int(c.get("kb_idx") or -1),
              "t_enc": int(c.get("t_enc") or 0),
              "fact": list(c.get("fact") or [])}
             for c in cands if list((c.get("fact") or [])[:2]) == list(CUE)]
    patients = sorted(str(s["fact"][2]) for s in slots if len(s["fact"]) == 3)
    if "vole" in patients and "frog" in patients:
        verdict = "APPEND"
    elif patients == ["frog"]:
        verdict = "REWRITE"
    else:
        verdict = "UNRESOLVED"

    v = Verdict("cs-race step 0: does a second patient APPEND or REWRITE "
                "at the production default?")
    v.require("idle organ record present (path idle-replay)",
              _si(idle_body).get("path") == "idle-replay")
    v.require("candidates listed", bool(cands))
    v.require("cue slots resolved", verdict != "UNRESOLVED")
    v.require("backend numpy", True)
    # decide() BEFORE to_dict(): a verdict must be earned before it is written.
    ok = v.decide(verdict in ("APPEND", "REWRITE"))

    out = {
        "probe": "cs_race_step0",
        "seed": args.seed,
        "session": args.session,
        "backend": "numpy",
        "device": "cpu",
        "regime": {"BRAIN_RECONSOLIDATION": "unset (production default = on)",
                   "BRAIN_CS_RACE": cs_flag, "renderer": "stub"},
        "turns": len(responses),
        "turn_errors": [i for i, r in enumerate(responses)
                        if not isinstance(r, dict) or r.get("_error")],
        "idle_path": _si(idle_body).get("path"),
        "cue": list(CUE),
        "cue_slots": sorted(slots, key=lambda s: s["kb_idx"]),
        "patients": patients,
        "all_candidates": [{"kb_idx": int(c.get("kb_idx") or -1),
                            "fact": list(c.get("fact") or [])} for c in cands],
        "verdict": verdict,
        "claims": {
            "host_share_py36_lbf_1392": "rewrites in place (second patient unteachable)",
            "prereg_2026_10_12_line394": "APPENDED on this route (production gap of B.2)"},
        "Verdict": v.to_dict(),
        "preconditions": {
            "env": "production default (BRAIN_RECONSOLIDATION unset), "
                   "renderer=stub, SIM_BACKEND=numpy",
            "route": "battery RE-TELL session C5B_RETELL turns via brain_chat",
            "read": "idle-turn candidate record (c5_battery_score.retell_read path)",
            "one_dev_seed": args.seed},
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    from research.runners import declare_output
    declare_output(out_path)
    print("verdict=%s cue_slots=%s patients=%s wall=%.1fs -> %s"
          % (verdict, out["cue_slots"], patients, out["wall_s"], out_path),
          flush=True)
    # GO = the route answered cleanly (an UNRESOLVED read = NO-GO).
    return 0 if ok.get("go") else 1


if __name__ == "__main__":
    sys.exit(main())
