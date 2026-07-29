#!/usr/bin/env python3
"""Which ROADMAP LANES is the machine actually serving right now? Exit 1 (loud) on monoculture.

WHY THIS EXISTS (2026-07-29, owner-flagged twice in one day). Parallelization was fixed with a dispatcher:
GPU at 100%, queue stocked, no idle lane. It looked correct from every angle — and **every job in the queue
served ONE lane (H · Memory)** while lane **F · gap#4**, which the master roadmap calls *"the single
load-bearing dependency (the crux the whole roadmap pivots on)"*, had ZERO allocation, and lane **E ·
Language** (`[CPU]`, "disjoint from A/B/C") sat unqueued beside 36 idle pool cores. The first lane-E runner
dispatched returned a GO in 40 seconds.

**A full queue and a busy GPU look exactly like good prioritization from the inside.** Queue DEPTH was
already monitored; lane COVERAGE was not. This closes that gap the same way: a check that fails loudly.

    .venv/bin/python tools/lane_check.py            # report + exit 1 on monoculture
    .venv/bin/python tools/lane_check.py --quiet    # one line, for the heartbeat

The lane map is keyword→lane, derived from the roadmap's own parallelization table (§ lanes). It is
deliberately coarse: the point is to notice "everything is one lane", not to classify perfectly.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lane -> (compute tag, substrings that identify a job as serving that lane)
LANES = {
    "A · Affect":        ("CPU",  ["affect", "appraisal", "valence", "liking", "_dr2", "emotion"]),
    "B · Curiosity":     ("CPU",  ["curiosity", "novelty", "question_gen", "_dr1", "learning_progress"]),
    "C · Self/Workspace": ("CPU/GPU", ["self_schema", "meta_d", "false_belief", "workspace", "_dr3", "_p1_2", "tom"]),
    "D · Perception":    ("CPU/GPU", ["v1_selforg", "_b1_", "visual", "retina", "gabor", "v2_", "_it_", "nav_"]),
    "E · Language":      ("CPU",  ["emerge6", "emerge7", "construction", "morpholog", "lexicon", "grammar",
                                    "comprehension", "producer", "confidence_gate"]),
    "F · gap#4 CRUX":    ("GPU",  ["gap4", "deep_credit", "selfpredict", "microcircuit", "bdsp", "eprop",
                                    "burst", "credit"]),
    "G · Teacher-loop":  ("GPU",  ["develop_loop", "develop_run", "teacher", "curriculum", "_p2_1", "_p3_1"]),
    "H · Memory":        ("GPU/CPU", ["consol", "sparse_distributed", "concept_pool", "replay", "ca3", "ca1",
                                       "hippo", "btsp", "schaffer", "engram"]),
}
CRUX = "F · gap#4 CRUX"


def classify(cmd: str):
    c = cmd.lower()
    hits = [ln for ln, (_, keys) in LANES.items() if any(k in c for k in keys)]
    return hits or ["(unclassified)"]


def _queue_jobs():
    p = os.path.join(ROOT, "research/queue/gpu.queue")
    if not os.path.exists(p):
        return []
    return [l for l in open(p).read().split("\n") if l.strip() and not l.strip().startswith("#")]


def _running():
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    return [l for l in out.split("\n") if "research.runners" in l and "grep" not in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    jobs = _queue_jobs()
    run = _running()
    tally = {}
    for src, items in (("running", run), ("queued", jobs)):
        for it in items:
            for ln in classify(it):
                tally.setdefault(ln, {"running": 0, "queued": 0})[src] += 1

    active = {k: v for k, v in tally.items() if k != "(unclassified)"}
    n_lanes = len(active)
    crux = tally.get(CRUX, {"running": 0, "queued": 0})
    crux_total = crux["running"] + crux["queued"]
    cpu_lanes = [ln for ln, (tag, _) in LANES.items() if tag.startswith("CPU")]
    cpu_active = [ln for ln in cpu_lanes if ln in active]

    alarms = []
    if n_lanes <= 1 and (jobs or run):
        alarms.append("LANE-MONOCULTURE(all work serves %s)" % (list(active) or ["nothing"])[0])
    if crux_total == 0:
        alarms.append("CRUX-UNSERVED(F/gap#4 has 0 jobs; roadmap calls it 'the must-solve core')")
    # A FALSE ALARM IS AS CORROSIVE AS A MISSED ONE (this tool's own first lesson). Pool jobs finish in
    # SECONDS-to-minutes, so "no CPU lane running right now" is usually SUCCESS, not neglect — alarming on
    # instantaneous idleness would fire every cycle and train the reader to ignore it. Alarm instead on
    # STALENESS: no CPU-lane work dispatched for STALE_MIN. Any dispatch touches the marker.
    STALE_MIN = 45
    marker = os.path.join(ROOT, "research/queue/.last_cpu_dispatch")
    if cpu_active:
        try:
            open(marker, "w").write("dispatched")
        except Exception:
            pass
    age_min = None
    if os.path.exists(marker):
        age_min = (time.time() - os.path.getmtime(marker)) / 60.0
    if not cpu_active and (age_min is None or age_min > STALE_MIN):
        alarms.append("CPU-LANES-STALE(%s; 5 disjoint CPU lanes cost nothing beside GPU work)"
                      % ("never dispatched" if age_min is None else "%.0f min since last dispatch" % age_min))

    if args.quiet:
        print("lanes=%d crux=%d%s" % (n_lanes, crux_total, (" -- ACT: " + " ".join(alarms)) if alarms else ""))
        return 1 if alarms else 0

    print("=" * 74)
    print("ROADMAP LANE COVERAGE  (%d running, %d queued)" % (len(run), len(jobs)))
    print("=" * 74)
    for ln, (tag, _) in LANES.items():
        t = tally.get(ln, {"running": 0, "queued": 0})
        mark = "  <-- CRUX" if ln == CRUX else ""
        flag = "" if (t["running"] or t["queued"]) else "   (unserved)"
        print("  %-20s [%-8s] running=%-3d queued=%-3d%s%s" % (ln, tag, t["running"], t["queued"], flag, mark))
    if "(unclassified)" in tally:
        print("  %-20s %s" % ("(unclassified)", tally["(unclassified)"]))
    print()
    if alarms:
        for a in alarms:
            print("  ⚠️  %s" % a)
        print()
        print("  FIX: stock research/queue/gpu.queue across lanes, and dispatch a CPU lane to the pool.")
        print("       The 5 CPU-tagged lanes (A/B/C/D/E) are explicitly disjoint — they cost nothing")
        print("       alongside GPU work, and leaving them unqueued is why 36 cores idled for a day.")
        return 1
    print("  OK — work spans %d lanes and the crux is served." % n_lanes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
