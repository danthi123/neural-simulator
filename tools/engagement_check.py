#!/usr/bin/env python3
"""Did the mechanism actually ENGAGE? Scan a result artifact for the signatures of a void arm.

WHY (2026-07-29 — SIX instances in one day, each a different proximate cause, all the same error):
  1. a metaplasticity toy whose threshold was unreachable, so a fallback branch did all the work;
  2. a saturation arm where the soft bound never bound (`sat_frac` 0.000 in every cell);
  3. a DG lesion on a gate that was never declared (drift exactly +0.000000 = a "perfect" freeze);
  4. the crux running 47 min on the CPU while every liveness indicator read healthy;
  5. alpha sweeps whose control sat at ceiling, so no arm could rank;
  6. on-bridge V1 with `v1_firing_rate_mean` 0.0007 — selectivity "measured" in silent neurons.
Every one returned a NUMBER while the thing being measured never happened. `tools/lab.py` has helpers for
this, but they must be imported and remembered; this runs on the ARTIFACT afterwards, so it catches the
cases where the probe was written without them.

    .venv/bin/python tools/engagement_check.py <result.json> [more.json ...]

Exit 1 if any artifact shows a void signature. It flags, it does not adjudicate — a flagged number may
still be real (a genuinely silent population IS the finding sometimes), but it must be looked at, not
reported.
"""
from __future__ import annotations

import io
import json
import os
import sys

# metric-name fragments whose near-zero value means "nothing happened"
ACTIVITY = ("firing_rate", "fire_rate", "active_frac", "n_active", "spike", "sat_frac", "n_blocked",
            "engaged", "n_registered", "dw", "drift", "n_committed", "blocks")
# names that are accuracies/scores (used for the exact-chance test)
SCORE = ("acc", "decode", "recall", "render", "top1", "correct", "score", "cue")
# CONTROL metrics are SUPPOSED to be zero — a permuted/lesioned/no-corpus arm scoring 0 is the result
# working, not a void arm. Flagging them made the first version fire on a known-GOOD artifact, which is
# how a checker trains its reader to ignore it (the day's other monitors did this twice).
CONTROL = ("nocorpus", "no_corpus", "permut", "perm_", "shuffle", "shuf", "scramble", "lesion", "cross_",
           "derange", "chance", "floor", "abstain", "control", "_ctrl", "moat_calls")


def _flat(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, pre + k + "."))
    elif isinstance(o, list):
        if o and all(isinstance(x, (int, float)) for x in o):
            out[pre.rstrip(".")] = o
    elif isinstance(o, (int, float, bool)):
        out[pre.rstrip(".")] = o
    return out


def check(path):
    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except Exception as e:
        print("  %s: UNREADABLE (%s)" % (os.path.basename(path), e))
        return ["unreadable"]
    flat = _flat(d)
    flags = []

    for k, v in flat.items():
        kl = k.lower()
        val = v if isinstance(v, (int, float)) else None
        if val is None:
            continue
        if any(c in kl for c in CONTROL):
            continue  # a control at zero is the control WORKING
        if any(a in kl for a in ACTIVITY) and abs(val) < 1e-3 and "frac" not in kl.split(".")[-1][:4]:
            flags.append("NEAR-ZERO ACTIVITY  %s = %g  -> the mechanism may never have engaged" % (k, val))
        elif any(a in kl for a in ACTIVITY) and abs(val) < 1e-9:
            flags.append("ZERO ACTIVITY       %s = %g" % (k, val))

    # exact-chance detection: a score equal to 1/n for small n is the signature of "nothing was learned"
    for k, v in flat.items():
        if not isinstance(v, (int, float)) or not any(s in k.lower() for s in SCORE):
            continue
        if any(c in k.lower() for c in CONTROL):
            continue  # a control AT chance is the control working
        for n in range(2, 33):
            if abs(v - 1.0 / n) < 1e-3:
                flags.append("EXACTLY CHANCE      %s = %.4f = 1/%d -> read as 'nothing happened', not 'it failed'" % (k, v, n))
                break

    # IDENTICAL-ARMS detection was REMOVED after it fired on a known-GOOD artifact twice: from an
    # artifact alone there is no way to tell "two arms of one comparison" (inert lever) from "two different
    # quantities that legitimately coincide" (n_registered == n_rendered_exact IS the GO). A heuristic that
    # cannot be made reliable is worse than none — it trains the reader to ignore the checker. That test
    # belongs at PROBE time, where the arms are known: use tools/lab.py::lever(before, after).
    return flags


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__.split("\n\n")[-1].strip())
        return 2
    bad = 0
    for p in paths:
        name = os.path.basename(p)
        flags = check(p)
        if flags:
            bad += 1
            print("⚠️  %s" % name)
            for f in flags[:6]:
                print("      %s" % f)
        else:
            print("✅ %s — no void signature" % name)
    if bad:
        print("\n  %d artifact(s) show a VOID SIGNATURE. Flagged, not adjudicated: look before reporting." % bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
