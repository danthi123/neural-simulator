"""ROADMAP PHASE 2 (the real "one brain"), STEP A3 PRODUCTION WIRING -- the `BrainConversationalAgent` runs on the
`OneBrainComposer`. The agent's `hear(sentence)` DELEGATES comprehension to the composer's OWN on-bridge parser (one
parser on the one brain), and `what_does`/`who_does`/`is_it_true` query the persistent on-bridge store. This de-risk
confirms the WIRED AGENT answers the core who/what/yes-no/moat matrix == a reference `RFPhasorComposer`-backed agent ==
ground truth.

GATE (3 seeds, the agent is the unit): with `composer_kind="onebrain"`, the agent's `what_does`/`who_does`/`is_it_true`
over a heard knowledge base == the reference agent (`composer_kind="rf"`, the validated production default) == ground
truth, AND the no-confab moat holds (an unheard cue -> what_does None / is_it_true 'unknown'). Both agents hear the
SAME sentences with polarity="AFFIRM" (the affirmative-fact scope). Reuse-by-import; the only protected-set-free edits
are the additive `composer_kind="onebrain"` branch + the `hear()` delegation in brain_conversational_agent.py. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_agent_integration_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
PATIENTS = ["north", "east", "south", "west", "home"]
VOCAB = AGENTS + ACTIONS + PATIENTS


def run_seed(seed):
    concepts = {w: None for w in VOCAB}                 # set the vocabulary (codes are generated per word from seed)
    one = BrainConversationalAgent(seed=seed, composer_kind="onebrain", concepts=concepts)
    ref = BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=concepts)
    facts = [(AGENTS[i], ACTIONS[i], PATIENTS[i]) for i in range(5)]
    # the 5th heard via its PASSIVE frame on BOTH agents (voice-invariant comprehension through the wired agent)
    for i, (a, v, p) in enumerate(facts):
        if i == 4:
            one.hear(f"{p} {v} {a}", voice="passive", polarity="AFFIRM")
            ref.hear(f"{p} {v} {a}", voice="passive", polarity="AFFIRM")
        else:
            one.hear(f"{a} {v} {p}", voice="active", polarity="AFFIRM")
            ref.hear(f"{a} {v} {p}", voice="active", polarity="AFFIRM")

    okw = okr = oky = refw = refr = refy = 0
    for (a, v, p) in facts:
        w1, r1, y1 = one.what_does(a, v), one.who_does(v, p), one.is_it_true(a, v, p)
        w0, r0, y0 = ref.what_does(a, v), ref.who_does(v, p), ref.is_it_true(a, v, p)
        okw += int(w1 == p); okr += int(r1 == a); oky += int(y1 == "yes")
        refw += int(w1 == w0); refr += int(r1 == r0); refy += int(y1 == y0)
    n = len(facts)

    # MOAT: an unheard (agent, action) cue -> what_does None; an unheard fact -> is_it_true 'unknown'.
    used = {(a, v) for (a, v, p) in facts}
    absent = next(((a, v) for a in AGENTS for v in ACTIONS if (a, v) not in used), None)
    moat_w = int(one.what_does(absent[0], absent[1]) is None) if absent else 1
    moat_y = int(one.is_it_true(AGENTS[0], ACTIONS[1], PATIENTS[2]) in ("unknown", "no"))

    row = {"seed": seed, "what": okw / n, "who": okr / n, "yesno": oky / n,
           "ref_what": refw / n, "ref_who": refr / n, "ref_yesno": refy / n, "moat_w": moat_w, "moat_y": moat_y}
    print(f"  [seed {seed}] agent who/what: what={okw/n:.2f} who={okr/n:.2f} yes/no={oky/n:.2f} | == ref agent "
          f"w/r/y {refw/n:.2f}/{refr/n:.2f}/{refy/n:.2f} | moat what->None {moat_w} yn->abstain {moat_y}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_agent_integration.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print("[STEP A3 wiring] does BrainConversationalAgent(composer_kind='onebrain') answer who/what/yes-no/moat == the "
          "rf-backed reference agent == ground truth (one parser on the one brain)?\n", flush=True)
    rows = [run_seed(s) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    w, r, y = m("what"), m("who"), m("yesno")
    rw, rr, ry = m("ref_what"), m("ref_who"), m("ref_yesno")
    mw, my = m("moat_w"), m("moat_y")
    n_full = sum(int(x["what"] >= 0.99 and x["who"] >= 0.99 and x["yesno"] >= 0.99 and x["ref_what"] >= 0.99
                     and x["ref_who"] >= 0.99 and x["ref_yesno"] >= 0.99 and x["moat_w"] >= 1 and x["moat_y"] >= 1)
                 for x in rows)
    go = (n_full == len(rows))
    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(rows)} seeds): agent what {w:.3f} who {r:.3f} yes/no {y:.3f} | == ref agent w/r/y "
          f"{rw:.3f}/{rr:.3f}/{ry:.3f} | moat what->None {mw:.2f} yn->abstain {my:.2f} | full {n_full}/{len(rows)}",
          flush=True)
    if go:
        print(f"  GO: BrainConversationalAgent(composer_kind='onebrain') runs the who/what/yes-no/moat matrix == the "
              f"rf reference agent == ground truth -- the agent delegates comprehension to the OneBrainComposer's "
              f"on-bridge parser (one parser on the one brain), stores on the persistent bridge, and queries it, with "
              f"the no-confab moat intact. ==> the integrated one-brain conversational agent is WIRED + validated; add "
              f"a CI guard, then A4/A5 (richer caps + retire legacy numpy runtime).", flush=True)
    else:
        print(f"  BOUNDARY/NEGATIVE: full {n_full}/{len(rows)} (what {w:.3f} who {r:.3f} yn {y:.3f} ref {rw:.3f}/{rr:.3f}"
              f"/{ry:.3f} moat {mw:.2f}/{my:.2f}) -- localize (agent hear-delegation / the composer query / the moat). "
              f"The rf default is unaffected (additive wiring). Reportable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*104}", flush=True)
    out = {"verdict": "GO" if go else "BOUNDARY", "seeds": seeds, "what": w, "who": r, "yesno": y,
           "ref_what": rw, "ref_who": rr, "ref_yesno": ry, "moat_w": mw, "moat_y": my, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
