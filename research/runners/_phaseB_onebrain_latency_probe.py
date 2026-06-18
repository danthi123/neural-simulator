"""Quick latency probe for the OneBrainComposer (informs the A5 speed design): time a who/what query on the onebrain
agent vs the rf reference agent, over a K-fact store. The onebrain scan is reconstruct-per-block (O(K) blocks x a
208-step resonate per op); the rf agent is the speed reference. Prints ms/query for each + the ratio.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_latency_probe --k 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

AG = ["dog", "cat", "bird", "river", "apple", "tree", "sun", "moon"]
AC = ["go", "come", "look", "stop", "swim", "walk", "run", "jump"]
PA = ["north", "east", "south", "west", "home", "hill", "lake", "sky"]
VOCAB = AG + AC + PA


def _time_queries(agent, facts, n_rep):
    t0 = time.time()
    for _ in range(n_rep):
        for (a, v, p) in facts:
            agent.what_does(a, v)
    return 1000.0 * (time.time() - t0) / (n_rep * len(facts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rep", type=int, default=2)
    args = ap.parse_args()
    k = min(args.k, len(AG))
    facts = [(AG[i], AC[i], PA[i]) for i in range(k)]
    concepts = {w: None for w in VOCAB}
    print(f"[onebrain latency probe] K={k} facts, {args.n_rep} reps/fact\n", flush=True)

    one = BrainConversationalAgent(seed=args.seed, composer_kind="onebrain", concepts=concepts)
    for (a, v, p) in facts:
        one.hear(f"{a} {v} {p}", polarity="AFFIRM")
    ms_one = _time_queries(one, facts, args.n_rep)

    ref = BrainConversationalAgent(seed=args.seed, composer_kind="rf", concepts=concepts)
    for (a, v, p) in facts:
        ref.hear(f"{a} {v} {p}", polarity="AFFIRM")
    ms_ref = _time_queries(ref, facts, args.n_rep)

    print(f"  onebrain what_does: {ms_one:.1f} ms/query", flush=True)
    print(f"  rf       what_does: {ms_ref:.1f} ms/query", flush=True)
    print(f"  ratio (onebrain / rf): {ms_one / max(ms_ref, 1e-6):.1f}x  (the gap A5 must close)", flush=True)
    print(f"\n  Structure: onebrain query = reconstruct-per-block over K={k} blocks, each ~2 resonate windows of "
          f"~208 steps = O(K) * the 208-step resonate. A5 levers: (1) batched scan (1 resonate over K blocks), "
          f"(2) indexed store (cue->block, O(K)->O(1)), (3) masked-megakernel (fuse the resonate step).", flush=True)


if __name__ == "__main__":
    main()
