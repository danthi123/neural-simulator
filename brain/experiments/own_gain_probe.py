# Capability: Language
# Record: brain/research/2026-11-16-own-gain-1.5b.md (research round 11-16: the brain's own reply written into the
# conversation state at reduced gain, so a sampled copy cannot reinforce itself)
"""Report-only probe, no training: does a copy, once in the conversation state, persist because the brain's own reply
is written there? Replays open_conv's day-2 script of --script-seed on a copy of a slept home with the first --force
day-2 replies FORCED to those of a recorded collapsed conversation (the onset), then lets the brain speak freely, once
per sampling seed, for each own_gain. Counts, on the free turns: pronoun-normalised echoes of the user's turn, and
repeats of the forced onset reply.
  python -m brain.experiments.own_gain_probe --home <slept home> --script-seed 1121 --force-json <rows or probe json>
      [--force-index K] [--force-arm intact] --force 2 --gains none 1 0.5 0 --samples 6 [--hippocampus-off --night-off]
      --out <json>
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval.open_conv import build
from brain.experiments.echo_onset_probe import _norm


def forced_replies(path, index, arm, n):
    d = json.load(open(path))
    if isinstance(d, list):   # echo_onset_probe output
        return [t["brain"] for t in d[index]["turns"]][:n]
    return [r["brain"] for r in d["rows"] if r["day"] == 2 and r["arm"] == arm][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--home", required=True)
    ap.add_argument("--script-seed", type=int, required=True)
    ap.add_argument("--force-json", required=True)
    ap.add_argument("--force-index", type=int, default=0)
    ap.add_argument("--force-arm", default="intact")
    ap.add_argument("--force", type=int, default=2)
    ap.add_argument("--gains", nargs="+", default=["none", "1", "0.5", "0"])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--hippocampus-off", action="store_true")
    ap.add_argument("--night-off", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    _, _, day2 = build(a.script_seed)
    forced = forced_replies(a.force_json, a.force_index, a.force_arm, a.force)
    onset = _norm(forced[-1])
    res = []
    for gs in a.gains:
        g = None if gs == "none" else float(gs)
        for k in range(a.samples):
            np.random.seed(1000 * a.script_seed + k)
            torch.manual_seed(1000 * a.script_seed + k)
            tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
            cp = os.path.join(tmp, "home")
            shutil.copytree(a.home, cp)
            b = Brain(cp, cortex=c, sleep_on_restart=False, sleep_seed=a.script_seed, own_gain=g)
            b.set_lesions(hippocampus_off=a.hippocampus_off, consolidated_off=a.night_off)
            real = b.cortex.speak
            queue = list(forced)
            b.cortex.speak = lambda lg, st, **kw: (queue.pop(0), st) if queue else real(lg, st, **kw)
            turns = []
            for i, (kind, turn, _) in enumerate(day2):
                reply, info = b.hear(turn)
                turns.append({"kind": kind, "you": turn, "brain": reply, "forced": i < len(forced),
                              "recalled": info.get("recalled")})
            shutil.rmtree(tmp, ignore_errors=True)
            free = [t for t in turns if not t["forced"]]
            r = {"gain": gs, "sample": k, "n_free": len(free),
                 "echo": sum(_norm(t["brain"]) == _norm(t["you"]) for t in free),
                 "onset_repeat": sum(_norm(t["brain"]) == onset for t in free), "turns": turns}
            res.append(r)
            print(gs, k, "echo", r["echo"], "onset_repeat", r["onset_repeat"], "of", r["n_free"], flush=True)
            json.dump({"forced": forced, "args": vars(a), "runs": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
