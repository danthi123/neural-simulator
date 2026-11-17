# Capability: Language
# Record: brain/experiments/results/2026-11-16-reply-rehearse-pool-replication/README.md (seed 1121 echoes the user
# 14/16 on day 2 with the night and the hippocampus both off: the pretrained cortex alone)
"""Report-only probe, no training: is the day-2 echo cascade a property of the conversation script, or a sampled
onset that then repeats in context? Runs the pretrained cortex alone (hippocampus off, night off, on a copy of any
slept home) over open_conv's day-2 script of each --script-seed, once per sampling seed, and counts verbatim echoes
and the turn of the first one.
  python -m brain.experiments.echo_onset_probe --home <slept home> --script-seeds 1121 1118 --samples 8 --out <json>
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


def _norm(s):
    return " ".join(s.lower().replace("your", "my").replace("you", "i").split()).strip(" ?.!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--home", required=True)
    ap.add_argument("--script-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--carry-penalty", action="store_true", help="Brain(carry_penalty=True)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    res = []
    for s in a.script_seeds:
        _, _, day2 = build(s)
        for k in range(a.samples):
            np.random.seed(1000 * s + k)
            torch.manual_seed(1000 * s + k)
            tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
            cp = os.path.join(tmp, "home")
            if a.home != "fresh":   # "fresh": a newly born home (any cortex size; nothing lived, no night)
                shutil.copytree(a.home, cp)
            b = Brain(cp, cortex=c, sleep_on_restart=False, sleep_seed=s, carry_penalty=a.carry_penalty,
                      # the probe never sleeps; explicit settings let a size with no SLEEP_DEFAULTS row load
                      sleep_steps=480, sleep_lr=5e-4, sleep_kl_weight=3.0, adapter_rank=8)
            b.set_lesions(hippocampus_off=True, consolidated_off=True)
            turns = []
            for kind, turn, _ in day2:
                reply, _ = b.hear(turn)
                turns.append({"kind": kind, "you": turn, "brain": reply,
                              "echo": reply.strip() == turn.strip(), "echo_flip": _norm(reply) == _norm(turn)})
            shutil.rmtree(tmp, ignore_errors=True)
            first = next((i for i, t in enumerate(turns) if t["echo_flip"]), None)
            r = {"script_seed": s, "sample": k, "echo": sum(t["echo"] for t in turns),
                 "echo_flip": sum(t["echo_flip"] for t in turns), "first_echo_turn": first,
                 "questions_back": sum(t["brain"].strip().endswith("?") and not t["echo_flip"] for t in turns),
                 "greet_reply": turns[0]["brain"], "turns": turns}
            res.append(r)
            print(s, k, "echo", r["echo"], "echo_flip", r["echo_flip"], "first", first, "q", r["questions_back"], "|", r["greet_reply"][:60],
                  flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
