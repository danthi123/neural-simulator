# Capability: Sleep consolidation
# Record: brain/research/2026-11-14-near-miss-guard-1.5b.md (frame choice, seed 7 only: not a test seed)
"""Prototype for the near-miss guard: which rewrite frame gives usable near-miss questions? For each of the 12 told
facts and each frame (N1-N3) it logs the cortex's own question, its near-miss rewrites and the pretrained reply.
Score per frame: usable = a question, not the own question, and naming no told key word. Not a result.

  python -m brain.experiments.near_miss_proto --model 1.5b --seed 7 --out proto.json
"""
import argparse
import json

import torch

from brain.brain import BIRTH
from brain.cortex import Cortex, resolve_model
from brain.experiments.facts import FACTS
from brain.near_miss import REWRITE, guard_texts, near_miss_cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", default=",".join(REWRITE))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    _, base = c.read(BIRTH, None)
    keys = [k for _, _, k in FACTS]
    res = {}
    for frame in a.frames.split(","):
        rows, usable, total = [], 0, 0
        for i, (s, _, _) in enumerate(FACTS):
            own, cues = near_miss_cues(c, base, s, k=a.k, frame=frame, seed=a.seed * 100 + i)
            texts = guard_texts(c, base, cues)
            ok = [not any(k in q.lower() for k in keys) for q in cues]
            usable += sum(ok)
            total += a.k
            rows.append({"statement": s, "own": own, "cues": cues, "usable": ok, "texts": texts})
            print(frame, s, "|", own, "|", cues, flush=True)
        res[frame] = {"usable": usable, "of": total, "rows": rows}
        print(frame, "usable", usable, "of", total, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
