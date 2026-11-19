# Capability: Curiosity
# Record: brain/research/2026-11-18-novelty-inquiry-record.md (stage 1 dev: can an "asking" direction move the reply)
# Expected-fail: the drive is a host-built direction with a host gain (declared scaffold); additive state pushes moved
#   style more than intent in the mood record (results/2026-11-18-mood4-1.5b.md)
"""Stage-1 DEV probe for "it asks about what is new to it". Dev bank only; never a test.

The brain's own replies that asked what a thing is (both blind readers agreed, dev seeds 1140-1142) and its other
replies to the SAME statements are read back by the cortex from the birth state ("Assistant: <reply>"). The asking
direction is the mean matrix-memory trace of the asking replies minus that of the others, per layer (the same form as
the affect axis). Arms: off; ask at gain g (the direction added to the brain's state before it reads the turn, the
hook `Affect.modulate` uses); rand at gain g (a per-layer norm-matched random direction). Always on, no novelty gate:
this step only asks whether the direction moves asking at all, and on which items (common items must not move).

usage: python -m brain.experiments.inquiry_dev --seed 1143 --arms off,ask4,rand4 --samples 2 --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import tempfile
import time

import numpy as np
import torch

from brain.brain import BIRTH, Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.novelty_stage0 import DEV_FRAMES, CLASSES

DEV = "brain/experiments/results/2026-11-18-novelty-dev-1.5b"


def harvest(dev_dir=DEV):
    """(asking replies, other replies to the same statements) from the dev harvest and its blind labels."""
    labels = json.load(open(f"{dev_dir}/blind_labels.json"))
    asks, other = {}, {}
    for s in (1140, 1141, 1142):
        for i, r in enumerate(json.load(open(f"{dev_dir}/dev_s{s}.json"))["rows"]):
            lab = labels[f"d{s}_{i}"]
            if all(x["asks_what"] for x in lab):
                asks.setdefault(r["stmt"], []).append(r["reply"].strip())
            elif not any(x["asks_what"] for x in lab):
                other.setdefault(r["stmt"], []).append(r["reply"].strip())
    stmts = sorted(set(asks) & set(other))
    return [x for s in stmts for x in asks[s]], [x for s in stmts for x in other[s]], stmts


def build_axis(cortex, base, ask_texts, other_texts):
    def trace(t):
        return cortex.matrix_delta(cortex.read(f"Assistant: {t}\n\n", base)[1], base)
    A = [trace(t) for t in ask_texts]
    O = [trace(t) for t in other_texts]
    return [None if i % 3 != 1 else (torch.stack([a[i] for a in A]).mean(0) - torch.stack([o[i] for o in O]).mean(0))
            for i in range(len(base))]


def random_like(axis, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in axis:
        if d is None:
            out.append(None)
            continue
        r = torch.randn(d.shape, generator=g, dtype=torch.float32).to(d.dtype)
        out.append(r * (d.float().norm() / r.float().norm().clamp_min(1e-8)).to(d.dtype))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=1143)
    ap.add_argument("--arms", default="off,ask2,ask4,ask8,rand4,rand8")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cortex = Cortex(resolve_model(a.model))
    _, base = cortex.read(BIRTH)
    ask_t, other_t, stmts = harvest()
    axis = build_axis(cortex, base, ask_t, other_t)
    rnd = random_like(axis, a.seed)
    norms = [float(d.float().norm()) for d in axis if d is not None]
    rows, t0 = [], time.time()
    for arm in a.arms.split(","):
        kind, g = ("off", 0.0) if arm == "off" else (arm.rstrip("0123456789."), float(arm.lstrip("askrand")))
        vec = axis if kind == "ask" else rnd
        for fi, frame in enumerate(DEV_FRAMES):
            for ci, cls in enumerate(CLASSES):
                item = frame[1 + ci]
                stmt = frame[0].format(item)
                for k in range(a.samples):
                    # common random numbers across arms: the same sampling seed per (statement, sample)
                    sd = a.seed * 100000 + fi * 100 + ci * 10 + k
                    random.seed(sd); np.random.seed(sd % (2 ** 32)); torch.manual_seed(sd)
                    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as home:
                        b = Brain(home, cortex=cortex)
                        if kind != "off":
                            b.state = cortex.add_delta(b.state, vec, g)
                        reply, _ = b.hear(stmt)
                    rows.append({"arm": arm, "frame": fi, "cls": cls, "item": item, "stmt": stmt, "sample": k,
                                 "reply": reply})
            print(f"[{arm}] frame {fi} {time.time() - t0:.0f}s", flush=True)
    out = {"seed": a.seed, "model": a.model, "arms": a.arms, "samples": a.samples, "host": socket.gethostname(),
           "torch": torch.__version__, "threads": a.threads, "n_ask": len(ask_t), "n_other": len(other_t),
           "n_stmts": len(stmts), "axis_layer_norms": norms, "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_ask", "n_other", "n_stmts", "seconds")}))


if __name__ == "__main__":
    main()
