# Capability: Curiosity
# Record: brain/research/2026-11-18-novelty-inquiry-record.md (additive NO-GO: results/2026-11-18-inquiry-dev-1.5b.md)
# Expected-fail: the lived lesson may be copied verbatim (confab M2 at 1.5B: ~96% verbatim), and the bar is host-set
"""DEV probe for the novelty-gated inquiry episode (Brain(inquiry=bar), Brain.live_inquiry). Dev bank only.

Per statement (dev bank, C/R/N) and sample, a fresh brain:
  off         - Brain defaults (no lesson, no gate)
  lesson      - the lesson lived, gate off (inquiry=None): does having lived it change replies on its own?
  gate        - the lesson lived, gate at --bar (nats)
  forced      - the lesson lived, gate bar -1e9 (reinstated on every telling): what the gate adds over always-on
The lesson is an earlier conversation: after it the working state is reset to the birth state (episode kept).
Common random numbers across arms (the same sampling seed per statement and sample, set after the lesson).
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

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.novelty_stage0 import CLASSES, DEV_FRAMES, FRAMES

LESSON = "My friend keeps a zorbit in her kitchen."
GUIDED = "I don't know that word. What is it?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=1144)
    ap.add_argument("--bank", choices=["dev", "test"], default="dev")
    ap.add_argument("--arms", default="off,lesson,gate,forced")
    ap.add_argument("--bar", type=float, default=12.4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--guided", default=GUIDED, help="the lesson's own reply (dev variants)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cortex = Cortex(resolve_model(a.model))
    frames = DEV_FRAMES if a.bank == "dev" else FRAMES
    rows, t0 = [], time.time()
    for arm in a.arms.split(","):
        for fi, frame in enumerate(frames):
            for ci, cls in enumerate(CLASSES):
                item = frame[1 + ci]
                stmt = frame[0].format(item)
                for k in range(a.samples):
                    sd = a.seed * 100000 + fi * 100 + ci * 10 + k
                    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as home:
                        kw = {"inquiry": None} if arm in ("off", "lesson") else {"inquiry": a.bar if arm == "gate" else -1e9}
                        b = Brain(home, cortex=cortex, **kw)
                        spont = None
                        if arm != "off":
                            random.seed(sd + 7); np.random.seed((sd + 7) % 2 ** 32); torch.manual_seed(sd + 7)
                            spont = b.live_inquiry(LESSON, a.guided)
                            # the lesson was an earlier conversation: the episode stays in the hippocampus, the
                            # working state starts fresh (otherwise the brain repeats its own last turn)
                            b.state = [x.clone() for x in b.base]
                            b._recent_replies = []
                        random.seed(sd); np.random.seed(sd % 2 ** 32); torch.manual_seed(sd)
                        reply, info = b.hear(stmt)
                    rows.append({"arm": arm, "frame": fi, "cls": cls, "item": item, "stmt": stmt, "sample": k,
                                 "reply": reply, "fired": bool(info.get("inquiry")),
                                 "novelty": getattr(b, "_last_novelty", None), "lesson_spontaneous": spont,
                                 "verbatim_guided": reply.strip() == a.guided.strip()})
            print(f"[{arm}] frame {fi} {time.time() - t0:.0f}s", flush=True)
    out = {"seed": a.seed, "model": a.model, "bank": a.bank, "bar": a.bar, "arms": a.arms, "samples": a.samples,
           "lesson": LESSON, "guided": a.guided, "host": socket.gethostname(), "torch": torch.__version__,
           "threads": a.threads, "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    fired = {arm: {c: sum(r["fired"] for r in rows if r["arm"] == arm and r["cls"] == c) for c in CLASSES}
             for arm in a.arms.split(",")}
    verb = {arm: sum(r["verbatim_guided"] for r in rows if r["arm"] == arm) for arm in a.arms.split(",")}
    print(json.dumps({"fired": fired, "verbatim_guided": verb, "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
