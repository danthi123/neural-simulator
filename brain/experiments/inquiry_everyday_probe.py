# Capability: Curiosity
# Record: brain/research/2026-11-19-curiosity-test-prereg.md
# Expected-fail: the bar is host-set and fitted on common words, so real but rare names may cross it
"""Cost check for the novelty-gated inquiry episode: everyday tellings with real names, places and brands.

Per telling and sample, a fresh brain with the lesson lived (as in inquiry_gate_dev), arms gate (--bar) and lesson
(gate off = the reply without the switch). Records whether the gate fired, the novelty, and both replies, so a reader
can see whether real names are treated as unknown words ("What's a Marguerite?").
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
from brain.experiments.inquiry_gate_dev import LESSON

TELLINGS = [
    "My cat is called Pixel.", "My sister's name is Marguerite.", "I drive a red Volvo.",
    "I work as a system administrator, mostly Linux servers.", "My brother lives in Edinburgh.",
    "My daughter is called Siobhan.", "We went to Reykjavik last summer.", "My dog's name is Biscuit.",
    "I grew up in Saskatoon.", "My husband is called Tomasz.", "I play the cello in an orchestra.",
    "My favourite band is Radiohead.", "I just bought a Kindle.", "My grandmother was born in Gdansk.",
    "My best friend is called Oluwaseun.", "I had pho for lunch.", "My son plays rugby on Saturdays.",
    "I use Kubernetes at work.", "My parrot is called Kiwi.", "We live near Lake Winnipesaukee.",
    "My boss is called Mr. Abernathy.", "I'm reading a book by Ishiguro.", "My niece studies at Dartmouth.",
    "I fixed my Raspberry Pi yesterday.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bar", type=float, default=12.4)
    ap.add_argument("--guided", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cortex = Cortex(resolve_model(a.model))
    rows, t0 = [], time.time()
    for ti, stmt in enumerate(TELLINGS):
        for k in range(a.samples):
            sd = a.seed * 100000 + ti * 10 + k
            rec = {"stmt": stmt, "sample": k}
            for arm in ("lesson", "gate"):
                with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as home:
                    b = Brain(home, cortex=cortex, inquiry=a.bar if arm == "gate" else None)
                    random.seed(sd + 7); np.random.seed((sd + 7) % 2 ** 32); torch.manual_seed(sd + 7)
                    b.live_inquiry(LESSON, a.guided)
                    b.state = [x.clone() for x in b.base]
                    b._recent_replies = []
                    random.seed(sd); np.random.seed(sd % 2 ** 32); torch.manual_seed(sd)
                    reply, info = b.hear(stmt)
                rec[arm] = reply
                if arm == "gate":
                    rec["fired"] = bool(info.get("inquiry"))
                    rec["novelty"] = getattr(b, "_last_novelty", None)
            rows.append(rec)
        print(f"{ti} {time.time() - t0:.0f}s", flush=True)
    json.dump({"seed": a.seed, "model": a.model, "bar": a.bar, "guided": a.guided, "host": socket.gethostname(),
               "rows": rows, "seconds": round(time.time() - t0, 1)}, open(a.out, "w"), indent=1)
    print(json.dumps({"fired": sum(r["fired"] for r in rows), "n": len(rows)}))


if __name__ == "__main__":
    main()
