# Capability: Self and identity
# Record: brain/experiments/results/2026-11-13-live-gain-1.5b.md (flip check for Brain(gain="size"))
"""Does gain="size" (1.5B: 0.5) change the brain's answers about itself and its friend? Fresh 1.5B brain, no sleep, the
three permanent self episodes plus a short chat; each self question asked of a fresh brain, gain default vs size,
S_all key with the size bar (the live f/r setting) and S0. Greedy, deterministic, CPU.

  python -m brain.experiments.gain_identity_check --out <json>
"""
import argparse
import json
import os
import shutil
import tempfile

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model

QS = ["What's your name?", "Who are you?", "Do you remember my name?", "What is your name?", "Tell me about yourself.",
      "Who am I?", "Are you a digital brain?", "What's my name?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    rows = []
    for key, bar in (("S0", "default"), ("S_all", "size")):
        for gain in ("default", "size"):
            for i, q in enumerate(QS):
                home = f"{root}/{key}_{gain}_{i}"
                b = Brain(home, cortex=c, sleep_on_restart=False, key=key, key_bar=bar, gain=gain)
                reply, info = b.hear(q)
                rows.append({"key": key, "gain": gain, "gain_value": b.hippocampus.GAIN, "q": q, "reply": reply,
                             "recalled": info["recalled"], "nova": "nova" in reply.lower(), "dan": "dan" in reply.lower()})
                print(key, gain, q, "|", info["recalled"] and info["recalled"][:40], "|", reply[:80], flush=True)
                b.release()
    shutil.rmtree(root, ignore_errors=True)
    json.dump({"model": c.model_name, "host": os.uname().nodename, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
