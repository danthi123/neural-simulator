# Capability: Episodic recall
# Record: brain/research/2026-11-14-lived-episode-1.5b.md (pre-registration, E0-E3)
"""Does storing the LIVED exchange (the user's telling + the brain's own reply to it) instead of the telling alone stop
the first-person echo ("My cat is called Pixel." to "What is my cat called?")? Oracle routing, unslept 1.5B cortex.

Per fact, format "tell" = Hippocampus.store("User: <s>") as the live Brain does now; format "lived" = store("User:
<s>\n\nAssistant: <r>") where r is the brain's own greedy reply to the telling, read from the same Brain state with no
reinstatement (what it would have said live; no host text). Then, as hippo_gain_refit, the right episode is reinstated
at each gain and the cortex answers greedily (40 tokens). Classes: right / echo (strict_scorer.is_echo) / generic /
key_other / none. Deterministic, CPU fp32.

  python -m brain.experiments.episode_format_probe --bank dev --gains 0.25 0.5 1 --out <json>
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.hippo_gain_refit import bank_items, klass
from brain.experiments.strict_scorer import is_echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--bank", choices=("dev", "test", "fresh"), required=True)
    ap.add_argument("--gains", type=float, nargs="+", required=True)
    ap.add_argument("--formats", default="tell,lived")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    home = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    b = Brain(f"{home}/h", cortex=c, sleep_on_restart=False)
    items = bank_items(a.bank)[:a.limit or None]
    own = []
    for s, _, _ in items:
        lg, st = c.read(f"User: {s}\n\nAssistant:", b.state)
        own.append(c.speak(lg, st, max_tokens=40, temperature=0)[0].strip())
    formats = a.formats.split(",")
    eps = {"tell": [b.hippocampus.store(b.cortex, b.base, s, speaker="User") for s, _, _ in items]}
    contrast = [f for f in formats if f.startswith("contrast")]   # contrast<mu*100>: d_user + mu (d_user - d_self)
    if contrast:
        selfd = [b.hippocampus.store(b.cortex, b.base, s, speaker="Assistant").delta for s, _, _ in items]
        for f in contrast:
            mu = int(f[len("contrast"):]) / 100
            eps[f] = [type("E", (), {"delta": [None if u is None else (u.float() + mu * (u.float() - v.float())).half()
                                               for u, v in zip(e.delta, sd)]})() for e, sd in zip(eps["tell"], selfd)]
    if "lived" in formats:
        eps["lived"] = [b.hippocampus.store(b.cortex, b.base, f"{s}\n\nAssistant: {r}", speaker="User")
                        for (s, _, _), r in zip(items, own)]
    rows = []
    for i, (s, cues, k) in enumerate(items):
        for fmt in formats:
            for kind, cue in cues:
                for g in a.gains:
                    active = c.add_delta(b.state, eps[fmt][i].delta, g)
                    logits, st = c.read(f"User: {cue}\n\nAssistant:", active)
                    reply, _ = c.speak(logits, st, max_tokens=40, temperature=0)
                    cl = klass(reply, k, cue, s)
                    if cl == "key_other" and is_echo(reply, k, s):
                        cl = "echo"
                    rows.append({"q": cue, "kind": kind, "key": k, "format": fmt, "gain": g, "reply": reply,
                                 "class": cl, "correct": cl == "right"})
        print(f"{k} {time.time() - t0:.0f}s own={own[i][:50]!r}", flush=True)
    b.release()
    shutil.rmtree(home, ignore_errors=True)
    json.dump({"model": c.model_name, "bank": a.bank, "gains": a.gains, "formats": formats, "own_replies": own,
               "rows": rows, "host": os.uname().nodename, "elapsed_s": round(time.time() - t0, 1)},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
