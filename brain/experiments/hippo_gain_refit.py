# Capability: Episodic recall
# Record: brain/research/2026-11-13-gain-refit-1.5b.md (pre-registration, R0-R3)
"""Refit the hippocampal reinstatement gain on an UNSLEPT 1.5B cortex with the fixed scorer. Oracle routing.

Same path as hippo_gain_probe (the right episode, stored as the live Brain stores it, reinstated into a fresh Brain's
state at gain g; the cortex answers greedily, at most 40 tokens), on a chosen bank:
  dev  = DEV_FACTS36 (36 facts, cues orig / para_a / para_b): the gain is fitted here only.
  test = FACTS + PARA (12 facts, cues orig / para): scored at the fitted gain only.
Each row records the reply class: right (strict_scorer.correct_answer), generic (is_generic: talks about the key word),
key_other (key present, rejected otherwise), none (key absent). No sleep, no sampling: deterministic, CPU fp32.

  python -m brain.experiments.hippo_gain_refit --bank dev --gains 0 1 2 --out <json> [--threads 12]
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
from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_bank import DEV_FACTS36
from brain.experiments.strict_scorer import correct_answer, has_key, is_generic


def bank_items(bank):
    """[(statement, [(kind, cue)], key)]"""
    if bank == "dev":
        return [(s, [("orig", o), ("para_a", pa), ("para_b", pb)], keys[0]) for s, o, pa, pb, keys in DEV_FACTS36]
    return [(s, [("orig", q), ("para", p)], k) for (s, q, k), p in zip(FACTS, PARA)]


def klass(reply, key, cue, told):
    if correct_answer(reply, key, cue, told):
        return "right"
    if not has_key(reply, key):
        return "none"
    return "generic" if is_generic(reply, key) else "key_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--bank", choices=("dev", "test"), required=True)
    ap.add_argument("--gains", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="first N facts only (smoke)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    home = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    b = Brain(f"{home}/h", cortex=c, sleep_on_restart=False)
    items = bank_items(a.bank)[:a.limit or None]
    eps = [b.hippocampus.store(b.cortex, b.base, s, speaker="User") for s, _, _ in items]
    rows = []
    for (s, cues, k), ep in zip(items, eps):
        for kind, cue in cues:
            for g in a.gains:
                active = c.add_delta(b.state, ep.delta, g) if g else b.state
                logits, st = c.read(f"User: {cue}\n\nAssistant:", active)
                reply, _ = c.speak(logits, st, max_tokens=40, temperature=0)
                cl = klass(reply, k, cue, s)
                rows.append({"q": cue, "kind": kind, "key": k, "gain": g, "reply": reply, "class": cl,
                             "correct": cl == "right"})
        print(f"{k} {time.time() - t0:.0f}s", [(r["gain"], r["class"]) for r in rows[-len(cues) * len(a.gains):]],
              flush=True)
    b.release()
    shutil.rmtree(home, ignore_errors=True)
    json.dump({"model": c.model_name, "bank": a.bank, "gains": a.gains, "rows": rows, "host": os.uname().nodename,
               "threads": a.threads, "elapsed_s": round(time.time() - t0, 1)}, open(a.out, "w"), indent=1)
    print("done", a.out, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
