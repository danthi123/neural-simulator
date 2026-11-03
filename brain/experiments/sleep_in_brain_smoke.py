# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""End-to-end smoke: tell the live brain 3 facts, restart (= one night of sleep), ask with the hippocampus OFF.

Arms, each a fresh Brain on the same home after the restart (so each starts from the same day-2 state):
  hippo_off            hippocampus off, consolidated synapses on    (does sleep alone answer?)
  hippo_off_consol_off hippocampus off, consolidated synapses off   (the pretrained cortex: lesion contrast)
  intact               both on
Questions: the 3 told facts, their 3 paraphrases, 1 never-told (false recall). Replies sampled as in chat (np seeded per arm).
--seed sets the sleep seed (adapter init + replay order) and the reply sampling.

  python -m brain.experiments.sleep_in_brain_smoke --home <dir> --seed 42 --out <json>
"""
import argparse
import json
import re
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments.facts import FACTS, PARA

TOLD = [FACTS[0], FACTS[3], FACTS[5]]          # cat Pixel, sister Marguerite, red Volvo
NEVER = ("What is my horse called?", None)
KEYS = {**{q: k for _, q, k in TOLD}, **{PARA[FACTS.index(f)]: f[2] for f in TOLD}}
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|mine)\b")


def is_hit(reply, key):
    """Answer word present AND the reply is not a first-person echo of the user's own line.
    "I drive a red Volvo." repeats what the user said (speaker's perspective); a correct answer speaks to the user
    ("Your cat is called Pixel."). Skeptic 2026-11-03: the bare substring test counted 7/15 hippo-off hits that were echoes."""
    low = reply.lower()
    return key in low and not FIRST_PERSON.search(low)


def rescore(path):
    """Re-score a saved result JSON with is_hit (no model rerun). Returns {arm: (recall, paraphrase)} and rewrites nothing."""
    d = json.load(open(path))
    out = {}
    for arm, v in d["arms"].items():
        keyed = {r["q"]: r for r in v["rows"]}
        told = sum(is_hit(r["reply"], KEYS[r["q"]]) for r in v["rows"] if r["kind"] == "told")
        para = sum(is_hit(r["reply"], KEYS[r["q"]]) for r in v["rows"] if r["kind"] == "para")
        out[arm] = (told, para)
    return out


ARMS = {"hippo_off": (True, False), "hippo_off_consol_off": (True, True), "intact": (False, False)}


def main():
    ap = argparse.ArgumentParser()
    if len(__import__("sys").argv) > 2 and __import__("sys").argv[1] == "--rescore":
        for p in __import__("sys").argv[2:]:
            print(p, rescore(p))
        return
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    shutil.rmtree(a.home, ignore_errors=True)
    c = Cortex(model_path=a.model, device="cpu")
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    day1 = Brain(a.home, cortex=c, sleep_steps=a.steps, sleep_seed=a.seed)
    told = []
    for s, _, _ in TOLD:
        reply, info = day1.hear(s)
        told.append({"you": s, "brain": reply, "stored": info["stored"]})
    day1.save()                                   # not closed: the restart itself must sleep
    t1 = time.time()
    day2 = Brain(a.home, cortex=c, sleep_steps=a.steps, sleep_seed=a.seed, sleep_log=lambda m: print(m, flush=True))
    slept_s = time.time() - t1
    res = {"model": a.model, "seed": a.seed, "steps": a.steps, "day1": told, "sleeps_after_restart": day2.meta["sleeps"],
           "sleep_seconds": round(slept_s, 1), "episodes_replayed": len(day2.hippocampus.episodes),
           "adapter_norm": day2.cortex.plastic.adapter_norm() if day2.cortex.plastic else 0.0, "arms": {}}
    for arm, (hip_off, con_off) in ARMS.items():
        rows = []
        qs = [(q, k, "told") for _, q, k in TOLD] + [(PARA[FACTS.index(f)], f[2], "para") for f in TOLD] + \
             [(NEVER[0], None, "never")]
        for q, key, kind in qs:
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            b = Brain(a.home, cortex=c, sleep_steps=a.steps)   # nothing pending: no further sleep
            assert b.meta["sleeps"] == 1
            b.set_lesions(hippocampus_off=hip_off, consolidated_off=con_off)
            reply, info = b.hear(q)
            hit = None if key is None else is_hit(reply, key)
            leak = key is None and any(k in reply.lower() for _, _, k in FACTS)
            rows.append({"q": q, "kind": kind, "reply": reply, "hit": hit, "false_recall": leak, "recalled": info["recalled"]})
            print(arm, "|", q, "->", reply, "|", "hit" if hit else ("LEAK" if leak else "-"), flush=True)
        res["arms"][arm] = {"recall": sum(bool(r["hit"]) for r in rows if r["kind"] == "told"),
                            "paraphrase": sum(bool(r["hit"]) for r in rows if r["kind"] == "para"), "false_recall": sum(r["false_recall"] for r in rows),
                            "rows": rows}
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
