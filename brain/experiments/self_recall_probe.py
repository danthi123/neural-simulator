# Capability: Self and identity
# Record: brain/experiments/results/2026-11-15-self-completion-weekly-1.5b.md (session 2: "What's your name?" recalls the sister telling on 6/6)
"""Which episode wins the self questions after the night, and why? No training, no sleep, CPU, a copy of each home.

Each home is a slept weekly brain (`<homes>/_weekly_intact` after session 2, or after session 1 and its night). For
each self cue the brain's own recall competition is read (Hippocampus.scores and the bar/margin, as recall() does it,
with no use boost: `best3`, `winner`, `passes`) in four arms:
  key  S0 (the default) | S_all with its size-fitted bar (Brain(key="S_all", key_bar="size"))
  read live (the cue read through the slept cortex, as session 2 does) | raw (consolidated_off: the pretrained
       cortex, as on day 1 when the stored keys were made)
No reply is generated; this is a selection read-out.

  python -m brain.experiments.self_recall_probe --out <json> <home> [<home> ...]
"""
import argparse
import json
import os
import shutil
import tempfile

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model

CUES = ["What's your name?", "Who are you?", "Hi again! Do you remember my name?", "What is my sister's name?",
        "What is my cat called?"]


def read_arm(b, cue):
    h = b.hippocampus
    score = h.scores(b.cortex, b.base, cue, "User")
    order = torch.argsort(score, descending=True)
    best3 = [(h.episodes[int(i)].text[:60], round(float(score[int(i)]), 4)) for i in order[:3]]
    top, run = float(score[order[0]]), (float(score[order[1]]) if len(order) > 1 else -1.0)
    passes = top >= h.FAMILIARITY and top - run >= h.MARGIN
    return {"best3": best3, "winner": h.episodes[int(order[0])].text[:60], "passes": passes,
            "bar": h.FAMILIARITY, "margin_needed": h.MARGIN, "margin": round(top - run, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("homes", nargs="+")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    out = []
    for home in a.homes:
        tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
        cp = os.path.join(tmp, "home")
        shutil.copytree(home, cp)
        res = {"home": home}
        for key in ("S0", "S_all"):
            kw = {} if key == "S0" else {"key": "S_all", "key_bar": "size"}
            try:
                b = Brain(cp, cortex=c, sleep_on_restart=False, **kw)
            except ValueError:   # no size-fitted S_all bar (0.1B smoke only): the 0.1B-fit default bar, recorded
                kw["key_bar"] = "default"
                b = Brain(cp, cortex=c, sleep_on_restart=False, **kw)
            res[f"{key}_key_bar"] = kw.get("key_bar", "default")
            assert not b.pending(), f"{home}: pending episodes; this probe reads a slept home"
            res["n_episodes"] = len(b.hippocampus.episodes)
            for read in ("live", "raw"):
                b.set_lesions(consolidated_off=(read == "raw"))
                for cue in CUES:
                    r = read_arm(b, cue)
                    res[f"{key}|{read}|{cue}"] = r
                    print(os.path.basename(os.path.dirname(home.rstrip("/"))), key, read, repr(cue[:28]), "->",
                          repr(r["winner"][:40]), "passes" if r["passes"] else "none", r["best3"][0][1], r["margin"],
                          flush=True)
            b.release()
        out.append(res)
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"model": c.model_name, "host": os.uname().nodename, "cues": CUES, "homes": out}, open(a.out, "w"),
              indent=1)


if __name__ == "__main__":
    main()
