# Capability: Self and identity
# Record: brain/research/2026-11-15-self-key-question-1.5b.md (K1-K3)
"""Self episodes keyed by their question alone, on slept homes: no training, no reply, CPU, a copy of each home.

For each home the S0 recall competition (Hippocampus.scores, the bar recall() uses: FAMILIARITY for a cue ending in
"?", else FAMILIARITY_SPONTANEOUS; margin MARGIN) is read for every session-2 cue and the session-1 cat question, twice:
"exchange" (as stored) and "question" (the three permanent self episodes re-keyed from their question, as
Brain(self_key="question") would key them at birth; the S0 key is the layer-0 attention delta, which the night does not
change, so this is the key that home would hold). Nothing is written back.

  python -m brain.experiments.self_key_probe --out <json> <home> [<home> ...]
"""
import argparse
import json
import os
import shutil
import socket
import tempfile

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval.weekly import SESSION_2

CUES = list(SESSION_2) + ["What is my cat called? (session 1)"]


def read(b, cue):
    h = b.hippocampus
    text = cue.replace(" (session 1)", "")
    score = h.scores(b.cortex, b.base, text, "User")
    order = torch.argsort(score, descending=True)
    top, run = float(score[order[0]]), (float(score[order[1]]) if len(order) > 1 else -1.0)
    bar = h.FAMILIARITY if text.endswith("?") else h.FAMILIARITY_SPONTANEOUS
    return {"best3": [(h.episodes[int(i)].text[:60], round(float(score[int(i)]), 4)) for i in order[:3]],
            "winner": h.episodes[int(order[0])].text[:60], "passes": top >= bar and top - run >= h.MARGIN,
            "bar": bar, "margin": round(top - run, 4)}


def rekey(b):
    n = 0
    for e in b.hippocampus.episodes:
        if e.permanent and "\n\nAssistant:" in e.text:
            q = e.text.split("User: ", 1)[1].split("\n\nAssistant:")[0]
            _, st = b.raw_cortex.read(f"User: {q}\n\n", b.base)
            e.key = b.raw_cortex.key_of(b.raw_cortex.matrix_delta(st, b.base)).cpu()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("homes", nargs="+")
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    out = {"model": getattr(c, "model_name", None), "host": socket.gethostname(), "homes": []}
    for home in a.homes:
        tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
        cp = os.path.join(tmp, "home")
        shutil.copytree(home, cp)
        try:
            b = Brain(cp, cortex=c, sleep_on_restart=False)
            assert b.key == "S0", b.key
            res = {"home": home, "n_episodes": len(b.hippocampus.episodes), "exchange": {}, "question": {}}
            for cue in CUES:
                res["exchange"][cue] = read(b, cue)
            res["n_rekeyed"] = rekey(b)
            for cue in CUES:
                res["question"][cue] = read(b, cue)
            out["homes"].append(res)
            print(home, res["n_rekeyed"], {k: (res[k]["What's your name?"]["winner"][:30], res[k]["What's your name?"]["passes"])
                                           for k in ("exchange", "question")}, flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
