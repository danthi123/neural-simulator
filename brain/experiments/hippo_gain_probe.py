# Capability: Episodic recall
# Record: brain/experiments/results/2026-11-03-live-brain-1.5b.md (arm b: right episode recalled on 6/12, still wrong
#         answers on about half of those); brain/hippocampus.py GAIN = 4.0 (grid 11-02 at 0.1B)
"""Does the RIGHT episode, reinstated at the hippocampus gain, make the unslept cortex answer? Oracle routing.

The 12 live-probe facts are stored as the live Brain stores them; for each told question the right episode is
reinstated into a fresh Brain's state at each gain (0 = no reinstatement), then the pretrained cortex answers
(greedy, at most 40 tokens). Scored with strict_scorer.correct_answer. No sleep, deterministic, CPU.

  python -m brain.experiments.hippo_gain_probe --model 1.5b --out <json> [--gains 0 1 2 3 4]
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
from brain.experiments.strict_scorer import correct_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gains", type=float, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    home = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    b = Brain(f"{home}/h", cortex=c, sleep_on_restart=False)
    eps = [b.hippocampus.store(b.cortex, b.base, s, speaker="User") for s, _, _ in FACTS]
    rows = []
    for (s, q, k), p, ep in zip(FACTS, PARA, eps):
        for kind, cue in (("orig", q), ("para", p)):
            for g in a.gains:
                active = c.add_delta(b.state, ep.delta, g) if g else b.state
                logits, st = c.read(f"User: {cue}\n\nAssistant:", active)
                reply, _ = c.speak(logits, st, max_tokens=40, temperature=0)
                rows.append({"q": cue, "kind": kind, "gain": g, "reply": reply,
                             "correct": bool(correct_answer(reply, k, cue, s))})
        print(q, [(r["gain"], r["correct"]) for r in rows[-2 * len(a.gains):]], flush=True)
    b.release()
    shutil.rmtree(home, ignore_errors=True)
    tab = {kind: {str(g): sum(r["correct"] for r in rows if r["kind"] == kind and r["gain"] == g) for g in a.gains}
           for kind in ("orig", "para")}
    print("correct of 12 by gain:", tab)
    json.dump({"model": c.model_name, "gains": a.gains, "table": tab, "rows": rows,
               "elapsed_s": round(time.time() - t0, 1)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
