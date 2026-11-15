# Capability: Self and identity
# Record: brain/experiments/results/2026-11-15-weekly-1.5b-milestone.md (addendum: own name on day 1); prereg brain/research/2026-11-15-own-name-day1-probe-1.5b.md
"""Why does the recalled self episode not carry the own name on day 1? No training, no sleep, CPU.

A fresh brain (birth state, the three permanent self episodes) hears one first-turn question. Through the brain's own
recall (Brain.hippocampus.recall, as Brain.hear does), the recalled episode is reinstated at a grid of gains, and
the reply's next-token distribution is read:
  - p_first: top-5 first tokens after "Assistant:";
  - p_nova:  P(name token | "Assistant: My name is") and its rank, top-5 there;
  - samples: n replies with the brain's own speak() defaults (temperature 1, top_p 0.3, penalties), share saying the
    name.
Arms: prompt live ("Hi! What's your name?") vs bare ("What's your name?"); delta full (as stored) vs answer-only
(d(question+answer) - d(question), norm-matched to the full delta). Gain 0 = no reinstatement (hippocampus off).

  python -m brain.experiments.own_name_probe --out <json> [--samples 12]
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model

GAINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
PROMPTS = {"live": "Hi! What's your name?", "bare": "What's your name?"}


def top5(c, logits):
    p = torch.softmax(logits.float(), -1)
    v, i = p.topk(5)
    return [(c.pipe.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]


def answer_only(c, base, name):
    """d(QA) - d(Q) for the self episode, both read from the birth state as Hippocampus.store reads them."""
    _, qa = c.read(f"User: What's your name?\n\nAssistant: My name is {name}.\n\n", base)
    _, q = c.read("User: What's your name?\n\n", base)
    return [(qa[i] - q[i]).clone() if i % 3 == 1 else None for i in range(len(base))]


def norm(delta):
    return float(torch.sqrt(sum((d.float() ** 2).sum() for d in delta if d is not None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gains", default=",".join(str(g) for g in GAINS))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    gains = [float(g) for g in a.gains.split(",")]
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    b = Brain(f"{root}/home", cortex=c, sleep_on_restart=False)
    name = b.meta["name"]
    name_tok = c.pipe.encode(f" {name}")
    assert all(torch.equal(x, y) for x, y in zip(b.state, b.base)), "day-1 turn-1 state is not the birth state"
    rows, recalls = [], {}
    for pk, prompt in PROMPTS.items():
        # the brain's own recall, as hear() does it (one read; recall's use boost does not change reinstatement)
        saved = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
        ep, sim = b.hippocampus.recall(b.cortex, b.base, prompt, question=prompt.endswith("?"))
        for e, s in zip(b.hippocampus.episodes, saved):
            e.strength, e.uses, e.last_used = s
        recalls[pk] = {"recalled": ep.text if ep else None, "similarity": round(sim, 4)}
        if ep is None:
            continue
        full = ep.delta
        ao = answer_only(c, b.base, name)
        ao = [None if d is None else d * (norm(full) / max(norm(ao), 1e-8)) for d in ao]
        for dk, delta in (("full", full), ("answer_only", ao)):
            for g in gains:
                if dk == "answer_only" and g == 0.0:
                    continue
                active = c.add_delta(b.state, delta, g) if g else b.state
                logits, st = b.cortex.read(f"User: {prompt}\n\nAssistant:", active)
                first = top5(c, logits)
                l2, st2 = b.cortex.read(f"User: {prompt}\n\nAssistant: My name is", active)
                p2 = torch.softmax(l2.float(), -1)
                rank = int((p2 > p2[name_tok[0]]).sum()) + 1
                lp = 0.0   # log P of the whole name continuation " Nova." under teacher forcing
                lg, s = l2, st2
                for t in c.pipe.encode(f" {name}."):
                    lp += float(torch.log_softmax(lg.float(), -1)[t])
                    lg, s = b.cortex.forward([t], s)
                replies = []
                for k in range(a.samples):
                    torch.manual_seed(1000 + k)
                    np.random.seed(1000 + k)   # speak() samples with numpy on the CPU
                    r, _ = b.cortex.speak(logits, st, max_tokens=24)
                    replies.append(r)
                row = {"prompt": pk, "delta": dk, "gain": g, "first_top5": first,
                       "p_name": round(float(p2[name_tok[0]]), 4), "rank_name": rank, "top5_after_my_name_is": top5(c, l2),
                       "logp_name_cont": round(lp, 3), "samples_with_name": sum(name.lower() in r.lower() for r in replies),
                       "n_samples": a.samples, "replies": replies}
                rows.append(row)
                print(pk, dk, g, "p_name", row["p_name"], "rank", rank, "first", first[:2], "samples",
                      row["samples_with_name"], "/", a.samples, "|", replies[0][:50], flush=True)
    b.release()
    shutil.rmtree(root, ignore_errors=True)
    json.dump({"model": c.model_name, "host": os.uname().nodename, "name": name, "name_tokens": name_tok,
               "recalls": recalls, "gain_by_size_default": Brain.GAIN_BY_SIZE, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
