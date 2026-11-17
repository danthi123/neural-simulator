# Capability: Metacognition
# Record: brain/experiments/results/2026-11-17-confab-m1-1.5b.md (M1 falsified at 1.5B: untold attributes of told
#         topics look known to every read-out)
"""In-context abstention ceiling: does the pretrained cortex itself decline a never-told question when everything it was
told is plainly in its context?

Birth context, then all 12 FACTS told in ONE user turn ("User: A few things about me. <fact> <fact> ...") and one
acknowledgement (one ack, not twelve: twelve identical acks are copied back by 0.1B),
then each fresh M1 question (abstain-v2 fresh v2 + M1 v3: told rewordings, untold topics, untold attributes of told
topics, general) asked from that same state. No hippocampus, no sleep, pretrained synapses. Greedy, so deterministic:
one run per cortex (seeds do not apply). Scored with the teacher-side marks of brain/eval/open_conv.py: told = the key
is named; never = invented (a confident answer, no "don't know", not a question back), and the told name reused.

If the cortex invents here too, no read-out of its own comprehension can supply "I was not told this": the ceiling
for any signal-only method at this size is this number.

  python -m brain.experiments.incontext_abstain_probe --model <path.pth> --out <json>
"""
import argparse
import json
import time

import torch

from brain.brain import BIRTH
from brain.cortex import DEFAULT_MODEL, Cortex, resolve_model
from brain.eval.open_conv import invented
from brain.experiments.abstain_in_brain import keys_in
from brain.experiments.confab_m1 import FRESH, kind
from brain.experiments.facts import FACTS

ACK = "Thanks for telling me. I'll keep that in mind."


@torch.no_grad()
def answer(c, state, q, n=40):
    logits, st = c.read(f"User: {q}\n\nAssistant:", state)
    toks = []
    for _ in range(n):
        t = int(torch.argmax(logits))
        if t == 0:
            break
        toks.append(t)
        if "\n\n" in c.pipe.decode(toks):
            toks.pop()
            break
        logits, st = c.forward([t], st)
    return c.pipe.decode(toks).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--layout", default="turn", choices=["turn", "episodes"],
                    help="turn: all facts in one user turn + one ack (the 11-17 run); episodes: each telling as the "
                         "hippocampus stores it ('User: <fact>' + blank line), in order, no ack (the teacher night's context)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    _, st = c.read(BIRTH, None)
    if a.layout == "turn":
        _, st = c.read("User: A few things about me. " + " ".join(s for s, _, _ in FACTS) + f"\n\nAssistant: {ACK}\n\n", st)
    else:
        _, st = c.read("".join(f"User: {s}\n\n" for s, _, _ in FACTS), st)
    rows = []
    for set_name, items in FRESH.items():
        for q, key, allowed in items:
            r = answer(c, st, q)
            row = {"set": set_name, "q": q, "key": key, "allowed": allowed, "reply": r}
            k = kind(row)
            names = [n for n in keys_in(r) if n != allowed]
            echo = "keep that in mind" in r.lower() or "thanks for telling" in r.lower()
            row.update(kind=k, told_ok=bool(key) and key in keys_in(r), echo=echo, invented=invented(r) and not echo,
                       reused=bool(names))
            rows.append(row)
            print(set_name, row["told_ok"] if k == "told" else row["invented"], "|", q, "|", r, flush=True)
    summ = {}
    for name, sel in (("told", lambda r: r["kind"] == "told"), ("never", lambda r: r["kind"] == "never"),
                      ("attr", lambda r: r["kind"] == "never" and r["allowed"]),
                      ("topic", lambda r: r["kind"] == "never" and not r["allowed"])):
        rs = [r for r in rows if sel(r)]
        summ[name] = {"n": len(rs), "told_ok": sum(r["told_ok"] for r in rs), "invented": sum(r["invented"] for r in rs),
                      "reused": sum(r["reused"] for r in rs), "echo": sum(r["echo"] for r in rs)}
    res = {"model": a.model, "device": a.device, "layout": a.layout, "summary": summ, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(summ), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
