# Capability: Sleep consolidation
# Record: brain/research/2026-11-14-near-miss-guard-1.5b.md (frame choice, seed 7 only: not a test seed)
"""Prototype for the near-miss guard: which rewrite frame gives usable near-miss questions? For each of the 12 told
facts and each frame (N1-N3) it logs the cortex's own question, its near-miss rewrites and the pretrained reply.
Score per frame: usable = a question, not the own question, and naming no told key word, in the first person (my / I / me). Not a result.

  python -m brain.experiments.near_miss_proto --model 1.5b --seed 7 --out proto.json
"""
import argparse
import re
import json

import torch

from brain.brain import BIRTH, Brain
from brain.eval.open_conv import invented
from brain.experiments.abstain_in_brain import keys_in
from brain.cortex import Cortex, resolve_model
from brain.experiments.facts import FACTS
from brain.near_miss import REWRITE, frame_cues, guard_texts, near_miss_cues, topic_cues


# the told facts' topic words (instrument only: how often a frame cue asks about a told topic)
TOPICS = ("cat", "colour", "color", "born", "sister", "dog", "car", "drive", "friend", "work", "brother", "allerg",
          "grandmother", "grandma", "son")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", default=",".join(REWRITE))
    ap.add_argument("--teacher-context", action="store_true",
                    help="replies from the cortex after it has read all 12 tellings (the teacher night's layout)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    _, base = c.read(BIRTH, None)
    keys = [k for _, _, k in FACTS]
    ctx = None
    if a.teacher_context:
        ctx = (f"User: {Brain.TEACHER_OPEN} " + " ".join(s for s, _, _ in FACTS) + f"\n\nAssistant: {Brain.TEACHER_ACK}\n\n")
    res = {}
    if a.frames == "T":   # paired night topic frame: cues on the told topics, teacher replies in context
        cues = topic_cues(c, base, [f"User: {s}" for s, _, _ in FACTS], k=a.k, seed=a.seed)
        texts = guard_texts(c, base, cues, context=ctx)
        rows = []
        for q, t in zip(cues, texts):
            r = t[1].strip() if isinstance(t, tuple) else t
            rows.append({"q": q, "reply": r, "named": keys_in(r), "invented": invented(r)})
            print("T", q, "->", r, "| named", keys_in(r), "invented", invented(r), flush=True)
        print("T cues", len(rows), "named", sum(bool(x["named"]) for x in rows), "invented", sum(x["invented"] for x in rows),
              flush=True)
        json.dump({"T": rows}, open(a.out, "w"), indent=1)
        return
    if a.frames == "F":
        cues = frame_cues(c, base, k=a.k, seed=a.seed)
        texts = guard_texts(c, base, cues)
        told_topic = [any(w in q.lower() for w in TOPICS) for q in cues]
        for q, t, tt in zip(cues, texts, told_topic):
            print("F", "TOLD-TOPIC" if tt else "", repr(t), flush=True)
        print("F cues", len(cues), "told-topic", sum(told_topic), flush=True)
        json.dump({"F": {"cues": cues, "texts": texts, "told_topic": told_topic}}, open(a.out, "w"), indent=1)
        return
    for frame in a.frames.split(","):
        rows, usable, total = [], 0, 0
        for i, (s, _, _) in enumerate(FACTS):
            own, cues = near_miss_cues(c, base, s, k=a.k, frame=frame, seed=a.seed * 100 + i)
            texts = guard_texts(c, base, [own] + cues, context=ctx)
            rep = [t[1].strip() if isinstance(t, tuple) else t.split("Assistant:", 1)[1].strip() for t in texts]
            own_named = FACTS[i][2] in keys_in(rep[0])
            inv = [invented(r) for r in rep[1:]]
            ok = [not any(k in q.lower() for k in keys) and bool(re.search(r"\b(my|i|me)\b", q.lower())) for q in cues]
            usable += sum(ok)
            total += a.k
            rows.append({"statement": s, "own": own, "cues": cues, "usable": ok, "replies": rep, "own_named": own_named,
                         "rewrite_invented": inv})
            print(frame, s, "|", own, "->", rep[0], "|", list(zip(cues, rep[1:])), flush=True)
        oc = sum(r["own_named"] for r in rows)
        ninv = sum(sum(r["rewrite_invented"]) for r in rows)
        ncue = sum(len(r["cues"]) for r in rows)
        res[frame] = {"usable": usable, "of": total, "own_named": oc, "cues": ncue, "rewrite_invented": ninv, "rows": rows}
        print(frame, "usable", usable, "of", total, "own named", oc, "/12 rewrites invented", ninv, "/", ncue, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
