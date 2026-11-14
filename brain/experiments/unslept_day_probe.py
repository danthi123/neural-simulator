# Capability: Episodic recall
# Record: brain/research/2026-11-14-key-default-1.5b.md (pre-registration, part A)
"""Unslept day: the 12 FACTS told to a live Brain, nothing slept, then every question asked of a fresh Brain on that home
(sleep_on_restart=False), per arm (mixed_day_probe.ARM_SPECS; default c, s, r). Questions: 12 told, 12 PARA, 12
never-told, 3 self questions. Scoring as mixed_day_probe (correct_answer; never = any FACTS key; self = name word).
Sampled replies seeded per question by --seed (as the live probes), so seeds matter. CPU is fine (no sleep).

  python -m brain.experiments.unslept_day_probe --home <dir> --model 1.5b --seed 42 --out <json> [--arms c,s,r]
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.confidence_gate_probe import fresh
from brain.experiments.confidence_probe import NEVER_TOLD_EXTRA
from brain.experiments.facts import FACTS, PARA
from brain.experiments.mixed_day_probe import ARM_SPECS, SELF_Q
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.experiments.strict_scorer import correct_answer, has_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="c,s,r")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-q", type=int, default=0, help="smoke only: first N questions per kind")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    home = f"{a.home}/unslept"
    shutil.rmtree(home, ignore_errors=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(home, cortex=c, sleep_on_restart=False, sleep_seed=a.seed)
    day = [{"you": s, "brain": b.hear(s)[0]} for s, _, _ in FACTS]
    b.save()
    n_pending = len(b.pending())
    b.release()
    keys = [k for _, _, k in FACTS]
    qs = [(q, k, "told", s) for s, q, k in FACTS] + [(p, f[2], "para", f[0]) for p, f in zip(PARA, FACTS)] + \
         [(q, None, "never", None) for q in NEVER_TOLD + NEVER_TOLD_EXTRA] + [(q, w, "self", None) for q, w in SELF_Q]
    if a.max_q:
        qs = [x for kind in ("told", "para", "never", "self") for x in [y for y in qs if y[2] == kind][:a.max_q]]
    names = a.arms.split(",")
    rows = []
    for q, key, kind, told in qs:
        row = {"q": q, "kind": kind, "key": key}
        for n in names:
            hip_off, hk, bar, gain = ARM_SPECS[n]
            bb = fresh(home, c, a.seed, None, hip_off, hk, bar, gain)
            assert bb.meta["sleeps"] == 0, bb.meta
            reply, info = bb.hear(q)
            if kind == "self":
                ok = key in reply.lower()
            elif key is None:
                ok = any(has_key(reply, k) for k in keys)
            else:
                ok = bool(correct_answer(reply, key, q, told))
            row[n] = {"reply": reply, "recalled": info["recalled"], "ok": ok, "gain": bb.hippocampus.GAIN,
                      "key": bb.key}
            bb.release()
        rows.append(row)
        print(a.seed, kind, *[row[n]["ok"] for n in names], repr(row[names[-1]]["reply"][:60]), flush=True)
    shutil.rmtree(home, ignore_errors=True)
    json.dump({"model": c.model_name, "seed": a.seed, "arms": names, "day": day, "n_pending": n_pending, "rows": rows,
               "host": os.uname().nodename, "elapsed_s": round(time.time() - t0, 1)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
