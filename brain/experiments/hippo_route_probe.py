# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-live-brain-1.5b.md (caveat 1: 6/12 told questions misrouted at 1.5B);
#         brain/experiments/results/2026-11-03-meaning-selection-realpath.md (S0 vs S_all at 0.1B / 0.4B, offline)
"""Hippocampal routing at one cortex size: which stored episode each cue retrieves, for every candidate key.

No sleep, no speech: only cortex reads (deterministic), so it runs on CPU at 1.5B. Banks:
  test  the 12 live-probe FACTS stored exactly as the live Brain stores them (a fresh Brain's 3 permanent self
        episodes + `hippocampus.store` of each telling from the birth state); cues = 12 told questions, 12 PARA,
        4 sleep_probe NEVER_TOLD + 12 facts_bank NEVER_TOLD.
  dev   per seed 42/43/44: Random(seed) shuffle of DEV_FACTS36, first 12 stored (same way), cues = their orig /
        para_a / para_b questions, the 24 untold orig questions and DEV_NEVER_TOLD. Used only to fit bars.
Keys (conventions of brain/hippocampus_keys.py): one-layer keys S<l>/att<l>/ffn<l>/last<l>/pool<l> = raw read-out
centred on the stored keys' mean, cosine; *_all = every layer's read-out L2-normalised and concatenated, centred,
cosine (= KeyedHippocampus S_all). Scores are strength-weighted as recall() does (tanh of effective strength).
Fidelity: S0 is also scored through the live `Hippocampus.scores` path and must match the read-out S0.

  python -m brain.experiments.hippo_route_probe --model 1.5b --out <json> [--threads 16]
"""
import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time

import torch
import torch.nn.functional as F

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_bank import DEV_FACTS36, DEV_NEVER_TOLD, NEVER_TOLD as BANK_NEVER
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.hippocampus_keys import read_activity
from brain.plastic_cortex import PlasticCortex

KINDS = ("S", "att", "ffn", "last", "pool")


def readouts(pc, base, text):
    """Per-layer read-outs of one turn (text already carries the speaker and the blank line)."""
    act = read_activity(pc, base, text)
    return {k: [t.detach().cpu() for t in act[k]] for k in KINDS}


def key_vec(act, variant):
    kind, rest = (variant[:-4], "all") if variant.endswith("_all") else \
        next((k, variant[len(k):]) for k in KINDS if variant.startswith(k) and variant[len(k):].isdigit())
    if rest == "all":
        return torch.cat([F.normalize(b.flatten().float(), dim=0) for b in act[kind]])
    return act[kind][int(rest)].flatten().float()


def stored(ep_acts, variant):
    """Centred, normalised stored keys and their mean (the centre a cue is moved by)."""
    K = torch.stack([key_vec(a, variant) for a in ep_acts])
    mu = K.mean(0)
    return F.normalize(K - mu, dim=1), mu


def scores(Kmu, cue_act, variant, w):
    K, mu = Kmu
    s = K @ F.normalize(key_vec(cue_act, variant) - mu, dim=0)
    return (s * w).tolist()


def bank(c, pc, home, statements, cues):
    """Store `statements` as the live Brain does; return episode texts, weights, read-outs, live S0 rows, cue reads."""
    shutil.rmtree(home, ignore_errors=True)
    b = Brain(home, cortex=c, sleep_on_restart=False)
    for s in statements:
        b.hippocampus.store(b.cortex, b.base, s, speaker="User")   # = Brain.hear's store (from the birth state)
    eps = b.hippocampus.episodes
    now = time.time()
    w = torch.tensor([math.tanh(b.hippocampus.effective_strength(e, now)) for e in eps])
    ep_acts = [readouts(pc, b.base, f"{e.text}\n\n") for e in eps]
    rows = []
    for q, kind, right in cues:
        live = b.hippocampus.scores(b.cortex, b.base, q, now=now).tolist()   # live S0 path (what recall competes on)
        rows.append({"q": q, "kind": kind, "right": right, "live_S0": [round(x, 5) for x in live],
                     "_act": readouts(pc, b.base, f"User: {q}\n\n")})
    texts = [e.text for e in eps]
    b.release()
    return texts, w, ep_acts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dev-seeds", type=int, nargs="*", default=[42, 43, 44])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    pc = PlasticCortex(c)   # fp32 cortex: z is shared, not copied
    L = c.n_layers
    variants = [f"{k}{l}" for k in KINDS for l in range(L)] + [f"{k}_all" for k in KINDS]
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    out = {"model": c.model_name, "n_layers": L, "variants": variants, "banks": {}}
    plans = [("test", [s for s, _, _ in FACTS],
              [(q, "orig", f"User: {s}") for s, q, _ in FACTS] + [(p, "para", f"User: {f[0]}") for p, f in zip(PARA, FACTS)]
              + [(q, "never", None) for q in NEVER_TOLD + BANK_NEVER])]
    for seed in a.dev_seeds:
        facts = list(DEV_FACTS36)
        random.Random(seed).shuffle(facts)
        told, untold = facts[:12], facts[12:]
        plans.append((f"dev{seed}", [f[0] for f in told],
                      [(f[1], "orig", f"User: {f[0]}") for f in told] + [(f[2], "para_a", f"User: {f[0]}") for f in told]
                      + [(f[3], "para_b", f"User: {f[0]}") for f in told] + [(f[1], "untold", None) for f in untold]
                      + [(q, "never", None) for q in DEV_NEVER_TOLD]))
    for name, statements, cues in plans:
        texts, w, ep_acts, rows = bank(c, pc, f"{tmp}/{name}", statements, cues)
        for r in rows:
            r["scores"] = {}
        for v in variants:
            Kmu = stored(ep_acts, v)
            for r in rows:
                r["scores"][v] = [round(x, 5) for x in scores(Kmu, r["_act"], v, w)]
        for r in rows:
            del r["_act"]
        out["banks"][name] = {"episodes": texts, "weights": w.tolist(), "rows": rows}
        print(name, "done", round(time.time() - t0), "s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
