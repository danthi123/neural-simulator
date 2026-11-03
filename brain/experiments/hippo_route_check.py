# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-03-hippo-routing-1.5b.md (the fix this checks)
"""Fidelity of the fix path: a real Brain(key=..., key_bar=...) stores the 12 live-probe facts and recall() is asked
every told / PARA / never-told cue. Counts the right, wrong and no recall per kind. Deterministic, no sleep, no speech.

  python -m brain.experiments.hippo_route_check --model 1.5b --key S_all --key-bar size --out <json>
"""
import argparse
import json
import os
import shutil
import tempfile

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_bank import NEVER_TOLD as BANK_NEVER
from brain.experiments.sleep_probe import NEVER_TOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--key", default="S0")
    ap.add_argument("--key-bar", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    home = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    b = Brain(f"{home}/h", cortex=c, sleep_on_restart=False, key=a.key, key_bar=a.key_bar)
    for s, _, _ in FACTS:
        b.hippocampus.store(b.cortex, b.base, s, speaker="User")
    cues = [(q, "orig", f"User: {s}") for s, q, _ in FACTS] + [(p, "para", f"User: {f[0]}") for p, f in zip(PARA, FACTS)] \
        + [(q, "never", None) for q in NEVER_TOLD + BANK_NEVER]
    rows = []
    for q, kind, right in cues:
        saved = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
        ep, sim = b.hippocampus.recall(b.cortex, b.base, q, question=True)
        for e, (st, u, lu) in zip(b.hippocampus.episodes, saved):   # each cue on a fresh store (as the probe's fresh Brain)
            e.strength, e.uses, e.last_used = st, u, lu
        got = None if ep is None else ep.text
        rows.append({"q": q, "kind": kind, "recalled": got, "sim": round(sim, 4),
                     "outcome": "none" if got is None else ("right" if got == right else "wrong")})
    b.release()
    shutil.rmtree(home, ignore_errors=True)
    tab = {k: {o: sum(r["kind"] == k and r["outcome"] == o for r in rows) for o in ("right", "wrong", "none")}
           for k in ("orig", "para", "never")}
    print(a.key, a.key_bar, json.dumps(tab))
    json.dump({"model": c.model_name, "key": a.key, "key_bar": a.key_bar, "table": tab, "rows": rows},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
