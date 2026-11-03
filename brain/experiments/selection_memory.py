# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-meaning-selection.md (skeptic: S_all duplicated the memory)
"""Bytes per stored episode, before (11-03 KeyedHippocampus, commit 8c52620b9) and after (key read from the delta).

  git show 8c52620b9:brain/hippocampus_keys.py > <old.py>
  python -m brain.experiments.selection_memory --old <old.py> --model <pth> [--model <pth>] --out <json>
"""
import argparse
import importlib.util
import json
import os

import torch

from brain.brain import BIRTH
from brain.cortex import Cortex
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus, episode_bytes
from brain.plastic_cortex import PlasticCortex

FACTS = ["My cat is called Pixel.", "I was born in Lyon.", "My dog is named Biscuit.", "I drive a red Volvo."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("hk_before", a.old)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    out = {}
    for m in a.model:
        c = Cortex(model_path=m)
        pc = PlasticCortex(c, rank=1, layers=[])
        pc.enabled = False
        _, base = c.read(BIRTH, None)
        base = [s.float() for s in base]
        r = {}
        for name, h in (("default_Hippocampus", Hippocampus()),
                         ("before_S0", old.KeyedHippocampus(pc, key="S0")), ("before_S_all", old.KeyedHippocampus(pc, key="S_all")),
                         ("after_S0", KeyedHippocampus(key="S0")), ("after_S_all", KeyedHippocampus(key="S_all"))):
            b = [episode_bytes(h.store(c, base, f)) for f in FACTS]
            ep = h.episodes[0]
            r[name] = {"bytes_per_episode": sum(b) / len(b),
                       "delta_bytes": sum(d.numel() * d.element_size() for d in ep.delta if d is not None),
                       "key_bytes": ep.key.numel() * ep.key.element_size() if ep.key is not None else 0,
                       "activity_bytes": sum(t.numel() * t.element_size() for t in (getattr(ep, "activity", None) or []))}
        out[os.path.basename(m)] = r
        print(m, json.dumps(r, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
