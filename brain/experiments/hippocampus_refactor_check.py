# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-meaning-selection-realpath.md
"""Old Hippocampus (8c52620b9) vs refactored: identical recall decisions and similarities on 24 facts x 4 cue sets.

  (old.py = brain/hippocampus.py at commit 8c52620b9)  python brain/experiments/hippocampus_refactor_check.py <old.py>
"""
import importlib.util
import sys

import torch

from brain.brain import BIRTH
from brain.cortex import Cortex
from brain.experiments.facts_bank import FACTS48, NEVER_TOLD
from brain.hippocampus import Hippocampus

spec = importlib.util.spec_from_file_location("hb", sys.argv[1])
hb = importlib.util.module_from_spec(spec)
sys.modules["hb"] = hb
spec.loader.exec_module(hb)
torch.set_num_threads(2)
c = Cortex(model_path="/home/dant123/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
_, base = c.read(BIRTH, None)
base = [s.float() for s in base]
h0, h1 = hb.Hippocampus(), Hippocampus()
for f in FACTS48[:24]:
    h0.store(c, base, f[0])
    h1.store(c, base, f[0])
n = same = 0
for q in [f[j] for f in FACTS48 for j in (1, 2, 3)] + NEVER_TOLD:
    e0, s0 = h0.recall(c, base, q)
    e1, s1 = h1.recall(c, base, q)
    n += 1
    same += (e0 is None) == (e1 is None) and (e0 is None or e0.text == e1.text) and abs(s0 - s1) < 1e-6
print(f"identical {same}/{n}")
