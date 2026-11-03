# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-false-recall-feeling-of-knowing/README.md
"""Dev only (fitting set, seed 42): what does the slept cortex say when the read-out gates a question to 'not known'?"""
import sys

import numpy as np
import torch

import brain.brain as bb
from brain.cortex import Cortex
from brain.experiments.abstain_in_brain import FIT, score

torch.set_num_threads(3)
home = sys.argv[1]
c = Cortex(device="cpu")
b = bb.Brain(home, cortex=c, sleep_seed=42, abstain=True)
b.know_bar = {"hippocampus": 1e9, "cortex": 1e9}   # force the not-known path for every question
snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
base = [x.clone() for x in b.base]
VARIANTS = {
    "g0": "",
    "g1": bb.NOT_KNOWN,
    "g2": "System: {name} has no memory of being told about this.\n\n",
}
qs = [x for x in FIT if x[2] == "never"] + [FIT[0], FIT[3], FIT[13]] + \
     [("What is the capital of France?", None, "gen"), ("How many legs does a spider have?", None, "gen")]
for v, line in VARIANTS.items():
    b.abstain = line is not None
    bb.NOT_KNOWN = bb.NOT_KNOWN if line is None else line
    for hip_off in (False,):
        tot = {"false_recall": 0, "abstained": 0}
        for qi, (q, k, kind) in enumerate(qs):
            for e, (st, us, lu) in zip(b.hippocampus.episodes, snap):
                e.strength, e.uses, e.last_used = st, us, lu
            b.state = [x.clone() for x in base]
            b.set_lesions(hippocampus_off=hip_off)
            np.random.seed(42000 + qi)
            torch.manual_seed(42000 + qi)
            r, _ = b.hear(q)
            s = score(r, k, None)
            if kind == "never":
                tot["false_recall"] += s["false_recall"]
                tot["abstained"] += s["abstained"]
            print(v, "hip_off" if hip_off else "hip_on", kind, "|", q, "->", r, flush=True)
        print("TOTAL", v, hip_off, tot, flush=True)
