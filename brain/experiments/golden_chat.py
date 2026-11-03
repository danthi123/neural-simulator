# Capability: Episodic recall
# Record: brain/experiments/results/2026-11-02-scale-ladder.txt
"""Record a seeded two-session conversation from the importable Brain, so a change can be checked byte-equal.

The fixture tests/fixtures/brain_golden_0.1b.json was recorded with the brain BEFORE sleep was wired in (origin/main
c231bf752); tests/test_brain_sleep.py replays it against today's brain with no consolidated adapters.

  python -m brain.experiments.golden_chat <home_dir> <out.json>
"""
import inspect
import json
import shutil
import sys

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex

TURNS = ["Hi! What's your name?", "My cat is called Pixel.", "I drive a red Volvo.", "What is my cat called?",
         "Tell me something you find interesting."]


def record(home, cortex):
    shutil.rmtree(home, ignore_errors=True)
    kw = {"sleep_on_restart": False} if "sleep_on_restart" in inspect.signature(Brain).parameters else {}
    res = []
    for session in range(2):
        np.random.seed(1234 + session)
        torch.manual_seed(1234 + session)
        b = Brain(home, cortex=cortex, **kw)
        for t in TURNS:
            r, info = b.hear(t)
            res.append([t, r, info["recalled"], info["similarity"], info["stored"]])
        b.save()
    return res


if __name__ == "__main__":
    out = record(sys.argv[1], Cortex(device="cpu"))
    with open(sys.argv[2], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
