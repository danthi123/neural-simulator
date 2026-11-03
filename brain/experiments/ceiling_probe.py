# Capability: Episodic recall
# Record: brain/experiments/results/2026-11-03-high-use-facts/README.md (next step 1)
"""Per-fact in-context ceiling: tell ONE fact in context (birth context + the told turn, no sleep, no hippocampus), ask the
original question and the reworded one. Greedy, so deterministic (no seed). Which facts can this cortex hold at all?

  python -m brain.experiments.ceiling_probe --model <path.pth> --out <json>
"""
import argparse
import json

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_extra import EXTRA, EXTRA_PARA
from brain.experiments.sleep_probe import ask
from brain.plastic_cortex import PlasticCortex

POOL = [(s, q, k, p) for (s, q, k), p in zip(FACTS + EXTRA, PARA + EXTRA_PARA)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=a.model)
    pc = PlasticCortex(c)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    rows = []
    for i, (s, q, k, p) in enumerate(POOL):
        _, st = c.read(f"User: {s}\n\nAssistant: Okay, I'll remember that.\n\n", base)
        st = [x.float() for x in st]
        a_q, a_p = ask(pc, st, q), ask(pc, st, p)
        rows.append({"key": k, "statement": s, "answer": a_q, "para_answer": a_p, "original": k in a_q.lower(),
                     "para": k in a_p.lower(), "from_original_12": i < len(FACTS)})
        print(k, rows[-1]["original"], rows[-1]["para"], "|", a_q, "|", a_p, flush=True)
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
