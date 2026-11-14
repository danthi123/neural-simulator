# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-13-mixed-day-1.5b.md (scorer credited general statements about the key)
"""Re-score told + paraphrase rows of saved probe JSONs with the current strict_scorer.correct_answer.

Writes copies under <dir>/rescored/ with the same names, so the run's own table reads them unchanged:
  python -m brain.experiments.rescore_told <dir> --glob 'mixday_s*.json'
Handles confidence_gate_probe / mixed_day_probe rows ({c,f}: {reply, ok}) and live_brain_probe arms (reply, correct).
Never-told rows are left as recorded. Prints, per arm, how many told/para credits changed.
"""
import argparse
import glob
import json
import os

from brain.experiments.facts import FACTS
from brain.experiments.strict_scorer import correct_answer

TOLD = {k: s for s, _, k in FACTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", required=True)
    a = ap.parse_args()
    out = os.path.join(a.dir, "rescored")
    os.makedirs(out, exist_ok=True)
    lost = {}
    for p in sorted(glob.glob(os.path.join(a.dir, a.glob))):
        d = json.load(open(p))
        if "arms" in d:
            for arm, x in d["arms"].items():
                for r in x["rows"]:
                    kind = r.get("kind", r.get("set"))   # told-gate rows: set told | told_para, plus cond
                    if kind in ("told", "para", "told_para"):
                        k = r.get("key") or _key(r, d)
                        new = bool(correct_answer(r["reply"] or "", k, r["q"], TOLD[k]))
                        old = bool(r.get("correct", r.get("hit")))   # 11-03 0.1B live rows: 'hit' (strict score)
                        key = arm + (f"/{r['cond']}" if "cond" in r else "")
                        lost[key] = lost.get(key, 0) + (old and not new) - (new and not old)
                        r["correct"] = new
                k_told = lambda r: r.get("kind", r.get("set")) == "told"
                k_para = lambda r: r.get("kind", r.get("set")) in ("para", "told_para")
                x["recall_correct"] = sum(bool(r.get("correct")) for r in x["rows"] if k_told(r))
                x["paraphrase_correct"] = sum(bool(r.get("correct")) for r in x["rows"] if k_para(r))
        else:
            for r in d["rows"]:
                if r["kind"] in ("told", "para"):
                    for arm in ("c", "f"):
                        new = bool(correct_answer(r[arm]["reply"], r["key"], r["q"], TOLD[r["key"]]))
                        lost[arm] = lost.get(arm, 0) + (r[arm]["ok"] and not new) - (new and not r[arm]["ok"])
                        r[arm]["ok"] = new
        json.dump(d, open(os.path.join(out, os.path.basename(p)), "w"), indent=1)
    print("credits removed by the current scorer (told + para, all seeds):", lost)


def _key(r, d):
    """live_brain_probe rows carry no key: told/para rows follow FACTS order."""
    for s, q, k in FACTS:
        if r["q"] == q:
            return k
    from brain.experiments.facts import PARA
    return FACTS[PARA.index(r["q"])][2]


if __name__ == "__main__":
    main()
