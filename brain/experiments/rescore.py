# Capability: Episodic recall
# Record: brain/experiments/strict_scorer.py (is_disclaimed, added 2026-11-14)
"""Re-score banked probe results with the current strict scorer, into a copy (the originals are never touched).

Rewrites, from each stored reply: oracle rows ("class"/"correct": hippo_gain_refit, episode_format_probe, speaker
contrast) via hippo_gain_refit.klass plus the echo split; arm rows ({"q","kind","key", arm: {"reply","ok"}}: unslept,
mixed-day, confidence gate) for keyed told/para/new/slept questions via correct_answer. Never-told and self rows are
unchanged (they do not use correct_answer). Prints how many verdict fields changed per file.

  python -m brain.experiments.rescore <src_dir> <dst_dir>
"""
import json
import os
import shutil
import sys

from brain.experiments.facts import FACTS
from brain.experiments.facts_bank import DEV_FACTS36, FACTS48
from brain.experiments.hippo_gain_refit import klass
from brain.experiments.strict_scorer import correct_answer, is_echo

TOLD = {k: s for s, _, k in FACTS}
TOLD.update({keys[0]: s for s, _, _, _, keys in DEV_FACTS36})
TOLD.update({keys[0]: s for s, _, _, _, keys in FACTS48})


def rescore_rows(rows):
    changed = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = r.get("key")
        if "class" in r and "reply" in r and k:
            cl = klass(r["reply"], k, r["q"], TOLD.get(k))
            if cl == "key_other" and is_echo(r["reply"], k, TOLD.get(k)):
                cl = "echo"
            changed += cl != r["class"]
            r["class"], r["correct"] = cl, cl == "right"
            continue
        if not k or r.get("kind") in ("never", "self"):
            continue
        for v in r.values():
            if isinstance(v, dict) and "reply" in v and "ok" in v:
                ok = bool(correct_answer(v["reply"], k, r["q"], TOLD.get(k)))
                changed += ok != v["ok"]
                v["ok"] = ok
    return changed


def main():
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for root, _, files in os.walk(dst):
        for f in sorted(files):
            if not f.endswith(".json") or f.endswith(".prov.json"):
                continue
            p = os.path.join(root, f)
            try:
                d = json.load(open(p))
            except ValueError:
                continue
            if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
                continue
            n = rescore_rows(d["rows"])
            d["rescored"] = "strict_scorer 2026-11-14 (is_disclaimed)"
            json.dump(d, open(p, "w"), indent=1)
            if n:
                print(f"{os.path.relpath(p, dst)}: {n} verdicts changed")


if __name__ == "__main__":
    main()
