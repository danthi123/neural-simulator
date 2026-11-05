# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-live-brain-12facts.md
"""Rescore the live-brain 12-fact replies with the correct-perspective scorer (strict_scorer.correct_answer).

The six-seed result (results/2026-11-03-live-brain-12facts.md) was scored with `strict_scorer.score`, which voids
every first-person reply and credits tautologies ("Pixel is called Pixel."); its caveat section says so. This
re-analysis computes `correct_answer` for every told/para row of every arm, with the key and the told sentence from
FACTS by position (told rows and para rows are both in FACTS order; arm e is scored against the ORIGINAL keys and
sentences, as the probe does). False recall is unchanged. New files only: `rescored.json` and `table_correct.txt`
next to the seed JSONs; the seed JSONs and `table.txt` are never modified. `table_correct.txt` is
live_brain_table.py's P1-P4 logic with the metric names as parameters (the rescored counts under
recall_correct/paraphrase_correct). Pi writes no verdict: the table prints the P-checks and Claude rules on them.

  python -m brain.experiments.live_brain_rescore brain/experiments/results/2026-11-03-live-brain-12facts
"""
import argparse
import contextlib
import glob
import io
import json

from brain.experiments.facts import FACTS, PARA
from brain.experiments.live_brain_table import main as table
from brain.experiments.strict_scorer import correct_answer

# The rescored counts live under these names; false_recall / echo / the lenient counts are the stored ones.
METRICS = {"recall": "recall_correct", "paraphrase": "paraphrase_correct"}
KEPT = ("false_recall", "echo", "recall_lenient", "paraphrase_lenient", "false_lenient")


def rescore_rows(rows, facts=FACTS, para=PARA):
    """Told and para rows re-scored with correct_answer, key and told sentence from `facts` by position (both kinds
    are in FACTS order); never rows (false recall) are copied unchanged. Each row's question is checked against the
    FACTS/PARA entry at its position: a file whose rows are not in FACTS order errors here instead of being silently
    mis-mapped to the wrong key and sentence."""
    expect = {"told": [f[1] for f in facts], "para": list(para)}
    pos = {"told": 0, "para": 0}
    out = []
    for r in rows:
        r = dict(r)
        if r["kind"] in expect:
            i = pos[r["kind"]]
            pos[r["kind"]] += 1
            if i >= len(expect[r["kind"]]):
                raise ValueError(f"more {r['kind']} rows than FACTS entries")
            if r["q"] != expect[r["kind"]][i]:
                raise ValueError(f"{r['kind']} row {i}: question {r['q']!r} != {expect[r['kind']][i]!r} "
                                 f"— the rows are not in FACTS order")
            statement, _q, key = facts[i]
            r["correct"] = correct_answer(r["reply"], key, r["q"], statement)
        out.append(r)
    return out


def rescore_seed(obj, facts=FACTS):
    """One probe JSON -> the rescored shape: per arm the correct_answer counts (recall_correct / paraphrase_correct)
    and the stored metrics kept unchanged."""
    arms = {}
    for arm, r in obj["arms"].items():
        rows = rescore_rows(r["rows"], facts)
        n = lambda kind: sum(bool(x["correct"]) for x in rows if x["kind"] == kind)
        arms[arm] = {"recall_correct": n("told"), "paraphrase_correct": n("para"), **{k: r[k] for k in KEPT},
                     "rows": rows}
    return {"seed": obj["seed"], "arms": arms}


def main(d, pattern="seed*.json", facts=FACTS):
    """Rescore every seed JSON in d into d/rescored.json and write d/table_correct.txt (the P1-P4 table over the
    rescored counts). The seed JSONs and table.txt are read-only."""
    rescored = [rescore_seed(json.load(open(f)), facts) for f in sorted(glob.glob(d + "/" + pattern))]
    with open(d + "/rescored.json", "w") as fh:
        json.dump(rescored, fh, indent=1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        table(d, pattern="rescored.json", metrics=METRICS,
              header="rescored with strict_scorer.correct_answer (key + told from FACTS by position); false recall unchanged")
    with open(d + "/table_correct.txt", "w") as fh:
        fh.write(buf.getvalue())
    return rescored


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="seed*.json")
    a = ap.parse_args()
    main(a.dir, a.glob)