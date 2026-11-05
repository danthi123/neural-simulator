# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-live-brain-12facts.md
"""Read the per-seed live_brain_probe JSONs, print per-seed/per-arm tables and the pre-registered P1-P4 checks.

  python -m brain.experiments.live_brain_table brain/experiments/results/2026-11-03-live-brain-12facts
  1.5B (results/2026-11-03-live-brain-1.5b.md): ... <dir> --glob 'live15_s*.json' --scorer correct --p1-mean 4
  --scorer correct reads recall_correct / paraphrase_correct (strict_scorer.correct_answer) as recall / paraphrase.

`main(d, ...)` takes the metric names as parameters (which JSON field backs recall / paraphrase / false_recall /
echo / recall_lenient), so brain/experiments/live_brain_rescore.py reuses this exact P1-P4 logic on its rescored
counts. The old call `main(d)` prints exactly the committed table.txt, byte for byte (a test checks this); the CLI
keeps its 'scorer <x>' identification line by passing `header`. A JSON file may hold one seed object or a list of
them (rescored.json holds all seeds).
"""
import argparse
import glob
import json
import statistics as st

ARMS = ["a_intact_sleep", "b_intact_nosleep", "c_hippo_off", "d_hippo_off_consol_off", "e_shuffled_hippo_off"]
M = ["recall", "paraphrase", "false_recall", "echo"]
METRIC_FIELDS = {"recall": "recall", "paraphrase": "paraphrase", "false_recall": "false_recall", "echo": "echo",
                 "recall_lenient": "recall_lenient"}


def load(d, pattern):
    """seed -> arms from every JSON matching d/pattern (each file: one probe object or a list of them)."""
    out = {}
    for f in sorted(glob.glob(d + "/" + pattern)):
        obj = json.load(open(f))
        for o in obj if isinstance(obj, list) else [obj]:
            out[o["seed"]] = o["arms"]
    return out


def main(d, pattern="seed*.json", scorer="strict", p1_mean=2.0, metrics=None, header=None):
    """metrics: the JSON field behind each role (recall / paraphrase / false_recall / echo / recall_lenient); the
    P1-P4 checks and the printed table read those fields. header: an identification line printed first (the CLI
    passes 'scorer <x>'); with header=None the output is exactly table.txt's bytes (the old call)."""
    m = dict(METRIC_FIELDS, **(metrics or {}))
    if scorer == "correct":
        m["recall"], m["paraphrase"] = "recall_correct", "paraphrase_correct"
    R = load(d, pattern)
    if header is not None:
        print(header)
    seeds = sorted(R)
    print("seeds", seeds)
    for role in M + ["recall_lenient"]:
        print(f"\n{role} (per seed {seeds}, mean)")
        for a in ARMS:
            v = [R[s][a][m[role]] for s in seeds]
            print(f"  {a:24s} {v} mean {st.mean(v):.2f}")
    mean = lambda a, role: st.mean(R[s][a][m[role]] for s in seeds)
    p1 = all(R[s]["c_hippo_off"][m["recall"]] > max(R[s]["d_hippo_off_consol_off"][m["recall"]], R[s]["e_shuffled_hippo_off"][m["recall"]]) for s in seeds) \
        and mean("c_hippo_off", "recall") >= p1_mean
    p2 = sum(R[s]["a_intact_sleep"][m["recall"]] >= R[s]["b_intact_nosleep"][m["recall"]] for s in seeds) >= 5 \
        and mean("a_intact_sleep", "recall") - mean("b_intact_nosleep", "recall") >= 1
    p3 = mean("a_intact_sleep", "false_recall") <= mean("b_intact_nosleep", "false_recall") + 0.5 \
        and mean("c_hippo_off", "false_recall") <= mean("e_shuffled_hippo_off", "false_recall")
    p4 = mean("c_hippo_off", "paraphrase") > mean("e_shuffled_hippo_off", "paraphrase") + 1
    print(f"\nP1 {p1}  P2 {p2}  P3 {p3}  P4 {p4}")
    print("sleep adds over no-sleep (a-b): recall", round(mean("a_intact_sleep", "recall") - mean("b_intact_nosleep", "recall"), 2),
          "paraphrase", round(mean("a_intact_sleep", "paraphrase") - mean("b_intact_nosleep", "paraphrase"), 2),
          "false", round(mean("a_intact_sleep", "false_recall") - mean("b_intact_nosleep", "false_recall"), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="seed*.json")
    ap.add_argument("--scorer", choices=["strict", "correct"], default="strict")
    ap.add_argument("--p1-mean", type=float, default=2.0)
    a = ap.parse_args()
    main(a.dir, a.glob, a.scorer, a.p1_mean, header=f"scorer {a.scorer}")