"""Backfill `.prov.json` sidecars for the AMENDMENT 3 dev-check artifacts (one-off, declared).

WHY. `_lexicon_closed_class_junction_dev.py` takes `--out <DIRECTORY>`. The automatic provenance door
(`research/runners/__init__.py`) treats any `--out` flag as a declared output, and a directory is not a file, so
it stamped nothing (and the fresh-file fallback is off once a flag is present). The runs' own start records ARE in
`research/findings/raw/_provenance/runs.jsonl`; this script copies each run's record into a sidecar for every file
the run wrote into its `--out` directory, marked `"backfilled": true` with the reason. The dev script now calls
`declare_output()` on each file it writes, so later runs are stamped by the door itself.

    SIM_NO_PROVENANCE=1 .venv/bin/python research/runners/_lexicon_closed_class_a3_backfill_prov.py \
        research/findings/raw/_lexicon_closed_class/dev_s7_amendment3 [...more dirs]
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNNER = "research/runners/_lexicon_closed_class_junction_dev.py"


def _corpus_check_at(start_iso: str, max_age_h: float = 24.0):
    """The corpus-check state AT THE RUN'S START (the provenance door computes the same fields at exit): the last
    `before_you_build.sh` record in research/queue/.corpus_checks.jsonl written before the run started."""
    import time
    t_start = time.mktime(time.strptime(start_iso, "%Y-%m-%dT%H:%M:%S"))
    log = os.path.join(_ROOT, "research", "queue", ".corpus_checks.jsonl")
    last = None
    for ln in open(log, errors="ignore"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if float(r.get("when", 0)) <= t_start:
            last = r
    if last is None:
        return {"corpus_check_age_s": None, "corpus_check_query": None, "corpus_check_fresh": False}
    age = t_start - float(last["when"])
    return {"corpus_check_age_s": round(age, 1), "corpus_check_query": str(last.get("query", ""))[:200],
            "corpus_check_fresh": bool(age <= max_age_h * 3600.0), "corpus_check_age_measured_at": "run start"}


def main(dirs):
    log = os.path.join(_ROOT, "research", "findings", "raw", "_provenance", "runs.jsonl")
    recs = [json.loads(ln) for ln in open(log) if ln.strip()]
    for d in dirs:
        rel = os.path.relpath(os.path.join(_ROOT, d) if not os.path.isabs(d) else d, _ROOT)
        mine = [r for r in recs if "--out" in r.get("argv", [])
                and r["argv"][r["argv"].index("--out") + 1].rstrip("/") == rel.rstrip("/")]
        if len(mine) != 1:
            raise SystemExit(f"{rel}: expected exactly one run record with --out {rel}, found {len(mine)}")
        rec = mine[0]
        for fn in sorted(os.listdir(os.path.join(_ROOT, rel))):
            if not fn.endswith(".json") or fn.endswith(".prov.json"):
                continue
            p = os.path.join(_ROOT, rel, fn)
            side = {"run_id": rec["run_id"], "runner": RUNNER, "argv": rec["argv"], "git_sha": rec["git_sha"],
                    "git_dirty": rec["git_dirty"], "started": rec["started"], "env": rec.get("env"),
                    "host": rec.get("host"), **_corpus_check_at(rec["started"]),
                    "artifact": os.path.relpath(p, _ROOT), "backfilled": True,
                    "backfill_reason": "--out names a directory, so research/runners/__init__.py stamped no "
                                       "sidecar; copied from this run's own runs.jsonl start record",
                    "git_dirty_note": "this lane's record: at launch the only untracked or modified paths were "
                                      "scratch files under .a3_scratch/ (never imported by the runner); tracked "
                                      "source equalled git_sha"}
            with open(p + ".prov.json", "w") as fh:
                json.dump(side, fh, indent=1)
            print("stamped", os.path.relpath(p, _ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
