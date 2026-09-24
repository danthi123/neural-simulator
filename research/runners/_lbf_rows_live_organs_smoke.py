"""Seed-7 SMOKE for `research/runners/lbf_rows/live_organs.py`'s seven `neural-lesion` rows (S12, lane A1).

Builds intact + lesion arms through the REAL `webapp.server.brain_chat` handler (numpy backend, stub renderer, no
LLM), reusing `onebrain_regression_battery`'s `PROBE_TURNS`/`_EXTRA_TURNS`/`_TURN_BY_LABEL`/`_get_path` verbatim
(imported directly, never copied), exactly as `load_bearing_fraction.py` itself does for turn resolution. It does
NOT reuse `_spawn_arm` -- that helper spawns `-m research.runners.onebrain_regression_battery --worker`, whose
worker process would only see the BASE `_TURN_BY_LABEL` (this module's `EXTRA_TURNS` are added to a LOCAL dict
here, in this process, never patched into the battery module) and so could not resolve this module's own new turn
labels (`lbf_tom1`, `lbf_cau_*`). Instead this script re-execs ITSELF (`_spawn` below, `sys.executable -u
__file__ --worker ...`) so the worker's `_TURN_BY_LABEL` lookup runs inside this module, where
`ROWS.EXTRA_TURNS` has already been merged in -- the same reasoning the sibling A2/`learning.py` smoke uses
(documented there and here, never touching either literal registry on disk). Because AG-REG's import hook
(S08/S26) has not yet merged `lbf_rows/*.py` into `FACULTY_LESIONS`/`FACULTY_PROBES` at the time this runs, this
script does not need that merge either: it drives each row's (flag, value, turn) directly, reading
`lbf_rows.live_organs.EXTRA_LESIONS`/`EXTRA_PROBES` itself.

Each arm is a FRESH subprocess (RNG-isolated, matches every other flip-verify in this repo); one arm at a time,
so peak RSS is one brain build. Run under the repo's standing RAM discipline:

  bash tools/mem_ok.sh 6 4 && OMP_NUM_THREADS=1 bash tools/memcap.sh 6 -- \\
      .venv/bin/python -u -m research.runners._lbf_rows_live_organs_smoke --seed 7 \\
      --out-dir research/findings/raw/_lbf_rows_live_organs

Only rows: `--only self-schema` (repeatable) restricts to a subset -- useful when RAM only allows one row's two
arms at a time (this lane's own experience: a full-brain build stalled for minutes in D-state under system-wide
swap pressure from ~13 concurrent sibling lanes on one box; see live_organs.py's module docstring).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
os.environ.setdefault("SIM_DISABLE_LLM", "1")

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from research.runners.onebrain_regression_battery import (  # noqa: E402
    PROBE_TURNS, _EXTRA_TURNS, _TURN_BY_LABEL as _BASE_TURN_BY_LABEL, _get_path,
)
from research.runners.lbf_rows import live_organs as ROWS  # noqa: E402
# Attribution discipline (tools.lab): `run_row` below builds an intact arm and a lesion (control-shaped) arm and
# reports both -- `lever()` forces the "did the compared field actually MOVE" question to execute per field,
# instead of two JSON blobs sitting one key apart in `summary_s<seed>.json` with nobody subtracting them.
from tools.lab import lever  # noqa: E402

# local turn-label table = the battery's own + this module's EXTRA_TURNS (additive; never mutates the imported dict).
_TURN_BY_LABEL = dict(_BASE_TURN_BY_LABEL)
_TURN_BY_LABEL.update({t[0]: t for t in ROWS.EXTRA_TURNS})


def _turn_group(label: str) -> list[str]:
    """Same semantics as load_bearing_fraction.turn_group(): every same-session turn up to and INCLUDING `label`,
    in declaration order, over PROBE_TURNS + _EXTRA_TURNS + this module's own EXTRA_TURNS."""
    target = _TURN_BY_LABEL[label]
    sess = target[2]
    grp = []
    for t in list(PROBE_TURNS) + list(_EXTRA_TURNS) + list(ROWS.EXTRA_TURNS):
        if t[2] == sess:
            grp.append(t[0])
        if t[0] == label:
            break
    return grp


def _worker_labels_env(env_json: str, labels_csv: str, out_path: str) -> int:
    """--worker entrypoint (spawned by _spawn_arm as `-m research.runners.onebrain_regression_battery --worker`
    normally; this smoke instead spawns ITSELF so `_TURN_BY_LABEL` includes this module's EXTRA_TURNS)."""
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v
    from webapp.server import brain_chat, BrainChatRequest
    responses = {}
    for label in labels_csv.split(","):
        _, msg, session, reset, percept, rich = _TURN_BY_LABEL[label]
        try:
            kwargs = dict(session=session, message=msg, brain="tiny-demo", renderer="stub",
                          rich=bool(rich), reset=reset)
            if percept is not None:
                kwargs["percept"] = percept
            r = brain_chat(BrainChatRequest(**kwargs))
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(responses, open(out_path, "w"), indent=2, default=str)
    print("[worker] env=%s -> %d turns -> %s" % (env, len(responses), out_path), flush=True)
    return 0


def _spawn(env: dict, labels: list[str], out_path: str):
    import subprocess
    p = subprocess.run([sys.executable, "-u", __file__, "--worker",
                        "--env", json.dumps(env), "--turns", ",".join(labels), "--out", out_path],
                       env=dict(os.environ))
    if p.returncode != 0 or not os.path.exists(out_path):
        print("[spawn] FAILED env=%s rc=%s" % (env, p.returncode))
        return None
    return json.load(open(out_path))


def run_row(key: str, seed: int, out_dir: str) -> dict:
    spec = ROWS.EXTRA_LESIONS[key]
    if spec["kind"] != "neural-lesion":
        return {"key": key, "skipped": "kind=%s (no arm to build)" % spec["kind"]}
    row = next(p for p in ROWS.EXTRA_PROBES if p[0] == key)
    turn_label = row[1]
    labels = _turn_group(turn_label)
    base_env = {"BRAIN_CHAT_SEED": str(seed)}
    intact_path = os.path.join(out_dir, "%s_intact_s%d.json" % (key.replace("-", "_"), seed))
    lesion_path = os.path.join(out_dir, "%s_lesion_s%d.json" % (key.replace("-", "_"), seed))
    intact_full = _spawn(dict(base_env), labels, intact_path)
    les_env = dict(base_env); les_env[spec["flag"]] = spec["value"]
    lesion_full = _spawn(les_env, labels, lesion_path)
    result = {"key": key, "flag": spec["flag"], "turn": turn_label, "labels": labels,
              "intact_ok": intact_full is not None, "lesion_ok": lesion_full is not None}
    if intact_full is not None and lesion_full is not None:
        intact_t, lesion_t = intact_full.get(turn_label, {}), lesion_full.get(turn_label, {})
        result["fields"] = row[2]
        result["diffs"] = []
        for path in row[2]:
            _, on_v = _get_path(intact_t, path)
            _, off_v = _get_path(lesion_t, path)
            moved = lever("%s %s" % (key, path), on_v, off_v, required=False)
            result["diffs"].append({"field": path, "intact": on_v, "lesion": off_v, "changed": moved})
        result["assertion"] = ROWS.assert_lesion_holds(key, intact_t, lesion_t)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env")
    ap.add_argument("--turns")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="research/findings/raw/_lbf_rows_live_organs")
    ap.add_argument("--only", action="append", default=None)
    args = ap.parse_args()
    if args.worker:
        sys.exit(_worker_labels_env(args.env, args.turns, args.out))

    if args.seed not in (7,):
        print("WARNING: seed %d is not the dev/calibration seed 7 -- this smoke is for calibration only; the "
              "6-seed gate (42/43/44/100/101/102) is B2b's job, not this script's." % args.seed, file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)
    keys = args.only or [k for k, v in ROWS.EXTRA_LESIONS.items() if v["kind"] == "neural-lesion"]
    summary = {}
    for key in keys:
        print("=== row: %s ===" % key, flush=True)
        summary[key] = run_row(key, args.seed, args.out_dir)
        json.dump(summary, open(os.path.join(args.out_dir, "summary_s%d.json" % args.seed), "w"),
                  indent=2, default=str)
        print(json.dumps(summary[key], indent=2, default=str), flush=True)
    print("=== SUMMARY: %s ===" % os.path.join(args.out_dir, "summary_s%d.json" % args.seed))


if __name__ == "__main__":
    main()
