"""Runner modules for research pipelines (G1, G2, ...).

⭐ THIS FILE IS THE UNAVOIDABLE DOOR FOR RUNS, and it is now used as one.

WHY (2026-07-31). The repo's worst-measured structural defect is that artifacts cannot say what produced them:
**7127 result JSONs, only 393 (5.5%) with a sibling .cmd.json and 443 (6.2%) carrying any provenance key inside
=> ~94% unprovenanced.** The cause is structural, not sloppiness: **645 runners accept `--out`**, so the output
path is chosen in `argv` at invocation and nothing durably captures `argv`. The queue's done-log holds 204 lines
against 7127 artifacts (2.9%).

What that cost: ~94 GPU-hours re-deriving a NO-GO banked a week earlier, because nothing tied the banked artifact
to the command that produced it. And `_gap5_fieldquality_gpu6.py` wrote UNCONDITIONALLY to the path holding a
banked 6-seed GPU GO -- a CPU smoke test would have silently clobbered it. **An artifact that cannot say what it
is cannot be protected from being overwritten.**

Every previous attempt to fix this asked runner authors to opt in. Measured result of that approach across this
repo: `tools/lab.py` imported by **2 of 1330** runners, `tools/experiment.py` by **0**. Opt-in does not work here.
But **990 of 992 documented invocations use `-m research.runners.X`** (99.8%), and `-m` imports THIS package
first, every time. So provenance is captured here, automatically, for runners nobody has to modify.

WHAT IT DOES
  1. On import: stamp a run record (argv, cwd, git SHA, dirty flag, python, relevant env, pid, start) into
     research/findings/raw/_provenance/runs.jsonl, and export SIM_RUN_ID.
  2. At exit: write an `<artifact>.prov.json` sidecar for every file created under research/findings/raw/ during
     this run, naming the run id and the exact argv. Nothing to remember; no runner edited.

SAFETY, because this executes before EVERY run and must never be why one dies:
  * everything wrapped -- a provenance failure warns and is never fatal;
  * stdlib only, no heavy imports;
  * append-only writes to a dedicated directory;
  * SIM_NO_PROVENANCE=1 disables it entirely (byte-identical reruns, CI).
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time

# THREE levels: this file is <root>/research/runners/__init__.py. Two dirnames land on <root>/research and
# silently create <root>/research/research/findings/raw/_provenance -- caught on the first real invocation,
# by the provenance log being absent while SIM_RUN_ID was set.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROV_DIR = os.path.join(_ROOT, "research", "findings", "raw", "_provenance")
_RAW_DIR = os.path.join(_ROOT, "research", "findings", "raw")
_ENABLED = os.environ.get("SIM_NO_PROVENANCE", "") != "1"
_START = time.time()


def _git_head():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                               capture_output=True, text=True, timeout=15).stdout.strip() != ""
        return sha, dirty
    except Exception:
        return "unknown", None


def _record_start():
    sha, dirty = _git_head()
    rec = {
        "run_id": "%d-%d" % (int(_START), os.getpid()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_START)),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.executable,
        "pid": os.getpid(),
        # The env vars that have silently changed results here before: SIM_BACKEND made `SIM_BACKEND=numpy` run
        # on the GPU for months, and gap#5's read-density lived ONLY in an env var -- a knob with no other record.
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("SIM_", "GAP5_", "HEBB_", "POOL_", "GAP4_")) or k == "CUDA_VISIBLE_DEVICES"},
    }
    os.makedirs(_PROV_DIR, exist_ok=True)
    with open(os.path.join(_PROV_DIR, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    os.environ["SIM_RUN_ID"] = rec["run_id"]
    return rec


def _resolve_argv(rec):
    """At IMPORT time `-m` has not yet rewritten sys.argv[0], so the record says "-m" and never names the runner.
    That is the single most important field -- "which runner produced this artifact" is the question provenance
    exists to answer. By atexit, runpy has set argv[0] to the module path, so re-read it then and prefer it.
    Caught on the first end-to-end run: the sidecar carried every flag and no module name."""
    try:
        argv = list(sys.argv)
        if argv and argv[0] not in ("-m", "-c", ""):
            rec["argv"] = argv
            rec["runner"] = os.path.relpath(argv[0], _ROOT) if os.path.isabs(argv[0]) else argv[0]
        else:
            mod = getattr(sys.modules.get("__main__"), "__file__", None)
            if mod:
                rec["runner"] = os.path.relpath(mod, _ROOT)
    except Exception:
        pass
    return rec


def _resolved_backend():
    """The backend ACTUALLY used, resolved at exit rather than read at import.

    EARNED 2026-07-31. This package's __init__ runs BEFORE the runner body, and runners apply
    `os.environ.setdefault("SIM_BACKEND", "numpy")` in that body -- so the `env` block captured at import is
    EMPTY for every caller who did not set it explicitly, and the sidecar cannot say what device ran. That is
    the request recorded and called provenance, one layer below the defect this door exists to close: a
    four-cell "GPU" test spent 30 minutes on the CPU and nothing in the record could have revealed it."""
    try:
        requested = os.environ.get("SIM_BACKEND")
        try:
            import cupy  # noqa: F401
            importable = True
        except Exception:
            importable = False
        return {"sim_backend": "cupy" if (requested == "cupy" and importable) else "numpy",
                "sim_backend_requested": requested or "(unset -> runner default)",
                "sim_backend_cupy_importable": importable}
    except Exception:
        return {}


def _stamp_outputs(rec):
    """Sidecar every artifact this run created. mtime-bounded, so it can never claim another run's files."""
    _resolve_argv(rec)
    made = []
    for dirpath, dirnames, filenames in os.walk(_RAW_DIR):
        if os.path.basename(dirpath) == "_provenance":
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".prov.json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getmtime(p) < _START:
                    continue
            except OSError:
                continue
            try:
                with open(p + ".prov.json", "w") as fh:
                    json.dump({"run_id": rec["run_id"], "runner": rec.get("runner", "unknown"),
                               "argv": rec["argv"], "git_sha": rec["git_sha"], "git_dirty": rec["git_dirty"],
                               "started": rec["started"], "env": rec["env"], **_resolved_backend(),
                               "artifact": os.path.relpath(p, _ROOT)}, fh, indent=1)
                made.append(p)
            except OSError:
                pass
    return made


if _ENABLED:
    try:
        _REC = _record_start()

        @atexit.register
        def _finish():
            try:
                n = len(_stamp_outputs(_REC))
                if n:
                    print("[provenance] stamped %d artifact(s) | run_id %s" % (n, _REC["run_id"]), file=sys.stderr)
            except Exception as e:                      # never fatal, never silent
                print("[provenance] WARNING: output stamping failed: %s: %s" % (type(e).__name__, e),
                      file=sys.stderr)
    except Exception as e:
        print("[provenance] WARNING: run record failed: %s: %s" % (type(e).__name__, e), file=sys.stderr)
