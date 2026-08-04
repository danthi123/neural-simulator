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
  2. At exit: write an `<artifact>.prov.json` sidecar for the output path declared by `--out`, `--output`, or
     `--json`. Runners without one of those arguments use a guarded fresh-file fallback.

SAFETY, because this executes before EVERY run and must never be why one dies:
  * everything wrapped -- a provenance failure warns and is never fatal;
  * stdlib only, no heavy imports;
  * append-only writes to a dedicated directory;
  * SIM_NO_PROVENANCE=1 disables it entirely (byte-identical reruns, CI).
"""
from __future__ import annotations

import atexit
import hashlib
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
_OUTPUT_FLAGS = frozenset(("--out", "--output", "--json"))


def _source_snapshot():
    """Read identity for a clean exported tree that intentionally has no .git."""
    path = os.path.join(_ROOT, ".source_revision")
    try:
        values = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, sep, value = line.strip().partition("=")
                if sep and key:
                    values[key] = value
        return values
    except Exception:
        return {}


def verify_immutable_source_manifest(snapshot=None):
    """Verify an exported source tree against its complete provisioned manifest."""
    snapshot = dict(_source_snapshot() if snapshot is None else snapshot)
    result = {
        "source_manifest_verified": False,
        "source_manifest_verification_error": None,
    }
    if snapshot.get("source_kind") != "git_archive":
        result["source_manifest_verification_error"] = "source is not a Git archive"
        return result
    expected_manifest_hash = snapshot.get("source_manifest_sha256")
    if not expected_manifest_hash:
        result["source_manifest_verification_error"] = "source manifest digest is missing"
        return result
    manifest_path = os.path.join(_ROOT, ".source_manifest.sha256")
    try:
        with open(manifest_path, "rb") as fh:
            manifest_bytes = fh.read()
        actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_hash != expected_manifest_hash:
            raise ValueError("manifest file digest does not match .source_revision")

        expected_files = {}
        for raw_line in manifest_bytes.decode("utf-8").splitlines():
            digest, separator, relative_path = raw_line.partition("  ")
            if not separator or len(digest) != 64 or not relative_path:
                raise ValueError("manifest contains a malformed line")
            int(digest, 16)
            normalized = os.path.normpath(relative_path)
            if normalized != relative_path or os.path.isabs(normalized) or normalized.startswith(".." + os.sep):
                raise ValueError("manifest contains an unsafe path")
            if normalized in expected_files:
                raise ValueError("manifest contains a duplicate path")
            expected_files[normalized] = digest

        actual_files = set()
        for relative_root in ("sim", "research/runners", "experiment", "tools"):
            root = os.path.join(_ROOT, relative_root)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name != "__pycache__"]
                for filename in filenames:
                    if not filename.endswith((".py", ".sh")):
                        continue
                    path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(path, _ROOT)
                    actual_files.add(relative_path)
        research_init = os.path.join(_ROOT, "research", "__init__.py")
        if os.path.isfile(research_init):
            actual_files.add("research/__init__.py")
        specs_root = os.path.join(_ROOT, "research", "specs")
        if os.path.isdir(specs_root):
            for dirpath, _, filenames in os.walk(specs_root):
                for filename in filenames:
                    if filename.endswith(".json"):
                        actual_files.add(os.path.relpath(os.path.join(dirpath, filename), _ROOT))
        ancestry_attestation = os.path.join(_ROOT, ".source_ancestry.json")
        if os.path.isfile(ancestry_attestation):
            actual_files.add(".source_ancestry.json")
        if actual_files != set(expected_files):
            missing = sorted(set(expected_files) - actual_files)[:3]
            extra = sorted(actual_files - set(expected_files))[:3]
            raise ValueError(f"source file set differs from manifest; missing={missing}, extra={extra}")

        for relative_path, expected_digest in expected_files.items():
            hasher = hashlib.sha256()
            with open(os.path.join(_ROOT, relative_path), "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_digest:
                raise ValueError(f"source digest mismatch: {relative_path}")
        result["source_manifest_verified"] = True
    except Exception as exc:
        result["source_manifest_verification_error"] = str(exc)
    return result


def _git_head():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                               capture_output=True, text=True, timeout=15).stdout.strip() != ""
        if sha != "unknown":
            return sha, dirty
    except Exception:
        pass
    snapshot = _source_snapshot()
    if snapshot.get("git_sha"):
        return snapshot["git_sha"], False
    return "unknown", None


def _record_start():
    sha, dirty = _git_head()
    snapshot = _source_snapshot()
    rec = {
        "run_id": "%d-%d" % (int(_START), os.getpid()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_START)),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_sha": sha,
        "git_dirty": dirty,
        "source_kind": snapshot.get("source_kind"),
        "source_manifest_sha256": snapshot.get("source_manifest_sha256"),
        "python": sys.executable,
        "pid": os.getpid(),
        # The env vars that have silently changed results here before: SIM_BACKEND made `SIM_BACKEND=numpy` run
        # on the GPU for months, and gap#5's read-density lived ONLY in an env var -- a knob with no other record.
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("SIM_", "GAP5_", "HEBB_", "POOL_", "GAP4_")) or k == "CUDA_VISIBLE_DEVICES"},
    }
    if snapshot.get("source_kind") == "git_archive":
        rec.update(verify_immutable_source_manifest(snapshot))
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


def _corpus_check_state(max_age_h=24.0):
    """How long since `before_you_build.sh` last ran. Stamped into every run record.

    EARNED 2026-07-31, expensively. The corpus check returns the priors for a question in 0.63 s and was
    purely ADVISORY: nothing bound running it to launching anything. A nine-hour, eight-cell crux was
    launched against a question already answered three weeks earlier at six seeds, with its root cause
    named in a second finding. The heartbeat flagged the missing check about fifteen times that day and was
    read past every time -- so this is recorded as a FACT of the run rather than as a reminder, and
    `gates/corpus_check_required` refuses an expensive artifact whose run carries no recent check."""
    try:
        log = os.path.join(_ROOT, "research", "queue", ".corpus_checks.jsonl")
        if not os.path.exists(log):
            return {"corpus_check_age_s": None, "corpus_check_query": None}
        last = None
        with open(log, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except ValueError:
                        continue
        if not last:
            return {"corpus_check_age_s": None, "corpus_check_query": None}
        age = max(0.0, time.time() - float(last.get("when", 0)))
        return {"corpus_check_age_s": round(age, 1),
                "corpus_check_query": str(last.get("query", ""))[:200],
                "corpus_check_fresh": bool(age <= max_age_h * 3600.0)}
    except Exception:
        return {"corpus_check_age_s": None, "corpus_check_query": None}


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


def _declared_output_paths(rec):
    """Return (output_flag_seen, existing artifacts under raw named by argv).

    A fresh-file scan cannot establish ownership when several runners overlap:
    every process can see every peer's new artifact. Explicit output arguments
    are the stronger ownership record and are used whenever present.
    """
    argv = list(rec.get("argv") or ())
    cwd = rec.get("cwd") or os.getcwd()
    raw = os.path.realpath(_RAW_DIR)
    seen = False
    values = []
    for i, arg in enumerate(argv):
        value = None
        if arg in _OUTPUT_FLAGS:
            seen = True
            if i + 1 < len(argv):
                value = argv[i + 1]
        else:
            for flag in _OUTPUT_FLAGS:
                prefix = flag + "="
                if arg.startswith(prefix):
                    seen = True
                    value = arg[len(prefix):]
                    break
        if not value:
            continue
        candidate = os.path.realpath(os.path.join(cwd, os.path.expanduser(value)))
        try:
            inside_raw = os.path.commonpath((raw, candidate)) == raw
        except ValueError:
            inside_raw = False
        if (inside_raw and os.path.isfile(candidate)
                and not candidate.endswith(".prov.json")):
            values.append(candidate)
    return seen, list(dict.fromkeys(values))


def _fresh_output_paths():
    paths = []
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
                sidecar = p + ".prov.json"
                if (os.path.exists(sidecar)
                        and os.path.getmtime(sidecar) >= os.path.getmtime(p)):
                    continue
            except OSError:
                continue
            paths.append(p)
    return paths


def _stamp_outputs(rec):
    """Sidecar artifacts owned by this run without claiming concurrent outputs."""
    _resolve_argv(rec)
    declared, explicit_paths = _declared_output_paths(rec)
    candidates = explicit_paths if declared else _fresh_output_paths()
    made = []
    exit_verification = (
        verify_immutable_source_manifest(
            {
                "source_kind": rec.get("source_kind"),
                "source_manifest_sha256": rec.get("source_manifest_sha256"),
            }
        )
        if rec.get("source_kind") == "git_archive"
        else {
            "source_manifest_verified": None,
            "source_manifest_verification_error": None,
        }
    )
    for p in candidates:
        try:
            with open(p + ".prov.json", "w") as fh:
                json.dump({"run_id": rec["run_id"], "runner": rec.get("runner", "unknown"),
                           "argv": rec["argv"], "git_sha": rec["git_sha"], "git_dirty": rec["git_dirty"],
                           "source_kind": rec.get("source_kind"),
                           "source_manifest_sha256": rec.get("source_manifest_sha256"),
                           "source_manifest_verified_at_start": rec.get("source_manifest_verified"),
                           "source_manifest_start_error": rec.get("source_manifest_verification_error"),
                           "source_manifest_verified_at_exit": exit_verification["source_manifest_verified"],
                           "source_manifest_exit_error": exit_verification["source_manifest_verification_error"],
                           "started": rec["started"], "env": rec["env"],
                           **_resolved_backend(), **_corpus_check_state(),
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
