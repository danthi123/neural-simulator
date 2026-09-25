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
_START_UTC_NS = time.time_ns()
_OUTPUT_FLAGS = frozenset(("--out", "--output", "--json"))
_PRIVATE_PROVENANCE_PREFIX = "SIM_PROVENANCE_"
# Extra output paths registered via declare_output() below, sidecared at exit exactly like an argv-declared
# --out/--output/--json path (see declare_output's docstring for why this exists).
_EXTRA_DECLARED_OUTPUTS = []


def declare_output(path):
    """Register PATH as an output this run owns, sidecared at exit alongside anything named by --out/--output/--json
    on argv.

    EARNED (2026-09-25, closing a provenance gap on the B2a/B2b load-bearing shards): a runner can write a real
    artifact from the PARENT process to a path its own `--out` argument never names. `load_bearing_fraction.py`
    writes `oed_distributional<seed>.json` into the same `out_dir` as its declared `--out lb.json`, from inside
    `measure_open_ended_distributional()`, well before `main()` writes `lb.json` itself. `_declared_output_paths`
    only ever sidecars paths named by an output FLAG on argv -- and once ANY output flag is present, the
    fresh-file fallback that would otherwise have caught this file is switched OFF for the whole run, so the side
    artifact was a silent, permanent orphan (no `.prov.json`, ever, from any run of that runner).

    Call this once, right after writing the file, to make it a first-class declared output. It only registers the
    path -- the same existence/location checks `_declared_output_paths` already applies to argv-declared paths
    (must resolve under `research/findings/raw`, must exist as a file, must not itself be a `.prov.json`) are
    applied again at sidecar time, so calling this before the file exists, or on a path outside `raw/`, is
    harmless. Never fatal, by the same contract as the rest of this module -- a bad PATH here must never be why
    the run it is instrumenting fails.

    ONE PER-VALUE GUARANTEE (fix round, review 2026-09-25): a registration only counts as "declared" for the
    PURPOSE OF DISABLING THE FRESH-FILE FALLBACK once IT ITSELF resolves to a real file under `raw/` at sidecar
    time. A registration that never validates (a typo'd path, a call before the file was ever actually written,
    a path outside `raw/`) is simply skipped -- it does NOT, by itself, turn off the fresh-file fallback for the
    rest of a run's genuine, undeclared outputs. Before this guarantee, ANY call here -- valid or not -- set the
    module-level "an output was declared" flag unconditionally, so one bad `declare_output()` call anywhere in a
    run with no `--out`/`--output`/`--json` on argv silently made every OTHER real artifact of that run permanently
    un-sidecared, with no error and no test (`_declared_output_paths` never distinguished "declared" from
    "declared and real"). See `test_declare_output_bad_path_does_not_disable_fresh_file_fallback`.
    """
    try:
        _EXTRA_DECLARED_OUTPUTS.append(str(path))
    except Exception:
        pass


def _provenance_v2_enabled():
    return os.environ.get("SIM_PROVENANCE_V2") == "1"


def _required_v2_identity():
    values = {
        "run_id": os.environ.get("SIM_PROVENANCE_RUN_ID", "").strip(),
        "source_kind": os.environ.get("SIM_PROVENANCE_SOURCE_KIND", "").strip(),
        "source_manifest_sha256": os.environ.get(
            "SIM_PROVENANCE_SOURCE_MANIFEST_SHA256", ""
        ).strip(),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(
            "SIM_PROVENANCE_V2=1 requires private provenance identity: "
            + ", ".join(missing)
        )
    if values["source_kind"] not in ("git", "git_archive"):
        raise ValueError("SIM_PROVENANCE_SOURCE_KIND must be git or git_archive")
    digest = values["source_manifest_sha256"]
    try:
        valid_digest = len(digest) == 64 and digest == digest.lower() and int(digest, 16) >= 0
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError("SIM_PROVENANCE_SOURCE_MANIFEST_SHA256 must be lowercase SHA-256")
    return values


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
        from tools.pool.provisioning.source_manifest import verify_manifest
        verify_manifest(_ROOT, manifest_path, expected_manifest_hash)
        result["source_manifest_verified"] = True
    except Exception as exc:
        result["source_manifest_verification_error"] = str(exc)
    return result


def _git_head(full=False):
    try:
        command = ["git", "rev-parse", "HEAD"] if full else ["git", "rev-parse", "--short", "HEAD"]
        sha = subprocess.run(command, cwd=_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        # AMENDMENT B / D6-capacity-curve review (2026-09-23): runs.jsonl is the provenance door's OWN append-only
        # log (written by _record_start below, every run) -- it is self-referentially "modified" by the very act
        # of recording provenance, so counting it makes git_dirty=true UNCONDITIONALLY for every provenanced run
        # in a checkout that has ever produced one (all 13 D6-capacity-curve sidecars read git_dirty=true for
        # exactly this reason, never a real source change). Excluded by exact pathspec -- nothing else is silenced.
        _prov_log_rel = os.path.relpath(os.path.join(_PROV_DIR, "runs.jsonl"), _ROOT)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)%s" % _prov_log_rel],
            cwd=_ROOT, capture_output=True, text=True, timeout=15).stdout.strip() != ""
        if sha != "unknown":
            return sha, dirty
    except Exception:
        pass
    snapshot = _source_snapshot()
    if snapshot.get("git_sha"):
        return snapshot["git_sha"], False
    return "unknown", None


def _hostname():
    """The machine this run executed on (2026-09-24, research/lbf-row-registry-hook): pool jobs, AWS pool1/pool2 and
    local runs currently produce byte-identical artifacts with no record of WHICH node ran them, so a mixed-host
    battery (e.g. the flip-defaults s102 shards, some local, some pool) cannot honestly say which shards differ by
    host. Never fatal: an unresolvable hostname records 'unknown', never raises into the run it is instrumenting."""
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


def _record_start():
    v2 = _provenance_v2_enabled()
    identity = _required_v2_identity() if v2 else None
    sha, dirty = _git_head(full=v2)
    snapshot = _source_snapshot()
    if v2:
        if sha == "unknown" or len(sha) != 40:
            raise ValueError("provenance v2 requires a full Git revision")
        try:
            int(sha, 16)
        except ValueError as exc:
            raise ValueError("provenance v2 requires a hexadecimal Git revision") from exc
        if identity["source_kind"] == "git_archive":
            expected = {
                "git_sha": sha,
                "source_kind": identity["source_kind"],
                "source_manifest_sha256": identity["source_manifest_sha256"],
            }
            actual = {key: snapshot.get(key) for key in expected}
            if actual != expected:
                raise ValueError("private provenance identity does not match archive source identity")
    rec = {
        "run_id": identity["run_id"] if v2 else "%d-%d" % (int(_START), os.getpid()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_START)),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_sha": sha,
        "git_dirty": dirty,
        "source_kind": identity["source_kind"] if v2 else snapshot.get("source_kind"),
        "source_manifest_sha256": (
            identity["source_manifest_sha256"] if v2 else snapshot.get("source_manifest_sha256")
        ),
        "python": sys.executable,
        "pid": os.getpid(),
        "host": _hostname(),
        # The env vars that have silently changed results here before: SIM_BACKEND made `SIM_BACKEND=numpy` run
        # on the GPU for months, and gap#5's read-density lived ONLY in an env var -- a knob with no other record.
        "env": {k: v for k, v in os.environ.items()
                if not k.startswith(_PRIVATE_PROVENANCE_PREFIX)
                # LB_ / BRAIN_MULTIREF_: the load-bearing battery's opt-in probe flags and the D6 lesion knobs
                # (adversarial review v2:7a3b94367: a flag-ON wm-binding artifact recorded only SIM_BACKEND).
                # BRAIN_: every production flag (a flip battery's arms differ ONLY in these). *_NUM_THREADS: the math-
                # library thread count changed a ridge decode on one seed (2026-09-24, perception G0: 12 threads read
                # 0.6458, 1 or 4 threads 0.625), and an unset count means "every core", so cpu_count is recorded too.
                and (k.startswith(("SIM_", "GAP5_", "HEBB_", "POOL_", "GAP4_", "LB_", "BRAIN_"))
                     or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                              "NUMEXPR_NUM_THREADS"))},
        "cpu_count": os.cpu_count(),
    }
    if v2:
        rec["provenance_schema"] = "sim-run-provenance-v2"
        rec["started_utc_ns"] = _START_UTC_NS
    if rec.get("source_kind") == "git_archive":
        rec.update(verify_immutable_source_manifest(rec))
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
    """Return (declared, existing artifacts under raw named by argv or declare_output()).

    A fresh-file scan cannot establish ownership when several runners overlap:
    every process can see every peer's new artifact. Explicit output arguments
    are the stronger ownership record and are used whenever present.

    `declared` (the first element) is True once at least one output path -- from an argv --out/--output/--json
    flag OR a declare_output() registration -- VALIDATES (resolves under raw/, exists as a file, is not itself a
    .prov.json). An argv flag alone sets it as soon as the flag is present (the declared path may not exist yet
    at import time; the run is still trusted to write it by exit). A declare_output() registration is held to a
    stricter bar: since the contract for calling it is "right after writing the file", a registration that never
    validates is treated as if it had never been made, and does NOT flip `declared` -- otherwise one bad
    declare_output() call (a typo, or a call before the file existed) would silently turn off the fresh-file
    fallback for a run's entire remaining, undeclared output with no error and no replacement (see
    declare_output's docstring).
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
    # declare_output() registrations: same validation as an argv-declared path. A registration marks this run as
    # "declared" (so a runner with NO --out/--output/--json flag but at least one VALID declare_output() call
    # still gets the explicit-path treatment instead of falling through to fresh-file scanning) ONLY once it
    # actually validates -- a registration that fails validation (never became a real file under raw/, or points
    # outside it) is silently skipped and must NOT flip `seen` on its own, or it would disable the fresh-file
    # fallback for the rest of this run's genuine outputs with nothing to replace it (see declare_output's
    # docstring: "ONE PER-VALUE GUARANTEE").
    for value in list(_EXTRA_DECLARED_OUTPUTS):
        if not value:
            continue
        candidate = os.path.realpath(os.path.join(cwd, os.path.expanduser(value)))
        try:
            inside_raw = os.path.commonpath((raw, candidate)) == raw
        except ValueError:
            inside_raw = False
        if (inside_raw and os.path.isfile(candidate)
                and not candidate.endswith(".prov.json")):
            seen = True
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
    ended_utc_ns = time.time_ns()
    for p in candidates:
        try:
            sidecar = {"run_id": rec["run_id"], "runner": rec.get("runner", "unknown"),
                       "argv": rec["argv"], "git_sha": rec["git_sha"], "git_dirty": rec["git_dirty"],
                       "source_kind": rec.get("source_kind"),
                       "source_manifest_sha256": rec.get("source_manifest_sha256"),
                       "source_manifest_verified_at_start": rec.get("source_manifest_verified"),
                       "source_manifest_start_error": rec.get("source_manifest_verification_error"),
                       "source_manifest_verified_at_exit": exit_verification["source_manifest_verified"],
                       "source_manifest_exit_error": exit_verification["source_manifest_verification_error"],
                       "started": rec["started"], "env": rec["env"], "host": rec.get("host", "unknown"),
                       **_resolved_backend(), **_corpus_check_state(),
                       "artifact": os.path.relpath(p, _ROOT)}
            if rec.get("provenance_schema") == "sim-run-provenance-v2":
                sidecar.update({
                    "schema": "sim-run-provenance-v2",
                    "started_utc_ns": rec["started_utc_ns"],
                    "ended_utc_ns": ended_utc_ns,
                })
            with open(p + ".prov.json", "w") as fh:
                json.dump(sidecar, fh, indent=1)
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
