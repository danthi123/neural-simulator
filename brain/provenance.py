"""Provenance stamp for brain/ result JSONs: which host, which code, which model file produced a result.

Every probe that writes a result JSON calls `stamp(res, model)` on the dict just before `json.dump`; it adds one
top-level key, "provenance", and changes nothing else (no scoring, no row). tools/where_ran.py reads it.

  host       socket.gethostname()
  git_sha    HEAD of the checkout the code runs from, `git_dirty` whether brain/ has uncommitted changes; both None
             where there is no .git (AWS gets `git archive HEAD brain`, the pool a synced copy)
  src_hash   sha256 over brain/**/*.py (relative path + bytes, sorted), recorded everywhere, so an AWS or pool result
             can be matched to a commit by hashing that commit's brain/ (`src_hash(root)` on a checkout of it)
  model_path the --model value resolved by brain.cortex.resolve_model, its realpath, size and mtime
"""
from __future__ import annotations

import functools
import hashlib
import os
import socket
import subprocess
import sys
import time

BRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_SKIP_DIRS = {"__pycache__", "results", "transcripts"}


def src_hash(brain_dir: str = BRAIN_DIR) -> str:
    """sha256 of every .py under brain_dir (relative path and bytes, in sorted order); stable across copies."""
    files = []
    for dp, dn, fns in os.walk(brain_dir):
        dn[:] = sorted(d for d in dn if d not in _SKIP_DIRS and not d.startswith("."))
        files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    h = hashlib.sha256()
    for p in sorted(files, key=lambda p: os.path.relpath(p, brain_dir)):
        h.update(os.path.relpath(p, brain_dir).replace(os.sep, "/").encode() + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def _git(*args, cwd):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@functools.lru_cache(maxsize=None)
def code_version(brain_dir: str = BRAIN_DIR) -> dict:
    """git sha / dirty (None without a .git) and the brain/ source hash; computed once per process."""
    sha = _git("rev-parse", "HEAD", cwd=brain_dir)
    dirty = None
    if sha:
        st = _git("status", "--porcelain", "--", ".", cwd=brain_dir)
        dirty = bool(st) if st is not None else None
    return {"git_sha": sha or None, "git_dirty": dirty, "src_hash": src_hash(brain_dir)}


def model_info(model) -> dict:
    """The resolved checkpoint path for a --model value (path, size tag or None = DEFAULT_MODEL), realpath/size/mtime."""
    out = {"model_arg": model, "model_path": None}
    try:
        from brain.cortex import resolve_model
        path = resolve_model(model)
    except Exception as e:   # a stamp must never fail the run it records
        out["model_error"] = f"{type(e).__name__}: {e}"
        return out
    out["model_path"] = path
    real = os.path.realpath(path)
    if real != path:
        out["model_realpath"] = real
    try:
        s = os.stat(real)
        out.update(model_bytes=s.st_size, model_mtime=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(s.st_mtime)))
    except OSError:
        out["model_missing"] = True
    return out


def provenance(model=None) -> dict:
    return {"host": socket.gethostname(), "stamped": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "argv": list(sys.argv), **code_version(), **model_info(model)}


def stamp(res: dict, model=None) -> dict:
    """Add res["provenance"] (host, code version, resolved model file) and return res, for `json.dump(stamp(res, m))`."""
    res["provenance"] = provenance(model)
    return res
