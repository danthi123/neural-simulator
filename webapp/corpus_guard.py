"""Repo-wide guard for the corpus-learned production organs (2026-09-30, branch research/chat-fact-fixes).

THE DEFECT (research/FAILURE_LOG.md 2026-09-23 x3, 2026-09-25, 2026-09-29). The production brain LEARNS parts of
itself from `data/corpus/` at build time (the one-brain cross-edge pool and the learned-animacy lexicon read
`tinystories.txt`). `data/` is git-excluded, so a git worktree or a pool revision directory whose provision did not
sync it builds a DIFFERENT, degraded brain: `onebrain_xedge_production` logs "ONEBRAIN XEDGE build FAILED -> degrading
to standalone organs", the lexicon raises inside a caught branch, and every turn still answers with nothing in the
reply saying so. Five runners grew their own CORPUS GUARD one incident at a time; nothing guarded the server itself.

THE GUARD (one place, every surface that builds a chat brain):
  * `check_at_build()` runs at the top of `webapp.server._build_chat_brain`. Corpus present: silent, no effect.
    Corpus missing: a loud stderr WARNING (once per process and root) -- or, under BRAIN_REQUIRE_CORPUS=1,
    `CorpusMissingError` (the build refuses; the /api/brain-chat handler turns it into a 400 naming the files).
  * `mark_response()` wraps the /api/brain-chat handler's single exit. Corpus present: the SAME response object is
    returned untouched (byte-identical reply). Corpus missing: the reply gains `corpus_present: false` and
    `corpus_missing: [...]`, so every consumer of a degraded turn can see it.
The marker is deliberately absent when the corpus is present: a field on every reply would change the bytes of every
production reply and break the cross-revision byte-identity checks the integration packets rely on. So read it as
`reply.get("corpus_present", True)`, and remember a reply from a revision older than this guard never carries it.

CORPUS_FILES is the set the production tiny-demo brain was MEASURED to open under data/corpus/ (an audit hook over a
full 13-turn /api/brain-chat session, `research/runners/chat_fact_fixes_identity.py --log-corpus-opens`, dev seed 7,
2026-09-30); see the HANDOFF of research/packets/chat-fact-fixes/.
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_FILES = ("tinystories.txt",)
_WARNED: set = set()


class CorpusMissingError(RuntimeError):
    """Raised by `check_at_build` under BRAIN_REQUIRE_CORPUS=1 when a corpus file the production brain learns from is
    missing (the build refuses instead of degrading)."""


def corpus_status(root: str | None = None) -> dict:
    """{"corpus_present": bool, "corpus_missing": [file, ...]} for `root`/data/corpus (default: this checkout)."""
    root = REPO_ROOT if root is None else root
    missing = [f for f in CORPUS_FILES if not os.path.exists(os.path.join(root, "data", "corpus", f))]
    return {"corpus_present": not missing, "corpus_missing": missing}


def require_corpus() -> bool:
    return os.environ.get("BRAIN_REQUIRE_CORPUS", "").strip().lower() in ("1", "true", "yes", "on")


def check_at_build(root: str | None = None) -> dict:
    """Run before a chat brain is built. Present: returns the status, no side effect. Missing: raises
    `CorpusMissingError` under BRAIN_REQUIRE_CORPUS=1, else warns LOUDLY on stderr (once per process and root)."""
    st = corpus_status(root)
    if st["corpus_present"]:
        return st
    where = os.path.join(REPO_ROOT if root is None else root, "data", "corpus")
    msg = ("CORPUS MISSING: %s lacks %s. The corpus-learned organs (the one-brain cross-edge pool, the learned-animacy "
           "lexicon) will DEGRADE to their standalone paths: this is a DIFFERENT brain from the primary checkout's. "
           "Link or sync data/ (a worktree: `ln -s <primary>/data data`; a pool node: tools/pool_provision.sh). "
           "Every /api/brain-chat reply carries corpus_present=false." % (where, st["corpus_missing"]))
    if require_corpus():
        raise CorpusMissingError(msg + " BRAIN_REQUIRE_CORPUS=1: refusing to build.")
    if where not in _WARNED:
        _WARNED.add(where)
        print("[webapp] ⛔ %s Set BRAIN_REQUIRE_CORPUS=1 to refuse instead." % msg, file=sys.stderr, flush=True)
    return st


def mark_response(resp, root: str | None = None):
    """The /api/brain-chat reply, marked when the corpus is missing. Present (or a non-JSON response): the SAME
    object, untouched. Missing: a new JSONResponse with the same status and background, the body's dict plus
    `corpus_present: false` and `corpus_missing`."""
    st = corpus_status(root)
    if st["corpus_present"]:
        return resp
    try:
        from starlette.responses import JSONResponse
        if not isinstance(resp, JSONResponse):
            return resp
        body = json.loads(resp.body)
        if not isinstance(body, dict):
            return resp
        body.update(st)
        out = JSONResponse(body, status_code=resp.status_code, background=resp.background)
        for k, v in resp.headers.items():
            if k.lower() not in ("content-length", "content-type"):
                out.headers[k] = v
        return out
    except Exception as e:   # the marker must never be why a turn fails; say so loudly instead
        print("[webapp] ⛔ corpus_guard.mark_response failed (%s: %s); the reply is UNMARKED although the corpus "
              "is missing %s" % (type(e).__name__, e, st["corpus_missing"]), file=sys.stderr, flush=True)
        return resp
