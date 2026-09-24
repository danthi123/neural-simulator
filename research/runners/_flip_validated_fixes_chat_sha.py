"""Scripted multi-turn /api/brain-chat session -> sha256 of every full response body (flip-validated-fixes, 2026-09-23).

PURPOSE. The 2026-09-23 production default-flip of BRAIN_EPISODIC_STORE_VERIFY, BRAIN_PMEM_FACILITATION and
BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE must keep the OFF arm reachable: with every flipped flag set EXPLICITLY to "0", the
chat path must be byte-identical to the pinned pre-flip commit 033e385b8. This runner drives one fixed scripted session
through the REAL in-process FastAPI handler (TestClient, as tests/test_webapp_server.py does) and writes the sha256 of
each response body. Run it once in this tree and once in an export of 033e385b8 (same file copied there), then compare
the per-turn hashes exactly.

  python -m research.runners._flip_validated_fixes_chat_sha --arm off     --out A.json   # the 3 flags forced "0"
  python -m research.runners._flip_validated_fixes_chat_sha --arm default --out B.json   # the 3 flags UNSET

The OFF arm writes os.environ[F] = "0" for each flag (never pop / unset: once a default is ON, unset == ON and the
comparison would prove nothing -- gates/flip_offarm_staleness).

VOLATILE FIELDS. A response can carry wall-clock timings. Each body is hashed twice: `sha_raw` (the exact bytes) and
`sha_canon` (JSON re-serialized with sort_keys after dropping keys whose NAME matches VOLATILE_KEY_RE). The dropped key
paths are listed per turn so a reader can see nothing semantic was removed. The identity verdict uses sha_canon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

FLIPPED = ("BRAIN_EPISODIC_STORE_VERIFY", "BRAIN_PMEM_FACILITATION", "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE")
SESSION = "flipfix-sha"
TURNS = [
    ("what does the dog chase", False),
    ("the wolf hunts the deer", False),
    ("what does the wolf hunt", False),
    ("remind me to feed the cat when we talk about dinner", False),
    ("what does the dog chase", False),
    ("tell me about the bird", False),
    ("what should we have for dinner", False),
    ("earlier you mentioned the wolf", False),
    ("do you remember talking about the dog", False),
    ("what does the dog chase", True),
]
VOLATILE_KEY_RE = re.compile(r"(latency|elapsed|_ms$|_sec$|_secs$|seconds|time|timestamp|wall|duration|perf)", re.I)


def _canon(obj, path, dropped):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = "%s.%s" % (path, k)
            if VOLATILE_KEY_RE.search(str(k)):
                dropped.append(p)
                continue
            out[k] = _canon(v, p, dropped)
        return out
    if isinstance(obj, list):
        return [_canon(v, "%s[%d]" % (path, i), dropped) for i, v in enumerate(obj)]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("off", "default"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    for f in FLIPPED:
        if a.arm == "off":
            os.environ[f] = "0"
        else:
            os.environ.pop(f, None)   # the DEFAULT arm: unset on purpose (it measures the shipped default, not an OFF arm)

    sys.path.insert(0, os.getcwd())
    from fastapi.testclient import TestClient  # noqa: E402
    from webapp.server import app  # noqa: E402
    client = TestClient(app)

    rows = []
    t0 = time.time()
    for i, (msg, rich) in enumerate(TURNS):
        res = client.post("/api/brain-chat", json={"session": SESSION, "brain": "tiny-demo", "renderer": "stub",
                                                   "message": msg, "rich": rich})
        body = res.content
        dropped = []
        try:
            canon = json.dumps(_canon(json.loads(body), "$", dropped), sort_keys=True).encode()
        except Exception:
            canon = body
        try:
            ans = json.loads(body).get("answer")
        except Exception:
            ans = None
        rows.append({"turn": i, "message": msg, "rich": rich, "status": res.status_code,
                     "sha_raw": hashlib.sha256(body).hexdigest(), "sha_canon": hashlib.sha256(canon).hexdigest(),
                     "dropped_volatile_keys": dropped, "answer": ans})
        print("[chat-sha] turn %d status=%d canon=%s ans=%r" % (i, res.status_code, rows[-1]["sha_canon"][:12],
                                                                 (ans or "")[:80]), flush=True)
    client.post("/api/brain-chat/reset", json={"session": SESSION, "brain": "tiny-demo", "renderer": "stub"})
    out = {"arm": a.arm, "label": a.label, "cwd": os.getcwd(), "flags_env": {f: os.environ.get(f) for f in FLIPPED},
           "sim_backend": os.environ.get("SIM_BACKEND"), "n_turns": len(TURNS), "wall_s": round(time.time() - t0, 1),
           "session_sha_canon": hashlib.sha256("".join(r["sha_canon"] for r in rows).encode()).hexdigest(),
           "turns": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("[chat-sha] session_sha_canon=%s wrote %s" % (out["session_sha_canon"], a.out), flush=True)


if __name__ == "__main__":
    main()
