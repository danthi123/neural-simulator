"""Is ONE caller's brain_chat byte-identical with the M0 turn lock ON and OFF? (one-brain plan M0; docs/TERMS.md:
"byte-identical" is asserted in the data, never inferred from reading the code.)

Three fresh processes run the same single-session turn script through the REAL `webapp.server.brain_chat` (stub
renderer, numpy, `rich` omitted = the production default path, production LTM default):
  on        BRAIN_TURN_LOCK=1
  off       BRAIN_TURN_LOCK=0
  on_rep    BRAIN_TURN_LOCK=1 again: the NULL (two identical configurations across processes). If on vs on_rep
            already differ, a cross-process byte compare cannot be read and the differing keys are reported instead.
Each turn's raw response body is hashed (sha256). Verdict: byte_identical = (on == off) for every turn, readable only
when (on == on_rep) for every turn.

Run (dev seed 7; brain lock + memcap):  ... -m research.runners.turn_lock_identity_check --seed 7 \
    --out research/findings/raw/_one_brain_probe/turn_lock_identity_s7.json
Self-test (no brain):  .venv/bin/python -m research.runners.turn_lock_identity_check --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from research.runners.one_brain_probe import BRAIN, RENDERER, REPO, SESSION_A, seed_allowed

SCRIPT = ["the wug eats the dax", "what does the wug eat", "what does the dog chase",
          "Tell me about angora_turkey.", "what does the wug eat"]
ARMS = {"on": "1", "off": "0", "on_rep": "1"}


def worker(out_path):
    from webapp import server as S
    rows = []
    for msg in SCRIPT:
        r = S.brain_chat(S.BrainChatRequest(session=SESSION_A, message=msg, brain=BRAIN, renderer=RENDERER))
        body = bytes(r.body)
        rows.append({"message": msg, "sha256": hashlib.sha256(body).hexdigest(), "body": json.loads(body)})
    Path(out_path).write_text(json.dumps({"rows": rows, "turn_lock_env": os.environ.get("BRAIN_TURN_LOCK"),
                                          "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                                               / 1024.0, 1)}, default=str))
    return 0


def diff_keys(a, b, path=""):
    """Dotted paths where two JSON values differ (for the report when hashes differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_keys(a.get(k, "<absent>"), b.get(k, "<absent>"), "%s.%s" % (path, k) if path else str(k))
        return out
    return [] if a == b else [path or "<root>"]


def verdict(res):
    n = len(SCRIPT)
    same = lambda x, y: [res[x]["rows"][i]["sha256"] == res[y]["rows"][i]["sha256"] for i in range(n)]  # noqa: E731
    null_ok, on_off = same("on", "on_rep"), same("on", "off")
    out = {"null_identical_per_turn": null_ok, "on_off_identical_per_turn": on_off,
           "readable": all(null_ok), "byte_identical": all(on_off) if all(null_ok) else None}
    out["on_off_diff_keys"] = {i: diff_keys(res["on"]["rows"][i]["body"], res["off"]["rows"][i]["body"])
                               for i in range(n) if not on_off[i]}
    out["null_diff_keys"] = {i: diff_keys(res["on"]["rows"][i]["body"], res["on_rep"]["rows"][i]["body"])
                             for i in range(n) if not null_ok[i]}
    return out


def selftest():
    row = lambda h, b: {"sha256": h, "body": b}  # noqa: E731
    mk = lambda hs: {"rows": [row(h, {"answer": h}) for h in hs]}  # noqa: E731
    base = ["a", "b", "c", "d", "e"]
    v = verdict({"on": mk(base), "off": mk(base), "on_rep": mk(base)})
    assert v["byte_identical"] is True and v["readable"]
    v = verdict({"on": mk(base), "off": mk(["a", "x", "c", "d", "e"]), "on_rep": mk(base)})
    assert v["byte_identical"] is False and v["on_off_diff_keys"] == {1: ["answer"]}
    v = verdict({"on": mk(base), "off": mk(base), "on_rep": mk(["a", "b", "y", "d", "e"])})
    assert v["byte_identical"] is None and not v["readable"]     # a failing null makes the compare unreadable
    print("turn_lock_identity_check selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.worker:
        return worker(a.out)
    ok, why = seed_allowed(a.seed, a.prereg)
    if not ok:
        print("REFUSED: " + why, file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res, meta = {}, {}
    for arm, val in ARMS.items():
        po = out.parent / ("_tli_%s_s%d.json" % (arm, a.seed))
        env = dict(os.environ, SIM_NO_PROVENANCE="1", SIM_BACKEND=os.environ.get("SIM_BACKEND", "numpy"),
                   BRAIN_CHAT_RENDERER=RENDERER, SIM_DISABLE_LLM="1", BRAIN_CHAT_SEED=str(a.seed),
                   BRAIN_PERSIST_LEARNING="0", BRAIN_TURN_LOCK=val)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", "-m", "research.runners.turn_lock_identity_check", "--worker",
                              "--out", str(po)], cwd=str(REPO), env=env)
        if rc != 0:
            print("worker %s failed rc=%d" % (arm, rc), file=sys.stderr)
            return 1
        res[arm] = json.loads(po.read_text())
        meta[arm] = {"seconds": round(time.time() - t0, 1), "peak_rss_mb": res[arm]["peak_rss_mb"],
                     "turn_lock_env": res[arm]["turn_lock_env"]}
        po.unlink()
    v = verdict(res)
    rec = {"runner": "turn_lock_identity_check", "seed": a.seed, "script": SCRIPT, "arms": meta, "verdict": v,
           "hashes": {arm: [r["sha256"] for r in res[arm]["rows"]] for arm in ARMS},
           "answers": {arm: [r["body"].get("answer") for r in res[arm]["rows"]] for arm in ARMS}}
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v[k] for k in ("readable", "byte_identical", "on_off_diff_keys", "null_diff_keys")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
