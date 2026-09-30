"""chat-fact-fixes: before/after reply-identity check through the REAL /api/brain-chat handler (2026-09-30).

Evidence instrument for branch research/chat-fact-fixes (two defects from the Stage-1 exit packet,
research/FAILURE_LOG.md rows 2026-09-29): (1) `ChatBrain._maybe_acquire` stored listener-directed requests such as
"tell me about the sky" as SVO facts; (2) corpus-learned organs degraded SILENTLY without data/corpus/. Each fix must
leave every reply outside its defect byte-identical. This runner plays a fixed script through the real handler in ONE
FRESH PROCESS per arm (numpy, stub renderer, LLM disabled: the integration-wave1 identity discipline, whose runner
`research/runners/iw1_default_identity.py` lives on branch research/integration-wave1) and records, per turn, the
sha256 of the raw body and of the canonical body (keys whose NAME looks like a wall-clock timing dropped, the same
VOLATILE_KEY_RE) plus the canonical body itself, so a byte difference can be explained key by key.

Scripts (`pmem` = the load_bearing_fraction PMEM driving group verbatim; `short` = the 3-turn degraded-brain probe):
  iw1       the 13 turns of iw1_default_identity.TURNS, verbatim (two sessions). No turn is a request, so fix (1)
            must leave all 13 byte-identical; with the corpus present fix (2) must too.
  requests  13 turns in one session: legitimate teach/recall turns interleaved with request-shaped turns (the defect
            class). Before the fix the request turns are ACQUIRED (recalled_svo = the junk triple); after, they are not.
            Differences are expected ONLY on request turns and on turns whose reply depends on the junk facts.

Arms are not flags: run the SAME runner file against two checkouts (`--root`), the pre-fix base and the fixed branch.

  python -u research/runners/chat_fact_fixes_identity.py --root <checkout> --script iw1 --seed 7 --out A.json
  python research/runners/chat_fact_fixes_identity.py --compare A.json B.json [--out V.json]

CORPUS: refuses (exit 3) to build without the four core corpus files unless --allow-degraded (then it measures the
degraded brain on purpose and records it). `--log-corpus-opens` installs an audit hook that records every file opened
under data/corpus/ during the run (the empirical list of corpus files the production brain reads).
No sim/ edit; reads only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

SESSION = "iw1-identity"
SESSION_B = "iw1-identity-b"
POS, NEG = "that was wonderful, I am so happy", "that was terrible, I am so sad"
SCRIPTS = {
    # verbatim research/runners/iw1_default_identity.py TURNS (branch research/integration-wave1 @ 35825741d)
    "iw1": [
        ("the dog and the cat walked in", SESSION),
        ("what does it chase", SESSION),
        ("the cat and the dog walked in", SESSION_B),
        ("what does it chase", SESSION_B),
        ("the wolf hunts the deer", SESSION),
        ("what does the cat eat", SESSION),
        ("what does the wolf hunt", SESSION),
        (POS, SESSION),
        (NEG, SESSION),
        (POS, SESSION),
        (NEG, SESSION),
        (POS, SESSION),
        ("what do you expect?", SESSION),
    ],
    "requests": [
        ("the wolf hunts the deer", "cff-requests"),          # legitimate teach
        ("tell me about the sky", "cff-requests"),            # REQUEST (the Stage-1 dev-smoke case)
        ("what does the wolf hunt", "cff-requests"),          # legitimate recall
        ("tell me about wolves", "cff-requests"),             # REQUEST
        ("talk about the moon", "cff-requests"),              # REQUEST (bare imperative)
        ("can you describe the moon", "cff-requests"),        # REQUEST (modal inversion)
        ("please remember the red ball", "cff-requests"),     # REQUEST (politeness marker)
        ("the cat eats fish", "cff-requests"),                # legitimate teach
        ("what does the cat eat", "cff-requests"),            # legitimate recall
        ("tell me more", "cff-requests"),                     # follow-up request (2 content words: never acquired)
        ("i like the sky", "cff-requests"),                   # control (2 content words: never acquired)
        ("the dog chases the ball", "cff-requests"),          # legitimate teach
        ("what does the dog chase", "cff-requests"),          # legitimate recall
    ],
    # the load_bearing_fraction PMEM driving group, verbatim (research/runners/onebrain_regression_battery.py labels
    # pmem_form2..pmem_cue): its third intervening turn is the request "tell me about the sky", which the pre-fix brain
    # ACQUIRED as a junk fact. Does removing that write disturb the cue turn's `prospective` read (the row's measure)?
    "pmem": [
        ("remind me to feed the dog when the bird sings", "pmem2"),
        ("what does the cat eat", "pmem2"),
        ("how is the weather today", "pmem2"),
        ("tell me about the sky", "pmem2"),
        ("the bird sings", "pmem2"),
    ],
    # the degraded-brain arms (no data/corpus/): does the reply SAY it is degraded? (fix 2)
    "short": [
        ("the wolf hunts the deer", "cff-short"),
        ("what does the wolf hunt", "cff-short"),
        ("tell me about the sky", "cff-short"),
    ],
}
CORPUS_FILES = ("tinystories.txt", "wikitext.txt", "simplewiki.txt", "websters1913.json")
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


def corpus_status(root: str) -> dict:
    missing = [f for f in CORPUS_FILES if not os.path.exists(os.path.join(root, "data", "corpus", f))]
    return {"corpus_present": not missing, "corpus_missing": missing}


def run_arm(root: str, script: str, seed: int, out: str, allow_degraded: bool, log_opens: bool) -> dict:
    root = os.path.abspath(root)
    cs = corpus_status(root)
    if not cs["corpus_present"] and not allow_degraded:
        print("⛔ chat_fact_fixes_identity: data/corpus/ in %s is missing %s; refusing to build a degraded brain "
              "(--allow-degraded measures it on purpose)." % (root, cs["corpus_missing"]), file=sys.stderr)
        raise SystemExit(3)
    opened = []
    if log_opens:
        def _hook(event, args):
            if event == "open" and args and isinstance(args[0], str) and "/data/corpus/" in args[0]:
                p = os.path.basename(args[0])
                if p not in opened:
                    opened.append(p)
        sys.addaudithook(_hook)
    os.chdir(root)
    here = os.path.dirname(os.path.abspath(__file__))       # run as a script: never let this file's own directory
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]   # shadow the checkout under test
    sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = root                          # any subprocess the brain spawns sees the same tree
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["SIM_DISABLE_LLM"] = "1"
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    from fastapi.testclient import TestClient  # noqa: E402
    from webapp import server as _server  # noqa: E402
    ltm_dir = _server._default_ltm_bundle_dir() if _server._ltm_ship_default_on() else None
    client = TestClient(_server.app)
    rows, t0 = [], time.time()
    for i, (msg, session) in enumerate(SCRIPTS[script]):
        t1 = time.time()
        res = client.post("/api/brain-chat", json={"session": session, "brain": "tiny-demo", "renderer": "stub",
                                                   "message": msg, "rich": False})
        body, dropped = res.content, []
        try:
            d = json.loads(body)
            cobj = _canon(d, "$", dropped)
            canon = json.dumps(cobj, sort_keys=True).encode()
        except Exception:
            d, cobj, canon = {}, None, body
        rows.append({"turn": i, "message": msg, "session": session, "status": res.status_code,
                     "sha_raw": hashlib.sha256(body).hexdigest(), "sha_canon": hashlib.sha256(canon).hexdigest(),
                     "dropped_volatile_keys": dropped, "answer": d.get("answer"), "abstained": d.get("abstained"),
                     "recalled_svo": d.get("recalled_svo"), "source": d.get("source"),
                     "corpus_present_field": d.get("corpus_present", "<absent>"),
                     "body_canon": cobj, "turn_wall_s": round(time.time() - t1, 2)})
        print("[cff-identity] %s turn %d status=%d canon=%s svo=%r ans=%r" % (
            script, i, res.status_code, rows[-1]["sha_canon"][:12], d.get("recalled_svo"),
            (d.get("answer") or "")[:90]), flush=True)
    rec = {"script": script, "seed": int(seed), "root": root, "corpus_present": cs["corpus_present"],
           "corpus_missing": cs["corpus_missing"], "allow_degraded": allow_degraded,
           "ltm_bundle": os.path.basename(ltm_dir) if ltm_dir else None,
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "sim_backend": os.environ.get("SIM_BACKEND"),
           "corpus_files_opened": opened if log_opens else None,
           "n_turns": len(SCRIPTS[script]), "wall_s": round(time.time() - t0, 1), "turns": rows,
           "session_sha_canon": hashlib.sha256("".join(r["sha_canon"] for r in rows).encode()).hexdigest()}
    try:
        import subprocess
        rec["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=root).stdout.strip() or None
        rec["git_dirty_tracked"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                                       capture_output=True, text=True, cwd=root).stdout.strip())
    except Exception:
        rec["git_sha"] = None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1)
    print("[cff-identity] %s session_sha_canon=%s wrote %s" % (script, rec["session_sha_canon"], out), flush=True)
    return rec


def _key_diff(x, y, path="$", out=None, limit=40):
    """Paths at which two canonical bodies differ (bounded)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append("%s.%s (%s)" % (path, k, "only-before" if k in x else "only-after"))
            else:
                _key_diff(x[k], y[k], "%s.%s" % (path, k), out, limit)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (a, b) in enumerate(zip(x, y)):
            _key_diff(a, b, "%s[%d]" % (path, i), out, limit)
    elif x != y:
        out.append(path)
    return out


def compare(a: dict, b: dict) -> dict:
    pre = {"same_script": a.get("script") == b.get("script"), "same_seed": a.get("seed") == b.get("seed"),
           "all_status_200": all(r.get("status") == 200 for r in a.get("turns", []) + b.get("turns", [])),
           "corpus_present_both": a.get("corpus_present") is True and b.get("corpus_present") is True,
           "same_ltm_bundle": bool(a.get("ltm_bundle")) and a.get("ltm_bundle") == b.get("ltm_bundle"),
           "same_pythonhashseed": a.get("pythonhashseed") == b.get("pythonhashseed") is not None}
    turns = []
    for x, y in zip(a.get("turns", []), b.get("turns", [])):
        if x.get("sha_canon") == y.get("sha_canon"):
            continue
        turns.append({"turn": x["turn"], "message": x["message"],
                      "reply_differs": [x.get(k) for k in ("answer", "abstained", "recalled_svo")]
                      != [y.get(k) for k in ("answer", "abstained", "recalled_svo")],
                      "before": {k: x.get(k) for k in ("answer", "abstained", "recalled_svo", "source")},
                      "after": {k: y.get(k) for k in ("answer", "abstained", "recalled_svo", "source")},
                      "differing_paths": _key_diff(x.get("body_canon"), y.get("body_canon"))})
    v = {"runner": "research/runners/chat_fact_fixes_identity.py --compare", "script": a.get("script"),
         "git_shas": [a.get("git_sha"), b.get("git_sha")], "preconditions": pre,
         "differing_turns": [t["turn"] for t in turns], "details": turns}
    v["verdict"] = ("UNDEFINED" if not all(pre.values()) else ("IDENTICAL" if not turns else "DIFFERS"))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="iw1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-degraded", action="store_true")
    ap.add_argument("--log-corpus-opens", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args(argv)
    if a.compare:
        v = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        v["inputs"] = [os.path.abspath(p) for p in a.compare]
        print(json.dumps({k: v[k] for k in ("verdict", "preconditions", "differing_turns")}, indent=1))
        for t in v["details"]:
            print("turn %d %r reply_differs=%s\n  before=%s\n  after =%s\n  paths=%s" % (
                t["turn"], t["message"], t["reply_differs"], t["before"], t["after"], t["differing_paths"][:12]))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(v, open(a.out, "w"), indent=1)
        return 0
    if not a.out:
        ap.error("--out is required unless --compare is given")
    run_arm(a.root, a.script, a.seed, a.out, a.allow_degraded, a.log_corpus_opens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
