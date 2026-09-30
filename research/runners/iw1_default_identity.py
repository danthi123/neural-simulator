"""Integration wave 1: production-default identity check through the REAL /api/brain-chat handler (2026-09-29).

A fixed script that reaches all three wave-1 mechanisms (focus-bind: a two-referent intro then an anaphor, both orders;
D6 write: teach then recall; C1: alternating valence then an expectation query) is run in ONE FRESH PROCESS per arm,
and the sha256 of every response body is recorded (canonical: keys whose NAME looks like a wall-clock timing are
dropped first, the same VOLATILE_KEY_RE as research/runners/_flip_validated_fixes_chat_sha.py, and every dropped
path is listed). Arms:

  default              every wave-1 flag UNSET (whatever the checkout's anchors say)
  off                  every wave-1 flag set EXPLICITLY to "0" (never popped: gates/flip_offarm_staleness)
  on:<packet>          that packet's flags "1", the others unset
  on:<p1>+<p2>...      several packets' flags "1"

Checks the packets use (HANDOFF step D and step F):
  pre-flip (pin P):  default == off on every turn            (byte-identical-off)
                     on:<packet> != default on >= 1 turn      (the script reaches the flag; else the check is vacuous)
  post-flip (pin F): default == on:<flipped set> on every turn (the flipped default IS the capability)
                     off@F == default@P on every turn         (the explicit-off escape reproduces the pre-flip default)

  python -m research.runners.iw1_default_identity --arm default --seed 7 --out A.json
  python -m research.runners.iw1_default_identity --compare A.json B.json --expect same|differs [--out V.json]

Numpy backend, stub renderer, LLM disabled (the battery workers' discipline). No sim/ edit; reads only.

DEGRADED-BRAIN GUARD (adversarial review 2026-09-29). A pool revision dir without data/corpus/ builds a DEGRADED brain
with no error anywhere in the replies (the ONEBRAIN XEDGE build falls back to standalone organs: research/
FAILURE_LOG.md 2026-09-23 and 2026-09-25), and a node without the knowledge-bundle data lake builds a brain with no
LTM. Either would make a byte comparison across arms (which land on different nodes) meaningless. So an arm REFUSES
(exit 3, before touching the environment or building anything) when a load_bearing_fraction CORPUS GUARD file is
missing or the shipped LTM default resolves no bundle, and records `corpus_present` and `ltm_bundle`; `--compare` is
UNDEFINED unless both arms read corpus_present True and the same non-empty ltm_bundle. IW1_ALLOW_DEGRADED=1 measures
the degraded brain on purpose (recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

PACKET_FLAGS = {
    "d6-hebbian-write": ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME"),
    "wm-focus-bind": ("BRAIN_MULTIREF_FOCUS_BIND",),
    "xedge-surprise-worldmodel": ("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL",),
}
SESSION = "iw1-identity"
SESSION_B = "iw1-identity-b"   # the focus probe's swapped intro order, in its own session (per-session organs)
POS, NEG = "that was wonderful, I am so happy", "that was terrible, I am so sad"
# (message, which packet the turn is there for, session). The two orderings of the focus intro mirror
# LB_WMB_FOCUS_PROBE's pair A: with the flag on, the two anaphor replies follow the held referent (they differ); the
# positional route answers both the same, so at least one ordering differs from the flag-off reply.
TURNS = [
    ("the dog and the cat walked in", "wm-focus-bind", SESSION),
    ("what does it chase", "wm-focus-bind", SESSION),
    ("the cat and the dog walked in", "wm-focus-bind", SESSION_B),
    ("what does it chase", "wm-focus-bind", SESSION_B),
    ("the wolf hunts the deer", "d6-hebbian-write", SESSION),
    ("what does the cat eat", "d6-hebbian-write", SESSION),
    ("what does the wolf hunt", "d6-hebbian-write", SESSION),
    (POS, "xedge-surprise-worldmodel", SESSION),
    (NEG, "xedge-surprise-worldmodel", SESSION),
    (POS, "xedge-surprise-worldmodel", SESSION),
    (NEG, "xedge-surprise-worldmodel", SESSION),
    (POS, "xedge-surprise-worldmodel", SESSION),
    ("what do you expect?", "xedge-surprise-worldmodel", SESSION),
]
# the same four files load_bearing_fraction's CORPUS GUARD requires (research/runners/load_bearing_fraction.py)
CORPUS_FILES = ("tinystories.txt", "wikitext.txt", "simplewiki.txt", "websters1913.json")
VOLATILE_KEY_RE = re.compile(r"(latency|elapsed|_ms$|_sec$|_secs$|seconds|time|timestamp|wall|duration|perf)", re.I)


def arm_env(arm: str) -> dict:
    """{flag: value or None(=unset)} for every wave-1 flag under `arm`."""
    flags = [f for fl in PACKET_FLAGS.values() for f in fl]
    if arm == "default":
        return {f: None for f in flags}
    if arm == "off":
        return {f: "0" for f in flags}
    if arm.startswith("on:"):
        want = [p for p in arm[3:].split("+") if p]
        bad = [p for p in want if p not in PACKET_FLAGS]
        if bad or not want:
            raise SystemExit("unknown packet(s) in %r (known: %s)" % (arm, sorted(PACKET_FLAGS)))
        on = {f for p in want for f in PACKET_FLAGS[p]}
        return {f: ("1" if f in on else None) for f in flags}
    raise SystemExit("unknown arm %r" % arm)


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


def _refuse(msg: str):
    print("⛔ iw1_default_identity: %s. Refusing to build a degraded brain (set IW1_ALLOW_DEGRADED=1 to measure it "
          "on purpose)." % msg, file=sys.stderr, flush=True)
    raise SystemExit(3)


def run_arm(arm: str, seed: int, out: str) -> dict:
    allow_degraded = os.environ.get("IW1_ALLOW_DEGRADED") == "1"
    cs = corpus_status(os.getcwd())
    if not cs["corpus_present"] and not allow_degraded:
        _refuse("data/corpus/ in %s is missing %s" % (os.getcwd(), cs["corpus_missing"]))
    env = arm_env(arm)
    for f, v in env.items():
        if v is None:
            os.environ.pop(f, None)
        else:
            os.environ[f] = v
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["SIM_DISABLE_LLM"] = "1"
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    sys.path.insert(0, os.getcwd())
    from fastapi.testclient import TestClient  # noqa: E402
    from webapp import server as _server  # noqa: E402
    ltm_dir = _server._default_ltm_bundle_dir() if _server._ltm_ship_default_on() else None
    if ltm_dir is None and not allow_degraded:
        _refuse("the shipped LTM default resolves no knowledge bundle (ship-default %s, BRAIN_DATA_ROOT=%r)"
                % (_server._ltm_ship_default_on(), os.environ.get("BRAIN_DATA_ROOT")))
    client = TestClient(_server.app)
    rows, t0 = [], time.time()
    for i, (msg, packet, session) in enumerate(TURNS):
        t1 = time.time()
        res = client.post("/api/brain-chat", json={"session": session, "brain": "tiny-demo", "renderer": "stub",
                                                   "message": msg, "rich": False})
        body, dropped = res.content, []
        try:
            d = json.loads(body)
            canon = json.dumps(_canon(d, "$", dropped), sort_keys=True).encode()
        except Exception:
            d, canon = {}, body
        rows.append({"turn": i, "message": msg, "for": packet, "session": session, "status": res.status_code,
                     "sha_raw": hashlib.sha256(body).hexdigest(), "sha_canon": hashlib.sha256(canon).hexdigest(),
                     "dropped_volatile_keys": dropped, "answer": d.get("answer"), "abstained": d.get("abstained"),
                     "recalled_svo": d.get("recalled_svo"), "source": d.get("source"),
                     "turn_wall_s": round(time.time() - t1, 2)})
        print("[iw1-identity] %s turn %d status=%d canon=%s ans=%r" % (arm, i, res.status_code,
              rows[-1]["sha_canon"][:12], (d.get("answer") or "")[:90]), flush=True)
    rec = {"arm": arm, "seed": int(seed), "cwd": os.getcwd(), "flags_env": {f: os.environ.get(f) for f in env},
           "corpus_present": cs["corpus_present"], "corpus_missing": cs["corpus_missing"],
           "ltm_bundle": os.path.basename(ltm_dir) if ltm_dir else None, "ltm_bundle_dir": ltm_dir,
           "allow_degraded": allow_degraded, "sources": sorted({str(r["source"]) for r in rows}),
           "sim_backend": os.environ.get("SIM_BACKEND"), "n_turns": len(TURNS),
           "wall_s": round(time.time() - t0, 1), "turns": rows,
           "session_sha_canon": hashlib.sha256("".join(r["sha_canon"] for r in rows).encode()).hexdigest()}
    try:
        import subprocess
        rec["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=os.getcwd()).stdout.strip() or None
    except Exception:
        rec["git_sha"] = None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1)
    print("[iw1-identity] %s session_sha_canon=%s wrote %s" % (arm, rec["session_sha_canon"], out), flush=True)
    return rec


def compare(a: dict, b: dict, expect: str) -> dict:
    """Per-turn sha_canon comparison. expect='same': every turn identical (and every status 200). expect='differs':
    at least one turn differs. Returns a verdict dict; `ok` is the check's outcome; UNDEFINED when a precondition
    (same seed, same script length, every status 200) fails."""
    pre = {"same_seed": a.get("seed") == b.get("seed"), "same_n_turns": a.get("n_turns") == b.get("n_turns"),
           "all_status_200": all(r.get("status") == 200 for r in a.get("turns", []) + b.get("turns", [])),
           "all_turns_present": all(len(x.get("turns", [])) == x.get("n_turns") for x in (a, b)),
           # degraded-brain guard (module docstring): both arms built with the corpus and the same shipped LTM
           "corpus_present_both": a.get("corpus_present") is True and b.get("corpus_present") is True,
           "same_ltm_bundle": bool(a.get("ltm_bundle")) and a.get("ltm_bundle") == b.get("ltm_bundle")}
    diff = [i for i, (x, y) in enumerate(zip(a.get("turns", []), b.get("turns", [])))
            if x.get("sha_canon") != y.get("sha_canon")]
    reply_diff = [i for i, (x, y) in enumerate(zip(a.get("turns", []), b.get("turns", [])))
                  if [x.get(k) for k in ("answer", "abstained", "recalled_svo")]
                  != [y.get(k) for k in ("answer", "abstained", "recalled_svo")]]
    backends = sorted({str(a.get("sim_backend")), str(b.get("sim_backend"))})
    v = {"runner": "research.runners.iw1_default_identity --compare", "argv": list(sys.argv),
         "sim_backend": backends[0] if len(backends) == 1 else backends,   # the compared arms' backend (no build here)
         "arms": [a.get("arm"), b.get("arm")], "git_shas": [a.get("git_sha"), b.get("git_sha")], "expect": expect,
         "ltm_bundles": [a.get("ltm_bundle"), b.get("ltm_bundle")], "sources": [a.get("sources"), b.get("sources")],
         "preconditions": pre, "differing_turns": diff, "reply_differing_turns": reply_diff}
    if not all(pre.values()):
        v["verdict"], v["ok"] = "UNDEFINED", None
    elif expect == "same":
        v["ok"] = not diff
        v["verdict"] = "IDENTICAL" if v["ok"] else "DIFFERS"
    else:
        v["ok"] = bool(diff)
        v["verdict"] = "DIFFERS" if v["ok"] else "IDENTICAL (vacuous: the script does not reach this flag)"
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--expect", choices=("same", "differs"), default="same")
    a = ap.parse_args(argv)
    if a.compare:
        v = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])), a.expect)
        v["inputs"] = [os.path.abspath(p) for p in a.compare]
        print(json.dumps(v, indent=1))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(v, open(a.out, "w"), indent=1)
        return 0 if v["ok"] else 1
    if not (a.arm and a.out):
        ap.error("--arm and --out are required unless --compare is given")
    run_arm(a.arm, a.seed, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
