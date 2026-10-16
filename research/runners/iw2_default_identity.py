"""Integration wave 2: production-default identity check through the REAL /api/brain-chat handler (2026-09-30).

The wave-1 instrument (research/runners/iw1_default_identity.py, branch research/integration-wave1) with the wave-1
review's lessons built in from the start, a SCOPE check added, and one brain alive at a time. A fixed 16-turn script
in ONE session, split into three segments that each begin with `reset: true` (the handler drops the session's
ChatBrain and every per-session organ, so only one brain is alive at a time: the pool dispatches at most 11 GB to a
15 GB node). Each segment first plays SCOPE turns (ordinary turns the packet must not change), then the turns that
reach its packet:

  segment fb   scope: "what does the dog chase"                         then the Sally-Anne story, 4 turns
  segment tr   scope: "does the cat sleep?" (a yes/no question that is not a taught chain)
               then teach cat->ball->worm->river ('precede'), ask "does cat precede river?" and "... ball?"
  segment si   scope: "the cat eats fish", "what does it eat?" (ordinary anaphora; 'it' is also a self-alias)
               then "what do you use?", "what are you?", "who are you?"

Each arm runs in ONE FRESH PROCESS, and the sha256 of every response body is recorded (canonical: keys whose NAME
looks like a wall-clock timing are dropped first -- the wave-1 VOLATILE_KEY_RE -- and every dropped path is listed).
Arms:
  default              every wave-2 flag UNSET (whatever the checkout's anchors say)
  off                  every wave-2 flag set EXPLICITLY to "0" (never popped: gates/flip_offarm_staleness)
  on:<packet>          that packet's flag "1", the others unset
  on:<p1>+<p2>...      several packets' flags "1"

Checks the packets use (shared prereg, steps D and F):
  pre-flip (pin P):  default == off on every turn                                  (D0, byte-identical-off)
                     on:<p> differs from default on >= 1 of p's own turns           (D_p, else vacuous: UNDEFINED)
                     on:<p> == default on every turn NOT tagged p (scope + others)  (S_p, else the flag over-reaches)
  post-flip (pin F): default == on:<flipped set> on every turn                     (F2)
                     off@F == default@P on every turn                              (F3)

  python -m research.runners.iw2_default_identity --arm default --seed 7 --out A.json
  python -m research.runners.iw2_default_identity --compare A.json B.json --expect same [--out V.json]
  python -m research.runners.iw2_default_identity --compare A.json B.json --expect differs --packet <p> [--out V.json]

Numpy backend, stub renderer, LLM disabled (the battery workers' discipline). No sim/ edit; reads only.

DEGRADED-BRAIN GUARD (wave-1 review, built in). A revision dir without data/corpus/ builds a DEGRADED brain with no
error anywhere in the replies (research/FAILURE_LOG.md 2026-09-23 and 2026-09-25), and a node without the knowledge-
bundle data lake builds a brain with no LTM. So an arm REFUSES (exit 3, before touching the environment or building
anything) when a load_bearing_fraction CORPUS GUARD file is missing or the shipped LTM default resolves no bundle, and
records `corpus_present` and `ltm_bundle`; `--compare` is UNDEFINED unless both arms read corpus_present True and the
same non-empty ltm_bundle. IW2_ALLOW_DEGRADED=1 measures the degraded brain on purpose (recorded). The arm also
records its peak RSS (`peak_rss_gb`, getrusage) so the job lines' declared mem_gb is a measured number.

DETERMINISM GUARD (dev seed 7, 2026-09-30). One turn's body ("does the cat sleep?", right after a segment reset)
carried a rounding residue in `curiosity.novelty` / `curiosity.graded_novelty.value` that took the values +1.1e-16,
-1.1e-16 or 0.0 in fresh processes of the SAME code and flags (measured with 2 BLAS threads, and again with 1), so a byte
comparison read a difference no flag caused. The residue follows the order of a floating-point sum, and that order
follows Python's per-process string-hash randomization (the Stage-1 exit runner pins PYTHONHASHSEED=0 for the same
reason). A byte-identity instrument must be deterministic: an arm refuses (exit 3) unless PYTHONHASHSEED is "0" and
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are all "1" (the job lines set them; PYTHONHASHSEED must be
in the environment BEFORE the interpreter starts), records them, and `--compare` is UNDEFINED unless both arms ran
that way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time

PACKET_FLAGS = {
    "false-belief-chat": ("BRAIN_FALSE_BELIEF_CHAT",),
    "transitive-chat": ("BRAIN_TRANSITIVE_CHAT",),
    "neural-selfid": ("BRAIN_NEURAL_SELFID",),
    "word-role": ("BRAIN_WORD_ROLE_PARSE", "BRAIN_WORD_ROLE_COMMIT"),
}
SESSION = "iw2-identity"
SCOPE = "scope"
# (message, tag, reset). tag = the packet whose flag the turn is there to reach, or SCOPE (no wave-2 flag may change
# it). Every segment starts with reset=True: one ChatBrain alive at a time. Scope turns come FIRST in their segment,
# so a scope difference can only be the flag acting on that turn itself, never carried-over state.
TURNS = [
    ("what does the dog chase", SCOPE, True),
    ("Sally puts the marble in the basket.", "false-belief-chat", False),
    ("Sally leaves the room.", "false-belief-chat", False),
    ("Anne moves the marble to the box.", "false-belief-chat", False),
    ("Where will Sally look for the marble?", "false-belief-chat", False),
    ("does the cat sleep?", SCOPE, True),
    ("the cat precedes the ball", "transitive-chat", False),
    ("the ball precedes the worm", "transitive-chat", False),
    ("the worm precedes the river", "transitive-chat", False),
    ("does cat precede river?", "transitive-chat", False),
    ("does cat precede ball?", "transitive-chat", False),
    ("the cat eats fish", SCOPE, True),
    ("what does it eat?", SCOPE, False),
    ("what do you use?", "neural-selfid", False),
    ("what are you?", "neural-selfid", False),
    ("who are you?", "neural-selfid", False),
]
# the same four files load_bearing_fraction's CORPUS GUARD requires (research/runners/load_bearing_fraction.py)
CORPUS_FILES = ("tinystories.txt", "wikitext.txt", "simplewiki.txt", "websters1913.json")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DETERMINISM_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
VOLATILE_KEY_RE = re.compile(r"(latency|elapsed|_ms$|_sec$|_secs$|seconds|time|timestamp|wall|duration|perf)", re.I)


def arm_env(arm: str) -> dict:
    """{flag: value or None(=unset)} for every wave-2 flag under `arm`."""
    flags = [f for fl in PACKET_FLAGS.values() for f in fl]
    if arm == "default":
        return {f: None for f in flags}
    if arm == "off":
        return {f: "0" for f in flags}
    if arm.startswith("off:"):
        # word-role-only-off (RULING wrc-f3-1015): zero ONLY the named packet's flags, every other flag at its
        # production default. Amendment B's "off" is this arm, not the all-flags `off` above.
        want = [p for p in arm[4:].split("+") if p]
        bad = [p for p in want if p not in PACKET_FLAGS]
        if bad or not want:
            raise SystemExit("unknown packet(s) in %r (known: %s)" % (arm, sorted(PACKET_FLAGS)))
        zero = {f for p in want for f in PACKET_FLAGS[p]}
        return {f: ("0" if f in zero else None) for f in flags}
    if arm.startswith("on:"):
        want = [p for p in arm[3:].split("+") if p]
        bad = [p for p in want if p not in PACKET_FLAGS]
        if bad or not want:
            raise SystemExit("unknown packet(s) in %r (known: %s)" % (arm, sorted(PACKET_FLAGS)))
        on = {f for p in want for f in PACKET_FLAGS[p]}
        return {f: ("1" if f in on else None) for f in flags}
    raise SystemExit("unknown arm %r" % arm)


def arm_file_name(arm: str) -> str:
    return arm.replace(":", "_").replace("+", "_")


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
    print("⛔ iw2_default_identity: %s. Refusing to build a degraded brain (set IW2_ALLOW_DEGRADED=1 to measure it "
          "on purpose)." % msg, file=sys.stderr, flush=True)
    raise SystemExit(3)


def _refuse_threads(th):
    print("⛔ iw2_default_identity: the environment %s must equal %s (hash-randomized iteration order or a "
          "multi-threaded reduction is not bit-reproducible; see the module docstring). Refusing to run."
          % (th, DETERMINISM_ENV), file=sys.stderr, flush=True)
    raise SystemExit(3)


def _rss_now_gb() -> float:
    try:
        return round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9, 2)
    except Exception:
        return None


def _peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2)


def threads_status(environ=None) -> dict:
    """The determinism-relevant environment of this process (see DETERMINISM GUARD)."""
    environ = os.environ if environ is None else environ
    return {k: environ.get(k) for k in DETERMINISM_ENV}


def run_arm(arm: str, seed: int, out: str) -> dict:
    allow_degraded = os.environ.get("IW2_ALLOW_DEGRADED") == "1"
    th = threads_status()
    if th != DETERMINISM_ENV:
        _refuse_threads(th)
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
    from research.runners.wr_route import wr_route as _wr_route  # noqa: E402 (Amendment E readout)
    ltm_dir = _server._default_ltm_bundle_dir() if _server._ltm_ship_default_on() else None
    if ltm_dir is None and not allow_degraded:
        _refuse("the shipped LTM default resolves no knowledge bundle (ship-default %s, BRAIN_DATA_ROOT=%r)"
                % (_server._ltm_ship_default_on(), os.environ.get("BRAIN_DATA_ROOT")))
    client = TestClient(_server.app)
    rows, t0 = [], time.time()
    for i, (msg, tag, reset) in enumerate(TURNS):
        if reset and i:
            # the handler drops the previous segment's ChatBrain on this reset; collect it now, so a reference cycle
            # does not keep two brains alive (the arm's peak RSS is the job line's declared mem_gb)
            import gc
            gc.collect()
        t1 = time.time()
        res = client.post("/api/brain-chat", json={"session": SESSION, "brain": "tiny-demo", "renderer": "stub",
                                                   "message": msg, "rich": False, "reset": bool(reset)})
        body, dropped = res.content, []
        try:
            d = json.loads(body)
            canon = json.dumps(_canon(d, "$", dropped), sort_keys=True).encode()
        except Exception:
            d, canon = {}, body
        rows.append({"turn": i, "message": msg, "for": tag, "reset": bool(reset), "status": res.status_code,
                     "sha_raw": hashlib.sha256(body).hexdigest(), "sha_canon": hashlib.sha256(canon).hexdigest(),
                     "dropped_volatile_keys": dropped, "answer": d.get("answer"), "abstained": d.get("abstained"),
                     "recalled_svo": d.get("recalled_svo"), "source": d.get("source"),
                     "wr_route": _wr_route(d.get("word_role_trace")),
                     "turn_wall_s": round(time.time() - t1, 2), "peak_rss_gb_after": _peak_rss_gb(),
                     "rss_gb_after": _rss_now_gb()})
        print("[iw2-identity] %s turn %d status=%d canon=%s ans=%r" % (arm, i, res.status_code,
              rows[-1]["sha_canon"][:12], (d.get("answer") or "")[:100]), flush=True)
    rec = {"arm": arm, "seed": int(seed), "cwd": os.getcwd(), "flags_env": {f: os.environ.get(f) for f in env},
           "corpus_present": cs["corpus_present"], "corpus_missing": cs["corpus_missing"],
           "ltm_bundle": os.path.basename(ltm_dir) if ltm_dir else None, "ltm_bundle_dir": ltm_dir,
           "allow_degraded": allow_degraded, "sources": sorted({str(r["source"]) for r in rows}),
           "threads": th,
           "sim_backend": os.environ.get("SIM_BACKEND"), "n_turns": len(TURNS),
           "script_sha": hashlib.sha256(json.dumps(TURNS).encode()).hexdigest(),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb(), "turns": rows,
           "session_sha_canon": hashlib.sha256("".join(r["sha_canon"] for r in rows).encode()).hexdigest()}
    try:
        import subprocess
        rec["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=os.getcwd()).stdout.strip() or None
    except Exception:
        rec["git_sha"] = None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1)
    print("[iw2-identity] %s session_sha_canon=%s peak_rss_gb=%s wrote %s" % (arm, rec["session_sha_canon"],
          rec["peak_rss_gb"], out), flush=True)
    return rec


def _reply(t):
    return [t.get(k) for k in ("answer", "abstained", "recalled_svo")]


def compare(a: dict, b: dict, expect: str, packet: str = None) -> dict:
    """Per-turn sha_canon comparison. expect='same': every turn identical (and every status 200). expect='differs'
    with `packet`: the S_p/D_p pair -- `ok` only if at least one turn tagged `packet` differs (else VACUOUS, UNDEFINED)
    AND every turn not tagged `packet` is identical (else OUT-OF-SCOPE, a FAIL). UNDEFINED when a precondition (same
    seed, same script, every status 200, every turn present, corpus + the same LTM on both arms) fails."""
    if expect == "differs" and packet not in PACKET_FLAGS:
        raise SystemExit("--expect differs needs --packet (one of %s)" % sorted(PACKET_FLAGS))
    ta, tb = a.get("turns", []), b.get("turns", [])
    pre = {"same_seed": a.get("seed") == b.get("seed"), "same_n_turns": a.get("n_turns") == b.get("n_turns"),
           "same_script": bool(a.get("script_sha")) and a.get("script_sha") == b.get("script_sha"),
           "all_status_200": all(r.get("status") == 200 for r in ta + tb),
           "all_turns_present": all(len(x.get("turns", [])) == x.get("n_turns") for x in (a, b)),
           "corpus_present_both": a.get("corpus_present") is True and b.get("corpus_present") is True,
           "same_ltm_bundle": bool(a.get("ltm_bundle")) and a.get("ltm_bundle") == b.get("ltm_bundle"),
           "deterministic_env_both": all((x.get("threads") or {}) == DETERMINISM_ENV for x in (a, b))}
    diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if x.get("sha_canon") != y.get("sha_canon")]
    reply_diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if _reply(x) != _reply(y)]
    backends = sorted({str(a.get("sim_backend")), str(b.get("sim_backend"))})
    v = {"runner": "research.runners.iw2_default_identity --compare", "argv": list(sys.argv),
         "sim_backend": backends[0] if len(backends) == 1 else backends,   # the compared arms' backend (no build here)
         "arms": [a.get("arm"), b.get("arm")], "git_shas": [a.get("git_sha"), b.get("git_sha")], "expect": expect,
         "packet": packet, "ltm_bundles": [a.get("ltm_bundle"), b.get("ltm_bundle")],
         "sources": [a.get("sources"), b.get("sources")], "preconditions": pre, "differing_turns": diff,
         "reply_differing_turns": reply_diff,
         "reply_diffs": [{"turn": i, "message": ta[i].get("message"), "for": ta[i].get("for"),
                          "a": ta[i].get("answer"), "b": tb[i].get("answer")} for i in reply_diff]}
    if not all(pre.values()):
        v["verdict"], v["ok"] = "UNDEFINED", None
    elif expect == "same":
        v["ok"] = not diff
        v["verdict"] = "IDENTICAL" if v["ok"] else "DIFFERS"
    else:
        own = [i for i in diff if ta[i].get("for") == packet]
        other = [i for i in diff if ta[i].get("for") != packet]
        v["own_differing_turns"], v["out_of_scope_differing_turns"] = own, other
        if other:
            v["ok"], v["verdict"] = False, "OUT-OF-SCOPE (the flag changes a turn it should not reach: %s)" % other
        elif not own:
            v["ok"], v["verdict"] = None, "UNDEFINED (vacuous: the script does not reach this flag)"
        else:
            v["ok"], v["verdict"] = True, "DIFFERS-IN-SCOPE"
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--expect", choices=("same", "differs"), default="same")
    ap.add_argument("--packet", default=None, choices=sorted(PACKET_FLAGS))
    a = ap.parse_args(argv)
    if a.compare:
        v = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])), a.expect, a.packet)
        v["inputs"] = [os.path.abspath(p) for p in a.compare]
        print(json.dumps(v, indent=1))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(v, open(a.out, "w"), indent=1)
        # exit: 0 PASS, 1 FAIL, 3 UNDEFINED (the registered scorer convention)
        return 0 if v["ok"] is True else (3 if v["ok"] is None else 1)
    if not (a.arm and a.out):
        ap.error("--arm and --out are required unless --compare is given")
    run_arm(a.arm, a.seed, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
