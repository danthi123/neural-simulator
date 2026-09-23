"""FLAG-OFF IDENTITY, ASSERTED IN DATA -- the default-OFF open-ended routes change nothing when off.

Lane research/open-ended-production-turn-lb, fix round (2026-09-23). The review required byte-identity of the flag-off
reply to be checked in DATA against a PINNED pre-change SHA, not inferred from reading the short-circuit.

  dump    : run a fixed chat script through the REAL `webapp.server.brain_chat` of the tree at --repo (a clean
            `git archive` of one commit) and write every full JSON response body, in order, to --out.
  compare : exact compare of two dumps (json.dumps(body, sort_keys=True) per turn). IDENTICAL only if every turn is
            equal and both dumps are complete; a missing / errored dump is UNDEFINED, never a pass.

Run as a FILE (not -m) so the imported `webapp`/`research` packages come from --repo, not from this checkout:
  python research/runners/_lbf_oe_route_flag_off_identity.py dump --repo <tree> --env-set default --out a.json
  python research/runners/_lbf_oe_route_flag_off_identity.py compare --a pre.json --b post.json --out verdict.json

Env sets. `default`: no BRAIN_OPEN_ENDED* variable at all (today's production turn). `oe_off`: BRAIN_OPEN_ENDED=1 +
BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK=1 with the warm Qwen faculty stubbed (declared: FORM not measured), and the two
route flags UNSET (their default). Both run numpy, SIM_DISABLE_LLM=1, renderer stub, BRAIN_CHAT_SEED=42.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCRIPT = [
    ("the wolf chase the rabbit", False),
    ("the lion chase the deer", False),
    ("the crow chase the beetle", False),
    ("the coyote chase the dog", False),
    ("what might a dog chase", True),
    ("what does the wolf chase?", True),
    ("tell me about the ocean", True),
    ("the owl chase the mouse", False),
    ("what might a dog chase", True),
]
ENV_SETS = {
    "default": {},
    "oe_off": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
}
ROUTE_FLAGS = ("BRAIN_OPEN_ENDED_GENERATE_ROUTE", "BRAIN_OPEN_ENDED_ACQUIRE_ROUTE")


class _StubFaculty:
    def __getattr__(self, name):
        raise RuntimeError("stub Qwen faculty used (%s)" % name)


def dump(repo, env_set, out):
    repo = os.path.abspath(repo)
    if not os.path.isfile(os.path.join(repo, "data", "corpus", "tinystories.txt")):
        # untracked corpus missing -> the one-brain XEDGE build fails and the webapp degrades to standalone organs:
        # not the production brain. Refuse rather than compare two degraded brains.
        raise SystemExit("REFUSED: %s has no data/corpus (symlink data -> the main checkout's data/)" % repo)
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ["SIM_BACKEND"] = "numpy"
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    os.environ["SIM_DISABLE_LLM"] = "1"
    os.environ["BRAIN_CHAT_SEED"] = "42"
    for f in ROUTE_FLAGS:
        os.environ.pop(f, None)                      # OFF = the default = unset
    for k in [k for k in os.environ if k.startswith("BRAIN_OPEN_ENDED")]:
        os.environ.pop(k)
    os.environ.update(ENV_SETS[env_set])
    import logging
    logging.disable(logging.INFO)
    import webapp.server as S
    assert os.path.dirname(os.path.abspath(S.__file__)) == os.path.join(repo, "webapp"), S.__file__
    if env_set == "oe_off":
        S._get_warm_qwen_renderer = lambda: type("R", (), {"_fac": _StubFaculty()})()
    turns = []
    t0 = time.time()
    for i, (msg, rich) in enumerate(SCRIPT):
        try:
            r = S.brain_chat(S.BrainChatRequest(session="ident", message=msg, brain="tiny-demo", renderer="stub",
                                                rich=rich, reset=(i == 0)))
            turns.append({"msg": msg, "status": getattr(r, "status_code", None), "body": json.loads(r.body)})
        except Exception as e:
            turns.append({"msg": msg, "error": "%s: %s" % (type(e).__name__, e)})
    rec = {"repo": repo, "server_file": S.__file__, "env_set": env_set, "env": ENV_SETS[env_set],
           "route_flags_in_env": {f: os.environ.get(f) for f in ROUTE_FLAGS},
           "has_route_fn": hasattr(S, "_open_ended_brain_route"), "turns": turns,
           "complete": len(turns) == len(SCRIPT), "t_s": round(time.time() - t0, 1)}
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print("[identity dump] %s env=%s turns=%d errors=%d -> %s" % (
        repo, env_set, len(turns), sum(1 for t in turns if "error" in t), out), flush=True)
    return 0


# WALL-CLOCK fields: measured durations, different on every run of the SAME code (the pre-vs-pre control shows it).
# They are stripped for the content verdict; the raw verdict keeps them, and every differing path is listed so the
# exclusion can be audited. Found on the first real compare: `open_ended.gen_seconds` was the ONLY differing path.
WALLCLOCK_KEYS = frozenset({"gen_seconds"})


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in WALLCLOCK_KEYS}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _canon(turn, strip=True):
    return json.dumps(_strip(turn) if strip else turn, sort_keys=True, default=str)


def _diff_paths(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff_paths(a.get(k), b.get(k), p + "." + str(k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, "%s[%d]" % (p, i))
        return out
    return [] if a == b else [p]


def compare(a_path, b_path, out, a_sha=None, b_sha=None):
    try:
        A, B = json.load(open(a_path)), json.load(open(b_path))
    except Exception as e:
        rec = {"verdict": "UNDEFINED", "reason": "missing dump: %r" % e}
    else:
        if not (A.get("complete") and B.get("complete")) or A.get("env_set") != B.get("env_set"):
            rec = {"verdict": "UNDEFINED", "reason": "incomplete dump or env-set mismatch"}
        else:
            same_len = len(A["turns"]) == len(B["turns"])
            per = [(_canon(x) == _canon(y)) for x, y in zip(A["turns"], B["turns"])]
            per_raw = [(_canon(x, strip=False) == _canon(y, strip=False)) for x, y in zip(A["turns"], B["turns"])]
            paths = sorted({q for x, y in zip(A["turns"], B["turns"]) for q in _diff_paths(x, y)})
            n_err = sum(1 for t in A["turns"] + B["turns"] if "error" in t)
            raw = "IDENTICAL" if (all(per_raw) and same_len) else "DIFFERENT"
            content = "IDENTICAL" if (all(per) and same_len) else "DIFFERENT"
            # AMENDMENT 3 (review 2026-09-23): `verdict` IS the PRE-REGISTERED criterion (amendment 2 item 9: exact
            # compare of every full response body). The wall-clock-stripped comparison was added AFTER the first
            # compare had failed raw, so it is reported beside it as a POST-HOC content verdict -- never as the
            # headline. A reader of `verdict` must see the raw FAIL.
            rec = {"verdict": raw, "verdict_preregistered_raw": raw,
                   "verdict_content_wallclock_stripped_POSTHOC": content,
                   "headline": ("%s raw (pre-registered criterion)%s; content %s with wall-clock keys %s stripped "
                                "(post-hoc exclusion, added after the raw compare was seen)"
                                % (raw, (" -- FAILED at " + ", ".join(paths)) if raw != "IDENTICAL" else "",
                                   content, sorted(WALLCLOCK_KEYS))),
                   "wallclock_keys_excluded": sorted(WALLCLOCK_KEYS), "differing_paths_raw": paths,
                   "per_turn_equal": per, "n_turns": len(per), "n_error_turns_total": n_err,
                   "env_set": A["env_set"], "a_has_route_fn": A.get("has_route_fn"),
                   "b_has_route_fn": B.get("has_route_fn")}
    rec.update({"a": a_path, "b": b_path, "a_sha": a_sha, "b_sha": b_sha,
                "compare": "json.dumps(turn, sort_keys=True) exact string equality, every turn"})
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print("[identity compare] %s" % rec["verdict"], flush=True)
    return 0 if rec["verdict"] == "IDENTICAL" else 1


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    base = {"env_set": "default", "complete": True, "turns": [{"msg": "x", "body": {"answer": "a", "n": 1}}]}
    same = {"env_set": "default", "complete": True, "turns": [{"msg": "x", "body": {"n": 1, "answer": "a"}}]}
    diff = {"env_set": "default", "complete": True, "turns": [{"msg": "x", "body": {"answer": "b", "n": 1}}]}
    inc = dict(base, complete=False)
    clk = {"env_set": "default", "complete": True,
           "turns": [{"msg": "x", "body": {"answer": "a", "n": 1, "oe": {"gen_seconds": 0.4}}}]}
    clk2 = {"env_set": "default", "complete": True,
            "turns": [{"msg": "x", "body": {"answer": "a", "n": 1, "oe": {"gen_seconds": 0.3}}}]}
    clk3 = {"env_set": "default", "complete": True,
            "turns": [{"msg": "x", "body": {"answer": "a", "n": 1, "oe": {"gen_seconds": 0.3, "raw": "z"}}}]}
    paths = {}
    for k, v in (("base", base), ("same", same), ("diff", diff), ("inc", inc), ("clk", clk), ("clk2", clk2),
                 ("clk3", clk3)):
        paths[k] = os.path.join(d, k + ".json")
        json.dump(v, open(paths[k], "w"))
    o = os.path.join(d, "v.json")
    ok = True
    for name, a, b, want in (("key order ignored -> IDENTICAL", "base", "same", "IDENTICAL"),
                             ("changed answer -> DIFFERENT", "base", "diff", "DIFFERENT"),
                             ("incomplete -> UNDEFINED", "base", "inc", "UNDEFINED"),
                             ("wall-clock-only difference -> headline verdict DIFFERENT (pre-registered raw)", "clk",
                              "clk2", "DIFFERENT"),
                             ("nested content difference beside wall-clock -> DIFFERENT", "clk", "clk3",
                              "DIFFERENT")):
        compare(paths[a], paths[b], o)
        got = json.load(open(o))["verdict"]
        print(("PASS " if got == want else "FAIL ") + name)
        ok = ok and got == want
    compare(paths["clk"], paths["clk2"], o)
    r = json.load(open(o))
    c = (r["verdict_content_wallclock_stripped_POSTHOC"] == "IDENTICAL" and "FAILED" in r["headline"]
         and "post-hoc" in r["headline"])
    print(("PASS " if c else "FAIL ") + "wall-clock-only: content IDENTICAL reported beside, headline says raw FAILED")
    ok = ok and c
    compare(paths["base"], os.path.join(d, "nope.json"), o)
    got = json.load(open(o))["verdict"]
    print(("PASS " if got == "UNDEFINED" else "FAIL ") + "missing dump -> UNDEFINED")
    ok = ok and got == "UNDEFINED"
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["dump", "compare", "selftest"])
    ap.add_argument("--repo")
    ap.add_argument("--env-set", choices=sorted(ENV_SETS))
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--a-sha")
    ap.add_argument("--b-sha")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "dump":
        return dump(a.repo, a.env_set, a.out)
    return compare(a.a, a.b, a.out, a_sha=a.a_sha, b_sha=a.b_sha)


if __name__ == "__main__":
    raise SystemExit(main())
