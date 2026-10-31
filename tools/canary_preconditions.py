#!/usr/bin/env python3
"""tools/canary_preconditions.py -- a pool canary PASS also needs the runner's REGISTERED SCORER to read the canary's
own output with its preregistered preconditions holding, and a queued line must carry the flags its instrument
declares (2026-10-31).

WHY. On 2026-10-31 the BG live-seam step-1 sub-step grid (research/findings/raw/_bg_live_seam/step1/507b9c789/, 8
sets x seeds 71-78 = 64 cells, about 80 CPU-hours) was found VOID: its lines had dropped `--commit` (the e0b8608
grid's lines carried it), so no cell ran the commit stage and the registered scorer (`host_share_bg_seam --score
<set> --seeds ... --commit`) read every set UNDEFINED (commit_engaged 0.0, commit_op_ok 0.0, commit_seed_ok 0.0).
The grid's canary had PASSED, because tools/pool_canary.sh judged only rc, the JSON count and error fields, never
the instrument's own preconditions (research/FAILURE_LOG.md, 2026-10-31; RULING bg-t3-1031).

Two checks, both driven by the registry tools/canary_preconditions.json (override: CANARY_PRECONDITIONS_REGISTRY):

  check-line '<line>'   exit 0 iff the line carries every flag its runner's registry entry requires for the output
                        paths it writes (e.g. a host_share_bg_seam line writing under _bg_live_seam/step1/ must pass
                        --commit). Exit 1 with the reason otherwise. tools/pool_queue.sh `add` runs it on EVERY line,
                        dev seeds included (the 2026-10-31 grid ran dev seeds 71-78, where no gate-seed guard looks),
                        and tools/pool_canary.sh `run` refuses to launch a canary of a line that fails it.
  resolve '<line>'      print the scorer spec for the line as JSON (argv, env, the JSON key, the requirements), or
                        nothing when the runner has no registered checker. Templates: {python} is the interpreter
                        the line runs the module with, {module} the module, {arg:--x} the value of --x in the line.
                        A requirement with "when": "flag:--x" applies only when the line passes --x.
  node-run <b64 spec>   run the spec's scorer from the current directory (the pinned revision on the node), take the
                        LAST stdout line that parses as a JSON object holding the spec's key, and check each
                        requirement. Prints {"ok": bool, "checks": [...], "error": str|null}; exit 0 iff ok. A scorer
                        that crashes, prints no such line, or yields a missing/None field FAILS (a check that cannot
                        read its instrument must fail loudly, never pass).

Stdlib only: pool_canary.sh ships this file to the node and runs it with python3. It never reads an outcome value
that a prereg gates (G1-G5); it reads only fields the prereg declares as PRECONDITIONS (UNDEFINED, never NO-GO).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.environ.get("CANARY_PRECONDITIONS_REGISTRY") or os.path.join(HERE, "canary_preconditions.json")


def load_registry(path=None):
    with open(path or REGISTRY) as fh:
        reg = json.load(fh)
    return {k: v for k, v in reg.items() if not k.startswith("_") and isinstance(v, dict)}


def bare_line(s):
    """Accept a raw queue line too: drop a leading "<epoch><TAB>" and the "  #checked:" reason (as pool_canary.sh)."""
    m = re.match(r"^[0-9]+\t(.*)$", s, re.S)
    if m:
        s = m.group(1)
    return s.split("  #checked:", 1)[0]


def line_module(line):
    """The module the line runs: a research.runners module if any, else the first `-m X` (as pool_canary.sh)."""
    m = re.search(r"-m +(research\.runners\.[A-Za-z0-9_]+)", line)
    if not m:
        m = re.search(r"(?:^|\s)-m +([A-Za-z_][A-Za-z0-9_.]*)", line)
    return m.group(1) if m else ""


def _argv_after_module(line, module):
    """The tokens after `-m <module>` up to the end of that command (a shell separator ends it)."""
    i = line.find("-m " + module)
    if i < 0:
        m = re.search(r"-m +" + re.escape(module), line)
        if not m:
            return []
        i = m.start()
    rest = line[i:]
    try:
        toks = shlex.split(rest)
    except ValueError:
        toks = rest.split()
    out = []
    for t in toks[2:]:
        if t in ("&&", "||", ";", "|", "&") or t.endswith(";"):
            break
        out.append(t)
    return out


def flag_value(argv, flag):
    for i, t in enumerate(argv):
        if t == flag and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def has_flag(argv, flag):
    return any(t == flag or t.startswith(flag + "=") for t in argv)


def line_python(line, module):
    """The interpreter that runs `-m <module>`: the token before it, skipping interpreter options like -u."""
    m = re.search(r"(\S+)((?:\s+-[A-Za-z]+)*)\s+-m +" + re.escape(module), line)
    return m.group(1) if m else "python3"


def missing_flags(line, registry=None):
    """[(flag, why)] the registry requires of this line and it does not carry."""
    reg = registry if registry is not None else load_registry()
    line = bare_line(line)
    mod = line_module(line)
    entry = reg.get(mod)
    if not entry:
        return []
    argv = _argv_after_module(line, mod)
    joined = " ".join(argv)
    out = []
    for rule in entry.get("requires_flags", []):
        if re.search(rule["path_regex"], joined) and not has_flag(argv, rule["flag"]):
            out.append((rule["flag"], rule["why"]))
    return out


def _fill(tmpl, ctx, argv):
    def sub(m):
        key = m.group(1)
        if key.startswith("arg:"):
            v = flag_value(argv, key[4:])
            if v is None:
                raise KeyError("the line passes no %s" % key[4:])
            return v
        return ctx[key]
    return re.sub(r"\{([^{}]+)\}", sub, tmpl)


def resolve(line, registry=None):
    """The scorer spec for this line, or None when its runner has no registered checker."""
    reg = registry if registry is not None else load_registry()
    line = bare_line(line)
    mod = line_module(line)
    entry = reg.get(mod)
    if not entry or not entry.get("checker"):
        return None
    chk = entry["checker"]
    argv = _argv_after_module(line, mod)
    ctx = {"python": line_python(line, mod), "module": mod}
    cmd = [_fill(t, ctx, argv) for t in chk["argv"]]
    for flag, extra in (chk.get("add_if_flag") or {}).items():
        if has_flag(argv, flag):
            cmd += [_fill(t, ctx, argv) for t in extra]
    reqs = []
    for r in chk["require"]:
        when = r.get("when", "always")
        if when.startswith("flag:") and not has_flag(argv, when[5:]):
            continue
        reqs.append({"path": r["path"], "op": r["op"], "value": r["value"], "label": r.get("label", r["path"])})
    return {"module": mod, "argv": cmd, "env": chk.get("env", {}), "json_key": chk["json_key"], "require": reqs,
            "timeout_s": chk.get("timeout_s", 1800)}


def _get(obj, path):
    for p in path.split("."):
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    return obj


def _holds(x, op, v):
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    if op == "==":
        return abs(float(x) - float(v)) <= 1e-9
    if op == ">=":
        return float(x) >= float(v) - 1e-9
    if op == "<=":
        return float(x) <= float(v) + 1e-9
    raise ValueError("unknown op %r" % op)


def judge(spec, stdout):
    """Check the spec's requirements against the last JSON line of the scorer's stdout that holds json_key."""
    rec = None
    for ln in stdout.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            o = json.loads(ln)
        except ValueError:
            continue
        if isinstance(o, dict) and spec["json_key"] in o:
            rec = o
    if rec is None:
        return {"ok": False, "checks": [], "error": "the scorer printed no JSON line holding %r" % spec["json_key"]}
    checks = []
    for r in spec["require"]:
        x = _get(rec, r["path"])
        checks.append({"label": r["label"], "path": r["path"], "want": "%s %s" % (r["op"], r["value"]), "got": x,
                       "ok": _holds(x, r["op"], r["value"])})
    if not checks:
        return {"ok": False, "checks": [], "error": "the spec has no requirement to check"}
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "error": None}


def node_run(spec):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    try:
        p = subprocess.run(spec["argv"], env=env, capture_output=True, text=True, timeout=spec.get("timeout_s", 1800))
    except Exception as e:  # noqa: BLE001 -- any failure to run the scorer is a FAIL, reported
        return {"ok": False, "checks": [], "error": "the scorer did not run: %s" % str(e)[:200]}
    res = judge(spec, p.stdout)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
        res = {"ok": False, "checks": res["checks"],
               "error": "the scorer exited %d: %s" % (p.returncode, tail[0][:200])}
    return res


def summary(res):
    """One line for a canary record: pre=ok(...) / pre=FAIL(...)."""
    parts = ["%s=%s" % (c["path"].split(".")[-1], c["got"]) for c in res.get("checks", [])]
    if res.get("ok"):
        return "pre=ok(%s)" % " ".join(parts)
    bad = ["%s=%s want %s" % (c["path"].split(".")[-1], c["got"], c["want"]) for c in res.get("checks", [])
           if not c["ok"]]
    return "pre=FAIL(%s)" % ("; ".join(bad) or res.get("error") or "unknown")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "check-line":
        miss = missing_flags(a[1])
        if miss:
            print("; ".join("the line omits %s, which its instrument requires (%s)" % m for m in miss))
            return 1
        return 0
    if len(a) == 2 and a[0] == "resolve":
        try:
            spec = resolve(a[1])
        except KeyError as e:
            print("cannot resolve the precondition scorer: %s" % e.args[0], file=sys.stderr)
            return 1
        if spec is not None:
            print(json.dumps(spec))
        return 0
    if len(a) == 2 and a[0] == "node-run":
        res = node_run(json.loads(base64.b64decode(a[1]).decode()))
        res["summary"] = summary(res)
        print(json.dumps(res))
        return 0 if res["ok"] else 1
    print("usage: canary_preconditions.py check-line|resolve '<line>' | node-run <b64 spec>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
