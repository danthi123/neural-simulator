"""bar-sanity: can a PERFECT brain pass this pre-registered bar, and can a NULL brain fail it?

WHY (2026-10-06). Two registered bars cost about two days each in one week because nobody asked, before the first
run, whether the bar could be passed at all:
  * T1 long-delay credit, engagement bar (a): the LHb rate ratio over a 250 ms outcome window, with the reward
    delivered for 50 ms, can never exceed 250/200 = 1.25 however perfectly the reward silences the LHb, against a
    bar of "> 2". No brain passes it (docs/drafts/long-delay-credit/AMENDMENT-A0.1.md).
  * Stage-1 C4: "scrambled-map r < 0.20 on all 6 seeds" fails a brain that follows its learned values exactly,
    because a random permutation of 24 values can itself correlate with the human ratings
    (research/findings/2026-10-04-stage1-exit-gate-RESULT.md, "Claude review 2026-10-06").
Both are a bar that a perfect brain fails (UNREACHABLE). The mirror defect is a bar that a null brain passes
(TOOTHLESS). A bar is only worth registering when the perfect brain passes it >= 95% of the time and the null brain
passes <= 5% of the time. Run it BEFORE the bar is registered.

HOW. A packet writes a spec module (convention: docs/drafts/<packet>/bar_sanity_spec*.py, or any path) defining
    decide(data) -> bool          calls the packet's REAL scorer / decision function (never a reimplementation of
                                  the bar; thin glue is fine, and a decision inlined in a big function should be
                                  extracted into an importable function, with a test that it reads identically)
    perfect(rng)  -> data         synthetic data in the scorer's input format for a brain that does the thing
                                  perfectly (realistic noise is allowed and encouraged: the ideal brain is not the
                                  only brain the bar must admit)
    null(rng)     -> data         a brain that does NOT do the thing
    random(rng)   -> data         (optional) a brain that responds at random
    perfect_<x>, null_<x>         (optional) more of either kind; every one is thresholded like its kind
    info_<x>                      (optional) reported, never thresholded (a weak-effect brain, an exotic profile)
    SCORER_FILES = [..]           (optional) repo-relative files that hold the scorer; their sha256 is recorded and
                                  the gate refuses a record whose scorer changed since. Default: the spec file.
    NAME / DESCRIPTION            (optional) strings
    OVERRIDES = {"perfect_min": .., "null_max": .., "reason": "<>= 20 chars>"}   (optional; a looser threshold
                                  needs a reason, a tighter one does not)
`rng` is a numpy Generator seeded per (seed, generator); draws are sequential, so a record reproduces exactly.

    python tools/bar_sanity.py docs/drafts/<packet>/bar_sanity_spec.py [--n 1000] [--seed 0] [--out record.json]

writes a JSON record {spec, scorer sha256, n, pass_rate per generator, thresholds, verdict} and exits 0 only on OK.
Verdicts: OK; UNREACHABLE (a perfect brain passes < perfect_min); TOOTHLESS (a null brain passes > null_max). If both
hold the verdict is UNREACHABLE and `failures` lists both. `tools/gates/bar_sanity_line.py` requires a prereg or an
amendment to cite an OK record (or `bar-sanity: N/A <reason>`).

WHAT THIS CANNOT DO. It tests a bar against the brains the spec author wrote down. A perfect brain written to pass
proves nothing, and a null brain that is too easy proves nothing; the spec is read in review like the bar itself.
It does not say the bar is the right bar, only that it is neither impossible nor free.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERFECT_MIN = 0.95
NULL_MAX = 0.05
MIN_REASON_CHARS = 20
KIND_PREFIXES = (("perfect", "perfect"), ("null", "null"), ("random", "null"), ("info", "info"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def wilson(k, n, z=1.96):
    """95% Wilson interval for k passes in n draws (reported next to every rate; the verdict uses the point rate)."""
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4)]


def verdict_from(pass_rate, kinds, perfect_min, null_max):
    """Pure. ('OK'|'UNREACHABLE'|'TOOTHLESS', [failure strings]). A spec with no perfect generator, or no null one,
    is not a sanity check at all and reads INVALID."""
    fails, perf, nul = [], [g for g, k in kinds.items() if k == "perfect"], [g for g, k in kinds.items() if k == "null"]
    if not perf or not nul:
        return "INVALID", ["the spec needs at least one perfect generator and one null generator"]
    unreachable = [g for g in perf if pass_rate[g] < perfect_min]
    toothless = [g for g in nul if pass_rate[g] > null_max]
    for g in unreachable:
        fails.append("UNREACHABLE: %s passes %.3f < %.2f" % (g, pass_rate[g], perfect_min))
    for g in toothless:
        fails.append("TOOTHLESS: %s passes %.3f > %.2f" % (g, pass_rate[g], null_max))
    if unreachable:
        return "UNREACHABLE", fails
    if toothless:
        return "TOOTHLESS", fails
    return "OK", fails


def load_spec(path):
    path = os.path.abspath(path)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)                 # specs import research.* / tools.* the way the runners do
    name = "bar_sanity_spec_%s" % sha256_bytes(path.encode())[:10]
    s = importlib.util.spec_from_file_location(name, path)
    if s is None or s.loader is None:
        raise ImportError("cannot load a spec from %s" % path)
    mod = importlib.util.module_from_spec(s)
    sys.modules[name] = mod
    s.loader.exec_module(mod)
    return mod


def generators_of(mod):
    """{name: kind} for every callable the spec defines under a recognised name."""
    out = {}
    for attr in sorted(dir(mod)):
        if attr.startswith("_") or not callable(getattr(mod, attr)):
            continue
        for prefix, kind in KIND_PREFIXES:
            if attr == prefix or attr.startswith(prefix + "_"):
                out[attr] = kind
                break
    return out


def thresholds_of(mod):
    ov = dict(getattr(mod, "OVERRIDES", None) or {})
    pm, nm = float(ov.get("perfect_min", PERFECT_MIN)), float(ov.get("null_max", NULL_MAX))
    loosened = pm < PERFECT_MIN or nm > NULL_MAX
    reason = ov.get("reason")
    if loosened and not (isinstance(reason, str) and len(reason.strip()) >= MIN_REASON_CHARS):
        raise ValueError("OVERRIDES loosens a threshold (perfect_min %.2f, null_max %.2f) but gives no `reason` of at "
                         "least %d characters" % (pm, nm, MIN_REASON_CHARS))
    return {"perfect_min": pm, "null_max": nm, "override_reason": reason.strip() if loosened else None}


def _git_head(root):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _rel(path, root):
    p = os.path.relpath(os.path.abspath(path), root)
    return p.replace(os.sep, "/")


def run(spec_path, n=1000, seed=0, root=None):
    """Run the spec; return the record (a dict). Raises if the spec is malformed or `decide` raises."""
    root = os.path.abspath(root or _ROOT)
    mod = load_spec(spec_path)
    if not callable(getattr(mod, "decide", None)):
        raise ValueError("%s defines no decide(data)" % spec_path)
    kinds = generators_of(mod)
    th = thresholds_of(mod)
    t0 = time.time()
    pass_rate, n_pass = {}, {}
    for g in sorted(kinds):
        gen = getattr(mod, g)
        rng = np.random.default_rng([int(seed), zlib.crc32(g.encode())])
        k = 0
        for i in range(int(n)):
            data = gen(rng)
            try:
                k += 1 if bool(mod.decide(data)) else 0
            except Exception as e:                              # a crashing scorer is a spec bug, never a "fail"
                raise RuntimeError("decide() raised on %s draw %d: %s: %s" % (g, i, type(e).__name__, e)) from e
        n_pass[g] = k
        pass_rate[g] = round(k / float(n), 4)
    verdict, failures = verdict_from(pass_rate, kinds, th["perfect_min"], th["null_max"])
    files = [str(p) for p in (getattr(mod, "SCORER_FILES", None) or [_rel(spec_path, root)])]
    scorer = {}
    for f in files:
        full = os.path.join(root, f)
        if not os.path.isfile(full):
            raise FileNotFoundError("SCORER_FILES names %s, which does not exist under %s" % (f, root))
        scorer[f] = sha256_file(full)
    return {
        "tool": "tools/bar_sanity.py",
        "spec": _rel(spec_path, root),
        "spec_sha256": sha256_file(spec_path),
        "name": getattr(mod, "NAME", None),
        "description": getattr(mod, "DESCRIPTION", None),
        "scorer_files": scorer,
        "scorer_sha256": sha256_bytes(json.dumps(scorer, sort_keys=True).encode()),
        "n": int(n), "seed": int(seed),
        "kinds": {g: kinds[g] for g in sorted(kinds)},
        "pass_rate": pass_rate,
        "n_pass": n_pass,
        "pass_rate_ci95": {g: wilson(n_pass[g], int(n)) for g in sorted(kinds)},
        "thresholds": th,
        "verdict": verdict,
        "failures": failures,
        "backend": "numpy",                   # pure python + numpy on CPU; no brain is built (gates/device_and_cost reads this)
        "repo_head": _git_head(root),
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="path to the spec module (docs/drafts/<packet>/bar_sanity_spec.py)")
    ap.add_argument("--n", type=int, default=1000, help="draws per generator (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the JSON record here (default: print only)")
    a = ap.parse_args(argv)
    rec = run(a.spec, n=a.n, seed=a.seed)
    text = json.dumps(rec, indent=1, sort_keys=True) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    for g in sorted(rec["pass_rate"]):
        print("  %-34s %-8s pass %.3f  ci95 %s" % (g, rec["kinds"][g], rec["pass_rate"][g], rec["pass_rate_ci95"][g]))
    print("bar-sanity %s: %s" % (rec["verdict"], "; ".join(rec["failures"]) or "perfect >= %.2f, null <= %.2f" % (
        rec["thresholds"]["perfect_min"], rec["thresholds"]["null_max"])))
    return 0 if rec["verdict"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
