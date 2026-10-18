"""CLASS SMK -- a FLIP preregistration filed with no dev-seed smoke behind it.

EVIDENCE (2026-10-17/18). The word-role flip's reply regression ("what does the cat eat" abstains with the flag on, 4
of 6 gate seeds, flag-off answered at dev seed 7) was found only after the full six-seed gate plus an extra amendment
(Amendment G, FAILURE_LOG 2026-10-17); a two-seed dev smoke of flag-on vs flag-off scored with `tools/answer_rate.py
paired` shows it in about an hour. RULING smoke-first-1018 (owner-approved 2026-10-18) made the smoke the rule; this
gate makes the rule mechanical, as the same project did for the bar (CLASS BAR) and the order (CLASS PR / PRA).

THE GATE. A commit that ADDS a pre-registration (`research/findings/` or `docs/plans/`, file name containing `prereg`)
which is a FLIP prereg BLOCKS unless it carries `smoke-record: <path>` naming a COMMITTED (or staged-in-this-commit)
JSON written by `tools/answer_rate.py paired ... --seeds 7,8 --json`: two or more dev seeds (1-15, never a gate
seed), none UNMEASURED, counts and status agreeing, and no FLIP-HURTS-ANSWERS, unless the prereg also cites
`smoke-hurts-ruling: <KEY>` where KEY is a RULING in section 6 of research/coordination/LOCAL_LLM_RUNBOOK.md.
What a flip prereg IS, the record's rules and the limits of both are in the shared validator, tools/smoke_first.py
(its docstring is the specification; gate_preflight.py runs the same code when a gate-seed line is queued).

SCOPE. Only files git reports ADDED relative to EVERY parent of the commit being made are judged, so every existing
prereg is grandfathered and an amendment appended to one is not this gate's business (CLASS BAR reads those). The
registry passes only added files, so (like CLASS BAR and CLASS PRA) this gate reads `git diff --cached` itself; a clean
`git merge` is not checked (each side's commits were judged when made).

WHAT IT CANNOT CATCH. See tools/smoke_first.py: a smoke on the wrong probes, a flip prereg named and declared as
neither a flip nor with `flip:`, a ruling key picked for another decision.
"""
from __future__ import annotations

import json
import os

NAME = "flip_smoke_first"
CLASS_ID = "SMK"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SPECS = (":(icase)research/findings/*prereg*.md", ":(icase)docs/plans/*prereg*.md")


def _pra():
    from tools.gates import prereg_amendment_order as pra
    return pra


def _sf():
    from tools import smoke_first
    return smoke_first


def check(paths, root=None, invocation=None):
    """`paths` (the registry's --diff-filter=A list) is not used: the added set is read per parent, like CLASS BAR."""
    pra, sf = _pra(), _sf()
    root = os.path.abspath(root or _ROOT)
    try:
        env = pra._git_env(root)
        kind = invocation if invocation is not None else pra._detect_invocation(root)
        parents = pra._parents(root, env, kind)
        if not parents:
            return []                    # unborn branch / amended root commit; None: a clean auto-merge, not checked
        ch = [pra._changes(root, env, p, None, _SPECS, True) for p in parents]   # renames on: a moved old prereg is not ADDED
        cand = [p for p in sorted(set(ch[0]).intersection(*ch[1:]))
                if sf.PREREG_RE.match(p) and all(c[p][0] == "A" for c in ch)]
        if not cand:
            return []

        def read_blob(rel):
            r = pra._git(["cat-file", "blob", ":%s" % rel], root, env, ok_codes=(0, 128))
            return r.stdout if r.returncode == 0 else None

        rb = read_blob(sf.RUNBOOK_REL)
        runbook = rb.decode("utf-8", "replace") if rb is not None else sf.read_runbook(root)
        probs = []
        for p in cand:
            blob = read_blob(p)
            probs += sf.evaluate(p, (blob or b"").decode("utf-8", "replace"), read_blob, runbook)[2]
        return probs
    except pra._GitReadError as e:
        return ["CLASS SMK could not read the commit being made (%s) -- failing CLOSED. Retry; if git itself is "
                "broken, fix that first." % e]


# --- selftest ---------------------------------------------------------------------------------------------------
def _side(answered, n):
    return {"n": n, "answered": answered, "abstained": n - answered, "empty": 0, "text_only": 0,
            "answer_rate": (answered / n) if n else None}


def _rec(seeds=(7, 8), hurt=(), mode="paired", unmeasured=()):
    per = {}
    for s in seeds:
        if s in unmeasured:
            per[str(s)] = {"on": _side(0, 0), "off": _side(4, 4), "status": "UNMEASURED"}
        elif s in hurt:
            per[str(s)] = {"on": _side(2, 4), "off": _side(4, 4), "status": "FLIP-HURTS-ANSWERS"}
        else:
            per[str(s)] = {"on": _side(4, 4), "off": _side(4, 4), "status": "ok"}
    rec = {"check": "answer_rate", "label": "x", "per_seed": per, "n_seeds": len(per), "flags": (
        ["FLIP-HURTS-ANSWERS"] if hurt else [])}
    if mode:
        rec["mode"] = mode
    return rec


_RUNBOOK = "# runbook\n\n## 5. Idle\n\n- notakey: not a ruling\n\n## 6. RULINGS (x)\n\n- real-ruling-1018: decided (2026-10-18)\n"
_FLIP = "---\nflip: yes\n---\n# a flip prereg\n"


def _pure(bad):
    sf = _sf()
    blobs = {"docs/ok.json": json.dumps(_rec()).encode(),
             "docs/gate.json": json.dumps(_rec(seeds=(42, 43))).encode(),
             "docs/one.json": json.dumps(_rec(seeds=(7,))).encode(),
             "docs/hurts.json": json.dumps(_rec(hurt=(8,))).encode(),
             "docs/prefixed.json": json.dumps(_rec(mode="prefixed")).encode(),
             "docs/nomode.json": json.dumps(_rec(mode=None)).encode(),
             "docs/unmeasured.json": json.dumps(_rec(seeds=(7, 8, 9), unmeasured=(9,))).encode()}
    cases = [
        ("no smoke-record line", _FLIP, True),
        ("valid record", _FLIP + "smoke-record: docs/ok.json\n", False),
        ("bullet and backticks", _FLIP + "- **smoke-record:** `docs/ok.json`\n", False),
        ("record not committed", _FLIP + "smoke-record: docs/missing.json\n", True),
        ("gate-seed record", _FLIP + "smoke-record: docs/gate.json\n", True),
        ("single-seed record", _FLIP + "smoke-record: docs/one.json\n", True),
        ("prefixed-mode record", _FLIP + "smoke-record: docs/prefixed.json\n", True),
        ("record without mode", _FLIP + "smoke-record: docs/nomode.json\n", True),
        ("unmeasured seed in record", _FLIP + "smoke-record: docs/unmeasured.json\n", True),
        ("HURTS, no ruling", _FLIP + "smoke-record: docs/hurts.json\n", True),
        ("HURTS, real ruling", _FLIP + "smoke-record: docs/hurts.json\nsmoke-hurts-ruling: real-ruling-1018\n", False),
        ("HURTS, key not a ruling", _FLIP + "smoke-record: docs/hurts.json\nsmoke-hurts-ruling: notakey\n", True),
        ("HURTS, invented key", _FLIP + "smoke-record: docs/hurts.json\nsmoke-hurts-ruling: made-up\n", True),
        ("one good, one bad record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/hurts.json\n", True),
        ("two good records", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/ok.json\n", False),
        ("absolute record path", _FLIP + "smoke-record: /etc/passwd\n", True),
        ("escaping record path", _FLIP + "smoke-record: ../ok.json\n", True),
    ]
    for label, text, should_fail in cases:
        got = bool(sf.judge_prereg("p.md", text, blobs.get, _RUNBOOK))
        if got != should_fail:
            bad.append("judge_prereg case %r: expected %s, got %s" % (label, "a problem" if should_fail else "none",
                                                                      "a problem" if got else "none"))
    hand = _rec(hurt=(8,))
    hand["per_seed"]["8"]["status"] = "ok"
    hand["flags"] = []
    if not sf.judge_record("r.json", json.dumps(hand).encode())[0]:
        bad.append("did NOT refuse a hand-edited record (counts show a hurt, status says ok)")
    detect = [
        ("2026-10-18-x-flip-PREREGISTRATION.md", "# t\n", True),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: yes\n", True),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: BRAIN_X, BRAIN_Y\n", True),
        ("2026-10-18-x-flip-PREREGISTRATION.md", "flip: no\n", True),
        ("2026-10-18-x-flip-PREREGISTRATION.md", "flip: no this registers a battery and flips no default at all\n", False),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: no\n", False),
        ("2026-10-18-x-PREREGISTRATION.md", "# a battery\nvalidates the flips; becomes the production default\n", False),
        ("2026-10-18-x-flip-criteria-AMENDMENT-PREREG.md", "# t\n", False),
        ("2026-10-18-x-flip-criteria-AMENDMENT-PREREG.md", "flip: yes\n", True),
        ("2026-10-18-x-PREREGISTRATION.md", "| flag | old default | new default | anchor |\n|---|---|---|---|\n", True),
        ("2026-10-18-flipper-PREREGISTRATION.md", "# t\n", False),
    ]
    for name, text, want in detect:
        if sf.is_flip(name, text)[0] != want:
            bad.append("detector %r on %r: expected %s" % (name, text[:30], want))
    if sf.ruling_keys(_RUNBOOK) != {"real-ruling-1018"}:
        bad.append("ruling_keys read %r from section 6 only; expected {'real-ruling-1018'}" % sf.ruling_keys(_RUNBOOK))


def _repo(bad):
    """The wiring, in a scratch repo with real `git diff --cached`: an ADDED flip prereg without a record blocks; a
    non-flip prereg and an old (already committed) flip prereg are ignored; with a committed record it passes."""
    import subprocess
    import tempfile
    pra = _pra()
    sf = _sf()
    with tempfile.TemporaryDirectory() as td:
        env0 = pra._stripped_env()

        def g(*a):
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a),
                               cwd=td, env=env0, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError("selftest git %s: %s" % (a[:2], r.stderr.strip()[:120]))

        def w(rel, text):
            full = os.path.join(td, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)

        g("init", "-q", "-b", "main")
        g("config", "user.email", "gate-selftest@example.invalid")
        g("config", "user.name", "gate selftest")
        old = "research/findings/2026-10-01-old-flip-PREREGISTRATION.md"
        new = "research/findings/2026-10-19-new-flip-PREREGISTRATION.md"
        w(old, "# an old flip prereg, before the gate\n")
        w(sf.RUNBOOK_REL, _RUNBOOK)
        g("add", "-A")
        g("commit", "-q", "-m", "base")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a clean tree (grandfathered flip prereg) was flagged")
        w(old, "# an old flip prereg\n\n## AMENDMENT 1\nmore\n")
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a MODIFIED old flip prereg was blocked (only ADDED files are judged)")
        g("reset", "-q")
        w(old, "# an old flip prereg, before the gate\n")
        w("research/findings/2026-10-19-battery-PREREGISTRATION.md", "# a battery, flips nothing\n")
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a non-flip prereg was blocked")
        w(new, "# new flip prereg\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg with no smoke-record line")
        w(new, "# new flip prereg\nsmoke-record: research/findings/raw/smoke.json\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg whose smoke record is not committed")
        w("research/findings/raw/smoke.json", json.dumps(_rec(seeds=(42, 43))))
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg whose record is on gate seeds")
        w("research/findings/raw/smoke.json", json.dumps(_rec()))
        g("add", "-A")
        probs = check([], root=td, invocation="commit")
        if probs:
            bad.append("FALSE POSITIVE: a flip prereg with a staged valid record was blocked: %s" % probs)
        if check([], root=td, invocation="merge"):
            bad.append("a clean auto-merge must not be checked")


def selftest():
    """FAILING DIRECTION FIRST in each block, then the no-false-positive cases."""
    bad = []
    _pure(bad)
    try:
        _repo(bad)
    except (RuntimeError, OSError, _subprocess_error()) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


def _subprocess_error():
    import subprocess
    return subprocess.SubprocessError


if __name__ == "__main__":
    print("class SMK flip_smoke_first -- run via the registry (tools/gates); `python tools/smoke_first.py scan` lists "
          "the flip preregs of the tree.")
