"""CLASS SMK -- a FLIP preregistration filed with no dev-seed smoke behind it.

EVIDENCE (2026-10-17/18). The word-role flip's reply regression ("what does the cat eat" abstains with the flag on, 4
of 6 gate seeds, flag-off answered at dev seed 7) was found only after the full six-seed gate plus an extra amendment
(Amendment G, FAILURE_LOG 2026-10-17); a two-seed dev smoke of flag-on vs flag-off scored with `tools/answer_rate.py
paired` shows it in about an hour. RULING smoke-first-1018 (owner-approved 2026-10-18) made the smoke the rule; this
gate makes the rule mechanical, as the same project did for the bar (CLASS BAR) and the order (CLASS PR / PRA).

THE GATE. A commit that ADDS, MODIFIES or RENAMES a pre-registration (`research/findings/` or `docs/plans/`, file name
containing `prereg`) BLOCKS when the file is
  * NEW (first added on or after the cutoff, 2026-10-18) and declares neither `flip: yes` nor `flip: no <reason>`; or
  * a FLIP prereg (declared, or read as one from its name / flag table) that is NEW, or that BECAME a flip after the
    cutoff (edited in place or renamed into one), without valid `smoke-record:` lines: at least two committed (or
    staged-in-this-commit) JSONs written by `tools/answer_rate.py paired ... --seeds 7,8 --json`, dev seeds 7 and 8, no
    gate seed, none UNMEASURED, counts and status agreeing, evidence hashes matching, and no FLIP-HURTS-ANSWERS, unless
    the prereg also cites `smoke-hurts-ruling: <KEY>`, a Claude-signed RULING of section 6 of the runbook as MAIN holds
    it that cites this prereg (or the record). `smoke-exempt-ruling: <KEY>` (same test) replaces the records for a flip
    with no reply surface.
What a flip prereg IS, the record's rules and the limits of both are in the shared validator, tools/smoke_first.py
(its docstring is the specification; gate_preflight.py runs the same code when a gate-seed line is queued).

SCOPE. An OLD prereg (existed before the cutoff, already a flip then) is grandfathered whatever is appended to it; one
that was no flip then and is now is judged. The registry passes only added files, so (like CLASS BAR and CLASS PRA) this
gate reads `git diff --cached` itself (renames on); a clean `git merge` is not checked (each side's commits were judged
when made). The runbook is read from main, never from the index, so a commit cannot carry its own ruling.

WHAT IT CANNOT CATCH. See tools/smoke_first.py: a self-consistent invented record, a smoke on the wrong probes, a
forged committer date together with an undated file name.
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
    """`paths` (the registry's --diff-filter=A list) is not used: the changed set is read per parent, like CLASS BAR."""
    pra, sf = _pra(), _sf()
    root = os.path.abspath(root or _ROOT)
    try:
        env = pra._git_env(root)
        kind = invocation if invocation is not None else pra._detect_invocation(root)
        parents = pra._parents(root, env, kind)
        if not parents:
            return []                    # unborn branch / amended root commit; None: a clean auto-merge, not checked
        ch = [pra._changes(root, env, p, None, _SPECS, True) for p in parents]   # renames on: a moved old prereg is a R
        cand = [p for p in sorted(set(ch[0]).intersection(*ch[1:]))
                if sf.PREREG_RE.match(p) and all(c[p][0] != "D" for c in ch)]
        if not cand:
            return []

        def read_blob(rel):
            r = pra._git(["cat-file", "blob", ":%s" % rel], root, env, ok_codes=(0, 128))
            return r.stdout if r.returncode == 0 else None

        runbook = sf.read_runbook(root)                       # main's, never the index's or the working tree's
        probs = []
        for p in cand:
            blob = read_blob(p)
            st, old = ch[0][p]
            added = st == "A"
            probs += sf.decide(root, parents[0], p, (blob or b"").decode("utf-8", "replace"), read_blob, runbook,
                               force_new=added, history_rev=parents[0], history_rel=old)[2]
        return probs
    except pra._GitReadError as e:
        return ["CLASS SMK could not read the commit being made (%s) -- failing CLOSED. Retry; if git itself is "
                "broken, fix that first." % e]


# --- selftest ---------------------------------------------------------------------------------------------------
def _side(answered, n):
    return {"n": n, "answered": answered, "abstained": n - answered, "empty": 0, "text_only": 0,
            "answer_rate": (answered / n) if n else None}


def _rec(tag="a", seeds=(7, 8), hurt=(), mode="paired", unmeasured=(), faculty="fac-x"):
    """A record shaped as `tools/answer_rate.py paired --json` writes it (what the validator demands)."""
    import hashlib
    per, ev = {}, {}
    for s in seeds:
        n = 0 if s in unmeasured else 4
        on = _side(2 if s in hurt else 4, n)
        off = _side(4, 4)
        per[str(s)] = {"on": on, "off": off,
                       "status": "UNMEASURED" if s in unmeasured else ("FLIP-HURTS-ANSWERS" if s in hurt else "ok")}
        ev[str(s)] = {side: [{"file": "raw/%s/s%d/%s%d.json" % (tag, s, side, i),
                              "sha256": hashlib.sha256(("%s%d%s%d" % (tag, s, side, i)).encode()).hexdigest()}
                             for i in range(per[str(s)][side]["n"])] for side in ("on", "off")}
    rec = {"check": "answer_rate", "label": "%s: flip-%s vs off-%s" % (faculty, tag, tag), "per_seed": per,
           "n_seeds": len(per), "flags": ["FLIP-HURTS-ANSWERS"] if hurt else [], "faculty": faculty,
           "on_tag": "flip-" + tag, "off_tag": "off-" + tag, "evidence": ev, "script": "tools/answer_rate.py",
           "argv": ["paired"], "backend": "numpy", "device": "cpu"}
    if mode:
        rec["mode"] = mode
    return rec


_RUNBOOK = ("# runbook\n\n## 5. Idle\n\n- notakey: not a ruling (Claude)\n\n## 6. RULINGS (x)\n\n"
            "- real-ruling-1018: decided about p.md (Claude 2026-10-18)\n"
            "- unsigned-1018: about p.md (Hermes)\n- other-1018: signed but about another file (Claude)\n")
_FLIP = "---\nflip: yes\n---\n# a flip prereg for fac-x\n"
_TWO = "smoke-record: docs/ok.json\nsmoke-record: docs/ok2.json\n"


def _pure(bad):
    sf = _sf()
    blobs = {"docs/ok.json": json.dumps(_rec("a")).encode(), "docs/ok2.json": json.dumps(_rec("b")).encode(),
             "docs/gate.json": json.dumps(_rec("g", seeds=(42, 43))).encode(),
             "docs/one.json": json.dumps(_rec("o", seeds=(7,))).encode(),
             "docs/hurts.json": json.dumps(_rec("h", hurt=(8,))).encode(),
             "docs/prefixed.json": json.dumps(_rec("p", mode="prefixed")).encode(),
             "docs/nomode.json": json.dumps(_rec("n", mode=None)).encode(),
             "docs/unmeasured.json": json.dumps(_rec("u", seeds=(7, 8, 9), unmeasured=(9,))).encode(),
             "docs/other.json": json.dumps(_rec("c", faculty="another-flip")).encode(),
             "docs/self.json": json.dumps(dict(_rec("s"), off_tag="flip-s")).encode(),
             "docs/typed.json": json.dumps({"check": "answer_rate", "mode": "paired", "flags": [], "per_seed": {
                 "7": {"on": _side(4, 4), "off": _side(4, 4), "status": "ok"},
                 "8": {"on": _side(4, 4), "off": _side(4, 4), "status": "ok"}}}).encode(),
             "docs/list.json": json.dumps(dict(_rec("l"), per_seed={"7": [1], "8": "x"})).encode()}
    cases = [
        ("no smoke-record line", _FLIP, True),
        ("two valid records", _FLIP + _TWO, False),
        ("bullet and backticks", _FLIP + "- **smoke-record:** `docs/ok.json` faculty=fac-x\n" + "smoke-record: docs/ok2.json\n", False),
        ("ONE record only", _FLIP + "smoke-record: docs/ok.json\n", True),
        ("the same record twice", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/ok.json\n", True),
        ("record not committed", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/missing.json\n", True),
        ("gate-seed record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/gate.json\n", True),
        ("single-seed record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/one.json\n", True),
        ("prefixed-mode record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/prefixed.json\n", True),
        ("record without mode", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/nomode.json\n", True),
        ("unmeasured seed in record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/unmeasured.json\n", True),
        ("a hand-typed minimal record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/typed.json\n", True),
        ("on_tag == off_tag", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/self.json\n", True),
        ("malformed rows (must not raise)", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/list.json\n", True),
        ("another flip's record", _FLIP + "smoke-record: docs/ok.json\nsmoke-record: docs/other.json\n", True),
        ("HURTS, no ruling", _FLIP + _TWO.replace("ok2", "hurts"), True),
        ("HURTS, real ruling about this file", _FLIP + _TWO.replace("ok2", "hurts") + "smoke-hurts-ruling: real-ruling-1018\n", False),
        ("HURTS, unsigned ruling", _FLIP + _TWO.replace("ok2", "hurts") + "smoke-hurts-ruling: unsigned-1018\n", True),
        ("HURTS, ruling about another file", _FLIP + _TWO.replace("ok2", "hurts") + "smoke-hurts-ruling: other-1018\n", True),
        ("HURTS, wrong section", _FLIP + _TWO.replace("ok2", "hurts") + "smoke-hurts-ruling: notakey\n", True),
        ("HURTS, invented key", _FLIP + _TWO.replace("ok2", "hurts") + "smoke-hurts-ruling: made-up\n", True),
        ("exempt, real ruling", _FLIP + "smoke-exempt-ruling: real-ruling-1018\n", False),
        ("exempt, invented key", _FLIP + "smoke-exempt-ruling: made-up\n", True),
        ("absolute record path", _FLIP + "smoke-record: /etc/passwd\nsmoke-record: docs/ok.json\n", True),
        ("escaping record path", _FLIP + "smoke-record: ../ok.json\nsmoke-record: docs/ok.json\n", True),
    ]
    for label, text, should_fail in cases:
        try:
            got = bool(sf.judge_prereg("p.md", text, blobs.get, _RUNBOOK))
        except Exception as e:                                   # noqa: BLE001 -- a raise is a selftest failure
            bad.append("judge_prereg case %r RAISED %s: %s" % (label, type(e).__name__, e))
            continue
        if got != should_fail:
            bad.append("judge_prereg case %r: expected %s, got %s" % (label, "a problem" if should_fail else "none",
                                                                      "a problem" if got else "none"))
    hand = _rec(hurt=(8,))
    hand["per_seed"]["8"]["status"] = "ok"
    hand["flags"] = []
    if not sf.judge_record("r.json", json.dumps(hand).encode())[0]:
        bad.append("did NOT refuse a hand-edited record (counts show a hurt, status says ok)")
    detect = [
        ("2026-10-18-x-flip-PREREGISTRATION.md", "# t\n", "flip"),
        ("2026-10-18-x-flips-PREREGISTRATION.md", "# t\n", "flip"),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: yes\n", "flip"),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: BRAIN_X, BRAIN_Y\n", "flip"),
        ("2026-10-18-x-flip-PREREGISTRATION.md", "flip: no this registers a battery and flips no default at all\n", "flip"),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: no this registers a battery and flips no default at all\n", "nonflip"),
        ("2026-10-18-x-PREREGISTRATION.md", "flip: no\n", "undeclared"),
        ("2026-10-18-x-PREREGISTRATION.md", "# a battery\nvalidates the flips; becomes the production default\n", "undeclared"),
        ("2026-10-18-x-flip-criteria-AMENDMENT-PREREG.md", "# t\n", "undeclared"),
        ("2026-10-18-x-flip-criteria-AMENDMENT-PREREG.md", "flip: yes\n", "flip"),
        ("2026-10-18-x-PREREGISTRATION.md", "| flag | old default | new default | anchor |\n|---|---|---|---|\n", "flip"),
        ("2026-10-18-flipper-PREREGISTRATION.md", "# t\n", "undeclared"),
    ]
    for name, text, want in detect:
        if sf.classify(name, text)[0] != want:
            bad.append("classify %r on %r: expected %s, got %s" % (name, text[:30], want, sf.classify(name, text)[0]))
    if sf.ruling_keys(_RUNBOOK) != {"real-ruling-1018", "other-1018"}:
        bad.append("ruling_keys read %r from section 6 (signed entries only); expected the two signed ones"
                   % sorted(sf.ruling_keys(_RUNBOOK)))


def _repo(bad):
    """The wiring, in a scratch repo with real `git diff --cached`: a NEW flip prereg without the records blocks; a silent
    new prereg blocks; a non-flip prereg and an old (already committed) flip prereg are ignored, an old one edited INTO a
    flip is judged; with two committed records it passes; a ruling written in the commit itself does not count."""
    import subprocess
    import tempfile
    pra = _pra()
    sf = _sf()
    with tempfile.TemporaryDirectory() as td:
        env0 = pra._stripped_env()

        def g(*a, date=None):
            env = dict(env0)
            if date:
                env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = date + "T12:00:00+00:00"
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a),
                               cwd=td, env=env, capture_output=True, text=True, timeout=30)
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
        oldplain = "research/findings/2026-10-01-old-plain-PREREGISTRATION.md"
        new = "research/findings/2026-10-19-new-flip-PREREGISTRATION.md"
        w(old, "# an old flip prereg, before the gate\n")
        w(oldplain, "# an old prereg that flips nothing, before the gate\n")
        w(sf.RUNBOOK_REL, _RUNBOOK)
        g("add", "-A")
        g("commit", "-q", "-m", "base", date="2026-10-01")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a clean tree (grandfathered flip prereg) was flagged")
        w(old, "# an old flip prereg\n\n## AMENDMENT 1\nmore\n")
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a MODIFIED old flip prereg was blocked (grandfathered)")
        g("reset", "-q")
        w(old, "# an old flip prereg, before the gate\n")
        w(oldplain, "# an old prereg\n\nflip: yes\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT judge an OLD prereg edited in place into a flip")
        g("reset", "-q")
        w(oldplain, "# an old prereg that flips nothing, before the gate\n")
        w("research/findings/2026-10-19-battery-PREREGISTRATION.md", "# a battery\nflip: no it registers a battery only\n")
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a declared non-flip prereg was blocked")
        w("research/findings/2026-10-19-silent-PREREGISTRATION.md", "# silent\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block a NEW prereg that declares nothing")
        g("reset", "-q")
        os.remove(os.path.join(td, "research/findings/2026-10-19-silent-PREREGISTRATION.md"))
        os.remove(os.path.join(td, "research/findings/2026-10-19-battery-PREREGISTRATION.md"))
        w(new, "# new flip prereg for fac-x\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg with no smoke-record line")
        recs = "smoke-record: research/findings/smoke/a.json\nsmoke-record: research/findings/smoke/b.json\n"
        w(new, "# new flip prereg for fac-x\n" + recs)
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg whose smoke records are not committed")
        w("research/findings/smoke/a.json", json.dumps(_rec("a", seeds=(42, 43))))
        w("research/findings/smoke/b.json", json.dumps(_rec("b")))
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED flip prereg whose record is on gate seeds")
        w("research/findings/smoke/a.json", json.dumps(_rec("a")))
        g("add", "-A")
        probs = check([], root=td, invocation="commit")
        if probs:
            bad.append("FALSE POSITIVE: a flip prereg with two staged valid records was blocked: %s" % probs)
        w("research/findings/smoke/a.json", json.dumps(_rec("a", hurt=(7,))))
        w(sf.RUNBOOK_REL, _RUNBOOK + "- 2026-10-19 RULING self-made (Claude): fine, 2026-10-19-new-flip-PREREGISTRATION.md\n")
        w(new, "# new flip prereg for fac-x\n" + recs + "smoke-hurts-ruling: self-made\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("accepted a HURTS ruling written into the runbook IN THE SAME COMMIT (main must hold it)")
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
