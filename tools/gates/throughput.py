"""Failure class 7 — LIVENESS MISTAKEN FOR PROGRESS (4 incidents).

THE EVIDENCE. `research/findings/2026-07-30-CRITICAL-crux-throughput-14x-over-estimate-...md`: one arm of a
15-arm job printed `(81444s)` = 22.6 h against a pre-registered "~8-24 h for the WHOLE job" — a ~14x
over-estimate, projecting ~339 h. Liveness was checked every ~15 min for a whole session and "~20 crux-healthy
reports" were filed; **throughput was never checked once**. A second incident burned 47 min of GPU-priority time
running on CPU while `cpu-time/elapsed` read 99%, so every liveness signal looked perfect. The finding's own
closing rule: *"If the job emits no per-unit marker, that is itself a defect."*

WHAT THIS GATE IS. The failure happens at RUNTIME and a pre-commit hook cannot watch a process, so this is
deliberately an **honest reporting gate, BLOCKING=False**. It does not prevent the failure; it refuses to let
the *record* of a run be silent about its cost — the one statically checkable part, and the part that would
have made the 14x visible the moment the first arm landed. Two narrow checks on `research/findings/*.md`:

  P1  a stated runtime ESTIMATE (ETA / expected runtime / projected wall) with NO measured figure anywhere in
      the document. The incident verbatim: the estimate was recorded, reality never was.
  P2  a COMPLETED multi-arm run — a job size (`5 arms x 3 seeds`) plus a citation of its own
      `research/findings/raw/` artifacts, so it demonstrably ran — with NO duration figure of any kind.

CALIBRATION (measured, not asserted). Over all 1841 findings: P1 fires on 8, P2 on 4 — 0.65%, every one a
genuine silent-cost record. Both are suppressed by ANY measured evidence (`elapsed`, `wall-clock`, `h/arm`,
`(81444s)`, `took ~3 h`, "measured ... 40 min"), so the escape hatch is one honest sentence. Narrow beat broad
on purpose: a draft that also demanded a duration key in every `--out` runner fired on 517 of 690 (75%) — a
cry-wolf generator, and it is not built.

WHAT IT CANNOT CATCH — plainly, this is most of the class:
  · the actual failure. A live, slow, 14x-over-budget run commits nothing, stages nothing, trips no gate. Only
    a state-checking heartbeat that divides progress by elapsed catches that.
  · an arm on CPU while `cpu/elapsed` reads 99% — device placement is a runtime fact, invisible in prose.
  · a WRONG measured number. It checks that a cost was recorded, never that the recording is true.
  · a decoratively quoted duration ("the 3 h session") — that suppresses both checks.
  · `check([])` scans only findings dated within 30 days of the newest, so the 11 legacy pre-July hits report
    only when explicitly staged. Deliberate: a warn reprinting the same legacy list every commit is noise.
"""
from __future__ import annotations

import datetime
import os
import re

NAME = "throughput"
CLASS_ID = "7"
BLOCKING = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WINDOW_DAYS = 30

_U = r"(?:s|sec|secs|seconds|min|mins|minutes|h|hr|hrs|hour|hours)"   # duration units
_C = r"(?:arms?|seeds?|runs?|jobs?|configs?|conditions?)"            # job-size countables

# `ETA` is case-SENSITIVE on purpose: lowercase `eta` is a Gabor parameter and sits inside "meta-learning" —
# both produced false positives during calibration.
_EST = re.compile(r"(?:\bETA\b|(?:[Ee]stimated?|[Pp]rojected|[Ee]xpected)\s+(?:total\s+)?"
                  r"(?:runtime|run.time|wall|wall.clock|GPU.time|compute|job\s+time|duration))")
_DUR = re.compile(r"(?i)\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?\s*)?(?:" + _U + r"|d|day|days)\b")
# MEASURED cost. Any one of these suppresses both checks — require evidence, not a format.
_MEAS = re.compile(r"(?i)(?:elapsed|wall.?clock"
                   r"|\b\d+(?:\.\d+)?\s*" + _U + r"\s*/\s*(?:arm|seed|run|job|epoch|step|trial|cycle|sample)"
                   r"|\(\s*\d{3,}\s*s\s*\)|\btook\s+~?\s*\d"
                   r"|\b(?:measured|actual|observed|real)\b[^\n]{0,40}?\d+(?:\.\d+)?\s*" + _U +
                   r"|\bper[- ](?:arm|seed|run|job|step)\s+(?:cost|time))")
_SIZE = re.compile(r"(?i)\b\d+\s*" + _C + r"\s*[x×*]\s*\d+\s*" + _C)
_RAW = re.compile(r"research/findings/raw/")
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DEAD = re.compile(r"(?mi)^status:\s*(?:retracted|superseded)\s*$")


def _is_finding(path: str) -> bool:
    p = "/" + path.replace("\\", "/")
    return p.endswith(".md") and "/findings/" in p and "/findings/raw/" not in p


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _file_date(path):
    m = _DATE.search(os.path.basename(path))
    try:
        return datetime.date(*(int(g) for g in m.groups())) if m else None
    except ValueError:                       # a 2026-13-45 typo in a filename — not a date, not our class
        return None


def _problems_for(path: str, text: str) -> list:
    if _DEAD.search(text[:600]):             # a retracted/superseded doc is not worth nagging about
        return []
    if _MEAS.search(text):                   # a cost WAS recorded somewhere — that is all this gate asks
        return []
    rel = os.path.relpath(_abs(path), _ROOT) if os.path.isabs(path) else path
    rel = path if rel.startswith("..") else rel      # outside the repo (a fixture) — show it as given
    out = []
    for m in _EST.finditer(text):
        if _DUR.search(text[m.end():m.end() + 60]):
            out.append("%s: states a runtime estimate (%r) but records NO measured cost — class 7: the 14x "
                       "over-estimate stayed invisible because only the estimate was ever written down. Add "
                       "the measured elapsed / per-unit figure."
                       % (rel, " ".join(text[m.start():m.end() + 40].split())))
            break
    size = _SIZE.search(text)
    if size and _RAW.search(text) and not _DUR.search(text):
        out.append("%s: reports a completed multi-arm run (%r, cites its raw/ artifacts) with NO elapsed or "
                   "throughput figure — the next estimate for this lane will have no measured basis."
                   % (rel, " ".join(size.group(0).split())))
    return out


def _recent_findings() -> list:
    d = os.path.join(_ROOT, "research", "findings")
    if not os.path.isdir(d):
        return []
    dated = [(os.path.join(d, f), _file_date(f)) for f in os.listdir(d) if f.endswith(".md")]
    dated = [(f, dt) for f, dt in dated if dt is not None]
    if not dated:
        return []
    newest = max(dt for _, dt in dated)
    return sorted(f for f, dt in dated if (newest - dt).days <= _WINDOW_DAYS)


def check(paths) -> list:
    targets = [p for p in (paths or []) if _is_finding(p)]
    if not paths:
        targets = _recent_findings()
    problems = []
    for p in targets:
        ap = _abs(p)
        if not os.path.isfile(ap):           # staged-but-deleted; nothing to read, not a class-7 problem
            continue
        with open(ap, encoding="utf-8", errors="replace") as fh:
            problems += _problems_for(p, fh.read())
    return problems


def selftest() -> list:
    """FAILING DIRECTION FIRST: build docs the gate MUST flag, and fail if it does not."""
    import tempfile

    est = "# run\n\nLaunched the crux. **ETA ~8-24 h** for the whole job.\n\nAll arms healthy.\n"
    run = "# run\n\nRan 5 arms x 3 seeds.\n\nRaw: `research/findings/raw/gap4/seed42.json`\n"
    # (name, body, must_be_caught, why)
    cases = [("est.md", est, True, "a runtime estimate with no measured cost"),
             ("run.md", run, True, "a completed multi-arm run with no elapsed figure"),
             ("est-ok.md", est + "\nFirst arm landed: elapsed 81444s = 22.6 h/arm.\n", False, ""),
             ("run-ok.md", run + "\nMeasured 22.6 h/arm over the 15 arms.\n", False, ""),
             ("clean.md", "# prose\n\nNo run, no estimate, no job size.\n", False, ""),
             ("dead.md", "---\nstatus: retracted\n---\n" + est, False, "")]
    fails = []
    with tempfile.TemporaryDirectory() as td:
        fdir = os.path.join(td, "research", "findings")
        os.makedirs(fdir)
        for name, body, must_catch, why in cases:
            p = os.path.join(fdir, "2026-07-30-" + name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            got = check([p])
            if must_catch and not got:
                fails.append("gate did NOT catch %s (%s) — it cannot fail, i.e. failure class 3" % (name, why))
            elif got and not must_catch:
                fails.append("FALSE POSITIVE on %s: %s" % (name, got[0][:110]))
        src = os.path.join(td, "runner.py")   # a non-finding path must be ignored, or the gate leaks
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("# ETA ~8-24 h\n")
        if check([src]):
            fails.append("gate fired on a non-finding path (%s) — scope leak" % src)
    return fails
