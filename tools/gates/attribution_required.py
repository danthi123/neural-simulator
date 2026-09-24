"""CLASS AT — A TREATMENT/CONTROL PAIR MEASURED, THE DIFFERENCE NEVER ATTRIBUTED.

THE DEFECT. A runner computes a treatment and its control, reports both, and never asks WHOSE the difference
was. The two numbers sit one key apart in the same JSON and nobody subtracts them. gap#5 is the worked example
and it cost weeks:

    treatment (lr on)  mean |dW| = 22.6098
    control   (lr = 0) mean |dW| = 21.9393        <- both banked, both correct, both in one file

`attributable_to("lr @ tuned point", 22.6098, 21.9393)` returns **0.0297**. The lever moved 3% of the measured
change; the other **97% was the CLAMP**, running identically in both arms. The tuning then walked DEEPER into
the trap (w_max 110 -> 150 -> 220, with 150 selected as "optimal") because clamp depth was exactly what the
metric rewarded. Note what this exposes about the control: `lr=0` holds the LEARNING fixed, it does not hold
the PROXY fixed. "The control is clean" was never the right reading — a control only bounds the terms it
actually varies.

WHY A GATE AND NOT A RULE. `tools/lab.py` exists precisely to make this one call, and it is imported by
**2 of 1330 runners**. The helpers are opt-in, so they are used by whoever already remembered the lesson —
which is the population that did not need them. This is the same shape as failure class 3: the machinery to
check the claim existed and nothing invoked it.

WHAT THIS GATE ENFORCES, narrowly. A file under `research/runners/` that computes a CONTROL-shaped quantity
(lesion / shuffle / permuted / null / baseline / no-credit / frozen / sham / scrambled) alongside a treatment
must make at least ONE attribution call from `tools.lab`: `attributable_to`, `term_budget`, `lever`,
`before_after` or `sign_budget`. Any one of them forces the question to be asked out loud.

WHAT IT CANNOT CATCH: whether the attribution is CORRECT, or whether the named terms are the right ones. A
control only bounds the terms it varies, and no static check knows which terms a mechanism has. That judgement
is `verify-go`'s. This gate only ensures the subtraction is not skipped entirely.

SCOPED TO NEWLY ADDED FILES. Applying it to the 1330-file legacy corpus would emit hundreds of hits and get
the gate switched off, which is strictly worse than no gate (the doc-type gate learned this the hard way when
a Tier-1 classification pulled 192 legacy findings into scope in one commit). Legacy runners are audited on
next touch.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "attribution-required"
CLASS_ID = "AT"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CONTROL-shaped names. Anchored to word boundaries so `nullable` / `baseline_cfg` do not trip it, and kept to
# terms this project actually uses for an experimental control (grepped from the runner corpus).
CONTROL_RE = re.compile(
    r"\b(?:[a-z0-9_]*_)?(?:lesion|lesioned|shuffle|shuffled|permuted|scrambled|sham|"
    r"no_credit|no_teaching|nocred|untrained|frozen_ctrl|ctrl|control_arm)(?:_[a-z0-9_]*)?\b", re.I)
# The treatment side: a second measured quantity the control is meant to be compared AGAINST.
TREATMENT_RE = re.compile(r"\b(?:treatment|arm|expanded|test_fixed|test_learned|on_mean|with_|full|intact)\b", re.I)
# Any one of these forces the question to be asked out loud.
ATTRIB_RE = re.compile(r"\b(?:attributable_to|term_budget|lever|before_after|sign_budget)\s*\(")
IMPORT_LAB_RE = re.compile(r"from\s+tools\.lab\s+import|import\s+tools\.lab")


def _strip_noncode(text):
    """Docstrings and comments discuss controls constantly — this gate is about what the code COMPUTES."""
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"#.*", "", text)


def _check_one(path, text=None):
    rel = os.path.relpath(path, _ROOT) if os.path.isabs(path) else path
    rel = rel.replace("\\", "/")
    if "research/runners/" not in rel or not rel.endswith(".py"):
        return []
    try:
        text = text if text is not None else open(path, errors="ignore").read()
    except OSError:
        return []
    code = _strip_noncode(text)
    controls = sorted({m.group(0) for m in CONTROL_RE.finditer(code)})
    if not controls or not TREATMENT_RE.search(code):
        return []                                        # no treatment/control pair computed here
    if ATTRIB_RE.search(code) and IMPORT_LAB_RE.search(text):
        return []
    return ["%s: computes a treatment/control pair (%s) but makes NO attribution call. Add one of "
            "attributable_to / term_budget / lever / before_after / sign_budget from tools.lab — measuring "
            "both arms is not the same as asking whose the difference was. gap#5 banked both numbers one key "
            "apart for weeks; the subtraction showed the lever owned 3%% of the change and the CLAMP owned 97%%."
            % (rel, ", ".join(controls[:4]) + ("..." if len(controls) > 4 else ""))]


def _regressed(rel, old_text, new_text):
    """REGRESSION MODE. A runner that SATISFIED this gate before a modification and FAILS it after has had its
    attribution call removed. Legacy files that already failed before are NOT flagged (no corpus flood; they are
    audited when written, as above) -- only a file that was clean and was made dirty.

    Earned 2026-09-23: the D3 affect-pool verify runner imported `lever` + `attributable_to` and called both on its
    edge-lesion pair (7b46d761e, clean). The v2 gate amendment (cd5288018) rewrote the arm and dropped both calls.
    The gate went red on the file -- but pre-commit only passes ADDED files (--diff-filter=A) and the runner was
    already tracked, so the commit, the push and a fix round all went through; an adversarial re-review caught it."""
    if _check_one(rel, old_text):
        return []                                        # already failing before: legacy, not a regression
    if not _check_one(rel, new_text):
        return []
    return ["%s: REGRESSION -- this MODIFICATION removed the file's tools.lab attribution call (it passed "
            "attribution-required before the change and fails it after). A rewrite of an arm must keep asking "
            "whose the difference was; restore lever / attributable_to / term_budget / before_after / "
            "sign_budget on the treatment/control pair." % rel]


def _git(args):
    import subprocess
    try:
        r = subprocess.run(["git"] + args, cwd=_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _staged_modification_regressions():
    """Staged MODIFIED runners (index vs HEAD) checked in regression mode. Empty outside a commit (nothing staged)."""
    out = _git(["diff", "--cached", "--name-only", "--diff-filter=M", "--", "research/runners/"])
    problems = []
    for rel in (out or "").split("\n"):
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        old, new = _git(["show", "HEAD:" + rel]), _git(["show", ":" + rel])
        if old is None or new is None:
            continue
        problems += _regressed(rel, old, new)
    return problems


def check(paths):
    # paths=None means "standalone run": the legacy corpus is audited on next touch, so nothing is checked.
    # Otherwise (staged mode) two things are checked: (1) staged MODIFIED runners in REGRESSION mode (a clean
    # file made dirty -- see `_regressed`); this runs even when no file was ADDED, because the pre-commit driver
    # passes only --diff-filter=A paths and a modification never appears in them; (2) the ADDED paths, as before.
    # An empty added-list is NOT a corpus fallback -- that is how doc-type fired 192 legacy hits in one commit.
    if paths is None:
        return []
    problems = _staged_modification_regressions()
    targets = [p for p in paths if p.endswith(".py")]
    for p in targets:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: build files the gate MUST catch, and fail if it does not."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "research", "runners")
        os.makedirs(run)
        # 1. THE DEFECT: a treatment and its lesion control, both measured, never attributed.
        p1 = os.path.join(run, "_x_derisk.py")
        open(p1, "w").write(
            "def main():\n"
            "    arm = train(full=True)\n"
            "    apical_lesion = train(full=False)\n"
            "    print(arm, apical_lesion)\n")
        if not _check_one(p1):
            bad.append("did NOT catch a treatment/control pair with no attribution call")
        # 2. NEGATIVE CONTROL — the same file WITH an attribution call must pass, else the gate is unpassable
        #    and gets disabled (a gate nobody can satisfy is a gate nobody keeps).
        p2 = os.path.join(run, "_y_derisk.py")
        open(p2, "w").write(
            "from tools.lab import attributable_to\n"
            "def main():\n"
            "    arm = train(full=True)\n"
            "    apical_lesion = train(full=False)\n"
            "    attributable_to('apical drive', arm, apical_lesion)\n")
        if _check_one(p2):
            bad.append("FALSE POSITIVE: flagged a file that DOES call attributable_to")
        # 3. NEGATIVE CONTROL — a control word appearing only in prose must not fire. Docstrings in this repo
        #    discuss lesions constantly; flagging those would bury the real hits.
        p3 = os.path.join(run, "_z_derisk.py")
        open(p3, "w").write('"""We ran an apical_lesion control in a prior arm."""\ndef main():\n    return 1\n')
        if _check_one(p3):
            bad.append("FALSE POSITIVE: flagged a control mentioned only in a docstring")
        # 4. NEGATIVE CONTROL — a file outside research/runners/ is out of scope.
        p4 = os.path.join(d, "notarunner.py")
        open(p4, "w").write("arm = 1\napical_lesion = 2\n")
        if _check_one(p4):
            bad.append("FALSE POSITIVE: flagged a file outside research/runners/")
        clean = open(p2).read()
        dirty = open(p1).read()
    # 5. REGRESSION MODE -- a modification that DROPS the attribution call from a clean runner MUST be caught.
    rel = "research/runners/_x_derisk.py"
    if not _regressed(rel, clean, dirty):
        bad.append("did NOT catch a modification that removed a runner's attribution call (regression mode)")
    # 6. NEGATIVE CONTROLS -- a legacy file that already failed is not a regression; a clean->clean edit passes.
    if _regressed(rel, dirty, dirty):
        bad.append("FALSE POSITIVE: flagged a legacy (already failing) runner as a regression")
    if _regressed(rel, clean, clean + "\n# edited\n"):
        bad.append("FALSE POSITIVE: flagged a clean->clean modification")
    return bad
