"""CLASS KR — A KNOB THAT CHANGES THE SUBSTRATE BUT CANNOT BE SET FROM THE COMMAND LINE.

THE DEFECT, measured 2026-07-31, and it wasted a whole arc. `_onbridge_eprop_port_derisk.py` takes
constructor parameters `ou_noise=False, cond_noise=False` and writes them straight into the substrate:

    self.cfg.enable_ou_process       = bool(ou_noise)      # :171
    self.cfg.enable_conductance_noise = bool(cond_noise)   # :172

Independent background noise is not a detail — it is the difference between neurons that decorrelate and
neurons that are redundant copies, which is exactly what population coding depends on. And a same-day
finding NAMED enabling it as the biology-grounded fix (Destexhe-Rudolph high-conductance state).

**There is no `--ou-noise` flag. There is no `--cond-noise` flag. No runner in the tree passes either as
True.** So the fix the record prescribed was STRUCTURALLY UNRUNNABLE — you could not apply it without
editing source — and seventeen days later nobody could tell whether the banked K=8 closure had it, because
an unreachable knob is also an unrecorded one.

WHY THIS IS ITS OWN CLASS. `artifact_provenance` asks whether a run recorded *something*. `device_and_cost`
asks whether it recorded its device. Neither asks whether an experiment-changing parameter could be VARIED
AT ALL. A knob wired to `cfg` but not to argparse is invisible to every check here while silently fixing a
substrate property at its default — the "an absent flag means DEFAULT, not OFF" trap, one level deeper,
because here there is no flag whose default you could even inspect.

WHAT THIS GATE ENFORCES, on newly-added or newly-modified runners: any constructor parameter assigned into
`cfg.*` / `self.cfg.*` must have a matching `add_argument`. Substrate configuration is the experiment; if a
parameter reaches `cfg`, someone must be able to change it without a patch.

WHAT IT CANNOT CATCH: a knob that IS reachable and never recorded in the artifact, and a knob set from a
literal rather than a parameter (which is a fixed design choice, not a knob). The first is a candidate for
this gate's second half once runners uniformly write their config; the second is deliberately out of scope,
since not every constant deserves a flag.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "knob-reachable"
CLASS_ID = "KR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `self.cfg.enable_x = bool(param)` / `cfg.enable_x = param` -- capture the cfg key and the source identifier.
CFG_ASSIGN_RE = re.compile(
    r"^\s*(?:self\.)?cfg\.([A-Za-z_][\w]*)\s*=\s*(?:bool|int|float|str)?\(?\s*([A-Za-z_][\w]*)\s*\)?\s*$", re.M)
ADD_ARG_RE = re.compile(r"""add_argument\(\s*["']--([A-Za-z0-9][\w-]*)["']""")
DEST_RE = re.compile(r"""dest\s*=\s*["']([A-Za-z_]\w*)["']""")
# Identifiers that are not knobs: locals derived in-function, loop vars, and the obvious pass-throughs.
NOT_A_KNOB = {"self", "true", "false", "none", "cfg", "value", "v", "x"}


def _declared_flags(text):
    flags = {m.group(1).replace("-", "_") for m in ADD_ARG_RE.finditer(text)}
    flags |= {m.group(1) for m in DEST_RE.finditer(text)}
    return flags


def _ctor_params(text):
    """Parameters of any `def __init__(...)` — the population a cfg assignment can legitimately draw on."""
    params = set()
    for m in re.finditer(r"def\s+__init__\s*\(([^)]*)\)", text, re.S):
        for piece in m.group(1).split(","):
            name = piece.split("=")[0].split(":")[0].strip()
            if name and name.isidentifier() and name != "self":
                params.add(name)
    return params


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    if "research/runners/" not in rel or not rel.endswith(".py"):
        return []
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return []
    if "add_argument" not in text:
        return []                                          # a library, not a runnable experiment
    flags, ctor = _declared_flags(text), _ctor_params(text)
    problems = []
    seen = set()
    for m in CFG_ASSIGN_RE.finditer(text):
        cfg_key, src = m.group(1), m.group(2)
        if src.lower() in NOT_A_KNOB or src not in ctor or src in seen:
            continue
        if src in flags or cfg_key in flags:
            continue
        seen.add(src)
        problems.append(
            "%s: constructor parameter %r is written into `cfg.%s` but has NO --flag. A parameter that "
            "configures the SUBSTRATE and cannot be set from the command line is silently pinned at its "
            "default and invisible in every artifact. On 2026-07-31 `ou_noise`/`cond_noise` set "
            "`enable_ou_process`/`enable_conductance_noise` with no flags, so the biology-grounded fix a "
            "same-day finding PRESCRIBED was structurally unrunnable, and the banked closure's substrate "
            "config is unrecoverable." % (rel, src, cfg_key))
    return problems


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".py")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the real 2026-07-31 shape, then the controls that keep it usable."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "research", "runners")
        os.makedirs(run)

        def w(name, src):
            p = os.path.join(run, name)
            open(p, "w").write(src)
            return p

        # 1. THE REAL CASE: a substrate knob with no flag.
        p = w("_a.py", "class N:\n    def __init__(self, ou_noise=False):\n"
                       "        self.cfg.enable_ou_process = bool(ou_noise)\n"
                       "def main():\n    ap.add_argument('--seeds')\n")
        if not _check_one(p, "research/runners/_a.py"):
            bad.append("did NOT catch a cfg-writing constructor param with no flag")
        # 2. NEGATIVE CONTROL — the same knob WITH a flag passes.
        p = w("_b.py", "class N:\n    def __init__(self, ou_noise=False):\n"
                       "        self.cfg.enable_ou_process = bool(ou_noise)\n"
                       "def main():\n    ap.add_argument('--ou-noise')\n")
        if _check_one(p, "research/runners/_b.py"):
            bad.append("FALSE POSITIVE: flagged a knob that HAS a --flag")
        # 3. NEGATIVE CONTROL — a flag named for the cfg key, not the parameter, also satisfies it.
        p = w("_c.py", "class N:\n    def __init__(self, ou_noise=False):\n"
                       "        self.cfg.enable_ou_process = bool(ou_noise)\n"
                       "def main():\n    ap.add_argument('--enable-ou-process')\n")
        if _check_one(p, "research/runners/_c.py"):
            bad.append("FALSE POSITIVE: flagged a knob whose flag matches the cfg key")
        # 4. NEGATIVE CONTROL — a LITERAL is a design choice, not a knob.
        p = w("_d.py", "class N:\n    def __init__(self):\n        self.cfg.enable_ou_process = False\n"
                       "def main():\n    ap.add_argument('--seeds')\n")
        if _check_one(p, "research/runners/_d.py"):
            bad.append("FALSE POSITIVE: flagged a literal cfg assignment as a knob")
        # 5. NEGATIVE CONTROL — a library with no argparse is out of scope.
        p = w("_e.py", "class N:\n    def __init__(self, ou_noise=False):\n"
                       "        self.cfg.enable_ou_process = bool(ou_noise)\n")
        if _check_one(p, "research/runners/_e.py"):
            bad.append("FALSE POSITIVE: flagged a library with no CLI")
        # 6. NEGATIVE CONTROL — a `dest=` alias counts as reachable.
        p = w("_f.py", "class N:\n    def __init__(self, ou_noise=False):\n"
                       "        self.cfg.enable_ou_process = bool(ou_noise)\n"
                       "def main():\n    ap.add_argument('--noise', dest='ou_noise')\n")
        if _check_one(p, "research/runners/_f.py"):
            bad.append("FALSE POSITIVE: flagged a knob reachable via dest=")
        # 7. SCOPING
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
    return bad
