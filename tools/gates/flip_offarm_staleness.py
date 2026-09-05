"""flip_offarm_staleness (CLASS OS, BLOCK) — a flip/soak/verify runner's OFF arm must FORCE the flag OFF
explicitly (`os.environ[F] = "0"`), never rely on `os.environ.pop(F)`.

WHY (the 2026-08-27 flip-soak-off-arm-staleness audit, research/findings/2026-08-27-flip-soak-off-arm-staleness-audit.md,
+ FAILURE_LOG 2026-08-27 which scoped THIS gate as the explicit follow-up). `_spiking_mouth_recall_soak.py::_set_flag(False)`
did `os.environ.pop("BRAIN_SPIKING_MOUTH_RECALL", None)`, written when that flag defaulted OFF (unset==OFF). The
2026-08-26 wave-3 flip made the flag default ON *without* touching the soak — so `unset` silently started reading ON,
and every "flag-off vs flag-on" comparison the soak made collapsed to a vacuous ON-vs-ON compare, with NO error. The
soak had banked 6/6 GO; re-run it read 0/6. An audit found 5 MORE soaks with the identical bug. This is a
check-that-cannot-fail sibling: a comparison whose control arm is silently equal to its treatment arm proves nothing.

WHAT THIS GATE ENFORCES. In an in-scope runner (research/runners|tools, basename ~ soak|flip|verify), for each
non-`*_LESION* BRAIN_ flag F, the STALE-ON-vs-ON pattern is:

    F is POPPED (os.environ.pop(F, ...))  — the OFF/baseline arm is `unset`, not explicit
    AND F is EXPLICITLY set ON somewhere (os.environ[F] = "1"/"on"/...)  — so there IS a real ON-vs-OFF comparison
    AND F is NEVER explicitly set to a falsy literal (os.environ[F] = "0"/"off"/...)  — the OFF arm is unset-only
    AND F's CURRENT production default resolves ON  — so `unset` now == ON == the treatment arm

All four ⇒ the OFF arm is not OFF; BLOCK. The fix is the reference fix: write `os.environ[F] = "0"` for the OFF arm.

WHY EACH CONJUNCT (calibrated against the audit's full table so the current tree passes with 0 false positives):
  * `*_LESION` flags are EXCLUDED — a production wave-flip turns a FACULTY on, never a LESION; a lesion flag's `unset`
    correctly means "not lesioned" and that default does not flip (audit: "safe by construction").
  * requiring an EXPLICIT-ON arm excludes direct-function tests that pop to TEST the real default (e.g.
    `_verify_d5_episodic_organ.py`, `_d6_multiref_wm_production_verify.py`): they assert the on-default is correct, they
    do not treat the pop as an OFF arm — there is no `="1"` force-on to compare against.
  * requiring NO explicit-falsy-set excludes every soak that already writes `="0"` for its OFF arm (the post-fix state
    of all 6 known instances, and every soak that was always safe) — the pop that remains there is cleanup/reset.
  * requiring the default to resolve ON excludes flags that legitimately still default OFF (e.g. `BRAIN_OPEN_ENDED`,
    `BRAIN_GNW_BUS`): for those, pop==OFF is currently correct. (When such a flag is later flipped ON, its default
    resolves ON and this gate then fires — which is exactly the forward-looking protection.)

HONEST BOUNDARY. Static, config-as-source: it reads the flag's default off its reader's `.get(F, <lit>)` literal or a
same-function `_*_DEFAULT_ON` constant. A default that is neither (a runtime brain.json override, a default computed by
a helper this gate cannot follow) resolves UNKNOWN and is NOT blocked — a conservative miss, never a false block (a
false block gets the whole hook bypassed with --no-verify, disabling every OTHER gate). The instrument the audit used —
importing each reader with the flag unset — is the ground truth; this is its cheap static shadow.
"""
from __future__ import annotations

import glob
import os
import re

NAME = "flip-offarm-staleness"
CLASS_ID = "OS"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# in-scope: a flag on/off comparison runner (the family the audit's follow-up named).
_SCOPE_DIRS = ("research/runners/", "tools/")
_SCOPE_BASENAME = re.compile(r"(soak|flip|verify)", re.I)

_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no", ""}

# reader dirs to resolve a flag's current production default (its "owning module").
_READER_GLOBS = ("webapp/*.py", "research/runners/*.py", "webapp/**/*.py")


# ---------------------------------------------------------------- source parsing (flag-aware, var-indirection-aware)
def _flag_var_map(text):
    """`_FLAG = "BRAIN_X"` -> {'_FLAG': 'BRAIN_X'} so os.environ[_FLAG]/pop(_FLAG) resolve to the flag."""
    out = {}
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\'](BRAIN_[A-Z0-9_]+)["\']\s*(?:#.*)?$', text, re.M):
        out[m.group(1)] = m.group(2)
    return out


def _pops(text, varmap):
    """Every BRAIN_ flag popped (literal `pop("BRAIN_X"` or var `pop(_FLAG`)."""
    out = set()
    for m in re.finditer(r'os\.environ\.pop\(\s*(?:["\'](BRAIN_[A-Z0-9_]+)["\']|([A-Za-z_][A-Za-z0-9_]*))', text):
        if m.group(1):
            out.add(m.group(1))
        elif m.group(2) in varmap:
            out.add(varmap[m.group(2)])
    return out


def _sets(text, varmap):
    """flag -> set of ALL assigned literals (lowercased) on the RHS of `os.environ[F] = ...`. Captures every
    quoted literal to end-of-line so the ternary idiom `os.environ[F] = "1" if on else "0"` (very common here)
    contributes BOTH "1" and "0" — missing the `else "0"` was a false-positive bug (the whole RHS is the OFF arm)."""
    out = {}
    key = re.compile(r'os\.environ\[\s*(?:["\'](BRAIN_[A-Z0-9_]+)["\']|([A-Za-z_][A-Za-z0-9_]*))\s*\]\s*=\s*(.*)$',
                     re.M)
    for m in key.finditer(text):
        flag = m.group(1) or varmap.get(m.group(2))
        if not flag:
            continue
        for lit in re.findall(r'["\']([^"\']*)["\']', m.group(3)):
            out.setdefault(flag, set()).add(lit.strip().lower())
    return out


def _classify(values):
    """(has_truthy_set, has_falsy_set) over a set of assigned literals."""
    return (any(v in _TRUTHY for v in values), any(v in _FALSY for v in values))


# ---------------------------------------------------------------- default resolution (the flag's owning-module reader)
def _reader_function_slice(lines, get_line_idx):
    """The body of the def enclosing line `get_line_idx` (for constant-indirection default lookup)."""
    start = 0
    indent = None
    for i in range(get_line_idx, -1, -1):
        m = re.match(r'^(\s*)def\s+\w+', lines[i])
        if m:
            start = i
            indent = len(m.group(1))
            break
    if indent is None:
        return "\n".join(lines[max(0, get_line_idx - 6): get_line_idx + 3])
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r'^(\s*)(def|class)\s', lines[j])
        if m and len(m.group(1)) <= indent:
            end = j
            break
    return "\n".join(lines[start:end])


def _default_from_text(flag, text):
    """Resolve `flag`'s default from one source file. Returns 'on'/'off'/None(unknown)."""
    lines = text.splitlines()
    verdict = None
    # (1) literal fallback: os.environ.get("F", "<lit>")  / os.getenv("F", "<lit>")
    for m in re.finditer(r'os\.(?:environ\.get|getenv)\(\s*["\']%s["\']\s*,\s*["\']([^"\']*)["\']' % re.escape(flag),
                         text):
        lit = m.group(1).strip().lower()
        if lit in _TRUTHY:
            return "on"          # any truthy literal default is decisive
        if lit in _FALSY:
            verdict = verdict or "off"
    # (2) constant-indirection: bare os.environ.get("F") in a function that gates on a `_*_DEFAULT_ON` constant.
    for m in re.finditer(r'os\.(?:environ\.get|getenv)\(\s*["\']%s["\']\s*\)' % re.escape(flag), text):
        idx = text[:m.start()].count("\n")
        body = _reader_function_slice(lines, idx)
        for cm in re.finditer(r'(_[A-Z0-9_]*DEFAULT(?:_ON)?)\b', body):
            const = cm.group(1)
            am = re.search(r'^\s*%s\s*=\s*(True|False|["\'][^"\']*["\'])' % re.escape(const), text, re.M)
            if not am:
                continue
            val = am.group(1).strip().strip('"\'').lower()
            if val in ("true",) or val in _TRUTHY:
                return "on"
            if val in ("false",) or val in _FALSY:
                verdict = verdict or "off"
    return verdict


def _default_state(flag, root):
    """Resolve across the flag's owning-module files. 'on' wins if ANY reader defaults it ON."""
    off_seen = False
    seen_files = set()
    for pat in _READER_GLOBS:
        for path in glob.glob(os.path.join(root, pat), recursive=True):
            if path in seen_files:
                continue
            seen_files.add(path)
            try:
                text = open(path, errors="ignore").read()
            except Exception:
                continue
            if flag not in text:
                continue
            v = _default_from_text(flag, text)
            if v == "on":
                return "on"
            if v == "off":
                off_seen = True
    return "off" if off_seen else None


# ---------------------------------------------------------------- the check
def _in_scope(rel):
    rel = rel.replace("\\", "/")
    if not any(rel.startswith(d) for d in _SCOPE_DIRS) and not os.path.isabs(rel):
        # allow absolute fixture paths (selftest) to pass the dir test on basename alone
        return _SCOPE_BASENAME.search(os.path.basename(rel)) is not None and rel.endswith(".py")
    return _SCOPE_BASENAME.search(os.path.basename(rel)) is not None and rel.endswith(".py")


def _violations_in(path, root, _default_fn=None):
    default_fn = _default_fn or (lambda f: _default_state(f, root))
    try:
        text = open(path, errors="ignore").read()
    except Exception:
        return []
    varmap = _flag_var_map(text)
    popped = _pops(text, varmap)
    sets = _sets(text, varmap)
    probs = []
    for flag in sorted(popped):
        if "_LESION" in flag:
            continue                                   # a wave-flip never targets a lesion flag's default
        has_truthy, has_falsy = _classify(sets.get(flag, set()))
        if not has_truthy or has_falsy:
            continue                                   # no explicit ON arm, or an explicit OFF arm already exists
        if default_fn(flag) != "on":
            continue                                   # default is OFF/unknown -> pop==OFF is (currently) fine
        rel = os.path.relpath(path, root)
        probs.append(
            "%s: OFF arm of %s uses os.environ.pop() while %s DEFAULTS ON in production -> the OFF arm silently "
            "reads ON (a vacuous ON-vs-ON comparison). Force it explicitly: os.environ[%s] = \"0\". "
            "(2026-08-27 flip-soak-off-arm-staleness class)" % (rel, flag, flag, flag))
    return probs


def check(paths):
    root = _ROOT
    # ALWAYS corpus-scan every in-scope runner, regardless of what THIS commit staged. The 2026-08-27 bug and its
    # gate assumed the stale OFF arm lives in a STAGED file — but the recurrence (2026-09-05: the one-brain flip AND
    # the composer flip, twice in one session) came the OTHER way: a default-flip landed in an OWNING MODULE
    # (one_brain_composer.py / onebrain_single_pool_production.py) whose commit did NOT stage the dependent verify
    # runner, so the staged-files scope never re-scanned the runner whose pop-based OFF arm the flip had just made
    # stale. Because _violations_in resolves each flag's CURRENT default from its owning module, a corpus scan of
    # every runner catches exactly that cross-file staleness. Cheap (regex over ~N small files); the current tree is
    # clean, so this only fires on a genuinely-stale OFF arm (a flip that outran its verifier). `paths` kept for the
    # gate contract but intentionally not used to narrow scope.
    cand = []
    for pat in ("research/runners/*.py", "tools/*.py"):
        cand += glob.glob(os.path.join(root, pat))
    cand = [p for p in cand if _SCOPE_BASENAME.search(os.path.basename(p))]
    problems = []
    for path in cand:
        if not os.path.exists(path):
            continue
        problems += _violations_in(path, root)
    return problems


# ---------------------------------------------------------------- selftest (MUST fail in the failing direction)
def selftest():
    import tempfile
    bad = []

    # A REAL flag that genuinely defaults ON via a literal reader in the repo, so default resolution is decisive.
    on_flag = "BRAIN_GNW_2ORGAN"      # webapp/server.py: os.environ.get("BRAIN_GNW_2ORGAN", "on")
    off_flag = "BRAIN_OPEN_ENDED"     # webapp/server.py: os.environ.get("BRAIN_OPEN_ENDED", "0")  (default OFF)
    if _default_state(on_flag, _ROOT) != "on":
        bad.append("PRECONDITION: expected %s to resolve default-ON in the repo (reader idiom changed?)" % on_flag)
    if _default_state(off_flag, _ROOT) not in ("off", None):
        bad.append("PRECONDITION: expected %s to resolve default-OFF in the repo" % off_flag)

    def _write(name, body):
        d = tempfile.mkdtemp(prefix="offarm_selftest_")
        p = os.path.join(d, name)
        open(p, "w").write(body)
        return p

    # (1) FAILING DIRECTION: explicit ON arm + pop-based OFF arm, flag defaults ON -> MUST be flagged.
    bad_src = (
        "import os\n_FLAG = '%s'\n"
        "def _set(on):\n"
        "    if on:\n        os.environ[_FLAG] = '1'\n"
        "    else:\n        os.environ.pop(_FLAG, None)   # OFF arm (stale: default is ON)\n" % on_flag)
    bad_p = _write("_selftest_offarm_flip_soak.py", bad_src)
    if not _violations_in(bad_p, _ROOT):
        bad.append("did NOT flag a pop-based OFF arm on a default-ON flag (the exact 2026-08-27 bug)")

    # (2) the FIX must PASS: explicit ='0' OFF arm.
    good_src = (
        "import os\n_FLAG = '%s'\n"
        "def _set(on):\n"
        "    if on:\n        os.environ[_FLAG] = '1'\n"
        "    else:\n        os.environ[_FLAG] = '0'   # explicit OFF (the reference fix)\n" % on_flag)
    good_p = _write("_selftest_offarm_flip_soak.py", good_src)
    if _violations_in(good_p, _ROOT):
        bad.append("FALSE POSITIVE: flagged a runner whose OFF arm is an explicit ='0'")

    # (3) a LESION flag pop must NOT be flagged (excluded by construction).
    les_src = (
        "import os\n"
        "def _set(on, lesion=False):\n"
        "    os.environ['%s'] = '1' if on else '0'\n"
        "    if lesion:\n        os.environ['%s_LESION'] = '1'\n"
        "    else:\n        os.environ.pop('%s_LESION', None)\n" % (on_flag, on_flag, on_flag))
    les_p = _write("_selftest_offarm_flip_soak.py", les_src)
    if _violations_in(les_p, _ROOT):
        bad.append("FALSE POSITIVE: flagged a *_LESION flag pop (wave-flips never target lesion defaults)")

    # (4) a DEFAULT-OFF flag with the same asymmetric arms must NOT be flagged (pop==OFF is currently correct).
    offdef_src = (
        "import os\n_FLAG = '%s'\n"
        "def _set(on):\n"
        "    if on:\n        os.environ[_FLAG] = '1'\n"
        "    else:\n        os.environ.pop(_FLAG, None)\n" % off_flag)
    offdef_p = _write("_selftest_offarm_flip_soak.py", offdef_src)
    if _violations_in(offdef_p, _ROOT):
        bad.append("FALSE POSITIVE: flagged a pop OFF arm on a flag that still DEFAULTS OFF")

    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
    print("full-tree check (in-scope runners):")
    for p in check(None):
        print("  ", p)
