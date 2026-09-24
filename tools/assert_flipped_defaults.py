#!/usr/bin/env python3
"""Guard for the 2026-09-23 production-default batteries (tags flipdefaults-thin / flipdefaults-adequate).

Those batteries pass NO fix flag, so they measure whatever the checked-out code defaults to. Run on a revision from
before the flip, they would silently measure the OLD (OFF) defaults and still look like a valid result. Each job line
in their JOBS.txt therefore starts with `.venv/bin/python tools/assert_flipped_defaults.py &&`, which exits 1 unless:
  * all three reader constants exist and are True (the revision contains the flip), and
  * none of the three flags is set in the environment (the job measures the default, not an override).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

FLIPPED = {
    "BRAIN_EPISODIC_STORE_VERIFY": ("research.runners._episodic_dap_dialogue_memory", "_STORE_VERIFY_DEFAULT_ON"),
    "BRAIN_PMEM_FACILITATION": ("research.runners.prospective_memory_production_organ",
                                "_PMEM_FACILITATION_DEFAULT_ON"),
    "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE": ("research.runners.source_provenance_honesty", "_ABSTAIN_AT_TIE_DEFAULT_ON"),
    # S09 (AG-FLIP, 2026-09-24) -- PARKED on this branch only (`research/settle-default-on-prep`), pending the
    # owner's S00(a) fork answer; NOT present on main/M1 (bd391aa31 does not carry this entry). Only takes effect
    # for a guard run against a revision that has THIS branch merged in -- see _affect_marker_wta_derisk.py's
    # _SETTLE_DEFAULT_ON block for why it is safe to add here without forcing the flip on M1's own battery.
    "BRAIN_AFFECT_MARKER_SETTLE": ("research.runners._affect_marker_wta_derisk", "_SETTLE_DEFAULT_ON"),
}


_MISSING = object()


def _source_constant(mod, const):
    """Value of a module-level `CONST = <literal>` assignment, parsed from the file (no import)."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, *mod.split(".")) + ".py"
    try:
        tree = ast.parse(open(path).read(), filename=path)
    except (OSError, SyntaxError):
        return _MISSING
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == const for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return _MISSING
    return _MISSING


def problems(environ=None):
    environ = os.environ if environ is None else environ
    out = []
    # Review (2026-09-23, guard scope): ANY other BRAIN_* override in the node's ambient environment (e.g. a leftover
    # BRAIN_PMEM_OP_STABILIZER or BRAIN_AFFECT_MARKER_SETTLE) would make the battery measure something other than the
    # shipped default while still passing. The production-default batteries set no BRAIN_* variable at all.
    for k in sorted(environ):
        if k.startswith("BRAIN_") and k not in FLIPPED:
            out.append("%s is set in the environment (=%r); a production-default battery must set no BRAIN_* flag"
                       % (k, environ[k]))
    for flag, (mod, const) in FLIPPED.items():
        if flag in environ:
            out.append("%s is set in the environment (=%r); this battery must measure the default" % (flag, environ[flag]))
        # Read the constant from the module SOURCE, never by importing it (2026-09-24): importing
        # _episodic_dap_dialogue_memory pulls in cupy, which CPU-only AWS/pool nodes do not have, so every job's guard
        # failed with "No module named 'cupy'" and 156 shards ran nothing.
        val = _source_constant(mod, const)
        if val is _MISSING:
            out.append("%s.%s not found in source: this revision predates the 2026-09-23 flip" % (mod, const))
            continue
        if val is not True:
            out.append("%s.%s = %r, expected True" % (mod, const, val))
    return out


def selftest():
    """FAILING DIRECTION FIRST (2026-09-24, AGFLIP review of `research/settle-default-on-prep`): this guard's
    FLIPPED registry had no self-verifying check that the guard can actually FAIL -- the same property
    `tools/gates/` requires of every commit-time gate (docs/FAILURE_GATE_MATRIX.md: "the registry REFUSES to
    trust a gate whose selftest() does not itself fail in its failing direction"). Before this function, that
    property rested on one session's ad hoc sandbox check (copy the guard + one module into an isolated dir,
    mutate the constant, eyeball the exit code) -- reproducible, but not mechanically re-run. This script is
    NOT itself registered under tools/gates/ (it is a runtime job-shell guard invoked from JOBS.txt lines on
    research pool/AWS nodes, not a pre-commit doc/claim check on the repo), so it keeps its own selftest here;
    the return convention matches tools/gates/*.selftest() (a list of problems; empty == the guard proved it
    can fail every way it claims to).

    Exercises, for EVERY registered flag (not just BRAIN_AFFECT_MARKER_SETTLE): (1) the clean-checkout baseline
    is problem-free, (2) an env override is reported, (3) a constant forced to False is reported, (4) a constant
    forced MISSING (simulating a pre-flip revision) is reported. (3)/(4) monkeypatch `_source_constant` for the
    single (mod, const) pair under test only, falling back to the real reader for everything else, so this never
    touches a file on disk.
    """
    bad = []

    base = problems({})
    if base:
        bad.append("baseline problems() on a clean env is non-empty on this checkout: %r" % base)

    for flag in FLIPPED:
        p = problems({flag: "1"})
        if not any(flag in line for line in p):
            bad.append("problems({%r: '1'}) did NOT report the override -- guard cannot fail this way" % flag)

    orig_reader = _source_constant

    def _fake_reader(target_mod, target_const, fake_val):
        def _reader(mod, const):
            if mod == target_mod and const == target_const:
                return fake_val
            return orig_reader(mod, const)
        return _reader

    g = globals()
    for flag, (mod, const) in FLIPPED.items():
        for fake_val, label in ((False, "False"), (_MISSING, "MISSING (pre-flip revision)")):
            g["_source_constant"] = _fake_reader(mod, const, fake_val)
            try:
                p = problems({})
            finally:
                g["_source_constant"] = orig_reader
            if not any(mod in line and const in line for line in p):
                bad.append("problems() with %s.%s forced to %s did NOT report it -- guard cannot fail this way"
                           % (mod, const, label))

    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        st = selftest()
        for line in st:
            print("[assert_flipped_defaults] SELFTEST FAILED: " + line, file=sys.stderr)
        sys.exit(1 if st else 0)
    p = problems()
    for line in p:
        print("[assert_flipped_defaults] FAIL: " + line, file=sys.stderr)
    sys.exit(1 if p else 0)
