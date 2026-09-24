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
}


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
        try:
            val = getattr(importlib.import_module(mod), const)
        except Exception as e:  # noqa: BLE001
            out.append("%s.%s missing (%s): this revision predates the 2026-09-23 flip" % (mod, const, e))
            continue
        if val is not True:
            out.append("%s.%s = %r, expected True" % (mod, const, val))
    return out


if __name__ == "__main__":
    p = problems()
    for line in p:
        print("[assert_flipped_defaults] FAIL: " + line, file=sys.stderr)
    sys.exit(1 if p else 0)
