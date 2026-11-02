"""bar-sanity spec (amendment A6, ruling Claude 2026-11-01, criterion-only variant): the Stage-2 STAGE FAIL verdict.

decide = the real aggregate (`research.runners.stage2_exit_gate.aggregate` at level `bar_sanity_s2_common.LEVEL`, "A6")
scores the STAGE FAIL over six synthetic gate seeds x five groups. The ruling: the stage is FAIL whenever any counted
criterion FAILs, and UNDEFINED only when no counted criterion FAILs and at least one is UNDEFINED/INCOMPLETE. An
undefined C2 must never hide a defined C1 failure.
  'perfect' kind = a brain with a DEFINED criterion failure (FAIL is the right stage verdict; it must be FAIL >= 0.95):
                   C1 missing a capability (the 10-26 named gap `no_elaboration`; random replies) with the C2 permuted
                   hop teach refused (the 10-26 / 11-01 shape: C2 INCOMPLETE), and the same C1 failure with C2 received.
  'null' kind    = a brain the stage FAIL would MISDESCRIBE (it must be FAIL <= 0.05): every criterion capable but the
                   C2 permuted hop teach refused (C2 INCOMPLETE, nothing FAILs: the stage is UNDEFINED), and the fully
                   capable brain (PASS)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2 stage FAIL-claim (A6, criterion-only)"
DESCRIPTION = ("Stage-2 stage verdict at amendment A6 (criterion-only ruling 2026-11-01): the real aggregate's stage verdict "
               "is FAIL. 'perfect' = a defined C1 failure, with or without the C2 permuted hop teach refused; 'null' = every "
               "criterion capable with the C2 permuted teach refused (UNDEFINED), and the fully capable brain (PASS).")
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]
decide = C.decide_stage_fail

_REFUSED = {"permuted": "permuted+refused"}
_C1_GAP = {"intact_a": "no_elaboration", "intact_b": "no_elaboration"}


def _c1_random(rng, refused=True):
    d = C.make_stage(rng, {"infer": _REFUSED} if refused else None)
    d["conv"] = C.make_group(rng, "conv", degenerate="random")
    return d


perfect = lambda rng: C.make_stage(rng, {"conv": _C1_GAP, "infer": _REFUSED})                     # noqa: E731
perfect_exact = lambda rng: C.make_stage(rng, {"conv": _C1_GAP, "infer": _REFUSED}, realistic=False)  # noqa: E731
perfect_c1_random_c2_refused = _c1_random
perfect_c1_gap_c2_received = lambda rng: C.make_stage(rng, {"conv": _C1_GAP})                     # noqa: E731
null = lambda rng: C.make_stage(rng, {"infer": _REFUSED})                                          # noqa: E731
null_refused_exact = lambda rng: C.make_stage(rng, {"infer": _REFUSED}, realistic=False)           # noqa: E731
null_capable = lambda rng: C.make_stage(rng)                                                       # noqa: E731
