"""bar-sanity spec: the Stage-2 STAGE verdict (all five criteria PASS and every precondition earned) under the real
aggregate. A perfect brain passes all five groups; each null makes one group fail by one clause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2 stage verdict"
DESCRIPTION = "the real aggregate's stage verdict is PASS over six gate seeds x five groups; one group broken at a time for the nulls."
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]
decide = C.decide_stage
globals().update(C.stage_generators())
