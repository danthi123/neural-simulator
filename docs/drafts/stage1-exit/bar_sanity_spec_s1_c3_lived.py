"""bar-sanity spec: Stage-1 C3 (lived-fact recall), the real aggregate's `lived` verdict (Amendment A1: lesion_recall = the
lesion_lived_recall patch with a call count; lesion_episodic = the D5 env flag, no engagement record, so an inert one makes the
seed UNDEFINED). Real data re-score as PASS 6/6 under this runner (gate_c3rerun)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s1_c123_common as C  # noqa: E402

NAME = "stage1-C3 lived"
DESCRIPTION = ("C3 `lived` verdict PASS under the real stage1_exit_gate.aggregate, six gate seeds; perfect = the lived fact recalled "
               "after the distractors and the episode disclosed; nulls = either lesion / permuted inert, unreproduced, degenerate, "
               "one brain per missing sub-capability.")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_s1_c123_common.py"]


def decide(data):
    return C.decide_group(data, "lived")


globals().update(C.generators("lived", null_kinds={"null_no_preexposure": "no_preexposure", "null_no_disclosure": "no_disclosure"}))
