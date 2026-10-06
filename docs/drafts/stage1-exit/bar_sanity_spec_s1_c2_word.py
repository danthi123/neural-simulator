"""bar-sanity spec: Stage-1 C2 (word -> concept recall), the real aggregate's `word` verdict (lesion_recruit patch with call count;
role-permuted exposure). Real attempt-1 data re-score as PASS 6/6 under this runner, so a real brain passes it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s1_c123_common as C  # noqa: E402

NAME = "stage1-C2 word"
DESCRIPTION = ("C2 `word` verdict PASS under the real stage1_exit_gate.aggregate, six gate seeds; perfect = the word bound and "
               "reused in both roles with the moats intact; nulls = lesion/permuted inert, unreproduced, degenerate, one brain per "
               "missing sub-capability.")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_s1_c123_common.py"]


def decide(data):
    return C.decide_group(data, "word")


globals().update(C.generators("word", null_kinds={"null_no_preexposure": "no_preexposure", "null_no_moat": "no_moat",
                                                  "null_one_role": "one_role"}))
