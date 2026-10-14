"""bar-sanity spec: Stage-3 C3 (false belief), the DRAFT group `tom` CRITERION verdict (capability AND ownership) under the
real aggregate (research/runners/_stage3_exit_draft.py) over synthetic six-seed shards (bar_sanity_s3_common.py). Bar:
PLAN.md sections 1 and 5. Nulls: every host shortcut PLAN.md section 7 names plus one inert-instrument brain per clause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s3_common as C  # noqa: E402

NAME = "stage3-C3 tom (DRAFT)"
DESCRIPTION = ("Stage-3 C3 (false belief, counterbalanced + true-belief control): the real aggregate's `tom` criterion verdict is "
               "PASS over six gate seeds; perfect = belief named from the store on every scenario, store-lesion abolishes it, "
               "steer on the store moves it; nulls = first-location, present-means-reality, host rules that never read the "
               "store or veto it, a host-written belief, each instrument made inert, an unreproduced lesion, "
               "read-out-dependent decisions.")
SCORER_FILES = ["research/runners/_stage3_exit_draft.py", "docs/drafts/stage3-exit/bar_sanity_s3_common.py"]


def decide(data):
    return C.decide_group(data, "tom")


globals().update(C.generators("tom"))
