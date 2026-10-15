"""bar-sanity spec: Stage-3 C4 (held-out credit after a correction), the DRAFT group `credit` CRITERION verdict (capability
AND ownership) under the real aggregate (research/runners/_stage3_exit_draft.py) over synthetic six-seed shards
(bar_sanity_s3_common.py). Bar: PLAN.md sections 1 and 6. Nulls: every host shortcut PLAN.md section 7 names plus one
inert-instrument brain per clause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s3_common as C  # noqa: E402

NAME = "stage3-C4 credit (DRAFT)"
DESCRIPTION = ("Stage-3 C4 (held-out credit after a correction): the real aggregate's `credit` criterion verdict is PASS over six "
               "gate seeds; perfect = the held-out item is corrected while controls stand, no_teacher and replay_only do not, "
               "learning lesion abolishes it; nulls = production at main, recency, exemplar nearest-neighbour, a host-inferred "
               "block, activity-only, each instrument made inert, an unreproduced lesion, read-out-dependent decisions.")
SCORER_FILES = ["research/runners/_stage3_exit_draft.py", "docs/drafts/stage3-exit/bar_sanity_s3_common.py"]


def decide(data):
    return C.decide_group(data, "credit")


globals().update(C.generators("credit"))
