"""bar-sanity spec: Stage-3 C1 (3-hop chain via the workspace), the DRAFT group `chain` CRITERION verdict (capability AND
ownership) under the real aggregate (research/runners/_stage3_exit_draft.py) over synthetic six-seed shards
(bar_sanity_s3_common.py). Bar: PLAN.md sections 1 and 3. Nulls: every host shortcut PLAN.md section 7 names plus one
inert-instrument brain per clause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s3_common as C  # noqa: E402

NAME = "stage3-C1 chain (DRAFT)"
DESCRIPTION = ("Stage-3 C1 (3-hop chain via the workspace): the real aggregate's `chain` criterion verdict is PASS over six gate "
               "seeds; perfect = the capability present and owned, realistic reply surfaces; nulls = production at main, the "
               "named host shortcuts (kb closure, relay, unmarked terminal, single-branch fork, order dependence, host "
               "k-select), each instrument (lesion, steer, permuted, null) made inert, an unreproduced lesion, "
               "read-out-dependent decisions.")
SCORER_FILES = ["research/runners/_stage3_exit_draft.py", "docs/drafts/stage3-exit/bar_sanity_s3_common.py"]


def decide(data):
    return C.decide_group(data, "chain")


globals().update(C.generators("chain"))
