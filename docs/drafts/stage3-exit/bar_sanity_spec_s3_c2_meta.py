"""bar-sanity spec: Stage-3 C2 (self-report tracks confidence), the DRAFT group `meta` CRITERION verdict (capability AND
ownership) under the real aggregate (research/runners/_stage3_exit_draft.py) over synthetic six-seed shards
(bar_sanity_s3_common.py). Bar: PLAN.md sections 1 and 4. Nulls: every host shortcut PLAN.md section 7 names plus one
inert-instrument brain per clause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s3_common as C  # noqa: E402

NAME = "stage3-C2 meta (DRAFT)"
DESCRIPTION = ("Stage-3 C2 (self-report tracks confidence, natural uncertainty): the real aggregate's `meta` criterion verdict is "
               "PASS over six gate seeds; perfect = type-2 AUROC over the guess stratum >= 0.75 with a load-bearing, steerable "
               "inheritance read-out; nulls = never abstains, hedge-by-taught, all-abstain, item-keyed, random opt-out, a host "
               "conflict counter, each instrument made inert, an unreproduced lesion, read-out-dependent decisions.")
SCORER_FILES = ["research/runners/_stage3_exit_draft.py", "docs/drafts/stage3-exit/bar_sanity_s3_common.py"]


def decide(data):
    return C.decide_group(data, "meta")


globals().update(C.generators("meta"))
