"""bar-sanity spec: Stage-2 C4 (discrete-emotion read-out), the registered group `emo` verdict under the real scorer.
decide = `research.runners.stage2_exit_gate.aggregate` over synthetic six-seed shards (bar_sanity_s2_common.py; surfaces
copied from the dev-seed-7 shards). Bar: correct on 6/6 AND every lesion effect 6/6 AND reproduced 6/6 AND permuted tracks 6/6
(prereg sections 4 and 7). Needs research/runners/stage2_exit_gate.py at or after e125827f2 (research/stage2-exit-gate)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2-C4 emo"
DESCRIPTION = ("Stage-2 C4 (discrete-emotion read-out): the real aggregate's `emo` verdict is PASS over six gate seeds; perfect = "
               "the capability present, realistic dev-7 reply surfaces; nulls = each lesion / the permuted arm made inert, "
               "unreproduced lesions, a non-deterministic build, degenerate and random brains, one brain per missing sub-capability.")
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return C.decide_group(data, "emo")


globals().update(C.generators("emo"))
