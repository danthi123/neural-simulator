"""bar-sanity spec: Stage-2 C3 (curiosity ask-rate and learning-progress), the registered group `ask` verdict under the real scorer.
decide = `research.runners.stage2_exit_gate.aggregate` over synthetic six-seed shards (bar_sanity_s2_common.py; surfaces
copied from the dev-seed-7 shards). Bar: correct on 6/6 AND every lesion effect 6/6 AND reproduced 6/6 AND permuted tracks 6/6
(prereg sections 4 and 7). Needs research/runners/stage2_exit_gate.py at or after e125827f2 (research/stage2-exit-gate).
Amendment A4 (2026-10-13; docs/drafts/stage2-exit/AMENDMENT-A4.md): correct now means every C3 clause in BOTH the registered
and the counted reversed-order arm, and a taught moat whose teach was not acquired is undefined; the generators gain
perfect brains with one / two unacquired moats and nulls for an order-carried verdict and for no acquired moat."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2-C3 ask"
DESCRIPTION = ("Stage-2 C3 (curiosity ask-rate and learning-progress): the real aggregate's `ask` verdict is PASS over six gate seeds; perfect = "
               "the capability present, realistic dev-7 reply surfaces; nulls = each lesion / the permuted arm made inert, "
               "unreproduced lesions, a non-deterministic build, degenerate and random brains, one brain per missing sub-capability; "
               "A4: perfect brains with unacquired taught moats, nulls with the verdict carried by one probe order or no moat acquired.")
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return C.decide_group(data, "ask")


globals().update(C.generators("ask"))
