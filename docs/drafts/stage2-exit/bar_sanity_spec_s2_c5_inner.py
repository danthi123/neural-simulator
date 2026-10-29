"""bar-sanity spec: Stage-2 C5 (private-to-inner-speech chain), the registered group `inner` verdict under the real scorer.
decide = `research.runners.stage2_exit_gate.aggregate` over synthetic six-seed shards (bar_sanity_s2_common.py; surfaces
copied from the dev-seed-7 shards). Bar: correct on 6/6 AND every lesion effect 6/6 AND reproduced 6/6 AND permuted tracks 6/6
(prereg sections 4 and 7) as amended by AMENDMENT-A3 (docs/drafts/stage2-exit/AMENDMENT-A3.md): C5 permuted-tracks clause = the voiced idle fact differs between the intact and the permuted exposure. Needs research/runners/stage2_exit_gate.py with the A3 scorer (research/stage2-exit-gate + claude-review/stage2-c4c5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2-C5 inner"
DESCRIPTION = ("Stage-2 C5 (private-to-inner-speech chain): the real aggregate's `inner` verdict is PASS over six gate seeds; perfect = "
               "the capability present, realistic dev-7 reply surfaces; nulls = each lesion / the permuted arm made inert, "
               "unreproduced lesions, a non-deterministic build, degenerate and random brains, one brain per missing sub-capability; nulls also include (amendment A3) the fixed-fact brain that voices one stored fact on every idle turn whatever the exposure, and an info residual (a host first-taught-fact FIFO that does vary with the exposure).")
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return C.decide_group(data, "inner")


globals().update(C.generators("inner"))
