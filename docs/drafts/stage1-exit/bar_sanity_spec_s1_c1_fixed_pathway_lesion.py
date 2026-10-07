"""bar-sanity spec: Stage-1 C1 for ATTEMPT 2 with the outcome-blind instrument fix: the counted lesion arm is the pathway cut
(`lesion_pathway`, an in-process patch with a call count, as docs/drafts/stage1-c1-inheritance/PLAN.md "Controls" already
specifies) instead of the DRAW lesion (which the pathway route never touches). The fix is applied here by replacing
GROUPS["cat_disc"]["lesions"] in the imported scorer module (the real aggregate then runs unchanged over the new arm names);
the registered runner file itself is not edited by this spec. The bar (inherits on 6/6, permuted tracks 6/6, lesion effect
6/6 reproduced) is the registered one; only WHICH lesion is counted changes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s1_c123_common as C  # noqa: E402

C.G.GROUPS["cat_disc"]["lesions"] = {"lesion_pathway": {"env": {}, "patch": "lesion_inherit_pathway",
                                                        "cuts": "the learned agent->context pathway of the inheritance route"}}
NAME = "stage1-C1 cat_disc, attempt 2 (pathway lesion counted)"
DESCRIPTION = ("C1 `cat_disc` verdict PASS under the real stage1_exit_gate.aggregate with GROUPS['cat_disc']['lesions'] = "
               "{lesion_pathway} (patch with call count); perfect = the pathway inheritor of the C1 PLAN.")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_s1_c123_common.py"]


def decide(data):
    return C.decide_group(data, "cat_disc")


def _k(**kw):
    return lambda rng: C.make_group(rng, "cat_disc", **kw)


# the default brain of this spec is the pathway inheritor (the build attempt 2 runs)
C.DEFAULT_KIND["cat_disc"] = "pathway"
globals().update(C.generators(
    "cat_disc", lesions=["lesion_pathway"],
    extra={"null_frequency_shortcut": _k(kind="frequency")},
    info_kinds={"info_perfect_both_routes": dict(kind="both")}))
