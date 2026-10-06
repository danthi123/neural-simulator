"""bar-sanity spec: Stage-1 C1 (category held-out inheritance) AS REGISTERED: the real aggregate's `cat_disc` verdict, with the
registered lesion `lesion_draw` (the generative DRAW's likelihood) counted. Perfect brains: `gen` (inheritance on the generative
route, the mechanism the prereg names) and `pathway` (inheritance on the NEW route of docs/drafts/stage1-c1-inheritance/PLAN.md,
the build attempt 2 will run: a learned agent->context pathway, DRAW not on its path). Expected verdict: UNREACHABLE (the
pathway brain: `lesion_draw` changes nothing -> A1 rule -> UNDEFINED seeds -> INCOMPLETE, never PASS)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s1_c123_common as C  # noqa: E402

NAME = "stage1-C1 cat_disc as registered (lesion_draw counted)"
DESCRIPTION = ("C1 `cat_disc` verdict PASS under the real stage1_exit_gate.aggregate, six gate seeds, registered lesion lesion_draw; "
               "perfect = generative-route inheritor and pathway inheritor (the planned attempt-2 build).")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_s1_c123_common.py"]


def decide(data):
    return C.decide_group(data, "cat_disc")


def _k(**kw):
    return lambda rng: C.make_group(rng, "cat_disc", **kw)


globals().update(C.generators(
    "cat_disc", perfect_kinds={"pathway": "pathway"},
    extra={"null_frequency_shortcut": _k(kind="frequency")},
    info_kinds={
        "info_perfect_both_routes": dict(kind="both"),
        "info_gen_draw_lesion_uniform_k2": dict(over={"lesion_draw": ("gen", "lesion_draw_u2")}),
        "info_gen_draw_lesion_uniform_k3": dict(over={"lesion_draw": ("gen", "lesion_draw_u3")}),
        "info_gen_draw_lesion_uniform_k5": dict(over={"lesion_draw": ("gen", "lesion_draw_u5")}),
        "info_gen_abstains_on_permuted": dict(reply=C.reply_cat_disc_abstain_perm),
    }))
