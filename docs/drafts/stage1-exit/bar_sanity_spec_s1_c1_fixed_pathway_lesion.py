"""bar-sanity spec: Stage-1 C1 for ATTEMPT 2 under Amendment A2 (prereg section 12, 2026-10-13): the counted lesion arm is
`lesion_inherit_pathway` (one name for the arm and the in-process patch it installs; applied = calls > 0) instead of the DRAW
lesion (which the pathway route never touches). Since A2 this is the runner's own GROUPS["cat_disc"]["lesions"] entry (asserted
below, not overridden), and the real aggregate also applies A2's C1 validity record (effective flags recorded in every arm;
the direct probe's organ read records hop: true; a patch never called voids the seed). The bar (inherits on 6/6, permuted
tracks 6/6, lesion effect 6/6 reproduced) is the registered one; only WHICH lesion is counted changes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s1_c123_common as C  # noqa: E402

assert list(C.G.GROUPS["cat_disc"]["lesions"]) == ["lesion_inherit_pathway"], C.G.GROUPS["cat_disc"]["lesions"]
assert C.G.GROUPS["cat_disc"]["lesions"]["lesion_inherit_pathway"]["patch"] == "lesion_inherit_pathway"
NAME = "stage1-C1 cat_disc, attempt 2 (Amendment A2: lesion_inherit_pathway counted, hop record required)"
DESCRIPTION = ("C1 `cat_disc` verdict PASS under the real stage1_exit_gate.aggregate with the runner's A2 GROUPS entry "
               "{lesion_inherit_pathway} (patch with call count) and A2's hop/flag validity record; perfect = the pathway "
               "inheritor of the C1 PLAN at the flip revision.")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_s1_c123_common.py"]


def decide(data):
    return C.decide_group(data, "cat_disc")


def _k(**kw):
    return lambda rng: C.make_group(rng, "cat_disc", **kw)


# the default brain of this spec is the pathway inheritor (the build attempt 2 runs)
C.DEFAULT_KIND["cat_disc"] = "pathway"
globals().update(C.generators(
    "cat_disc", lesions=["lesion_inherit_pathway"],
    extra={"null_frequency_shortcut": _k(kind="frequency"),
           # A2's validity record: a perfect reply pattern without the record must never PASS
           "null_hop_not_recorded": _k(c1_hop=False),                      # the 09-30 organ (no hop) answering
           "null_flags_not_recorded": _k(c1_flags=False),                  # a shard written before A2
           "null_lesion_never_called": _k(zero_calls=("lesion_inherit_pathway",))},   # effect without engagement
    info_kinds={"info_perfect_both_routes": dict(kind="both")}))
