"""bar-sanity spec: Stage-1 C4 with the RULING s1-c4-control draw -- the scrambled control is the scorer module's own
`_scrambled_valence_draw` (rejection-sampled until |r(scrambled values, human ratings)| <= SCRAMBLE_R_MAX).
Needs research/runners/stage1_exit_gate.py at or after bc8f804b9 (research/stage1-exit-gate)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_c4_common as C  # noqa: E402

NAME = "stage1-C4 constrained draw (RULING s1-c4-control)"
DESCRIPTION = ("C4 val_heldout verdict PASS under the real scorer, six gate-style seeds; the scrambled arm uses "
               "stage1_exit_gate._scrambled_valence_draw (SCRAMBLE_R_MAX = %s)." % getattr(C.G, "SCRAMBLE_R_MAX", "absent"))
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_c4_common.py"]
decide = C.decide


def _make(**kw):
    return lambda rng: C.make_data(rng, C.derangement_constrained, **kw)


perfect = _make()
perfect_exact = _make(rho=None)
null_control_ignored = _make(control="ignores")
null_lesion_inert = _make(lesion="inert")
random = _make(intact="random")
