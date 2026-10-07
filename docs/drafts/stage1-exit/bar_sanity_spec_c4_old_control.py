"""bar-sanity spec: Stage-1 C4 as REGISTERED at attempt 1 -- the scrambled control is the first derangement the fixed RNG
offers (no constraint on its correlation with the human ratings). Expected verdict: UNREACHABLE (RULING s1-c4-control)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_c4_common as C  # noqa: E402

NAME = "stage1-C4 old control (unconstrained derangement)"
DESCRIPTION = ("C4 val_heldout verdict PASS under the real scorer, six gate-style seeds; the scrambled arm uses the "
               "unconstrained derangement (research/runners/stage1_exit_gate.py at 8c56ff97b).")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_c4_common.py"]
decide = C.decide


def _make(**kw):
    return lambda rng: C.make_data(rng, C.derangement_old, **kw)


perfect = _make()                                       # realistic: reads follow the held values at r 0.93-0.96
perfect_exact = _make(rho=None)                         # exact: the brain follows its values with no read noise
null_control_ignored = _make(control="ignores")        # the scrambled control does not reach the reply
null_lesion_inert = _make(lesion="inert")              # the ladder lesion changes nothing
random = _make(intact="random")                        # replies independent of the map
