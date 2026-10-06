"""bar-sanity spec: Stage-1 C4 with the constrained draw TIGHTENED to SCRAMBLE_R_MAX = 0.05 (the proposal in the
2026-10-06 bar-sanity review: at 0.10 a realistic perfect brain passes the six-seed clause 94.3-94.5%, just under 95%).
This spec PINS the constant to 0.05 for the draw; the scorer file it hashes is the one on disk, so once
research/runners/stage1_exit_gate.py itself says 0.05, cite the record of `bar_sanity_spec_c4_constrained.py` instead
(it reads the module's own constant). Needs stage1_exit_gate.py at or after bc8f804b9 (research/stage1-exit-gate)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_c4_common as C  # noqa: E402

C.G.SCRAMBLE_R_MAX = 0.05            # the proposed constant (the module ships 0.10); affects this process only

NAME = "stage1-C4 constrained draw, SCRAMBLE_R_MAX pinned to 0.05"
DESCRIPTION = ("C4 val_heldout verdict PASS under the real scorer, six gate-style seeds; the scrambled arm uses "
               "stage1_exit_gate._scrambled_valence_draw with SCRAMBLE_R_MAX = 0.05 (pinned by this spec).")
SCORER_FILES = ["research/runners/stage1_exit_gate.py", "docs/drafts/stage1-exit/bar_sanity_c4_common.py"]
decide = C.decide


def _make(**kw):
    return lambda rng: C.make_data(rng, C.derangement_constrained, **kw)


perfect = _make()
perfect_exact = _make(rho=None)
null_control_ignored = _make(control="ignores")
null_lesion_inert = _make(lesion="inert")
random = _make(intact="random")
