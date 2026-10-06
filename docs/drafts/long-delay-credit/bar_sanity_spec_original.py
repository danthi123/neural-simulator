"""bar-sanity spec: T1 engagement bar (a) AS REGISTERED (STEP4B section 2): the whole-window LHb rate ratio > 2.
Expected verdict: UNREACHABLE at both timings (the pulse-only brain cannot exceed 250/200 = 1.25 at the Step 4a timing, or
exactly 2.0 -- a strict ">" -- at the registered default timing; even silencing from reward onset to the end of the
window reaches only 250/150 = 1.67 at the Step 4a timing)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_t1_common as C  # noqa: E402

NAME = "T1 engagement bar (a), whole outcome window (as registered)"
DESCRIPTION = "lhb_window_rates(lhb_spikes, OUTCOME_MS) + engagement_a(factor 2), research/runners/_ldc_omission_pilot.py"
SCORER_FILES = ["research/runners/_ldc_omission_pilot.py", "docs/drafts/long-delay-credit/bar_sanity_t1_common.py"]
decide = C.decide_whole_window

perfect = lambda rng: C.make_run(rng, "silent_from_onset", C.TIMING_4A)           # noqa: E731  the strongest brain, tune timing
perfect_pulse_only = lambda rng: C.make_run(rng, "silent_in_window", C.TIMING_4A)  # noqa: E731
perfect_registered_timing_pulse = lambda rng: C.make_run(rng, "silent_in_window", C.TIMING_REGISTERED)  # noqa: E731
# REPORTED, not thresholded: at the registered default timing a silencing that outlasts the pulse reaches 100/30 = 3.3
info_registered_timing_silent_from_onset = lambda rng: C.make_run(rng, "silent_from_onset", C.TIMING_REGISTERED)  # noqa: E731
null = lambda rng: C.make_run(rng, "none", C.TIMING_4A)                           # noqa: E731
