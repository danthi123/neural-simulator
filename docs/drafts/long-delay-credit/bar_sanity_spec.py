"""bar-sanity spec: T1 engagement bar (a) AS AMENDED by A0.1: the LHb rate ratio > 2 over the reward-time window
[REWARD_ON_MS, REWARD_ON_MS + REWARD_MS). Expected verdict: OK (a brain whose reward silences the LHb for the 50 ms
pulse passes; a brain with no reward-time change does not)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_t1_common as C  # noqa: E402

NAME = "T1 engagement bar (a), reward-time window (A0.1)"
DESCRIPTION = ("lhb_window_rates(lhb_spikes_rewardwin, REWARD_MS) + engagement_a(factor 2), "
               "research/runners/_ldc_omission_pilot.py")
SCORER_FILES = ["research/runners/_ldc_omission_pilot.py", "docs/drafts/long-delay-credit/bar_sanity_t1_common.py"]
decide = C.decide_reward_window

perfect = lambda rng: C.make_run(rng, "silent_in_window", C.TIMING_4A)                   # noqa: E731  complete silencing in the window
perfect_registered_timing = lambda rng: C.make_run(rng, "silent_in_window", C.TIMING_REGISTERED)  # noqa: E731
null = lambda rng: C.make_run(rng, "none", C.TIMING_4A)                                   # noqa: E731  no reward-time change
null_registered_timing = lambda rng: C.make_run(rng, "none", C.TIMING_REGISTERED)        # noqa: E731
random = lambda rng: C.make_run(rng, "none", C.TIMING_4A, gain_sd=0.5)                    # noqa: E731  noisy drive, independent of reward
# REPORTED, not thresholded: the round-1 release arm (about 17% suppression) must NOT pass the amended bar
info_weak_suppression = lambda rng: C.make_run(rng, "weak", C.TIMING_4A)                  # noqa: E731
