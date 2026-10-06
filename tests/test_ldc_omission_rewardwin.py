"""AMENDMENT A0.1 (docs/drafts/long-delay-credit/AMENDMENT-A0.1.md, 2026-10-06): engagement bar (a) of STEP4B is read in
the reward-time window of the outcome period, not over the whole outcome window.

Pins three things without building a brain (the outcome loop is driven by a stub `steps`):
1. the reward-time window is the SAME steps on every trial ([REWARD_ON_MS, REWARD_ON_MS + REWARD_MS)), counted on
   unrewarded trials too (the expected reward time), and reward_us is driven only inside it on rewarded learn trials;
2. the whole-window rate ratio is capped at OUTCOME_MS / (OUTCOME_MS - REWARD_MS) = 1.25 at onset 150 / outcome 250
   even when the reward silences the LHb completely (why the registered bar could never pass);
3. the amended bar is still falsifiable: a weak reward-time suppression (the round-1 release arm, ~17%) FAILS it.
"""
import pytest

import research.runners._ldc_omission_pilot as pilot


class _P:
    ctx_pA = 1.0
    rus_pA = 2.0


def _stub_run(rate_per_step_free, rate_per_step_reward):
    """A Run whose `steps` emits `rate_per_step_free` LHb spikes per step, or `rate_per_step_reward` while reward_us is
    being driven (a stand-in for the lhb_inh -> lhb GABA_A brake)."""
    run = object.__new__(pilot.Run)
    run.p, run.learn, run.lhb_idx = _P(), True, object()
    run.lhb_last = run.lhb_last_rwin = 0
    run.reward_steps = []
    k = {"i": 0}

    def steps(n, stim=(), arousal=False, extra=(), da=False, record=()):
        assert n == 1 and record == ("lhb",)
        names = [e[0] for e in extra]
        if "reward_us" in names:
            run.reward_steps.append(k["i"])
        k["i"] += 1
        c = rate_per_step_reward if "reward_us" in names else rate_per_step_free
        return {"lhb": c}, [0.5]

    run.steps = steps
    return run


@pytest.fixture
def timing(monkeypatch):
    monkeypatch.setattr(pilot, "OUTCOME_MS", 250)
    monkeypatch.setattr(pilot, "REWARD_ON_MS", 150)
    monkeypatch.setattr(pilot, "REWARD_MS", 50)


def _trial(rewarded, free, rew):
    run = _stub_run(free, rew)
    das = run.outcome(rewarded)
    assert len(das) == 250
    return run, {"rewarded": rewarded, "lhb_spikes": run.lhb_last, "lhb_spikes_rewardwin": run.lhb_last_rwin}


def test_reward_window_is_the_same_steps_on_every_trial(timing):
    run_r, rew = _trial(True, 2, 0)
    run_u, unrew = _trial(False, 2, 0)
    assert run_r.reward_steps == list(range(150, 200))      # reward_us only inside the window
    assert run_u.reward_steps == []
    assert (unrew["lhb_spikes"], unrew["lhb_spikes_rewardwin"]) == (500, 100)   # expected reward time still counted
    assert (rew["lhb_spikes"], rew["lhb_spikes_rewardwin"]) == (400, 0)


def test_whole_window_ratio_is_capped_and_reward_window_is_not(timing):
    _, rew = _trial(True, 2, 0)          # total silencing during the reward
    _, unrew = _trial(False, 2, 0)
    trials = [rew, unrew]
    r, u = pilot.lhb_window_rates(trials, "lhb_spikes", 20, 250)
    assert u[0] / r[0] == pytest.approx(1.25)               # the registered bar (> 2x) is unreachable here
    assert pilot.engagement_a(r[0], u[0])["pass"] is False
    r, u = pilot.lhb_window_rates(trials, "lhb_spikes_rewardwin", 20, 50)
    assert (r[0], u[0]) == (0.0, 100.0)                      # 2 spikes/step / 20 neurons / 1 ms = 100 Hz
    assert pilot.engagement_a(r[0], u[0]) == {"ratio": None, "factor": 2.0, "pass": True}


def test_weak_reward_suppression_still_fails_the_amended_bar(timing):
    _, rew = _trial(True, 6, 5)          # ~17% suppression during the reward (round-1 release arm)
    _, unrew = _trial(False, 6, 5)
    r, u = pilot.lhb_window_rates([rew, unrew], "lhb_spikes_rewardwin", 20, 50)
    out = pilot.engagement_a(r[0], u[0])
    assert out["pass"] is False and out["ratio"] == pytest.approx(1.2)


def test_engagement_undefined_without_an_omission_response():
    assert pilot.engagement_a(None, 10.0)["pass"] is None
    assert pilot.engagement_a(0.0, 0.0)["pass"] is None
    assert pilot.lhb_window_rates([{"rewarded": True, "k": 3}], "k", 20, 50, present=False) == ([], [])
