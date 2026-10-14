"""Long-delay temporal credit on the production basal-ganglia organ (plan 2026-09-25 section 4).

COPY (STEP4B step 2, docs/drafts/long-delay-credit/STEP4B.md): research/runners/_ldc_omission_pilot.py is
_ldc_outcome_timing_pilot.py importing from _ldc_credit_omission, so every CreditParams omission flag lands on the
command line automatically and the lr0 declaration (RULING T1-lr0) rides along. Per run it additionally records
`lhb_rate_rewarded` / `lhb_rate_unrewarded` (LHb mean rate in the outcome window, measured on TRAIN trials because
reward_us is only driven while learning) and `membrane_ok` (tools.lab.membrane_bound_check on lhb, lhb_inh, rmtg and
snc after training; route-off runs have no LHb regions, so those rates read None there).

INSTRUMENT COPY (RULING ldc-d2-floor 2026-10-09, T1): _ldc_dip_observability.py is _ldc_omission_pilot.py plus
read-only observability for (a) which afferent still drives the omission dip in the cut arm and (b) when the
corticostriatal weights hit the floor and how. Adds: per-ms spike counts of the candidate dip drivers (lhb,
lhb_inh, rmtg, snc, str_* per channel - whatever the organ has) alongside the per-ms DA trace during OUTCOME_MS,
stashed per trial as info["_obs"] (full [DA + per-region spike trains + reward-window flags] kept for the first
40 omission trials, compact min/argmin/max + totals after that); and per-trial corticostriatal weight snapshots
info["wt"] = {pathway: [mean, floor_frac]} on every TRAIN trial, floor_frac = fraction of synapses at
<= 0.05 * that pathway's initial mean (the runner's own floor_ceiling test, per synapse instead of per mean).
No RNG draw, no weight/eligibility/DA write: dynamics are byte-identical to _ldc_omission_pilot.py.

AMENDMENT A0.1 (2026-10-06, docs/drafts/long-delay-credit/AMENDMENT-A0.1.md): engagement bar (a) is read in the
REWARD-TIME window [REWARD_ON_MS, REWARD_ON_MS + REWARD_MS) of the outcome period, the same steps on every trial (the
reward period on rewarded trials, the expected reward time on unrewarded ones): `lhb_rate_rewardwin_rewarded` /
`lhb_rate_rewardwin_unrewarded` and `lhb_engagement_a_rewardwin`. The whole-window rates above stay as recorded
(diagnostic); with a 50 ms pulse in a 250 ms window they cannot exceed 250/200 = 1.25 whatever the LHb does.

Preregistration: research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md (schedules, arms, bars, the
event-decomposition instrument, what may be tuned). The substrate is the production Gate-A v2 selector plus the
default-off `BRAIN_BG_CREDIT` extension (research/runners/bg_corticostriatal_credit.py); this runner turns the flag on
for its own process and never edits a weight, an eligibility or a dopamine value.

Every trial: trial-end clear -> ITI -> EARLY race on a cue (ctx_a / ctx_b) -> delay (base 250 ms x 0.5..8) with 1-2
distractor races -> DECOY race (ctx_n, or the other cue) -> TERMINAL race (ctx_t, identical every trial) ->
[late_delayed: one more distractor race] -> OUTCOME screen (ctx_o) with the reward 30-80 ms into it if earned.
A race = the organ's own: shared arousal on, the stimulus pool on, the first motor pool to cross the commit threshold
with a clean loser is the commit (the body's read), then the organ's reset + washout. In the immediate schedules the
outcome starts at the commit while the race inputs stay on, then the reset follows.

    python -m research.runners.long_delay_credit_bg --schedule credit --arms full,trace_ablated_integral \
        --seeds 7 --out research/findings/raw/_long_delay_credit_bg/dev/credit_s7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._ldc_credit_omission import (  # noqa: E402
    CreditParams, MSN, Synapses, baseline_currents, build_credit_selector, calibrate_da, drive,
)
from research.runners._vocal_action_selector_gate import CHANNELS, _indices, _step  # noqa: E402
from tools.lab import attributable_to, membrane_bound_check  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

# hashed at import: this runner AND the extension module that holds the circuit constants (the source that ran)
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_SHA256 = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()
                               + open(os.path.join(_HERE, "_ldc_credit_omission.py"), "rb").read()).hexdigest()

BASE_DELAY_MS = 250
DELAY_MULTS = (0.5, 1.0, 2.0, 4.0, 8.0)
CLEAR_MS = 250                     # trial-end event: > 2 tau_NMDA, else the held cue re-ignites (D3 slot)
OBS_FULL_CAP = 40                  # instrument: full per-ms outcome traces kept for this many omission trials
_OBS_FULL = {"n": 0}
ITI_QUIET_MS = (300, 450)
OUTCOME_MS = 100
REWARD_ON_MS, REWARD_MS = 30, 50
POST_TERM_MS = 50
LATE_OUTCOME_GAP_MS = (300, 800)   # late_delayed: terminal commit -> outcome onset
N_TRAIN, N_TEST, N_PROBE = 400, 200, 40
CHANCE = 0.5
CORRECT = {0: 0, 1: 1}             # cue A -> channel 0, cue B -> channel 1

SCHEDULES = {
    "credit": dict(relevant="early", immediate=False),
    "credit_immediate": dict(relevant="early", immediate=True),
    "late_delayed": dict(relevant="terminal", immediate=False, post_race=True),
    "late_immediate": dict(relevant="terminal", immediate=True),
    "sanity_early_ch0": dict(relevant="early", immediate=True, fixed_target=0),
    "sanity_terminal_ch0": dict(relevant="terminal", immediate=True, fixed_target=0),
}
ARMS = {
    "full": {},
    "trace_ablated_integral": {"tau": 20.0, "gain": 50.0},
    "trace_ablated_peak": {"tau": 20.0, "gain": 1.0},
    "shuffled_yoked": {"yoked": True},
    "misleading_final_cue": {"misleading": True},
    "memory_lesioned": {"lesion": (("mem_a", "mem_a"), ("mem_b", "mem_b"))},
    "critic_lesioned": {"lesion": (("striosome_value", "snc"),)},
    "reward_to_da_lesioned": {"lesion": (("reward_us", "snc"),)},
    # native syn-zero D2-output lesion (RULING a2-syn-zero-approved): the indirect pathway's
    # output synapses zeroed at build, held at test via lesion_weight_max_at_test (self-proving).
    "d2_output_lesioned": {"lesion": (("str_d2_0", "gpe_0"), ("str_d2_1", "gpe_1"))},
}


def arm_config(arm):
    """Named arms, plus the dose cells `dose_m<mult>_tau<ms>` (fixed delay multiplier, equal kernel integral against
    tau = 1000 ms)."""
    if arm in ARMS:
        return dict(ARMS[arm])
    if arm.startswith("dose_m"):
        m, tau = arm[len("dose_m"):].split("_tau")
        return {"fixed_mult": float(m), "tau": float(tau), "gain": 1000.0 / float(tau)}
    raise KeyError(arm)


def _git_state():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--",
                                              "research/runners/_ldc_omission_pilot.py",
                                              "research/runners/_ldc_credit_omission.py"], text=True).strip())
    except Exception:
        head, dirty = None, None
    return head, dirty


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# the world's schedule
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_trial(rng, acfg):
    mult = float(acfg["fixed_mult"]) if "fixed_mult" in acfg else float(rng.choice(DELAY_MULTS))
    delay = int(round(BASE_DELAY_MS * mult))
    cue = int(rng.integers(0, 2))
    n_d = 1 if delay < 500 else int(rng.integers(1, 3))
    # distractor races at fractions of the quiet delay (the delay counts quiet time; races add their own time)
    cuts = sorted(float(x) for x in rng.uniform(0.15, 0.85, size=n_d))
    distractors = [(int(round(delay * c)), str(rng.choice(["ctx_x", "ctx_y"]))) for c in cuts]
    decoy = ("ctx_b" if cue == 0 else "ctx_a") if acfg.get("misleading") else "ctx_n"
    return {"cue": cue, "delay": delay, "distractors": distractors, "decoy": decoy,
            "iti": int(rng.integers(ITI_QUIET_MS[0], ITI_QUIET_MS[1] + 1)),
            "late_gap": int(rng.integers(LATE_OUTCOME_GAP_MS[0], LATE_OUTCOME_GAP_MS[1] + 1)),
            "late_distractor": str(rng.choice(["ctx_x", "ctx_y"]))}


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# the brain run: the organ's race, the outcome period, the event-decomposition ledger
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Run:
    last_out: dict | None = None  # instrument copy: last outcome trace (set by outcome(), read by run_trial)
    def __init__(self, bridge, config, p: CreditParams, learn: bool):
        self.b, self.config, self.p = bridge, config, p
        self.learn = learn
        self.xp, _ = get_backend()
        self.syn = Synapses(bridge)
        self.motor = {ch: _indices(bridge, "motor_%d" % ch) for ch in CHANNELS}
        self.mem = {m: _indices(bridge, m) for m in ("mem_a", "mem_b")} if p.with_memory else {}
        names = {r.name for r in bridge.region_manager.regions()}
        self.lhb_idx = _indices(bridge, "lhb") if "lhb" in names else None
        self.lhb_last = 0                      # LHb spikes in the last outcome window (route-on builds only)
        self.lhb_last_rwin = 0                 # ... of which inside the reward-time window (AMENDMENT A0.1)
        self.t = 0                              # steps since the run began (the ledger's clock)
        self.marks = []                          # (label, t, eligibility of the plastic synapses)
        pos = {int(s): i for i, s in enumerate(self.syn.plastic_idx)}
        self.pos_of = {}                         # (pre, post) -> positions inside plastic_idx
        for a in ("ctx_a", "ctx_b", "mem_a", "mem_b"):
            for b in MSN:
                self.pos_of[(a, b)] = np.asarray([pos[int(x)] for x in self.syn.mask(a, b) if int(x) in pos],
                                                 dtype=np.int64)

    # -- stepping -------------------------------------------------------------------------------------------
    def _currents(self, stim=(), arousal=False, extra=()):
        baseline_currents(self.b, self.config, self.p)
        for region in stim:
            drive(self.b, region, self.p.ctx_pA)
        if arousal:
            drive(self.b, "practice_arousal", self.config.practice_pA)
        for region, pA in extra:
            drive(self.b, region, pA)

    def steps(self, n, stim=(), arousal=False, extra=(), record=(), da=False):
        self._currents(stim, arousal, extra)
        out = {r: 0 for r in record}
        das = []
        idx = {r: _indices(self.b, r) for r in record}
        nm = self.b.neuromodulator_manager
        for _ in range(int(n)):
            _step(self.b)
            self.t += 1
            if record:
                f = np.asarray(to_host(self.b.cp_firing_states), dtype=bool)
                for r in record:
                    out[r] += int(f[idx[r]].sum())
            if da:
                das.append(float(nm.get_concentration("dopamine")))
        return out, das

    def mark(self, label):
        e = self.syn.eligibility()[self.syn.plastic_idx]
        self.marks.append((label, self.t, e))

    # -- the organ's race ----------------------------------------------------------------------------------
    def race(self, stim, *, reward_of=None):
        """One race on `stim`. With `reward_of` (immediate schedules) the outcome starts at the commit while the race
        inputs stay on; returns (winner or None, decision step, rewarded or None, DA trace of the outcome)."""
        cfgs = self.config
        self._currents((stim,), arousal=True)
        counts = np.zeros(2, dtype=np.int64)
        first, simult, dstep = None, False, None
        for step in range(int(cfgs.action_steps)):
            _step(self.b)
            self.t += 1
            f = np.asarray(to_host(self.b.cp_firing_states), dtype=bool)
            prev = counts.copy()
            for ch in CHANNELS:
                counts[ch] += int(f[self.motor[ch]].sum())
            crossed = [ch for ch in CHANNELS if prev[ch] < cfgs.commit_threshold_spikes <= counts[ch]]
            if first is None and len(crossed) == 1:
                first, dstep = int(crossed[0]), int(step)
                break
            if first is None and len(crossed) > 1:
                simult, dstep = True, int(step)
                break
        winner = None
        if first is not None and not simult:
            loser = 1 - first
            if counts[loser] / max(1, counts[first]) <= cfgs.clean_loser_ratio:
                winner = first
        rewarded, das = None, []
        if reward_of is not None:
            rewarded = bool(reward_of(winner))
            self.mark("outcome")
            das = self.outcome(rewarded, stim=(stim,), arousal=True)
        self._currents(extra=(("selector_reset", cfgs.reset_pA),))
        for _ in range(int(cfgs.reset_steps)):
            _step(self.b); self.t += 1
        self._currents()
        for _ in range(int(cfgs.washout_steps)):
            _step(self.b); self.t += 1
        return winner, dstep, rewarded, das

    def outcome(self, rewarded, stim=(), arousal=False):
        """The outcome period: the feedback screen ctx_o for OUTCOME_MS; reward_us 30-80 ms into it if earned (and
        only while learning). Returns the DA trace; also records the LHb spike count of the window.
        Instrument copy: per-ms spike counts of every candidate dip-driver region the organ has are accumulated
        beside the DA trace and stashed on self.last_out for run_trial (read-only; dynamics unchanged)."""
        das = []
        names = {r.name for r in self.b.region_manager.regions()}
        cand = ("lhb", "lhb_inh", "rmtg", "snc", "str_d1_0", "str_d1_1", "str_d2_0", "str_d2_1", "gpe_0", "gpe_1")
        rec = tuple(r for r in cand if r in names)
        trace = {r: [] for r in rec}
        rwin_f = []
        total = rwin = 0
        for k in range(OUTCOME_MS):
            in_rwin = REWARD_ON_MS <= k < REWARD_ON_MS + REWARD_MS     # reward time, rewarded or not (A0.1)
            extra = [("ctx_o", self.p.ctx_pA)]
            if rewarded and self.learn and in_rwin:
                extra.append(("reward_us", self.p.rus_pA))
            o, d = self.steps(1, stim=stim, arousal=arousal, extra=extra, da=True, record=rec)
            n = int(o.get("lhb", 0))
            total += n
            rwin += n if in_rwin else 0
            das += d
            for r in rec:
                trace[r].append(int(o.get(r, 0)))
            rwin_f.append(1 if in_rwin else 0)
        self.lhb_last = total
        self.lhb_last_rwin = rwin
        self.last_out = {"da": [float(x) for x in das], "spk": trace, "rwin": rwin_f,
                         "rewarded": bool(rewarded)}
        return das


def run_trial(run: Run, tr, schedule, acfg, reward_rng, yoked_reward=None):
    sch = SCHEDULES[schedule]
    run.marks = []
    run.lhb_last = 0
    run.lhb_last_rwin = 0
    run.last_out = None
    info = {"cue": tr["cue"], "delay": tr["delay"]}
    target = sch.get("fixed_target")

    def correct_for(ch):
        return (ch == target) if target is not None else (ch == CORRECT[tr["cue"]])

    def reward_of(ch):
        if yoked_reward is not None:
            return bool(yoked_reward)
        return ch is not None and correct_for(ch)

    cue_name = "ctx_a" if tr["cue"] == 0 else "ctx_b"
    # trial-end event (clears working memory), then the quiet ITI
    run.mark("carry")
    run.steps(CLEAR_MS, extra=(("mem_clear", run.p.clear_pA),) if run.p.with_memory else ())
    rec_mem = ("mem_a", "mem_b") if run.p.with_memory else ()
    after_clear, iti_da = run.steps(tr["iti"], record=rec_mem, da=True)
    info["mem_after_clear_hz"] = (sum(after_clear.values()) / (2 * run.p.n_mem) / (tr["iti"] / 1000.0)
                                  if rec_mem else None)
    info["da_iti_mean"] = float(np.mean(iti_da)) if iti_da else None
    # EARLY race
    run.mark("early")
    immediate_early = sch["immediate"] and sch["relevant"] == "early"
    e_win, e_step, e_rew, e_da = run.race(cue_name, reward_of=reward_of if immediate_early else None)
    info.update(early=-1 if e_win is None else e_win, t_early=e_step)
    # DELAY: quiet segments with distractor races
    t_quiet = 0
    n_interv = 0
    for on, name in tr["distractors"]:
        run.mark("delay")
        run.steps(max(0, on - t_quiet))
        t_quiet = max(t_quiet, on)
        run.mark("interv")
        w, _, _, _ = run.race(name)
        n_interv += int(w is not None)
    run.mark("delay")
    tail = max(0, tr["delay"] - t_quiet)
    hold = {"mem_a": 0, "mem_b": 0}
    if tail > 50:
        run.steps(tail - 50)
        hold, _ = run.steps(50, record=rec_mem)
    else:
        hold, _ = run.steps(tail, record=rec_mem)
    info["n_intervening_commits"] = n_interv
    info["n_distractors"] = len(tr["distractors"])
    if rec_mem:
        info["mem_hold"] = 0 if hold["mem_a"] > hold["mem_b"] else 1 if hold["mem_b"] > hold["mem_a"] else -1
    # DECOY race, then TERMINAL race
    run.mark("decoy")
    d_win, _, _, _ = run.race(tr["decoy"])
    info["decoy_commit"] = -1 if d_win is None else d_win
    if rec_mem:   # which held cue survives the decoy (instrument: decoy resistance)
        after_decoy, _ = run.steps(20, record=rec_mem)
        info["mem_after_decoy"] = (0 if after_decoy["mem_a"] > after_decoy["mem_b"] else
                                   1 if after_decoy["mem_b"] > after_decoy["mem_a"] else -1)
    run.mark("terminal")
    immediate_term = sch["immediate"] and sch["relevant"] == "terminal"
    t_win, t_step, t_rew, t_da = run.race("ctx_t", reward_of=reward_of if immediate_term else None)
    info.update(terminal=-1 if t_win is None else t_win, t_terminal=t_step)
    # OUTCOME (delayed schedules)
    das = e_da or t_da
    rewarded = e_rew if immediate_early else t_rew if immediate_term else None
    if not sch["immediate"]:
        relevant = e_win if sch["relevant"] == "early" else t_win
        rewarded = reward_of(relevant)
        run.mark("post")
        if sch.get("post_race"):
            before = run.t
            run.race(tr["late_distractor"])
            run.steps(max(0, tr["late_gap"] - (run.t - before)))
        else:
            run.steps(POST_TERM_MS)
        run.mark("outcome")
        das = run.outcome(rewarded)
    obs = run.last_out          # instrument: stash the outcome trace on the trial row (read-only)
    run.last_out = None
    if obs is not None:
        da = obs["da"]
        rec_obs = {"rewarded": obs["rewarded"],
                   "da_min": float(min(da)) if da else None,
                   "da_max": float(max(da)) if da else None,
                   "da_argmin": int(min(range(len(da)), key=da.__getitem__)) if da else None,
                   "spk_tot": {r: int(sum(v)) for r, v in obs["spk"].items()}}
        if (not obs["rewarded"]) and _OBS_FULL["n"] < OBS_FULL_CAP:
            _OBS_FULL["n"] += 1
            rec_obs["da_full"] = da
            rec_obs["spk"] = obs["spk"]
            rec_obs["rwin"] = obs["rwin"]
        info["_obs"] = rec_obs
    run.mark("end")
    info["rewarded"] = bool(rewarded)
    info["lhb_spikes"] = int(run.lhb_last)
    info["lhb_spikes_rewardwin"] = int(run.lhb_last_rwin)
    info["correct_early"] = e_win is not None and correct_for(e_win)
    info["correct_terminal"] = t_win is not None and correct_for(t_win)
    info["da_outcome"] = (float(min(das)), float(max(das))) if das else None
    return info, das


def decompose(run: Run, das, p: CreditParams, tau_ms, sign_d2):
    """Exact split of the eligibility at outcome onset into per-window contributions, times the credited scalar
    A = lr * sum_t (DA_t - b) d^(t - t0) over the outcome. Returns {window: credited dw per plastic synapse}."""
    d = math.exp(-1.0 / float(tau_ms))
    labels = [m[0] for m in run.marks]
    if "outcome" not in labels:
        return None
    k_out = labels.index("outcome")
    t0 = run.marks[k_out][1]
    A = float(run.b.core_config.reward_learning_rate) * sum((x - p.da_baseline) * d ** i for i, x in enumerate(das))
    contrib = {}
    for k in range(k_out):
        label, t_a, e_a = run.marks[k]
        _, t_b, e_b = run.marks[k + 1]
        if k == 0:   # the eligibility present when this trial began (previous trials), decayed to t0
            contrib["carry"] = e_a * d ** (t0 - t_a)
            label = "iti"                      # the first window (trial-end clear + ITI) adds its own tags
        added = (e_b - e_a * d ** (t_b - t_a)) * d ** (t0 - t_b)
        contrib[label] = contrib.get(label, 0) + added
    return {w: A * sign_d2 * c for w, c in contrib.items()}


def mapping_credit(dw, run: Run, family):
    """Mapping-positive credited change for a pathway family ('cue': ctx_a/ctx_b, 'mem': mem_a/mem_b):
    [correct D1 - wrong D1] - [correct D2 - wrong D2], means over synapses (D2 dw already carries its sign)."""
    src = ("ctx_a", "ctx_b") if family == "cue" else ("mem_a", "mem_b")

    def mean_dw(a, b):
        idx = run.pos_of.get((a, b))
        return float(np.mean(dw[idx])) if idx is not None and idx.size else 0.0

    corr_d1 = mean_dw(src[0], "str_d1_0") + mean_dw(src[1], "str_d1_1")
    wrong_d1 = mean_dw(src[0], "str_d1_1") + mean_dw(src[1], "str_d1_0")
    corr_d2 = mean_dw(src[0], "str_d2_0") + mean_dw(src[1], "str_d2_1")
    wrong_d2 = mean_dw(src[0], "str_d2_1") + mean_dw(src[1], "str_d2_0")
    return (corr_d1 - wrong_d1) - (corr_d2 - wrong_d2)


def run_condition(schedule, arm, seed, *, n_train=N_TRAIN, n_test=N_TEST, n_probe=N_PROBE, p=CreditParams(),
                  yoke_sequence=None, verbose=False):
    acfg = arm_config(arm)
    if schedule in ("late_delayed", "late_immediate", "sanity_terminal_ch0") and not p.with_memory:
        raise ValueError("late schedules need the memory pair")
    t_start = time.time()
    tau = float(acfg.get("tau", p.elig_tau_ms))
    bridge, config = build_credit_selector(seed, p, elig_tau_ms=tau, tag_gain=float(acfg.get("gain", 1.0)))
    tonic = calibrate_da(bridge, config, p)
    syn = Synapses(bridge)
    lesions = {"%s->%s" % ab: syn.zero(*ab) for ab in acfg.get("lesion", ())}
    sign = np.asarray(to_host(bridge.cp_d1_d2_sign))[:syn.nnz][syn.plastic_idx].astype(np.float64)
    w_init = {"%s->%s" % (a, b): float(np.mean(syn.weights(a, b)))
              for a in ("ctx_a", "ctx_b", "ctx_x", "ctx_y", "ctx_n", "ctx_t", "mem_a", "mem_b") for b in MSN
              if syn.mask(a, b).size}
    world = np.random.default_rng(seed * 7919 + 11)      # the schedule
    coin = np.random.default_rng(seed * 104729 + 3)      # the yoked control's own stream (unused otherwise)
    cfg = bridge.core_config
    lr_on = float(cfg.reward_learning_rate)

    def block(n, learn, keep=False, yoke=None):
        cfg.reward_learning_rate = lr_on if learn else 0.0
        run = Run(bridge, config, p, learn)
        rows, dec = [], []
        for i in range(n):
            tr = make_trial(world, acfg)
            yr = None if yoke is None else yoke[i % len(yoke)]
            info, das = run_trial(run, tr, schedule, acfg, coin, yoked_reward=yr)
            if learn:   # instrument: per-trial corticostriatal weight trajectory (RULING ldc-d2-floor (b))
                wt = {}
                for k in w_init:
                    w = np.asarray(syn.weights(*k.split("->")))
                    if w.size == 0:
                        continue
                    wt[k] = [float(np.mean(w)), float((w <= 0.05 * w_init[k]).mean())]
                info["wt"] = wt
            if keep and learn:
                dw = decompose(run, das, p, tau, sign)
                if dw is not None:
                    dec.append({w: {"cue": mapping_credit(v, run, "cue"),
                                    "mem": mapping_credit(v, run, "mem") if p.with_memory else 0.0}
                                for w, v in dw.items()})
            rows.append(info)
            if verbose and learn and (i + 1) % 50 == 0:
                last = rows[-50:]
                sys.stderr.write("[%s/%s s%d] %d early %.2f term %.2f rew %.2f resp_e %.2f resp_t %.2f %.0fs\n" % (
                    schedule, arm, seed, i + 1, np.mean([x["correct_early"] for x in last]),
                    np.mean([x["correct_terminal"] for x in last]), np.mean([x["rewarded"] for x in last]),
                    np.mean([x["early"] >= 0 for x in last]), np.mean([x["terminal"] >= 0 for x in last]),
                    time.time() - t_start))
        return rows, dec

    probe, _ = block(n_probe, learn=False)
    yoke = yoke_sequence if acfg.get("yoked") else None
    if acfg.get("yoked") and not yoke:
        raise ValueError("shuffled_yoked needs the same seed's full-arm reward sequence (run 'full' first)")
    train, dec = block(n_train, learn=True, keep=True, yoke=yoke)
    test, _ = block(n_test, learn=False)
    cfg.reward_learning_rate = lr_on
    held = {k: float(np.abs(syn.weights(*k.split("->"))).max()) for k in lesions}
    assert all(v == 0.0 for v in held.values()), held
    assert float(cfg.current_reward_signal) == 0.0, "no host reward scalar: the SNc firing is the third factor"

    rel = SCHEDULES[schedule]["relevant"]
    # omission-route instrument (STEP4B step 2): LHb mean rate in the outcome window, split by trial outcome
    # (TRAIN trials only: reward_us is driven only while learning), and the membrane bounds after training
    names = {r.name for r in bridge.region_manager.regions()}
    rew_lhb, unrew_lhb = lhb_window_rates(train, "lhb_spikes", p.n_lhb, OUTCOME_MS, "lhb" in names)
    # AMENDMENT A0.1: bar (a) reads the reward-time window, where the reward is the only difference between trials
    rew_lhb_rw, unrew_lhb_rw = lhb_window_rates(train, "lhb_spikes_rewardwin", p.n_lhb, REWARD_MS, "lhb" in names)
    membrane_bad = []
    volt = np.asarray(to_host(bridge.cp_membrane_potential_v))
    for reg in ("lhb", "lhb_inh", "rmtg", "snc"):
        if reg not in names:
            continue
        if membrane_bound_check("omission pilot s%d %s" % (seed, reg), volt[_indices(bridge, reg)],
                                strict=False) is not True:
            membrane_bad.append(reg)

    def score(rows, key):
        resp = [x for x in rows if x[key] >= 0]
        return {"success": float(np.mean([x["correct_" + key] for x in rows])) if rows else None,
                "response_rate": len(resp) / max(1, len(rows)),
                "acc_among_responders": (float(np.mean([x["correct_" + key] for x in resp])) if resp else None),
                "chance": CHANCE}

    last = dec[-200:]

    def dsum(window, fam):
        return float(sum(r.get(window, {}).get(fam, 0.0) for r in last))

    windows = sorted({w for r in last for w in r})
    decomposition = {w: {"cue": dsum(w, "cue"), "mem": dsum(w, "mem")} for w in windows}
    m_early, m_decoy = dsum("early", "cue"), dsum("decoy", "cue")
    m_mem = sum(dsum(w, "mem") for w in windows)
    m_carry = dsum("carry", "cue") + dsum("carry", "mem")
    denom = abs(m_early) + abs(m_decoy) + abs(m_mem) + abs(m_carry)
    # ATTRIBUTION (gates/attribution_required): `early_share` already reports the FRACTION of the total credited
    # weight change attributable to the early causal event -- ask it explicitly too: treatment = the early-event
    # credit magnitude, control = the non-causal share (decoy + carryover) the SAME denominator sums. Diagnostic
    # only (out["early_vs_noncausal_attribution"]); does not feed early_share, G2, or any other registered gate.
    early_vs_noncausal = (attributable_to("%s/%s s%d early cue credit vs non-causal (decoy+carry)"
                                          % (schedule, arm, seed), abs(m_early), abs(m_decoy) + abs(m_carry))
                          if denom > 0 else None)
    w_final = {k: float(np.mean(syn.weights(*k.split("->")))) for k in w_init}
    floor_ceiling = {k: (v <= 0.05 * w_init[k] or v >= 0.95 * p.w_max) for k, v in w_final.items()}

    def cue_split(rows, ch):
        a = [x["early"] for x in rows if x["cue"] == ch and x["early"] >= 0]
        return float(np.mean([e == 0 for e in a])) if a else None

    probe_resp = [x for x in probe if x["early"] >= 0 and x["terminal"] >= 0]
    out = {
        "schedule": schedule, "arm": arm, "seed": seed, "n_train": n_train, "n_test": n_test,
        "relevant": rel, "test": {"early": score(test, "early"), "terminal": score(test, "terminal")},
        "success": score(test, rel)["success"], "response_rate": score(test, rel)["response_rate"],
        "chance": CHANCE,
        "mem_hold_acc": (float(np.mean([x["mem_hold"] == x["cue"] for x in test if x.get("mem_hold", -1) >= 0]))
                         if any(x.get("mem_hold", -1) >= 0 for x in test) else None),
        "decomposition_last200": decomposition,
        "M_early": m_early, "M_decoy": m_decoy, "M_mem": m_mem, "M_term_mem": dsum("terminal", "mem"),
        "M_carry": m_carry, "early_share": (m_early / denom) if denom > 0 else None,
        "early_vs_noncausal_attribution": early_vs_noncausal,
        "instrument": {
            "mem_after_clear_hz": _mean([x["mem_after_clear_hz"] for x in probe if x.get("mem_after_clear_hz")
                                         is not None]),
            "mem_hold_acc_probe": _mean([x["mem_hold"] == x["cue"] for x in probe if x.get("mem_hold", -1) >= 0]),
            "mem_after_decoy_acc_probe": _mean([x["mem_after_decoy"] == x["cue"] for x in probe
                                                if x.get("mem_after_decoy", -1) >= 0]),
            "da_iti_offset_probe": (_mean([x["da_iti_mean"] for x in probe if x.get("da_iti_mean") is not None])
                                    - p.da_baseline) if probe else None,
            "interv_commit_rate_probe": _mean([x["n_intervening_commits"] / max(1, x["n_distractors"])
                                               for x in probe]),
            "decoy_commit_rate_probe": _mean([x["decoy_commit"] >= 0 for x in probe]),
        },
        "probe_lr0": {"p_ch0_given_A": cue_split(probe, 0), "p_ch0_given_B": cue_split(probe, 1),
                      "p_terminal_eq_early": (float(np.mean([x["early"] == x["terminal"] for x in probe_resp]))
                                              if probe_resp else None),
                      "early_commit_rate": float(np.mean([x["early"] >= 0 for x in probe])),
                      "terminal_commit_rate": float(np.mean([x["terminal"] >= 0 for x in probe]))},
        "train_reward_rate_last100": float(np.mean([x["rewarded"] for x in train[-100:]])),
        "train_reward_sequence": [bool(x["rewarded"]) for x in train],
        "da_outcome_rewarded_max_mean": _mean([x["da_outcome"][1] for x in train[-100:]
                                               if x["rewarded"] and x["da_outcome"]]),
        "da_outcome_unrewarded_min_mean": _mean([x["da_outcome"][0] for x in train[-100:]
                                                 if not x["rewarded"] and x["da_outcome"]]),
        "weights_initial": w_init, "weights_final": w_final, "floor_or_ceiling": floor_ceiling,
        "void_floor_ceiling": any(floor_ceiling.values()),
        "lhb_rate_rewarded": _mean(rew_lhb), "lhb_rate_unrewarded": _mean(unrew_lhb),
        "lhb_rewardwin_ms": [REWARD_ON_MS, REWARD_ON_MS + REWARD_MS],
        "lhb_rate_rewardwin_rewarded": _mean(rew_lhb_rw), "lhb_rate_rewardwin_unrewarded": _mean(unrew_lhb_rw),
        "lhb_engagement_a_rewardwin": engagement_a(_mean(rew_lhb_rw), _mean(unrew_lhb_rw)),
        "membrane_ok": not membrane_bad, "membrane_bad_regions": membrane_bad,
        "snc_tonic_fraction": tonic, "lesions": lesions, "lesion_weight_max_at_test": held,
        "elig_tau_ms": tau, "tag_gain": float(acfg.get("gain", 1.0)),
        "host_reward_signal": float(cfg.current_reward_signal), "wall_s": round(time.time() - t_start, 1),
        # instrument copy (RULING ldc-d2-floor): the per-trial info dicts are otherwise score-only, so carry
        # the trajectory (info["wt"] = {pathway: [mean, floor_frac]} on TRAIN trials) and the outcome-window
        # traces (info["_obs"], read-only instrumentation of Run.outcome) into the cell record.
        "weights_per_trial": [x.get("wt") for x in train],
        "obs": [x.get("_obs") for x in train],
    }
    return out


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def lhb_window_rates(trials, key, n_lhb, window_ms, present=True):
    """Per-trial LHb mean rate (Hz per neuron) from the spike count `key` over a `window_ms` window, split into
    (rewarded, unrewarded) trials. Empty lists when the build has no LHb."""
    if not present:
        return [], []
    scale = n_lhb * (window_ms / 1000.0)
    rew = [x[key] / scale for x in trials if x.get("rewarded")]
    unrew = [x[key] / scale for x in trials if not x.get("rewarded")]
    return rew, unrew


def engagement_a(rate_rewarded, rate_unrewarded, factor=2.0):
    """STEP4B engagement bar (a) as amended by A0.1: the reward-time-window LHb rate on unrewarded trials exceeds
    `factor` x the rate on rewarded trials. UNDEFINED (pass None) when either side has no trials or the LHb is silent
    on unrewarded trials (no omission response to compare)."""
    if rate_rewarded is None or rate_unrewarded is None or rate_unrewarded <= 0.0:
        return {"ratio": None, "factor": factor, "pass": None}
    ratio = (rate_unrewarded / rate_rewarded) if rate_rewarded > 0.0 else None
    return {"ratio": ratio, "factor": factor, "pass": bool(rate_unrewarded > factor * rate_rewarded)}


# LEARNING-OFF CELLS (RULING T1-lr0, cloud Claude session 7, 2026-10-05). With `--lr 0` the weights never move, so the
# cue->choice map is the seed's untrained one and `success` sits at chance by design: the 27 Step 4b.1 clear-dose
# cells read acc_among_responders 0.52 on average (9 seeds x 3 candidates; per-cell spread close to the binomial 0.05
# of n_test 100), with no bias below 0.5. Each per-race score dict holds two arms the below-chance gate reads, and
# success <= acc_among_responders by construction (a non-response scores as wrong; `response_rate` is a rate, which
# the gate does not count), so a chance-level cell reads "every arm below 0.5" whenever its accuracy draw lands under
# 0.5, about half the time. The behaviour of such a cell is not a measurement: the runner says so in the artifact
# (tools/gates/below_chance.py accepts an UNDEFINED verdict string) and only the instrument fields are read.
LR0_BEHAVIOR_VERDICT = ("UNDEFINED (lr 0 diagnostic cell): the weights never learn, so behaviour sits at chance by "
                        "design and reads below 0.5 on about half the cells by sampling noise alone; not a test of "
                        "the hypothesis. Read only the instrument fields (mem_after_clear_hz, mem_hold_acc, ...).")


def lr0_declaration(params):
    """The below-chance declaration for a learning-off cell (`params.lr == 0`), or None for a learning cell, whose
    behaviour IS the measurement and is never declared."""
    lr = params.get("lr") if isinstance(params, dict) else getattr(params, "lr", None)
    try:
        lr = float(lr)
    except (TypeError, ValueError):
        return None
    if lr != 0.0:
        return None
    return {"behavior_verdict": LR0_BEHAVIOR_VERDICT, "behavior_undefined_reason": "lr0-diagnostic"}


def declare_lr0(paths):
    """`--declare-lr0 FILE...`: add the same declaration to records this runner wrote before the declaration existed
    (Step 4b.1, 27 cells at 50f52c23e). Refuses any record whose params.lr is not 0 or that has no rows. Records the
    declaring runner's sha256 and the time beside the declaration; changes no measured value. Returns an exit code."""
    import datetime
    rc = 0
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        decl = lr0_declaration(doc.get("params") or {})
        if decl is None or not doc.get("rows"):
            sys.stderr.write("LDCBG_DECLARE_REFUSED %s: params.lr is not 0 or the record has no rows (a learning "
                             "cell's behaviour is a measurement and is never declared)\n" % path)
            rc = 2
            continue
        if all(doc.get(k) == v for k, v in decl.items()):
            sys.stderr.write("LDCBG_DECLARE_ALREADY %s\n" % path)
            continue
        doc.update(decl)
        doc["declared_after_run"] = {
            "by": "research/runners/_ldc_omission_pilot.py --declare-lr0",
            "declaring_runner_sha256": RUNNER_SHA256,
            "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fields": sorted(decl)}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, default=str)
        os.replace(tmp, path)
        sys.stderr.write("LDCBG_DECLARED %s\n" % path)
    return rc


def main(argv=None):
    global OUTCOME_MS, REWARD_ON_MS, CLEAR_MS
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--declare-lr0"]:
        if len(argv) < 2:
            sys.stderr.write("usage: --declare-lr0 RECORD.json [RECORD.json ...]\n")
            return 2
        return declare_lr0(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), required=True)
    ap.add_argument("--arms", required=True, help="comma list; shuffled_yoked must follow full")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-probe", type=int, default=N_PROBE)
    # Step 4a pilot options (HANDOFF Step 4a.1): defaults unchanged, so with no flags this
    # reproduces the registered runner exactly.
    ap.add_argument("--outcome-ms", type=int, default=OUTCOME_MS,
                    help="outcome screen duration, ms (default: %(default)s)")
    ap.add_argument("--reward-on-ms", type=int, default=REWARD_ON_MS,
                    help="reward onset into the outcome screen, ms (default: %(default)s)")
    ap.add_argument("--clear-ms", type=int, default=CLEAR_MS,
                    help="trial-end clear duration, ms (HANDOFF 4a.3 knob; default: %(default)s)")
    ap.add_argument("--param-json", default=None, help="DEV ONLY (instrument tuning, prereg section 7)")
    # one flag per CreditParams field (DEV ONLY, prereg section 7): tools/sweep_pool.sh appends extra axes as
    # `--<axis> <value>`, so both the dashed and the underscored spelling are accepted.
    from dataclasses import fields
    for f in fields(CreditParams):
        kind = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__)
        if f.name == "with_memory":
            continue
        ap.add_argument("--" + f.name.replace("_", "-"), "--" + f.name, dest="p_" + f.name, type=kind or float,
                        default=None)
    ap.add_argument("--out", "--json", dest="out", required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if not 0 <= a.reward_on_ms < a.outcome_ms:
        ap.error("--reward-on-ms must lie inside [0, --outcome-ms)")
    if a.reward_on_ms + REWARD_MS > a.outcome_ms:
        ap.error("--reward-on-ms + REWARD_MS must fit inside --outcome-ms")
    if a.clear_ms <= 0:
        ap.error("--clear-ms must be positive")
    OUTCOME_MS, REWARD_ON_MS, CLEAR_MS = a.outcome_ms, a.reward_on_ms, a.clear_ms
    os.environ["BRAIN_BG_CREDIT"] = "1"      # the benchmark turns the default-off extension on for this process
    from dataclasses import replace
    p = CreditParams()
    if a.param_json:
        p = replace(p, **json.loads(a.param_json))
    over = {k[2:]: v for k, v in vars(a).items() if k.startswith("p_") and v is not None}
    if over:
        p = replace(p, **over)
    head, dirty = _git_state()
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        full_seq = None
        for arm in a.arms.split(","):
            r = run_condition(a.schedule, arm, seed, n_train=a.n_train, n_test=a.n_test, n_probe=a.n_probe, p=p,
                              yoke_sequence=full_seq, verbose=a.verbose)
            if arm == "full":
                full_seq = r["train_reward_sequence"]
            rows.append(r)
            sys.stderr.write("LDCBG_RESULT %s\n" % json.dumps({k: r[k] for k in (
                "schedule", "arm", "seed", "success", "response_rate", "early_share", "mem_hold_acc")}))
    doc = {"runner_sha256": RUNNER_SHA256, "git_head": head, "git_dirty": dirty,
           "params": p.__dict__, "schedule": a.schedule, "rows": rows}
    doc.update(lr0_declaration(p.__dict__) or {})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    sys.stderr.write("LDCBG_WROTE %s\n" % a.out)


if __name__ == "__main__":
    sys.exit(main() or 0)
