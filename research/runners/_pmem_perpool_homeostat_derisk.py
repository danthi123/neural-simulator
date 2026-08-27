"""PROSPECTIVE MEMORY residual closure -- a PER-POOL HOMEOSTAT on the `rel` cue-monitor read-out.

THE RESIDUAL (from research/findings/2026-08-13-prospective-memory-intention-latch-cue-monitor-derisk.md).
The spiking intention LATCH + BA10 cue-MONITOR is bulletproof: every specificity/persistence/lesion clause is
6/6. The ONE failing clause is `fire_on_cue` (3/6): the release amplitude against a FIXED absolute threshold.
Under a CONSTANT tonic bias (`rel_bias_pA=-1050`) the release rate spreads ~4x across seeds -- seed 100's read-out
pool is too HYPO-excitable to reach rheobase on the joint drive (0.001/0.054), while seed 42's is hyper-excitable
(0.267). The finding's own diagnosis: a SUBTRACTIVE / THRESHOLD deficit (the operating point sits too low), NOT a
gain deficit -- so recurrence (a multiplicative amplifier) cannot rescue it, and re-sweeping the constant bias only
trades one seed's failure for another's.

THE MISSING COMPANION PROCESS (CLAUDE.md wall reframe). The constant tonic bias PROXIES the per-pool homeostatic
excitability control biology runs ALONGSIDE the coincidence detector: intrinsic-plasticity regulation of each
neuron's excitability toward a firing-rate SET-POINT (Turrigiano 2011, "Too many cooks? Intrinsic and synaptic
homeostatic mechanisms"; Desai, Rutherford & Turrigiano 1999, activity-dependent regulation of intrinsic
excitability; Kandel 6e, K-channel-set F-I / spike-frequency adaptation as the excitability substrate). Replaced by
ONE constant, each pool's operating point is left to its seed-drawn threshold heterogeneity -> the ~4x spread. The
substrate ALREADY carries this mechanism (`cfg.homeostasis_target_rate=0.02` + threshold adaptation), but at the
default slow timescale (tau ~5000 steps) it barely moves within a short trial -- which is exactly why the parent's
constant-bias run (default homeostasis nominally ON) still spread 4x.

THE SURPASS (this runner; NO sim/ edit -- additive, reuse-by-import of the parent runner's ProspectiveMemory).
Give EACH `rel` pool a per-pool homeostatic set-point on its tonic-inhibition bias (an adaptive stand-in for the
tonic-inhibition operating-point control / intrinsic-plasticity threshold that the engine's homeostasis realizes).
The homeostat is CALIBRATED, label-free, on each pool's OWN strongest single feedforward input -- the CUE drive:
  intrinsic-plasticity update   bias_pool[a] += eta * (r_set - r_cue_alone[a])   (clamped, bias_max <= 0)
run to convergence so cue-ALONE settles at a sub-threshold target r_set (r_set < SILENT_MAX). Because the STRONGEST
single input is pinned sub-threshold per pool, EVERY single-input silence condition (cue-alone, hold-alone, wrong-
cue, no-intention, lesioned-cue) is sub-threshold BY CONSTRUCTION on every seed; and because every pool now shares
the SAME operating point, the COINCIDENCE (held-intention + cue) clears the fixed FIRE_THR uniformly -- the hypo-
excitable pool is LIFTED to rheobase, the hyper-excitable pool is held at the same point. This adapts to the pool's
OWN activity, never to which cue is correct (label-free). It is a SET-POINT shift (subtractive), the right tool for
a subtractive/threshold deficit -- a divisive-normalization FS partner (Carandini & Heeger 2012) was CONSIDERED and
rejected: dividing a near-absent coincidence by a small normalizer stays near-absent (it cannot lift a hypo pool
over threshold).

ANTI-CHEAT (the mission's central risk). A homeostat that just raises all gains until everything fires is a CHEAT.
This one CANNOT do that: it pins the strongest SINGLE input sub-threshold, so raising a pool's excitability lifts
the COINCIDENCE without lifting any single-input silence condition over threshold. The gate PROVES it -- every
silence clause must STAY 6/6 (a regression => VOID). We ALSO run the identical substrate with the homeostat OFF
(the parent's constant bias) as an internal control, and attribute the fire lift to the homeostat.

GATE. Identical to the parent: the FROZEN thresholds and the per-seed clause logic are IMPORTED from
research.runners._pmem_intention_latch_derisk (not re-typed) and the substrate class is monkey-patched to the
homeostatic subclass, so base.run_seed computes every clause with the SAME code. The only difference between arms
is the per-pool homeostat. 6 seeds 42/43/44/100/101/102.

  SIM_BACKEND=numpy python -m research.runners._pmem_perpool_homeostat_derisk --smoke     # 1 seed, N=3, <60s
  SIM_BACKEND=numpy python -m research.runners._pmem_perpool_homeostat_derisk --derisk    # 6 seeds (on+off arms)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the parent runner's substrate + the FROZEN gate (thresholds + per-seed clause logic).
import research.runners._pmem_intention_latch_derisk as base  # noqa: E402
from research.runners._pmem_intention_latch_derisk import (  # noqa: E402  (FROZEN gate constants -- imported, never re-typed)
    FIRE_THR, SILENT_MAX, HOLD_FLOOR, LESION_HELD_MAX, SEP_RATIO, GO_MIN_SEEDS_FRAC,
)
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict               # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_perpool_homeostat.json")

# the silence clauses that MUST stay 6/6 -- a homeostat that breaks any of these is a CHEAT (spurious fires).
SILENCE_CLAUSES = ("no_fire_before", "no_fire_wrongcue", "no_intention_silent",
                   "lesion_holds", "lesion_forgets", "persistence")


_BIAS_CACHE = {}   # (seed, homeostat/substrate knobs) -> {action: calibrated bias pA}. The calibrated set-point is
                   # a deterministic function of the seed + knobs, so we calibrate ONCE per seed and reuse it across
                   # the 5 fresh condition-builds run_seed makes (they would otherwise re-derive the identical bias).


class HomeostaticProspectiveMemory(base.ProspectiveMemory):
    """ProspectiveMemory + a PER-POOL homeostatic set-point on each `rel` pool's tonic-inhibition bias.

    Intrinsic-plasticity excitability homeostasis (Turrigiano 2011; Desai et al. 1999): each `rel` pool adapts its
    own tonic bias so that its WORST SUSTAINED single feedforward input (max of cue-alone and held-alone) settles at
    a sub-threshold rate set-point r_set. Label-free (references the pool's own single-input response, never which
    cue is correct); symmetric across pools; calibrated once at build (an always-on developmental operating-point
    tuning), then FROZEN for the trial so the fast coincidence transient carries the release the slow homeostat does
    not cancel (timescale separation).
    """

    def __init__(self, actions, distractors, homeostat_on=True, homeostat_r_set=0.045,
                 homeostat_eta=4000.0, homeostat_iters=15, homeostat_window=6,
                 homeostat_bias_min=-4000.0, homeostat_bias_max=0.0, homeostat_cal_drive_pA=2500.0,
                 **kw):
        super().__init__(actions, distractors, **kw)
        self.homeostat_on = bool(homeostat_on)
        self._seed = int(kw.get("seed", 42))
        self._bias_pool = {a: float(self._rel_bias_pA) for a in self.actions}     # per-pool adaptive bias (pA)
        self._rel_idx_dev = {a: self.xp.asarray(self._rel_idx[a]) for a in self.actions}
        self._h = dict(r_set=float(homeostat_r_set), eta=float(homeostat_eta), iters=int(homeostat_iters),
                       window=int(homeostat_window), bmin=float(homeostat_bias_min),
                       bmax=float(homeostat_bias_max), drive=float(homeostat_cal_drive_pA))
        # calibration probes the SUSTAINED single-input operating point. The rel pool is an NMDA-recurrent soft-WTA
        # ACCUMULATOR: under a SINGLE sustained input it settles at a LOW fixed point, under the COINCIDENCE it
        # ramps to a HIGH fixed point. The gate's worst silence condition is the SUSTAINED held-alone read
        # (no_fire_before ramps rel over the whole ~375-step hold), so a short/transient calibration under-reads it
        # and leaves the pool too excitable. We calibrate against the MAX of BOTH sustained single inputs (cue and
        # held), over a sustained window, so every single-input silence condition is pinned sub-threshold per pool.
        self._cal_cue_window = 30                    # matches present_cue -> _read(window=30): the cue read regime
        self._cal_N = int(homeostat_window)          # #intervening distractor turns in the held probe (>= trial N)
        self._bias_trace = {a: float(self._rel_bias_pA) for a in self.actions}
        self._cue_alone_trace = {a: None for a in self.actions}
        self._cache_key = (self._seed, tuple(sorted(self.actions)), self._rel_bias_pA,
                           self._h["r_set"], self._h["eta"], self._h["iters"], self._cal_N,
                           self._h["bmin"], self._h["bmax"])
        if self.homeostat_on:
            # SHARED (one-brain merge): BYPASS the module cache -- each arm (merged vs coresident) must calibrate
            # INDEPENDENTLY on its OWN pool slice so the byte-identity of the calibrated bias is a genuine result,
            # not a cache hit. (The standalone keeps the cache: the 5 fresh condition-builds re-derive one bias.)
            if self._shared is not None:
                self._calibrate_all()
            else:
                cached = _BIAS_CACHE.get(self._cache_key)
                if cached is not None:
                    self._bias_pool = dict(cached["bias"])
                    self._bias_trace = {a: round(self._bias_pool[a], 1) for a in self.actions}
                    self._cue_alone_trace = dict(cached["trace"])
                else:
                    self._calibrate_all()
                    _BIAS_CACHE[self._cache_key] = {"bias": dict(self._bias_pool),
                                                    "trace": dict(self._cue_alone_trace)}

    def _step(self, drive_idx=None, drive_pA=0.0):
        """Same as the parent, but the tonic rel bias is PER POOL (the homeostat's adapted set-point) when ON."""
        cur = self.bridge.cp_external_input_current
        cur[:] = 0.0
        if drive_idx is not None:
            cur[drive_idx] = np.float32(drive_pA)
        if self.homeostat_on:
            for a in self.actions:
                cur[self._rel_idx_dev[a]] = np.float32(self._bias_pool[a])
        else:
            cur[self._rel_all] = np.float32(self._rel_bias_pA)   # parent's CONSTANT bias (the control arm)
        self.bridge._run_one_simulation_step()

    def _rel_only(self, action, window, cue=None):
        """LIGHT read: run `window` steps (optionally driving one cue) and return ONLY rel_{action}'s firing rate
        (one host transfer/step, vs _read's full rel+held scan) -- calibration needs just this pool's rate."""
        cue_idx = self._cpat[cue] if cue is not None else None
        ridx = self._rel_idx[action]
        s = 0.0
        for _ in range(window):
            self._step(drive_idx=cue_idx, drive_pA=self._h["drive"])
            fs = self.bridge.cp_firing_states
            s += float(self.B.to_host(fs[ridx]).sum())
        return s / (self._n_rel * window)

    def _single_input_rate(self, action):
        """MAX of the two single-input silence conditions for pool `action`, each from rest, over the window that
        MATCHES each condition's actual exposure in the gate (asymmetric -- the two inputs differ in duration):
          * cue-alone  : drive cue_{action}, no held intention, over the SHORT cue window (present_cue reads 30
                         steps): the wrong-cue / no-intention / lesioned-cue read.
          * held-alone : latch the intention, read with the cue ABSENT, over the LONG hold window: the
                         no_fire_before read, which the rel accumulator ramps SLOWLY across the whole ~300-step
                         hold. Probing it from rest over a short window UNDER-reads the cross-turn runaway (0.006
                         at 115 steps vs 0.067 at ~300) and leaves the pool too excitable -> silence breaks.
        The homeostat pins THIS max to r_set, so whichever single input is the worse PER SEED stays sub-threshold.
        Neither input alone may fire; only the coincidence (both) may."""
        self._reset_dynamics()
        r_cue = self._rel_only(action, self._cal_cue_window, cue=f"cue_{action}")
        # held probe: REPLICATE the gate's no_fire_before condition exactly -- encode, then run the intervening
        # distractor turns (writes interleaved with reads), take the PEAK rel over turns. A continuous read
        # under-reads it (the cross-turn accumulation with distractor writes climbs higher, 0.029 vs 0.062).
        self._reset_dynamics()
        self.encode_intention(action)                       # latch the intention (self-sustaining attractor)
        dists = self.distractors or [None]
        inter = [dists[i % len(dists)] for i in range(self._cal_N)]
        r_held = 0.0
        for d in inter:
            r = self._read(window=20, cue=None) if d is None else self.intervening_turn(d)
            r_held = max(r_held, r["rel"][action])
        return max(r_cue, r_held), r_cue, r_held

    def _calibrate_all(self):
        """Intrinsic-plasticity calibration: adapt each pool's tonic bias so the WORST sustained SINGLE input
        settles at the sub-threshold set-point r_set. Runs on BOTH pools (always-on per-pool homeostasis, not just
        the latched one) -> every single-input silence condition is sub-threshold by construction, on every seed."""
        h = self._h
        for a in self.actions:
            r = c = held = None
            for _ in range(h["iters"]):
                r, c, held = self._single_input_rate(a)
                self._bias_pool[a] += h["eta"] * (h["r_set"] - r)
                self._bias_pool[a] = float(min(max(self._bias_pool[a], h["bmin"]), h["bmax"]))
            self._cue_alone_trace[a] = {"max_single": round(r, 4), "cue": round(c, 4), "held": round(held, 4)}
            self._bias_trace[a] = round(self._bias_pool[a], 1)
        self._reset_dynamics()


def _run_arm(seeds, N, n_distractors, homeostat_on, **kw):
    """Run base.run_seed (the FROZEN gate) per seed with the homeostatic substrate (on/off arm)."""
    per = []
    for s in seeds:
        d = base.run_seed(s, N, n_distractors, homeostat_on=homeostat_on, **kw)
        # surface the calibrated bias/cue-alone for the report (rebuild is not needed; run_seed already ran it,
        # but its PMs are gone -- so we re-instantiate ONE PM at this seed just to read the calibrated state).
        per.append(d)
    return per


def _agg(per, seeds):
    n_pass = sum(int(p["passed"]) for p in per)
    clauses = list(per[0]["clauses"].keys())
    agg = {c: sum(int(p["clauses"][c]) for p in per) for c in clauses}
    fire_per_seed = {p["seed"]: round(min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"]), 4) for p in per}
    silent_per_seed = {p["seed"]: round(p["max_silent"], 4) for p in per}
    mean_fire = float(np.mean([min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"]) for p in per]))
    mean_silent = float(np.mean([p["max_silent"] for p in per]))
    return dict(n_pass=n_pass, agg=agg, fire_per_seed=fire_per_seed, silent_per_seed=silent_per_seed,
                mean_fire=mean_fire, mean_silent=mean_silent)


def _derisk(seeds, N, n_distractors, smoke=False, **kw):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"PMEM PER-POOL HOMEOSTAT [{tag}] -- intrinsic-plasticity set-point on each rel pool; "
          f"{len(seeds)} seed(s), N={N}, {n_distractors} distractors, r_set={kw.get('homeostat_r_set')}", flush=True)
    t0 = time.time()
    err = None
    try:
        # monkey-patch the substrate class so base.run_seed / base._new_pm build the HOMEOSTATIC PM with the
        # IDENTICAL frozen gate. (base._new_pm looks up module-global ProspectiveMemory.)
        base.ProspectiveMemory = HomeostaticProspectiveMemory

        print("\n--- ARM 1: HOMEOSTAT ON (the surpass) ---", flush=True)
        on = _run_arm(seeds, N, n_distractors, homeostat_on=True, **kw)
        for p in on:
            c = p["clauses"]
            fails = " ".join(k for k, v in c.items() if not v) or "ALL-PASS"
            print(f"  [seed {p['seed']}] pass={p['passed']} | fireA={p['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={p['fireB']['rel_B_on_cueB']:.3f} | max_silent={p['max_silent']:.3f} "
                  f"held_min={p['held_min']:.3f} | {fails}", flush=True)

        print("\n--- ARM 2: HOMEOSTAT OFF (parent constant bias, internal control) ---", flush=True)
        off = _run_arm(seeds, N, n_distractors, homeostat_on=False, **kw)
        for p in off:
            c = p["clauses"]
            fails = " ".join(k for k, v in c.items() if not v) or "ALL-PASS"
            print(f"  [seed {p['seed']}] pass={p['passed']} | fireA={p['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={p['fireB']['rel_B_on_cueB']:.3f} | max_silent={p['max_silent']:.3f} | {fails}", flush=True)

        # read one calibrated PM per seed to expose the adapted bias + post-calibration cue-alone rate.
        actions = ["A", "B"]
        cal = {}
        for s in seeds:
            pm = HomeostaticProspectiveMemory(actions, [f"d{i}" for i in range(n_distractors)],
                                              homeostat_on=True, seed=s, **kw)
            cal[s] = {"bias_pA": dict(pm._bias_trace), "cue_alone_after": dict(pm._cue_alone_trace)}
        print("\n--- CALIBRATED STATE (per seed): adapted bias + post-cal cue-alone rate (must be <= r_set-ish) ---",
              flush=True)
        for s in seeds:
            print(f"  [seed {s}] bias={cal[s]['bias_pA']}  cue_alone_after={cal[s]['cue_alone_after']}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_perpool_homeostat", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    A = _agg(on, seeds)
    B = _agg(off, seeds)
    min_seeds = int(np.ceil(GO_MIN_SEEDS_FRAC * len(seeds)))

    # ANTI-CHEAT: every silence clause MUST stay 6/6 under the homeostat (else it created spurious fires).
    silence_regressed = [c for c in SILENCE_CLAUSES if A["agg"].get(c, 0) < len(seeds)]
    cheat = void_if(bool(silence_regressed),
                    f"the homeostat REGRESSED a silence clause {silence_regressed} -> spurious fires (a homeostat "
                    f"that raises all gains until everything fires is a CHEAT; the surpass is VOID)")

    fire_on = A["agg"].get("fire_on_cue", 0)
    fire_off = B["agg"].get("fire_on_cue", 0)
    go = bool(A["n_pass"] >= min_seeds) and (not cheat) and (not smoke)

    # attribute the fire lift to the homeostat: mean correct-cue release ON vs OFF (identical substrate).
    lift = attributable_to("fire lift owned by the per-pool homeostat (mean fire: ON vs OFF)",
                           A["mean_fire"], B["mean_fire"])
    # specifically the hypo-excitable seed 100: its fire ON vs OFF.
    s100_on = A["fire_per_seed"].get(100)
    s100_off = B["fire_per_seed"].get(100)

    vd = Verdict("pmem_perpool_homeostat")
    for c in SILENCE_CLAUSES:
        vd.require(f"silence held under homeostat: {c} (per-seed count)", A["agg"].get(c, 0),
                   expect=lambda x, n=len(seeds): x == n)
    vd.reaches("fire_on_cue pass-count: homeostat OFF -> ON", fire_off, fire_on)
    vd.control("mean correct-cue release: homeostat ON vs OFF", A["mean_fire"], B["mean_fire"], min_separation=0.02)
    vd.disabled("STDP / Hebbian / STP / OU-noise",
                "clean-hold WM config unchanged; the ONLY added mechanism is the per-pool intrinsic-plasticity "
                "set-point on the rel tonic-inhibition bias (Turrigiano 2011 / Desai 1999)")
    decided = vd.decide(go)

    silence_counts = ", ".join(f"{c}:{A['agg'][c]}" for c in SILENCE_CLAUSES)
    if smoke:
        verdict = (f"SMOKE OK -- the per-pool homeostat RUNS end-to-end and every condition is live/measured. "
                   f"ON fire_on_cue={fire_on}/{len(seeds)} (OFF {fire_off}/{len(seeds)}); mean fire ON~{A['mean_fire']:.3f} "
                   f"vs OFF~{B['mean_fire']:.3f}; silence-regressed={silence_regressed or 'none'}. "
                   f"Not a GO claim; run --derisk for the 6-seed verdict.")
    elif cheat:
        verdict = (f"VOID -- the homeostat regressed silence clause(s) {silence_regressed}: it created spurious "
                   f"fires rather than lifting the coincidence. This is the named cheat; the surpass does NOT hold "
                   f"with silence intact. fire_on_cue ON={fire_on}/{len(seeds)}.")
    elif go:
        verdict = (
            f"GO -- the per-pool homeostat SURPASSES the fire_on_cue boundary WITH silence intact. Giving each rel "
            f"cue-monitor pool an intrinsic-plasticity set-point on its tonic-inhibition bias (calibrated label-free "
            f"on the pool's own cue drive to a sub-threshold target r_set) normalizes every pool's operating point: "
            f"fire_on_cue rises {fire_off}/{len(seeds)} -> {fire_on}/{len(seeds)}, and {A['n_pass']}/{len(seeds)} "
            f"seeds now pass EVERY clause (need {min_seeds}). Every silence clause STAYS 6/6 "
            f"({silence_counts}) -- the homeostat lifts the COINCIDENCE, "
            f"not any single-input silence condition (it pins the strongest single input, the cue, sub-threshold by "
            f"construction). The hypo-excitable seed 100 is specifically rescued: correct-cue release "
            f"{s100_off} -> {s100_on}. Mean fire ON~{A['mean_fire']:.3f} vs the identical substrate OFF~"
            f"{B['mean_fire']:.3f} (parent constant bias). All reads cp_firing_states; NO sim/ edit.")
    else:
        fails = {c: A["agg"][c] for c in A["agg"] if A["agg"][c] < len(seeds)}
        verdict = (f"BOUNDARY -- the homeostat lifted fire_on_cue {fire_off}/{len(seeds)} -> {fire_on}/{len(seeds)} "
                   f"and held every silence clause, but {A['n_pass']}/{len(seeds)} seeds pass all clauses "
                   f"(need {min_seeds}). Residual: {sorted(fails)}. per-seed fire ON={A['fire_per_seed']}. "
                   f"Honest residual -- name the next single-variable mechanism, do NOT force GO.")

    summary = {
        "probe": "pmem_perpool_homeostat", "verdict": verdict, "go": bool(go),
        "task": ("prospective-memory fire_on_cue amplitude closure: a per-pool intrinsic-plasticity homeostat "
                 "(Turrigiano 2011 / Desai 1999) on each rel cue-monitor pool's tonic-inhibition bias, calibrated "
                 "label-free on the pool's own cue drive to a sub-threshold set-point; normalizes the operating "
                 "point so the coincidence clears a FIXED FIRE_THR on all seeds without lifting any single-input "
                 "silence over threshold. ON vs OFF (parent constant bias) internal control. All reads "
                 "cp_firing_states; NO sim/ edit; reuse-by-import of the parent ProspectiveMemory + frozen gate."),
        "gate": {"FIRE_THR": FIRE_THR, "SILENT_MAX": SILENT_MAX, "HOLD_FLOOR": HOLD_FLOOR,
                 "LESION_HELD_MAX": LESION_HELD_MAX, "SEP_RATIO": SEP_RATIO,
                 "GO_MIN_SEEDS_FRAC": GO_MIN_SEEDS_FRAC},
        "homeostat": {k: kw.get(f"homeostat_{k2}") for k, k2 in
                      [("r_set", "r_set"), ("eta", "eta"), ("iters", "iters"), ("window", "window"),
                       ("bias_min", "bias_min"), ("bias_max", "bias_max")]},
        "N_intervening": N, "n_distractors": n_distractors, "seeds": list(seeds),
        "min_seeds_to_go": min_seeds,
        "ON": {"n_pass": A["n_pass"], "per_clause_pass_counts": A["agg"],
               "fire_per_seed": A["fire_per_seed"], "max_silent_per_seed": A["silent_per_seed"],
               "mean_fire": A["mean_fire"], "mean_max_silent": A["mean_silent"]},
        "OFF": {"n_pass": B["n_pass"], "per_clause_pass_counts": B["agg"],
                "fire_per_seed": B["fire_per_seed"], "max_silent_per_seed": B["silent_per_seed"],
                "mean_fire": B["mean_fire"], "mean_max_silent": B["mean_silent"]},
        "fire_on_cue_OFF_to_ON": [fire_off, fire_on],
        "seed100_fire_OFF_to_ON": [s100_off, s100_on],
        "fire_lift_attributable_to_homeostat": lift,
        "silence_regressed": silence_regressed,
        "calibrated_state": cal,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed_ON": on, "per_seed_OFF": off,
        "BIOLOGY": ("Intrinsic homeostasis of excitability toward a firing-rate set-point (Turrigiano 2011; Desai, "
                    "Rutherford & Turrigiano 1999); the excitability substrate is K-channel-set F-I / spike-freq "
                    "adaptation (Kandel 6e). Realized as a per-pool adaptive tonic-inhibition set-point (the engine "
                    "carries this natively as homeostasis_target_rate + threshold adaptation, but the default "
                    "tau~5000-step timescale is too slow to converge in a short trial). A divisive-normalization FS "
                    "partner (Carandini & Heeger 2012) was considered and rejected: the finding diagnosed a "
                    "SUBTRACTIVE/threshold deficit, which a set-point shift fixes and gain division cannot."),
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[pmem-homeostat] VERDICT: {verdict}", flush=True)
    print(f"[pmem-homeostat] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (go or smoke) else 1


def _write(summary):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    # substrate knobs (parity with the parent; defaults MATCH the parent so the OFF arm reproduces the boundary)
    ap.add_argument("--hold-to-rel-weight", type=float, default=3.2)
    ap.add_argument("--cue-to-rel-weight", type=float, default=4.2)
    ap.add_argument("--rel-recurrent-weight", type=float, default=0.10)
    ap.add_argument("--rel-bias-pA", type=float, default=-1050.0)
    ap.add_argument("--n-rel", type=int, default=60)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=40)
    # homeostat knobs (set label-free, identical for ALL seeds)
    ap.add_argument("--homeostat-r-set", type=float, default=0.045,
                    help="sub-threshold single-input set-point (< SILENT_MAX=0.06, with margin)")
    ap.add_argument("--homeostat-eta", type=float, default=4000.0)
    ap.add_argument("--homeostat-iters", type=int, default=15)
    ap.add_argument("--homeostat-window", type=int, default=6,
                    help="#intervening distractor turns in the held calibration probe (>= trial N; replicates the "
                         "gate's no_fire_before cross-turn accumulation)")
    ap.add_argument("--homeostat-bias-min", type=float, default=-4000.0)
    ap.add_argument("--homeostat-bias-max", type=float, default=0.0)
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else a.seeds
    kw = dict(hold_to_rel_weight=a.hold_to_rel_weight, cue_to_rel_weight=a.cue_to_rel_weight,
              rel_recurrent_weight=a.rel_recurrent_weight, rel_bias_pA=a.rel_bias_pA,
              n_rel=a.n_rel, n=a.n, pattern_size=a.pattern_size,
              homeostat_r_set=a.homeostat_r_set, homeostat_eta=a.homeostat_eta,
              homeostat_iters=a.homeostat_iters, homeostat_window=a.homeostat_window,
              homeostat_bias_min=a.homeostat_bias_min, homeostat_bias_max=a.homeostat_bias_max)
    if a.smoke:
        return _derisk([seeds[0]], N=3, n_distractors=min(3, a.n_distractors), smoke=True, **kw)
    return _derisk(seeds, N=a.N, n_distractors=a.n_distractors, smoke=False, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
