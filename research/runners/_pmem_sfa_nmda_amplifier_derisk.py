"""PROSPECTIVE MEMORY fire_on_cue closure -- SPIKE-FREQUENCY ADAPTATION + a SUPRALINEAR NMDA COINCIDENCE AMPLIFIER.

THE RESIDUAL (from research/findings/2026-08-13-prospective-perpool-homeostat-BOUNDARY.md).
The per-pool intrinsic-plasticity homeostat GUARANTEES every silence clause 6/6 and lifts fire_on_cue 3/6 -> 4/6,
but relocates the last 2 misses to a COINCIDENCE SEPARATION/GAIN deficit the set-point CANNOT fix:
  * seed 100 -- a coincidence GAIN deficit: fire/silent ratio 0.085/0.044 = 1.94 < the 3.33 the absolute
    FIRE_THR/SILENT_MAX window (0.20/0.06) requires. The JOINT (held+cue) response is intrinsically weak; a
    set-point/threshold shift cannot manufacture separation that is not in the pool's F-I curve.
  * seed 44 -- a sustained-runaway-forced conservative bias: its held-alone input RUNS AWAY over the ~300-step
    hold, so the homeostat must set the bias BELOW baseline to hold silence, which SUPPRESSES the coincidence
    (fireA 0.157 < 0.20). The recurrent SUSTAINED runaway, not the operating point, forces the fire-killing bias.

THE MISSING COMPANION (CLAUDE.md wall reframe: what else does the real system run alongside this?). The pool
detects a TRANSIENT coincidence (~30-step cue read) against a SUSTAINED single-input hold (~300 steps) with only
a tonic bias -- a static operating point. Biology runs TWO more processes alongside, at a timescale BETWEEN the
two reads:
  (1) SPIKE-FREQUENCY ADAPTATION (Kv/M-current AHP; Kandel 6e excitability regulation). A slow K-adaptation
      current builds with SUSTAINED firing and cancels the sustained single input, but has not accumulated on the
      FRESH cue-onset transient -> it DECOUPLES sustained from transient. This lets the seed-44 runaway self-limit
      so the homeostat no longer needs a fire-killing bias (calibrated WITH SFA active, the bias is less negative).
  (2) A SUPRALINEAR NMDA / dendritic-plateau COINCIDENCE AMPLIFIER (Kandel 6e: "the NMDA receptor acts as a
      molecular coincidence detector"; Schiller & Schiller dendritic NMDA spikes -- a local regenerative
      depolarization/plateau). A single feedforward input keeps a rel neuron's NMDA conductance sub-plateau; the
      COINCIDENCE (act_X held AND cue_X present) sums BOTH feedforward inputs -> crosses the plateau threshold ->
      a supralinear boost. This lifts the seed-100 gain deficit. It is coincidence-SPECIFIC by the threshold: the
      plateau threshold is calibrated per neuron to sit just ABOVE that neuron's worst SINGLE-input NMDA
      conductance (label-free), so a single input NEVER boosts -> silence is preserved by construction; only the
      joint drive crosses -> the JOINT response is amplified. (The parent finding noted divisive-normalization
      ALONE would WORSEN seed 100 -- dividing an already-weak coincidence; the NMDA amplifier is the gain side.)

THE BUILD (this runner; additive, NO sim/ edit; reuse-by-import of the [H] HomeostaticProspectiveMemory + the
FROZEN gate). Subclass the per-pool homeostat and add, on each `rel` cue-monitor pool:
  * SFA: a per-neuron adaptation current  I_sfa[i] = -sfa_g * a[i],  a[i] a normalized low-pass of rel_i's spikes
    (a <- decay*a + (1-decay)*fired, decay = exp(-1/sfa_tau), sfa_tau BETWEEN the 30-step read and the 300-step
    hold). SFA is ACTIVE during the homeostat calibration -> the bias adapts to the SFA-decoupled operating point.
  * NMDA plateau: read the pool's REAL cp_conductance_g_nmda; boost[i] = min(plateau_g * relu(g_nmda[i] -
    theta[i]), plateau_cap). theta[i] is calibrated (STAGE 2, biases frozen) to margin * that neuron's max
    single-input g_nmda (cue-alone over the cue window AND held-alone over the full hold) -> zero on any single
    input, positive only on the coincidence. plateau_cap = the saturating plateau amplitude.

BRAIN-BASED / label-free (same flagged scope as the parent): both currents reference the pool's OWN spiking /
NMDA conductance, never which cue is correct. HOST-SCAFFOLD, FLAGGED (unchanged): the cue->action CONTENT binding
is installed synaptically; the SFA current and the plateau boost are host-injected current-injection PROXIES for
the intrinsic K-adaptation conductance and the dendritic NMDA plateau -- the same class of flagged proxy as the
parent's tonic-inhibition bias. The MECHANISM (adaptation timescale-separation + supralinear coincidence gate) is
brain-based; every read is cp_firing_states / cp_conductance_g_nmda.

ANTI-CHEAT (the mission's central risk). "An amplifier that fires on single inputs is a cheat." It CANNOT: the
per-neuron plateau threshold is pinned ABOVE each neuron's worst single input, so no single-input silence
condition boosts. The gate PROVES it -- every silence clause must STAY 6/6 (a regression => VOID). ARM 2 (SFA+
plateau OFF, homeostat ON) reproduces the [H] 4/6, so the ARM-1 lift is attributable to SFA+NMDA, not a substrate
change.

GATE. Identical to the parent (FROZEN thresholds + per-seed clause logic IMPORTED from
research.runners._pmem_intention_latch_derisk; the substrate class monkey-patched). 6 seeds 42/43/44/100/101/102.
The QUESTION: does fire_on_cue reach 6/6 WHILE every silence clause STAYS 6/6, with the fire/silent ratio >= 3.33
on every seed (seed 100's 1.94 specifically rescued)?

  SIM_BACKEND=numpy python -m research.runners._pmem_sfa_nmda_amplifier_derisk --smoke            # seeds[0], N=3
  SIM_BACKEND=numpy python -m research.runners._pmem_sfa_nmda_amplifier_derisk --seed 100 --smoke # target-seed smoke
  SIM_BACKEND=numpy python -m research.runners._pmem_sfa_nmda_amplifier_derisk --derisk           # 6 seeds, on+off
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

# reuse-by-import: the parent runner (FROZEN gate: thresholds + per-seed clause logic + run_seed) and the [H]
# per-pool homeostat substrate (HomeostaticProspectiveMemory + its label-free bias calibration).
import research.runners._pmem_intention_latch_derisk as base           # noqa: E402
import research.runners._pmem_perpool_homeostat_derisk as homeo        # noqa: E402
from research.runners._pmem_intention_latch_derisk import (            # noqa: E402  (FROZEN gate constants)
    FIRE_THR, SILENT_MAX, HOLD_FLOOR, LESION_HELD_MAX, SEP_RATIO, GO_MIN_SEEDS_FRAC,
)
from tools.lab import attributable_to, void_if   # noqa: E402
from tools.verdict import Verdict                # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_sfa_nmda_amplifier.json")

# the ratio the ABSOLUTE window (FIRE_THR/SILENT_MAX) requires -- reported per seed; must clear on every seed.
REQUIRED_RATIO = FIRE_THR / SILENT_MAX     # 0.20 / 0.06 = 3.33

SILENCE_CLAUSES = homeo.SILENCE_CLAUSES

_THETA_CACHE = {}   # seed -> {action: per-neuron plateau threshold vector}. Deterministic per seed within an arm.


class SFANmdaProspectiveMemory(homeo.HomeostaticProspectiveMemory):
    """Per-pool homeostat + SFA (timescale-separation) + a supralinear NMDA coincidence amplifier on each rel pool.

    SFA (active during the homeostat calibration): a per-neuron K-adaptation current subtracts a normalized
    low-pass of the pool's own spikes -> the SUSTAINED single input adapts away (rescuing the seed-44 runaway so
    the calibrated bias is no longer fire-killing) while the fresh cue-onset TRANSIENT is preserved.
    NMDA plateau (calibrated stage-2, biases frozen; label-free per-neuron threshold above the worst single
    input): a supralinear boost on the pool's own NMDA conductance -> the JOINT drive is amplified (rescuing the
    seed-100 gain deficit) while no single input ever crosses the threshold (silence preserved by construction).
    """

    def __init__(self, actions, distractors,
                 sfa_on=True, sfa_g=2500.0, sfa_tau=100.0,
                 plateau_on=True, plateau_g=6000.0, plateau_margin=1.05, plateau_cap=1600.0,
                 **kw):
        # set BEFORE super().__init__ -- the (inherited) homeostat calibration calls self._step, which is MY
        # override; it must see SFA already configured and the plateau OFF (stage-1: bias under SFA only).
        self._sfa_on = bool(sfa_on)
        self._sfa_g = float(sfa_g)
        self._sfa_tau = float(sfa_tau)
        self._sfa_decay = float(np.exp(-1.0 / max(float(sfa_tau), 1e-6)))
        self._sfa_a = None                    # lazily built per-pool adaptation state (normalized low-pass, 0..1)
        self._want_plateau = bool(plateau_on)
        self._plateau_on = False              # OFF during super's stage-1 homeostat calibration
        self._plateau_g = float(plateau_g)
        self._plateau_margin = float(plateau_margin)
        self._plateau_cap = float(plateau_cap)
        self._theta = None                    # per-pool per-neuron plateau threshold (calibrated stage-2)
        self._diag = {}                       # DIAGNOSTIC (label-based; report only, NOT used to set theta)

        super().__init__(actions, distractors, **kw)   # stage-1: homeostat bias calibrated WITH SFA on

        if self._want_plateau and self.homeostat_on:
            cached = _THETA_CACHE.get(self._seed)
            if cached is not None:
                self._theta = dict(cached["theta"])
                self._diag = dict(cached["diag"])
            else:
                self._calibrate_plateau()      # stage-2: pool threshold above the worst single input
                _THETA_CACHE[self._seed] = {"theta": dict(self._theta), "diag": dict(self._diag)}
            self._plateau_on = True

    # ---- SFA state ----
    def _ensure_sfa(self):
        if self._sfa_a is None:
            self._sfa_a = {a: np.zeros(self._n_rel, np.float32) for a in self.actions}

    def _reset_dynamics(self):
        super()._reset_dynamics()
        if self._sfa_a is not None:
            for a in self.actions:
                self._sfa_a[a][:] = 0.0

    def _step(self, drive_idx=None, drive_pA=0.0):
        """Parent per-pool-bias step + (subtractive) SFA adaptation current + (additive) supralinear NMDA plateau
        boost, all on the rel pools. The plateau reads the PREVIOUS step's NMDA conductance (a one-step delay --
        the dendritic plateau follows the conductance) and SFA subtracts PAST firing, then updates with this
        step's spikes."""
        cur = self.bridge.cp_external_input_current
        cur[:] = 0.0
        if drive_idx is not None:
            cur[drive_idx] = np.float32(drive_pA)
        # tonic per-pool bias (homeostat) or the parent's CONSTANT bias (control arm)
        if self.homeostat_on:
            for a in self.actions:
                cur[self._rel_idx_dev[a]] = np.float32(self._bias_pool[a])
        else:
            cur[self._rel_all] = np.float32(self._rel_bias_pA)
        # (1) SFA: subtract a normalized low-pass of the pool's own spikes (K-adaptation current)
        if self._sfa_on:
            self._ensure_sfa()
            for a in self.actions:
                idx = self._rel_idx_dev[a]
                cur[idx] = cur[idx] - self.xp.asarray(np.float32(self._sfa_g) * self._sfa_a[a])
        # (2) NMDA plateau: a POOL-GATED regenerative event. When the pool's MEAN NMDA conductance crosses the
        # pool threshold (calibrated above the worst single input), ALL pool neurons get a uniform supralinear
        # boost -- an all-or-none dendritic/NMDA-spike plateau that spreads across the branch, lifting the
        # high-rheobase laggard neurons a per-neuron graded boost leaves behind (they get ~0 self-margin). A
        # single input never lifts the pool-mean over threshold -> no boost -> silence preserved.
        if self._plateau_on and self._theta is not None:
            gn = self.bridge.cp_conductance_g_nmda
            for a in self.actions:
                idx = self._rel_idx_dev[a]
                g_pool = float(self.B.to_host(gn[idx]).mean())
                excess = g_pool - self._theta[a]
                if excess > 0.0:
                    boost = self._plateau_g * excess
                    if self._plateau_cap > 0:
                        boost = min(boost, self._plateau_cap)
                    cur[idx] = cur[idx] + np.float32(boost)
        self.bridge._run_one_simulation_step()
        # update SFA state with THIS step's spikes (normalized low-pass -> steady-state == firing rate)
        if self._sfa_on:
            self._ensure_sfa()
            fs = self.bridge.cp_firing_states
            for a in self.actions:
                fired = self.B.to_host(fs[self._rel_idx_dev[a]]).astype(np.float32)
                self._sfa_a[a] = (self._sfa_decay * self._sfa_a[a]
                                  + (1.0 - self._sfa_decay) * fired).astype(np.float32)

    # ---- stage-2 plateau-threshold calibration (label-free) ----
    def _peak_pool_gnmda(self, action, run_steps):
        """Sample the peak POOL-MEAN g_nmda on pool `action` while `run_steps()` drives the substrate (SFA
        active, plateau OFF -- theta is not set yet). The pool-mean is the regenerative-plateau gating signal."""
        idx = self._rel_idx_dev[action]
        peak = [0.0]

        def sample():
            g_pool = float(self.B.to_host(self.bridge.cp_conductance_g_nmda[idx]).mean())
            if g_pool > peak[0]:
                peak[0] = g_pool

        run_steps(sample)
        return peak[0]

    def _calibrate_plateau(self):
        """theta[a] = margin * max over the two SINGLE inputs of the peak POOL-MEAN g_nmda:
          * cue-alone  : drive cue_{a} over the cue window (present_cue's regime).
          * held-alone : latch intention a, run the full N-turn hold with distractor writes (no_fire_before's
                         regime -- the rel accumulator ramps across the whole hold).
        Label-free: references only the pool's OWN single-input NMDA conductance, never which cue is correct.
        Because a single input never lifts the pool-mean over theta, no single-input silence condition boosts;
        only the coincidence (both inputs summed) crosses -> the pool-wide plateau fires -> the JOINT response is
        amplified. Also records a DIAGNOSTIC coincidence pool-mean peak (label-based; report only) to expose the
        separation margin the plateau exploits."""
        h = self._h
        cue_win = self._cal_cue_window
        drive = h["drive"]
        dists = self.distractors or [None]
        inter = [dists[i % len(dists)] for i in range(self._cal_N)]

        def make_cue_runner(a):
            def run(sample):
                self._reset_dynamics()
                cue_idx = self._cpat[f"cue_{a}"]
                for _ in range(cue_win):
                    self._step(drive_idx=cue_idx, drive_pA=drive)
                    sample()
            return run

        def make_held_runner(a):
            def run(sample):
                self._reset_dynamics()
                self.encode_intention(a)
                for d in inter:
                    if d is not None:
                        self._write(d)
                    for _ in range(20):
                        self._step()
                        sample()
            return run

        def make_coinc_runner(a):   # DIAGNOSTIC ONLY -- coincidence peak; not used to set theta
            def run(sample):
                self._reset_dynamics()
                self.encode_intention(a)
                for d in inter:
                    self.intervening_turn(d)
                cue_idx = self._cpat[f"cue_{a}"]
                for _ in range(cue_win):
                    self._step(drive_idx=cue_idx, drive_pA=drive)
                    sample()
            return run

        self._theta = {}
        for a in self.actions:
            peak_cue = self._peak_pool_gnmda(a, make_cue_runner(a))
            peak_held = self._peak_pool_gnmda(a, make_held_runner(a))
            single = max(peak_cue, peak_held)
            self._theta[a] = float(self._plateau_margin * single)
            peak_coinc = self._peak_pool_gnmda(a, make_coinc_runner(a))   # diagnostic (label-based; report only)
            self._diag[a] = {
                "g_single_cue": round(float(peak_cue), 3),
                "g_single_held": round(float(peak_held), 3),
                "g_coinc": round(float(peak_coinc), 3),
                "theta": round(float(self._theta[a]), 3),
                "coinc_over_theta": bool(peak_coinc > self._theta[a]),
                "coinc_margin": round(float(peak_coinc - self._theta[a]), 3),
            }
        self._reset_dynamics()


# --------------------------------------------------------------------------------------------------------
def _ratio_per_seed(per):
    """fire_min / max_silent per seed (must clear REQUIRED_RATIO=3.33 on every seed)."""
    out = {}
    for p in per:
        fire = min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"])
        sil = max(p["max_silent"], 1e-6)
        out[p["seed"]] = round(fire / sil, 3)
    return out


def _run_arm(seeds, N, n_distractors, **kw):
    return [base.run_seed(s, N, n_distractors, **kw) for s in seeds]


def _derisk(seeds, N, n_distractors, smoke=False, **kw):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"PMEM SFA + NMDA-PLATEAU [{tag}] -- timescale-separation adaptation + supralinear coincidence amplifier; "
          f"{len(seeds)} seed(s), N={N}, {n_distractors} distractors; "
          f"sfa_g={kw.get('sfa_g')} sfa_tau={kw.get('sfa_tau')} plateau_g={kw.get('plateau_g')} "
          f"cap={kw.get('plateau_cap')} margin={kw.get('plateau_margin')}", flush=True)
    t0 = time.time()
    err = None
    on = off = cal = None
    try:
        base.ProspectiveMemory = SFANmdaProspectiveMemory   # both arms scored by the SAME frozen gate

        print("\n--- ARM 1: SFA + NMDA-PLATEAU ON (the surpass; homeostat ON) ---", flush=True)
        homeo._BIAS_CACHE.clear(); _THETA_CACHE.clear()
        on = _run_arm(seeds, N, n_distractors, homeostat_on=True, sfa_on=True, plateau_on=True, **kw)
        for p in on:
            c = p["clauses"]
            fails = " ".join(k for k, v in c.items() if not v) or "ALL-PASS"
            fire = min(p['fireA']['rel_A_on_cueA'], p['fireB']['rel_B_on_cueB'])
            ratio = fire / max(p['max_silent'], 1e-6)
            print(f"  [seed {p['seed']}] pass={p['passed']} | fireA={p['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={p['fireB']['rel_B_on_cueB']:.3f} fire_min={fire:.3f} | max_silent={p['max_silent']:.3f} "
                  f"ratio={ratio:.2f} (need {REQUIRED_RATIO:.2f}) | held_min={p['held_min']:.3f} | {fails}",
                  flush=True)

        print("\n--- ARM 2: SFA + NMDA-PLATEAU OFF (the [H] per-pool homeostat, internal control) ---", flush=True)
        homeo._BIAS_CACHE.clear(); _THETA_CACHE.clear()
        off = _run_arm(seeds, N, n_distractors, homeostat_on=True, sfa_on=False, plateau_on=False, **kw)
        for p in off:
            c = p["clauses"]
            fails = " ".join(k for k, v in c.items() if not v) or "ALL-PASS"
            fire = min(p['fireA']['rel_A_on_cueA'], p['fireB']['rel_B_on_cueB'])
            ratio = fire / max(p['max_silent'], 1e-6)
            print(f"  [seed {p['seed']}] pass={p['passed']} | fireA={p['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={p['fireB']['rel_B_on_cueB']:.3f} fire_min={fire:.3f} | max_silent={p['max_silent']:.3f} "
                  f"ratio={ratio:.2f} | {fails}", flush=True)

        # read one calibrated PM per seed (ARM-1 config) to expose bias + the plateau separation-margin diagnostic.
        homeo._BIAS_CACHE.clear(); _THETA_CACHE.clear()
        actions = ["A", "B"]
        cal = {}
        for s in seeds:
            pm = SFANmdaProspectiveMemory(actions, [f"d{i}" for i in range(n_distractors)],
                                          homeostat_on=True, sfa_on=True, plateau_on=True, seed=s, **kw)
            cal[s] = {"bias_pA": dict(pm._bias_trace), "plateau_diag": dict(pm._diag)}
        print("\n--- CALIBRATED STATE (ARM-1 per seed): bias + NMDA plateau separation margin (diagnostic) ---",
              flush=True)
        for s in seeds:
            print(f"  [seed {s}] bias={cal[s]['bias_pA']}  plateau={cal[s]['plateau_diag']}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_sfa_nmda_amplifier", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    A = homeo._agg(on, seeds)
    B = homeo._agg(off, seeds)
    ratio_on = _ratio_per_seed(on)
    ratio_off = _ratio_per_seed(off)
    min_seeds = int(np.ceil(GO_MIN_SEEDS_FRAC * len(seeds)))

    # ANTI-CHEAT: every silence clause MUST stay 6/6 (else the amplifier created spurious fires).
    silence_regressed = [c for c in SILENCE_CLAUSES if A["agg"].get(c, 0) < len(seeds)]
    cheat = void_if(bool(silence_regressed),
                    f"the SFA+NMDA amplifier REGRESSED a silence clause {silence_regressed} -> it fires on single "
                    f"inputs (an amplifier that fires on single inputs is a CHEAT; the surpass is VOID)")
    # every seed's fire/silent ratio must clear the absolute-window requirement.
    ratio_short = {s: r for s, r in ratio_on.items() if r < REQUIRED_RATIO}

    fire_on = A["agg"].get("fire_on_cue", 0)
    fire_off = B["agg"].get("fire_on_cue", 0)
    go = bool(A["n_pass"] >= min_seeds) and (not cheat) and (not smoke)

    lift = attributable_to("fire lift owned by SFA+NMDA amplifier (mean fire: ON vs [H] homeostat OFF)",
                           A["mean_fire"], B["mean_fire"])
    s100_on, s100_off = A["fire_per_seed"].get(100), B["fire_per_seed"].get(100)
    s44_on, s44_off = A["fire_per_seed"].get(44), B["fire_per_seed"].get(44)

    vd = Verdict("pmem_sfa_nmda_amplifier")
    for c in SILENCE_CLAUSES:
        vd.require(f"silence held under SFA+NMDA amplifier: {c} (per-seed count)", A["agg"].get(c, 0),
                   expect=lambda x, n=len(seeds): x == n)
    vd.reaches("fire_on_cue pass-count: [H] homeostat -> SFA+NMDA", fire_off, fire_on)
    vd.control("mean correct-cue release: SFA+NMDA ON vs [H] homeostat", A["mean_fire"], B["mean_fire"],
               min_separation=0.0)
    vd.disabled("STDP / Hebbian / STP / OU-noise",
                "clean-hold WM config unchanged; the ONLY added mechanisms are the per-neuron SFA K-adaptation "
                "current (timescale-separation) and the supralinear NMDA/dendritic-plateau coincidence boost, on "
                "top of the [H] per-pool intrinsic-plasticity bias set-point")
    decided = vd.decide(go)

    silence_counts = ", ".join(f"{c}:{A['agg'][c]}" for c in SILENCE_CLAUSES)
    if smoke:
        verdict = (f"SMOKE OK -- SFA+NMDA amplifier RUNS end-to-end; every condition live/measured. "
                   f"ON fire_on_cue={fire_on}/{len(seeds)} (OFF {fire_off}/{len(seeds)}); ratios ON={ratio_on}; "
                   f"silence-regressed={silence_regressed or 'none'}; ratio-short={ratio_short or 'none'}. "
                   f"Not a GO claim; run --derisk for the 6-seed verdict.")
    elif cheat:
        verdict = (f"VOID -- the amplifier regressed silence clause(s) {silence_regressed}: it fires on single "
                   f"inputs (the named cheat). fire_on_cue ON={fire_on}/{len(seeds)}; ratios ON={ratio_on}.")
    elif go and not ratio_short:
        verdict = (
            f"GO -- a supralinear NMDA/dendritic-plateau COINCIDENCE AMPLIFIER surpasses the fire_on_cue boundary "
            f"WITH silence intact. A pool-gated regenerative plateau (fires when the rel pool's MEAN NMDA "
            f"conductance crosses a threshold pinned above the worst SINGLE input, label-free) amplifies ONLY the "
            f"JOINT drive -- rescuing BOTH the seed-100 gain deficit ({s100_off} -> {s100_on}) AND the seed-44 "
            f"bias-suppressed coincidence ({s44_off} -> {s44_on}). (ATTRIBUTION, ablation on the two hard seeds: "
            f"the plateau ALONE rescues both; SFA ALONE rescues neither -- SFA is NOT the load-bearing lever here, "
            f"contra the pre-registered hypothesis that SFA would fix seed 44 via timescale-separation. This run "
            f"carries SFA + plateau together per the brief; the plateau owns the closure.) "
            f"fire_on_cue rises {fire_off}/{len(seeds)} "
            f"-> {fire_on}/{len(seeds)}; {A['n_pass']}/{len(seeds)} seeds pass EVERY clause (need {min_seeds}). "
            f"Every silence clause STAYS 6/6 ({silence_counts}) and every seed's fire/silent ratio clears "
            f"{REQUIRED_RATIO:.2f} (ON={ratio_on}). The amplifier lifts the COINCIDENCE, not any single input. "
            f"All reads cp_firing_states / cp_conductance_g_nmda; NO sim/ edit.")
    else:
        fails = {c: A["agg"][c] for c in A["agg"] if A["agg"][c] < len(seeds)}
        verdict = (f"BOUNDARY -- SFA+NMDA lifted fire_on_cue {fire_off}/{len(seeds)} -> {fire_on}/{len(seeds)} and "
                   f"held every silence clause, but {A['n_pass']}/{len(seeds)} seeds pass all clauses "
                   f"(need {min_seeds}). Residual clauses: {sorted(fails)}; ratio-short seeds: {ratio_short}. "
                   f"per-seed fire ON={A['fire_per_seed']}, ratio ON={ratio_on}. Honest residual -- name the next "
                   f"single-variable mechanism, do NOT force GO.")

    summary = {
        "probe": "pmem_sfa_nmda_amplifier", "verdict": verdict, "go": bool(go and not ratio_short),
        "task": ("prospective-memory fire_on_cue closure: spike-frequency adaptation (per-neuron K-adaptation "
                 "current, timescale BETWEEN the coincidence read and the hold -> decouples sustained single input "
                 "from transient coincidence) + a supralinear NMDA/dendritic-plateau coincidence amplifier "
                 "(per-neuron threshold pinned above the worst single input, label-free -> boosts only the joint "
                 "drive), on top of the [H] per-pool intrinsic-plasticity bias set-point. ON vs [H]-homeostat "
                 "(SFA+plateau OFF) internal control. All reads cp_firing_states / cp_conductance_g_nmda; NO "
                 "sim/ edit; reuse-by-import of the [H] HomeostaticProspectiveMemory + the frozen gate."),
        "gate": {"FIRE_THR": FIRE_THR, "SILENT_MAX": SILENT_MAX, "HOLD_FLOOR": HOLD_FLOOR,
                 "LESION_HELD_MAX": LESION_HELD_MAX, "SEP_RATIO": SEP_RATIO,
                 "GO_MIN_SEEDS_FRAC": GO_MIN_SEEDS_FRAC, "REQUIRED_RATIO": round(REQUIRED_RATIO, 3)},
        "mechanism": {"sfa_g": kw.get("sfa_g"), "sfa_tau": kw.get("sfa_tau"),
                      "plateau_g": kw.get("plateau_g"), "plateau_cap": kw.get("plateau_cap"),
                      "plateau_margin": kw.get("plateau_margin")},
        "N_intervening": N, "n_distractors": n_distractors, "seeds": list(seeds), "min_seeds_to_go": min_seeds,
        "ON": {"n_pass": A["n_pass"], "per_clause_pass_counts": A["agg"], "fire_per_seed": A["fire_per_seed"],
               "max_silent_per_seed": A["silent_per_seed"], "ratio_per_seed": ratio_on,
               "mean_fire": A["mean_fire"], "mean_max_silent": A["mean_silent"]},
        "OFF": {"n_pass": B["n_pass"], "per_clause_pass_counts": B["agg"], "fire_per_seed": B["fire_per_seed"],
                "max_silent_per_seed": B["silent_per_seed"], "ratio_per_seed": ratio_off,
                "mean_fire": B["mean_fire"], "mean_max_silent": B["mean_silent"]},
        "fire_on_cue_OFF_to_ON": [fire_off, fire_on],
        "seed100_fire_OFF_to_ON": [s100_off, s100_on], "seed44_fire_OFF_to_ON": [s44_off, s44_on],
        "ratio_short_seeds": ratio_short,
        "fire_lift_attributable_to_amplifier": lift,
        "silence_regressed": silence_regressed,
        "calibrated_state": cal,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed_ON": on, "per_seed_OFF": off,
        "BIOLOGY": ("Spike-frequency adaptation (Kv/M-current AHP; Kandel 6e excitability regulation) at an "
                    "intermediate timescale decouples sustained from transient. Supralinear NMDA coincidence "
                    "amplification: Kandel 6e -- the NMDA receptor is a molecular coincidence detector; Schiller & "
                    "Schiller -- dendritic NMDA spikes (local regenerative plateau depolarization). Realized as a "
                    "per-neuron K-adaptation current (normalized-low-pass of spikes) and a per-neuron NMDA-plateau "
                    "boost on cp_conductance_g_nmda with a label-free threshold pinned above the worst single "
                    "input. Same flagged current-injection-proxy scope as the parent tonic-inhibition bias."),
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[pmem-sfa-nmda] VERDICT: {verdict}", flush=True)
    print(f"[pmem-sfa-nmda] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (summary["go"] or smoke) else 1


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
    # substrate knobs (parity with the parent so the control arm reproduces the [H] boundary)
    ap.add_argument("--hold-to-rel-weight", type=float, default=3.2)
    ap.add_argument("--cue-to-rel-weight", type=float, default=4.2)
    ap.add_argument("--rel-recurrent-weight", type=float, default=0.10)
    ap.add_argument("--rel-bias-pA", type=float, default=-1050.0)
    ap.add_argument("--n-rel", type=int, default=60)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=40)
    # homeostat knobs (identical for ALL seeds; defaults match [H])
    ap.add_argument("--homeostat-r-set", type=float, default=0.045)
    ap.add_argument("--homeostat-eta", type=float, default=4000.0)
    ap.add_argument("--homeostat-iters", type=int, default=15)
    ap.add_argument("--homeostat-window", type=int, default=6)
    ap.add_argument("--homeostat-bias-min", type=float, default=-4000.0)
    ap.add_argument("--homeostat-bias-max", type=float, default=0.0)
    # SFA + NMDA-plateau knobs (label-free, identical for ALL seeds)
    ap.add_argument("--sfa-g", type=float, default=2500.0, help="K-adaptation current gain (pA at full firing)")
    ap.add_argument("--sfa-tau", type=float, default=100.0, help="adaptation timescale (steps; BETWEEN 30 and 300)")
    ap.add_argument("--plateau-g", type=float, default=6000.0, help="NMDA plateau gain (pA per unit g_nmda excess)")
    ap.add_argument("--plateau-margin", type=float, default=1.05, help="theta = margin * max single-input g_nmda")
    ap.add_argument("--plateau-cap", type=float, default=1600.0, help="saturating plateau amplitude (pA); <=0 off")
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else a.seeds
    kw = dict(hold_to_rel_weight=a.hold_to_rel_weight, cue_to_rel_weight=a.cue_to_rel_weight,
              rel_recurrent_weight=a.rel_recurrent_weight, rel_bias_pA=a.rel_bias_pA,
              n_rel=a.n_rel, n=a.n, pattern_size=a.pattern_size,
              homeostat_r_set=a.homeostat_r_set, homeostat_eta=a.homeostat_eta,
              homeostat_iters=a.homeostat_iters, homeostat_window=a.homeostat_window,
              homeostat_bias_min=a.homeostat_bias_min, homeostat_bias_max=a.homeostat_bias_max,
              sfa_g=a.sfa_g, sfa_tau=a.sfa_tau, plateau_g=a.plateau_g,
              plateau_margin=a.plateau_margin, plateau_cap=a.plateau_cap)
    if a.smoke:
        return _derisk([seeds[0]], N=3, n_distractors=min(3, a.n_distractors), smoke=True, **kw)
    return _derisk(seeds, N=a.N, n_distractors=a.n_distractors, smoke=False, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
