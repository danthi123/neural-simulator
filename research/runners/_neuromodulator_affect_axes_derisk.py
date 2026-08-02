"""Affect-axis neuromodulator de-risk (LANE A, 2026-08-02).

WHAT / WHY (roadmap §2.5 line 102 + S2 line 291). The affective world-model needs
three SLOW affect axes that the phasic reward/curiosity machinery does not yet
instantiate. This probe shows all three can be built PURELY over the EXISTING
`sim/neuromodulators.py` subsystem — with NO sim/ edit, config only — by choosing
the right (existing) production rule + decay_tau_ms regime for each:

  (1) mood            = 5-HT analog. `from_reward` (reads current_reward_signal,
                        used project-wide to carry the RPE) with a LONG
                        decay_tau_ms (~2000 ms, the slow-peptide regime that
                        already exists). The long tau makes the concentration a
                        running AVERAGE of the phasic RPE stream (Eldar-Niv
                        avg-delta mood), LAGGING the phasic DA that reads the same
                        stream with a short tau.
  (2) arousal         = NA analog. `from_surprise` (phasic on |RPE - expectation|,
                        Aston-Jones-Cohen LC) + a TONIC baseline > 0.
  (3) learning_eager. = ACh analog. `from_novelty` (reads current_novelty_signal,
                        the Bogacz-Brown familiarity-gate novelty the moat writes;
                        Yu-Dayan ACh learning-eagerness).

A phasic-DA reference axis (`da_phasic`, `from_reward`, SHORT tau) is included ONLY
to demonstrate the mood-lags-DA timescale dissociation on the identical stream.

The three affect rules only read SCALAR core_config fields (current_reward_signal /
reward_baseline / current_novelty_signal / novelty_baseline) via getattr-with-
default, so this probe drives the REAL NeuromodulatorManager with a tiny mock
bridge — no SimulationBridge, no GPU. Truly disjoint [CPU].

ADDITIVE: new file, imports sim.neuromodulators unchanged. Byte-identical when not
invoked (the subsystem is opt-in and this runner adds nothing to it).

GO gate (per seed, then 6/6):
  G1 domain-dissociation : own-is-max 3/3 — each axis's response is MAXIMAL for
                           its MATCHED statistical driver vs the other two
                           (mood<-sustained, arousal<-surprise, eager<-novelty).
  G2 permuted-selectivity: the identity axis<->driver assignment is the UNIQUE
                           permutation scoring 3/3 (all 5 others score < 3) —
                           rules out a trivial always-max metric.
  G3 lesion-collapse     : rebuilding each axis with its production sensitivity=0
                           (rule present but inert) drives its matched response to
                           <= 0.10 x the intact response — the response comes from
                           the rule, not a decay/driver artifact.
  G4 mood-lags-DA        : on an impulse-then-off stream, mood post-offset
                           retention >= 3 x the phasic-DA retention (the slow
                           peptide regime actually lags the phasic channel).

Run (1-seed CPU smoke):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._neuromodulator_affect_axes_derisk \
      --smoke --out research/findings/raw/lanes/affect_axes_smoke.json
Run (6-seed CPU GO):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._neuromodulator_affect_axes_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/affect_axes_6seed.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np

from sim.neuromodulators import (
    NeuromodulatorConfig,
    NeuromodulatorManager,
    ProductionRule,
    ModulatorTarget,
)
from tools.lab import attributable_to

# ----- axis / driver identifiers -----
AXES = ("mood", "arousal", "learning_eagerness")
DRIVERS = ("sustained", "surprise", "novelty")
# matched axis<->driver assignment (the diagonal the GO gate requires)
MATCH = {"mood": "sustained", "arousal": "surprise", "learning_eagerness": "novelty"}

# tonic baseline for the NA/arousal axis (Aston-Jones-Cohen tonic LC)
AROUSAL_TONIC = 0.2


def build_configs(dt_ms: float, lesion_axis: str | None = None):
    """Construct the four modulator configs over the EXISTING subsystem.

    lesion_axis: if given, that axis's production sensitivity is 0.0 (rule present
    but inert) — the G3 lesion-collapse control.
    """
    def s(axis, val):
        return 0.0 if lesion_axis == axis else val

    return [
        # mood = 5-HT: from_reward, LONG tau -> running average of the RPE stream.
        # concentration_min < 0 so mood can go negative (bad mood on negative RPE).
        NeuromodulatorConfig(
            name="mood",
            baseline=0.0,
            decay_tau_ms=2000.0,
            concentration_min=-5.0,
            concentration_max=5.0,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="all",
                                     sensitivity=0.0)],  # 0.0 = data-only, no bridge here
            production_rules=[ProductionRule(rule_type="from_reward",
                                             sensitivity=s("mood", 1.0))],
        ),
        # arousal = NA: from_surprise (phasic RPE) + tonic baseline.
        NeuromodulatorConfig(
            name="arousal",
            baseline=AROUSAL_TONIC,
            decay_tau_ms=300.0,
            concentration_min=0.0,
            concentration_max=5.0,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="all",
                                     sensitivity=0.0)],
            production_rules=[ProductionRule(rule_type="from_surprise",
                                             sensitivity=s("arousal", 1.0),
                                             threshold=0.15,
                                             window_ms=500.0)],
        ),
        # learning_eagerness = ACh: from_novelty (the reserved rule, now filled).
        NeuromodulatorConfig(
            name="learning_eagerness",
            baseline=0.0,
            decay_tau_ms=500.0,
            concentration_min=0.0,
            concentration_max=5.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="plastic_only",
                                     sensitivity=0.0)],
            production_rules=[ProductionRule(rule_type="from_novelty",
                                             sensitivity=s("learning_eagerness", 1.0))],
        ),
        # da_phasic = phasic DA reference: from_reward, SHORT tau (same stream).
        NeuromodulatorConfig(
            name="da_phasic",
            baseline=0.0,
            decay_tau_ms=100.0,
            concentration_min=-5.0,
            concentration_max=5.0,
            targets=[ModulatorTarget(target_type="synaptic_gain", scope="all",
                                     sensitivity=0.0)],
            production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
        ),
    ]


def make_manager(dt_ms: float, lesion_axis: str | None = None) -> NeuromodulatorManager:
    mgr = NeuromodulatorManager(build_configs(dt_ms, lesion_axis), dt_ms=dt_ms)
    mgr.initialize(n_neurons=1, cp_module=np)  # cp unused by scalar rules; numpy is honest CPU
    return mgr


def driver_stream(driver: str, n_steps: int, rng, amp: float = 1.0,
                  pulse_period_steps: int = 200):
    """Return (reward[n], novelty[n]) for one driver condition.

    sustained : constant RPE (high average, low surprise once expected) -> mood.
    surprise  : sparse UNEXPECTED RPE pulses (low average, high surprise)  -> arousal.
    novelty   : constant novelty, zero RPE                                 -> eagerness.
    Small per-seed multiplicative jitter differentiates seeds honestly.
    """
    reward = np.zeros(n_steps, dtype=np.float64)
    novelty = np.zeros(n_steps, dtype=np.float64)
    jit = 1.0 + 0.05 * rng.standard_normal(n_steps)  # ~5% per-seed jitter
    if driver == "sustained":
        reward[:] = amp * jit
    elif driver == "surprise":
        idx = np.arange(0, n_steps, pulse_period_steps)
        reward[idx] = amp * jit[idx]
    elif driver == "novelty":
        novelty[:] = amp * jit
    else:
        raise ValueError(driver)
    return reward, novelty


def run_condition(dt_ms: float, reward, novelty, lesion_axis: str | None = None):
    """Drive a fresh manager with a reward/novelty stream; return per-axis traces."""
    mgr = make_manager(dt_ms, lesion_axis)
    n = len(reward)
    names = mgr.modulator_names()
    traces = {nm: np.empty(n, dtype=np.float64) for nm in names}
    bridge = SimpleNamespace(core_config=SimpleNamespace(
        current_reward_signal=0.0, reward_baseline=0.0,
        current_novelty_signal=0.0, novelty_baseline=0.0))
    for t in range(n):
        bridge.core_config.current_reward_signal = float(reward[t])
        bridge.core_config.current_novelty_signal = float(novelty[t])
        mgr.step(bridge)
        for nm in names:
            traces[nm][t] = mgr.get_concentration(nm)
    return traces


def _baseline_of(axis: str) -> float:
    return AROUSAL_TONIC if axis == "arousal" else 0.0


def response_amplitude(trace: np.ndarray, axis: str, warmup: int) -> float:
    """Mean absolute deviation from the axis baseline over the post-warmup block.

    Uniform across integrator (mood/eager) and phasic (arousal) axes: the total
    drive delivered above/below baseline. Non-negative."""
    dev = np.abs(trace[warmup:] - _baseline_of(axis))
    return float(dev.mean())


def build_response_matrix(dt_ms: float, n_steps: int, rng, warmup: int,
                          lesion_axis: str | None = None):
    """3x3 matrix R[axis][driver] of response amplitudes."""
    R = {a: {} for a in AXES}
    for driver in DRIVERS:
        reward, novelty = driver_stream(driver, n_steps, rng)
        traces = run_condition(dt_ms, reward, novelty, lesion_axis)
        for axis in AXES:
            R[axis][driver] = response_amplitude(traces[axis], axis, warmup)
    return R


def own_is_max(R) -> tuple[int, list[str]]:
    """Count axes whose MATCHED driver is the row argmax; also return the argmax."""
    n_ok = 0
    argmax = []
    for axis in AXES:
        row = R[axis]
        best = max(row, key=row.get)
        argmax.append(best)
        if best == MATCH[axis]:
            n_ok += 1
    return n_ok, argmax


def permuted_selectivity(R) -> dict:
    """G2: identity assignment must be the UNIQUE permutation scoring 3/3.

    score(perm) = # axes for which R[axis][perm(axis)] is the row max.
    """
    perms = list(itertools.permutations(DRIVERS))
    ident = tuple(MATCH[a] for a in AXES)
    scores = {}
    for perm in perms:
        assign = dict(zip(AXES, perm))
        sc = sum(1 for a in AXES if R[a][assign[a]] == max(R[a].values()))
        scores["|".join(perm)] = sc
    ident_key = "|".join(ident)
    ident_score = scores[ident_key]
    others_max = max(v for k, v in scores.items() if k != ident_key)
    unique_perfect = (ident_score == 3) and (others_max < 3)
    return {"identity": ident_key, "identity_score": ident_score,
            "others_max_score": others_max, "unique_perfect": bool(unique_perfect),
            "all_scores": scores}


def mood_lags_da(dt_ms: float, rng, impulse_ms: float = 100.0,
                 post_ms: float = 500.0) -> dict:
    """G4: impulse-then-off; mood post-offset retention >= 3x phasic-DA retention.

    retention = |dev at offset+post_ms| / |peak dev| for each axis. mood's long tau
    keeps it elevated long after DA has returned to baseline."""
    impulse_steps = max(1, int(round(impulse_ms / dt_ms)))
    post_steps = max(1, int(round(post_ms / dt_ms)))
    n = impulse_steps + post_steps + 1
    reward = np.zeros(n, dtype=np.float64)
    reward[:impulse_steps] = 1.0
    traces = run_condition(dt_ms, reward, np.zeros(n), lesion_axis=None)
    out = {}
    for axis in ("mood", "da_phasic"):
        tr = traces[axis]
        peak = float(np.max(np.abs(tr))) or 1e-9
        offset_idx = impulse_steps  # first post-offset sample
        ret_idx = min(n - 1, offset_idx + post_steps)
        retention = float(abs(tr[ret_idx])) / peak
        out[axis] = {"peak": peak, "retention": retention}
    mood_ret = out["mood"]["retention"]
    da_ret = out["da_phasic"]["retention"]
    out["mood_lags_da"] = bool(mood_ret >= 3.0 * da_ret)
    out["ratio"] = mood_ret / (da_ret + 1e-9)
    return out


def run_seed(seed: int, dt_ms: float, n_steps: int, warmup: int) -> dict:
    rng = np.random.default_rng(seed)
    # G1: intact response matrix + domain dissociation
    R = build_response_matrix(dt_ms, n_steps, rng, warmup, lesion_axis=None)
    n_ok, argmax = own_is_max(R)
    g1 = (n_ok == 3)
    # G2: permuted selectivity
    perm = permuted_selectivity(R)
    g2 = perm["unique_perfect"]
    # G3: lesion-collapse — each axis's matched response with sensitivity=0
    lesion = {}
    g3_all = True
    for axis in AXES:
        rng_l = np.random.default_rng(seed)  # same stream jitter as intact
        Rl = build_response_matrix(dt_ms, n_steps, rng_l, warmup, lesion_axis=axis)
        intact = R[axis][MATCH[axis]]
        lesioned = Rl[axis][MATCH[axis]]
        # ATTRIBUTION (tools.lab, gap#5 lesson: measuring both arms != attributing the difference): what FRACTION
        # of the matched response is NOT present when the axis's production rule is lesioned to sensitivity=0 (rule
        # present but inert). A genuine axis => ~100% attributable (the response is produced BY the rule, not by a
        # decay/driver artifact). warn_below=-1.0 keeps a non-collapsing axis from tripping a spurious warning.
        attrib = attributable_to(f"{axis} response from its production rule", intact, lesioned, warn_below=-1.0)
        ok = lesioned <= 0.10 * intact
        lesion[axis] = {"intact": intact, "lesioned": lesioned,
                        "ratio": lesioned / (intact + 1e-12),
                        "attributable_to_rule": attrib, "collapsed": bool(ok)}
        g3_all = g3_all and ok
    # G4: mood lags DA
    tl = mood_lags_da(dt_ms, np.random.default_rng(seed))
    g4 = tl["mood_lags_da"]

    go = bool(g1 and g2 and g3_all and g4)
    return {
        "seed": seed,
        "response_matrix": R,
        "own_is_max_count": n_ok,
        "row_argmax": dict(zip(AXES, argmax)),
        "G1_domain_dissociation": bool(g1),
        "permuted": perm,
        "G2_permuted_selectivity": bool(g2),
        "lesion": lesion,
        "G3_lesion_collapse": bool(g3_all),
        "timescale": tl,
        "G4_mood_lags_da": bool(g4),
        "GO": go,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed (42), short blocks — CPU sanity only")
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=3000,
                    help="steps per driver block (>= a few mood taus)")
    ap.add_argument("--warmup", type=int, default=1500,
                    help="steps to skip before measuring the block response. MUST exceed ~3x the surprise "
                         "reward-expectation window (window_ms=500 => >=1500 steps) so the expectation EMA has "
                         "CONVERGED before the block is scored; otherwise a fully-EXPECTED sustained stream still "
                         "reads as surprising (a decaying transient) and the arousal axis fails own-is-max. The "
                         "first-attempt 200 was a measurement artifact, NOT a wrong 500ms biology (2026-08-02).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [42] if args.smoke else args.seeds
    n_steps = 800 if args.smoke else args.n_steps

    per_seed = [run_seed(s, args.dt_ms, n_steps, args.warmup) for s in seeds]
    n_go = sum(1 for r in per_seed if r["GO"])
    overall = {
        "runner": "_neuromodulator_affect_axes_derisk",
        "axes": {"mood": "5-HT/from_reward/tau2000", "arousal": "NA/from_surprise+tonic",
                 "learning_eagerness": "ACh/from_novelty"},
        "seeds": seeds,
        "dt_ms": args.dt_ms,
        "n_steps": n_steps,
        "warmup": args.warmup,
        "per_seed": per_seed,
        "n_go": n_go,
        "n_seeds": len(seeds),
        "GO_ALL": bool(n_go == len(seeds)),
        "gate": "G1 own-is-max 3/3 & G2 unique-perfect-permutation & "
                "G3 lesion<=0.10x intact & G4 mood-retention>=3x DA; 6/6 seeds",
    }

    print(f"[affect-axes] seeds={seeds} GO {n_go}/{len(seeds)}"
          f"  ALL={'GO' if overall['GO_ALL'] else 'NO-GO'}")
    for r in per_seed:
        print(f"  seed {r['seed']:>4}: GO={r['GO']}  "
              f"G1={r['G1_domain_dissociation']} G2={r['G2_permuted_selectivity']} "
              f"G3={r['G3_lesion_collapse']} G4={r['G4_mood_lags_da']}  "
              f"argmax={r['row_argmax']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(overall, f, indent=2, default=float)
        print(f"[affect-axes] wrote {args.out}")


if __name__ == "__main__":
    main()
