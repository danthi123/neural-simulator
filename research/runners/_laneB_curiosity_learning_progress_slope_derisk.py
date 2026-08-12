"""Lane B curiosity: CPU-cheap learning-progress-slope differentiator de-risk.

This runner tests the 2026-08-02 lane-B next mechanism without editing `sim/`:
a per-concept fast-minus-slow familiarity/progress trace gates the existing
reward-omission veto. The framing is Oudeyer-Kaplan expected learning progress
implemented as an SNc/LHb-style phasic-minus-tonic read:

  progress read = post-ask familiarity (1 - novelty)
  phasic pool   = fast EMA of that read
  tonic pool    = slow EMA of that read
  slope         = phasic - tonic

The target dissociation is the residual from the reward-omission runner:
per-ask omission alone vetoes a slow-but-improving concept before mastery,
while the history/slope trace protects it. A noisy/unlearnable concept stays
flat, receives no slope protection, and is still vetoed.

Controls:
  * omission_only: ignore the slope gate. This is the failing per-ask veto.
  * slope_lesion: force the slope read to zero. This should match omission.
  * permuted_history: slow concepts read noisy traces and noisy concepts read
    slow traces. This should waste asks on noise and collapse slow mastery.
  * curiosity_lesion: zero the curiosity drive. This should stop asking.

CPU-only and lane-B-specific: imports the real Bogacz-Brown familiarity gate,
but does not import `sim` or modify any summary documents.
Run:
  python -u -m research.runners._laneB_curiosity_learning_progress_slope_derisk --smoke --out /tmp/laneB_lp_slope_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_biologize_moat_streamcodes_derisk import (  # noqa: E402
    RealAntiHebbianFamiliarity,
)


@dataclass(frozen=True)
class Config:
    d: int = 1024
    n_slow: int = 5
    n_noisy: int = 3
    n_turns: int = 400
    ask_budget: int = 120
    novel_thresh: float = 0.35
    eps: float = 0.08
    obs_noise: float = 1.00
    mod_sensor_noise: float = 0.02
    omission_lp_floor: float = 0.11
    omission_gain: float = 0.90
    omission_decay_on_progress: float = 0.18
    veto_floor: float = 1.80
    fast_alpha: float = 0.75
    slow_alpha: float = 0.16
    slope_protect_floor: float = 0.035
    confidence_mastery_floor: float = 0.50
    min_conf_rise: float = 0.15
    min_conf_vs_noise_margin: float = 0.30
    min_slope_sep_at_protect: float = 0.03


class World:
    """Host environment/teacher.

    Slow learnable concepts are fixed unit codes with high observation jitter:
    repeated imprints improve familiarity, but many asks are needed. Noisy
    concepts render a fresh random code every time and are unlearnable.
    """

    def __init__(self, seed: int, cfg: Config):
        self.rng = np.random.default_rng(seed * 7 + 1)
        self.cfg = cfg
        self.concepts = list(range(cfg.n_slow + cfg.n_noisy))
        self.is_noisy = {c: c >= cfg.n_slow for c in self.concepts}
        self._code = {}
        for c in range(cfg.n_slow):
            v = self.rng.standard_normal(cfg.d)
            self._code[c] = v / (np.linalg.norm(v) + 1e-12)

    def render(self, c: int) -> np.ndarray:
        if self.is_noisy[c]:
            v = self.rng.standard_normal(self.cfg.d)
        else:
            n = self.rng.standard_normal(self.cfg.d)
            n = n / (np.linalg.norm(n) + 1e-12) * self.cfg.obs_noise
            v = self._code[c] + n
        return v / (np.linalg.norm(v) + 1e-12)


class PhasicMinusTonicTrace:
    """Fast progress pool minus slow tonic pool."""

    def __init__(self, cfg: Config):
        self.fast = 0.0
        self.tonic = 0.0
        self.slope = 0.0
        self.cfg = cfg

    def update(self, progress_read: float) -> float:
        x = float(progress_read)
        self.fast += self.cfg.fast_alpha * (x - self.fast)
        self.tonic += self.cfg.slow_alpha * (x - self.tonic)
        self.slope = self.fast - self.tonic
        return self.slope

    def snapshot(self) -> Dict[str, float]:
        return {"fast": float(self.fast), "tonic": float(self.tonic), "slope": float(self.slope)}


def _history_read_map(c: int, cfg: Config) -> int:
    """Permuted-history anti-cheat: slow reads noisy, noisy reads slow."""
    if c < cfg.n_slow:
        return cfg.n_slow + (c % cfg.n_noisy)
    return c - cfg.n_slow


def _mean(xs: Iterable[float], default: float = 0.0) -> float:
    rows = list(xs)
    return float(np.mean(rows)) if rows else float(default)


def run(seed: int, mode: str, cfg: Config, *, verbose: bool = False) -> Dict[str, object]:
    rng = np.random.default_rng(seed * 101 + 5)
    world = World(seed, cfg)
    gate = RealAntiHebbianFamiliarity()
    concepts = world.concepts
    traces = {c: PhasicMinusTonicTrace(cfg) for c in concepts}
    omission_veto = {c: 0.0 for c in concepts}

    use_slope_gate = mode in {"real", "permuted_history"}
    curiosity_lesion = mode == "curiosity_lesion"
    slope_lesion = mode == "slope_lesion"

    corr_gap: List[float] = []
    corr_mod: List[float] = []
    asked = set()
    ask_events: List[Tuple[int, int, float, float, bool, float, float, float]] = []
    first_conf = {}
    n_asks = 0
    elig_unknown = elig_known = ask_unknown = ask_known = 0
    third = max(1, cfg.n_turns // 3)
    noisy_elig = [0, 0, 0]
    noisy_ask = [0, 0, 0]
    slow_protected_asks = 0
    noisy_protected_asks = 0
    slope_at_slow_protect: List[float] = []
    slope_at_noisy_protect: List[float] = []

    def slope_read(c: int) -> float:
        if slope_lesion:
            return 0.0
        if mode == "permuted_history":
            return traces[_history_read_map(c, cfg)].slope
        return traces[c].slope

    for turn in range(cfg.n_turns):
        if n_asks >= cfg.ask_budget:
            break

        true_gaps = {c: gate.novelty(world.render(c)) for c in concepts}
        if curiosity_lesion:
            mod = {c: 0.0 for c in concepts}
        else:
            mod = {c: float(true_gaps[c] + cfg.mod_sensor_noise * rng.standard_normal()) for c in concepts}

        for c in concepts:
            corr_gap.append(true_gaps[c])
            corr_mod.append(mod[c])
            unknown = true_gaps[c] > cfg.novel_thresh
            elig_unknown += int(unknown)
            elig_known += int(not unknown)
            if world.is_noisy[c] and unknown:
                noisy_elig[min(turn // third, 2)] += 1

        def not_vetoed(c: int) -> bool:
            if omission_veto[c] < cfg.veto_floor:
                return True
            return bool(use_slope_gate and slope_read(c) > cfg.slope_protect_floor)

        cands = [
            c for c in concepts
            if true_gaps[c] > cfg.novel_thresh and mod[c] > 1e-9 and not_vetoed(c)
        ]
        if not cands:
            continue

        if rng.random() < cfg.eps:
            c_ask = int(rng.choice(cands))
        else:
            mx = max(mod[c] for c in cands)
            c_ask = int(rng.choice([c for c in cands if mod[c] >= mx - 1e-12]))

        was_protected = (
            use_slope_gate
            and omission_veto[c_ask] >= cfg.veto_floor
            and slope_read(c_ask) > cfg.slope_protect_floor
        )
        if was_protected and world.is_noisy[c_ask]:
            noisy_protected_asks += 1
            slope_at_noisy_protect.append(slope_read(c_ask))
        elif was_protected:
            slow_protected_asks += 1
            slope_at_slow_protect.append(slope_read(c_ask))

        if true_gaps[c_ask] > cfg.novel_thresh:
            ask_unknown += 1
        else:
            ask_known += 1
        if world.is_noisy[c_ask]:
            noisy_ask[min(turn // third, 2)] += 1

        g_before = true_gaps[c_ask]
        if (not world.is_noisy[c_ask]) and c_ask not in first_conf:
            first_conf[c_ask] = 1.0 - g_before

        gate.imprint(world.render(c_ask))
        g_after = gate.novelty(world.render(c_ask))
        raw_lp = float(g_before - g_after)
        progress_read = max(0.0, 1.0 - g_after)

        if raw_lp < cfg.omission_lp_floor:
            omission_veto[c_ask] += cfg.omission_gain * (
                (cfg.omission_lp_floor - raw_lp) / cfg.omission_lp_floor
            )
        else:
            omission_veto[c_ask] = max(
                0.0,
                omission_veto[c_ask] - cfg.omission_decay_on_progress
                * ((raw_lp - cfg.omission_lp_floor) / cfg.omission_lp_floor),
            )

        if not slope_lesion:
            traces[c_ask].update(progress_read)

        asked.add(c_ask)
        n_asks += 1
        ask_events.append(
            (
                turn,
                c_ask,
                float(g_before),
                raw_lp,
                bool(world.is_noisy[c_ask]),
                float(omission_veto[c_ask]),
                float(slope_read(c_ask)),
                float(progress_read),
            )
        )

        if verbose and n_asks <= 14:
            kind = "noisy" if world.is_noisy[c_ask] else "slow"
            print(
                f"    [ask {n_asks:03d}] {kind} c={c_ask} "
                f"g {g_before:.3f}->{g_after:.3f} LP {raw_lp:+.3f} "
                f"progress {progress_read:.3f} slope {slope_read(c_ask):+.3f} "
                f"omitV {omission_veto[c_ask]:.2f} "
                f"{'PROTECT' if was_protected else ''}",
                flush=True,
            )

    corr_gap_a = np.asarray(corr_gap)
    corr_mod_a = np.asarray(corr_mod)
    corr = (
        float(np.corrcoef(corr_gap_a, corr_mod_a)[0, 1])
        if corr_mod_a.std() > 1e-9 and corr_gap_a.std() > 1e-9
        else 0.0
    )
    rate_unknown = ask_unknown / max(elig_unknown, 1)
    rate_known = ask_known / max(elig_known, 1)
    ratio_b = rate_unknown / (rate_known + 1e-9)

    conf_after = {c: 1.0 - gate.novelty(world.render(c)) for c in concepts}
    slow_conf = [conf_after[c] for c in range(cfg.n_slow)]
    noisy_conf = [conf_after[c] for c in range(cfg.n_slow, cfg.n_slow + cfg.n_noisy)]
    first_slow_conf = [first_conf.get(c, 0.0) for c in range(cfg.n_slow) if c in first_conf]
    conf_rise = _mean(slow_conf) - _mean(first_slow_conf)
    abstain_floor = _mean(noisy_conf)
    slow_mastered = int(sum(conf_after[c] > cfg.confidence_mastery_floor for c in range(cfg.n_slow)))

    late_asks = [e for e in ask_events if e[0] >= 2 * third]
    late_slow_frac = (
        sum(1 for e in late_asks if not e[4]) / len(late_asks)
        if late_asks
        else 1.0
    )
    noisy_early_rate = noisy_ask[0] / max(noisy_elig[0], 1)
    noisy_late_rate = noisy_ask[2] / max(noisy_elig[2], 1)
    noisy_gap_final = _mean(
        gate.novelty(world.render(c)) for c in range(cfg.n_slow, cfg.n_slow + cfg.n_noisy)
    )
    noisy_vetoed_frac = _mean(
        omission_veto[c] >= cfg.veto_floor for c in range(cfg.n_slow, cfg.n_slow + cfg.n_noisy)
    )
    slow_false_vetoed = int(
        sum(
            omission_veto[c] >= cfg.veto_floor
            and slope_read(c) <= cfg.slope_protect_floor
            and conf_after[c] <= cfg.confidence_mastery_floor
            for c in range(cfg.n_slow)
        )
    )
    confident_set = {c for c in concepts if conf_after[c] > cfg.confidence_mastery_floor}
    moat_ok = confident_set.issubset(asked)

    slow_events = [e for e in ask_events if not e[4]]
    noisy_events = [e for e in ask_events if e[4]]
    mean_lp_slow = _mean(e[3] for e in slow_events)
    mean_lp_noisy = _mean(e[3] for e in noisy_events)
    mean_omit_slow = _mean(omission_veto[c] for c in range(cfg.n_slow))
    mean_omit_noisy = _mean(omission_veto[c] for c in range(cfg.n_slow, cfg.n_slow + cfg.n_noisy))
    mean_slope_slow = _mean(traces[c].slope for c in range(cfg.n_slow))
    mean_slope_noisy = _mean(traces[c].slope for c in range(cfg.n_slow, cfg.n_slow + cfg.n_noisy))

    return {
        "mode": mode,
        "seed": int(seed),
        "corr_gap_mod": corr,
        "rate_unknown": rate_unknown,
        "rate_known": rate_known,
        "ratio_b": ratio_b,
        "total_asks": len(ask_events),
        "slow_mastered": slow_mastered,
        "slow_conf_mean": _mean(slow_conf),
        "conf_rise": conf_rise,
        "abstain_floor": abstain_floor,
        "noisy_conf_mean": _mean(noisy_conf),
        "noisy_gap_final": noisy_gap_final,
        "noisy_asks_total": int(sum(1 for e in ask_events if e[4])),
        "noisy_early_rate": noisy_early_rate,
        "noisy_late_rate": noisy_late_rate,
        "noisy_vetoed_frac": noisy_vetoed_frac,
        "slow_false_vetoed": slow_false_vetoed,
        "slow_protected_asks": int(slow_protected_asks),
        "noisy_protected_asks": int(noisy_protected_asks),
        "mean_slope_at_slow_protect": _mean(slope_at_slow_protect),
        "mean_slope_at_noisy_protect": _mean(slope_at_noisy_protect),
        "mean_slope_slow_final": mean_slope_slow,
        "mean_slope_noisy_final": mean_slope_noisy,
        "slope_sep_final": mean_slope_slow - mean_slope_noisy,
        "mean_omission_veto_slow": mean_omit_slow,
        "mean_omission_veto_noisy": mean_omit_noisy,
        "mean_lp_slow": mean_lp_slow,
        "mean_lp_noisy": mean_lp_noisy,
        "late_slow_frac": late_slow_frac,
        "moat_ok": bool(moat_ok),
        "trace_final": {str(c): traces[c].snapshot() for c in concepts},
    }


def evaluate(seed: int, cfg: Config, *, verbose: bool = False) -> Dict[str, object]:
    real = run(seed, "real", cfg, verbose=verbose)
    omission = run(seed, "omission_only", cfg)
    slope_lesion = run(seed, "slope_lesion", cfg)
    permuted = run(seed, "permuted_history", cfg)
    curiosity_lesion = run(seed, "curiosity_lesion", cfg)

    gate_a = real["corr_gap_mod"] >= 0.9
    gate_b = real["ratio_b"] >= 2.0
    gate_c = (
        real["conf_rise"] > cfg.min_conf_rise
        and real["slow_conf_mean"] > real["abstain_floor"] + cfg.min_conf_vs_noise_margin
    )
    slow_protected = (
        real["slow_mastered"] == cfg.n_slow
        and real["slow_protected_asks"] > 0
        and real["mean_slope_at_slow_protect"] >= cfg.min_slope_sep_at_protect
    )
    noisy_still_vetoed = (
        real["noisy_gap_final"] > 0.7
        and real["noisy_vetoed_frac"] >= 1.0
        and real["noisy_protected_asks"] == 0
    )
    omission_fails_slow = (
        omission["slow_mastered"] == 0
        and omission["total_asks"] < real["total_asks"]
        and omission["noisy_vetoed_frac"] >= 1.0
    )
    slope_lesion_collapses = (
        slope_lesion["slow_mastered"] < real["slow_mastered"]
        and slope_lesion["total_asks"] <= omission["total_asks"] + 2
    )
    permuted_history_collapses = (
        permuted["slow_mastered"] < real["slow_mastered"]
        and permuted["noisy_asks_total"] >= real["noisy_asks_total"] * 4
    )
    curiosity_lesion_collapses = (
        curiosity_lesion["total_asks"] == 0
        and curiosity_lesion["conf_rise"] < 0.05
    )

    go = bool(
        gate_a
        and gate_b
        and gate_c
        and real["moat_ok"]
        and slow_protected
        and noisy_still_vetoed
        and omission_fails_slow
        and slope_lesion_collapses
        and permuted_history_collapses
        and curiosity_lesion_collapses
    )
    return {
        "seed": int(seed),
        "real": real,
        "omission_only": omission,
        "slope_lesion": slope_lesion,
        "permuted_history": permuted,
        "curiosity_lesion": curiosity_lesion,
        "gate_a_corr": bool(gate_a),
        "gate_b_askratio": bool(gate_b),
        "gate_c_conf_rise": bool(gate_c),
        "slow_protected": bool(slow_protected),
        "noisy_still_vetoed": bool(noisy_still_vetoed),
        "omission_fails_slow": bool(omission_fails_slow),
        "slope_lesion_collapses": bool(slope_lesion_collapses),
        "permuted_history_collapses": bool(permuted_history_collapses),
        "curiosity_lesion_collapses": bool(curiosity_lesion_collapses),
        "moat_ok": bool(real["moat_ok"]),
        "GO": go,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="one-seed CPU smoke using the tuned lane-B config")
    ap.add_argument("--verbose", action="store_true", help="print first asks for the real arm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = Config(ask_budget=100 if args.smoke else Config.ask_budget)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    if args.out is None:
        args.out = "research/findings/raw/lanes/curiosity_lp_slope_derisk.json"

    print(
        "[lane-B LP-slope differentiator] CPU numpy proxy, no sim import: "
        "fast progress pool minus slow tonic pool gates the LHb/RMTg omission veto.\n"
        "  Target: slow-but-improving concepts stay askable while unlearnable noisy concepts are vetoed.\n"
        "  Controls: omission-only, slow-integrator lesion, permuted history, curiosity lesion.\n",
        flush=True,
    )

    results = []
    for seed in seeds:
        r = evaluate(seed, cfg, verbose=args.verbose)
        results.append(r)
        re = r["real"]
        om = r["omission_only"]
        sl = r["slope_lesion"]
        ph = r["permuted_history"]
        print(
            f"  [seed {seed}] corr(gap,drive) {re['corr_gap_mod']:+.3f} | "
            f"slow mastered {re['slow_mastered']}/{cfg.n_slow} "
            f"(conf {re['slow_conf_mean']:.3f} vs noisy floor {re['abstain_floor']:.3f}) | "
            f"LP slow {re['mean_lp_slow']:+.3f} vs noisy {re['mean_lp_noisy']:+.3f}",
            flush=True,
        )
        print(
            f"            veto: slow omissionV {re['mean_omission_veto_slow']:.2f} "
            f"and protected asks {re['slow_protected_asks']} "
            f"(slope@protect {re['mean_slope_at_slow_protect']:+.3f}); "
            f"noisy omissionV {re['mean_omission_veto_noisy']:.2f}, "
            f"vetoed {re['noisy_vetoed_frac']:.2f}, protected-noisy {re['noisy_protected_asks']}",
            flush=True,
        )
        print(
            f"            controls: omission-only slow mastered {om['slow_mastered']}/{cfg.n_slow} "
            f"asks {om['total_asks']} | slope-lesion mastered {sl['slow_mastered']}/{cfg.n_slow} "
            f"asks {sl['total_asks']} | permuted-history mastered {ph['slow_mastered']}/{cfg.n_slow}, "
            f"noisy asks {ph['noisy_asks_total']} vs real {re['noisy_asks_total']} | "
            f"curiosity-lesion asks {r['curiosity_lesion']['total_asks']} | moat {r['moat_ok']}",
            flush=True,
        )
        flags = (
            f"a={r['gate_a_corr']} b={r['gate_b_askratio']} c={r['gate_c_conf_rise']} "
            f"slow-protect={r['slow_protected']} noisy-veto={r['noisy_still_vetoed']} "
            f"omission-fails={r['omission_fails_slow']} slope-lesion={r['slope_lesion_collapses']} "
            f"perm-history={r['permuted_history_collapses']} curiosity-lesion={r['curiosity_lesion_collapses']}"
        )
        print(f"            [{flags}]  ==>  {'GO' if r['GO'] else 'NO'}\n", flush=True)

    n_go = sum(1 for r in results if r["GO"])
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"results": results, "config": asdict(cfg), "smoke": bool(args.smoke)},
            fh,
            indent=2,
            default=str,
        )

    print("=" * 100, flush=True)
    print(
        f"  LP-SLOPE DIFFERENTIATOR: {n_go}/{len(results)} seeds GO "
        f"({'ALL GO' if n_go == len(results) else 'partial/negative - inspect per-seed flags'})",
        flush=True,
    )
    print(f"  [saved] {args.out}\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    main()
