"""Lane B curiosity: learning-progress-MAXIMIZING ask SELECTION de-risk (CPU numpy proxy).

Re-anchor (2026-08-07): every existing lane-B runner selects the ask by ``argmax want``
(a novelty-driven VTA/striosome salience read). Learning progress (LP) is used ONLY as a
protective VETO GATE (`--lp-slope`, omission-veto protection). The CORE Oudeyer-Kaplan
intrinsic-motivation thesis -- that the drive should be to MAXIMISE learning progress
itself (pick what improves competence fastest), which is immune to the "noisy-TV" trap by
construction -- has never been tested as the ASK-SELECTION drive here. This runner de-risks
that distinct mechanism, CPU-cheap, no `sim/` import.

Contrast (proactive vs reactive):
  * Novelty-max selection (the current default) is CAPTURED by unlearnable high-novelty
    stimuli: a noisy concept renders a fresh random code every time, so its novelty stays
    ~maximal forever. Pure novelty curiosity keeps re-asking it and only escapes via the
    reactive omission veto (a separate, already-GO mechanism).
  * LP-max selection targets the LEARNABLE FRONTIER: a concept's expected LP (phasic-minus-
    tonic progress slope) is positive only while it is actually improving. Noisy concepts
    have ~zero LP, so LP-max never wastes budget on them -- no veto required.

Arms:
  real        : exploit = max(0, LP slope)          -- learning-progress maximisation
  novelty_max : exploit = novelty (the current selector's drive)     [like-for-like control]
  lp_lesion   : exploit = 0 (pure count-based explore bonus)         [LP removed -> uniform]
  permuted_lp : exploit = LP slope read from a MIS-ASSIGNED concept  [LP-specificity anti-cheat]
  novelty_min : exploit = -novelty (avoid novelty)   [shows LP != "just avoid novel"]

All arms share an identical count-based exploration bonus (novelty-agnostic) so the ONLY
difference is the exploitation term. The real familiarity gate is the imported Bogacz-Brown
anti-Hebbian model; the LP traces are numpy EMAs (this is a CPU proxy, like the LP-slope
differentiator that preceded it -- the on-bridge realisation is the next step IF this holds).

Run:
  env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_selection_derisk \
    --smoke --out research/findings/raw/lanes/curiosity/lp_max_selection_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List

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
    d: int = 512
    n_fast: int = 3          # learnable, low observation jitter (few asks to master)
    n_slow: int = 3          # learnable, high observation jitter (many asks to master)
    n_noisy: int = 3         # unlearnable: fresh random code every render
    n_turns: int = 600
    ask_budget: int = 90
    eps: float = 0.06
    fast_alpha: float = 0.55
    slow_alpha: float = 0.12
    explore_beta: float = 0.20   # count-based exploration bonus weight (identical across arms)
    fast_obs_noise: float = 0.28
    slow_obs_noise: float = 1.05
    mastery_conf_floor: float = 0.50


class World:
    """Host environment/teacher: fast-learnable, slow-learnable, and unlearnable-noisy concepts."""

    def __init__(self, seed: int, cfg: Config):
        self.rng = np.random.default_rng(seed * 7 + 1)
        self.cfg = cfg
        self.n_learn = cfg.n_fast + cfg.n_slow
        self.concepts = list(range(self.n_learn + cfg.n_noisy))
        self.is_noisy = {c: c >= self.n_learn for c in self.concepts}
        self.is_fast = {c: c < cfg.n_fast for c in self.concepts}
        self._code = {}
        for c in range(self.n_learn):
            v = self.rng.standard_normal(cfg.d)
            self._code[c] = v / (np.linalg.norm(v) + 1e-12)

    def obs_noise(self, c: int) -> float:
        return self.cfg.fast_obs_noise if self.is_fast[c] else self.cfg.slow_obs_noise

    def render(self, c: int) -> np.ndarray:
        if self.is_noisy[c]:
            v = self.rng.standard_normal(self.cfg.d)
        else:
            n = self.rng.standard_normal(self.cfg.d)
            n = n / (np.linalg.norm(n) + 1e-12) * self.obs_noise(c)
            v = self._code[c] + n
        return v / (np.linalg.norm(v) + 1e-12)


class Trace:
    """Phasic (fast EMA) minus tonic (slow EMA) of the progress read = LP slope."""

    def __init__(self, cfg: Config):
        self.fast = 0.0
        self.tonic = 0.0
        self.cfg = cfg

    def update(self, progress_read: float) -> None:
        x = float(progress_read)
        self.fast += self.cfg.fast_alpha * (x - self.fast)
        self.tonic += self.cfg.slow_alpha * (x - self.tonic)

    @property
    def slope(self) -> float:
        return self.fast - self.tonic


def _permute_map(c: int, cfg: Config, n_learn: int) -> int:
    """Anti-cheat: learnable concepts read a noisy trace, noisy concepts read a learnable trace."""
    if c < n_learn:
        return n_learn + (c % cfg.n_noisy)
    return (c - n_learn) % n_learn


def run(seed: int, mode: str, cfg: Config, *, verbose: bool = False) -> Dict[str, object]:
    rng = np.random.default_rng(seed * 101 + 5)
    world = World(seed, cfg)
    gate = RealAntiHebbianFamiliarity()
    concepts = world.concepts
    n_learn = world.n_learn
    traces = {c: Trace(cfg) for c in concepts}
    count = {c: 0 for c in concepts}

    n_asks = 0
    ask_class = {"fast": 0, "slow": 0, "noisy": 0}
    early_third = max(1, cfg.ask_budget // 3)
    early_noisy = 0
    asks_to_master_all = cfg.ask_budget  # default: never
    mastered_at = {}

    def conf(c: int, k: int = 1) -> float:
        # Mastery is a terminal read; average over k renders so a concept sitting right at the
        # conf~=0.5 floor is not counted differently between calls due to single-render jitter.
        return float(np.mean([1.0 - gate.novelty(world.render(c)) for _ in range(k)]))

    def exploit(c: int) -> float:
        if mode == "novelty_max":
            return float(gate.novelty(world.render(c)))
        if mode == "novelty_min":
            return -float(gate.novelty(world.render(c)))
        if mode == "lp_lesion":
            return 0.0
        if mode == "permuted_lp":
            return max(0.0, traces[_permute_map(c, cfg, n_learn)].slope)
        # real: learning-progress maximisation
        return max(0.0, traces[c].slope)

    for _turn in range(cfg.n_turns):
        if n_asks >= cfg.ask_budget:
            break

        # candidate score: arm-specific exploitation + identical count-based exploration bonus
        scores = {
            c: exploit(c) + cfg.explore_beta / np.sqrt(count[c] + 1.0)
            for c in concepts
        }
        if rng.random() < cfg.eps:
            c_ask = int(rng.choice(concepts))
        else:
            mx = max(scores.values())
            c_ask = int(rng.choice([c for c in concepts if scores[c] >= mx - 1e-12]))

        g_before = gate.novelty(world.render(c_ask))
        gate.imprint(world.render(c_ask))
        g_after = gate.novelty(world.render(c_ask))
        progress_read = max(0.0, 1.0 - g_after)
        traces[c_ask].update(progress_read)
        count[c_ask] += 1

        if world.is_noisy[c_ask]:
            ask_class["noisy"] += 1
            if n_asks < early_third:
                early_noisy += 1
        elif world.is_fast[c_ask]:
            ask_class["fast"] += 1
        else:
            ask_class["slow"] += 1

        n_asks += 1

        # mastery bookkeeping
        for c in range(n_learn):
            if c not in mastered_at and conf(c) > cfg.mastery_conf_floor:
                mastered_at[c] = n_asks
        if len(mastered_at) == n_learn and asks_to_master_all == cfg.ask_budget:
            asks_to_master_all = n_asks

        if verbose and n_asks <= 18:
            kind = "noisy" if world.is_noisy[c_ask] else ("fast" if world.is_fast[c_ask] else "slow")
            print(f"    [ask {n_asks:03d}] {kind} c={c_ask} g {g_before:.3f}->{g_after:.3f} "
                  f"slope {traces[c_ask].slope:+.3f} score {scores[c_ask]:+.3f}", flush=True)

    K_EVAL = 8
    is_master = {c: conf(c, K_EVAL) > cfg.mastery_conf_floor for c in range(n_learn)}
    learn_mastered = int(sum(is_master.values()))
    fast_mastered = int(sum(is_master[c] for c in range(cfg.n_fast)))
    slow_mastered = int(sum(is_master[c] for c in range(cfg.n_fast, n_learn)))
    noisy_conf = float(np.mean([conf(c, K_EVAL) for c in range(n_learn, n_learn + cfg.n_noisy)]))

    return {
        "mode": mode,
        "seed": int(seed),
        "total_asks": n_asks,
        "ask_fast": ask_class["fast"],
        "ask_slow": ask_class["slow"],
        "ask_noisy": ask_class["noisy"],
        "early_noisy_asks": int(early_noisy),
        "noisy_ask_frac": float(ask_class["noisy"] / max(n_asks, 1)),
        "learn_mastered": learn_mastered,
        "fast_mastered": fast_mastered,
        "slow_mastered": slow_mastered,
        "all_learn_mastered": bool(learn_mastered == n_learn),
        "asks_to_master_all": int(asks_to_master_all),
        "noisy_conf_mean": noisy_conf,
        "final_slope_fast": float(np.mean([traces[c].slope for c in range(cfg.n_fast)])),
        "final_slope_slow": float(np.mean([traces[c].slope for c in range(cfg.n_fast, n_learn)])),
        "final_slope_noisy": float(np.mean([traces[c].slope for c in range(n_learn, n_learn + cfg.n_noisy)])),
    }


def evaluate(seed: int, cfg: Config, *, verbose: bool = False) -> Dict[str, object]:
    real = run(seed, "real", cfg, verbose=verbose)
    nov = run(seed, "novelty_max", cfg)
    lesion = run(seed, "lp_lesion", cfg)
    perm = run(seed, "permuted_lp", cfg)
    nmin = run(seed, "novelty_min", cfg)

    n_learn = cfg.n_fast + cfg.n_slow

    # G1: LP-max masters ALL learnable concepts within budget.
    g1_mastery = real["all_learn_mastered"]
    # G2: LP-max wastes far fewer asks on unlearnable noise than novelty-max (proactive noisy-TV immunity).
    g2_noise = real["ask_noisy"] <= 0.5 * max(nov["ask_noisy"], 1)
    # G3: LP-max reaches full mastery more efficiently than novelty-max (fewer asks, or novelty-max never does).
    g3_efficiency = (
        real["all_learn_mastered"]
        and (not nov["all_learn_mastered"] or real["asks_to_master_all"] < nov["asks_to_master_all"])
    )
    # G4: LP is load-bearing -- removing it (uniform) loses the noise-avoidance advantage.
    g4_lp_loadbearing = lesion["ask_noisy"] >= 1.5 * max(real["ask_noisy"], 1)
    # G5: LP-specificity anti-cheat -- mis-assigning the trace redirects budget onto noise and/or breaks mastery.
    g5_specificity = (
        perm["ask_noisy"] >= 1.5 * max(real["ask_noisy"], 1)
        or perm["learn_mastered"] < real["learn_mastered"]
    )

    go = bool(g1_mastery and g2_noise and g3_efficiency and g4_lp_loadbearing and g5_specificity)
    return {
        "seed": int(seed),
        "real": real,
        "novelty_max": nov,
        "lp_lesion": lesion,
        "permuted_lp": perm,
        "novelty_min": nmin,
        "g1_mastery": bool(g1_mastery),
        "g2_noise_avoidance": bool(g2_noise),
        "g3_efficiency": bool(g3_efficiency),
        "g4_lp_loadbearing": bool(g4_lp_loadbearing),
        "g5_lp_specificity": bool(g5_specificity),
        "GO": go,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="one-seed CPU smoke")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = Config()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    if args.out is None:
        args.out = "research/findings/raw/lanes/curiosity/lp_max_selection_derisk.json"

    print(
        "[lane-B LP-max SELECTION] CPU numpy proxy, no sim import: the ask is chosen by "
        "MAXIMISING learning-progress slope (Oudeyer-Kaplan), not novelty salience.\n"
        "  Target: allocate the ask budget to the LEARNABLE FRONTIER; ignore unlearnable noise "
        "WITHOUT a reactive veto.\n"
        "  Controls: novelty-max (current selector), LP-lesion (uniform), permuted-LP (anti-cheat), "
        "novelty-min.\n",
        flush=True,
    )

    results = []
    for seed in seeds:
        r = evaluate(seed, cfg, verbose=args.verbose)
        results.append(r)
        re, nv, le, pe, nm = r["real"], r["novelty_max"], r["lp_lesion"], r["permuted_lp"], r["novelty_min"]
        print(
            f"  [seed {seed}] REAL(lp-max): learn mastered {re['learn_mastered']}/{cfg.n_fast + cfg.n_slow} "
            f"(fast {re['fast_mastered']}/{cfg.n_fast}, slow {re['slow_mastered']}/{cfg.n_slow}) | "
            f"asks fast/slow/noisy {re['ask_fast']}/{re['ask_slow']}/{re['ask_noisy']} | "
            f"asks-to-master-all {re['asks_to_master_all']} | slope f/s/n "
            f"{re['final_slope_fast']:+.3f}/{re['final_slope_slow']:+.3f}/{re['final_slope_noisy']:+.3f}",
            flush=True,
        )
        print(
            f"            novelty-max: mastered {nv['learn_mastered']}/{cfg.n_fast + cfg.n_slow}, "
            f"noisy asks {nv['ask_noisy']}, to-master {nv['asks_to_master_all']} | "
            f"lp-lesion: noisy {le['ask_noisy']}, mastered {le['learn_mastered']} | "
            f"perm-lp: noisy {pe['ask_noisy']}, mastered {pe['learn_mastered']} | "
            f"nov-min: noisy {nm['ask_noisy']}, mastered {nm['learn_mastered']}",
            flush=True,
        )
        flags = (
            f"g1-mastery={r['g1_mastery']} g2-noise={r['g2_noise_avoidance']} "
            f"g3-eff={r['g3_efficiency']} g4-lp-load={r['g4_lp_loadbearing']} "
            f"g5-specificity={r['g5_lp_specificity']}"
        )
        print(f"            [{flags}]  ==>  {'GO' if r['GO'] else 'NO'}\n", flush=True)

    n_go = sum(1 for r in results if r["GO"])
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "config": asdict(cfg), "smoke": bool(args.smoke)},
                  fh, indent=2, default=str)

    print("=" * 100, flush=True)
    print(
        f"  LP-MAX SELECTION: {n_go}/{len(results)} seeds GO "
        f"({'ALL GO' if n_go == len(results) else 'partial/negative - inspect per-seed flags'})",
        flush=True,
    )
    print(f"  [saved] {args.out}\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    main()
