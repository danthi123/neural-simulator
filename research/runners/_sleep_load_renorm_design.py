"""DESIGN MEASUREMENT (no brain) for BRAIN_SLEEP_LOAD_RENORM (webapp/sleep_replay_capture.py, r3): how fast does a
weak, never re-mentioned fact fade when the night's downscaling is set by how much the day added, as a function of how
many other facts the brain is told each day?

The real v3 ledger, the real gamma calibration and the real SleepReplayCapture epochs (reactivation, re-tag, SWR DA,
and the downscaling step in its constant or load mode) run on a FAKE store and a FAKE D1 pool:
  * FAKE STORE. Five build-time blocks of unit synaptic magnitude (the tiny-demo's five build facts, unmanaged; their
    magnitude is ASSUMED 1.0, the composer's unit-phasor write at gain 1) plus one block per fact told. A block's
    read-back R is the Hill curve of its expressed increment-to-baseline ratio that `_awake_replay_capture_design`
    FITS to the ten (ratio, R) pairs the committed seed-42 brain smokes measured (imported, not refitted).
  * FAKE RECALL. The brain recalls the fact while its ratio is above a boundary taken from the committed seed-42
    ten-night horizon (research/findings/raw/_sleep_replay_capture_r2_horizon_smoke/seed42/d10w_shy.json): recalled
    at ratio 0.638 (night 6), not recalled at 0.537 (night 7). The design uses the midpoint 0.588 and reports the
    nights at both ends of the band.
  * FAKE D1: linear from tonic 0.5 (a = 0) to the pool's ceiling 1.24 (a = 1).
  * NOT MODELLED: the composer's Turrigiano pass (it rescales a block's baseline and increment together, so it does
    not change a ratio, but it does change the store total W, so the fake's delta is approximate).
Every told fact has the weak telling's full-expression ratio (1.461, measured seed 42) unless stated. Not a gate, not
brain evidence: it fixes the dose protocol of Amendment 6 and states its predictions before any brain run.

  .venv/bin/python -m research.runners._sleep_load_renorm_design \
      --out research/findings/raw/_sleep_forgetting_interference/design_fake_substrate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REPO)

TURN_H = 30.0 / 3600.0
N_BUILD = 5                          # the tiny-demo's build-time facts (unmanaged blocks)
U_BUILD = 1.0                        # ASSUMED synaptic magnitude of a build-time block (unit-phasor write at gain 1)
R_WEAK = 1.461                       # weak telling: inc/base at full expression (seed 42: 1.2259 / 0.8390)
R_SALIENT = 2.7467                   # salient telling: the measured (ratio, R) pair with the highest ratio (seed 42)
RECALL_LO, RECALL_HI = 0.537, 0.638  # seed-42 d10w_shy: abstained at 0.537 (night 7), recalled at 0.638 (night 6)
RECALL_MID = 0.5 * (RECALL_LO + RECALL_HI)
DA_WEAK = [0.382, 0.507, 0.297, 0.166, 0.135]   # seed-42 d10w telling: the brain's DA on turns 1-5 (fact on turn 5)
DA_SALIENT = [1.0, 1.1, 0.5, 1.1, 1.0]           # a surprising-news telling (fact on turn 3), as in the unit tests
DA_PROBE = 0.2                       # the morning recall question (no store)
DA_FACT = 0.35                       # a plainly told later fact


class FakeStore:
    """Block-major store: N_BUILD unmanaged unit blocks + the told facts. R = Hill(ratio) of the ledger's block."""

    def __init__(self, hill, D=64, seed=0):
        self.D, self.store_conns, self.hill, self.L = D, [], hill, None
        self._rng = np.random.default_rng(seed)
        for _ in range(N_BUILD):
            self._append(U_BUILD)

    def _append(self, g):
        u = g * np.exp(2j * np.pi * self._rng.random(self.D))
        trig = 1000 + (len(self.store_conns) // self.D) * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(u[k])) for k in range(self.D)]

    def store_ratio(self, ratio):
        i = len(self.store_conns) // self.D
        base = self.L._baseline(i, self.D)
        self._append(ratio * float(np.mean(np.abs(base))))

    def ratio(self, j):                 # j = managed block index
        b = self.L.blocks[j]
        wf = self.L.weight_factor(b)
        return float(np.mean(np.abs(wf * b["inc"]))) / float(np.mean(np.abs(b["base"])))

    def _block_role_scores(self, i):
        Rm, c, n = self.hill
        x = self.ratio(i - N_BUILD)
        m = Rm * x ** n / (c ** n + x ** n)
        return {"agent": ("a", 1.0, m, None), "action": ("b", 1.0, m, None), "patient": ("c", 1.0, m, None)}


def run_case(hill, mode, k_per_day, n_nights=7, telling="weak", remention_after=(), lesion=False):
    """mode: 'none' (route only), 'const' (r2 BRAIN_SLEEP_DOWNSCALING), 'load' (r3 BRAIN_SLEEP_LOAD_RENORM)."""
    from webapp import da_tag_capture as T
    from webapp import sleep_replay_capture as S
    from research.runners._awake_replay_capture_design import FakeD1
    env = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1", "BRAIN_SLEEP_DOWNSCALING": "1" if mode == "const" else "0",
           "BRAIN_SLEEP_LOAD_RENORM": "1" if mode == "load" else "0",
           "BRAIN_SLEEP_LOAD_RENORM_LESION": "1" if lesion else "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        d1 = FakeD1()
        L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1, block_offset=N_BUILD)
        comp = FakeStore(hill)
        comp.L = L
        src = S.SleepReplayCapture(7, d1)
        clock = {"t": 0.0, "n": 0, "t_last": 0.0}
        fact_blocks = []

        def turn(da, store_ratio=None, fact=False):
            t = clock["t"]
            src.catch_up(L, comp, clock["t_last"], clock["n"], t) if clock["n"] > 0 else None
            L.advance(comp, max(t, L.t))
            L.observe_turn(max(t, L.t), TURN_H, da)
            if store_ratio is not None:
                comp.store_ratio(store_ratio)
                L.on_store(comp, max(t, L.t))
                if fact:
                    fact_blocks.append(len(L.blocks) - 1)
            clock["t_last"] = max(t, L.t)
            clock["n"] += 1
            clock["t"] = clock["t_last"] + TURN_H

        das = DA_WEAK if telling == "weak" else DA_SALIENT
        fact_turn = 4 if telling == "weak" else 2
        r_fact = R_WEAK if telling == "weak" else R_SALIENT
        for i, da in enumerate(das):
            turn(da, r_fact if i == fact_turn else None, fact=(i == fact_turn))
        nights = []
        for n in range(1, n_nights + 1):
            clock["t"] = clock["t_last"] + 24.0                        # the environment's night
            src.catch_up(L, comp, clock["t_last"], clock["n"], clock["t"])
            L.advance(comp, clock["t"])
            ep = src.epochs[-1] if src.epochs else {}
            ratios = [comp.ratio(j) for j in fact_blocks]
            nights.append({"night": n, "fact_ratio": [round(r, 6) for r in ratios],
                           "fact_R": [round(ep.get("R", [None] * 99)[j], 6) for j in fact_blocks],
                           "delta": (ep.get("load") or {}).get("delta",
                                                               S.SHY_DELTA if mode == "const" else 0.0),
                           "dW": (ep.get("load") or {}).get("dW"), "W": (ep.get("load") or {}).get("W"),
                           "n_blocks": len(L.blocks),
                           "recalled_mid": bool(max(ratios) > RECALL_MID),
                           "recalled_lo": bool(max(ratios) > RECALL_LO),
                           "recalled_hi": bool(max(ratios) > RECALL_HI)})
            turn(DA_PROBE)                                                # the morning recall question (no store)
            if n < n_nights:
                if n in remention_after:
                    turn(DA_FACT, R_WEAK, fact=True)
                for _ in range(k_per_day):
                    turn(DA_FACT, R_WEAK)

        def first_lost(key):
            return next((x["night"] for x in nights if not x[key]), None)
        return {"mode": mode, "k_per_day": k_per_day, "telling": telling, "remention_after": list(remention_after),
                "lesion": lesion, "n_epochs": len(src.epochs), "nights": nights,
                "first_night_not_recalled": {"mid": first_lost("recalled_mid"), "band_lo_boundary":
                                             first_lost("recalled_lo"), "band_hi_boundary": first_lost("recalled_hi")},
                "n_told_facts": len(L.blocks) - len(fact_blocks)}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_sleep_forgetting_interference/design_fake_substrate.json")
    a = ap.parse_args()
    for k in ("BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION", "BRAIN_SLEEP_REPLAY_CAPTURE_LESION"):
        os.environ.pop(k, None)
    from research.runners._awake_replay_capture_design import fit_hill, MEASURED
    hill, max_err = fit_hill()
    rows = {
        "route_only_vacuum_10n": run_case(hill, "none", 0, 10),
        "const_vacuum_10n": run_case(hill, "const", 0, 10),       # the committed d10w_shy protocol (fake replica)
        "const_k3_7n": run_case(hill, "const", 3),                # the registered REPORTED arm fih_shy
    }
    for k in (0, 1, 2, 3, 4):
        rows["load_k%d_7n" % k] = run_case(hill, "load", k)
    rows["load_k3_7n_lesion"] = run_case(hill, "load", 3, lesion=True)
    rows["load_k3_7n_salient"] = run_case(hill, "load", 3, telling="salient")
    rows["load_k3_7n_remention12"] = run_case(hill, "load", 3, remention_after=(1, 2))
    rows["load_k2_10n"] = run_case(hill, "load", 2, 10)
    out = {"what": "fake-substrate design sweep: a weak fact told once, then n nights; on each later day k other "
                   "facts are told; the night's downscaling is off, constant (r2) or set by the day's load (r3)",
           "substrate": "FAKE (Hill read-back fitted to measured seed-42 (ratio, R) pairs; linear D1; five unit "
                        "build-time blocks; no Turrigiano pass); not brain evidence",
           "hill_fit": {"Rmax": hill[0], "c": hill[1], "n": hill[2], "max_abs_err_on_measured": max_err,
                        "measured_pairs": MEASURED},
           "recall_boundary": {"lo": RECALL_LO, "hi": RECALL_HI, "mid": RECALL_MID,
                               "source": "research/findings/raw/_sleep_replay_capture_r2_horizon_smoke/seed42/"
                                         "d10w_shy.json (nights 6 and 7)"},
           "constants": {"N_BUILD": N_BUILD, "U_BUILD": U_BUILD, "R_WEAK": R_WEAK, "R_SALIENT": R_SALIENT,
                         "DA_FACT": DA_FACT, "DA_PROBE": DA_PROBE},
           "rows": rows,
           "summary": {k: {"first_night_not_recalled": v["first_night_not_recalled"],
                           "ratio_by_night": [x["fact_ratio"] for x in v["nights"]],
                           "delta_by_night": [x["delta"] for x in v["nights"]]} for k, v in rows.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
