"""DESIGN MEASUREMENT (no brain) for webapp/awake_replay_capture.py: how much quiet rest keeps a 4-h-old fact capturable?

The real v3 ledger, the real gamma calibration, the real SleepReplayCapture night epoch and the real AwakeReplayCapture
bouts run on a FAKE substrate and a FAKE D1 pool:
  * FAKE READ-BACK. The composer's cleanup margin R is replaced by a Hill curve of the block's expressed
    increment-to-baseline ratio r = mean(weight factor) x |inc| / |base|, FITTED to the ten (r, R) pairs the committed
    seed-42 brain smokes measured (research/findings/raw/_sleep_replay_capture_smoke/seed42.json and
    research/findings/raw/_sleep_replay_capture_r2_smoke/seed42.json; the pairs are listed in the output). The one
    low-r point (r = 0.14, R = 0.0056, the Amendment-1 long-delay fact) anchors the low end; nothing between r = 0.14
    and r = 1.1 was measured, so the curve's shape there is an interpolation, declared.
  * FAKE D1: activation linear from tonic 0.5 (a = 0) to the pool's ceiling 1.24 (a = 1), no tonic noise floor.
The fact is the Amendment-1 neutral telling (r = 2.19 when fully expressed: measured, seed 42). Not a gate; not brain
evidence. It fixes the rest protocol of the pre-registration amendment and states its predictions before any brain run.

  .venv/bin/python -m research.runners._awake_replay_capture_design \
      --out research/findings/raw/_awake_replay_capture/design_fake_substrate.json
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
AWAKE_H = 4.0
R_FULL = 2.19                      # the neutral datl/datn telling's increment/baseline ratio at full expression (seed 42)
# (r, R) measured on the real composer (seed 42; the two committed smokes). R = the sleep epoch's read-back.
MEASURED = [(0.1414, 0.005609012), (1.1151, 0.306307325), (1.1389, 0.31162051), (1.292, 0.34171585),
            (1.3823, 0.356535268), (1.4612, 0.367943946), (2.0469, 0.426299712), (2.3961, 0.447724412),
            (2.654, 0.460067608), (2.7467, 0.466426613)]


def fit_hill():
    from scipy.optimize import curve_fit
    r = np.array([p[0] for p in MEASURED]); R = np.array([p[1] for p in MEASURED])

    def hill(x, Rm, c, n):
        return Rm * x ** n / (c ** n + x ** n)
    p, _ = curve_fit(hill, r, R, p0=[0.6, 1.0, 2.0], maxfev=20000)
    return [float(v) for v in p], float(np.max(np.abs(hill(r, *p) - R)))


class FakeD1:
    def __init__(self):
        from webapp import da_tag_capture as T
        self.T = T
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        from webapp.sleep_replay_capture import DA_SWR_FULL
        return float(np.clip((float(da) - self.T._DA_TONIC) / (DA_SWR_FULL - self.T._DA_TONIC), 0.0, 1.0)), None


class FakeComposer:
    """One-block store; its read-back is the fitted Hill curve of the ledger's expressed ratio (declared fake)."""

    def __init__(self, hill, D=64, seed=0):
        self.D, self.store_conns, self.hill, self.L = D, [], hill, None
        self._rng = np.random.default_rng(seed)

    def store(self, g):
        u = g * np.exp(2j * np.pi * self._rng.random(self.D))
        trig = 1000 + (len(self.store_conns) // self.D) * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(u[k])) for k in range(self.D)]

    def ratio(self, i):
        b = self.L.blocks[i]
        wf = float(np.mean(self.L.weight_factor(b)))
        return wf * float(np.mean(np.abs(b["inc"]))) / float(np.mean(np.abs(b["base"])))

    def _block_role_scores(self, i):
        Rm, c, n = self.hill
        x = self.ratio(i)
        m = Rm * x ** n / (c ** n + x ** n)
        return {"agent": ("a", 1.0, m, None), "action": ("b", 1.0, m, None), "patient": ("c", 1.0, m, None)}


def run_case(hill, rest_period_h=None, rest_from_h=0.0, rest_to_h=AWAKE_H, awake_lesion=False, sleep_lesion=False,
             da_level=0.5):
    """Tell the fact at t=0 (one turn), stay awake AWAKE_H with rest bouts every rest_period_h in [rest_from, rest_to],
    then sleep (one SWR epoch at the awake mark + onset), read at 24 h after the awake mark."""
    from webapp import da_tag_capture as T
    from webapp import sleep_replay_capture as S
    from webapp import awake_replay_capture as A
    env = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1", "BRAIN_AWAKE_REPLAY_CAPTURE": "1",
           "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1" if awake_lesion else "0",
           "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1" if sleep_lesion else "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        comp, d1 = FakeComposer(hill), FakeD1()
        L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1)
        comp.L = L
        g = R_FULL * float(np.mean(np.abs(L._baseline(0, comp.D))))
        L.observe_turn(0.0, TURN_H, da_level); comp.store(g); L.on_store(comp, 0.0)
        arc, src = A.AwakeReplayCapture(7), S.SleepReplayCapture(7, d1)
        if rest_period_h:
            k = 1
            while True:
                t = k * rest_period_h
                if t > AWAKE_H + 1e-9:
                    break
                if rest_from_h - 1e-9 <= t <= rest_to_h + 1e-9:
                    arc.maybe_bout(L, comp, t, t)          # the awake mark is at t (the body awake through t)
                k += 1
        L.advance(comp, AWAKE_H)
        e_sleep = L.early_expression(L.blocks[0])
        src.catch_up(L, comp, AWAKE_H, 1, AWAKE_H + 24.0)
        L.advance(comp, AWAKE_H + 24.0)
        s = L.summary()[0]
        ep = src.epochs[0]
        return {"rest_period_h": rest_period_h, "rest_from_h": rest_from_h, "rest_to_h": rest_to_h,
                "awake_lesion": awake_lesion, "sleep_lesion": sleep_lesion, "da_at_telling": da_level,
                "n_bouts": len(arc.bouts), "early_expression_at_4h": round(e_sleep, 6),
                "R_at_sleep_onset": ep["R"][0], "da_swr": ep["da_swr"],
                "frac_z_gt_half_28h": s["frac_synapses_z_gt_half"], "ratio_28h": round(comp.ratio(0), 6),
                "R_28h": round(comp._block_role_scores(0)["agent"][2], 6),
                "R_first_bouts": [b["R"][0] for b in arc.bouts[:3]], "R_last_bout": (arc.bouts[-1]["R"][0]
                                                                                    if arc.bouts else None)}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_awake_replay_capture/design_fake_substrate.json")
    a = ap.parse_args()
    for k in ("BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION", "BRAIN_SLEEP_DOWNSCALING"):
        os.environ.pop(k, None)
    hill, max_err = fit_hill()
    bout = 5.0 / 60.0
    rows = {"no_rest": run_case(hill, None)}
    for p_min in (5, 10, 15, 20, 30, 60):
        rows["rest_every_%dmin" % p_min] = run_case(hill, p_min / 60.0)
    rows["rest_every_5min_awake_lesion"] = run_case(hill, bout, awake_lesion=True)
    rows["rest_every_5min_sleep_lesion"] = run_case(hill, bout, sleep_lesion=True)
    rows["rest_first_hour_only"] = run_case(hill, bout, 0.0, 1.0)
    rows["rest_last_hour_only"] = run_case(hill, bout, 3.0, 4.0)
    rows["rest_last_2h_only"] = run_case(hill, bout, 2.0, 4.0)
    kept = {k: bool(v["frac_z_gt_half_28h"] > 0.5) for k, v in rows.items()}
    out = {"what": "fake-substrate design sweep: a neutral fact told 4 h before sleep onset, with quiet-rest awake "
                   "replay bouts at various schedules, then one night of the sleep route",
           "substrate": "FAKE (Hill read-back fitted to measured seed-42 (ratio, R) pairs; linear D1); not brain evidence",
           "hill_fit": {"Rmax": hill[0], "c": hill[1], "n": hill[2], "max_abs_err_on_measured": max_err,
                        "measured_pairs": MEASURED},
           "R_FULL": R_FULL, "rows": rows, "captured_by_28h": kept}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: (v["n_bouts"], v["early_expression_at_4h"], v["R_at_sleep_onset"], v["da_swr"],
                          v["frac_z_gt_half_28h"]) for k, v in rows.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
