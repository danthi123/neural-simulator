"""DESIGN MEASUREMENT (no brain) for webapp/sleep_replay_capture.py: how many SWR epochs per night?

On a FAKE substrate (a block store whose read-back is the coherence of each block's current synapses with the fact it
was written with, a stand-in for the composer's cleanup margin) and a FAKE D1 pool (activation linear from tonic 0.5 to
its ceiling 1.24), with the real v3 ledger, the real gamma calibration and the real SleepReplayCapture epoch: fact A is
told `age` hours before a fact B that is told just before sleep; the night then runs 1 or 5 SWR epochs (5 = one per
90-min NREM cycle, emulated by starting a new episode at each cycle time). Records, per age, A's read-back R at each
epoch and whether A / B are captured (fraction of synapses with z > 1/2) and still expressed (coherence) at 24 h.

This is what decided ONE epoch per night (pre-registration 2026-09-24-sleep-replay-capture-PREREGISTRATION.md): with 5
epochs a trace already at baseline is resurrected. Not a gate; not brain evidence.

  .venv/bin/python -m research.runners._sleep_replay_capture_design \
      --out research/findings/raw/_sleep_replay_capture/design_fake_substrate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REPO)

AGES_H = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
N_EPOCH_VARIANTS = [0, 1, 5]
CYCLE_H = 1.5          # Kandel 6e ch.44: each NREM/REM cycle "takes about 90 minutes" (only for the 5-epoch variant)
TURN_H = 30.0 / 3600.0


class FakeD1:
    def __init__(self):
        from webapp import da_tag_capture as T
        self.T = T
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        from webapp.sleep_replay_capture import DA_SWR_FULL
        return float(np.clip((float(da) - self.T._DA_TONIC) / (DA_SWR_FULL - self.T._DA_TONIC), 0.0, 1.0)), None


class FakeComposer:
    def __init__(self, D=64, seed=0):
        self.D, self.store_conns, self.patterns = D, [], []
        self._rng = np.random.default_rng(seed)

    def store(self):
        u = np.exp(2j * np.pi * self._rng.random(self.D))
        i = len(self.patterns)
        self.patterns.append(u)
        trig = 1000 + i * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(u[k])) for k in range(self.D)]

    def coherence(self, i):
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        return float(abs(np.mean(np.conj(self.patterns[i]) * w)) / max(1e-12, float(np.mean(np.abs(w)))))

    def _block_role_scores(self, i):
        c = self.coherence(i)
        return {"agent": ("a", 1.0, c, None), "action": ("b", 1.0, c, None), "patient": ("c", 1.0, c, None)}


def run_case(age_h, n_epochs):
    from webapp import da_tag_capture as T
    from webapp import sleep_replay_capture as S
    comp, d1 = FakeComposer(), FakeD1()
    L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1)
    src = S.SleepReplayCapture(7, d1)
    L.observe_turn(0.0, TURN_H, 0.5); comp.store(); L.on_store(comp, 0.0)             # fact A
    t_b = float(age_h)
    L.observe_turn(t_b, TURN_H, 0.5); comp.store(); L.on_store(comp, t_b)             # fact B, just before sleep
    for k in range(n_epochs):                                                          # one epoch per NREM cycle
        src.catch_up(L, comp, t_b + k * CYCLE_H, k + 1, 24.0)
    L.advance(comp, 24.0)
    s = L.summary()
    return {"age_h": age_h, "n_epochs": n_epochs, "R_A_per_epoch": [e["R"][0] for e in src.epochs],
            "R_B_epoch0": (src.epochs[0]["R"][1] if src.epochs else None),
            "A_frac_z_gt_half_24h": s[0]["frac_synapses_z_gt_half"], "B_frac_z_gt_half_24h": s[1]["frac_synapses_z_gt_half"],
            "A_coherence_24h": round(comp.coherence(0), 6), "B_coherence_24h": round(comp.coherence(1), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_sleep_replay_capture/design_fake_substrate.json")
    a = ap.parse_args()
    os.environ["BRAIN_SLEEP_REPLAY_CAPTURE"] = "1"
    for k in ("BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION"):
        os.environ.pop(k, None)
    rows = [run_case(age, n) for n in N_EPOCH_VARIANTS for age in AGES_H]
    out = {"what": "fake-substrate design sweep: capture of an older fact A vs its age at sleep onset, 0/1/5 SWR epochs",
           "substrate": "FAKE (coherence read-back, linear D1); not brain evidence", "rows": rows,
           "summary": {str(n): {"ages_A_captured": [r["age_h"] for r in rows
                                                    if r["n_epochs"] == n and r["A_frac_z_gt_half_24h"] > 0.5],
                                "B_captured": all(r["B_frac_z_gt_half_24h"] > 0.5 for r in rows if r["n_epochs"] == n)}
                       for n in N_EPOCH_VARIANTS}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
