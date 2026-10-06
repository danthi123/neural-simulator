"""Review probe (DEV seed 7, numpy): per-region firing of the local model's N-channel build at N=2 vs N=5,
default v2 tonic (gpi=1000/thal=900), equal saliences, full action window (no early break).
Question: does the shared STN's drive onto each GPi fall with N (GPe->STN inhibition summed over N channels),
i.e. is the N=5 saturation the missing STN/GPe capacity scaling (Gurney-Prescott-Redgrave 2001)?"""
import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim.backend import get_backend, to_host
from research.runners._vocal_action_selector_gate import selector_config, _indices, _step, DIRECT_PATH_GATE
import _nch_sel as S

def run(n, seed=7, steps=600, gpi=None, thal=None):
    import dataclasses
    cfg = selector_config("v2")
    if gpi is not None:
        cfg = dataclasses.replace(cfg, gpi_tonic_pA=gpi, thalamus_tonic_pA=thal)
    b = S.build_nchannel_bridge(n, seed, cfg)
    b.set_transmission_gate(DIRECT_PATH_GATE, 1.0)
    ch = tuple(range(n))
    S._set_nchannel_tonic(b, cfg, ch)
    _step(b, cfg.warmup_steps)
    xp, _ = get_backend()
    S._set_nchannel_tonic(b, cfg, ch)
    b.cp_external_input_current[xp.asarray(_indices(b, "practice_arousal"))] = xp.float32(cfg.practice_pA)
    names = ["selector_stn"] + [f"{r}_{c}" for c in ch for r in ("str_d1", "gpe", "gpi", "thal", "commit", "motor")]
    idx = {nm: _indices(b, nm) for nm in names}
    cnt = {nm: 0 for nm in names}
    for _ in range(steps):
        _step(b)
        f = np.asarray(to_host(b.cp_firing_states), dtype=bool)
        for nm, ii in idx.items():
            cnt[nm] += int(f[ii].sum())
    # per-neuron-per-step rate
    rate = {nm: round(cnt[nm] / (len(idx[nm]) * steps), 4) for nm in names}
    out = {"n": n, "stn": rate["selector_stn"]}
    for r in ("str_d1", "gpe", "gpi", "thal", "commit", "motor"):
        out[r] = [rate[f"{r}_{c}"] for c in ch]
    return out

if __name__ == "__main__":
    res = [run(2), run(5)]
    print(json.dumps(res, indent=1))
