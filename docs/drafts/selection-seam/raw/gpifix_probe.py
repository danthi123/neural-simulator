import os, sys, json, dataclasses
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim.backend import get_backend, to_host
from research.runners._vocal_action_selector_gate import selector_config, _indices, _step
import _nch_sel as S
def run(gpi, thal, xw=40.0, seed=7, steps=600):
    cfg = dataclasses.replace(selector_config("v2"), gpi_tonic_pA=gpi, thalamus_tonic_pA=thal, commit_fsi_cross_weight=xw)
    b = S.build_nchannel_bridge(5, seed, cfg); ch = tuple(range(5))
    S._set_nchannel_tonic(b, cfg, ch); _step(b, cfg.warmup_steps)
    xp, _ = get_backend(); S._set_nchannel_tonic(b, cfg, ch)
    b.cp_external_input_current[xp.asarray(_indices(b, "practice_arousal"))] = xp.float32(cfg.practice_pA)
    idx = {n: _indices(b, n) for n in [f"{r}_{c}" for c in ch for r in ("commit", "commit_fs", "thal")]}
    cnt = {n: 0 for n in idx}; vmin = 0.0
    allc = np.concatenate([idx[f"commit_{c}"] for c in ch])
    for t in range(steps):
        _step(b); f = np.asarray(to_host(b.cp_firing_states), dtype=bool)
        for n, ii in idx.items(): cnt[n] += int(f[ii].sum())
        vmin = min(vmin, float(np.min(to_host(b.cp_membrane_potential_v)[allc])))
    r = lambda k: [round(cnt[f"{k}_{c}"] / (len(idx[f"{k}_{c}"]) * steps), 3) for c in ch]
    return {"gpi": gpi, "thal": thal, "xw": xw, "commit": r("commit"), "commit_fs": r("commit_fs"), "thal_rate": r("thal"), "commit_vmin": round(vmin, 1)}
for a in [(1000, 900), (500, 45), (50, 45)]:
    print(json.dumps(run(*a)), flush=True)
