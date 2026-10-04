import os, sys, json, dataclasses
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim.backend import get_backend, to_host
from research.runners._vocal_action_selector_gate import selector_config, _indices, _step, DIRECT_PATH_GATE
import _nch_sel as S

def run(n, xw, seed=7, steps=600, arousal=True):
    cfg = dataclasses.replace(selector_config("v2"), commit_fsi_cross_weight=float(xw))
    b = S.build_nchannel_bridge(n, seed, cfg)
    ch = tuple(range(n))
    S._set_nchannel_tonic(b, cfg, ch); _step(b, cfg.warmup_steps)
    xp, _ = get_backend()
    S._set_nchannel_tonic(b, cfg, ch)
    if arousal:
        b.cp_external_input_current[xp.asarray(_indices(b, "practice_arousal"))] = xp.float32(cfg.practice_pA)
    names = [f"{r}_{c}" for c in ch for r in ("commit", "commit_fs", "motor", "thal")]
    idx = {nm: _indices(b, nm) for nm in names}
    cnt = {nm: 0 for nm in names}
    vmin = []
    cidx = idx["commit_0"]
    for t in range(steps):
        _step(b)
        f = np.asarray(to_host(b.cp_firing_states), dtype=bool)
        for nm, ii in idx.items(): cnt[nm] += int(f[ii].sum())
        if hasattr(b, "cp_membrane_potential_v") and t < 60:
            vmin.append(float(np.min(to_host(b.cp_membrane_potential_v)[cidx])))
    r = lambda k: [round(cnt[f"{k}_{c}"] / (len(idx[f"{k}_{c}"]) * steps), 3) for c in ch]
    return {"n": n, "xw": xw, "arousal": arousal, "commit": r("commit"), "commit_fs": r("commit_fs"), "thal": r("thal"),
            "motor": r("motor"), "commit0_vmin_first60": (round(min(vmin), 1) if vmin else None)}

if __name__ == "__main__":
    for args in [(2, 40.0), (5, 40.0), (5, 10.0), (5, 0.0), (3, 40.0), (5, 40.0, 7, 600, False)]:
        print(json.dumps(run(*args)), flush=True)
