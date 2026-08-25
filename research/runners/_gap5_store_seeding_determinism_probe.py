"""SCRATCH: does cfg.seed control the gap5 decoupled store build? Build the SAME seed TWICE in ONE process and compare
(a) cp_neuron_firing_thresholds hash, (b) cp_connections.data hash after encode, (c) the fb_drive=10 readout member_frac.
If build2 != build1, the substrate is NOT fully seeded (each build advances a global RNG) -> the 'member_frac 0.17 vs
0.38' cross-process discrepancy is that bug, and the 6-seed runs are each on a partially-random store."""
import sys, hashlib
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np
from sim.backend import get_backend, to_host
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG
from research.runners._gap5_dg_detonator_ignition_derisk import _rest_and_detonate, _score

def h(arr):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(to_host(arr))).tobytes()).hexdigest()[:12]

cfg = {**DECOUPLED_CFG, "n_ca3": 2000, "n_mem": 3, "freeze_between_refresh": True}
SEED = 42
res = []
for build in (1, 2):
    prep = _prepare_sequence(SEED, cfg, do_encode=True)
    br = prep["bridge"]
    th = h(br.cp_neuron_firing_thresholds) if getattr(br, "cp_neuron_firing_thresholds", None) is not None else "n/a"
    cw = h(br.cp_connections.data)
    al = prep["assemblies_local"]
    r = _rest_and_detonate(prep, ("assembly", 0, 0.15, 3000.0, 15), 700, SEED, 0.1, adapt=True, d_abs=40.0,
                           a_abs=0.008, det_period=150, det_settle=50, apical_gc_read=None, fb_read=None, fb_drive=10.0)
    ev, seq = _score(r["F"], al, al, 0, SEED, 5, 0.5, 4.0, 0.30, 0.12, 0.08)
    res.append((th, cw, ev["member_frac"], ev["n_events"], seq["forward_frac"], r["basket_mean"]))
    print(f"build{build}: thresh={th} conn={cw} member_frac={ev['member_frac']:.4f} n_events={ev['n_events']} "
          f"FWD={seq['forward_frac']:.3f} basket={r['basket_mean']:.4f}", flush=True)
print()
print("SUBSTRATE DETERMINISTIC (thresh+conn identical build1==build2):",
      res[0][0] == res[1][0] and res[0][1] == res[1][1])
print("READOUT member_frac identical:", abs(res[0][2] - res[1][2]) < 1e-9)
