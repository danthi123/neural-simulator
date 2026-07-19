"""DECISIVE isolation for rung 3 (homeostatic op-point): is the objrel seed-fragility an OPERATING-POINT problem (fixable
by per-draw floor calibration) or deeper (the signed deploy can't do it regardless of weights)? Take the FIXED ridge Ws
(the structural-optimal weights that read objrel LINEARLY at 1.00 every seed) and SWEEP (floor x scale) per seed on the
DEGRADED draws 44/100. Pick the combo that maximizes CANONICAL (a fair operating point, no objrel-label leakage), report
OBJREL at it. If a calibrated op-point makes objrel work on 44/100 -> operating-point fragility -> HOMEOSTATIC floor is the
fix. If no (floor,scale) recovers objrel -> the signed deploy is fundamentally limited -> rung 4 (phase-domain read)."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_signed_readout_derisk as SR
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 44
N_TEST = 12
FLOORS = [15.0, 25.0, 40.0, 60.0, 90.0]
SCALES_C = [60.0, 90.0, 130.0, 180.0, 240.0]
corpus = SR.setup_corpus(seed=seed)
disc, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(disc)
rng = np.random.default_rng(seed * 101 + 5)
from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import N_TRAIN_PER, _fit_Ws_spiking
train = _gen(_TRAIN_KINDS, N_TRAIN_PER, rng, subj, verb, obj)

ub0 = SR._build_bridge(seed, corpus)
ens0, relay0 = SR.wire_signed_readout(ub0, np.arange(int(ub0.reservoir_base), int(ub0.reservoir_base) + SR.RES_N), seed)
res0, res_idx0, _w0 = SR._make_reservoir(ub0, enc, seed)
SR._wire_relay_ie(ub0, relay0, ens0)
res0.snapshot_after_wiring()
Ws = _fit_Ws_spiking(res0, enc, train)
n_res = len(res_idx0)
trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
proj = max(1e-9, float((res0.final_state(enc.encode(canon[0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max()))


def deploy(scale, facts):
    ub = SR._build_bridge(seed, corpus)
    ens, relay = SR.wire_signed_readout(ub, np.arange(int(ub.reservoir_base), int(ub.reservoir_base) + SR.RES_N), seed)
    res, res_idx, _w = SR._make_reservoir(ub, enc, seed)
    SR._wire_relay_ie(ub, relay, ens)
    ub.bridge.set_pathway_weights("res2ens", *SR._wpos_edges(res_idx, ens),
                                  np.zeros(3 * SR.ENS_P * len(res_idx), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2relay", *SR._wpos_edges(res_idx, relay),
                                  np.zeros(3 * SR.RELAY_P * len(res_idx), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    sig = SR.SignedReadout(ub, res_idx, ens, relay, Ws, scale, positive=False)
    ca = s0 = t0 = 0
    for toks, _r in facts:
        hs = SR._host_signed_winners(res0, enc, Ws, toks)
        for k in (0, 1, 2):
            es = SR._run_read(ub, res, ens, relay, enc.encode(toks), sig, k)
            hit = int(int(np.argmax(es)) == hs[k]); ca += hit
            if k == 0:
                s0 += hit; t0 += 1
    return ca / (3 * len(facts)), s0 / max(t0, 1)


best = None                                                   # pick (floor,scale) maximizing CANONICAL (fair, no objrel leak)
for fl in FLOORS:
    C.WS_ENS_FLOOR_C2 = fl; SR.WS_ENS_FLOOR_C2 = fl
    for c in SCALES_C:
        ca, _cs0 = deploy(c / proj, canon)
        if best is None or ca > best[0]:
            best = (ca, fl, c)
ca, fl, c = best
C.WS_ENS_FLOOR_C2 = fl; SR.WS_ENS_FLOOR_C2 = fl
oa, os0 = deploy(c / proj, objr)
print(f"RESULT seed {seed}: BEST-CANONICAL op-point floor={fl:.0f} scale_c={c:.0f} -> CANON {ca:.2f} | "
      f"OBJREL {oa:.2f} | objrel-slot0(THEME) {os0:.2f}  "
      f"(if slot0 recovers on this DEGRADED seed -> op-point is the fix -> homeostatic rung 3)", flush=True)
