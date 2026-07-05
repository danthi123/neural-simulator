"""DECISIVE final candidate: does the CONDUCTANCE-DOMAIN SIGNED read-out (Wp exc + Wn inh relay, net drive = Ws@f signed,
NO positive-shift pedestal) recover OBJREL where the positive-shifted read fails (0.00)? The earlier signed arc found this
machinery DECORATIVE on CANONICAL (positive Wp alone carries canonical) -- but OBJREL is exactly where the SIGNED info
should be LOAD-BEARING (positive-shift objrel = 0.00). Reuses _rungB1c_signed_readout_derisk's SignedReadout + _run_read
(positive=True is the c2 baseline; positive=False is the signed opponent), at LOW floor (subtractive g_i regime). Scored
per-slot vs the host SIGNED winner (== TRUE role on objrel, since signed-linear solves objrel 100%). For a fair operating
point, sweep scale + pick the CANONICAL-best scale, then report objrel at it."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_signed_readout_derisk as SR
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
N_TEST = 12
C.WS_ENS_FLOOR_C2 = FLOOR
SR.WS_ENS_FLOOR_C2 = FLOOR
corpus = SR.setup_corpus(seed=seed)
disc, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(disc)
rng = np.random.default_rng(seed * 101 + 5)
from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import N_TRAIN_PER
train = _gen(_TRAIN_KINDS, N_TRAIN_PER, rng, subj, verb, obj)

# FIT the signed ridge Ws on the spiking reservoir feature
ub0 = SR._build_bridge(seed, corpus)
ens0, relay0 = SR.wire_signed_readout(ub0, np.arange(int(ub0.reservoir_base), int(ub0.reservoir_base) + SR.RES_N), seed)
res0, res_idx0, _w0 = SR._make_reservoir(ub0, enc, seed)
SR._wire_relay_ie(ub0, relay0, ens0)
from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import _fit_Ws_spiking
res0.snapshot_after_wiring()
Ws = _fit_Ws_spiking(res0, enc, train)
n_res = len(res_idx0)

trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)


def deploy(positive, scale, facts):
    ub = SR._build_bridge(seed, corpus)
    ens, relay = SR.wire_signed_readout(ub, np.arange(int(ub.reservoir_base), int(ub.reservoir_base) + SR.RES_N), seed)
    res, res_idx, _w = SR._make_reservoir(ub, enc, seed)
    SR._wire_relay_ie(ub, relay, ens)
    ub.bridge.set_pathway_weights("res2ens", *SR._wpos_edges(res_idx, ens),
                                  np.zeros(3 * SR.ENS_P * len(res_idx), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2relay", *SR._wpos_edges(res_idx, relay),
                                  np.zeros(3 * SR.RELAY_P * len(res_idx), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    sig = SR.SignedReadout(ub, res_idx, ens, relay, Ws, scale, positive=positive)
    ok = tot = s0ok = s0t = 0
    for toks, _roles in facts:
        hs = SR._host_signed_winners(res0, enc, Ws, toks)     # signed host winner == TRUE role (signed solves objrel)
        for k in (0, 1, 2):
            es = SR._run_read(ub, res, ens, relay, enc.encode(toks), sig, k)
            hit = int(int(np.argmax(es)) == hs[k]); ok += hit; tot += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return ok / max(tot, 1), s0ok / max(s0t, 1)


proj = max(1e-9, float((res0.final_state(enc.encode(canon[0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max()))
for positive in (True, False):
    label = "POSITIVE (shift)" if positive else "SIGNED (conductance)"
    best = None
    for c in (60.0, 90.0, 130.0, 180.0, 240.0):
        sc = c / proj
        ca, cs0 = deploy(positive, sc, canon)
        if best is None or ca > best[0]:
            best = (ca, cs0, sc)
    ca, cs0, sc = best
    oa, os0 = deploy(positive, sc, objr)                       # objrel at the CANONICAL-best scale (fair)
    print(f"seed {seed} FLOOR={FLOOR:.0f} {label:>22}: CANON {ca:.2f} | OBJREL {oa:.2f} | "
          f"objrel-slot0(THEME) {os0:.2f}  (scale {sc:.3g})", flush=True)
