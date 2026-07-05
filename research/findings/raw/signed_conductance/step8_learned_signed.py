"""LEARNED SIGNED read (the objrel surpass's specified next mechanism). The FIXED ridge signed conductance overfit
(seed 42 objrel 0.92 but 44/100 fail; and it TRADES OFF canonical 0.31). The delta rule ADAPTS per-draw (it fits THROUGH
the spiking deploy) — exactly what generalized the CANONICAL positive read 6/6 where the ridge was seed-fragile. Here the
delta rule learns a SIGNED W (unclipped) deployed via SR's signed CONDUCTANCE (Wp exc + Wn inh relay), so it can learn a
read that does BOTH canonical AND objrel through the actual nonlinear deploy. Cheap-first de-risk: seed 42, does learned-
signed do BOTH (unlike fixed-signed's tradeoff)? Scored per-slot vs TRUE roles (held-out canonical + objrel).

Mechanism per (sentence, slot k): set sig.Wpos[k]=max(W[k].T,0), Wneg[k]=max(-W[k].T,0); deploy via _run_read -> a =
ens firing; W[k] += eta*(T - a_norm)*rho (rho = cached reservoir feature; NO clip -> signed). Deploy: signed conductance."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_signed_readout_derisk as SR
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
SCALE_C = float(sys.argv[5]) if len(sys.argv) > 5 else 130.0
BALANCE = int(sys.argv[6]) if len(sys.argv) > 6 else 0     # extra objrel/subjrel examples to fix the 7:1 slot0-AGENT imbalance
N_TRAIN, N_TEST = 35, 12
C.WS_ENS_FLOOR_C2 = FLOOR; SR.WS_ENS_FLOOR_C2 = FLOOR
corpus = SR.setup_corpus(seed=seed)
disc, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(disc)
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
if BALANCE:                                               # oversample OBJREL (the only slot0=THEME construction) to fix the
    train = train + _gen(["objrel"], BALANCE, rng, subj, verb, obj)   # 7:1 slot0-AGENT:THEME imbalance that drives position


def build():
    ub = SR._build_bridge(seed, corpus)
    ens, relay = SR.wire_signed_readout(ub, np.arange(int(ub.reservoir_base), int(ub.reservoir_base) + SR.RES_N), seed)
    res, res_idx, _w = SR._make_reservoir(ub, enc, seed)
    SR._wire_relay_ie(ub, relay, ens)
    ub.bridge.set_pathway_weights("res2ens", *SR._wpos_edges(res_idx, ens),
                                  np.zeros(3 * SR.ENS_P * len(res_idx), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2relay", *SR._wpos_edges(res_idx, relay),
                                  np.zeros(3 * SR.RELAY_P * len(res_idx), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    return ub, ens, relay, res, res_idx


ub, ens, relay, res, res_idx = build()
n_res = len(res_idx)
# cache reservoir features (deterministic per sentence)
cache = {}
def rho_of(toks):
    key = tuple(toks)
    if key not in cache:
        cache[key] = np.asarray(res.final_state(enc.encode(toks)), float)[:n_res]
    return cache[key]

proj = max(1e-9, float((rho_of(train[0][0]) @ np.maximum(np.random.default_rng(0).standard_normal((n_res, 3)), 0)).max()))
scale = SCALE_C / proj
# a SignedReadout with a mutable Ws (we overwrite Wpos/Wneg per slot before each read)
sig = SR.SignedReadout(ub, res_idx, ens, relay, {k: np.zeros((n_res + 1, 5)) for k in (0, 1, 2)}, scale, positive=False)
INIT_RIDGE = int(os.environ.get("INIT_RIDGE", "0"))            # 1 = init W from the ridge SIGNED Ws (has the objrel structure)
W = [np.zeros((3, n_res)) for _ in range(3)]                   # SIGNED (unclipped) learned read-out
if INIT_RIDGE:
    from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import _fit_Ws_spiking
    Wsr = _fit_Ws_spiking(res, enc, train)                    # ridge fit on the reservoir feature -> has objrel structure
    _rw = float(os.environ.get("RIDGE_W", "1.0"))             # magnitude to bring the ridge init into the delta op range
    for k in (0, 1, 2):
        col = Wsr[k][:n_res, :3]                              # (n_res, 3), SIGNED
        col = col / (np.abs(col).mean() + 1e-9) * _rw         # normalize to a unit-ish scale, times RIDGE_W
        W[k] = col.T                                          # (3, n_res)
lrng = np.random.default_rng(seed)
t0 = time.time()
for ep in range(EPOCHS):
    order = list(range(len(train))); lrng.shuffle(order)
    for si in order:
        toks, roles = train[si]
        for k, pos in enumerate(sorted(roles)):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            sig.Wpos[k] = np.maximum(W[k].T, 0.0)              # (n_res, 3)
            sig.Wneg[k] = np.maximum(-W[k].T, 0.0)
            a = np.asarray(SR._run_read(ub, res, ens, relay, enc.encode(toks), sig, k), float)
            an = a / (a.sum() + 1e-9)
            T = np.zeros(3); T[tgt] = 1.0
            W[k] += ETA * np.outer(T - an, rho_of(toks))       # signed delta, NO clip
    print(f"[seed {seed}] epoch {ep+1}/{EPOCHS} done [{time.time()-t0:.0f}s]", flush=True)


def score(facts):
    ok = tot = s0ok = s0t = 0
    for toks, roles in facts:
        for k, pos in enumerate(sorted(roles)):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            sig.Wpos[k] = np.maximum(W[k].T, 0.0); sig.Wneg[k] = np.maximum(-W[k].T, 0.0)
            a = np.asarray(SR._run_read(ub, res, ens, relay, enc.encode(toks), sig, k), float)
            hit = int(int(np.argmax(a)) == tgt); ok += hit; tot += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return ok / max(tot, 1), s0ok / max(s0t, 1)


trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
ca, cs0 = score(canon)
oa, os0 = score(objr)
print(f"RESULT seed {seed} FLOOR={FLOOR:.0f} scale_c={SCALE_C:.0f} [E{EPOCHS} eta{ETA}]: "
      f"LEARNED-SIGNED CANON {ca:.2f} | OBJREL {oa:.2f} | objrel-slot0(THEME) {os0:.2f}  "
      f"(fixed-signed was CANON 0.31/OBJREL 0.92; positive c3 was CANON 1.00/OBJREL 0.00)", flush=True)
