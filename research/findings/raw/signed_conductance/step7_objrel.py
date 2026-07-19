"""OBJREL structural-role test for the c3 learned (delta-rule) read-out (audit defect D1).

The canonical SVO test is POSITION-trivial (slot0=AGENT always). The DECISIVE test of genuine ROLE-reading is the
OBJECT-RELATIVE construction, where slot0 (left-to-right) = THEME, not AGENT (role != canonical position). The delta rule
TRAINS on _TRAIN_KINDS (which includes objrel), so it CAN learn to discriminate canonical-slot0->AGENT from
objrel-slot0->THEME via the reservoir feature (the "that" marker). EMERGE-78 showed a SIGNED host read-out does this; the
open question is whether the POSITIVE (Dale-legal) spiking c3 read-out preserves it (the runner's c2 probe found the c2
POSITIVE host-ridge MISROUTES objrel -> position). We score PER-SLOT against the TRUE roles (from _gen's roles dict), NOT
against the host ridge argmax (which the audit flagged as circular).

Report: canonical per-slot acc, objrel per-slot acc, and objrel SLOT-0 (THEME) acc specifically (the discriminating slot).
A POSITION reader: canonical ~1.0, objrel-slot0 ~0 (says AGENT). A STRUCTURAL reader: both high."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42"])]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 35
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 12
# c3 validated read config (consistent learn+deploy)
C.WS_BIAS_SCALE_C2 = 0.0
C.WS_ENS_FLOOR_C2 = 150.0
C.WS_REPLAY = 1
C.READ_T_STEP_C2 = 18
corpus0 = C.setup_corpus(seed=42)
subj, verb, obj = corpus0["subj"], corpus0["verb"], corpus0["obj"]


def per_slot_acc(ub, res, ens, enc, Ws, sentences):
    """Deploy the LEARNED read-out per content slot; score argmax(ens firing) vs the TRUE role. Returns
    (overall_ok, overall_tot, slot0_ok, slot0_tot, per_slot_hits[3], per_slot_tot[3])."""
    n_res = len(res.res_idx)
    pre, post = C._ws_edges(res.res_idx, ens)
    Wdep = {k: np.array([Ws[k][:n_res, r] for r in range(3)], np.float64) for k in (0, 1, 2)}  # (3,n_res) per slot

    def write(Wk):
        w = np.empty(len(pre), np.float32); p = 0
        for r in range(3):
            for _e in ens[r]:
                w[p:p + n_res] = Wk[r]; p += n_res
        ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=False)

    ok = tot = s0_ok = s0_tot = 0
    ps_hit = [0, 0, 0]; ps_tot = [0, 0, 0]
    for toks, roles in sentences:
        content = sorted(roles)                                   # left-to-right content positions
        for k, pos in enumerate(content):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:                                          # GOAL/LOCATION not in the 3-way read
                continue
            write(Wdep[k])
            _rho, a = res.run_with_ens(enc.encode(toks), ens)
            pred = int(np.argmax(np.asarray(a, float)))
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0_ok += hit; s0_tot += 1
    return ok, tot, s0_ok, s0_tot, ps_hit, ps_tot


for seed in seeds:
    t0 = time.time()
    enc = Encoder(corpus0["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    ub, ens, inh = C._build_wired_bridge(seed, corpus0, mode="c2")
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
    Ws = C._learn_Ws_spiking(ub, res, ens, enc, train, seed, epochs=EPOCHS)
    # held-out test sets (distinct rng from train), scored vs TRUE roles
    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
    co, ct, cs0o, cs0t, cps, cpt = per_slot_acc(ub, res, ens, enc, Ws, canon)
    oo, ot, os0o, os0t, ops, opt = per_slot_acc(ub, res, ens, enc, Ws, objr)
    print(f"seed {seed} [E{EPOCHS} N{N_TRAIN}]: "
          f"CANONICAL per-slot {co}/{ct}={co/max(ct,1):.2f} (slots {cps}/{cpt}) | "
          f"OBJREL per-slot {oo}/{ot}={oo/max(ot,1):.2f} (slots {ops}/{opt}) | "
          f"OBJREL slot0(THEME) {os0o}/{os0t}={os0o/max(os0t,1):.2f}  <- the discriminating slot "
          f"[{time.time()-t0:.0f}s]", flush=True)
