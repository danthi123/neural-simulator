"""FULL validation of the biological learned (delta-rule) read-out: per seed, the LEARNED read + all anti-cheats, at a
FIXED protocol (no per-seed tune). Modes:
  perrole  = the per-role delta rule (the claim).
  scramble = deranged role labels -> must FAIL to learn (learning is role-specific, not a position artifact).
  global   = a GLOBAL SCALAR error (mean of the per-role errors, same to all 3 ensembles) -> must FAIL near chance
             (proves the PER-ROLE-local credit is load-bearing; reproduces the project's global-scalar failure).
  +lesion  = after learning, zero the res2ens synapses -> deploy -> must COLLAPSE (the LEARNED synapses ARE the read-out).
Optional population lever: N_ENS_P (ens neurons/role) via env for the seed-101 tail. Source-clean is a separate code check."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44", "100", "101", "102"])]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ETA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
N_TRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 35
CONTROLS = int(sys.argv[5]) if len(sys.argv) > 5 else 1        # 1 = run scramble+global+lesion anti-cheats; 0 = learn only
FLOOR = 150.0
ENS_P = int(os.environ.get("N_ENS_P", str(C.WTA_P_C2)))         # population lever (default = c2 P=80)
C.WTA_P_C2 = ENS_P
C.ROLE_WTA_N_C2 = 3 * ENS_P + C.WTA_INH_C2
C.WS_BIAS_SCALE_C2 = 0.0
C.WS_ENS_FLOOR_C2 = FLOOR
C.WS_REPLAY = 1
C.READ_T_STEP_C2 = int(os.environ.get("READ_T", "18"))    # temporal read-out resolution lever (c2 CRUX uses 30)
corpus = S.setup_corpus(seed=42); test = corpus["test"]


def write_W(ub, pre, post, ens, res_idx, Wk):
    n = len(res_idx); w = np.empty(len(pre), np.float32); p = 0
    for r in range(3):
        col = Wk[r]
        for _e in ens[r]:
            w[p:p + n] = col; p += n
    ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=False)


def run_seed(seed, mode="perrole", lesion=False):
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in)
    n_res = len(res_idx)
    pre, post = C._ws_edges(res_idx, ens)
    ub.bridge.set_pathway_weights("res2ens", pre, post, np.zeros(len(pre), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    hostWs = C._fit_Ws_spiking(res, enc, train)                # ridge, ONLY to define the host argmax the read must match
    host = [[int(np.argmax((np.concatenate([res.final_state(enc.encode(t)), [1.0]]) @ hostWs[k])[[0, 1, 2]]))
             for k in (0, 1, 2)] for t, *_ in test]
    W = [np.zeros((3, n_res)) for _ in range(3)]
    lrng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        order = list(range(len(train))); lrng.shuffle(order)
        for si in order:
            toks, roles = train[si]
            content = sorted(roles)
            for k, t in enumerate(content):
                tgt = _ROLE_IDX[roles[content[(k + 1) % len(content)] if mode == "scramble" else t]]
                if tgt >= 3:
                    continue
                write_W(ub, pre, post, ens, res_idx, W[k])
                rho, a = res.run_with_ens(enc.encode(toks), ens)
                a = np.asarray(a, float); an = a / (a.sum() + 1e-9)
                T = np.zeros(3); T[tgt] = 1.0
                if mode == "global":
                    # GLOBAL scalar credit (the project's documented FAILED rule): ONE scalar reward for ALL roles'
                    # synapses (no per-role target) -> every role-row grows identically -> no discrimination -> chance.
                    R = 1.0 if int(np.argmax(an)) == tgt else -1.0
                    err = np.full(3, R)
                else:
                    err = T - an                               # per-role LOCAL error (the load-bearing credit rule)
                W[k] += ETA * np.outer(err, rho[:n_res])
                np.clip(W[k], 0.0, None, out=W[k])
    if lesion:
        for k in range(3):
            W[k][:] = 0.0                                       # syn-readout lesion: no reservoir signal to the ens
    ok = 0
    for (toks, *_), hs in zip(test, host):
        for k in (0, 1, 2):
            write_W(ub, pre, post, ens, res_idx, W[k])
            _rho, a = res.run_with_ens(enc.encode(toks), ens)
            ok += int(int(np.argmax(a)) == hs[k])
    return ok, len(test) * 3


for seed in seeds:
    t0 = time.time()
    ok, tot = run_seed(seed, "perrole")
    if CONTROLS:
        sc, _ = run_seed(seed, "scramble")
        gl, _ = run_seed(seed, "global")
        le, _ = run_seed(seed, "perrole", lesion=True)
        print(f"seed {seed} P{ENS_P}: LEARNED {ok}/{tot} | scramble {sc} | global {gl} | syn-lesion {le} "
              f"(controls must be ~chance {tot//3}) [E{EPOCHS} eta{ETA} N{N_TRAIN}] [{time.time()-t0:.0f}s]", flush=True)
    else:
        print(f"seed {seed} P{ENS_P}: LEARNED {ok}/{tot} [E{EPOCHS} eta{ETA} N{N_TRAIN}] [{time.time()-t0:.0f}s]", flush=True)
