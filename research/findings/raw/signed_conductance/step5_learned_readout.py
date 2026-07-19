"""BIOLOGIZE the read-out (research-gate verdict #1): a per-role DELTA RULE learns the res->ens synapses ON the frozen
spiking reservoir, replacing the host np.linalg.solve ridge fit. Per training sentence + slot k:
  drive reservoir -> (rho = reservoir firing, a = ACTUAL ensemble firing via run_with_ens);  error_r = T_r - a_norm_r;
  W_k[r,:] += eta * error_r * rho  (clip >=0, Dale-legal).  T = one-hot on the KNOWN slot-k role label (environmental
supervision, per-role-LOCAL -- NOT a global scalar). The learned W_k ARE the read-out (no host f@Ws, no ridge). Because a
is the REAL spiking ensemble firing (f-I nonlinearity + WTA ignition-order INSIDE the error), it learns to make the correct
ensemble WIN THE SPIKING COMPETITION on THIS draw -> generalizes across draws by construction. De-risk on seed 44 (host-fit
11/18) first, then the 6-seed blind protocol."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["44"])]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ETA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
N_TRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 40      # sentences/construction for learning (reduced for CPU speed)
FLOOR = 150.0
corpus = S.setup_corpus(seed=42); test = corpus["test"]
C.WS_BIAS_SCALE_C2 = 0.0
C.WS_ENS_FLOOR_C2 = FLOOR
C.WS_REPLAY = 1                                              # 1 replay during learning (delta rule averages over epochs)
C.READ_T_STEP_C2 = 18                                        # shorter read window during learning (faster; noisier a is OK)


def write_W(ub, pre, post, ens, res_idx, Wk):
    """Wk is (3 x n_res); every ens[r] neuron <- res_idx[i] with weight Wk[r,i] (role-major, matches _ws_edges order)."""
    n = len(res_idx); w = np.empty(len(pre), np.float32); p = 0
    for r in range(3):
        col = Wk[r]
        for _e in ens[r]:
            w[p:p + n] = col; p += n
    ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=False)


def run_seed(seed, scramble=False):
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
    # host target (for the AGREE metric only -- NOT used in learning; learning uses the environmental role LABEL)
    hostWs = C._fit_Ws_spiking(res, enc, train)                # ridge, only to define the host argmax the read must match
    host = [[int(np.argmax((np.concatenate([res.final_state(enc.encode(t)), [1.0]]) @ hostWs[k])[[0, 1, 2]]))
             for k in (0, 1, 2)] for t, *_ in test]
    W = [np.zeros((3, n_res)) for _ in range(3)]               # 3 slot read-outs, learned from zero
    lrng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        order = list(range(len(train))); lrng.shuffle(order)
        for si in order:
            toks, roles = train[si]
            content = sorted(roles)
            for k, t in enumerate(content):
                tgt = _ROLE_IDX[roles[content[(k + 1) % len(content)] if scramble else t]]  # scramble = deranged label
                if tgt >= 3:
                    continue                                  # GOAL/LOCATION not in the 3-way canonical read (ens 0/1/2)
                write_W(ub, pre, post, ens, res_idx, W[k])
                rho, a = res.run_with_ens(enc.encode(toks), ens)      # rho=reservoir firing, a=ACTUAL ens firing
                a = np.asarray(a, float); an = a / (a.sum() + 1e-9)
                T = np.zeros(3); T[tgt] = 1.0
                err = T - an
                W[k] += ETA * np.outer(err, rho[:n_res])              # delta rule (per-role error x presyn firing)
                np.clip(W[k], 0.0, None, out=W[k])
    # deploy: argmax over ACTUAL ens firing, no host f@Ws
    ok = 0
    for (toks, *_), hs in zip(test, host):
        for k in (0, 1, 2):
            write_W(ub, pre, post, ens, res_idx, W[k])
            _rho, a = res.run_with_ens(enc.encode(toks), ens)
            ok += int(int(np.argmax(a)) == hs[k])
    return ok, len(test) * 3


DO_SCRAMBLE = int(sys.argv[5]) if len(sys.argv) > 5 else (1 if len(seeds) == 1 else 0)
for seed in seeds:
    t0 = time.time()
    ok, tot = run_seed(seed, scramble=False)
    if DO_SCRAMBLE:
        ok_s, _ = run_seed(seed, scramble=True)               # scrambled-label anti-cheat (must fail to learn)
        print(f"seed {seed} LEARNED-DELTA: {ok}/{tot}  | scrambled-label {ok_s}/{tot} (must be ~chance)  "
              f"[E{EPOCHS} eta{ETA} N{N_TRAIN}] [{time.time()-t0:.0f}s]", flush=True)
    else:
        print(f"seed {seed} LEARNED-DELTA: {ok}/{tot}  [E{EPOCHS} eta{ETA} N{N_TRAIN}] [{time.time()-t0:.0f}s]", flush=True)
