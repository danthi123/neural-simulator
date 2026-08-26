"""RUNG 4 de-risk — the PHASE-DOMAIN structural read (numpy concept test, gate the RF-spiking build). The reservoir
feature f encodes objrel (linear argmax 100%), but rate/count/signed spiking reads lose it (common-mode pedestal) or are
seed-fragile. The composer escaped this exact family via FHRR PHASE coding. Test the phase-coherence classifier here in
numpy, MULTI-SEED, before any spiking build:
  P_f = exp(i·angle(Phi @ f))         (rate -> unit phasor via a fixed seeded complex projection; magnitude discarded)
  P_r = exp(i·angle(Phi @ W[:,r]))    (the 3 role read-out vectors as a 3-phasor codebook)
  role = argmax_r | sum_k P_f[k] * conj(P_r[k]) |    (phase coherence; a sum of phase DIFFERENCES, no additive pedestal)
Scored per-slot vs TRUE roles on held-out canonical + objrel, seeds 42/44/100 (the draws that broke the rate/signed reads).
Also: the COMMON-MODE control (f -> f + c) and the RATE-linear baseline (argmax(f·W), the ceiling). PASS = objrel slot0 high
+ canonical high on ALL seeds + pedestal-invariant. Cheap: numpy reservoir, no bridge."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX, Reservoir

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "44", "100"])]
N_PHASE = int(sys.argv[2]) if len(sys.argv) > 2 else 512
N_TRAIN, N_TEST = 60, 16
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])


def fit_W(res, sentences):
    """Ridge per-slot read-out on the reservoir feature (reservoir rows only). Returns W[k] : (n_res, 3)."""
    F, Yk = [], {0: [], 1: [], 2: []}
    for toks, roles in sentences:
        f = np.asarray(res.final_state(enc.encode(toks)), float)
        content = sorted(roles)
        row = None
        for k, pos in enumerate(content):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            if row is None:
                row = f; F.append(f)
            y = np.zeros(3); y[tgt] = 1.0; Yk[k].append((len(F) - 1, y))
    F = np.array(F); n = F.shape[1]
    W = {}
    A = F.T @ F + 1e-2 * np.eye(n)
    for k in (0, 1, 2):
        Y = np.zeros((len(F), 3))
        for idx, y in Yk[k]:
            Y[idx] = y
        W[k] = np.linalg.solve(A, F.T @ Y)                    # (n_res, 3)
    return W


def phases(Phi, v):
    return np.angle(Phi @ v)                                  # (N_PHASE,) real phases


for seed in seeds:
    rng = np.random.default_rng(seed * 101 + 5)
    res = Reservoir(enc.dim, seed)
    n_res = res.n
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    W = fit_W(res, train)
    prng = np.random.default_rng(seed * 31 + 7)
    Phi = (prng.standard_normal((N_PHASE, n_res)) + 1j * prng.standard_normal((N_PHASE, n_res))) / np.sqrt(n_res)
    if os.environ.get("MEANREMOVE", "0") == "1":
        Phi = Phi - Phi.mean(axis=1, keepdims=True)           # zero row-sums -> Phi @ (f + c*1) == Phi @ f (pedestal-immune)
    Pr = {k: np.stack([np.exp(1j * phases(Phi, W[k][:, r])) for r in range(3)], 1) for k in (0, 1, 2)}  # (N_PHASE,3)

    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    def score(sentences, mode, pedestal=0.0):
        ok = tot = s0ok = s0t = 0
        for toks, roles in sentences:
            f = np.asarray(res.final_state(enc.encode(toks)), float) + pedestal
            Pf = np.exp(1j * phases(Phi, f))
            for k, pos in enumerate(sorted(roles)):
                if k >= 3:
                    break
                tgt = _ROLE_IDX[roles[pos]]
                if tgt >= 3:
                    continue
                if mode == "linear":
                    pred = int(np.argmax(f @ W[k]))
                else:                                         # phase-coherence
                    pred = int(np.argmax([np.abs(np.sum(Pf * np.conj(Pr[k][:, r]))) for r in range(3)]))
                hit = int(pred == tgt); ok += hit; tot += 1
                if k == 0:
                    s0ok += hit; s0t += 1
        return ok / max(tot, 1), s0ok / max(s0t, 1)

    lc, _ = score(canon, "linear"); lo, ls0 = score(objr, "linear")
    pc, _ = score(canon, "phase"); po, ps0 = score(objr, "phase")
    # common-mode control: add a big uniform pedestal to f, re-score objrel under phase
    po_ped, ps0_ped = score(objr, "phase", pedestal=5.0)
    print(f"seed {seed} [Nphase {N_PHASE}]: LINEAR canon {lc:.2f}/objrel {lo:.2f}(slot0 {ls0:.2f}) | "
          f"PHASE canon {pc:.2f}/objrel {po:.2f}(slot0 {ps0:.2f}) | PHASE+pedestal objrel {po_ped:.2f}(slot0 {ps0_ped:.2f}) "
          f"[pedestal-invariant={abs(ps0-ps0_ped)<0.13}]", flush=True)
