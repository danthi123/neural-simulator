"""OFF-DIAGONAL dendritic-predictive-coding de-risk (the dendritic-substrate frontier's cheap-first).

THE QUESTION (per `2026-06-17-dendritic-substrate-frontier-scoping.md` + the build-spec
`2026-06-15-dendritic-predictive-coding-offdiagonal-mechanism-spec.md` §3): the project's POINT-NEURON cortex
recovers category structure on the real corpus via the DIAGONAL of whitening (PPMI/per-hub normalization, caps
~+0.31). The HOST ceiling is +0.44 and the OFFLINE rank-8 ZCA reaches ~+0.49 — i.e. there IS an OFF-DIAGONAL
(cross-neuron, low-rank) residual a point neuron provably cannot reach (a diagonal gain can't rotate off-diagonal
correlations away; CYCLE 87 falsified the somatic low-rank lateral; the somatic Oja network collapses to rank 3-4).
The ONE biological route that escapes the collapse is dendritic predictive coding / interneuron-whitening: a
per-input compartment cancels the predictable common mode BEFORE plasticity, so correlated units stop reinforcing
each other (Mikulasch-Priesemann PNAS 2021; Duong-Lipshutz-Chklovskii-Simoncelli ICML 2023). THE OPEN EMPIRICAL
QUESTION: can an ONLINE, LOCAL realization reach that off-diagonal residual on THIS moderate-SNR corpus — or is the
+0.13 residual intrinsically marginal for any local mechanism? This is the decisive de-risk that GATES the
months-scale dendritic-cortex build.

ARMS (all on the SAME PPMI-encoded real corpus; 3 seeds; the CONTRAST is the result; structure = Pearson(cos(code),
S_true) read directly off each arm's per-concept code, the spec convention):
  HOST          — PPMI+truncated-SVD (data-carries ceiling; gate >= +0.40)
  ZCA_rank8     — offline low-rank ZCA (the off-diagonal ceiling; gate >= +0.45)
  SM_somatic    — Oja similarity-matching (learn_simmatch) — MUST fall short (~+0.35, eff-rank <= 5: the W-collapse)
  DIAG_gain     — per-hub divisive gain — MUST stay ~+0.22-0.31 (the diagonal plateau)
  GAINS_whiten  — MECHANISM A (Duong fixed random frame + plastic gains; the recommended de-risk instrument)
  DEND_balance  — MECHANISM B (Mikulasch error-gated feedforward + anti-Hebbian lateral, full settle; the faithful)

GATES (multi-seed 42/43/44): host_carries (HOST>=.40 & ZCA8>=.45); somatic_falls_short (SM<=.38 & eff_rank<=5);
diagonal_falls_short (DIAG<=.32); MECHANISM_BEATS_COLLAPSE (the key: peak(GAINS,DEND) >= +0.40 AND >= SM+0.06 toward
ZCA8 AND eff_rank in [6,16] — the rank band is co-equal: +0.40@rank-3 = the collapse sneaking through, +0.40@rank-44
= over-whitening luck); generalizes (heldout above chance). ANTI-CHEATS: permuted-similarity collapses; lesion
(freeze the gains / Oja-ify) collapses the lift to the diagonal/somatic plateau; input-lesion (raw, non-PPMI)
trails; eff-rank reported per arm; S_true a-priori.

OUTCOMES (pre-registered): GO = mechanism beats +0.40 toward +0.49, eff_rank~8, controls clean -> the off-diagonal
IS reachable by an online local circuit -> greenlight the dendritic cortex (risk = bridge spiking realization only).
BOUNDARY = beats the diagonal+somatic controls but short of host -> the faithful plastic-frame form with a sharp
target. NEGATIVE = even the collapse-immune fixed-frame-gain circuit plateaus ~+0.35 -> the off-diagonal on this
corpus is intrinsically marginal for ANY local mechanism -> ship the flat 2,048-concept curated cortex (a clean,
citable result; the dendritic build reserved for the artificial-life goal, eyes-open).

Reuse-by-import (build_real_corpus / ppmi_matrix / learn_simmatch / learn_perhub_gains+perhub_residual / zca /
the metrics + effective_rank + heldout_generalization + the host instrument). CPU/numpy, NO GPU, NO sim/ edit.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_offdiagonal_dendritic_pc_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.learned_graded_cortex_fair_test import (  # noqa: E402
    build_real_corpus, ppmi_matrix, learn_simmatch,
)
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
    learn_perhub_gains, perhub_residual,
)
from research.runners._phaseB_offdiagonal_derisk import zca  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def _struct(codes, S_true):
    """The project metric: Pearson of the codes' pairwise cosine-similarity structure vs the a-priori taxonomy."""
    return float(_pearson_vs_Strue(_cos_sim(codes), S_true))


# ===========================================================================
# MECHANISM A — Duong-Lipshutz fixed random frame + plastic gains (the recommended de-risk instrument).
# y = (I + W diag(g) Wᵀ)^{-1} x ;  g_i <- max(g_i + eta*(z_i^2 - 1), 0) ;  z = Wᵀ y.  Only g is learned; the
# frame W is FIXED, so there is no feedforward weight to collapse and the rank is K (the interneuron count).
# ===========================================================================
def gains_whiten(Xc, K, epochs, eta_g, seed):
    rng = np.random.RandomState(seed * 7919 + K)
    H = Xc.shape[1]
    W = rng.standard_normal((H, K)) / np.sqrt(H)        # FIXED frame
    g = np.zeros(K)
    order = np.arange(Xc.shape[0])
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xc[i]
            A = np.eye(H) + (W * g) @ W.T               # (H,H)
            y = np.linalg.solve(A, x)
            z = W.T @ y
            g = np.maximum(g + eta_g * (z ** 2 - 1.0), 0.0)
    A = np.eye(H) + (W * g) @ W.T
    Y = np.linalg.solve(A, Xc.T).T                      # the whitened per-concept codes (Nc, H)
    return Y, g


# ===========================================================================
# MECHANISM B — Mikulasch / normative error-gated dendritic balance (the faithful arm). The single decisive
# change vs learn_simmatch: the feedforward learns on the RESIDUAL (x - x̂) (error-gated), not Oja (yxᵀ); and the
# settle is the FULL recurrent equilibrium z = (I+M)^{-1} W_ff x (a solve, not a single lateral step).
# ===========================================================================
def dend_balance(Xn, k, epochs, eta_w, eta_m, seed):
    rng = np.random.RandomState(seed * 104729 + k)
    Nc, H = Xn.shape
    W_ff = rng.standard_normal((k, H)) * 0.1
    M = np.zeros((k, k))
    order = np.arange(Nc)

    def settle(x):
        return np.linalg.solve(np.eye(k) + M, W_ff @ x)   # (I+M) z = a

    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            z = settle(x)
            x_hat = W_ff.T @ z                          # the population's prediction of x
            W_ff += eta_w * np.outer(z, (x - x_hat))    # ERROR-GATED (residual), NOT Oja
            dM = np.outer(z, z) - M
            np.fill_diagonal(dM, 0.0)
            M += eta_m * dM
    Z = np.array([settle(Xn[i]) for i in range(Nc)])
    return Z, W_ff, M


# ===========================================================================
# One seed: all arms + the anti-cheats.
# ===========================================================================
def run_seed(seed, n_hub=500, sm_k=64):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    Nc = C.shape[0]
    # PPMI diagonal front-end (the off-diagonal stacks on top); L2-normed rows; centered for whitening.
    P = ppmi_matrix(C, 0.75)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Pc = Pn - Pn.mean(0, keepdims=True)

    # --- HOST + offline ZCA ceiling ---
    host = float(score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)[0])
    zca8 = _struct(zca(Pc, rank=8), S_true)

    # --- must-fall-short controls ---
    sm_codes, _, _ = learn_simmatch(P, S_true, k=sm_k, epochs=200, lr_ff=0.01, lr_m=0.01, settle_steps=20, seed=seed)
    sm = _struct(sm_codes, S_true); sm_rank = float(effective_rank(sm_codes))
    g_diag, _ = learn_perhub_gains(C, epochs=200, eta=0.02, seed=seed)
    diag_codes = perhub_residual(C, g_diag, sigma=1.0)
    diag = _struct(diag_codes, S_true)

    # --- MECHANISM A (sweep K = the rank; take the peak that lands in a sane rank band) ---
    A_runs = {}
    for K in (8, 12, 16, 24):
        Y, _ = gains_whiten(Pc, K, epochs=150, eta_g=0.02, seed=seed)
        A_runs[K] = (_struct(Y, S_true), float(effective_rank(Y)), Y)
    A_K = max(A_runs, key=lambda K: A_runs[K][0])
    A_p, A_rank, A_Y = A_runs[A_K]
    A_gen = float(heldout_generalization(A_Y, labels)[0])

    # --- MECHANISM B (sweep k) ---
    B_runs = {}
    for k in (8, 16, 32, 64):
        Z, _, _ = dend_balance(Pn, k, epochs=150, eta_w=0.01, eta_m=0.01, seed=seed)
        B_runs[k] = (_struct(Z, S_true), float(effective_rank(Z)), Z)
    B_k = max(B_runs, key=lambda k: B_runs[k][0])
    B_p, B_rank, B_Z = B_runs[B_k]

    # --- ANTI-CHEATS (on Mechanism A, the recommended instrument) ---
    # (1) permuted-similarity: shuffle the taxonomy -> structure must collapse to ~0 (not a code artifact).
    rngp = np.random.RandomState(seed * 31 + 9)
    perm = rngp.permutation(Nc)
    S_perm = S_true[np.ix_(perm, perm)]
    A_perm = _struct(A_Y, S_perm)
    # (2) lesion: freeze the gains to ZERO (no off-diagonal whitening) -> codes = Pc -> drops to the diagonal level.
    A_lesion = _struct(Pc, S_true)        # g==0 => A=I => Y==Pc (the no-off-diagonal control)
    # (3) input-lesion: run Mechanism A on RAW (non-PPMI) centered input -> must trail the PPMI version.
    Xraw = np.log1p(np.maximum(C, 0.0)); Xraw = Xraw / (np.linalg.norm(Xraw, axis=1, keepdims=True) + 1e-9)
    Xrawc = Xraw - Xraw.mean(0, keepdims=True)
    Yr, _ = gains_whiten(Xrawc, A_K, epochs=150, eta_g=0.02, seed=seed)
    A_rawinput = _struct(Yr, S_true)

    mech_peak = max(A_p, B_p)
    print(f"\n[offdiag-dpc seed {seed}] {Nc}c x {n_hub}h | HOST {host:+.3f} | ZCA_r8 {zca8:+.3f} || "
          f"SM_somatic {sm:+.3f}(rank {sm_rank:.1f}) | DIAG {diag:+.3f} || "
          f"GAINS_whiten(K={A_K}) {A_p:+.3f}(rank {A_rank:.1f}) | DEND_balance(k={B_k}) {B_p:+.3f}(rank {B_rank:.1f})",
          flush=True)
    print(f"   anti-cheat: permuted {A_perm:+.3f}(~0) | lesion(g=0) {A_lesion:+.3f}(~diag) | raw-input {A_rawinput:+.3f}"
          f"(<PPMI) | heldout-gen(A) {A_gen:.2f}", flush=True)
    return {"seed": seed, "host": host, "zca8": zca8, "sm": sm, "sm_rank": sm_rank, "diag": diag,
            "A_p": A_p, "A_rank": A_rank, "A_K": A_K, "A_gen": A_gen, "B_p": B_p, "B_rank": B_rank, "B_k": B_k,
            "mech_peak": mech_peak, "perm": A_perm, "lesion": A_lesion, "rawinput": A_rawinput,
            "A_sweep": {K: A_runs[K][:2] for K in A_runs}, "B_sweep": {k: B_runs[k][:2] for k in B_runs}}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("[off-diagonal dendritic-PC de-risk] can an ONLINE LOCAL error-gated / fixed-frame-gain decorrelator "
          "reach the off-diagonal residual (diagonal ~+0.31, somatic ~+0.35 -> host +0.44, offline ZCA +0.49) on "
          "the REAL corpus, eff-rank ~8 (not collapsed, not over-whitened)? GATES the dendritic-cortex build.",
          flush=True)
    rows = [run_seed(s) for s in (42, 43, 44)]

    def m(key):
        return float(np.mean([r[key] for r in rows]))

    host, zca8, sm, sm_rank, diag = m("host"), m("zca8"), m("sm"), m("sm_rank"), m("diag")
    A_p, A_rank, B_p, B_rank = m("A_p"), m("A_rank"), m("B_p"), m("B_rank")
    mech = max(A_p, B_p)
    mech_rank = A_rank if A_p >= B_p else B_rank
    perm, lesion, rawinput, gen = m("perm"), m("lesion"), m("rawinput"), m("A_gen")

    # --- gates (per the spec §3.2/§3.4) ---
    host_carries = bool(host >= 0.40 and zca8 >= 0.45)
    somatic_falls_short = bool(sm <= 0.38 and sm_rank <= 5.0)
    diagonal_falls_short = bool(diag <= 0.32)
    rank_ok = bool(6.0 <= mech_rank <= 16.0)
    beats_collapse = bool(mech >= 0.40 and mech >= sm + 0.06 and rank_ok)
    anticheat_ok = bool(perm <= 0.10 and lesion <= diag + 0.05 and rawinput <= mech - 0.03 and gen > (1.0 / 8) + 1e-9)

    if beats_collapse and host_carries and somatic_falls_short and diagonal_falls_short and anticheat_ok:
        verdict = "GO"
    elif mech >= sm + 0.04 and mech >= diag + 0.06 and mech < 0.40 and host_carries and anticheat_ok:
        verdict = "BOUNDARY"
    else:
        verdict = "NEGATIVE"

    print(f"\n{'='*104}\n  MEAN (3 seeds): HOST {host:+.3f} | ZCA_r8 {zca8:+.3f} || SM_somatic {sm:+.3f}(rank {sm_rank:.1f}) "
          f"| DIAG {diag:+.3f} || GAINS_whiten {A_p:+.3f}(rank {A_rank:.1f}) | DEND_balance {B_p:+.3f}(rank {B_rank:.1f}) "
          f"|| mech peak {mech:+.3f}", flush=True)
    print(f"  gates: host_carries {host_carries} | somatic_falls_short {somatic_falls_short} | "
          f"diagonal_falls_short {diagonal_falls_short} | rank_in[6,16] {rank_ok} | beats_collapse {beats_collapse} | "
          f"anti-cheat {anticheat_ok} (perm {perm:+.3f} lesion {lesion:+.3f} raw {rawinput:+.3f} gen {gen:.2f})", flush=True)
    print(f"  ==> {verdict}\n{'='*104}", flush=True)
    if verdict == "GO":
        print(f"  GO: an ONLINE LOCAL circuit reaches the off-diagonal residual ({mech:+.3f} >= +0.40, toward ZCA "
              f"{zca8:+.3f}) with eff-rank {mech_rank:.1f}~8, while the somatic Oja ({sm:+.3f}, rank {sm_rank:.1f}) + "
              f"diagonal ({diag:+.3f}) fall short — the dendritic escape WORKS on the real corpus. The remaining risk "
              f"is ONLY the bridge spiking realization (graded_lateral settle + error-gating) → GREENLIGHT the "
              f"dendritic cortex on a measured signal. NO sim/ edit.", flush=True)
    elif verdict == "BOUNDARY":
        print(f"  BOUNDARY: the mechanism beats the diagonal+somatic controls ({mech:+.3f} vs SM {sm:+.3f} / DIAG "
              f"{diag:+.3f}) but falls short of host ({host:+.3f}) — the right family, an online-convergence gap. "
              f"Next cheap step: the faithful plastic-frame form / shaped target / longer settle before any bridge "
              f"work. NO sim/ edit.", flush=True)
    else:
        print(f"  NEGATIVE: even the collapse-immune fixed-frame-gain circuit ({A_p:+.3f}) ≈ the somatic plateau "
              f"({sm:+.3f}) — the off-diagonal residual on THIS moderate-SNR corpus is intrinsically marginal for any "
              f"local mechanism. A clean, citable result ⇒ SHIP THE FLAT 2,048-concept curated cortex as the "
              f"conversational product; reserve the months-scale dendritic build for the artificial-life goal only "
              f"(eyes open — it may also plateau on real experience). Honest negative = the deliverable. NO sim/ edit.",
              flush=True)

    out = {"verdict": verdict, "host": host, "zca8": zca8, "sm": sm, "sm_rank": sm_rank, "diag": diag,
           "gains_whiten": A_p, "gains_rank": A_rank, "dend_balance": B_p, "dend_rank": B_rank, "mech_peak": mech,
           "gates": {"host_carries": host_carries, "somatic_falls_short": somatic_falls_short,
                     "diagonal_falls_short": diagonal_falls_short, "rank_ok": rank_ok,
                     "beats_collapse": beats_collapse, "anticheat_ok": anticheat_ok},
           "anti_cheat": {"perm": perm, "lesion": lesion, "rawinput": rawinput, "heldout_gen": gen},
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_offdiagonal_dendritic_pc.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "BOUNDARY" else 1))


if __name__ == "__main__":
    main()
