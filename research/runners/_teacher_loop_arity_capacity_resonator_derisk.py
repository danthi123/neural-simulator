"""RESONATOR / EXPLAIN-AWAY CLEANUP READOUT for the naturalistic shared-channel capacity question (2026-08-10).

THE QUESTION IT SETTLES. The corrected finding (2026-08-10-shared-channel-arity-capacity-CORRECTED-DC-offset-artifact)
plus the correlated de-risk established: under CORRELATED codes (rho->1, shared-subspace rank r_shared small) the
composer's neural superposition is REPRESENTATION-robust (cos-to-true stays ~0.99, the reconstruction points at the
true prototype) yet the Euclidean NEAREST-PROTOTYPE readout CRATERS (recall ~0.44 at rho=0.99/r=1). The diagnosis: the
naturalistic "capacity break" is a READOUT-RULER artifact -- under correlation the bundle offset becomes FACT-DEPENDENT
(a single constant mean-centering under-removes it) and cosine saturates from collinearity. So the crater is the RULER,
not representational crosstalk. This runner builds the ruler the diagnosis implies and TESTS it.

THE FIX (offset-invariant, collinearity-robust readout). A RESONATOR / explain-away cleanup (Frady-Kanerva resonator
networks 2020; Plate 1995 HRR cleanup). The generator already computes per-family readout CODEBOOKS: the running-mean
Hebbian cleanups readout_m(v) = elig(off_m+v) @ W[m][v] + anchor (one d-vector per seen primitive). The shared-channel
regeneration is EXACTLY the sum of these atoms: regen[j][:d] = sum_m codebook_m(v_j[m]). A resonator FACTORIZES that
bundle back into per-family values by iterated per-family codebook projection + winner-take-all cleanup + explain-away
subtraction of the other families' current estimates, to a fixed point. It is offset-invariant BY CONSTRUCTION: the
codebooks are centered per family (mean over v removed), the exact fact-INDEPENDENT bundle offset M_off = sum_m mean_v
codebook_m(v) is subtracted (label-free, codebook-only), and the factorization runs on the pure centered-atom sum
s_c = sum_m Cb_m(v_j[m]). It matches per-family CODE STRUCTURE, never absolute magnitude/offset, so the DC that sinks
Euclidean cannot touch it. It reads ONLY the per-family codebooks (seen primitives) + the composer's OWN regeneration
-- NEVER the held-out fact's prototype (which is the test-time ruler only). It identifies the fact by the recovered
value-tuple (v_hat_0..v_hat_{M-1}), a bijection with the prototype index.

WHAT A RESONATOR CAN DO THAT NEAREST-PROTO CANNOT, AND ITS OWN FAILURE MODE. If the centered atoms are linearly
independent enough that the sum s_c is UNIQUELY decomposable, the resonator recovers v_j exactly regardless of any DC
offset -> recall 1.00 where Euclidean craters (CONFIRMS: composer robust, break was the ruler). If instead the codes
collapse to a low-rank subspace (rho->1, r_shared small), DISTINCT value-combos produce nearly the same centered sum ->
the factorization is genuinely AMBIGUOUS -> a REAL representational capacity limit. BUT (Kent 2020) a hard-WTA
resonator near capacity can also fall into a LIMIT CYCLE that never settles -- a NON-convergence that masquerades as a
"genuine limit". So a recall drop is only a real capacity limit if it comes WITH high convergence (a settled fixed
point) and cos-to-true dropping; a recall drop with LOW convergence is the RESONATOR failing, not the composer. This
runner reports the CONVERGENCE RATE explicitly to keep those two apart -- that distinction is the whole point.

SKEPTICAL CONTROLS (mandatory):
  (1) rho=0 must reproduce resonator recall ~1.00 (harness sound: independent codes factorize cleanly).
  (2) CONVERGENCE RATE reported for every (rho, r): a recall drop with low convergence is a resonator limit-cycle
      failure, NOT a composer capacity limit. Distinguish them.
  (3) The resonator reads only the per-family codebooks (seen primitives) + the composer's own regeneration; it NEVER
      reads the held-out fact's prototype. The value-tuple it outputs is scored against the true tuple (grading key
      only). No prototype leakage into the readout.
  (4) Like-for-like: the SAME regen[j], SAME held-out split, SAME codes feed the Euclidean-corrected readout (the
      established headline) and a disjoint (concatenation, no-crosstalk) control -- so the resonator-vs-Euclidean gap
      isolates the readout, not the setup. cos-to-true (a property of the representation, not the readout) is shared.

RUN (numpy CPU; tiny, fast):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_arity_capacity_resonator_derisk \
      --seeds 42 43 44 --M 4 --K 3 --d 8 --r-shared 2 1 --rho 0.0 0.5 0.8 0.95 0.99 \
      --out research/findings/raw/teacher_loop_arity_capacity_resonator.json

NO sim/ edit. Reuse-by-import: the correlated world (_make_world_corr, _mean_abs_offdiag_cos), the frozen spiking
generator CompositionalGeneratorM, the coverage-preserving split, and the recall/cos helpers.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402

from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import N_ACT  # noqa: E402
from research.runners._teacher_loop_compositional_generator_derisk import _action_ctx_const  # noqa: E402
from research.runners._teacher_loop_zeroshot_composition_derisk import _nearest_proto, _recall_fraction  # noqa: E402
from research.runners._teacher_loop_arity_capacity_derisk import (  # noqa: E402
    CompositionalGeneratorM, _FlatStoreM, _heldout_split_M,
)
from research.runners._teacher_loop_arity_capacity_correlated_derisk import (  # noqa: E402
    _make_world_corr, _mean_abs_offdiag_cos,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_arity_capacity_resonator.json"


# ============================ the codebook + the resonator (the new readout) ============================
def _build_codebook(gen):
    """Extract the composer's per-family readout CODEBOOKS (seen primitives only). codebook_m[v] = the running-mean
    Hebbian cleanup = elig(off_m+v) @ W[m][v] + anchor (the SAME atom regenerate() sums). Returns:
      Cb        : list over m of (K, d) CENTERED atoms (per-family mean over v removed) -- offset-free factor codebook.
      M_off     : (d,) the exact fact-INDEPENDENT bundle offset = sum_m mean_v codebook_m[v]  (label-free; codebook-only).
    The centering + M_off removal make the factorization operate on the pure centered-atom sum s_c = sum_m Cb_m[v]."""
    M, K, d = gen.M, gen.K, gen.d
    raw = np.zeros((M, K, d), dtype=np.float64)
    for m in range(M):
        for v in range(K):
            assert v in gen.W[m], f"codebook missing family {m} value {v} (coverage should guarantee all seen)"
            raw[m, v] = gen._elig(gen._off[m] + v) @ gen.W[m][v] + gen._anchor
    mu = raw.mean(axis=1)                       # (M, d) per-family mean over v
    Cb = raw - mu[:, None, :]                   # (M, K, d) centered atoms
    M_off = mu.sum(axis=0)                       # (d,) exact fact-independent offset (sum of per-family means)
    return Cb, M_off


def _resonator_single(s_c, Cb, norm, init_v, max_iters):
    """ONE resonator run from a given initial assignment. SEQUENTIAL (Gauss-Seidel) hard-WTA explain-away over a
    SUPERPOSITION (sum) bundle: sweep the families in order; for each m, residual_m = s_c - sum_{m'!=m} x_hat_{m'};
    pick the atom with the largest matched-filter projection onto residual_m (offset-invariant: centered codebooks),
    reconstruct, and IMMEDIATELY use it for the next family. Sequential updates are far more stable than synchronous
    ones (Kent 2020: synchronous hard-WTA oscillates even when the codes are cleanly separable). Fixed point = a full
    sweep with no change. A repeated post-sweep assignment => LIMIT CYCLE (converged=False). Returns
    (tuple v_hat, converged, n_iter)."""
    M, K, d = Cb.shape
    v_hat = list(init_v)
    x_hat = np.stack([Cb[m, v_hat[m]] for m in range(M)])   # (M, d)
    history = {tuple(v_hat)}
    converged, n_iter = False, 0
    for it in range(max_iters):
        n_iter = it + 1
        changed = False
        for m in range(M):
            residual = s_c - (x_hat.sum(axis=0) - x_hat[m])
            vm = int(np.argmax((Cb[m] @ residual) / norm[m]))
            if vm != v_hat[m]:
                changed = True
                v_hat[m] = vm
                x_hat[m] = Cb[m, vm]
        if not changed:                                     # FIXED POINT (a full sweep left it unchanged)
            converged = True
            break
        t = tuple(v_hat)
        if t in history:                                    # LIMIT CYCLE (period > 1)
            converged = False
            break
        history.add(t)
    return tuple(v_hat), converged, n_iter


def _resonator_factorize(s_c, Cb, max_iters, n_restarts, rng):
    """Frady-Kanerva resonator / matching-pursuit explain-away factorization of a SUM bundle, with random restarts
    (Frady/Kent standard practice near capacity). Runs _resonator_single from a warm start (each family matched to the
    full bundle) plus n_restarts-1 random initial assignments; keeps the best fixed point (converged first, then lowest
    reconstruction residual). Returns (v_hat, any_converged, mean n_iter, best residual). any_converged=True iff SOME
    restart reached a fixed point -- so a fact counts as non-convergent only when EVERY restart limit-cycled (the honest
    Kent-2020 signature of a resonator that cannot settle, distinct from settling on the WRONG factorization)."""
    M, K, d = Cb.shape
    norm = np.sqrt((Cb ** 2).sum(axis=2)) + 1e-12           # (M, K) atom norms for the matched filter
    warm = [int(np.argmax((Cb[m] @ s_c) / norm[m])) for m in range(M)]
    inits = [warm] + [[int(rng.integers(K)) for _ in range(M)] for _ in range(max(0, n_restarts - 1))]
    best = None                                             # (v_hat, converged, n_iter, resid)
    any_converged = False
    iters = []
    for init_v in inits:
        v_hat, converged, n_iter = _resonator_single(s_c, Cb, norm, init_v, max_iters)
        iters.append(n_iter)
        recon = np.stack([Cb[m, v_hat[m]] for m in range(M)]).sum(axis=0)
        resid = float(np.linalg.norm(s_c - recon))
        any_converged = any_converged or converged
        if best is None or ((converged and not best[1]) or (converged == best[1] and resid < best[3])):
            best = (v_hat, converged, n_iter, resid)
    return best[0], any_converged, float(np.mean(iters)), best[3]


def _exhaustive_decode(s_c, Cb):
    """WITNESS decoder: the ML-optimal factorization readout. Over ALL K^M value-combinations, reconstruct the
    centered-atom sum and return the combo with the smallest ||s_c - sum_m Cb_m[v_m]||, plus the SEPARATION MARGIN =
    the residual of the 2nd-best combo (the distance from s_c to the NEAREST WRONG bundle). Because regen is a noise-
    free exact atom sum (best residual ~1e-15), the margin is exactly the min gap to an alternative fact. It upper-
    bounds what ANY factorization readout can do, so:
      * exhaustive still 1.00 -> the noise-free bundle stays EXACTLY invertible (no two facts collide); a resonator
        drop is then its own local-minimum sub-optimality, NOT a representational limit.
      * the MARGIN shrinking toward the readout/noise scale -> the GENUINE capacity phenomenon: a margin (SNR) collapse
        (the VSA ~1/sqrt(#terms) statement is a margin statement), which erodes the practical readouts long before
        exact invertibility fails. Reads only codebooks + s_c; never the true tuple/prototype."""
    M, K, d = Cb.shape
    best_t, best_r, second_r = None, np.inf, np.inf
    for combo in itertools.product(range(K), repeat=M):
        recon = np.stack([Cb[m, combo[m]] for m in range(M)]).sum(axis=0)
        r = float(np.linalg.norm(s_c - recon))
        if r < best_r:
            second_r = best_r; best_r, best_t = r, combo
        elif r < second_r:
            second_r = r
    return best_t, float(second_r)


# ============================ one (rho, r_shared, channel_mode) arm ============================
def _run_arm(seed, M, K, d, rho, r_shared, channel_mode, m_held, noise, gen_hidden, gen_k, gen_settle, gen_lr,
             w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws, res_max_iters, res_restarts):
    facts, taught_idx, held_idx = _heldout_split_M(M, K, m_held, seed)
    N = len(facts)
    prims, proto_fn, d_p = _make_world_corr(M, K, d, channel_mode, seed, rho, r_shared)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    protos = np.stack([proto_fn(facts[j]) for j in range(N)]).astype(np.float64)     # test-time RULER only
    action_ctx = _action_ctx_const()
    draw_rng = np.random.default_rng(int(seed) + 909 + (0 if channel_mode == "shared" else 1))

    def engram_of(j):
        p = protos[j]
        noisy = p[None, :] + draw_rng.standard_normal((n_draws, d_p)) * noise
        e = np.zeros(n_in, dtype=np.float64)
        e[:d_p] = noisy.mean(axis=0)
        e[d_p:] = action_ctx
        return e

    # anti-cheats (disjoint split + coverage + no-leakage)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint_split = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    seen_vals = [set() for _ in range(M)]
    for j in taught_idx:
        for m in range(M):
            seen_vals[m].add(facts[j][m])
    coverage_ok = bool(all(all(facts[j][m] in seen_vals[m] for m in range(M)) for j in held_idx))
    assert disjoint_split, "taught/held-out must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive must appear in >= 1 taught fact"

    gen = CompositionalGeneratorM(gen_k, n_in, M, K, d, channel_mode, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                  bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    flat = _FlatStoreM(d_p, seed)
    fed = []
    for j in taught_idx:
        e = engram_of(j)
        gen.learn_fact(facts[j], e, action_ctx)
        flat.learn(j, e)
        fed.append(j)
    assert not (set(fed) & held_set), "held-out leaked into training"

    regen = {j: gen.regenerate(facts[j])[:d_p] for j in range(N)}

    # ---- Euclidean-CORRECTED readout (the established headline; common-mode removed, like the corrected runner) ----
    C_hat = np.mean([regen[j] for j in taught_idx], axis=0)
    protos_mu = protos.mean(axis=0)
    protos_c = protos - protos_mu[None, :]

    def pred_corrected(j):
        return _nearest_proto(regen[j] - C_hat, protos_c)

    euclid_corrected_held = _recall_fraction(held_idx, pred_corrected, protos_c)
    # cos-to-true of the common-mode-removed regeneration (a property of the REPRESENTATION, shared across readouts)
    cos_to_true_held = float(np.mean([_cos(regen[j] - C_hat, protos_c[j]) for j in held_idx])) if held_idx else float("nan")

    out = {
        "channel_mode": channel_mode, "rho": rho, "r_shared": r_shared, "M": M, "K": K, "d": d, "d_p": d_p, "N": N,
        "chance": chance, "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "offdiag_cos": _mean_abs_offdiag_cos(prims, M, K),
        "euclid_corrected_heldout_recall": euclid_corrected_held,
        "cos_to_true_heldout": cos_to_true_held,
        "coverage_ok": coverage_ok, "taught_heldout_disjoint": disjoint_split,
        "stored_raw_patterns": int(gen._stored_raw_patterns), "used_ruler": bool(gen._used_ruler),
    }

    # ---- RESONATOR readout: only defined for the SHARED bundle (disjoint is a trivial per-block factorization) ----
    if channel_mode == "shared":
        Cb, M_off = _build_codebook(gen)
        # WITNESS the codebook does not leak the held-out proto: it is built from seen primitives only. The offset we
        # remove is codebook-only (label-free). Record the reconstruction error of regen from the codebook atoms (must
        # be ~0: regen IS the atom sum -> proves the resonator factorizes the composer's own generation, not the ruler).
        recon_err = float(np.mean([np.linalg.norm(regen[j][:d] - (M_off + sum(Cb[m, facts[j][m]] for m in range(M))))
                                   for j in held_idx])) if held_idx else 0.0
        res_rng = np.random.default_rng(int(seed) + 515151)   # deterministic restart inits
        exhaustive_ok = bool(K ** M <= 20000)                 # ML-optimal witness feasible only for small K^M
        res_hits, res_conv, res_iters, res_conv_hits, exh_hits, margins = 0, 0, [], 0, 0, []
        for j in held_idx:
            s_c = regen[j][:d] - M_off
            v_hat, converged, n_iter, _resid = _resonator_factorize(s_c, Cb, res_max_iters, res_restarts, res_rng)
            correct = (tuple(v_hat) == tuple(facts[j]))
            res_hits += int(correct)
            res_conv += int(converged)
            res_iters.append(n_iter)
            if converged:
                res_conv_hits += int(correct)
            if exhaustive_ok:
                exh_t, margin = _exhaustive_decode(s_c, Cb)
                exh_hits += int(exh_t == tuple(facts[j]))
                margins.append(margin)                        # separation to the nearest WRONG bundle
        nH = max(1, len(held_idx))
        # noise scale on the d-channel percept mean-of-n_draws (the downstream read noise the margin competes with)
        noise_scale = float(noise * np.sqrt(d) / np.sqrt(max(1, n_draws)))
        out["resonator_heldout_recall"] = float(res_hits / nH)
        out["resonator_convergence_rate"] = float(res_conv / nH)
        out["resonator_recall_given_converged"] = float(res_conv_hits / res_conv) if res_conv > 0 else float("nan")
        out["resonator_mean_iters"] = float(np.mean(res_iters)) if res_iters else 0.0
        out["exhaustive_heldout_recall"] = float(exh_hits / nH) if exhaustive_ok else float("nan")  # ML-optimal witness
        out["separation_margin"] = float(np.mean(margins)) if margins else float("nan")   # min gap to a wrong fact
        out["separation_margin_over_noise"] = (float(np.mean(margins)) / (noise_scale + 1e-12)) if margins else float("nan")
        out["resonator_codebook_recon_err"] = recon_err       # ~0 witnesses regen == sum of codebook atoms
        out["resonator_offset_norm"] = float(np.linalg.norm(M_off))
    return out


def run(seed, M, K, d, r_shared_list, rhos, held_frac, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
        bdsp_wmax, conv_tol, conv_max_epochs, n_draws, res_max_iters, res_restarts):
    N = K ** M
    m_held = max(1, int(round(held_frac * N)))
    m_held = min(m_held, N - (K ** (M - 1)))
    by_cell = {}
    for r_shared in r_shared_list:
        for rho in rhos:
            key = f"rho{rho:.4f}_r{r_shared}"
            shared = _run_arm(seed, M, K, d, rho, r_shared, "shared", m_held, noise, gen_hidden, gen_k, gen_settle,
                              gen_lr, w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws, res_max_iters, res_restarts)
            disjoint = _run_arm(seed, M, K, d, rho, r_shared, "disjoint", m_held, noise, gen_hidden, gen_k, gen_settle,
                                gen_lr, w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws, res_max_iters, res_restarts)
            by_cell[key] = {
                "rho": rho, "r_shared": r_shared, "N": N, "chance": shared["chance"],
                "held_out_n": shared["held_out_n"], "offdiag_cos": shared["offdiag_cos"],
                "resonator_recall": shared["resonator_heldout_recall"],
                "resonator_convergence_rate": shared["resonator_convergence_rate"],
                "resonator_recall_given_converged": shared["resonator_recall_given_converged"],
                "resonator_mean_iters": shared["resonator_mean_iters"],
                "exhaustive_recall": shared["exhaustive_heldout_recall"],
                "separation_margin": shared["separation_margin"],
                "separation_margin_over_noise": shared["separation_margin_over_noise"],
                "resonator_codebook_recon_err": shared["resonator_codebook_recon_err"],
                "euclid_corrected_recall": shared["euclid_corrected_heldout_recall"],
                "disjoint_corrected_recall": disjoint["euclid_corrected_heldout_recall"],
                "cos_to_true": shared["cos_to_true_heldout"],
                "shared_full": shared, "disjoint_full": disjoint,
            }
            print(f"  [rho={rho:.2f} r={r_shared}] off|cos| {shared['offdiag_cos']:.3f} | RESONATOR recall "
                  f"{shared['resonator_heldout_recall']:.2f} (conv {shared['resonator_convergence_rate']:.2f}, "
                  f"iters {shared['resonator_mean_iters']:.1f}) | ML-exhaustive {shared['exhaustive_heldout_recall']:.2f} "
                  f"| EUCLID-corr {shared['euclid_corrected_heldout_recall']:.2f} "
                  f"| disjoint {disjoint['euclid_corrected_heldout_recall']:.2f} | cos-to-true "
                  f"{shared['cos_to_true_heldout']:.3f} | recon-err {shared['resonator_codebook_recon_err']:.1e}",
                  flush=True)
    return {"seed": seed, "M": M, "K": K, "d": d, "r_shared_list": r_shared_list, "rhos": rhos, "by_cell": by_cell,
            "config": {"held_frac": held_frac, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "n_draws": n_draws,
                       "res_max_iters": res_max_iters, "res_restarts": res_restarts}}


def _classify_cell(c):
    """Per-cell read. The offset-invariant readout has TWO nested versions: the RESONATOR (biological, iterative) and
    the ML-optimal EXHAUSTIVE decoder (the best any factorization readout can do on this representation). Their split
    separates the three hypotheses cleanly:
      * exhaustive high, Euclidean craters  -> RULER-ARTIFACT: the representation IS identifiable; Euclidean's crater
        was the offset-sensitive ruler. (If resonator also high, the biological readout captures it too.)
      * exhaustive drops (even the ML-optimal decoder is wrong) -> GENUINE bundle ALIASING = a real representational
        capacity limit (distinct facts map to near-identical bundles), regardless of cos-to-true (mean direction can
        still point right while identity is lost).
      * exhaustive high but resonator low WITH low convergence -> the RESONATOR is the bottleneck (limit cycles),
        not the representation."""
    res, eu, conv, cos = (c["resonator_recall"], c["euclid_corrected_recall"],
                          c["resonator_convergence_rate"], c["cos_to_true"])
    exh = c.get("exhaustive_recall", float("nan"))
    exh_ok = not (isinstance(exh, float) and np.isnan(exh))
    if exh_ok and exh < 0.85:
        return f"GENUINE-CAPACITY-LIMIT (ML-optimal exhaustive decoder also fails: recall {exh:.2f} -> bundle aliasing)"
    if res >= 0.90 and eu < 0.70:
        return "RULER-ARTIFACT (offset-invariant readout recovers where Euclidean craters -> composer robust)"
    if res >= 0.90:
        return "both-fine"
    if conv < 0.70:
        return "RESONATOR-CONVERGENCE-FAILURE (recall drop is limit-cycle, NOT a representational limit)"
    if exh_ok and exh >= 0.90 and res < 0.90:
        return f"RESONATOR-SUBOPTIMAL (representation identifiable @ exhaustive {exh:.2f}; resonator local-minima lose it)"
    return "AMBIGUOUS (check exhaustive vs resonator vs convergence)"


def main():
    ap = argparse.ArgumentParser(description="Offset-invariant RESONATOR readout vs Euclidean-nearest-proto for the "
                                             "naturalistic shared-channel capacity question: does it recover recall "
                                             "where Euclidean craters, and is any residual drop a real capacity limit "
                                             "or a resonator convergence failure?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--r-shared", type=int, nargs="+", default=[2, 1], help="shared-subspace rank(s)")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95, 0.99],
                    help="code-correlation knob(s): 0=idealized independent, ->1=all codes in the r_shared subspace")
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96)
    ap.add_argument("--gen-k", type=int, default=96)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--res-max-iters", type=int, default=50)
    ap.add_argument("--res-restarts", type=int, default=8, help="resonator random restarts (Frady/Kent standard)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  M={a.M} K={a.K} d={a.d} r_shared={a.r_shared} rho={a.rho}\n" + "#" * 100,
              flush=True)
        result = run(s, a.M, a.K, a.d, a.r_shared, a.rho, a.held_frac, a.noise, a.gen_hidden, a.gen_k, a.gen_settle,
                     a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.n_draws, a.res_max_iters,
                     a.res_restarts)
        for key, c in result["by_cell"].items():
            c["classification"] = _classify_cell(c)
        summary = {"probe": "teacher_loop_arity_capacity_resonator", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "M": a.M, "K": a.K, "d": a.d, "r_shared": a.r_shared, "rhos": a.rho,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "out": out_s, "by_cell": result["by_cell"]})
        print(f"[res] wrote {out_s}", flush=True)

    # aggregate table across seeds
    print("\n" + "=" * 116, flush=True)
    print(f"[res AGG] M={a.M} K={a.K} d={a.d} | seeds {seeds}", flush=True)
    print(f"{'rho':>5} {'r':>2} | {'RESON':>6} {'conv':>5} {'ML-exh':>6} {'marg/no':>7} | {'EUCLID':>6} {'disj':>5} | "
          f"{'cos-tru':>7} {'off|cos|':>8} | classification", flush=True)
    keys = list(per_seed[0]["by_cell"].keys())
    agg_cells = {}
    for key in keys:
        cells = [p["by_cell"][key] for p in per_seed]
        def mean(fld):
            vals = [c[fld] for c in cells if not (isinstance(c[fld], float) and np.isnan(c[fld]))]
            return float(np.mean(vals)) if vals else float("nan")
        rho = cells[0]["rho"]; r = cells[0]["r_shared"]
        row = {
            "rho": rho, "r_shared": r,
            "resonator_recall_mean": mean("resonator_recall"),
            "resonator_convergence_rate_mean": mean("resonator_convergence_rate"),
            "resonator_recall_given_converged_mean": mean("resonator_recall_given_converged"),
            "exhaustive_recall_mean": mean("exhaustive_recall"),
            "separation_margin_mean": mean("separation_margin"),
            "separation_margin_over_noise_mean": mean("separation_margin_over_noise"),
            "euclid_corrected_recall_mean": mean("euclid_corrected_recall"),
            "disjoint_corrected_recall_mean": mean("disjoint_corrected_recall"),
            "cos_to_true_mean": mean("cos_to_true"),
            "offdiag_cos_mean": mean("offdiag_cos"),
            "resonator_recall_per_seed": [c["resonator_recall"] for c in cells],
            "resonator_convergence_per_seed": [c["resonator_convergence_rate"] for c in cells],
            "exhaustive_recall_per_seed": [c["exhaustive_recall"] for c in cells],
            "euclid_recall_per_seed": [c["euclid_corrected_recall"] for c in cells],
        }
        row["classification"] = _classify_cell({
            "resonator_recall": row["resonator_recall_mean"], "euclid_corrected_recall": row["euclid_corrected_recall_mean"],
            "resonator_convergence_rate": row["resonator_convergence_rate_mean"], "cos_to_true": row["cos_to_true_mean"],
            "exhaustive_recall": row["exhaustive_recall_mean"]})
        agg_cells[key] = row
        print(f"{rho:>5.2f} {r:>2} | {row['resonator_recall_mean']:>6.2f} "
              f"{row['resonator_convergence_rate_mean']:>5.2f} {row['exhaustive_recall_mean']:>6.2f} "
              f"{row['separation_margin_over_noise_mean']:>7.2f} | "
              f"{row['euclid_corrected_recall_mean']:>6.2f} {row['disjoint_corrected_recall_mean']:>5.2f} | "
              f"{row['cos_to_true_mean']:>7.3f} {row['offdiag_cos_mean']:>8.3f} | {row['classification']}", flush=True)

    agg = {"probe": "teacher_loop_arity_capacity_resonator_AGG", "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
           "M": a.M, "K": a.K, "d": a.d, "r_shared": a.r_shared, "rhos": a.rho, "by_cell_means": agg_cells,
           "per_seed": [{"seed": p["seed"], "out": p["out"]} for p in per_seed]}
    agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json")) if len(seeds) > 1 else a.out
    if len(seeds) > 1:
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print(f"[res AGG] wrote {agg_out}", flush=True)
    print("=" * 116, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
