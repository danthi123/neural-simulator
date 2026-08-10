"""NATURALISTIC-CODE ARITY CAPACITY DE-RISK (2026-08-10): does the composer's SHARED-channel bundle survive
CORRELATED (non-idealized) codes? The named open question of the corrected finding
(2026-08-10-shared-channel-arity-capacity-CORRECTED-DC-offset-artifact.md):

  "The naive ~1/sqrt(#terms) VSA break did NOT manifest ... [it needs] genuinely interfering codes (not just more
   disjoint families). That is the open next probe."

ALL prior composer results (bundle / bind / faithful-spiking / arity-3 / the corrected M*) were on IDEALIZED codes:
each primitive drawn independently ~N(0,I_d), so different families are near-orthogonal and the ONLY cross-term is a
FACT-INDEPENDENT centroid offset -- a removable DC artifact. With that offset removed, corrected shared-channel recall
was 1.00 through arity 6. THAT is the idealization this probe attacks.

THE LEVER (new): a CODE-CORRELATION knob rho. Every primitive code is drawn from a SHARED low-rank subspace mixed with
an independent full-d part:
    code_{m,v} = sqrt(1-rho) * indep_{m,v}  +  sqrt(rho) * (B @ g_{m,v})              (B: d x r_shared, SHARED across
                                                                                       ALL families; unit per-dim var)
  * rho = 0  -> reduces EXACTLY to the idealized independent baseline (near-orthogonal at large d).
  * rho -> 1 -> all M terms live in the SAME r_shared-dim subspace; a bundle of M>r_shared terms is LINEARLY
                DEPENDENT, so distinct facts collide in the shared channels -> a REAL, fact-dependent capacity limit.
The variance is held at 1 per dim for every rho, so noise-to-signal is constant across the sweep (not a confound).

WHY THIS IS *NOT* A REMOVABLE DC OFFSET. The shared-subspace contribution has ZERO mean across facts (the coeffs g are
zero-mean), so it does NOT enter the common-mode estimate C_hat = mean(taught regenerations). The interference is
FACT-SPECIFIC (which values co-occur), so subtracting one constant vector cannot remove it. The corrected metric
(already common-mode-removed) therefore still DROPS -- that is the signature of naturalistic crosstalk, not DC.

THE CLEAN CONTROL (isolates SHARED-channel superposition crosstalk). At each rho, TWO arms on the SAME reservoir
readout, SAME K, SAME N, SAME held-out split, SAME codes -- differing ONLY in channel geometry:
  * SHARED  : the M codes summed into ONE d-channel space (real VSA bundling). Low-rank codes -> collisions.
  * DISJOINT: the M codes on M separate d-blocks (concatenation). Each block stays separable even at low rank
              (a difference of two r-dim codes still has norm ~sqrt(2)), so disjoint should HOLD ~1.00 across rho.
The shared-vs-disjoint gap AT each rho isolates the pure shared-channel bundling cost under naturalistic interference.

SKEPTICAL CONTROLS (the mandatory ones):
  (A) NO-LEVER BASELINE: rho=0 must REPRODUCE the known corrected 1.00 (the corrected finding's headline). If it does
      not, the harness is broken, not the codes.
  (B) REAL-CROSSTALK vs RESIDUAL-DC: report the CORRECTED (common-mode-removed) recall AND cosine-to-true AND the
      DC-offset ratio. A REAL capacity drop = corrected recall DROPS + cosine-to-true DROPS while the DC-ratio does
      NOT grow with rho (the drop is signal loss, not a growing removable offset). If instead corrected recall
      RECOVERED after common-mode removal, or the DC-ratio grew in lockstep, the drop would be DC and REFUTED.
  (C) DISJOINT HOLDS: same codes, concatenated, must stay high -> the drop is superposition, not unlearnable codes.

VERDICT (this is a capacity-LOCATION probe, not pass/fail):
  * POSITIVE (locates the naturalistic limit): rho=0 corrected ~1.00; corrected shared recall drops monotone-ish with
    rho by >= 0.30 total; cosine-to-true drops; DC-ratio does NOT explain it (flat/non-growing); disjoint holds.
  * NEGATIVE / refuted-lever: corrected recall stays ~1.00 across rho (correlation does not bite) OR the drop is a DC
    artifact (recovers after removal / DC-ratio grows in lockstep) OR disjoint also collapses (codes just unlearnable).

REUSE (no sim/ edit; no edit to the committed capacity runner): imports the frozen spiking reservoir generator
CompositionalGeneratorM, the FlatStore floor, the coverage-preserving split, and the recall/cos helpers.

RUN (single-seed smoke, numpy CPU):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_arity_capacity_correlated_derisk \
      --seed 42 --M 4 --K 3 --d 8 --r-shared 2 --rho 0.0 0.5 0.8 0.95 \
      --out research/findings/raw/teacher_loop_arity_capacity_correlated_s42.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
    CompositionalGeneratorM, _FlatStoreM, _all_facts, _heldout_split_M,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_arity_capacity_correlated.json"


# ============================ the CORRELATED-code world (host; legitimate as a retinal render is) ============================
def _make_world_corr(M, K, d, channel_mode, seed, rho, r_shared):
    """M attribute-families x K primitive codes. Each code mixes an INDEPENDENT full-d part with a SHARED low-rank
    part drawn from ONE basis B (shared across ALL families) -> cross-family confinement to an r_shared-dim subspace.
    rho=0 => independent (idealized baseline); rho->1 => all codes in span(B). Per-dim variance is 1 for every rho
    (norm ~ sqrt(d)) so noise-to-signal is constant across the sweep."""
    wr = np.random.default_rng(int(seed) + 30303030)
    r = int(r_shared)
    B = wr.standard_normal((d, r)).astype(np.float64) / np.sqrt(r)     # (B@g)_i has unit variance
    a_ind = float(np.sqrt(max(0.0, 1.0 - rho)))
    a_sub = float(np.sqrt(max(0.0, rho)))
    prims = []
    for m in range(M):
        fam = []
        for _v in range(K):
            indep = wr.standard_normal(d).astype(np.float64)
            g = wr.standard_normal(r).astype(np.float64)
            fam.append((a_ind * indep + a_sub * (B @ g)).astype(np.float64))
        prims.append(fam)
    d_p = d if channel_mode == "shared" else M * d

    def proto(values):
        if channel_mode == "shared":
            acc = np.zeros(d, dtype=np.float64)
            for m in range(M):
                acc += prims[m][int(values[m])]
            return acc
        return np.concatenate([prims[m][int(values[m])] for m in range(M)]).astype(np.float64)

    return prims, proto, d_p


def _mean_abs_offdiag_cos(prims, M, K):
    """Diagnostic: mean |cosine| between codes of DIFFERENT families (the naturalistic non-orthogonality)."""
    codes_by_fam = [[prims[m][v] / (np.linalg.norm(prims[m][v]) + 1e-12) for v in range(K)] for m in range(M)]
    vals = []
    for m in range(M):
        for m2 in range(m + 1, M):
            for v in range(K):
                for v2 in range(K):
                    vals.append(abs(float(codes_by_fam[m][v] @ codes_by_fam[m2][v2])))
    return float(np.mean(vals)) if vals else 0.0


# ============================ one (rho, channel_mode) arm ============================
def _run_arm(seed, M, K, d, rho, r_shared, channel_mode, m_held, noise, gen_hidden, gen_k, gen_settle, gen_lr,
             w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws):
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

    # anti-cheats
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

    # common-mode (DC) removal -- identical to the corrected runner. The corrected metric IS common-mode-removed.
    C_hat = np.mean([regen[j] for j in taught_idx], axis=0)
    protos_mu = protos.mean(axis=0)
    protos_c = protos - protos_mu[None, :]
    offset_norm = float(np.linalg.norm(C_hat - protos_mu))
    ip = [float(np.linalg.norm(protos[a] - protos[b])) for a in range(min(N, 24)) for b in range(min(N, 24)) if a < b]
    interproto = float(np.mean(ip)) if ip else 1.0
    offset_ratio = float(offset_norm / (interproto + 1e-12))

    def pred_biased(j):
        return _nearest_proto(regen[j], protos)

    def pred_corrected(j):
        return _nearest_proto(regen[j] - C_hat, protos_c)

    # ORACLE-DC control (the decisive real-crosstalk-vs-residual-DC discriminator, per the task). A DC artifact is BY
    # DEFINITION a single fact-INDEPENDENT constant; the best possible one to remove is C_exact = mean_j(regen-proto)
    # over ALL facts (uses true protos -> ORACLE/WITNESS ONLY, never usable in production, NOT part of the headline).
    # If removing this EXACT best constant RESTORES recall, the drop was residual DC; if recall STILL fails, the
    # residual is genuinely FACT-DEPENDENT crosstalk (a real capacity limit).
    C_exact = np.mean([regen[j] - protos[j] for j in range(N)], axis=0)
    def pred_oracle_dc(j):
        return _nearest_proto(regen[j] - C_exact, protos)

    comp_seen = _recall_fraction(taught_idx, pred_corrected, protos_c)
    comp_held = _recall_fraction(held_idx, pred_corrected, protos_c)              # HEADLINE (corrected)
    comp_held_biased = _recall_fraction(held_idx, pred_biased, protos)
    comp_held_oracle_dc = _recall_fraction(held_idx, pred_oracle_dc, protos)      # WITNESS: best-constant removal
    flat_held = _recall_fraction(held_idx, lambda j: flat.recall_nearest(j, protos), protos)
    # cosine-to-true of the COMMON-MODE-REMOVED regeneration vs the centered true proto (the real direction fidelity)
    comp_held_cos = float(np.mean([_cos(regen[j] - C_hat, protos_c[j]) for j in held_idx]))

    les_idx = held_idx[:min(6, len(held_idx))]
    lesion_delta = float(np.mean([float(np.linalg.norm(regen[j] - gen.regenerate(facts[j], lesion=0)[:d_p]))
                                  for j in les_idx])) if les_idx else 0.0

    return {
        "channel_mode": channel_mode, "rho": rho, "r_shared": r_shared, "M": M, "K": K, "d": d, "d_p": d_p, "N": N,
        "chance": chance, "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "offdiag_cos": _mean_abs_offdiag_cos(prims, M, K),
        "compositional_heldout_recall": comp_held,               # corrected = headline
        "compositional_heldout_recall_biased": comp_held_biased,
        "compositional_heldout_recall_oracle_dc": comp_held_oracle_dc,   # WITNESS: best-constant removal
        "compositional_seen_recall": comp_seen,
        "flat_heldout_recall": flat_held, "compositional_heldout_cos": comp_held_cos,
        "dc_offset_norm": offset_norm, "interproto_spacing": interproto, "dc_offset_ratio": offset_ratio,
        "lesion_delta": lesion_delta,
        "stored_raw_patterns": int(gen._stored_raw_patterns), "used_ruler": bool(gen._used_ruler),
        "coverage_ok": coverage_ok, "no_leakage": True, "taught_heldout_disjoint": disjoint_split,
    }


def run(seed, M, K, d, r_shared, rhos, held_frac, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
        conv_tol, conv_max_epochs, n_draws):
    N = K ** M
    m_held = max(1, int(round(held_frac * N)))
    m_held = min(m_held, N - (K ** (M - 1)))
    by_rho = {}
    for rho in rhos:
        print(f"\n{'=' * 96}\n# SEED {seed}  M={M} K={K} d={d} r_shared={r_shared} rho={rho}  (N={N}, held={m_held})\n"
              f"{'=' * 96}", flush=True)
        shared = _run_arm(seed, M, K, d, rho, r_shared, "shared", m_held, noise, gen_hidden, gen_k, gen_settle,
                          gen_lr, w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws)
        disjoint = _run_arm(seed, M, K, d, rho, r_shared, "disjoint", m_held, noise, gen_hidden, gen_k, gen_settle,
                            gen_lr, w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws)
        gap = float(disjoint["compositional_heldout_recall"] - shared["compositional_heldout_recall"])
        by_rho[f"{rho:.4f}"] = {
            "rho": rho, "N": N, "chance": shared["chance"], "held_out_n": shared["held_out_n"],
            "offdiag_cos": shared["offdiag_cos"],
            "shared_heldout_recall": shared["compositional_heldout_recall"],
            "shared_heldout_recall_biased": shared["compositional_heldout_recall_biased"],
            "shared_heldout_recall_oracle_dc": shared["compositional_heldout_recall_oracle_dc"],
            "disjoint_heldout_recall": disjoint["compositional_heldout_recall"],
            "flat_heldout_recall": shared["flat_heldout_recall"],
            "shared_seen_recall": shared["compositional_seen_recall"],
            "disjoint_minus_shared": gap,
            "shared_dc_offset_ratio": shared["dc_offset_ratio"],
            "shared_heldout_cos": shared["compositional_heldout_cos"],
            "disjoint_heldout_cos": disjoint["compositional_heldout_cos"],
            "shared_lesion_delta": shared["lesion_delta"], "disjoint_lesion_delta": disjoint["lesion_delta"],
            "shared_full": shared, "disjoint_full": disjoint,
        }
        print(f"  [rho={rho:.2f}] off-diag|cos| {shared['offdiag_cos']:.3f} | HELD-OUT (corrected): shared "
              f"{shared['compositional_heldout_recall']:.2f} [biased {shared['compositional_heldout_recall_biased']:.2f}] "
              f"| disjoint {disjoint['compositional_heldout_recall']:.2f} | flat {shared['flat_heldout_recall']:.2f} "
              f"(chance {shared['chance']:.4f}) | GAP {gap:+.2f} | DC-ratio {shared['dc_offset_ratio']:.2f} | cos "
              f"{shared['compositional_heldout_cos']:.3f}", flush=True)
    return {"seed": seed, "M": M, "K": K, "d": d, "r_shared": r_shared, "rhos": rhos, "by_rho": by_rho,
            "config": {"held_frac": held_frac, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "n_draws": n_draws}}


# ============================ verdict ============================
def _verdict(result):
    rhos = sorted(result["rhos"])
    by = result["by_rho"]
    r0 = by[f"{rhos[0]:.4f}"]
    rtop = by[f"{rhos[-1]:.4f}"]
    corrected = [by[f"{r:.4f}"]["shared_heldout_recall"] for r in rhos]
    disjoint = [by[f"{r:.4f}"]["disjoint_heldout_recall"] for r in rhos]
    oracle_dc = [by[f"{r:.4f}"]["shared_heldout_recall_oracle_dc"] for r in rhos]
    cos = [by[f"{r:.4f}"]["shared_heldout_cos"] for r in rhos]
    dcr = [by[f"{r:.4f}"]["shared_dc_offset_ratio"] for r in rhos]

    # (A) NO-LEVER BASELINE reproduces the known corrected ~1.00 at rho=0
    baseline_ok = bool(rhos[0] <= 1e-9 and r0["shared_heldout_recall"] >= 0.80)
    # (C) DISJOINT holds across the sweep (codes remain learnable; drop is superposition-specific, not bad codes)
    disjoint_holds = bool(min(disjoint) >= 0.70)
    # POSITIVE: corrected shared recall drops with rho by a real margin
    total_drop = float(corrected[0] - corrected[-1])
    real_drop = bool(total_drop >= 0.30 and corrected[-1] < 0.60)
    # (B) THE decisive skeptical control: at the top rho, removing the EXACT best constant (oracle DC) does NOT
    #     restore recall to the disjoint level -> the residual is FACT-DEPENDENT crosstalk, not residual DC.
    #     (If oracle-DC removal had recovered recall, the drop would have been a DC artifact and this REFUTES.)
    oracle_dc_top = oracle_dc[-1]
    not_dc_artifact = bool(oracle_dc_top < disjoint[-1] - 0.20 and oracle_dc_top < 0.70)

    located = bool(baseline_ok and real_drop and disjoint_holds and not_dc_artifact)
    return {
        "rhos": rhos,
        "shared_corrected_recall": corrected,
        "shared_biased_recall": [by[f"{r:.4f}"]["shared_heldout_recall_biased"] for r in rhos],
        "shared_oracle_dc_recall": oracle_dc,
        "disjoint_recall": disjoint,
        "shared_cos_to_true": cos,
        "shared_dc_offset_ratio": dcr,
        "offdiag_cos": [by[f"{r:.4f}"]["offdiag_cos"] for r in rhos],
        "gap_disjoint_minus_shared": [by[f"{r:.4f}"]["disjoint_minus_shared"] for r in rhos],
        "baseline_rho0_ok": baseline_ok,
        "disjoint_holds": disjoint_holds,
        "total_corrected_drop": total_drop,
        "real_capacity_drop": real_drop,
        "oracle_dc_recall_top": oracle_dc_top,
        "not_dc_artifact": not_dc_artifact,
        "naturalistic_limit_located": located,
        "verdict": ("POSITIVE-located-naturalistic-limit" if located
                    else ("NEGATIVE-correlation-does-not-bite" if total_drop < 0.30
                          else ("REFUTED-drop-was-residual-DC" if not not_dc_artifact and real_drop
                                else "AMBIGUOUS-check-controls"))),
        "lesion_delta_rho0_shared": r0["shared_lesion_delta"],
        "lesion_delta_top_shared": rtop["shared_lesion_delta"],
    }


def main():
    ap = argparse.ArgumentParser(description="Does the shared-channel bundle survive CORRELATED (naturalistic) codes? "
                                             "Sweep code-correlation rho at fixed arity; shared vs disjoint control.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--r-shared", type=int, default=2, help="rank of the shared subspace the codes are drawn from")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95],
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
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    result = run(a.seed, a.M, a.K, a.d, a.r_shared, a.rho, a.held_frac, a.noise, a.gen_hidden, a.gen_k, a.gen_settle,
                 a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.n_draws)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_arity_capacity_correlated", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "M": a.M, "K": a.K, "d": a.d, "r_shared": a.r_shared, "rhos": a.rho,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[corr] seed {a.seed}  M={a.M} K={a.K} d={a.d} r_shared={a.r_shared}", flush=True)
    for i, rho in enumerate(sorted(a.rho)):
        print(f"[corr] rho={rho:.2f}: off|cos|={verdict['offdiag_cos'][i]:.3f} | CORRECTED shared "
              f"{verdict['shared_corrected_recall'][i]:.2f} [biased {verdict['shared_biased_recall'][i]:.2f} | "
              f"oracle-DC {verdict['shared_oracle_dc_recall'][i]:.2f}] vs disjoint {verdict['disjoint_recall'][i]:.2f} "
              f"| cos-to-true {verdict['shared_cos_to_true'][i]:.3f} | DC-ratio "
              f"{verdict['shared_dc_offset_ratio'][i]:.2f}", flush=True)
    print(f"[corr] baseline(rho0)-ok={verdict['baseline_rho0_ok']} | corrected-drop "
          f"{verdict['total_corrected_drop']:+.2f} | disjoint-holds={verdict['disjoint_holds']} | "
          f"oracle-DC-top={verdict['oracle_dc_recall_top']:.2f} (not-DC={verdict['not_dc_artifact']}) | "
          f"VERDICT {verdict['verdict']}", flush=True)
    print(f"[corr] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["naturalistic_limit_located"] else 1


if __name__ == "__main__":
    sys.exit(main())
