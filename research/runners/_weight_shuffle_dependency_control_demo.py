"""DEMONSTRATION with TEETH of the weight-shuffle DEPENDENCY control (tools.lab.dependency_control), on a REAL
structured circuit already in this repo: the plastic context->CA3 heteroassociative pathway `W_ctx` of the gap#5
WHEN episodic store (research/runners/_gap5_episodic_temporal_context_when_derisk.py).

THE QUESTION the control answers (Shiu & Sterne et al. 2024, Nature 634:210-219): does the store's RECENCY function
depend on the ACTUAL learned/structured `W_ctx` weights, or merely on their gross statistics (the weight histogram,
or the per-cell row sums)? Take the trained `W_ctx`, SHUFFLE its entries while preserving its value distribution,
re-run the recency read at a FIXED substrate seed, and require the recency gradient to COLLAPSE. If it survives a
distribution-preserving shuffle, the "temporal-context code" was riding on gross statistics -- an overclaim.

WHY THIS CIRCUIT IS A CLEAN TARGET: the WHEN runner's OWN anti-cheat is a context-LESION (W_ctx := 0) that already
shows the recency gradient is 100% carried by this pathway (the runner reports recency_attributable_to_context).
So the effect is known to ride on THIS matrix -- exactly the case a shuffle should collapse, and NOT the geometry-
carried case (gap#5 replay shuffle-bar finding) where a weight shuffle is insensitive. We report the W_ctx:=0
lesion score alongside the shuffle null as the reference floor.

measure_fn = the runner's OWN recency metric: newest-third-minus-oldest-third of held-cell apical-UP completion
(the runner's `range_intact`), read through the SAME spiking dendritic-dAP completion machinery, given a context
matrix W. Large positive => a graded recency gradient exists; ~0 => no gradient (function collapsed).

  Run: OMP_NUM_THREADS=2 SIM_BACKEND=numpy python -m research.runners._weight_shuffle_dependency_control_demo \
         --seeds 42 43 --n-shuffles 24 \
         --out research/findings/raw/_weight_shuffle/wshuffle_when_Wctx.json
The runner harness (research/runners/__init__.py) auto-writes the <out>.prov.json provenance sidecar.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from tools.lab import dependency_control  # noqa: E402
from research.runners._gap5_episodic_temporal_context_when_derisk import (  # noqa: E402
    build_and_form, GO_DEFAULTS, _drive_context_read, _up_fraction, _ctx_current, _third_ratio, _spearman)

OUT = _REPO / "research" / "findings" / "raw" / "_weight_shuffle" / "wshuffle_when_Wctx.json"


def make_recency_measure(S, *, drive_pA, ctx_pA):
    """Return measure_fn(W) = the runner's recency RANGE (newest-third minus oldest-third of held-cell apical-UP
    completion) given a context->CA3 matrix W, read through the SAME spiking completion machinery the runner uses.
    Deterministic: each read hard-silences and resets the apical latch before driving (no plasticity at read)."""
    def measure(W):
        vals = [_up_fraction(_drive_context_read(S.bridge, S.R, S.cue_by_asm[i],
                                                 _ctx_current(S, S.c_test, W, ctx_pA),
                                                 drive_pA=drive_pA, warm=S.warm, read=S.read),
                             S.held_global[i], S.up_thresh) for i in range(S.n_items)]
        _ratio, old, new = _third_ratio(vals)
        return float(new - old)
    return measure


def run_one_seed(seed, *, n_items, n_ctx, rho, beta, k_active, ctx_pA, ctx_lr, cue_frac, drive_pA, p,
                 n_ca3, n_shuffles, ratio, modes, verbose=True):
    t = {"seed": seed, "backend": os.environ.get("SIM_BACKEND", "(unset)")}
    S = build_and_form(seed, n_items=n_items, n_ctx=n_ctx, rho=rho, beta=beta, k_active=k_active, ctx_lr=ctx_lr,
                       cue_frac=cue_frac, drive_pA=drive_pA, p=p, preassigned=True, n_ca3_pre=n_ca3, verbose=verbose)
    if S.error:
        t["error"] = S.error
        return t
    measure = make_recency_measure(S, drive_pA=drive_pA, ctx_pA=ctx_pA)
    t["assembly_sizes"] = S.sizes
    t["n_ca3"] = S.n_ca3
    t["ctx_overlap_probe"] = S.ctx_overlap_probe
    # the runner's OWN anti-cheat as the reference floor: recency with the pathway zeroed (W_ctx := 0)
    t["real_score"] = measure(S.W_ctx)
    t["lesion_score_Wctx_zero"] = measure(S.W_les)
    t["modes"] = {}
    for mode in modes:
        if verbose:
            print(f"  [s{seed}] dependency_control mode={mode} n_shuffles={n_shuffles} ratio={ratio}", flush=True)
        dc = dependency_control(measure, S.W_ctx, np.random.default_rng(seed * 101 + 7),
                                n_shuffles=n_shuffles, mode=mode, ratio=ratio)
        # multiset-preservation assert on a fresh shuffle: the control must not silently change the weight values
        from tools.lab import shuffle_preserving_marginal
        Wsh = shuffle_preserving_marginal(S.W_ctx, np.random.default_rng(1), mode=mode)
        dc["multiset_preserved"] = bool(np.array_equal(np.sort(Wsh.reshape(-1)), np.sort(S.W_ctx.reshape(-1))))
        t["modes"][mode] = dc
    t["collapsed_all_modes"] = bool(all(t["modes"][m]["collapsed"] for m in modes))
    if verbose:
        g = t["modes"].get("global", {})
        print(f"  [s{seed}] REAL recency range={t['real_score']:+.4f} | W_ctx=0 lesion={t['lesion_score_Wctx_zero']:+.4f}"
              f" | GLOBAL-shuffle null mean={g.get('shuffled_mean'):+.4f} p95={g.get('shuffled_p95'):+.4f}"
              f" collapsed={t['collapsed_all_modes']}", flush=True)
    del S
    return t


def build_summary(per, seeds, cfg, elapsed, ratio, modes, err=None):
    valid = [p for p in per if not p.get("error")]
    n = len(valid)
    n_collapsed = sum(1 for p in valid if p.get("collapsed_all_modes"))
    real_ok = all(p.get("real_score", 0.0) > 0.15 for p in valid) if valid else False
    status = "DEMONSTRATED" if (n > 0 and n_collapsed == n and real_ok) else ("PARTIAL" if n_collapsed else "NO-COLLAPSE")
    verdict = (f"weight-shuffle dependency control on gap#5 WHEN W_ctx: {n_collapsed}/{n} substrate seeds show the "
               f"recency function COLLAPSE under a distribution-preserving shuffle in ALL modes {modes} "
               f"(real recency gradient present at every seed: {real_ok}). Tests dependence-on-structure, not "
               f"correctness. Control: Shiu & Sterne et al. 2024 Nature 634:210-219 (motor neuron 100/100 real vs "
               f"1/100 shuffled).")
    if err is not None:
        verdict = f"ERROR -- {err}"
    return {"probe": "weight_shuffle_dependency_control_demo",
            "source_runner": "research/runners/_gap5_episodic_temporal_context_when_derisk.py",
            "source_circuit": "W_ctx (context->CA3 heteroassociative pathway)",
            "helper": "tools.lab.dependency_control / shuffle_preserving_marginal",
            "control_source": "Shiu & Sterne et al. 2024, Nature 634:210-219; Ecker et al. 2022 eLife 71850 (column-shuffle)",
            "measure_fn": "recency RANGE (newest-third minus oldest-third held-cell apical-UP completion) = the runner's range_intact",
            "collapse_criterion": f"real > shuffled_p95 AND real >= {ratio}x shuffled_mean AND real > 0",
            "ratio": ratio, "modes": modes, "status": status, "verdict": verdict,
            "seed_waiver": "single-substrate-seed by design: the statistical population is the >=20 distribution-preserving "
                           "shuffles per seed (Shiu's 100 sims), replicated across substrate seeds for robustness",
            "seeds": seeds, "config": cfg, "n_seeds": len(seeds), "elapsed_seconds": elapsed,
            "n_seeds_collapsed": n_collapsed, "n_seeds_valid": n, "real_gradient_present_all_seeds": real_ok,
            "per_seed": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-ctx", type=int, default=200)
    ap.add_argument("--k-active", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.72)
    ap.add_argument("--beta", type=float, default=0.60)
    ap.add_argument("--ctx-pA", type=float, default=700.0)
    ap.add_argument("--ctx-lr", type=float, default=1.0)
    ap.add_argument("--cue-frac", type=float, default=0.15)
    ap.add_argument("--drive-pa", type=float, default=50.0)
    ap.add_argument("--n-ca3", type=int, default=300)
    ap.add_argument("--n-shuffles", type=int, default=24)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--modes", nargs="+", default=["global", "per_row"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    p = dict(GO_DEFAULTS)
    cfg = dict(n_items=a.n_items, n_ctx=a.n_ctx, k_active=a.k_active, rho=a.rho, beta=a.beta, ctx_pA=a.ctx_pA,
               ctx_lr=a.ctx_lr, cue_frac=a.cue_frac, drive_pA=a.drive_pa, n_ca3=a.n_ca3, n_shuffles=a.n_shuffles,
               ratio=a.ratio, modes=a.modes, preassigned=True, backend=os.environ.get("SIM_BACKEND", "(unset)"))
    print(f"[wshuffle] weight-shuffle dependency control on gap#5 WHEN W_ctx | seeds={a.seeds} cfg={cfg}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_one_seed(s, n_items=a.n_items, n_ctx=a.n_ctx, rho=a.rho, beta=a.beta, k_active=a.k_active,
                             ctx_pA=a.ctx_pA, ctx_lr=a.ctx_lr, cue_frac=a.cue_frac, drive_pA=a.drive_pa, p=p,
                             n_ca3=a.n_ca3, n_shuffles=a.n_shuffles, ratio=a.ratio, modes=a.modes, verbose=True)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True)
            print(f"  [seed {s}] done ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()
    summary = build_summary(per, a.seeds, cfg, round(time.time() - t0, 1), a.ratio, a.modes,
                            err=(err if (err is not None or not [q for q in per if not q.get("error")]) else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[wshuffle] {summary['status']}: {summary['verdict']}\n[wshuffle] wrote {a.out}\n"
          + "=" * 100, flush=True)
    return 0 if summary["status"] == "DEMONSTRATED" else 1


if __name__ == "__main__":
    sys.exit(main())
