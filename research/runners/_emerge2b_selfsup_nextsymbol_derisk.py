"""EMERGE-2b DE-RISK: self-supervised NEXT-SYMBOL prediction via burst credit (a DISCRIMINATING-control test).

EMERGE-2 (regression) showed a real self-supervised depth signal but its wrong-sign anti-cheat was non-discriminating
in multi-output regression (the output fit b from any hidden features). wrong-sign DID discriminate cleanly in
EMERGE-1b's CLASSIFICATION setup (0.545 vs deep 0.796). So this reframes the self-supervised test as CLASSIFICATION,
reusing EMERGE-1b's proven BurstpropMLP (softmax) + DendriticMLP arms -- where the controls bite.

THE SELF-SUPERVISED TASK: an observation whose lawful NEXT SYMBOL y (one of 4 classes) is a depth-2 function of the
context a -- y = 2*b0 + b1 where b0, b1 are threshold-of-XORs over disjoint subsets of a's clean pair-XORs (each needs
one hidden layer for the XORs + a second for the threshold). Predicting y is SELF-SUPERVISED: y is the lawful next part
of the observation, NOT an external human label -- the burst rule uses whatever target drives the soma (Urbanczik-Senn
unifies sup/unsup). Held-out = unseen contexts. If deep burst predicts y where shallow/lesion/wrong-sign can't, the
substrate DEVELOPED the depth-2 structure from self-supervised experience, cleanly attributable (the controls bite).

ARMS (identical task/splits/seeds; classification, softmax+CE): deep_burst_linearized (TEST) · vanilla_FA
(DendriticMLP local rule) · single_layer · oracle_bp (fenced backprop ceiling) · apical_lesion (Y=0 -> frozen hidden) ·
wrong_sign (must anti-learn -- the discriminating control) · no_teaching_null (output error zeroed -> no learning).
GO = deep_burst held-out >= 0.70 AND > vanilla_FA + 0.08 AND > apical_lesion + 0.10; probe(XOR latents) >= 0.70;
apical-lesion collapses; wrong-sign AND no-teaching-null at/near chance (0.25, 4-class); oracle >= 0.80; no weight
transport; multi-seed (42/43/44). Reuse-by-import; NO `sim/` edit; CPU. Run via run_in_background per seed for parallelism.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# PERF: tiny matmuls -- multi-threaded BLAS OVERSUBSCRIBES a many-core box (measured ~30x slower; EMERGE-5 finding).
# Force ONE BLAS thread per process (before numpy imports) and parallelize across SEEDS instead (main()).
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
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
from research.runners._emerge1b_burstprop_derisk import BurstpropMLP, _train  # noqa: E402 -- proven softmax machinery
from research.runners._emerge1_deep_dendritic_representation_derisk import (  # noqa: E402
    _hidden_rep, _probe_latents, N_BITS)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge2b_selfsup_nextsymbol.json"


def make_task(seed):
    """Observation context a (N_BITS bits); lawful next symbol y in {0,1,2,3} = 2*b0 + b1, b0/b1 = threshold-of-XORs of
    DISJOINT subsets of a's clean disjoint-pair XORs (depth-2, learnable per EMERGE-1). Self-supervised: y is the lawful
    next part of the observation, no external label. Returns (a_tr, y_tr, px_tr), (a_te, y_te, px_te); a in +/-1."""
    rng = np.random.default_rng(seed)
    n = 1 << N_BITS
    a = ((np.arange(n)[:, None] >> np.arange(N_BITS)[None, :]) & 1).astype(np.float64)
    px = np.logical_xor(a[:, 0::2].astype(bool), a[:, 1::2].astype(bool)).astype(float)  # 5 clean pair-XORs
    b0 = (px[:, [0, 1, 2]].sum(1) >= 2).astype(int)                # depth-2 over subset {0,1,2}
    b1 = (px[:, [2, 3, 4]].sum(1) >= 2).astype(int)                # depth-2 over subset {2,3,4}
    y = (2 * b0 + b1).astype(np.int64)                             # 4-class lawful next symbol
    X = a * 2.0 - 1.0
    idx = rng.permutation(n); cut = int(0.65 * n)
    tr, te = idx[:cut], idx[cut:]
    return (X[tr], y[tr], px[tr]), (X[te], y[te], px[te])


def run(seed, epochs, lr, batch, hidden):
    (Xtr, ytr, Ptr), (Xte, yte, Pte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 4]; shal = [N_BITS, hidden, 4]
    res = {"chance": float(max(np.bincount(yte, minlength=4)) / len(yte))}

    def _probe(net):
        return _probe_latents(_hidden_rep(net, Xtr), Ptr, _hidden_rep(net, Xte), Pte)

    # TEST + burst controls (BurstpropMLP, softmax). probe_latent is computed for EVERY arm (not just the TEST) --
    # the fresh-look review's mandatory representation-level anti-cheat: in multi-class classification, the OUTPUT
    # layer's gradient is ALWAYS correct regardless of mode (only the HIDDEN update is flipped/zeroed), so it can
    # "launder" a corrupted hidden representation into above-floor TASK accuracy (more classes = more readout degrees
    # of freedom to exploit residual structure). Probing the frozen hidden reps directly for the XOR latents is not
    # fooled by that -- it is the same check EMERGE-5 added after finding exactly this effect.
    for name, mode in [("deep_burst_linearized", "burst_linearized"), ("wrong_sign", "wrong_sign"),
                       ("apical_lesion", "apical_lesion"), ("no_teaching_null", "no_teaching_null")]:
        net = BurstpropMLP(deep, seed=seed)
        wt = all(not any(np.array_equal(Yk, w) or np.array_equal(Yk, w.T) for w in net.W) for Yk in net.Y)
        _train(net, Xtr, ytr, mode, epochs, lr, batch, seed)
        entry = {"heldout": float(net.accuracy(Xte, yte)), "train": float(net.accuracy(Xtr, ytr)),
                 "no_weight_transport": bool(wt), "probe_latent": _probe(net)}
        res[name] = entry
    # vanilla FA + single-layer + oracle (DendriticMLP)
    for name, sizes, mode in [("vanilla_FA", deep, "local_correct"), ("single_layer", shal, "local_correct"),
                              ("oracle_bp", deep, "oracle")]:
        net = DendriticMLP(sizes, seed=seed)
        _train(net, Xtr, ytr, mode, epochs, lr, batch, seed)
        res[name] = {"heldout": float(net.accuracy(Xte, yte)), "train": float(net.accuracy(Xtr, ytr))}
    return {"seed": seed, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        try:
            import functools
            from concurrent.futures import ProcessPoolExecutor
            fn = functools.partial(run, epochs=a.epochs, lr=a.lr, batch=a.batch, hidden=a.hidden)
            with ProcessPoolExecutor(max_workers=min(len(a.seeds), os.cpu_count() or 1)) as ex:
                per = list(ex.map(fn, a.seeds))
        except Exception:
            per = [run(s, a.epochs, a.lr, a.batch, a.hidden) for s in a.seeds]
        for r in per:
            s = r["seed"]
            print(f"  [seed {s}] deep_burst held {r['deep_burst_linearized']['heldout']:.3f} "
                  f"(probe {r['deep_burst_linearized']['probe_latent']:.3f}) | vanilla_FA {r['vanilla_FA']['heldout']:.3f}"
                  f" | single {r['single_layer']['heldout']:.3f} | lesion {r['apical_lesion']['heldout']:.3f} (probe "
                  f"{r['apical_lesion']['probe_latent']:.3f}) | wrong {r['wrong_sign']['heldout']:.3f} (probe "
                  f"{r['wrong_sign']['probe_latent']:.3f}) | null {r['no_teaching_null']['heldout']:.3f} (probe "
                  f"{r['no_teaching_null']['probe_latent']:.3f}) | oracle {r['oracle_bp']['heldout']:.3f} | chance "
                  f"{r['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mh(k):
            return float(np.mean([p[k]["heldout"] for p in per]))
        deep, fa, sing = mh("deep_burst_linearized"), mh("vanilla_FA"), mh("single_layer")
        les, wrong, null, orac = mh("apical_lesion"), mh("wrong_sign"), mh("no_teaching_null"), mh("oracle_bp")
        ch = float(np.mean([p["chance"] for p in per])); probe = float(np.mean([p["deep_burst_linearized"]["probe_latent"] for p in per]))
        les_probe = float(np.mean([p["apical_lesion"]["probe_latent"] for p in per]))
        wrong_probe = float(np.mean([p["wrong_sign"]["probe_latent"] for p in per]))
        null_probe = float(np.mean([p["no_teaching_null"]["probe_latent"] for p in per]))
        wt = all(p["deep_burst_linearized"]["no_weight_transport"] for p in per)
        task_ok = orac >= 0.80
        generalizes = (deep >= 0.70) and (deep > fa + 0.08) and (deep > les + 0.10)
        rep_ok = probe >= 0.70
        # REPRESENTATION-level gate (added per the fresh-look review + the EMERGE-5 finding): in 4-way classification
        # the always-correctly-trained OUTPUT layer can launder a corrupted hidden rep into above-floor accuracy, so
        # accuracy-only anti-cheats are not trustworthy here -- require the probe (chance=0.5 on balanced bits) to
        # ALSO stay near floor for a control to count as a clean pass, not just its readout accuracy.
        PROBE_FLOOR = 0.65
        lesion_collapses = les <= max(fa, ch) + 0.06 and les_probe <= PROBE_FLOOR
        wrong_anti = (wrong <= ch + 0.08) and (wrong_probe <= PROBE_FLOOR)
        null_flat = (null <= ch + 0.08) and (null_probe <= PROBE_FLOOR)
        go = bool(task_ok and generalizes and rep_ok and lesion_collapses and wrong_anti and null_flat and wt)
        if not task_ok:
            verdict = f"INCONCLUSIVE -- oracle only {orac:.3f}; tune (epochs/lr/hidden) before reading the burst arms."
        elif go:
            verdict = (f"GO -- burst credit assignment learns deep structure SELF-SUPERVISED, cleanly attributable: "
                       f"predicting the lawful next symbol (depth-2, no external label), deep burst held-out {deep:.3f} "
                       f">> vanilla_FA {fa:.3f} + apical-lesion {les:.3f} + chance {ch:.3f}; XOR latents emerged (probe "
                       f"{probe:.3f}); the DISCRIMINATING controls bite on BOTH readout AND representation -- wrong-sign "
                       f"{wrong:.3f} (probe {wrong_probe:.3f}) + no-teaching-null {null:.3f} (probe {null_probe:.3f}) at "
                       f"floor, apical-lesion collapses (probe {les_probe:.3f}); oracle {orac:.3f}; no weight transport. "
                       f"Multi-seed. ⇒ the emergent-cortex primitive holds SELF-SUPERVISED with clean attribution -- "
                       f"carry it to the spiking substrate. NO sim/ edit.")
        else:
            miss = []
            if not generalizes: miss.append(f"deep didn't beat floors (deep {deep:.3f} vs FA {fa:.3f}/lesion {les:.3f})")
            if not rep_ok: miss.append(f"probe {probe:.3f} < 0.70")
            if not lesion_collapses: miss.append(f"lesion didn't collapse (readout {les:.3f}, probe {les_probe:.3f})")
            if not wrong_anti: miss.append(f"wrong-sign not at chance (readout {wrong:.3f}, probe {wrong_probe:.3f})")
            if not null_flat: miss.append(f"null not at chance (readout {null:.3f}, probe {null_probe:.3f})")
            verdict = ("BOUNDARY (next mechanism, not a stop) -- " + "; ".join(miss) + f" (oracle {orac:.3f}). If the "
                       f"MISS is readout-only (probe near floor) it is the known output-layer-laundering artifact in "
                       f"multi-class classification (accuracy-only was fooled; representation-level is not) -- not a "
                       f"mechanism failure. If the probe itself is elevated, that is a genuine credit-assignment leak. "
                       f"Iterate: width/ensemble, or defer the clean self-supervised test to the substrate/stream.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge2b_selfsup_nextsymbol", "GO": go, "verdict": verdict,
               "task": f"self-supervised next-symbol classification: predict the lawful 4-class next symbol (2 depth-2 "
                       f"threshold-of-XORs of {N_BITS} bits); NO external label; discriminating (classification) controls",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Classification objective so the wrong-sign/null controls discriminate (they didn't in "
                              "EMERGE-2's regression). Self-supervised = the target is the lawful next part of the "
                              "observation, not an external label. Boundaries = the next mechanism. Oracle = fenced "
                              "backprop task-sanity."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge2b] VERDICT: {verdict}", flush=True)
    print(f"[emerge2b] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
