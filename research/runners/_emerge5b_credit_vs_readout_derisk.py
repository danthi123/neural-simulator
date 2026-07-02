"""EMERGE-5b DE-RISK: is the EMERGE-5 spiking accuracy gap a CREDIT-signal failure or a READOUT failure?

EMERGE-5 (rate->spike Burstprop) at the realistic p0=0.03 found: the hidden representation PARTIALLY emerges under
spike noise (XOR-latent probe ~0.70, above the ~0.5 floor) but TASK ACCURACY sits at chance (~0.50 vs the rate
ceiling ~0.79). Two very different diagnoses -- and they point to different next levers:
  (A) READOUT bottleneck: the spiking-trained HIDDEN rep is actually good, but the noisy spike-count OUTPUT/readout
      can't turn it into task accuracy. Fix = a cleaner/averaged readout (cheap); the credit signal is fine.
  (B) CREDIT bottleneck: the noisy burst credit built an IMPAIRED hidden rep; no readout can rescue it. Fix = a
      better/ more-noise-robust credit rule (the Sacramento-Senn microcircuit's ACTIVE cancellation -- a real build).

THE TEST (cheapest-first, before committing to the microcircuit build): train the spiking Burstprop net at the healthy
width-384 config (where the oracle is ~1.0, so width-scaling is NOT a confound), FREEZE its hidden weights, then train
ONLY a fresh softmax readout on its (analytic, noise-free) hidden features with a CLEAN full-batch gradient. Compare:
  - own_spiking           : the spiking net's own end-to-end task accuracy (~chance, the symptom)
  - clean_readout_spiking  : a clean readout on the FROZEN spiking-trained hidden rep  <-- the diagnostic
  - rate_ceiling           : the analytic rate BurstpropMLP's task accuracy (the ceiling)
  - clean_readout_rate     : a clean readout on the rate net's frozen hidden (upper bound for "good rep + good readout")
  - clean_readout_untrained: a clean readout on a random/untrained net's hidden (random-features FLOOR)
VERDICT: READOUT-bottleneck if clean_readout_spiking >> own_spiking AND ~ rate_ceiling (the rep was good all along);
CREDIT-bottleneck if clean_readout_spiking stays near own_spiking / near the untrained floor (the rep is impaired).

NOTE: naive "population averaging" of the burst estimate is mathematically identical to raising the sample budget S
(pooling M independent Poisson/Binomial copies = Poisson(M*e*S) then Binomial), which EMERGE-5's S-sweep already tested
and which already failed to recover accuracy at p0=0.03 -- so that lever is exhausted; this diagnostic decides between
the two remaining ones (readout fix vs the microcircuit). Reuse-by-import; NO `sim/` edit; CPU.
Run: SIM_BACKEND=numpy python -m research.runners._emerge5b_credit_vs_readout_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
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
from research.runners._emerge1_deep_dendritic_representation_derisk import make_task, N_BITS  # noqa: E402
from research.runners._emerge1b_burstprop_derisk import BurstpropMLP, _train, _softmax  # noqa: E402
from research.runners._emerge5_spiking_burstprop_derisk import SpikingBurstpropMLP, _train_spk  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge5b_credit_vs_readout.json"


def _hidden(net, X):
    """The net's analytic (noise-free) LAST hidden-layer activations -- the representation to probe/read out."""
    acts, _lg = net._forward(np.asarray(X, float))
    return np.asarray(acts[-1])


def _clean_readout_acc(H_tr, y_tr, H_te, y_te, epochs=1200, lr=0.5, seed=0):
    """Train ONLY a fresh softmax readout on FROZEN features H (clean full-batch gradient, NO spike noise); return
    held-out task accuracy. Isolates 'is the task-relevant info linearly present in this hidden rep?'."""
    rng = np.random.default_rng(seed + 321)
    d = H_tr.shape[1]; nc = int(max(int(y_tr.max()), int(y_te.max())) + 1)
    W = rng.normal(0.0, 1.0 / np.sqrt(d), (d, nc)); b = np.zeros(nc)
    oh = np.eye(nc)[y_tr]; m = len(H_tr)
    vW = np.zeros_like(W); vb = np.zeros_like(b)
    for _ in range(epochs):
        g = _softmax(H_tr @ W + b) - oh
        vW = 0.9 * vW - lr * (H_tr.T @ g / m)
        vb = 0.9 * vb - lr * g.mean(0)
        W += vW; b += vb
    return float(np.mean(np.argmax(H_te @ W + b, 1) == y_te))


def run(seed, epochs, lr, batch, hidden, samples, p0):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]

    # spiking Burstprop net (the one whose accuracy lags) at the healthy width where the oracle is ~1.0
    spk = SpikingBurstpropMLP(deep, seed=seed, p0=p0)
    _train_spk(spk, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed, samples=samples)
    own = float(spk.accuracy(Xte, yte))
    cr_spk = _clean_readout_acc(_hidden(spk, Xtr), ytr, _hidden(spk, Xte), yte, seed=seed)

    # rate ceiling + its clean-readout upper bound
    rate = BurstpropMLP(deep, seed=seed)
    _train(rate, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed)
    ceil = float(rate.accuracy(Xte, yte))
    cr_rate = _clean_readout_acc(_hidden(rate, Xtr), ytr, _hidden(rate, Xte), yte, seed=seed)

    # random-features floor (untrained net's hidden)
    unt = BurstpropMLP(deep, seed=seed)
    cr_unt = _clean_readout_acc(_hidden(unt, Xtr), ytr, _hidden(unt, Xte), yte, seed=seed)

    chance = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return {"seed": seed, "own_spiking": own, "clean_readout_spiking": cr_spk, "rate_ceiling": ceil,
            "clean_readout_rate": cr_rate, "clean_readout_untrained": cr_unt, "chance": chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=384)              # the HEALTHY width (oracle ~1.0; no scaling confound)
    ap.add_argument("--samples", type=int, default=300)             # EMERGE-5 primary S
    ap.add_argument("--p0", type=float, default=0.03)               # EMERGE-4's measured resting burst prob
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        jobs = list(a.seeds)
        try:
            import functools
            from concurrent.futures import ProcessPoolExecutor
            fn = functools.partial(run, epochs=a.epochs, lr=a.lr, batch=a.batch, hidden=a.hidden,
                                   samples=a.samples, p0=a.p0)
            with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
                per = list(ex.map(fn, jobs))
        except Exception:
            per = [run(s, a.epochs, a.lr, a.batch, a.hidden, a.samples, a.p0) for s in jobs]
        for r in per:
            print(f"  [seed {r['seed']}] own_spiking {r['own_spiking']:.3f} | CLEAN-readout-on-spiking "
                  f"{r['clean_readout_spiking']:.3f} | rate_ceiling {r['rate_ceiling']:.3f} | clean-on-rate "
                  f"{r['clean_readout_rate']:.3f} | clean-on-untrained {r['clean_readout_untrained']:.3f} | "
                  f"chance {r['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([p[k] for p in per]))
        own, crs, ceil = m("own_spiking"), m("clean_readout_spiking"), m("rate_ceiling")
        crr, cru, ch = m("clean_readout_rate"), m("clean_readout_untrained"), m("chance")
        # readout-bottleneck: the FROZEN spiking hidden rep, read cleanly, recovers most of the ceiling AND far beats
        # the spiking net's own accuracy -> the rep was good; the noisy readout was the limit (cheap fix, credit is OK).
        readout_bottleneck = (crs >= ceil - 0.10) and (crs > own + 0.15) and (crs > cru + 0.15)
        # credit-bottleneck: even a clean readout can't rescue the spiking hidden rep (stays near own / near the
        # random-features floor) -> the noisy credit built an impaired rep -> need better credit (microcircuit arm).
        credit_bottleneck = (crs <= max(own, cru) + 0.10)
        sanity = (crr >= ceil - 0.10) and (cru <= ch + 0.15)         # clean-readout method itself is valid + floor is a floor
        if not sanity:
            verdict = (f"INCONCLUSIVE -- the clean-readout method itself misbehaves (clean-on-rate {crr:.3f} vs ceiling "
                       f"{ceil:.3f}; clean-on-untrained {cru:.3f} vs chance {ch:.3f}); fix the readout probe before "
                       f"reading the diagnosis.")
        elif readout_bottleneck:
            verdict = (f"READOUT-BOTTLENECK -- a clean readout on the FROZEN spiking-trained hidden rep recovers "
                       f"{crs:.3f} (vs the spiking net's own {own:.3f}, ~ the rate ceiling {ceil:.3f}, >> untrained "
                       f"floor {cru:.3f}). ⇒ the burst CREDIT signal built a GOOD representation under spike noise; the "
                       f"noisy spike-count READOUT was the accuracy bottleneck. Next lever = a cleaner/averaged OUTPUT "
                       f"read-out (cheap), NOT the microcircuit. The rate->spike credit assignment itself survives.")
        elif credit_bottleneck:
            verdict = (f"CREDIT-BOTTLENECK -- even a clean readout can't rescue the spiking hidden rep (clean-on-spiking "
                       f"{crs:.3f} ~ own {own:.3f} / untrained floor {cru:.3f}, far below the ceiling {ceil:.3f}). ⇒ the "
                       f"noisy burst credit built an IMPAIRED representation; a better readout won't fix it. Next lever = "
                       f"a more noise-robust credit rule -- the Sacramento-Senn microcircuit's ACTIVE cancellation "
                       f"(EMERGE-3 arm), per Urbanczik-Senn (population-feedback factor, not naive averaging).")
        else:
            verdict = (f"PARTIAL -- clean-on-spiking {crs:.3f} sits BETWEEN the floor ({max(own,cru):.3f}) and the "
                       f"ceiling ({ceil:.3f}): the spiking rep is PARTLY task-usable but not cleanly. BOTH a better "
                       f"readout AND better credit would help; the microcircuit arm (better credit) is the higher-"
                       f"leverage build, a cleaner readout the cheaper partial fix. Iterate, not a stop.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge5b_credit_vs_readout", "verdict": verdict,
               "question": "is EMERGE-5's spiking task-accuracy gap a CREDIT-signal failure (impaired hidden rep) or a "
                           "READOUT failure (good rep, noisy output)? -- decides microcircuit-arm vs readout-fix",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                                            "samples": a.samples, "p0": a.p0},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Diagnostic (not a GO/BOUNDARY gate): freezes the spiking-trained hidden rep + retrains "
                              "only a clean softmax readout on its noise-free activations. Runs at the HEALTHY width-384 "
                              "config (oracle ~1.0) so width-scaling is not a confound. Naive population-averaging was "
                              "excluded a priori (= raising the sample budget S, which EMERGE-5's S-sweep already tested "
                              "and which failed to recover accuracy at p0=0.03)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge5b] VERDICT: {verdict}", flush=True)
    print(f"[emerge5b] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
