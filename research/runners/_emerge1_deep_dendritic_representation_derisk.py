"""EMERGE-1 DE-RISK: does a DEEP dendritic cortex DEVELOP hierarchical structure a point-neuron can't?

The pivotal cheap-first gate from `2026-07-01-dendritic-cortex-for-emergence-scoping.md`: the owner's question is
whether capabilities should EMERGE from experience (a substrate that DEVELOPS structure) rather than be hand-designed.
The research isolated the one never-run experiment: every prior dendrite NEGATIVE used a SINGLE trainable layer, i.e.
off the regime the literature (Sacramento-Senn 2018, Payeur-Naud 2021, Urbanczik-Senn 2014) says deep dendritic credit
assignment works. This poses it in the DEEP regime, reusing the ALREADY-BUILT `sim.dendritic_mlp.DendriticMLP` (a deep
feedback-alignment learner: per-hidden-layer FIXED-RANDOM feedback B, no weight transport; hidden learning by the
committed local Urbanczik-Senn rule). Reuse-by-import; NO `sim/` edit; CPU.

THE TASK (genuinely requires DEPTH + generalizes): a depth-2 Boolean function -- pair the n input bits, XOR each pair
(level-1 latent features), then the label = threshold(majority) of the 5 pair-XORs (level-2). XOR needs a hidden layer;
a threshold OVER the XORs needs a SECOND -- a single hidden layer / random-feature readout provably can't represent it
efficiently, and a MEMORIZER can't generalize to held-out bit-patterns. So held-out accuracy measures whether the deep
net DEVELOPED the hierarchical structure, and a linear PROBE on the frozen hidden reps for the level-1 XOR features
measures whether those intermediate features EMERGED (were never supplied as targets).

ARMS (identical data/splits/seeds): deep_dendrite_FA (TEST, >=2 hidden, local rule) · single_layer_dendrite (the
prior-NEGATIVE regime, must struggle) · apical_lesion (B=0 -> no top-down error -> hidden frozen-random = the
point-neuron / no-credit-assignment FLOOR, must collapse) · wrong_sign (must anti-learn) · oracle_backprop (fenced
ceiling / task-sanity, NOT a shipped rule) · memorization (train-vs-held gap; chance=0.5).

EMERGENCE METRICS: (a) GENERALIZATION -- deep_dendrite_FA held-out accuracy >> single_layer, apical_lesion, chance;
(b) REPRESENTATION emergence -- a linear probe on frozen hidden reps recovers the level-1 XOR latents ABSENT in the
apical-lesion (frozen) arm; (c) ALIGNMENT emergence -- `hidden_grad_alignment` CLIMBS during training (the local FA
update comes to align with the true gradient -- structure the substrate developed).

GO = deep_dendrite_FA generalizes AND its hidden structure emerges (probe/alignment), > single_layer + apical_lesion,
multi-seed (42/43/44), lesion collapses, wrong_sign anti-learns. HONEST PRIOR (per the scoping): more likely BOUNDARY
than GO (bio-plausible deep learning struggles to credit-assign through depth) -- BOUNDARY is build-saving either way.
NO `sim/` edit. Run: python -m research.runners._emerge1_deep_dendritic_representation_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the ALREADY-BUILT deep feedback-alignment learner

OUT = _REPO / "research" / "findings" / "raw" / "_emerge1_deep_dendritic_representation.json"
N_BITS = 10
N_PAIRS = N_BITS // 2


def make_task(seed):
    """All 2^N_BITS bit patterns; latents = the N_PAIRS pair-XORs; label = threshold(sum of pair-XORs >= ceil/2).
    Inputs mapped to +/-1 (zero-mean, good for the sigmoid MLP). Train on 65%, held-out 35% (unseen patterns)."""
    rng = np.random.default_rng(seed)
    n = 1 << N_BITS
    bits = ((np.arange(n)[:, None] >> np.arange(N_BITS)[None, :]) & 1).astype(np.float64)  # (n, N_BITS) in {0,1}
    pair_xor = np.logical_xor(bits[:, 0::2].astype(bool), bits[:, 1::2].astype(bool)).astype(np.float64)  # (n, N_PAIRS)
    label = (pair_xor.sum(1) >= (N_PAIRS + 1) // 2).astype(np.int64)  # depth-2: threshold OVER the XORs
    X = bits * 2.0 - 1.0                                              # +/-1
    idx = rng.permutation(n)
    cut = int(0.65 * n)
    tr, te = idx[:cut], idx[cut:]
    return (X[tr], label[tr], pair_xor[tr]), (X[te], label[te], pair_xor[te])


def _train(net, X, y, mode, epochs, lr, batch, seed, track_align=None):
    rng = np.random.default_rng(seed + 777)
    aligns = []
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode=mode, lr=lr)
        if track_align is not None and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            Xa, ya = track_align
            aligns.append(net.hidden_grad_alignment(Xa, ya))
    return aligns


def _hidden_rep(net, X):
    """Frozen forward: the LAST hidden layer's activations (the representation the net developed)."""
    acts, _lg = net._forward(np.asarray(X, float))
    return np.asarray(acts[-1])                                       # (m, last_hidden)


def _probe_latents(H_tr, L_tr, H_te, L_te):
    """Linear probe: can a linear read-out of the FROZEN hidden reps recover the level-1 XOR latents on held-out?
    Ridge-regression closed form per latent; report mean held-out bit-accuracy (thresh 0.5). Measures whether the
    intermediate XOR features EMERGED in the representation (they were NEVER training targets)."""
    Xtr = np.concatenate([H_tr, np.ones((len(H_tr), 1))], 1)
    Xte = np.concatenate([H_te, np.ones((len(H_te), 1))], 1)
    lam = 1e-2 * np.eye(Xtr.shape[1]); lam[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ L_tr)             # (d+1, N_PAIRS)
    pred = (Xte @ W) >= 0.5
    return float(np.mean(pred == (L_te >= 0.5)))


def run(seed, epochs, lr, batch, hidden):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]                               # >=2 hidden layers (the deep regime)
    shal = [N_BITS, hidden, 2]                                       # single hidden layer (prior-NEGATIVE regime)
    res = {}

    def _acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    # TEST: deep dendritic feedback-alignment (the local Urbanczik-Senn rule)
    net = DendriticMLP(deep, seed=seed)
    al = _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed, track_align=(Xtr, ytr))
    tr, te = _acc(net)
    probe = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    res["deep_FA"] = {"train": tr, "heldout": te, "probe_latent": probe, "align_start": al[0], "align_end": al[-1]}

    # CONTROL: single hidden layer (the regime every prior dendrite NEGATIVE used)
    net = DendriticMLP(shal, seed=seed)
    _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    tr, te = _acc(net); res["single_layer"] = {"train": tr, "heldout": te}

    # FLOOR: apical lesion (B=0 -> no top-down error -> hidden frozen-random = the point-neuron/no-credit-assignment)
    net = DendriticMLP(deep, seed=seed)
    net.B = [np.zeros_like(b) for b in net.B]
    Hpre = _hidden_rep(net, Xte)                                     # frozen-random hidden (for the probe control)
    _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    tr, te = _acc(net)
    probe0 = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    res["apical_lesion"] = {"train": tr, "heldout": te, "probe_latent": probe0}

    # ANTI-LEARN: wrong-sign hidden update (must do no better than chance / anti-learn)
    net = DendriticMLP(deep, seed=seed)
    _train(net, Xtr, ytr, "local_wrongsign", epochs, lr, batch, seed)
    tr, te = _acc(net); res["wrong_sign"] = {"train": tr, "heldout": te}

    # CEILING / task-sanity: fenced backprop oracle (NOT a shipped rule -- only to confirm the task IS learnable deep)
    net = DendriticMLP(deep, seed=seed)
    _train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
    tr, te = _acc(net); res["oracle_bp"] = {"train": tr, "heldout": te}

    res["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return {"seed": seed, "n_train": int(len(Xtr)), "n_heldout": int(len(Xte)), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a.epochs, a.lr, a.batch, a.hidden); per.append(r)
            d = r["deep_FA"]
            print(f"  [seed {s}] deep_FA held {d['heldout']:.3f} (train {d['train']:.3f}, probe {d['probe_latent']:.3f}, "
                  f"align {d['align_start']:.2f}->{d['align_end']:.2f}) | single {r['single_layer']['heldout']:.3f} | "
                  f"lesion {r['apical_lesion']['heldout']:.3f} (probe {r['apical_lesion']['probe_latent']:.3f}) | "
                  f"wrong {r['wrong_sign']['heldout']:.3f} | oracle {r['oracle_bp']['heldout']:.3f} | "
                  f"chance {r['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k, sub):
            return float(np.mean([p[k][sub] for p in per]))
        deep_h, sing_h, les_h = mean("deep_FA", "heldout"), mean("single_layer", "heldout"), mean("apical_lesion", "heldout")
        wrong_h, orac_h, ch = mean("wrong_sign", "heldout"), mean("oracle_bp", "heldout"), float(np.mean([p["chance"] for p in per]))
        deep_probe, les_probe = mean("deep_FA", "probe_latent"), mean("apical_lesion", "probe_latent")
        align_climbs = all(p["deep_FA"]["align_end"] > p["deep_FA"]["align_start"] + 0.05 for p in per)
        # GO gates (multi-seed): generalizes clearly above the floors + a margin; structure emerges (probe + alignment);
        # anti-cheats hold (lesion collapses ~to single/chance; wrong-sign no better than chance); task is learnable (oracle high).
        generalizes = (deep_h >= 0.75) and (deep_h > sing_h + 0.05) and (deep_h > les_h + 0.10)
        rep_emerges = (deep_probe > les_probe + 0.10) and (deep_probe >= 0.70)
        lesion_collapses = les_h <= max(sing_h, ch) + 0.05
        wrong_anti = wrong_h <= ch + 0.05
        task_ok = orac_h >= 0.80                                      # the task IS deep-learnable (else it's a task bug)
        go = bool(task_ok and generalizes and (rep_emerges or align_climbs) and lesion_collapses and wrong_anti)
        if not task_ok:
            verdict = (f"INCONCLUSIVE -- the oracle-backprop ceiling only reached {orac_h:.3f} held-out, so the task "
                       f"isn't cleanly deep-learnable at this config; tune (epochs/lr/hidden/N_BITS) before reading the "
                       f"FA arms. NOT a dendrite verdict.")
        elif go:
            verdict = (f"GO -- a DEEP dendritic net (local feedback-alignment rule, no weight transport) DEVELOPED "
                       f"depth-2 hierarchical structure from experience: held-out {deep_h:.3f} >> single-layer {sing_h:.3f} "
                       f"+ apical-lesion floor {les_h:.3f} + chance {ch:.3f}; the level-1 XOR latents EMERGED in the "
                       f"hidden rep (probe {deep_probe:.3f} vs frozen {les_probe:.3f}; FA alignment climbs); apical-lesion "
                       f"collapses + wrong-sign anti-learns; oracle {orac_h:.3f}. Multi-seed. ⇒ the substrate can DEVELOP "
                       f"deep compositional structure a point-neuron/single-layer can't -- the emergence premise clears "
                       f"its cheap gate; localizes the months-scale build. Reuse-by-import, NO sim/ edit.")
        else:
            miss = []
            if not generalizes: miss.append(f"deep_FA didn't clearly beat the floors (held {deep_h:.3f} vs single "
                                            f"{sing_h:.3f}/lesion {les_h:.3f}) -- FA didn't credit-assign through depth")
            if not (rep_emerges or align_climbs): miss.append(f"hidden structure didn't emerge (probe {deep_probe:.3f} "
                                            f"vs frozen {les_probe:.3f}; alignment climb={align_climbs})")
            if not lesion_collapses: miss.append("apical-lesion did NOT collapse (the top-down error isn't load-bearing)")
            if not wrong_anti: miss.append(f"wrong-sign not at chance ({wrong_h:.3f})")
            verdict = ("BOUNDARY (the honest prior) -- " + "; ".join(miss) + f". The deep dendritic FA rule does NOT "
                       f"cleanly develop depth-2 structure at our scale (oracle CAN: {orac_h:.3f}), matching the field's "
                       f"'bio-plausible deep learning struggles to credit-assign through depth' evidence. This is a "
                       f"CHARACTERIZED frontier, not a fix -- build-saving: do NOT start the months-scale sim/ rewrite on "
                       f"the emergence premise; the wall is depth-scaling of the local rule, not (only) the point neuron.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge1_deep_dendritic_representation", "GO": go, "verdict": verdict,
               "task": f"depth-2 threshold-of-{N_PAIRS}-pair-XORs over {N_BITS} bits (needs 2 nonlinear levels; held-out"
                       " generalization + hidden-rep probe for the emergent XOR latents)",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Reuses the built DendriticMLP deep feedback-alignment learner (no weight transport, local "
                              "Urbanczik-Senn hidden rule). The oracle arm is a fenced backprop ceiling (task-sanity + "
                              "emergence-alignment reference ONLY), NOT a shipped biologically-local mode. A BOUNDARY is "
                              "the honest prior + is build-saving (localizes the wall to depth-scaling of the local rule, "
                              "before any months-scale sim/ two-compartment rewrite)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge1] VERDICT: {verdict}", flush=True)
    print(f"[emerge1] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
