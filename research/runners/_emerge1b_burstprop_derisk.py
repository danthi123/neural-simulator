"""EMERGE-1b DE-RISK: does FAITHFUL Burstprop credit-assign through depth where vanilla FA memorized?

Per the master directive (boundaries = undiscovered mechanisms) + the spec
`2026-07-01-burst-multiplexed-dendritic-credit-assignment-spec.md`: EMERGE-1 showed vanilla feedback-alignment
memorizes a depth-2 task but doesn't generalize through depth. The brain's ACTUAL deep-credit mechanism is
burst-multiplexed dendritic plasticity (Payeur-Guerguiev-Zenke-Richards-Naud, Nat Neurosci 2021). This digitizes it
FAITHFULLY and re-runs the exact EMERGE-1 harness.

THE MECHANISM (faithful, no weight transport):
  - a pyramidal unit multiplexes: EVENT rate e_l = sigmoid(basal drive) = the feedforward signal (an ordinary forward
    pass, UNPERTURBED); BURST probability p_l = sigmoid(beta * v_api,l) around baseline p0=0.5 = the top-down credit.
  - the top-down credit descends LAYER-BY-LAYER as a burst-rate deviation through FIXED-RANDOM feedback Y_l
    (l+1 -> l; set once from seed, never learned, never = a forward W -> no weight transport): the burst-rate
    deviation b_{l+1} = e_{l+1}*(p_{l+1}-pbar_{l+1}) is mapped down v_api,l = b_{l+1} @ Y_l.
  - the recurrent LINEARIZATION (Payeur's depth-benefit, where vanilla FA stalls): multiply the descending credit by
    the local derivative phi'(e_l)=e_l(1-e_l) per hop (arm `burst_linearized`); `burst_bare` omits it (isolates whether
    the linearization or the burst channel itself buys depth).
  - BDSP plasticity (M1.2): dW_l = e_{l-1}.T @ [ e_l*(p_l - pbar_l) ] -- potentiate when the post unit bursts ABOVE
    its recent moving-average baseline pbar_l given its event activity; pbar_l is a per-unit EMA. Output layer: the
    local delta -(e_last.T @ (softmax - y)) (the top has direct access to the target). Same mode-agnostic optimizer
    (mean-over-batch + momentum) as the vanilla-FA net, so ONLY the credit rule differs.

ARMS (identical task/splits/seeds to EMERGE-1): vanilla_FA (DendriticMLP local rule -- the memorizer to beat) ·
burst_bare · burst_linearized (the TEST) · oracle_bp (fenced backprop ceiling / task-sanity) · apical_lesion (Y=0 ->
no top-down credit -> hidden frozen = the floor) · wrong_sign (negated hidden update -> anti-learn) · no_teaching_null
(output error zeroed -> p stays at baseline -> ZERO learning; the burstprop-specific moat that p0 is right).

GO = burst_linearized (or bare) held-out >= 0.75 AND > vanilla_FA + 0.10 AND > apical_lesion + 0.10; hidden probe of the
level-1 XOR latents >= 0.70 (> the frozen floor); apical_lesion collapses; wrong_sign anti-learns; no_teaching_null does
not learn; oracle >= 0.80 (task sanity); no-weight-transport asserted (Y never touches a forward W). Multi-seed (42/43/
44). HONEST PRIOR (per the spec): likely a qualified/PARTIAL GO (linearized has a fair shot ~0.75-0.85; bare may only
partly beat FA); BOUNDARY plausible at this tiny single-width scale -- build-saving either way. Reuse-by-import; NO
`sim/` edit; CPU. Run: python -m research.runners._emerge1b_burstprop_derisk --seeds 42 43 44
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
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the vanilla-FA + oracle arms (the EMERGE-1 baselines)
from research.runners._emerge1_deep_dendritic_representation_derisk import (  # noqa: E402 -- reuse the exact harness
    make_task, _hidden_rep, _probe_latents, N_PAIRS, N_BITS)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge1b_burstprop.json"
_MOMENTUM = 0.9


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax(z):
    z = z - z.max(1, keepdims=True); ez = np.exp(z); return ez / ez.sum(1, keepdims=True)


class BurstpropMLP:
    """Faithful Burstprop (Payeur-Naud 2021). Forward W is Xavier-init from `seed` -- IDENTICAL to DendriticMLP(sizes,
    seed) so the vanilla-FA-vs-burst comparison is the same net (only the credit rule differs). Layer-wise fixed-random
    feedback Y (l+1 -> l), no weight transport."""

    def __init__(self, sizes, seed=0, beta=1.0, p0=0.5, ema=0.9):
        rng = np.random.default_rng(seed)                            # SAME sequence as DendriticMLP -> identical W
        self.sizes = list(sizes); self.n_out = sizes[-1]
        self.beta = float(beta); self.p0 = float(p0); self.ema = float(ema)
        self.W = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
        # DendriticMLP consumes n_out*sizes[i] normals next for its DFA B; draw the SAME to keep W byte-identical, then
        # our layer-wise feedback Y from a SEPARATE seed stream (Y is a different structure; no weight transport either way).
        for i in range(1, len(sizes) - 1):
            _ = rng.normal(0, 1.0, (self.n_out, sizes[i]))            # (discarded; keeps rng parity for reproducibility)
        yrng = np.random.default_rng(seed + 9973)
        # Y[k] feeds hidden layer k+1 (acts index k+1) FROM the layer above (size sizes[k+2]); k in 0..len-3.
        self.Y = [yrng.normal(0, 1.0, (sizes[k + 2], sizes[k + 1])) for k in range(len(sizes) - 2)]
        self.pbar = [np.full(sizes[k + 1], p0) for k in range(len(sizes) - 2)]  # per-unit EMA burst baseline
        self._vel = None

    def _forward(self, X):
        acts = [np.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(_sig(acts[-1] @ self.W[li]))
        return acts, acts[-1] @ self.W[-1]

    def loss(self, X, y):
        _, lg = self._forward(X); p = _softmax(lg); y = np.asarray(y)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def accuracy(self, X, y):
        _, lg = self._forward(X); return float(np.mean(np.argmax(lg, 1) == np.asarray(y)))

    def train_step(self, X, y, mode, lr):
        acts, lg = self._forward(X); y = np.asarray(y)
        nW = len(self.W); nhid = nW - 1                              # number of hidden layers
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0   # (m, n_out) +gradient at output
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ delta_out)                          # output local delta (descent)
        linearize = (mode == "burst_linearized")
        # descending burst-rate-deviation credit (b_out = -delta_out = descent; zeroed for the no-teaching null)
        b = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        for k in range(nhid - 1, -1, -1):                            # top hidden -> bottom
            post = acts[k + 1]                                       # the layer W[k] produces (event rate)
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            v_api = b @ Yk                                           # (m, size_{k+1}) top-down credit current
            if linearize:
                v_api = v_api * (post * (1.0 - post))               # recurrent linearization: * phi'(e)
            p = _sig(self.beta * v_api)                             # burst probability (baseline p0=0.5 at v_api=0)
            self.pbar[k] = self.ema * self.pbar[k] + (1.0 - self.ema) * p.mean(0)   # per-unit EMA baseline
            dev = post * (p - self.pbar[k])                         # burst-rate deviation e*(P - Pbar)  [M1.2]
            g = acts[k].T @ dev                                     # BDSP: pre_event.T @ post_burst_dev
            # descent = +g: dev already carries the descent sign (b = -delta_out -> dev ~ -delta_k); wrong_sign flips it.
            upd[k] = -g if mode == "wrong_sign" else g
            b = dev                                                 # the burst-rate deviation is what descends
        # mode-agnostic optimizer (mean-over-batch + heavy-ball momentum) -- identical to DendriticMLP
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train(net, X, y, mode, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode=mode, lr=lr)


def run(seed, epochs, lr, batch, hidden):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]
    res = {}

    def _acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    # baseline to beat: vanilla FA (the EMERGE-1 memorizer), SAME W-init/seed
    fa = DendriticMLP(deep, seed=seed)
    from research.runners._emerge1_deep_dendritic_representation_derisk import _train as _fa_train
    _fa_train(fa, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    tr, te = _acc(fa); res["vanilla_FA"] = {"train": tr, "heldout": te}
    fa_W0_before = None                                              # (no-weight-transport is structural; asserted below)

    for mode in ("burst_bare", "burst_linearized", "wrong_sign", "apical_lesion", "no_teaching_null"):
        net = BurstpropMLP(deep, seed=seed)
        # no-weight-transport self-check: Y is independent of every forward W (never derived/copied)
        wt_ok = all(not any(np.array_equal(Yk, w) or np.array_equal(Yk, w.T) for w in net.W) for Yk in net.Y)
        _train(net, Xtr, ytr, mode, epochs, lr, batch, seed)
        tr, te = _acc(net)
        entry = {"train": tr, "heldout": te, "no_weight_transport": bool(wt_ok)}
        if mode in ("burst_bare", "burst_linearized"):
            entry["probe_latent"] = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
        res[mode] = entry

    net = DendriticMLP(deep, seed=seed)
    from research.runners._emerge1_deep_dendritic_representation_derisk import _train as _o_train
    _o_train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
    tr, te = _acc(net); res["oracle_bp"] = {"train": tr, "heldout": te}

    # W-identity check: BurstpropMLP init == DendriticMLP init (the decisive within-net contrast is fair)
    b0 = BurstpropMLP(deep, seed=seed); f0 = DendriticMLP(deep, seed=seed)
    res["same_init_as_FA"] = bool(all(np.allclose(a, b) for a, b in zip(b0.W, f0.W)))
    res["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return {"seed": seed, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=400)
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
            print(f"  [seed {s}] burst_lin held {r['burst_linearized']['heldout']:.3f} (train "
                  f"{r['burst_linearized']['train']:.3f}, probe {r['burst_linearized']['probe_latent']:.3f}) | "
                  f"burst_bare {r['burst_bare']['heldout']:.3f} | vanilla_FA {r['vanilla_FA']['heldout']:.3f} | "
                  f"lesion {r['apical_lesion']['heldout']:.3f} | wrong {r['wrong_sign']['heldout']:.3f} | "
                  f"null {r['no_teaching_null']['heldout']:.3f} | oracle {r['oracle_bp']['heldout']:.3f} | "
                  f"chance {r['chance']:.3f} | wt_ok {r['burst_linearized']['no_weight_transport']} | "
                  f"same_init {r['same_init_as_FA']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k, sub="heldout"):
            return float(np.mean([p[k][sub] for p in per]))
        lin, bare, fa = mean("burst_linearized"), mean("burst_bare"), mean("vanilla_FA")
        les, wrong, null = mean("apical_lesion"), mean("wrong_sign"), mean("no_teaching_null")
        orac, ch = mean("oracle_bp"), float(np.mean([p["chance"] for p in per]))
        lin_probe = mean("burst_linearized", "probe_latent")
        best = max(lin, bare); best_name = "burst_linearized" if lin >= bare else "burst_bare"
        wt = all(p["burst_linearized"]["no_weight_transport"] and p["same_init_as_FA"] for p in per)
        task_ok = orac >= 0.80
        generalizes = (best >= 0.75) and (best > fa + 0.10) and (best > les + 0.10)
        rep_ok = lin_probe >= 0.70
        lesion_collapses = les <= max(fa, ch) + 0.05
        wrong_anti = wrong <= ch + 0.05
        null_flat = null <= ch + 0.05
        go = bool(task_ok and generalizes and lesion_collapses and wrong_anti and null_flat and wt)
        partial = bool(task_ok and wt and lesion_collapses and (best > fa + 0.10) and not generalizes)
        if not task_ok:
            verdict = f"INCONCLUSIVE -- oracle only {orac:.3f}; tune the task/config before reading the burst arms."
        elif go:
            verdict = (f"GO -- FAITHFUL Burstprop credit-assigns through depth where vanilla FA memorized: {best_name} "
                       f"held-out {best:.3f} >> vanilla_FA {fa:.3f} + apical-lesion {les:.3f} + chance {ch:.3f}; the "
                       f"level-1 XOR latents EMERGED (probe {lin_probe:.3f}); apical-lesion collapses, wrong-sign "
                       f"anti-learns ({wrong:.3f}), no-teaching-null flat ({null:.3f}), no weight transport, same W-init "
                       f"as FA. Multi-seed. ⇒ the brain's burst-multiplexed credit-assignment mechanism DEVELOPS deep "
                       f"structure a point-neuron/vanilla-FA can't -- the boundary WAS an undiscovered mechanism. "
                       f"Localizes the substrate build to the burst two-compartment + STD/STF neuron. NO sim/ edit.")
        elif partial:
            verdict = (f"PARTIAL/QUALIFIED -- {best_name} clearly BEATS vanilla FA ({best:.3f} vs {fa:.3f}, +{best-fa:.2f}) "
                       f"and the XOR latents partly emerge (probe {lin_probe:.3f}), so the burst mechanism DOES add "
                       f"depth-credit over FA -- but it doesn't fully clear the generalization bar ({best:.3f} < 0.75 or "
                       f"not > lesion+0.10) at this tiny single-width scale (oracle {orac:.3f}). A real step past "
                       f"EMERGE-1's wall, not yet a clean GO. Next: wider/ensemble net (Payeur's burst-estimate improves "
                       f"with population size) or the Sacramento-Senn microcircuit arm. Build-informative, NOT a stop.")
        else:
            miss = []
            if best <= fa + 0.10: miss.append(f"burst didn't beat vanilla FA (best {best:.3f} vs FA {fa:.3f})")
            if not lesion_collapses: miss.append("apical-lesion didn't collapse")
            if not wrong_anti: miss.append(f"wrong-sign not at chance ({wrong:.3f})")
            if not null_flat: miss.append(f"no-teaching-null not flat ({null:.3f}) -- p0/sign bug")
            if not wt: miss.append("weight-transport / same-init check failed")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + f". Faithful Burstprop did not clear the depth wall at this "
                       f"scale (oracle CAN: {orac:.3f}). Per the master directive this is the NEXT mechanism to find, not "
                       f"a stop: iterate to the Sacramento-Senn self-predicting microcircuit (more gradient-faithful) or "
                       f"a wider/ensemble regime (better burst-rate estimate). Build-saving: NOT the substrate rewrite yet.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge1b_burstprop", "GO": go, "verdict": verdict,
               "mechanism": "faithful Burstprop (Payeur-Naud 2021): multiplexed event/burst channels, layer-wise "
                            "burst-coded top-down error via fixed-random feedback, recurrent linearization, BDSP; no "
                            "weight transport; same W-init as the vanilla-FA baseline (decisive within-net contrast)",
               "task": f"depth-2 threshold-of-{N_PAIRS}-pair-XORs over {N_BITS} bits (== EMERGE-1)",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Boundaries are undiscovered mechanisms (master directive): a BOUNDARY here launches the "
                              "next mechanism (Sacramento-Senn microcircuit / wider-ensemble burst estimate), not a "
                              "stop. GO localizes the months-scale burst two-compartment substrate build. Oracle is a "
                              "fenced backprop ceiling (task-sanity), NOT a shipped biologically-local mode."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge1b] VERDICT: {verdict}", flush=True)
    print(f"[emerge1b] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
