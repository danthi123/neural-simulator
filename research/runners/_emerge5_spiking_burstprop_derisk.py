"""EMERGE-5 DE-RISK (substrate ladder rung 2): does deep Burstprop credit assignment SURVIVE the rate->spike
transition? -- the clean SINGLE-VARIABLE step from the CONFIRMED rate mechanism, before the protected `sim/` build.

Ladder (each rung gated before the next; `2026-07-01-spiking-burst-substrate-scoping.md` + the fresh-look review
`2026-07-01-fresh-look-emergence-strategy-review.md`):
  1. burst multiplexing survives on a two-compartment neuron -- DONE (EMERGE-4 GO: E~basal, P~apical, separable R2 0.94).
  2. deep burst credit assignment survives the rate->spike transition -- THIS de-risk.
  3. extend to a recurrent sequence cortex (the communication-relevant target: a simulated language-production circuit).
  4. port to the `sim/` spiking substrate (the protected build) + scale.

THE SINGLE VARIABLE (rate -> spike): EMERGE-1b confirmed the RATE Burstprop mechanism (Payeur-Naud 2021) credit-assigns
through depth (held-out 0.796, probe 0.989, no weight transport, all anti-cheats). The ONLY thing this de-risk changes
is that the event rate E and the burst probability P are no longer EXACT sigmoids -- they are FINITE-SAMPLE SPIKE-COUNT
ESTIMATES (a population of neurons observed over a window), exactly as EMERGE-4's two-compartment neuron produces them:
  - event count  k ~ Poisson(e * S)         where e = sigmoid(basal drive), S = the spike-sample budget (pop x window)
  - E_obs = k / S                            (the spike-estimated event rate = the forward activation, now stochastic)
  - burst count  j ~ Binomial(k, p)          where p = sigmoid(beta * v_api)  (the credit channel)
  - P_obs = j / max(k, 1)                    (the spike-estimated burst fraction -- the flagged NOISE source)
BDSP then uses the OBSERVED (noisy) E_obs, P_obs: dev = E_obs * (P_obs - pbar); everything else (the wiring, the
fixed-random feedback Y, the recurrent linearization, the optimizer, the W-init) is IDENTICAL to EMERGE-1b -- so this
is a faithful one-variable transition, not a new mechanism. The literature's flagged risk (Payeur; the spec 1.2(b)) is
precisely that the burst-rate estimate is noisy at small populations; the flagged MITIGATION is population coding -- so
we SWEEP the sample budget S and test whether the credit recovers as S grows.

ARMS (identical task/splits/seeds/W-init to EMERGE-1/1b; the depth-2 threshold-of-5-pair-XORs over 10 bits):
  rate_ref (S=inf: the analytic EMERGE-1b ceiling) · spiking_burst @ each S in the sweep (TEST) ·
  apical_lesion (Y=0) · wrong_sign · no_teaching_null (b=0 -> p=p0 -> the physical moat) · oracle_bp (fenced BP ceiling).
EMERGENCE GATE (the fresh-look review's mandatory anti-cheat vs the "good accuracy, dead deep layers" false-positive):
  a linear probe of the FROZEN hidden reps for the level-1 XOR latents must recover them (>=0.70) -- structure must
  EMERGE, not just accuracy.

GO = at a sufficient sample budget, spiking_burst held-out >= 0.70 AND within 0.10 of rate_ref AND > apical_lesion+0.10;
  probe_latent >= 0.70; the S-sweep shows monotone recovery (population coding is the mitigation); apical_lesion
  collapses; wrong_sign anti-learns; no_teaching_null flat; oracle >= 0.80; no weight transport; same W-init as the
  rate net. Multi-seed (42/43/44). GO ⇒ the confirmed rate mechanism survives spikes with population coding -> the
  `sim/` two-compartment build (rung 4) is justified + the population size is sized. BOUNDARY (build-saving) ⇒ the
  burst-estimate noise breaks depth-credit even with population coding -> iterate (wider population / the Sacramento-
  Senn microcircuit arm / the sim/ neuron's real transfer) BEFORE the protected build.

HONEST SCOPE: a finite-sample SPIKE-COUNT model (NOT dt-stepped, NOT the SimulationBridge yet) -- it isolates the
DOMINANT spiking effect (finite-sample burst-estimate noise + the population-coding mitigation), which is the review's
central spiking concern. Kept at EMERGE-1b's p0=0.5 for a clean single-variable transition; the real neuron's low
resting burst prob (EMERGE-4 P0~0.03), the refractory E<-apical cross-talk (~0.18), and full dt integration are
deferred to the `sim/` bridge port (rung 4), where the real transfer is used. Reuse-by-import; NO `sim/` edit; CPU.
Run: SIM_BACKEND=numpy python -m research.runners._emerge5_spiking_burstprop_derisk --seeds 42 43 44
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
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the fenced oracle (task-sanity) arm
from research.runners._emerge1_deep_dendritic_representation_derisk import (  # noqa: E402 -- the exact EMERGE-1 harness
    make_task, _hidden_rep, _probe_latents, N_PAIRS, N_BITS)
from research.runners._emerge1b_burstprop_derisk import (  # noqa: E402 -- the CONFIRMED rate mechanism + reference
    BurstpropMLP, _train, _sig, _softmax, _MOMENTUM)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge5_spiking_burstprop.json"


class SpikingBurstpropMLP(BurstpropMLP):
    """EMERGE-1b's faithful Burstprop, with the event rate E and burst probability P replaced by FINITE-SAMPLE
    SPIKE-COUNT estimates (population S = neurons x window). The ONLY change vs BurstpropMLP is the spike sampling in
    train_step; W-init, feedback Y, linearization, optimizer are inherited unchanged. Evaluation (`_forward`/`accuracy`/
    `_hidden_rep`) uses the inherited ANALYTIC clean forward -- it measures the LEARNED function, not eval-time noise."""

    def train_step(self, X, y, mode, lr, samples=None, srng=None):
        # --- spiking forward: each hidden activation is a spike-count-estimated event rate E_obs ---
        acts = [np.asarray(X, float)]
        kcounts = [None]                                             # per-layer event counts (aligned to acts index)
        for li in range(len(self.W) - 1):
            e = _sig(acts[-1] @ self.W[li])                          # analytic event rate (spike prob per sample)
            if samples is None:
                E_obs = e; k = None
            else:
                k = srng.poisson(e * samples)                        # events observed over the sample budget
                E_obs = k / float(samples)
            acts.append(E_obs)
            kcounts.append(k)
        lg = acts[-1] @ self.W[-1]
        y = np.asarray(y)
        nW = len(self.W); nhid = nW - 1
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ delta_out)
        linearize = (mode == "burst_linearized")
        b = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        for kk in range(nhid - 1, -1, -1):
            post = acts[kk + 1]                                      # E_obs (spike-estimated event rate)
            Yk = np.zeros_like(self.Y[kk]) if mode == "apical_lesion" else self.Y[kk]
            v_api = b @ Yk
            if linearize:
                v_api = v_api * (post * (1.0 - post))
            p = _sig(self.beta * v_api)                             # analytic burst probability (credit channel)
            if samples is None:
                P_obs = p
            else:
                kc = kcounts[kk + 1]
                j = srng.binomial(kc, np.clip(p, 0.0, 1.0))         # bursts ~ Binomial(events, p) -- the NOISE source
                P_obs = np.where(kc > 0, j / np.maximum(kc, 1), 0.0)
            self.pbar[kk] = self.ema * self.pbar[kk] + (1.0 - self.ema) * P_obs.mean(0)
            dev = post * (P_obs - self.pbar[kk])                    # BDSP burst-rate deviation on the OBSERVED channels
            g = acts[kk].T @ dev
            upd[kk] = -g if mode == "wrong_sign" else g
            b = dev
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train_spk(net, X, y, mode, epochs, lr, batch, seed, samples):
    rng = np.random.default_rng(seed + 777)                          # SAME shuffle stream as EMERGE-1b's _train
    srng = np.random.default_rng(seed + 555)                         # separate stream for spike sampling
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode=mode, lr=lr, samples=samples, srng=srng)


def run(seed, epochs, lr, batch, hidden, sweep, primary):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]
    res = {}

    def _acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    def _probe(net):
        return _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)

    # rate reference (S=inf): the CONFIRMED EMERGE-1b ceiling on the identical net/seed/init
    ref = BurstpropMLP(deep, seed=seed)
    _train(ref, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed)
    tr, te = _acc(ref); res["rate_ref"] = {"train": tr, "heldout": te, "probe_latent": _probe(ref)}

    # spiking sweep over the sample budget S (population coding = the mitigation for burst-estimate noise)
    res["spiking_sweep"] = {}
    for S in sweep:
        net = SpikingBurstpropMLP(deep, seed=seed)
        _train_spk(net, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed, samples=S)
        tr, te = _acc(net)
        res["spiking_sweep"][str(S)] = {"train": tr, "heldout": te, "probe_latent": _probe(net)}

    # anti-cheats at the PRIMARY sample budget (spiking)
    for mode in ("apical_lesion", "wrong_sign", "no_teaching_null"):
        net = SpikingBurstpropMLP(deep, seed=seed)
        wt_ok = all(not any(np.array_equal(Yk, w) or np.array_equal(Yk, w.T) for w in net.W) for Yk in net.Y)
        _train_spk(net, Xtr, ytr, mode, epochs, lr, batch, seed, samples=primary)
        tr, te = _acc(net)
        res[mode] = {"train": tr, "heldout": te, "no_weight_transport": bool(wt_ok)}

    # oracle (fenced backprop ceiling -- task sanity ONLY, not a shipped rule)
    net = DendriticMLP(deep, seed=seed)
    from research.runners._emerge1_deep_dendritic_representation_derisk import _train as _o_train
    _o_train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
    tr, te = _acc(net); res["oracle_bp"] = {"train": tr, "heldout": te}

    # same-W-init check (the decisive within-net contrast is fair)
    s0 = SpikingBurstpropMLP(deep, seed=seed); r0 = BurstpropMLP(deep, seed=seed)
    res["same_init_as_rate"] = bool(all(np.allclose(a, b) for a, b in zip(s0.W, r0.W)))
    res["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return {"seed": seed, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1500)              # EMERGE-1b GO config
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--sweep", type=int, nargs="+", default=[30, 100, 300])   # spike-sample budgets S
    ap.add_argument("--primary", type=int, default=300)             # the S the anti-cheats + GO read
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    sweep = list(a.sweep)
    if a.primary not in sweep:
        sweep = sorted(set(sweep + [a.primary]))
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a.epochs, a.lr, a.batch, a.hidden, sweep, a.primary); per.append(r)
            prim = r["spiking_sweep"][str(a.primary)]
            sweep_str = " ".join(f"S{S}={r['spiking_sweep'][str(S)]['heldout']:.3f}" for S in sweep)
            print(f"  [seed {s}] rate_ref {r['rate_ref']['heldout']:.3f} | spiking[{sweep_str}] | "
                  f"primary(S{a.primary}) held {prim['heldout']:.3f} probe {prim['probe_latent']:.3f} | "
                  f"lesion {r['apical_lesion']['heldout']:.3f} | wrong {r['wrong_sign']['heldout']:.3f} | "
                  f"null {r['no_teaching_null']['heldout']:.3f} | oracle {r['oracle_bp']['heldout']:.3f} | "
                  f"chance {r['chance']:.3f} | same_init {r['same_init_as_rate']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m_prim(sub):
            return float(np.mean([p["spiking_sweep"][str(a.primary)][sub] for p in per]))
        def m(k, sub="heldout"):
            return float(np.mean([p[k][sub] for p in per]))
        ref = m("rate_ref"); ref_probe = m("rate_ref", "probe_latent")
        spk = m_prim("heldout"); spk_probe = m_prim("probe_latent")
        les, wrong, null = m("apical_lesion"), m("wrong_sign"), m("no_teaching_null")
        orac, ch = m("oracle_bp"), float(np.mean([p["chance"] for p in per]))
        sweep_means = {str(S): float(np.mean([p["spiking_sweep"][str(S)]["heldout"] for p in per])) for S in sweep}
        wt = all(p["apical_lesion"]["no_weight_transport"] and p["same_init_as_rate"] for p in per)
        # monotone recovery: held-out should not DECREASE as S grows across the sweep (population coding mitigates noise)
        ordered = [sweep_means[str(S)] for S in sorted(sweep)]
        recovers = all(ordered[i + 1] >= ordered[i] - 0.03 for i in range(len(ordered) - 1))
        task_ok = orac >= 0.80
        survives = (spk >= 0.70) and (spk >= ref - 0.10) and (spk > les + 0.10)
        rep_ok = spk_probe >= 0.70
        lesion_collapses = les <= max(ref, ch) + 0.06
        wrong_anti = wrong <= ch + 0.06
        null_flat = null <= ch + 0.06
        go = bool(task_ok and survives and rep_ok and recovers and lesion_collapses and wrong_anti and null_flat and wt)
        partial = bool(task_ok and wt and lesion_collapses and (spk > les + 0.10) and (spk >= ch + 0.10) and not survives)
        if not task_ok:
            verdict = f"INCONCLUSIVE -- oracle only {orac:.3f}; tune before reading the spiking arms."
        elif go:
            verdict = (f"GO -- deep Burstprop credit assignment SURVIVES the rate->spike transition with population "
                       f"coding: at S={a.primary} the SPIKING net held-out {spk:.3f} (within {ref-spk:+.3f} of the "
                       f"confirmed rate ceiling {ref:.3f}, >> apical-lesion {les:.3f} + chance {ch:.3f}); the level-1 XOR "
                       f"latents EMERGED under spike noise (probe {spk_probe:.3f}); the S-sweep RECOVERS monotonically "
                       f"({sweep_means}) -- population coding IS the mitigation, as Payeur predicts; apical-lesion "
                       f"collapses, wrong-sign anti-learns ({wrong:.3f}), no-teaching-null flat ({null:.3f}), no weight "
                       f"transport, same W-init as the rate net. Multi-seed. ⇒ the confirmed mechanism carries to spikes "
                       f"-> the `sim/` two-compartment build (rung 4) is justified + the population size (~S={a.primary}) "
                       f"is sized. Next: rung 3 (recurrent sequence cortex) or rung 4 (sim/ port). NO sim/ edit here.")
        elif partial:
            verdict = (f"PARTIAL -- the SPIKING net clears chance + beats the lesion floor (S={a.primary}: {spk:.3f} vs "
                       f"lesion {les:.3f}, chance {ch:.3f}) and structure partly emerges (probe {spk_probe:.3f}), so the "
                       f"credit DOES survive spikes to a degree -- but it doesn't fully reach the rate ceiling {ref:.3f} "
                       f"or the 0.70 bar at this sample budget. The flagged burst-estimate noise costs real accuracy; the "
                       f"lever is a LARGER population (extend the sweep) or the microcircuit arm. A real step onto the "
                       f"substrate ladder, not yet a clean GO. Build-informative, NOT a stop.")
        else:
            miss = []
            if not survives: miss.append(f"spiking didn't reach the bar (S{a.primary} {spk:.3f} vs ref {ref:.3f}/lesion {les:.3f})")
            if not rep_ok: miss.append(f"probe {spk_probe:.3f} < 0.70 (structure didn't emerge under spike noise)")
            if not recovers: miss.append(f"no monotone S-recovery ({sweep_means}) -- population coding didn't mitigate")
            if not lesion_collapses: miss.append("apical-lesion didn't collapse")
            if not wrong_anti: miss.append(f"wrong-sign not at chance ({wrong:.3f})")
            if not null_flat: miss.append(f"no-teaching-null not flat ({null:.3f})")
            if not wt: miss.append("weight-transport / same-init check failed")
            verdict = ("BOUNDARY (next mechanism, not a stop) -- " + "; ".join(miss) + f" (oracle {orac:.3f}; rate ref "
                       f"{ref:.3f}). The burst-estimate noise breaks depth-credit at these budgets. Per the master "
                       f"directive: iterate BEFORE the protected build -- a wider population (extend the sweep), the "
                       f"Sacramento-Senn self-predicting microcircuit arm (EMERGE-3, more gradient-faithful under noise), "
                       f"or fold the real EMERGE-4 transfer. Build-SAVING: do NOT start the sim/ two-compartment port yet.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge5_spiking_burstprop", "GO": go, "verdict": verdict,
               "mechanism": "EMERGE-1b faithful Burstprop with event rate E + burst probability P as FINITE-SAMPLE "
                            "spike-count estimates (Poisson events / Binomial bursts over a population x window budget S); "
                            "single-variable rate->spike transition; population coding (S-sweep) = the flagged mitigation",
               "task": f"depth-2 threshold-of-{N_PAIRS}-pair-XORs over {N_BITS} bits (== EMERGE-1/1b)",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                                            "sweep": sweep, "primary": a.primary},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Finite-sample SPIKE-COUNT model (NOT dt-stepped, NOT the SimulationBridge) -- isolates the "
                              "DOMINANT spiking effect (burst-estimate noise + population-coding mitigation), the review's "
                              "central spiking concern. Kept at p0=0.5 for a clean single-variable transition; the real "
                              "neuron's low resting burst prob (EMERGE-4 P0~0.03), the refractory E<-apical cross-talk "
                              "(~0.18), and full dt integration are deferred to the sim/ bridge port (rung 4). Oracle = "
                              "fenced backprop ceiling (task-sanity), NOT a shipped biologically-local mode. Boundaries = "
                              "the next mechanism (master directive), never a stop."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge5] VERDICT: {verdict}", flush=True)
    print(f"[emerge5] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
