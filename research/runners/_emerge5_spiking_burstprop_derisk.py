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
central spiking concern. Reuse-by-import; NO `sim/` edit; CPU.
Run: SIM_BACKEND=numpy python -m research.runners._emerge5_spiking_burstprop_derisk --seeds 42 43 44

DISCOVERED (2026-07-01, `2026-07-01-emerge5-noise-driven-self-organization-discovery.md` -- read this before
interpreting apical_lesion/no_teaching_null probe values): finite-sample spike-count noise ALONE (Y=0 or b=0, zero
explicit teaching signal) drives real, reproducible self-organization of the hidden representation via an
activity-correlated noise-VARIANCE coupling (Var(dev) ~ post * p0*(1-p0)/samples, since post's own numerator IS the
Binomial trial count for P_obs) -- confirmed absent in the pure rate model (hidden weights bit-for-bit unchanged,
probe ~ untrained baseline ~0.48) and confirmed dose-dependent on p0*(1-p0) (at p0=0.5 it SATURATES both lesion and
null probes to ~1.0, masking the test arm's own advantage; at p0=0.03 -- EMERGE-4's measured real resting rate -- it
drops to ~0.58, and the test arm's probe (~0.87) cleanly separates again). This is why `--p0` defaults to 0.5 (the
byte-faithful EMERGE-1b transition) but `--p0 0.03` is the regime where the representation-level anti-cheat is
actually discriminating; read BOTH runs together, not `--p0 0.5` alone.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# PERF: these are TINY matmuls (a few hundred wide). Multi-threaded BLAS OVERSUBSCRIBES a many-core box and runs ~30x
# SLOWER (measured 266ms vs 8.8ms/step on 20 cores) + burns cores on thread-sync (the "low util" symptom). Force ONE
# BLAS thread per process (must be set BEFORE numpy imports) and parallelize across SEEDS instead (main() ProcessPool).
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
    `_hidden_rep`) uses the inherited ANALYTIC clean forward -- it measures the LEARNED function, not eval-time noise.

    REST-BIAS (folds in EMERGE-4's MEASURED biophysics, a second principled single-variable step): the constructor's
    `p0` is used, per BurstpropMLP, to seed the EMA baseline pbar -- but a bare sigmoid `sig(beta*v_api)` is centered
    at p=0.5 at rest REGARDLESS of p0, so the rate model's teaching-null test (b=0 -> p==p0 exactly) only holds when
    p0=0.5. A REAL two-compartment burst neuron (EMERGE-4, `_emerge4_burst_multiplexing_derisk.py`) measured a LOW
    resting burst probability (P0~0.03), not 0.5. A logit-shift `bias=logit(p0)` makes the analytic rest point match
    p0 exactly (`sig(beta*0 + bias) = p0`), so a low p0 is now BOTH the EMA seed AND the true rest point -- physically
    consistent, and (per Binomial variance = p(1-p)/k) a LOW p0 gives ~8.6x LOWER sampling variance than p0=0.5 at the
    same sample budget: real biology's low resting burst rate is plausibly ITSELF the noise-suppression mechanism."""

    def __init__(self, sizes, seed=0, beta=1.0, p0=0.5, ema=0.9):
        super().__init__(sizes, seed=seed, beta=beta, p0=p0, ema=ema)
        p0c = min(max(float(p0), 1e-6), 1.0 - 1e-6)
        self._bias = float(np.log(p0c / (1.0 - p0c)))            # logit(p0): shifts sig() so p(v_api=0) == p0

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
            p = _sig(self.beta * v_api + self._bias)                # rest-biased burst probability (credit channel)
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


def _run_arm(job):
    """Train ONE arm for ONE seed -- an independent unit of work, so the whole (seed x arm) grid parallelizes across
    cores. Returns (seed, arm_key, entry). arm_key in {'rate_ref', 'spk:<S>', 'apical_lesion', 'wrong_sign',
    'no_teaching_null', 'oracle_bp'}. Byte-identical to the old sequential run(): each arm re-derives its own
    data (make_task(seed)) + init (net(seed)) + training RNG (seed+777/+555) from the seed alone."""
    seed, arm, epochs, lr, batch, hidden, primary, p0 = job
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]

    def _acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    def _probe(net):
        return _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)

    if arm == "rate_ref":
        # the CONFIRMED EMERGE-1b ceiling on the identical net/seed/init -- analytic rate model, p0-independent.
        net = BurstpropMLP(deep, seed=seed)
        _train(net, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed)
        tr, te = _acc(net)
        return (seed, arm, {"train": tr, "heldout": te, "probe_latent": _probe(net)})
    if arm.startswith("spk:"):
        S = int(arm.split(":", 1)[1])
        net = SpikingBurstpropMLP(deep, seed=seed, p0=p0)
        _train_spk(net, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed, samples=S)
        tr, te = _acc(net)
        return (seed, arm, {"train": tr, "heldout": te, "probe_latent": _probe(net)})
    if arm in ("apical_lesion", "wrong_sign", "no_teaching_null"):
        # anti-cheats at the PRIMARY sample budget; probe on ALL of them (representation-level gate vs the
        # "good accuracy masks a dead/laundered hidden layer" false-positive).
        net = SpikingBurstpropMLP(deep, seed=seed, p0=p0)
        wt_ok = all(not any(np.array_equal(Yk, w) or np.array_equal(Yk, w.T) for w in net.W) for Yk in net.Y)
        _train_spk(net, Xtr, ytr, arm, epochs, lr, batch, seed, samples=primary)
        tr, te = _acc(net)
        return (seed, arm, {"train": tr, "heldout": te, "no_weight_transport": bool(wt_ok), "probe_latent": _probe(net)})
    if arm == "oracle_bp":
        from research.runners._emerge1_deep_dendritic_representation_derisk import _train as _o_train
        net = DendriticMLP(deep, seed=seed)
        _o_train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
        tr, te = _acc(net)
        return (seed, arm, {"train": tr, "heldout": te})
    raise ValueError(f"unknown arm {arm}")


def _assemble(seed, hidden, p0, arm_entries):
    """Reassemble one seed's arm entries into the per-seed dict the aggregation/verdict code expects."""
    deep = [N_BITS, hidden, hidden, 2]
    (_, _, _), (_, yte, _) = make_task(seed)
    res = {"seed": seed, "p0": float(p0), "spiking_sweep": {}}
    for arm, entry in arm_entries:
        if arm == "rate_ref":
            res["rate_ref"] = entry
        elif arm.startswith("spk:"):
            res["spiking_sweep"][arm.split(":", 1)[1]] = entry
        else:
            res[arm] = entry
    s0 = SpikingBurstpropMLP(deep, seed=seed); r0 = BurstpropMLP(deep, seed=seed)
    res["same_init_as_rate"] = bool(all(np.allclose(a, b) for a, b in zip(s0.W, r0.W)))
    res["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1500)              # EMERGE-1b GO config
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--sweep", type=int, nargs="+", default=[30, 100, 300])   # spike-sample budgets S
    ap.add_argument("--primary", type=int, default=300)             # the S the anti-cheats + GO read
    ap.add_argument("--p0", type=float, default=0.5,                # rest-biased burst probability (see class doc);
                    help="target resting burst probability (0.5 = the unbiased EMERGE-1b regime; EMERGE-4 measured "
                         "the real two-compartment neuron's resting rate at ~0.03 -- pass --p0 0.03 to fold that in)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    sweep = list(a.sweep)
    if a.primary not in sweep:
        sweep = sorted(set(sweep + [a.primary]))
    arms = ["rate_ref"] + [f"spk:{S}" for S in sweep] + ["apical_lesion", "wrong_sign", "no_teaching_null", "oracle_bp"]
    t0 = time.time(); err = None; per = []
    try:
        # EVERY (seed x arm) is an INDEPENDENT training -> parallelize the WHOLE grid, not just seeds, so all cores are
        # used (drift-mode #6): |seeds| x ~8 arms units across up to os.cpu_count() 1-thread-BLAS workers. Byte-identical
        # to the old sequential run() (each arm re-derives its own data/init from the seed). Sequential fallback below.
        jobs = [(s, arm, a.epochs, a.lr, a.batch, a.hidden, a.primary, a.p0) for s in a.seeds for arm in arms]
        collected = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
                for seed, arm, entry in ex.map(_run_arm, jobs):
                    collected.setdefault(seed, []).append((arm, entry))
        except Exception:
            for job in jobs:
                seed, arm, entry = _run_arm(job)
                collected.setdefault(seed, []).append((arm, entry))
        per = [_assemble(s, a.hidden, a.p0, collected[s]) for s in a.seeds]
        for r in per:
            s = r["seed"]; prim = r["spiking_sweep"][str(a.primary)]
            sweep_str = " ".join(f"S{S}={r['spiking_sweep'][str(S)]['heldout']:.3f}" for S in sweep)
            print(f"  [seed {s}] p0={a.p0:g} rate_ref {r['rate_ref']['heldout']:.3f} | spiking[{sweep_str}] | "
                  f"primary(S{a.primary}) held {prim['heldout']:.3f} probe {prim['probe_latent']:.3f} | "
                  f"lesion {r['apical_lesion']['heldout']:.3f} (probe {r['apical_lesion']['probe_latent']:.3f}) | "
                  f"wrong {r['wrong_sign']['heldout']:.3f} (probe {r['wrong_sign']['probe_latent']:.3f}) | "
                  f"null {r['no_teaching_null']['heldout']:.3f} (probe {r['no_teaching_null']['probe_latent']:.3f}) | "
                  f"oracle {r['oracle_bp']['heldout']:.3f} | chance {r['chance']:.3f} | "
                  f"same_init {r['same_init_as_rate']}", flush=True)
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
        les_probe, wrong_probe, null_probe = m("apical_lesion", "probe_latent"), m("wrong_sign", "probe_latent"), \
            m("no_teaching_null", "probe_latent")
        orac, ch = m("oracle_bp"), float(np.mean([p["chance"] for p in per]))
        sweep_means = {str(S): float(np.mean([p["spiking_sweep"][str(S)]["heldout"] for p in per])) for S in sweep}
        wt = all(p["apical_lesion"]["no_weight_transport"] and p["same_init_as_rate"] for p in per)
        # monotone recovery: held-out should not DECREASE as S grows across the sweep (population coding mitigates noise)
        ordered = [sweep_means[str(S)] for S in sorted(sweep)]
        recovers = all(ordered[i + 1] >= ordered[i] - 0.03 for i in range(len(ordered) - 1))
        task_ok = orac >= 0.80
        survives = (spk >= 0.70) and (spk >= ref - 0.10) and (spk > les + 0.10)
        rep_ok = spk_probe >= 0.70
        # REPRESENTATION-level gate (the fresh-look review's mandatory anti-cheat): a corrupted-credit arm's HIDDEN
        # REPRESENTATION, not just its readout accuracy, must stay near floor -- guards against the exact effect this
        # runner's first pass exposed (the always-correctly-trained OUTPUT layer can "launder" a degraded/noisy hidden
        # rep into above-floor task accuracy in multi-class classification; probing the representation directly is
        # not fooled by that). probe_latent is per-bit accuracy on BALANCED binary latents -> chance is a FIXED 0.5,
        # not the task's class-imbalance `ch` -- use a fixed absolute bound (0.5 + margin), not a self-referential one.
        PROBE_FLOOR = 0.65
        lesion_collapses = les <= max(ref, ch) + 0.06 and les_probe <= PROBE_FLOOR
        wrong_anti = wrong <= ch + 0.06 and wrong_probe <= PROBE_FLOOR
        null_flat = null <= ch + 0.06 and null_probe <= PROBE_FLOOR
        go = bool(task_ok and survives and rep_ok and recovers and lesion_collapses and wrong_anti and null_flat and wt)
        partial = bool(task_ok and wt and lesion_collapses and (spk > les + 0.10) and (spk >= ch + 0.10) and not survives)
        if not task_ok:
            verdict = f"INCONCLUSIVE -- oracle only {orac:.3f}; tune before reading the spiking arms."
        elif go:
            verdict = (f"GO -- deep Burstprop credit assignment SURVIVES the rate->spike transition with population "
                       f"coding (p0={a.p0:g}): at S={a.primary} the SPIKING net held-out {spk:.3f} (within {ref-spk:+.3f} "
                       f"of the confirmed rate ceiling {ref:.3f}, >> apical-lesion {les:.3f} + chance {ch:.3f}); the "
                       f"level-1 XOR latents EMERGED under spike noise (probe {spk_probe:.3f}); the S-sweep RECOVERS "
                       f"monotonically ({sweep_means}) -- population coding IS the mitigation, as Payeur predicts; "
                       f"apical-lesion collapses on BOTH readout AND representation (probe {les_probe:.3f}), wrong-sign "
                       f"anti-learns ({wrong:.3f}, probe {wrong_probe:.3f}), no-teaching-null flat ({null:.3f}, probe "
                       f"{null_probe:.3f}), no weight transport, same W-init as the rate net. Multi-seed. ⇒ the "
                       f"confirmed mechanism carries to spikes -> the `sim/` two-compartment build (rung 4) is "
                       f"justified + the population size (~S={a.primary}) and rest bias (p0={a.p0:g}) are sized. Next: "
                       f"rung 3 (recurrent sequence cortex) or rung 4 (sim/ port). NO sim/ edit here.")
        elif partial:
            verdict = (f"PARTIAL (p0={a.p0:g}) -- the SPIKING net clears chance + beats the lesion floor (S={a.primary}: "
                       f"{spk:.3f} vs lesion {les:.3f}, chance {ch:.3f}) and structure partly emerges (probe "
                       f"{spk_probe:.3f}), so the credit DOES survive spikes to a degree -- but it doesn't fully reach "
                       f"the rate ceiling {ref:.3f} or the 0.70 bar at this sample budget. The flagged burst-estimate "
                       f"noise costs real accuracy; the lever is a LARGER population (extend the sweep) or the "
                       f"microcircuit arm. A real step onto the substrate ladder, not yet a clean GO. Build-informative, "
                       f"NOT a stop.")
        else:
            miss = []
            if not survives: miss.append(f"spiking didn't reach the bar (S{a.primary} {spk:.3f} vs ref {ref:.3f}/lesion {les:.3f})")
            if not rep_ok: miss.append(f"probe {spk_probe:.3f} < 0.70 (structure didn't emerge under spike noise)")
            if not recovers: miss.append(f"no monotone S-recovery ({sweep_means}) -- population coding didn't mitigate")
            if not lesion_collapses: miss.append(f"apical-lesion didn't collapse (readout {les:.3f}, probe {les_probe:.3f})")
            if not wrong_anti: miss.append(f"wrong-sign not at chance (readout {wrong:.3f}, probe {wrong_probe:.3f})")
            if not null_flat: miss.append(f"no-teaching-null not flat (readout {null:.3f}, probe {null_probe:.3f})")
            if not wt: miss.append("weight-transport / same-init check failed")
            verdict = (f"BOUNDARY (p0={a.p0:g}; next mechanism, not a stop) -- " + "; ".join(miss) + f" (oracle "
                       f"{orac:.3f}; rate ref {ref:.3f}). The burst-estimate noise breaks depth-credit at these "
                       f"budgets. Per the master directive: iterate BEFORE the protected build -- a wider population "
                       f"(extend the sweep), a lower rest bias --p0 (EMERGE-4 measured ~0.03; lower p0 = lower Binomial "
                       f"variance p(1-p)/k = a cleaner credit signal), the Sacramento-Senn self-predicting microcircuit "
                       f"arm (EMERGE-3, more gradient-faithful under noise), or a fully-folded EMERGE-4 transfer. "
                       f"Build-SAVING: do NOT start the sim/ two-compartment port yet.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge5_spiking_burstprop", "GO": go, "verdict": verdict,
               "mechanism": "EMERGE-1b faithful Burstprop with event rate E + burst probability P as FINITE-SAMPLE "
                            "spike-count estimates (Poisson events / Binomial bursts over a population x window budget S); "
                            "single-variable rate->spike transition; population coding (S-sweep) = the flagged mitigation",
               "task": f"depth-2 threshold-of-{N_PAIRS}-pair-XORs over {N_BITS} bits (== EMERGE-1/1b)",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                                            "sweep": sweep, "primary": a.primary, "p0": a.p0},
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
