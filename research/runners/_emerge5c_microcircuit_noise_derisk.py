"""EMERGE-5c DE-RISK: under the SAME finite-sample spike noise, does the Sacramento-Senn microcircuit's ACTIVE
CANCELLATION build a cleaner representation than Burstprop's raw burst-rate estimation? -- the decided rung-2 lever.

EMERGE-5b (credit-vs-readout) established that at the realistic p0=0.03, Burstprop's rate->spike credit builds a
representation that is real but DEGRADED: a clean readout on its frozen hidden rep recovers only ~0.622 (vs the rate
ceiling ~0.796 / the random-features floor ~0.49). Credit QUALITY is the higher-leverage limit. Per Urbanczik-Senn
(the population mechanism is a population-FEEDBACK factor, not naive averaging) + the standing ladder, the decided next
mechanism is the microcircuit's ACTIVE interneuron cancellation -- structurally different from raw burst-rate
estimation, and possibly more robust to finite-sample noise (or it may hit the same wall -- both are build-informative).

THE HEAD-TO-HEAD (all at the healthy width-384 config, same depth-2 task/seeds, same S=300 finite-sample budget):
  - burstprop_spiking   : EMERGE-5's SpikingBurstpropMLP (p0=0.03)         -> own acc + CLEAN-readout-on-rep (~0.62 ref)
  - microcircuit_spiking: the NEW SpikingMicrocircuitMLP (this file)         -> own acc + CLEAN-readout-on-rep (THE TEST)
  - microcircuit_rate   : EMERGE-3's MicrocircuitMLP (noise-free)            -> own acc (the microcircuit's rate ceiling)
  - burstprop_rate      : EMERGE-1b's BurstpropMLP (noise-free)              -> own acc (Burstprop's rate ceiling)
  - untrained           : random net                                        -> CLEAN-readout (random-features FLOOR)
  - mc_spiking_lesion    : microcircuit_spiking with the apical feedback killed (W_PP_td=0) -> must collapse
  - mc_spiking_null      : microcircuit_spiking with no output teaching       -> must stay flat (self-cancel moat)

THE NOISE MODEL (identical injection point to EMERGE-5): the credit-carrying FIRING RATES phi(u^P) are replaced by
finite-sample spike-count estimates r_obs = Binomial(S, phi(u^P))/S -- the microcircuit's apical error v_A = W_PP_td @
(rate differences) and its phi'(r) modulation then ride on those noisy rates, exactly as Burstprop's burst-fraction
credit rides on noisy spike counts. The OUTPUT error stays target-exact (clean logits, direct target access -- as in
BOTH the rate microcircuit AND EMERGE-5's Burstprop; the noise is in the HIDDEN credit channel, the fair comparison).
Eval (_forward/accuracy) is the clean analytic forward, so the clean-readout probe reads the noise-free representation.

VERDICT: GO-microcircuit-more-robust = mc_spiking clean-rep > burstprop_spiking clean-rep + 0.08 AND approaches the
microcircuit rate ceiling; the apical-feedback lesion collapses it; the no-teaching null stays flat. BOUNDARY = mc
clean-rep ~= burstprop clean-rep (active cancellation does NOT beat the finite-sample wall) -> the noise limit is
MECHANISM-GENERAL, pointing the next research-gated move to a population-feedback factor / NMNC-style noise-geometry
(from the 2025-26 shortlist), not another local credit rule. Reuse-by-import; NO `sim/` edit; CPU; arm-parallel.
Run: SIM_BACKEND=numpy python -m research.runners._emerge5c_microcircuit_noise_derisk --seeds 42 43 44
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
from research.runners._emerge1b_burstprop_derisk import BurstpropMLP, _train as _train_rate_bp  # noqa: E402
from research.runners._emerge3_microcircuit_derisk import (  # noqa: E402
    MicrocircuitMLP, _train as _train_rate_mc, _softmax, _MOMENTUM)
from research.runners._emerge5_spiking_burstprop_derisk import SpikingBurstpropMLP, _train_spk  # noqa: E402
from research.runners._emerge5b_credit_vs_readout_derisk import _clean_readout_acc, _hidden  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge5c_microcircuit_noise.json"


class SpikingMicrocircuitMLP(MicrocircuitMLP):
    """EMERGE-3's Sacramento-Senn microcircuit with the credit-carrying FIRING RATES phi(u^P) replaced by
    finite-sample spike-count estimates Binomial(S, rate)/S -- the SAME noise model EMERGE-5 injects into Burstprop.
    The apical error v_A = W_PP_td @ e and its phi'(r) modulation ride on the noisy rates; the output error stays
    target-exact (clean logits). Interneuron maintenance is dropped (the credit is read in the self-predicting form,
    as in EMERGE-3, so it does not affect the within-step FF update). Eval forward is the inherited clean analytic one."""

    def train_step(self, X, y, mode, lr, samples=None, srng=None):
        acts, lg = self._forward(X); y = np.asarray(y)
        m = max(1, X.shape[0]); nW = len(self.W); nhid = nW - 1
        # spike-estimate the credit-carrying firing rates (input untouched; hidden + last-hidden estimated from S spikes)
        if samples is None:
            r = acts
        else:
            r = [acts[0]] + [srng.binomial(samples, np.clip(a, 0.0, 1.0)) / float(samples) for a in acts[1:]]
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0   # exact output error (target access)
        if mode == "no_teaching_null":
            delta_out = np.zeros_like(delta_out)
        elif mode == "wrong_sign":
            delta_out = -delta_out
        upd = [None] * nW
        upd[-1] = -(r[-1].T @ delta_out)
        # top->bottom apical-error recursion (M2.11), on the noisy rates
        v_A = [None] * nhid
        e_upper = -delta_out
        for k in range(nhid - 1, -1, -1):
            r_post = r[k + 1]
            Wtd = np.zeros_like(self.W_PP_td[k]) if mode == "feedback_lesion" else self.W_PP_td[k]
            v_A_k = e_upper @ Wtd.T
            v_A[k] = v_A_k
            e_upper = (r_post * (1.0 - r_post)) * v_A_k
        # feedforward somatic-error rule (M2.6), on the noisy rates
        for k in range(nhid):
            r_prev = r[k]; r_post = r[k + 1]; phi_prime = r_post * (1.0 - r_post)
            soma_err = (self.g_A / self._som_den) * v_A[k] * phi_prime
            upd[k] = r_prev.T @ soma_err
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train_spk_mc(net, X, y, mode, epochs, lr, batch, seed, samples):
    rng = np.random.default_rng(seed + 777); srng = np.random.default_rng(seed + 555)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode=mode, lr=lr, samples=samples, srng=srng)


def _run_arm(job):
    seed, arm, epochs, lr, batch, hidden, samples, p0 = job
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]

    def own(net):
        return float(net.accuracy(Xte, yte))

    def clean(net):
        return _clean_readout_acc(_hidden(net, Xtr), ytr, _hidden(net, Xte), yte, seed=seed)

    if arm == "burstprop_spiking":
        net = SpikingBurstpropMLP(deep, seed=seed, p0=p0)
        _train_spk(net, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed, samples=samples)
        return (seed, arm, {"own": own(net), "clean_readout": clean(net)})
    if arm in ("microcircuit_spiking", "mc_spiking_lesion", "mc_spiking_null"):
        md = {"microcircuit_spiking": "microcircuit", "mc_spiking_lesion": "feedback_lesion",
              "mc_spiking_null": "no_teaching_null"}[arm]
        net = SpikingMicrocircuitMLP(deep, seed=seed)
        _train_spk_mc(net, Xtr, ytr, md, epochs, lr, batch, seed, samples=samples)
        return (seed, arm, {"own": own(net), "clean_readout": clean(net)})
    if arm == "microcircuit_rate":
        net = MicrocircuitMLP(deep, seed=seed)
        _train_rate_mc(net, Xtr, ytr, "microcircuit", epochs, lr, batch, seed)
        return (seed, arm, {"own": own(net), "clean_readout": clean(net)})
    if arm == "burstprop_rate":
        net = BurstpropMLP(deep, seed=seed)
        _train_rate_bp(net, Xtr, ytr, "burst_linearized", epochs, lr, batch, seed)
        return (seed, arm, {"own": own(net), "clean_readout": clean(net)})
    if arm == "untrained":
        net = BurstpropMLP(deep, seed=seed)
        return (seed, arm, {"own": own(net), "clean_readout": clean(net)})
    raise ValueError(f"unknown arm {arm}")


ARMS = ["burstprop_spiking", "microcircuit_spiking", "mc_spiking_lesion", "mc_spiking_null",
        "microcircuit_rate", "burstprop_rate", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=384)              # healthy width (oracle ~1.0)
    ap.add_argument("--samples", type=int, default=300)             # EMERGE-5 primary S
    ap.add_argument("--p0", type=float, default=0.03)               # EMERGE-4 resting burst prob (Burstprop arm only)
    ap.add_argument("--max-workers", type=int, default=0,           # 0 = all cores; set small to leave CPU for other apps
                    help="cap parallel workers (each is 1-thread-BLAS = ~1 core); 0 = os.cpu_count(). Use e.g. 4 to "
                         "leave most of the CPU free (light contention) while something else runs.")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        jobs = [(s, arm, a.epochs, a.lr, a.batch, a.hidden, a.samples, a.p0) for s in a.seeds for arm in ARMS]
        _cap = a.max_workers if (a.max_workers and a.max_workers > 0) else (os.cpu_count() or 1)
        collected = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), _cap)) as ex:
                for seed, arm, entry in ex.map(_run_arm, jobs):
                    collected.setdefault(seed, {})[arm] = entry
        except Exception:
            for job in jobs:
                seed, arm, entry = _run_arm(job)
                collected.setdefault(seed, {})[arm] = entry
        for s in a.seeds:
            d = collected[s]; d["seed"] = s
            (_, _, _), (_, yte, _) = make_task(s)
            d["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
            per.append(d)
        for d in per:
            print(f"  [seed {d['seed']}] MC-spk own {d['microcircuit_spiking']['own']:.3f} clean "
                  f"{d['microcircuit_spiking']['clean_readout']:.3f} | BP-spk own {d['burstprop_spiking']['own']:.3f} "
                  f"clean {d['burstprop_spiking']['clean_readout']:.3f} | MC-rate {d['microcircuit_rate']['own']:.3f} | "
                  f"BP-rate {d['burstprop_rate']['own']:.3f} | floor {d['untrained']['clean_readout']:.3f} | "
                  f"MC-lesion clean {d['mc_spiking_lesion']['clean_readout']:.3f} | MC-null clean "
                  f"{d['mc_spiking_null']['clean_readout']:.3f} | chance {d['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="clean_readout"):
            return float(np.mean([p[arm][key] for p in per]))
        mc_clean, bp_clean = m("microcircuit_spiking"), m("burstprop_spiking")
        mc_own, mc_rate = m("microcircuit_spiking", "own"), m("microcircuit_rate", "own")
        bp_rate, floor = m("burstprop_rate", "own"), m("untrained")
        les, null = m("mc_spiking_lesion"), m("mc_spiking_null")
        ch = float(np.mean([p["chance"] for p in per]))
        # sanity: the rate microcircuit must be a real ceiling (learns the task), else my reuse is broken
        mc_rate_sane = mc_rate >= 0.70
        lesion_collapses = les <= floor + 0.08
        null_flat = null <= floor + 0.08
        beats_burstprop = mc_clean > bp_clean + 0.08
        approaches_ceiling = mc_clean >= mc_rate - 0.12
        go = bool(mc_rate_sane and beats_burstprop and approaches_ceiling and lesion_collapses and null_flat)
        same_wall = bool(mc_rate_sane and abs(mc_clean - bp_clean) <= 0.06)
        if not mc_rate_sane:
            verdict = (f"INCONCLUSIVE -- the rate microcircuit ceiling is only {mc_rate:.3f} (<0.70), so the "
                       f"microcircuit reuse/config isn't learning the task cleanly; fix before comparing the spiking "
                       f"reps. (BP rate ceiling {bp_rate:.3f} for reference.)")
        elif go:
            verdict = (f"GO -- ACTIVE CANCELLATION is MORE noise-robust than raw burst-rate estimation: under the same "
                       f"finite-sample noise (S={a.samples}), the spiking MICROCIRCUIT builds a cleaner representation "
                       f"(clean-readout {mc_clean:.3f}) than spiking Burstprop ({bp_clean:.3f}), approaching its own rate "
                       f"ceiling ({mc_rate:.3f}); apical-feedback lesion collapses it ({les:.3f} ~ floor {floor:.3f}), "
                       f"no-teaching null flat ({null:.3f}). ⇒ the microcircuit is the better rung-2 credit rule under "
                       f"spike noise -- carry IT (not Burstprop) toward the sim/ two-compartment port. NO sim/ edit.")
        elif same_wall:
            verdict = (f"BOUNDARY (mechanism-general noise wall) -- active cancellation does NOT beat raw burst-rate "
                       f"estimation under finite-sample spike noise: spiking microcircuit clean-rep {mc_clean:.3f} ~= "
                       f"spiking Burstprop {bp_clean:.3f} (both << the microcircuit rate ceiling {mc_rate:.3f}, > floor "
                       f"{floor:.3f}). ⇒ the finite-sample credit-noise limit is MECHANISM-GENERAL (not specific to "
                       f"Burstprop's burst channel). Per the master directive the next research-gated move is a "
                       f"population-FEEDBACK factor (Urbanczik-Senn 2009) or NMNC-style credit-noise GEOMETRY "
                       f"(shortlist), NOT another local point-credit rule. Build-informative. Lesion collapses "
                       f"{les:.3f}, null {null:.3f}.")
        else:
            verdict = (f"PARTIAL/MIXED -- spiking microcircuit clean-rep {mc_clean:.3f} vs spiking Burstprop {bp_clean:.3f} "
                       f"(rate ceiling {mc_rate:.3f}, floor {floor:.3f}, lesion {les:.3f}, null {null:.3f}). The "
                       f"microcircuit "
                       f"{'edges out Burstprop but not by the +0.08 bar / does not approach its ceiling' if mc_clean > bp_clean else 'does not beat Burstprop'}"
                       f"; anti-cheats: lesion-collapse={lesion_collapses}, null-flat={null_flat}. A real comparison "
                       f"data point, not a clean GO -- iterate (wider net / a full multi-step microcircuit relaxation / "
                       f"the population-feedback factor). NOT a stop.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge5c_microcircuit_noise", "verdict": verdict,
               "question": "under the same finite-sample spike noise (S), does the Sacramento-Senn microcircuit's active "
                           "interneuron cancellation build a cleaner hidden representation (clean-readout on frozen rep) "
                           "than Burstprop's raw burst-rate estimation? -- decides which credit rule to carry to the sim/ port",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                                            "samples": a.samples, "p0": a.p0},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Head-to-head at the healthy width-384 config. Noise model = Binomial(S, phi(u^P))/S on "
                              "the credit-carrying firing rates (same injection point as EMERGE-5's Burstprop); output "
                              "error stays target-exact (clean logits, as in both the rate microcircuit and Burstprop). "
                              "Microcircuit credit read in the self-predicting form (as EMERGE-3; interneuron maintenance "
                              "dropped -- it does not affect the within-step FF update). RATE-limit, NOT dt-integrated. "
                              "The clean-readout probe reads the noise-free analytic rep. A BOUNDARY here (cancellation "
                              "no better than Burstprop) is build-informative: it says the finite-sample noise limit is "
                              "mechanism-general and the next move is a population-feedback factor / noise-geometry, per "
                              "the 2025-26 shortlist."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge5c] VERDICT: {verdict}", flush=True)
    print(f"[emerge5c] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
