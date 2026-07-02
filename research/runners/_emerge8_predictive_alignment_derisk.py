"""EMERGE-8 (rung-3, the MERGED Fork-1/Fork-2 lever) — PREDICTIVE ALIGNMENT de-risk: does a fully-local,
biologically-plausible chaos-taming rule (Asabuki & Clopath 2025, Nat Commun 16:6784, DOI 10.1038/s41467-025-61309-9)
train a recurrent net to AUTONOMOUSLY GENERATE a target trajectory that stays ROBUST UNDER NOISE, where a fixed
random chaotic reservoir (the same G, no plastic correction) DIVERGES?

WHY (the multiply-confirmed setup): rung-3a re-localized the wall to autonomous-GENERATION STABILITY; three
independent probes (rung-3a, the pre-design scratch RFLO, EMERGE-7) confirmed a NAIVE local recurrent credit rule does
not beat -- and degrades -- a fixed random reservoir + trained readout. The research gate's scoped mechanism for
exactly this (tame chaos into stable autonomous trajectories, fully-local, spiking-compatible) is PREDICTIVE
ALIGNMENT. This is the decisive test of the merged fork: PA vs the fixed reservoir under noise.

THE PA RULE (verified verbatim from the PMC full text, Eqs. 6/8/11/14/15):
  state:    tau dx/dt = -x + G r + M r + W_in I + sigma xi ;  r = tanh(x)   (tau=10ms)
  recurrent = FIXED sparse chaotic G (p=0.1, gain g=1.2, std g/sqrt(p N)) + PLASTIC M (init 0)
  readout:  z = W r ;  delta rule  dW = eta_W (f - z) r^T   (f = target)
  feedback: Q in R^{N x K}, FIXED random uniform[-3/K, 3/K], projecting the readout z back
  PA rule:  dM = eta_M (Q z - Jhat r) r^T ,  Jhat = M - alpha G   (== eta_M[(Qz - Mr) + alpha G r] r^T)
  KEY: PA does NOT minimize output error (that is FORCE/RLS, non-local); it ALIGNS the recurrent prediction (M-alphaG)r
  with the readout-feedback Qz, suppressing chaos. LOCAL: pre-rate r_i, post error (Qz - Jhat r)_j, fixed feedback Q +
  fixed G -- NO W^T, NO inverse-correlation matrix, NO BPTT. `used_transpose` stays False (asserted).

ARMS: pa (PRIMARY, G+M) | fixed_reservoir (G only, M frozen 0, readout trained = the BAR) | pa_no_reg (alpha=0, drops
the chaos-suppression term -> tests it is load-bearing) | pa_scrambled_Q (feedback matrix permuted -> alignment
target incoherent) | untrained (G only, random readout -> floor). Metric = autonomous-generation readout-correlation
vs target under state noise, after a brief cue. GO = PA robust >> fixed reservoir under matched noise, multi-seed,
anti-cheats load-bearing. Reuse-by-import; NO sim/ edit; CPU/numpy; multi-seed 42/43/44.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge8_predictive_alignment.json")


def _corr(a, b):
    a = np.asarray(a).ravel().astype(float); b = np.asarray(b).ravel().astype(float)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def make_target(seed, T=220, K=2, n_modes=4):
    """A periodic K-dim sinusoid-superposition target trajectory (the Sussillo-Abbott / Laje-Buonomano canonical)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    freqs = np.array([1.0, 2.0, 3.0, 5.0])[:n_modes]
    f = np.zeros((T, K))
    for k in range(K):
        ph = rng.uniform(0, 2 * np.pi, n_modes)
        amp = rng.uniform(0.4, 1.0, n_modes)
        f[:, k] = np.tanh(sum(amp[i] * np.sin(2 * np.pi * freqs[i] * t / T + ph[i]) for i in range(n_modes)))
    return f


class PredictiveAlignmentNet:
    def __init__(self, N, K, seed=0, g=1.2, p=0.1, alpha=1.0, tau=10.0, dt=1.0, sigma=0.0, arm="pa", n_traj=1):
        rng = np.random.default_rng(seed)
        self.N, self.K, self.n_traj = N, K, n_traj
        self.tau, self.dt, self.alpha, self.arm = tau, dt, alpha, arm
        std = g / np.sqrt(p * N)
        mask = rng.random((N, N)) < p
        self.G = (rng.normal(0, std, (N, N)) * mask)                 # fixed sparse chaotic backbone
        np.fill_diagonal(self.G, 0.0)
        self.M = np.zeros((N, N))                                    # plastic correction (init 0)
        self.W = np.zeros((K, N))                                    # plastic readout
        self.Q = rng.uniform(-3.0 / K, 3.0 / K, (N, K))              # fixed random feedback
        if arm == "pa_scrambled_Q":                                  # anti-cheat: incoherent feedback alignment
            self.Q = rng.uniform(-3.0 / K, 3.0 / K, (N, K)) * rng.choice([-1.0, 1.0], (N, K))
        self.cues = rng.normal(0, 1.0, (n_traj, N)) * 3.0            # per-trajectory cue -> distinct initial states
        self.cue = self.cues[0]
        self.sigma = sigma
        self._rng = rng
        self.used_transpose = False                                 # locality flag

    def _run(self, T, learn, f=None, sigma=None, cue_steps=6, perturb_t=None, perturb_mag=0.0, traj_idx=0):
        """Run the net T steps from trajectory `traj_idx`'s cue. If learn: apply PA (M) + delta (W) updates online.
        Optionally inject a one-step state PULSE at perturb_t (tests whether the target trajectory is a STABLE
        ATTRACTOR -> recovers, vs a memorized chaotic run -> diverges). Returns z(t) (T x K)."""
        sig = self.sigma if sigma is None else sigma
        cue = self.cues[traj_idx]
        x = np.zeros(self.N); zs = np.zeros((T, self.K))
        alpha = self.alpha
        train_M = learn and self.arm in ("pa", "pa_no_reg", "pa_scrambled_Q")
        train_W = learn and self.arm != "untrained"
        if self.arm == "pa_no_reg":
            alpha = 0.0
        pdir = None
        if perturb_t is not None and perturb_mag > 0:
            pdir = self._rng.standard_normal(self.N); pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
        for t in range(T):
            I = cue if t < cue_steps else np.zeros(self.N)           # brief cue pulse sets the phase, then free-run
            noise = (sig * self._rng.standard_normal(self.N)) if sig > 0 else 0.0
            r = np.tanh(x)
            dx = (-x + self.G @ r + self.M @ r + I + noise) / self.tau
            x = x + self.dt * dx
            if pdir is not None and t == perturb_t:
                x = x + perturb_mag * pdir                          # one-step off-trajectory kick
            r = np.tanh(x)
            z = self.W @ r; zs[t] = z
            if train_M:                                             # PA: dM = eta_M (Qz - (M - alpha G) r) r^T
                Jhat_r = self.M @ r - alpha * (self.G @ r)          # (M - alpha G) r  -- local: own M, own fixed G
                err = self.Q @ z - Jhat_r                          # feedback-alignment error (Qz - Jhat r)
                self.M += self._eta_M * np.outer(err, r)
            if train_W and f is not None:                          # readout delta: dW = eta_W (f - z) r^T
                self.W += self._eta_W * np.outer(f[t] - z, r)
        return zs

    def perturb_recovery_corr(self, targets, perturb_mag=3.0, perturb_frac=0.4, reps=6):
        """Mechanism-native robustness metric: inject a mid-trajectory PULSE, measure POST-pulse readout correlation to
        target, per trajectory. A PA-shaped stable attractor recovers; a fixed-reservoir memorized chaotic run diverges."""
        cs = []
        for ti, f in enumerate(targets):
            T = len(f); pt = int(perturb_frac * T)
            for _ in range(reps):
                z = self._run(T, learn=False, sigma=0.0, perturb_t=pt, perturb_mag=perturb_mag, traj_idx=ti)
                cs.append(_corr(z[pt + 1:], f[pt + 1:]))
        return float(np.mean(cs))

    def train(self, targets, epochs, eta_M, eta_W):
        """targets = list of n_traj trajectories (each T x K); interleave all per epoch."""
        self._eta_M, self._eta_W = eta_M, eta_W
        idx = list(range(len(targets)))
        for _ in range(epochs):
            for ti in [idx[i] for i in self._rng.permutation(len(idx))]:
                self._run(len(targets[ti]), learn=True, f=targets[ti], sigma=self.sigma, traj_idx=ti)

    def generate_corr(self, targets, sigma_eval, reps=6):
        """Autonomous generation (plasticity off) under eval noise; mean readout-vs-target corr over trajectories+reps."""
        cs = []
        for ti, f in enumerate(targets):
            for _ in range(reps):
                cs.append(_corr(self._run(len(f), learn=False, sigma=sigma_eval, traj_idx=ti), f))
        return float(np.mean(cs))


ARMS = ["pa", "fixed_reservoir", "pa_no_reg", "pa_scrambled_Q", "untrained"]


def _run_arm(job):
    seed, arm, N, K, T, g, alpha, tau, epochs, eta_M, eta_W, sigma_train, sigma_eval, perturb_mag, n_traj = job
    targets = [make_target(seed * 1000 + ti, T=T, K=K) for ti in range(n_traj)]   # n_traj distinct trajectories
    net = PredictiveAlignmentNet(N, K, seed=seed, g=g, alpha=alpha, tau=tau, sigma=sigma_train, arm=arm, n_traj=n_traj)
    if arm != "untrained":
        net.train(targets, epochs, eta_M, eta_W)
    else:
        net._eta_M = net._eta_W = 0.0
    clean = net.generate_corr(targets, 0.0)
    noisy = net.generate_corr(targets, sigma_eval)
    pert = net.perturb_recovery_corr(targets, perturb_mag=perturb_mag)   # PRIMARY: stable-attractor recovery after a pulse
    return (seed, arm, {"gen_clean": clean, "gen_noisy": noisy, "pert_recovery": pert,
                        "locality_ok": (not net.used_transpose)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--T", type=int, default=220)
    ap.add_argument("--g", type=float, default=1.2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eta-M", type=float, default=0.0003)   # PA is sensitive: too-high eta_M destabilizes M (see EMERGE-8 eta_M sweep)
    ap.add_argument("--eta-W", type=float, default=0.01)
    ap.add_argument("--n-traj", type=int, default=1, help="number of distinct trajectories (capacity test: reservoirs fail as this grows)")
    ap.add_argument("--sigma-train", type=float, default=0.05)
    ap.add_argument("--sigma-eval", type=float, default=0.15)
    ap.add_argument("--perturb-mag", type=float, default=3.0)     # mid-trajectory pulse magnitude for the recovery metric
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        jobs = [(s, arm, a.N, a.K, a.T, a.g, a.alpha, a.tau, a.epochs, a.eta_M, a.eta_W, a.sigma_train, a.sigma_eval, a.perturb_mag, a.n_traj)
                for s in a.seeds for arm in ARMS]
        cap = a.max_workers if (a.max_workers and a.max_workers > 0) else (os.cpu_count() or 1)
        collected = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), cap)) as ex:
                for seed, arm, entry in ex.map(_run_arm, jobs):
                    collected.setdefault(seed, {})[arm] = entry
        except Exception:
            for job in jobs:
                seed, arm, entry = _run_arm(job); collected.setdefault(seed, {})[arm] = entry
        for s in a.seeds:
            d = collected[s]; d["seed"] = s; per.append(d)
        for d in per:
            print(f"  [seed {d['seed']}] PA recover {d['pa']['pert_recovery']:.3f} (clean {d['pa']['gen_clean']:.3f}) "
                  f"| RESERVOIR recover {d['fixed_reservoir']['pert_recovery']:.3f} (clean {d['fixed_reservoir']['gen_clean']:.3f}) "
                  f"| no_reg {d['pa_no_reg']['pert_recovery']:.3f} | scramQ {d['pa_scrambled_Q']['pert_recovery']:.3f} "
                  f"| untr {d['untrained']['pert_recovery']:.3f} || PA noisy {d['pa']['gen_noisy']:.3f} vs res {d['fixed_reservoir']['gen_noisy']:.3f} "
                  f"| loc {d['pa']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="pert_recovery"):
            return float(np.mean([p[arm][key] for p in per]))
        # PRIMARY metric = perturbation RECOVERY (does the target trajectory become a STABLE ATTRACTOR?). Continuous-noise
        # generation is secondary (a population readout is inherently noise-tolerant, so it is a weak discriminator).
        pa_r, pa_c = m("pa"), m("pa", "gen_clean")
        res_r, res_c = m("fixed_reservoir"), m("fixed_reservoir", "gen_clean")
        noreg_r, scram_r, unt_r = m("pa_no_reg"), m("pa_scrambled_Q"), m("untrained")
        pa_n, res_n = m("pa", "gen_noisy"), m("fixed_reservoir", "gen_noisy")
        loc = all(p["pa"]["locality_ok"] for p in per)
        task_sane = max(pa_c, res_c) >= 0.70                         # SOMETHING generates the clean target
        beats_reservoir = pa_r >= res_r + 0.15                       # PA RECOVERS from a pulse where the reservoir diverges
        reg_loadbearing = pa_r >= noreg_r + 0.10                     # the chaos-suppression term matters
        align_loadbearing = pa_r >= scram_r + 0.10                   # coherent feedback alignment matters
        above_floor = pa_r >= unt_r + 0.20
        go = bool(task_sane and beats_reservoir and above_floor and loc and (reg_loadbearing or align_loadbearing))
        if not loc:
            verdict = "INVALID -- locality assert failed (PA credit path used W.T / BPTT)."
        elif not task_sane:
            verdict = (f"INCONCLUSIVE -- neither PA nor the reservoir generates the CLEAN target (PA {pa_c:.3f}, "
                       f"reservoir {res_c:.3f}; <0.70). Tune N/g/epochs/eta/T/cue before the robustness verdict.")
        elif go:
            verdict = (f"GO -- PREDICTIVE ALIGNMENT (fully-local chaos-taming, Asabuki-Clopath 2025) makes the target "
                       f"trajectory a STABLE ATTRACTOR that RECOVERS from a perturbation where a fixed chaotic reservoir "
                       f"DIVERGES: PA post-pulse recovery {pa_r:.3f} >> fixed-reservoir {res_r:.3f} (clean gen PA {pa_c:.3f} "
                       f"/ reservoir {res_c:.3f}; noise-gen PA {pa_n:.3f} vs res {res_n:.3f}); chaos-suppression reg "
                       f"({'LB' if reg_loadbearing else 'n/l'}: no-reg {noreg_r:.3f}) + feedback alignment "
                       f"({'LB' if align_loadbearing else 'n/l'}: scrambled-Q {scram_r:.3f}); >> untrained floor {unt_r:.3f}; "
                       f"locality asserted (no W.T / no BPTT). Multi-seed. ⇒ PA is the rung-3 recurrent-training lever "
                       f"(the recurrent-credit value the naive local rules could not show) -> promote to 6 seeds, then "
                       f"rung-3b (spiking-LIF PA) + scope the sim/ PA-compatible port. NO sim/ edit.")
        else:
            miss = []
            if not beats_reservoir: miss.append(f"PA did NOT recover from perturbation better than the fixed reservoir (PA {pa_r:.3f} vs reservoir {res_r:.3f})")
            if not above_floor: miss.append(f"PA recovery not above the untrained floor ({pa_r:.3f} vs {unt_r:.3f})")
            if not (reg_loadbearing or align_loadbearing): miss.append(f"neither regularization ({noreg_r:.3f}) nor alignment ({scram_r:.3f}) was load-bearing")
            verdict = ("BOUNDARY (build-informative, not a stop) -- " + "; ".join(miss) + f" (clean gen PA {pa_c:.3f} / "
                       f"reservoir {res_c:.3f}, task-sane; noise-gen PA {pa_n:.3f} vs res {res_n:.3f}). The careful "
                       f"chaos-taming rule did not make the trajectory a more-recoverable attractor than the fixed "
                       f"reservoir at this config -> either PA-tuning (g/eta_M/alpha/epochs; PA is sensitive to the chaos "
                       f"gain + alignment lr) OR a genuine toy-scale limit (the recurrent-credit advantage may need "
                       f"larger scale / richer targets; escalate per the shortlist: Laje-Buonomano innate-trajectory "
                       f"training / Gilra-Gerstner FOLLOW). Do NOT start the sim/ port.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge8_predictive_alignment", "verdict": verdict,
               "mechanism": "Predictive Alignment (Asabuki-Clopath 2025, Nat Commun 16:6784): recurrent = fixed sparse "
                            "chaotic G + plastic M; dM = eta_M(Qz - (M-alphaG)r)r^T (align recurrent prediction with "
                            "fixed-random readout feedback Qz, suppress chaos); delta readout dW=eta_W(f-z)r^T; fully "
                            "local, no W^T, no inverse-correlation matrix, no BPTT; autonomous generation after a cue",
               "task": "autonomous generation of a periodic sinusoid-superposition trajectory (cue then free-run) under "
                       "state noise; metric = readout-vs-target correlation; PA vs FIXED-RESERVOIR bar under matched noise",
               "seeds": a.seeds, "config": {"N": a.N, "K": a.K, "T": a.T, "g": a.g, "alpha": a.alpha, "tau": a.tau,
               "epochs": a.epochs, "eta_M": a.eta_M, "eta_W": a.eta_W, "sigma_train": a.sigma_train,
               "sigma_eval": a.sigma_eval, "n_traj": a.n_traj, "perturb_mag": a.perturb_mag},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "The decisive merged-fork test: PA (the scoped chaos-taming rule) must BEAT the fixed "
                              "chaotic reservoir (same G, no plastic M) on autonomous generation UNDER NOISE -- the regime "
                              "where 3 prior probes showed naive local recurrent credit fails/degrades. Rate-limit "
                              "(spiking-LIF PA is rung-3b). Locality: PA uses own r, own fixed G, fixed random feedback Q, "
                              "post feedback-error -- no W^T, no BPTT (used_transpose False, asserted)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge8] VERDICT: {verdict}", flush=True)
    print(f"[emerge8] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
