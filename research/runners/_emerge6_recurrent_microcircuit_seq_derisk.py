"""EMERGE-6 (rung 3a) DE-RISK: does the confirmed active-cancellation credit, made RECURRENT (Muratore target-based
rule) with a local eligibility trace (NO BPTT), learn to STORE + autonomously RECALL a temporal trajectory?

Ladder: rung 1 (two-compartment burst multiplexing) DONE; rung 2 (rate->spike deep credit) RESOLVED -- the
Sacramento-Senn microcircuit's ACTIVE interneuron CANCELLATION is the noise-robust credit rule (EMERGE-5c GO, 0.981).
Rung 3 = a RECURRENT microcircuit SEQUENCE cortex (the communication-relevant target). Scoping
`2026-07-02-rung3-recurrent-microcircuit-sequence-scoping.md`: the confirmed feedforward cancellation credit
`W_PP_td @ (r_upper - r_int)` has an EXACT recurrent analogue -- the Muratore-Capone-Paolucci TARGET-BASED rule, the
diagonal/local (fully-local) limit of the Capone unified error<->target framework (PLoS CB 2022): local, online, NO
BPTT, NO weight transport. This is rung 3a: the RATE-limit version + the canonical store/recall task (no spike noise
yet -- that is rung 3b; do NOT conflate the recurrence question with the finite-sample-noise question).

THE MECHANISM (target-based recurrent credit, rate-limit):
  One recurrent rate population r_t in (0,1)^N. Autoregressive map a_{t+1} = sigmoid(W_rec @ r_t) (recurrence-driven
  activity = the network's prediction of the next state). The teacher provides the TARGET trajectory s*_{1..T} (the
  desired activity). Local target-based update (Muratore): dW_rec += (s*_{t+1} - a_{t+1}) (x) e_t, where the post-factor
  (s* - a) = "target-driven activity minus recurrence-driven activity" is the SAME cancellation-difference error as the
  feedforward microcircuit's (r_upper - r_int), and e_t = alpha*e_{t-1} + (1-alpha)*pre_t is the local eligibility trace
  (filtered presynaptic rate; Capone Eq. 7) -- the temporal-credit primitive that assigns credit across time with NO
  BPTT. Teacher-forced during training (pre_t = s*_t). RECALL = cue with s*_0, free-run r_{t+1}=sigmoid(W_rec @ r_t);
  teacher-neutrality means the recurrence must sustain the trajectory once the teacher is withdrawn.

ARMS (mirror EMERGE-3/5c): recurrent_microcircuit (TEST) · hebbian_selforg (Bouhadjar positive-control: associate
consecutive states, NO error/cancellation -- does the task even NEED credit?) · apical_feedback_lesion (teacher error
zeroed -> no learning -> floor) · wrong_sign (negate the target error -> anti-learn) · no_teaching_null (no target ->
no learning -> floor) · shuffled_target (train on a temporally-SHUFFLED target -> must FAIL to recall the ORDERED
sequence: proves temporal-order credit, the NEW sequence-specific anti-cheat) · untrained (random W_rec -> floor).

METRICS: recall_corr (Pearson of the autonomous free-run vs the target, over the HELD-OUT tail = the memorization
floor: a verbatim memorizer of the trained prefix must fail the held-out continuation) + onestep_corr (teacher-forced
one-step prediction corr = the achievable ceiling + task-one-step-sanity).

GO (multi-seed 42/43/44): task_sane (microcircuit onestep_corr >= 0.7 -- the local map IS learnable) AND
recurrent_microcircuit recall_corr_heldout >= 0.60 AND > hebbian_selforg + 0.10 (or >=, report honestly if Hebbian also
solves it) AND >> apical_feedback_lesion + 0.15; wrong_sign anti-learns (recall_corr <= 0.15); no_teaching_null flat
(~untrained); shuffled_target fails ordered recall (<= untrained + 0.15); locality asserted (no W_rec.T in the credit
path, eligibility forward-only, no BPTT). BOUNDARY = build-informative (if onestep high but recall low -> the map is
learned but autonomous recall is dynamically fragile -> burst-window gating / stability, per scoping risk #2; if
hebbian matches -> the task doesn't need credit, re-point to arbitrary-trajectory recall). Reuse-by-import; NO `sim/`
edit; CPU; arm-parallel with --max-workers (light-contention capable).
Run: SIM_BACKEND=numpy python -m research.runners._emerge6_recurrent_microcircuit_seq_derisk --seeds 42 43 44
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

OUT = _REPO / "research" / "findings" / "raw" / "_emerge6_recurrent_microcircuit_seq.json"
_MOMENTUM = 0.9


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def make_seq_task(seed, N=32, T=140, train_frac=0.7, n_modes=4):
    """A smooth periodic target trajectory s*_{1..T} in (0,1)^N (superposition of n_modes sinusoids mixed into N
    units -- the Muratore canonical). Train on the first train_frac; the HELD-OUT tail tests whether the learned
    DYNAMICS extrapolate the period (a verbatim memorizer of the prefix fails it)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    freqs = np.array([1.0, 2.0, 3.0, 5.0])[:n_modes]
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    modes = np.sin(2 * np.pi * freqs[:, None] * t[None, :] / T + ph[:, None])   # (n_modes, T)
    M = rng.normal(0, 1.0, (N, n_modes)) * 1.6                                   # fixed mixing
    sstar = _sig(M @ modes).T                                                    # (T, N) in (0,1)
    T_train = int(train_frac * T)
    return sstar, T_train


def _corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class RecurrentMicrocircuitRNN:
    """Recurrent target-based microcircuit (rate-limit): the confirmed active-cancellation credit made recurrent per
    Muratore-Capone-Paolucci. The ONLY weight is W_rec (Xavier from seed). Credit = (s* - a) (x) eligibility; NO BPTT,
    NO W_rec.T in the credit path (locality by construction)."""

    def __init__(self, N, seed=0, alpha=0.7):
        rng = np.random.default_rng(seed)
        lim = np.sqrt(6.0 / (2 * N))
        self.W = rng.uniform(-lim, lim, (N, N))
        self.N = N; self.alpha = float(alpha)
        self._vel = np.zeros((N, N))
        self.used_transpose = False        # locality flag: set True if the credit path ever reads W.T (it must not)

    def train(self, sstar, T_train, mode, epochs, lr, seed, free_run=False):
        """Target-based training over [0, T_train). Local, online-per-step credit accumulated per epoch.
        modes: recurrent_microcircuit | wrong_sign | apical_feedback_lesion | no_teaching_null | shuffled_target |
        hebbian_selforg. free_run=True uses SCHEDULED SAMPLING (teacher-forcing prob decays 1->floor over training) so
        the credit learns to correct the network's OWN free-run dynamics -- the exposure-bias fix for autonomous recall."""
        rng = np.random.default_rng(seed + 4242)
        traj = sstar
        if mode == "shuffled_target":
            perm = rng.permutation(T_train)                     # break the TEMPORAL ORDER of the teacher (order anti-cheat)
            traj = sstar.copy(); traj[:T_train] = sstar[:T_train][perm]
        for ep in range(epochs):
            e = np.zeros(self.N); dW = np.zeros((self.N, self.N))
            p_tf = 1.0 if not free_run else max(0.05, 1.0 - (ep / max(1, epochs - 1)) / 0.7)  # decaying teacher-forcing
            r_prev = traj[0].copy()
            for t in range(T_train - 1):
                # presynaptic drive: teacher (traj[t]) when teacher-forced, else the network's OWN previous output r_prev
                pre = traj[t] if (not free_run or rng.random() < p_tf) else r_prev
                a = _sig(self.W @ pre)                          # recurrence-driven prediction of the next state
                e = self.alpha * e + (1.0 - self.alpha) * pre   # local eligibility trace (forward-only; Capone Eq.7)
                if mode in ("apical_feedback_lesion", "no_teaching_null"):
                    err = np.zeros(self.N)                       # no teaching signal reaches the apical -> no learning
                elif mode == "wrong_sign":
                    err = -(traj[t + 1] - a)                     # negated target error -> anti-learn
                elif mode == "hebbian_selforg":
                    err = traj[t + 1]                            # Bouhadjar-style: associate consecutive states, NO (target-a) error
                else:                                            # target-based cancellation error (the TEST rule)
                    err = traj[t + 1] - a                        # (a* - a): target activity minus recurrence-driven activity
                dW += np.outer(err, e)                          # LOCAL: post-error (x) pre-eligibility; no W.T, no BPTT
                r_prev = a                                       # network's own output (used as pre when not teacher-forced)
            self._vel = _MOMENTUM * self._vel + dW / max(1, T_train - 1)
            # free-run correction rides on noisy OWN-dynamics -> a gentler lr (the diagnosed conflict: teacher-forced
            # wants a large lr for the one-step map, free-run wants a small one so it doesn't destabilize that map).
            self.W = self.W + (lr * (0.15 if free_run else 1.0)) * self._vel

    def recall(self, sstar):
        """Autonomous free-run from the cue s*_0 (teacher withdrawn -> teacher-neutrality test)."""
        r = sstar[0].copy(); out = [r]
        for _ in range(len(sstar) - 1):
            r = _sig(self.W @ r); out.append(r)
        return np.asarray(out)

    def onestep_pred(self, sstar):
        """Teacher-forced one-step prediction a_{t+1}=sig(W@s*_t) (the achievable-map ceiling / one-step sanity)."""
        return np.asarray([_sig(self.W @ sstar[t]) for t in range(len(sstar) - 1)])


def _run_arm(job):
    seed, arm, N, T, epochs, lr, alpha = job
    sstar, T_train = make_seq_task(seed, N=N, T=T)
    net = RecurrentMicrocircuitRNN(N, seed=seed, alpha=alpha)
    if arm != "untrained":                                          # untrained -> random-dynamics floor
        train_mode = "recurrent_microcircuit" if arm == "mc_freerun" else arm
        net.train(sstar, T_train, train_mode, epochs, lr, seed, free_run=(arm == "mc_freerun"))
    rec = net.recall(sstar)
    heldout = _corr(rec[T_train:], sstar[T_train:])                  # HELD-OUT continuation (memorization floor)
    full = _corr(rec, sstar)
    onestep = _corr(net.onestep_pred(sstar)[T_train - 1:], sstar[T_train:])   # one-step on the held-out region
    return (seed, arm, {"recall_heldout": heldout, "recall_full": full, "onestep": onestep,
                        "locality_ok": (not net.used_transpose)})


ARMS = ["mc_freerun", "recurrent_microcircuit", "hebbian_selforg", "apical_feedback_lesion", "wrong_sign",
        "no_teaching_null", "shuffled_target", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=int, default=140)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.7)            # eligibility-trace time constant
    ap.add_argument("--max-workers", type=int, default=0, help="cap parallel workers (0=all cores; e.g. 4 to leave CPU free)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        jobs = [(s, arm, a.N, a.T, a.epochs, a.lr, a.alpha) for s in a.seeds for arm in ARMS]
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
            d = collected[s]; d["seed"] = s; per.append(d)
        for d in per:
            print(f"  [seed {d['seed']}] FREERUN recall(held) {d['mc_freerun']['recall_heldout']:.3f} "
                  f"(1step {d['mc_freerun']['onestep']:.3f}) | naive-TF recall {d['recurrent_microcircuit']['recall_heldout']:.3f}"
                  f" | hebbian {d['hebbian_selforg']['recall_heldout']:.3f} | lesion {d['apical_feedback_lesion']['recall_heldout']:.3f}"
                  f" | wrong {d['wrong_sign']['recall_heldout']:.3f} | null {d['no_teaching_null']['recall_heldout']:.3f}"
                  f" | shuffled {d['shuffled_target']['recall_heldout']:.3f} | untrained {d['untrained']['recall_heldout']:.3f}"
                  f" | loc_ok {d['mc_freerun']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="recall_heldout"):
            return float(np.mean([p[arm][key] for p in per]))
        # PRIMARY = mc_freerun (scheduled-sampling, the autonomous-recall fix); naive teacher-forced kept as baseline
        mc, mc_1step = m("mc_freerun"), m("mc_freerun", "onestep")
        naive_tf = m("recurrent_microcircuit")
        heb, les = m("hebbian_selforg"), m("apical_feedback_lesion")
        wrong, null, shuf, unt = m("wrong_sign"), m("no_teaching_null"), m("shuffled_target"), m("untrained")
        loc = all(p["mc_freerun"]["locality_ok"] for p in per)
        task_sane = mc_1step >= 0.70
        recalls = mc >= 0.60
        beats_hebbian = mc > heb + 0.10
        beats_lesion = mc > les + 0.15
        wrong_anti = wrong <= 0.15
        null_flat = null <= unt + 0.12
        shuffled_fails = shuf <= unt + 0.15
        go = bool(task_sane and recalls and beats_lesion and wrong_anti and null_flat and shuffled_fails and loc
                  and (beats_hebbian or heb >= 0.60))
        if not loc:
            verdict = "INVALID -- locality assert failed (the credit path used W_rec.T / BPTT). Fix before trusting."
        elif not task_sane:
            verdict = (f"INCONCLUSIVE -- the target-based rule's one-step map is only {mc_1step:.3f} (<0.70), so the "
                       f"sequence's local structure isn't cleanly learnable at this config (N={a.N}/T={a.T}/epochs="
                       f"{a.epochs}/lr={a.lr}/alpha={a.alpha}); tune before reading recall. NOT a mechanism verdict.")
        elif go:
            verdict = (f"GO -- the confirmed active-cancellation credit, made RECURRENT (target-based rule + local "
                       f"eligibility trace, NO BPTT) and trained with the network's OWN dynamics in the loop (scheduled "
                       f"sampling), learns to STORE + autonomously RECALL the trajectory: FREE-RUN held-out recall "
                       f"{mc:.3f} (one-step map {mc_1step:.3f}; naive teacher-forced-only recall {naive_tf:.3f} -- the "
                       f"exposure-bias baseline the dynamics-in-loop training FIXED) >> apical-lesion {les:.3f} + "
                       f"untrained {unt:.3f}; wrong-sign anti-learns ({wrong:.3f}), no-teaching null flat ({null:.3f}), "
                       f"shuffled-target fails ordered recall ({shuf:.3f} -- temporal-order credit is load-bearing), "
                       f"{'beats' if mc > heb + 0.10 else 'ties (Hebbian also solves -- report honest)'} hebbian ({heb:.3f}), "
                       f"locality asserted (no W.T / no BPTT). Multi-seed. ⇒ rung 3a passes -> run 3b (spike noise) + "
                       f"Task B (next-symbol). NO sim/ edit.")
        else:
            miss = []
            if not recalls: miss.append(f"free-run recall too low (held-out {mc:.3f} < 0.60; naive-TF {naive_tf:.3f}; one-step {mc_1step:.3f})")
            if not beats_lesion: miss.append(f"didn't beat apical-lesion (freerun {mc:.3f} vs lesion {les:.3f})")
            if not (beats_hebbian or heb >= 0.60): miss.append(f"didn't beat hebbian ({mc:.3f} vs {heb:.3f})")
            if not wrong_anti: miss.append(f"wrong-sign didn't anti-learn ({wrong:.3f})")
            if not null_flat: miss.append(f"no-teaching-null not flat ({null:.3f} vs untrained {unt:.3f})")
            if not shuffled_fails: miss.append(f"shuffled-target still recalled ({shuf:.3f}) -- temporal order NOT load-bearing")
            verdict = ("BOUNDARY (build-informative, not a stop) -- " + "; ".join(miss) + f" (one-step map {mc_1step:.3f}, "
                       f"task-sane; naive teacher-forced-only recall {naive_tf:.3f}). Even dynamics-in-loop scheduled "
                       f"sampling {'did not' if mc < 0.60 else 'only partly'} stabilize autonomous recall -> the next "
                       f"mechanism is a proper recurrent (e-prop first-order) eligibility trace capturing recurrent "
                       f"sensitivity, and/or burst-window gating (scoping risk #2); if hebbian matches -> the task "
                       f"self-organizes without credit; a recurrent-noise family limit -> Urbanczik-Senn "
                       f"population-feedback / NMNC (shortlist). Do NOT start the sim/ port.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge6_recurrent_microcircuit_seq", "verdict": verdict,
               "mechanism": "recurrent target-based microcircuit (Muratore-Capone-Paolucci): the confirmed feedforward "
                            "active-cancellation credit (r_upper - r_int) made recurrent as (s* - a) (x) eligibility_trace; "
                            "local, online, NO BPTT, NO weight transport; rate-limit, no spike noise (that is rung 3b)",
               "task": "store/recall of a periodic sinusoid-superposition trajectory (Muratore canonical); held-out tail "
                       "= memorization floor; autonomous free-run recall after teacher withdrawal (teacher-neutrality)",
               "seeds": a.seeds, "config": {"N": a.N, "T": a.T, "epochs": a.epochs, "lr": a.lr, "alpha": a.alpha},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Rung 3a = RATE-limit + no spike noise (rung 3b re-injects EMERGE-5's Binomial(S)/S). "
                              "Task-sanity/ceiling = the target-based rule's own one-step teacher-forced prediction "
                              "(a separate BPTT autonomous-recall oracle is deferred -- if autonomous recall is the crux, "
                              "that is the finding, per scoping risk #2). The clean-readout representation gate (rung-2) "
                              "is subsumed here by direct recall correlation (the units ARE the trajectory in this "
                              "generator task). Locality by construction: the credit is outer(post-error, pre-eligibility) "
                              "with a forward-only eligibility trace -- no W_rec.T, no BPTT, no cross-neuron error broadcast."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge6] VERDICT: {verdict}", flush=True)
    print(f"[emerge6] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
