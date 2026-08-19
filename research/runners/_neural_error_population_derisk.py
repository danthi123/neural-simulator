"""Neuralise the "was I wrong?" TEACHING ERROR via the Urbanczik-Senn dendritic-prediction rule (board #39).

THE SHORTCUT BEING CLOSED. The read-out that learns the brain's word choices is trained by a delta rule
`W_O -= lr * outer(act, err)` whose per-output error `err_j = est_j - target_j` is a HOST subtraction (a Python
formula). Per the BRAIN-BASED-ONLY standard a prediction error computed by a host formula is a documented SHORTCUT
even when numerically correct -- the BRAIN must compute it. This de-risk replaces the host subtraction with a
POPULATION OF TWO-COMPARTMENT NEURONS that works the error out ITSELF via Urbanczik & Senn (Neuron 2014,
"Learning by the Dendritic Prediction of Somatic Spiking"): each read-out neuron's DENDRITE predicts its own
SOMATIC firing, and the intrinsic mismatch `(soma_rate - phi(v_dendrite))` IS the local teaching error that drives
plasticity -- no host error formula anywhere in the loop.

This is a DISTINCT mechanism from the 2026-06-17 neural-error-localrule GO, which used a Rao-Ballard TWO-neuron
ON/OFF predictive-coding population that computed `relu(target-est) - relu(est-target)` (a separate error unit fed
exc=target, inh=prediction). Here the error is NOT a separate population -- it is the *self-prediction mismatch
internal to each read-out neuron's own two compartments*: the soma is nudged by the teacher toward the target, the
basal dendrite predicts the soma from the forward synaptic drive through the plastic weights, and the neuron's own
soma-minus-dendrite voltage comparison is the error. The subtraction `target - prediction` is done by the neuron's
compartments (biophysics), not by host arithmetic on two host-held numbers.

REUSE-BY-IMPORT.
  * `sim.dendritic_plasticity.urbanczik_senn_update` -- THE SHIPPED, literature-faithful U-S rule
    (`dw = outer(pre, gate*(soma_rate - sigma(v_basal)))`). This function computes the error. We import it as the
    learning substrate; we do NOT reimplement it and we do NOT edit sim/.
  * `research.runners.cortex_learned_binder_systematicity_probe` -- the role-filler word/sequence acquisition task
    harness (make_role_codes / make_systematicity_splits / native_argmax) the prior read-out de-risks already use.
  * `LocalRuleBinder` below is a faithful reconstruction of the base binder used by the 2026-06-17 read-out arc
    (`_phaseB_neural_error_localrule_derisk.py`); the original base module `_phaseB_localrule_readout_derisk.py` was
    never committed to the repo as a .py (only its finding + raw json), so it is inlined here from its documented
    usage. It carries the SAME fixed-random encoder W_F + plastic delta-rule decoder W_O + fixed +-1 role gate.

ARMS (6 seeds: 42,43,44,100,101,102; systematicity protocol; bundled held-out generalization):
  1. HOST-err (reference)         -- the delta rule with the exact host subtraction err = est - target.
  2. NEURAL-err (the de-risk)     -- err computed by the U-S soma-vs-dendrite mismatch of the read-out neurons:
                                     soma nudged toward target (finite nudging beta), dendrite predicts est through
                                     the plastic weights, mismatch read via the SHIPPED urbanczik_senn_update, with
                                     a spiking (Poisson spike-count) soma. Decoded by dividing out the fixed
                                     small-signal transfer slope (rate decoding, exactly as the spike-rate runner
                                     divides counts by the spike window) -- NOT a host error.
  3. LESION-nodend (anti-cheat 1) -- keep the teacher, but SILENCE the dendritic self-prediction (pin v_basal=0 so
                                     the dendrite no longer predicts the soma). The mismatch stops tracking the
                                     per-output estimate -> learning must fail. Proves the DENDRITIC PREDICTION (the
                                     neural error computer), not a residual host formula, drives learning.
  4. LESION-noteach (anti-cheat 1)-- SILENCE the somatic teaching nudge (beta=0) so soma == dendrite prediction ->
                                     mismatch identically ~0 -> the error population emits nothing -> no learning.
  5. SCRAMBLE (anti-cheat)        -- permute the neural error across outputs so err_j no longer addresses output j
                                     -> must collapse (the per-output error is load-bearing, not a noise artifact).

GO = NEURAL-err >= 0.85x HOST-err held-out generalization in >=5/6 seeds AND both lesions AND scramble collapse
(< 0.5x NEURAL). If NEURAL-err UNDER-performs HOST, that gap IS the honest-negative deliverable (it maps what the
substrate's own somato-dendritic error can do) -- reported precisely, not hidden.

CPU/numpy. NO sim/ edit. Additive only.
Run:  OMP_NUM_THREADS=2 SIM_BACKEND=numpy python -u -m research.runners._neural_error_population_derisk --seeds 42,43,44,100,101,102

Biology: Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking," Neuron 81:521-528, 2014
(PubMed 24507189) -- the local dendritic-voltage third-factor rule shipped in sim/dendritic_plasticity.py. The
soma-vs-dendrite error framing: Mikulasch, Rudelt, Wibral & Priesemann, "Where is the error? Hierarchical
predictive coding through dendritic error computation," Trends Neurosci 46:45-59, 2023 (PubMed 36577388) --
prediction errors are computed locally in dendritic compartments, not in separate units.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.dendritic_plasticity import urbanczik_senn_update, _sig  # noqa: E402  (THE shipped U-S rule)
from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)
from tools.lab import attributable_to  # noqa: E402  (attribute the learning to each neural mechanism, not the proxy)
from tools.verdict import Verdict      # noqa: E402  (emit the preconditions the verdict is earned against)

R, F, N_SPLITS = 4, 16, 3
N_FACT_STEPS = 24000
N_EVAL_FACTS = 40
D_H = 256


class LocalRuleBinder:
    """Fixed +-1 role gate + fixed-random encoder W_F + LOCAL delta-rule decoder W_O. bind(r,f) = role_pm1[r] *
    (f @ W_F); unbind(bundle,r) = (bundle * role_pm1[r]) @ W_O. Faithful reconstruction of the 2026-06-17 base
    binder (original base module never committed as a .py; inlined from its documented usage)."""

    def __init__(self, D_in, role_pm1, D_h, seed, lr=0.02, lam=1e-4):
        rng = np.random.default_rng(seed * 17 + 3)
        self.D_in = int(D_in)
        self.D_h = int(D_h)
        self.W_F = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
        self.W_O = np.zeros((D_h, D_in))
        self.role_pm1 = role_pm1                      # (R, D_h), +-1
        self.lr = float(lr)
        self.lam = float(lam)

    def bind(self, r, filler):
        return self.role_pm1[r] * (np.asarray(filler, float) @ self.W_F)

    def unbind(self, bundle, r):
        return np.nan_to_num((bundle * self.role_pm1[r]) @ self.W_O)

    def _preact(self, roleids, fillerids, fillers, t):
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [self.role_pm1[r] * w for r, w in zip(roleids, ws)]
        bundle = sum(gs)
        return bundle * self.role_pm1[roleids[t]]

    def train_fact_step(self, roleids, fillerids, fillers, t):
        act = self._preact(roleids, fillerids, fillers, t)
        est = act @ self.W_O
        target = fillers[fillerids[t]]
        err = est - target                              # HOST subtraction (the shortcut being closed)
        self.W_O -= self.lr * (np.outer(act, err) + self.lam * self.W_O)


class UrbanczikSennBinder(LocalRuleBinder):
    """The read-out decoder trained by the neurons' OWN Urbanczik-Senn somato-dendritic mismatch instead of the host
    subtraction. Per output neuron j: dendrite v_basal_j = est_j = act @ W_O[:,j] (forward drive through plastic
    weights); soma membrane u_j = (1-beta)*est_j + beta*target_j (finite teacher nudging); soma fires at a spiking
    (Poisson) rate s_j = sigma(g*u_j); the SHIPPED urbanczik_senn_update returns dw = outer(act, s_j - sigma(g*est_j))
    -- the neuron's soma-minus-dendrite mismatch. Decoded to an error estimate by dividing out the fixed
    small-signal transfer slope (= beta*g/4), then applied with the SAME lr/lam as the host arm. No host err formula."""

    def __init__(self, D_in, role_pm1, D_h, seed, gain=1.0, beta=0.5, spike_gain=20.0,
                 scale=1.0, mode="neural", lr=0.02, lam=1e-4):
        super().__init__(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed, lr=lr, lam=lam)
        self.g = float(gain) / max(float(scale), 1e-9)   # sigmoid gain, scaled to the code magnitude
        self.beta = float(beta)                          # teacher nudging conductance (finite -> faithful U-S soma)
        self.win = float(spike_gain)                     # soma spike-count window (spikes per unit rate)
        self.mode = str(mode)
        self.slope = max(self.beta, 1e-6) * self.g / 4.0  # small-signal transfer slope (fixed rate-decode gain)
        self._rng = np.random.default_rng(seed * 233 + 11)
        self._perm = self._rng.permutation(self.D_in) if mode == "scramble" else None

    def train_fact_step(self, roleids, fillerids, fillers, t):
        act = self._preact(roleids, fillerids, fillers, t)
        est = act @ self.W_O                            # dendritic forward drive == the dendrite's estimate
        target = fillers[fillerids[t]]                  # env/teacher scaffold (legit supervised signal)
        beta = 0.0 if self.mode == "lesion_noteach" else self.beta
        u = (1.0 - beta) * est + beta * target          # somatic membrane: teacher-nudged mix
        s_clean = _sig(self.g * u)                       # somatic rate
        s_noisy = self._rng.poisson(np.clip(s_clean, 0.0, None) * self.win) / self.win  # SPIKING soma read
        # dendritic self-prediction via the SHIPPED U-S rule (v_basal = g*est). LESION-nodend pins v_basal=0 so the
        # dendrite no longer predicts the soma (sigma(0)=0.5 constant); the mismatch stops tracking the estimate.
        v_basal = np.zeros_like(est) if self.mode == "lesion_nodend" else self.g * est
        dw = urbanczik_senn_update(act, s_noisy, v_basal, np.ones(self.D_in), None, 1.0)  # outer(act, s_noisy - sig(v_basal))
        if self.mode == "scramble":
            dw = dw[:, self._perm]                        # error no longer addresses its output (anti-cheat)
        # decode the mismatch to an error estimate (divide out the fixed transfer slope, as the spike-rate runner
        # divides counts by the window) and descend with the SAME lr/lam as host; += because the U-S mismatch
        # points along (target - est) = -(est - target).
        self.W_O += (self.lr / self.slope) * dw - self.lr * self.lam * self.W_O


def _eval(binder, split, fillers, train_set, rng):
    single = sum(int(native_argmax(binder.unbind(binder.bind(r, fillers[f]), r), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
    h_ok = h = 0
    for _ in range(N_EVAL_FACTS):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(binder.bind(r, fillers[int(fids[r])]) for r in range(3))
        for r in range(3):
            if (r, int(fids[r])) not in train_set:
                h_ok += int(native_argmax(binder.unbind(bundle, r), fillers) == fids[r]); h += 1
    return single, (h_ok / h if h else 0.0)


def run_seed(codes, seed, scale, args):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; d_in = fillers.shape[1]
    roles = make_role_codes(R, d_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((d_in, D_H)) / np.sqrt(d_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)   # (R, D_H)
    acc = {k: [] for k in ("host", "neural", "neural_single", "lesion_nodend", "lesion_noteach", "scramble")}
    for split in splits:
        tr = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr[r]) for r in range(3)) == 0:
            continue
        train_set = set(split["train"])

        def _train(binder):
            rr = np.random.default_rng(seed * 53 + 9)
            for _ in range(args.steps):
                fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
                binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], fillers, int(rr.integers(3)))
            return binder

        def _us(mode):
            return UrbanczikSennBinder(D_in=d_in, role_pm1=role_pm1, D_h=D_H, seed=seed, gain=args.gain,
                                       beta=args.beta, spike_gain=args.spike_gain, scale=scale, mode=mode,
                                       lr=args.lr, lam=args.lam)
        host = _train(LocalRuleBinder(D_in=d_in, role_pm1=role_pm1, D_h=D_H, seed=seed, lr=args.lr, lam=args.lam))
        neural = _train(_us("neural"))
        les_d = _train(_us("lesion_nodend"))
        les_t = _train(_us("lesion_noteach"))
        scr = _train(_us("scramble"))
        ev = lambda b: _eval(b, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        _, h_host = ev(host)
        n_s, h_neu = ev(neural)
        _, h_ld = ev(les_d)
        _, h_lt = ev(les_t)
        _, h_sc = ev(scr)
        acc["host"].append(h_host); acc["neural"].append(h_neu); acc["neural_single"].append(n_s)
        acc["lesion_nodend"].append(h_ld); acc["lesion_noteach"].append(h_lt); acc["scramble"].append(h_sc)
    row = {"seed": seed, **{k: float(np.mean(v)) for k, v in acc.items()}}
    print(f"  [seed {seed}] HOST {row['host']:.3f} | NEURAL {row['neural']:.3f} (single {row['neural_single']:.3f}) "
          f"| lesion-nodend {row['lesion_nodend']:.3f} | lesion-noteach {row['lesion_noteach']:.3f} "
          f"| scramble {row['scramble']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--steps", type=int, default=N_FACT_STEPS)
    ap.add_argument("--gain", type=float, default=1.5)       # sigmoid gain (x code-std): >1 => genuinely nonlinear
    ap.add_argument("--beta", type=float, default=0.5)       # finite teacher nudging conductance
    ap.add_argument("--spike-gain", type=float, default=20.0)  # soma spike-count window
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_neural_error_population.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    scale = float(np.std(codes[:F]))                        # fixed code magnitude -> sigmoid gain normalization
    t0 = time.time()
    print(f"[neural-error population de-risk] Urbanczik-Senn soma-vs-dendrite mismatch replacing the host teaching "
          f"error. gain={args.gain} beta={args.beta} spike_gain={args.spike_gain} lr={args.lr} seeds={seeds}",
          flush=True)
    rows = [run_seed(codes, s, scale, args) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, neural, neu_s = m("host"), m("neural"), m("neural_single")
    les_d, les_t, scr = m("lesion_nodend"), m("lesion_noteach"), m("scramble")
    n_par = sum(int(r["neural"] >= 0.85 * r["host"]) for r in rows)
    bar = int(np.ceil(5 / 6 * len(seeds)))
    ref = max(neural, 1e-9)
    sep = 0.5 * ref
    lesion_d_collapses = les_d < sep
    lesion_t_collapses = les_t < sep
    scramble_collapses = scr < sep
    go = (n_par >= bar) and lesion_d_collapses and lesion_t_collapses and scramble_collapses
    # VALIDITY preconditions (the experiment is interpretable only if every control collapses): a control that
    # does NOT differ from the NEURAL arm means the "learning" is a noise/host artifact -> UNDEFINED, not a GO.
    verdict = Verdict("neural-error-population: Urbanczik-Senn soma-vs-dendrite mismatch replaces host teaching error")
    verdict.control("lesion-nodend collapses (dendritic self-prediction load-bearing)", ref, les_d, min_separation=sep)
    verdict.control("lesion-noteach collapses (somatic teaching load-bearing)", ref, les_t, min_separation=sep)
    verdict.control("scramble collapses (per-output addressing load-bearing)", ref, scr, min_separation=sep)
    decided = verdict.decide(go=go, verbose=True)
    status = decided["status"]                       # GO / NO-GO / UNDEFINED, downgraded if a control did not collapse
    # ATTRIBUTION: what fraction of the learning is NOT present in each lesion control -- i.e. owned by the neural
    # mechanism, not a residual host formula (the gap#5 97%-clamp lesson: measure whose the difference is).
    attribution = {
        "dendritic_self_prediction_frac": attributable_to(
            "dendritic self-prediction (NEURAL vs nodend-lesion)", neural, les_d),
        "somatic_teaching_frac": attributable_to(
            "somatic teaching (NEURAL vs noteach-lesion)", neural, les_t),
        "per_output_addressing_frac": attributable_to(
            "per-output addressing (NEURAL vs scramble)", neural, scr),
    }
    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): HOST {host:.3f} | NEURAL {neural:.3f} (single {neu_s:.3f}) | "
          f"lesion-nodend {les_d:.3f} | lesion-noteach {les_t:.3f} | scramble {scr:.3f} | "
          f"NEURAL>=0.85x HOST: {n_par}/{len(seeds)}", flush=True)
    if go:
        print(f"  GO: the read-out neurons' OWN Urbanczik-Senn soma-vs-dendrite mismatch drives the delta rule as "
              f"well as the host error -- NEURAL {neural:.3f} = {neural/max(host,1e-9):.0%} of HOST in "
              f"{n_par}/{len(seeds)} seeds; SILENCING the dendritic self-prediction (lesion-nodend {les_d:.3f}) OR "
              f"the somatic teaching (lesion-noteach {les_t:.3f}) OR mis-addressing it (scramble {scr:.3f}) all "
              f"collapse learning. ==> the 'was I wrong?' error is neuralisable; the host error formula is removable.",
              flush=True)
    elif not (lesion_d_collapses and lesion_t_collapses and scramble_collapses):
        print(f"  INVALID: a control did NOT collapse (lesion-nodend {les_d:.3f}, lesion-noteach {les_t:.3f}, "
              f"scramble {scr:.3f} vs NEURAL {neural:.3f}) -- the neural error may not be load-bearing; re-examine.",
              flush=True)
    else:
        print(f"  BOUNDARY (honest negative): the U-S somato-dendritic error under-performs the host error "
              f"({neural:.3f} vs {host:.3f} = {neural/max(host,1e-9):.0%}) while all controls collapse -- the "
              f"finite-nudging + spiking-soma mismatch loses {1-neural/max(host,1e-9):.0%} of the exact error. This "
              f"maps what the substrate's own error can do; residual = the transfer bias + spike-count noise.",
              flush=True)
    print(f"  ATTRIBUTION: learning owned by dendritic self-prediction {attribution['dendritic_self_prediction_frac']:.3f}"
          f" | somatic teaching {attribution['somatic_teaching_frac']:.3f}"
          f" | per-output addressing {attribution['per_output_addressing_frac']:.3f}", flush=True)
    print(f"  EARNED STATUS (preconditions guard): {status}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*104}", flush=True)
    out = {"verdict": status, "go_headline": bool(go), "preconditions": decided["preconditions"],
           "attribution": attribution, "chance": 1.0 / F,
           "seeds": seeds, "gain": args.gain, "beta": args.beta, "spike_gain": args.spike_gain, "lr": args.lr,
           "lam": args.lam, "steps": args.steps, "host": host, "neural": neural, "neural_single": neu_s,
           "lesion_nodend": les_d, "lesion_noteach": les_t, "scramble": scr, "n_parity": n_par, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
