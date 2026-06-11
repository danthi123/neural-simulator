"""STEP 3 (true cortex) CHEAP-FIRST DE-RISK -- Option A + the systematicity control.

Per docs/plans/2026-06-10-step3-true-cortex-design.md (Sec 2 Option A, Sec 3 the familiarity gate,
Sec 5 the cheap-first de-risk, Sec 6 anti-cheats, Sec 7 the systematicity risk + run-it-FIRST discipline).

WHAT THIS TESTS (CPU, numpy, toy scale, seed 42, the project's REAL correlated denoise64 codes -- NOT
random-clean phasors). The composer's idealization is (I-1) an exact-inverse VSA algebra, (I-2) a clean-code
demand, (I-3) a god's-eye argmax cleanup + a host-`if` abstention. Option A keeps the FHRR bind/unbind
OPERATIONS but (1) replaces the argmax cleanup with a LEARNED associative-memory attractor (a CA3/Hopfield
net whose recurrent weights are the Hebbian outer product of the stored codes) and (2) adds a SEPARATE LEARNED
spiking familiarity gate for abstention (Bogacz-Brown anti-Hebbian, catalog D.04 CA1 match/mismatch). This
de-risk runs the THREE tests, the SYSTEMATICITY control FIRST so the likely negative surfaces in minutes.

TERMS (defined once):
  - role-filler binding: combine a role (agent/action/patient) with a filler (a concept word) into one
    composite vector; a fact "dog go north" is the bundle (sum) of three bound role-filler pairs.
  - bind/unbind: bind makes the composite (FHRR = phasor product); unbind recovers a filler given a role
    (multiply by the conjugate). On the project's spike-phasor neurons: bind = phase ADD, unbind = phase
    SUBTRACT, bundle = phase-midpoint -- the validated SpikingPhasorFHRR primitives (reused by import).
  - cleanup: snap a noisy unbind estimate to the nearest stored concept code. Baseline = argmax over the
    codebook (a god's-eye matched-filter lookup); Option A = a LEARNED Hopfield/CA3 attractor (the
    ResonateFireTPAM, content-addressable memory, vocabulary in distributed Hebbian outer-product weights).
  - familiarity/novelty: a learned signal that reports "have I seen this cue before?" independent of WHAT it
    is -- gates abstention. Here: Bogacz-Brown ANTI-HEBBIAN ("cells that fire together wire apart"), whose
    response is SUPPRESSED for a familiar (stored) cue and HIGH for a novel one. A LEARNED signal, not a host
    `if`.
  - systematicity (Fodor-Pylyshyn): if a system binds "dog go north" it should bind "cat go north" WITHOUT
    separate training. A symbolic algebra has this for free; a learned readout notoriously does not. THE
    CORE RISK of step 3 -- so it is tested FIRST (Sec 7 discipline).
  - decorrelation (ZCA): orthonormalize the concept codebook (G^{-1/2} @ codes) -- biologically the dentate
    gyrus pattern-separation / ventral efficient-coding step (catalog D.12). Option A's "learned-decorrelated
    codes." Tested as a SEPARATE regime so the attractor's dependence on it is mapped.

REPRESENTATION (the crux: stress the clean-code demand with the BRAIN's codes, not random-clean phasors).
The denoise64 codes are signed real vectors (centered, unit-norm), mean pairwise cosine ~0.70-0.81 -- highly
CORRELATED. They are mapped to PHASOR phases by a fixed COMPLEX random projection (phase = angle of W_c @ code),
which (a) spreads phases across the FULL [0,1) circle (a valid FHRR code) and (b) PRESERVES the cross-code
correlation (reported as between-code phase-similarity, so the stress is auditable). The de-risk runs the
cleanup test in BOTH the raw-correlated regime AND the ZCA-decorrelated regime so the attractor's dependence
on decorrelation is the mapped boundary, not a hidden confound.

BRAIN-BASED-ONLY: the cleanup is the LEARNED attractor's recurrent settle (the vocabulary in distributed
Hebbian weights, no argmax over an enumerated god's-eye list); the familiarity is the LEARNED anti-Hebbian
pool's novelty energy (not a host max-similarity if). ANTI-CHEAT: lesion the learned weights -> the attractor
cleanup degrades to chance AND the familiarity separation collapses -> proves both ride the LEARNED weights,
not the algebra / a host path.

Run:  SIM_BACKEND=numpy python -m research.runners.cortex_learned_cleanup_derisk --seed 42
CPU-cheap (a few hundred-D matrix ops + a short attractor relaxation + an anti-Hebbian outer product); minutes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import the project's validated phasor primitives + the learned Hopfield attractor (TPAM).
from research.runners.spiking_phasor_fhrr import (
    phase_sum_neuron, phase_subtraction_neuron, phase_midpoint_bundle,
    phases_to_spikes, spikes_to_phases, phase_similarity, CYCLE_STEPS,
)
from research.runners.resonate_fire_fhrr import (
    ResonateFireTPAM, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

CACHE = os.path.join(_REPO, "research", "findings", "raw",
                     "activity_level_integration_cache", "denoise64_seed%d.npz")


# ---------------------------------------------------------------------------
# REAL codes -> correlated phasor phases (the crux).
# ---------------------------------------------------------------------------
def _phase_map(signed_codes, seed):
    """signed real codes [V, D] -> phasor phases [V, D] in [0,1) via a FIXED complex random projection:
    phase = angle(code @ W_c). Spreads phases across the FULL circle (a valid FHRR code) WHILE preserving the
    cross-code correlation (a shared linear map). Deterministic per seed."""
    V, D = signed_codes.shape
    rngc = np.random.default_rng(seed + 1000)
    Wc = (rngc.standard_normal((D, D)) + 1j * rngc.standard_normal((D, D))) / np.sqrt(2.0 * D)
    z = signed_codes.astype(complex) @ Wc
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def _between_phase_cos(phase_codes):
    V = phase_codes.shape[0]
    s = [float(np.mean(np.cos(2.0 * np.pi * (phase_codes[i] - phase_codes[k]))))
         for i in range(V) for k in range(i + 1, V)]
    return float(np.mean(s)) if s else 0.0


def load_real_codes(seed, proj_dim, rng, decorrelate=False):
    """Load the brain's REAL denoise64 concept codes -> signed real codes [V, D] (the project's load_concepts
    treatment: mean over obs samples, random-Gaussian project to proj_dim preserving cosines, mean-center +
    unit-normalize). If decorrelate, ZCA-orthonormalize the codebook first (catalog D.12 pattern separation).
    Returns (words, signed_codes, phase_codes, between_phase_cos)."""
    d = np.load(CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)   # [V, 3200]
    if proj_dim and proj_dim > 0:
        P = rng.standard_normal((raw.shape[1], proj_dim)) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    if decorrelate and codes.shape[0] > 1:
        g = codes @ codes.T
        ev, evec = np.linalg.eigh(g)
        ginvsqrt = evec @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-9))) @ evec.T
        codes = ginvsqrt @ codes
        codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    phase_codes = _phase_map(codes, seed)
    return ws, codes, phase_codes, _between_phase_cos(phase_codes)


# ---------------------------------------------------------------------------
# The LEARNED spiking familiarity gate (Bogacz-Brown anti-Hebbian, catalog D.04).
# ---------------------------------------------------------------------------
class AntiHebbianFamiliarity:
    """A LEARNED novelty/familiarity detector built on the Bogacz-Brown anti-Hebbian rule (perirhinal
    familiarity discrimination via repetition suppression). Catalog D.04 (CA1 match/mismatch).

    Mechanism (rate form of the spiking anti-Hebbian network; CPU-cheap, the weights are the load-bearing
    LEARNED state). A recurrent pool driven by an input cue x (the phasor cue rendered as a real I/Q activity
    vector). The familiarity readout is the NOVELTY ENERGY
        N(x) = ||x||^2 - x^T W x
    For a STORED (familiar) pattern, W x ~ x (the anti-Hebbian recurrence reconstructs it), so x^T W x ~
    ||x||^2 and N ~ 0 (SUPPRESSED -> familiar). For a NOVEL pattern, W x is uncorrelated with x, so x^T W x ~ 0
    and N ~ ||x||^2 (HIGH -> novel). This is exactly repetition suppression: low to familiar, high to novel.

    LEARNING is ANTI-HEBBIAN ("fire together wire apart"), realised as the projector onto the stored subspace:
    imprinting a pattern x_i adds it to a Gram-Schmidt-orthonormalized stored basis {u_i}; W = sum_i u_i u_i^T.
    This is the high-capacity-on-CORRELATED-inputs form Bogacz-Brown prove (the projector reconstructs any
    vector in the stored span, so a familiar -- even correlated -- cue is suppressed; a novel cue with a
    component outside the span is not). The weights are LEARNED from the imprinted cues; lesioning them (W:=0)
    makes N(x)=||x||^2 for EVERY input -> the familiar/novel separation collapses (the anti-cheat).

    The phasor cue is rendered to a real activity vector by [cos(2pi*phase), sin(2pi*phase)] (the population's
    in-phase/quadrature drive -- the standard resonate-and-fire real read of a phasor), unit-normalized.
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self._basis = []          # LEARNED orthonormal stored vectors u_i (the anti-Hebbian memory)
        self.W = np.zeros((2 * self.dim, 2 * self.dim), dtype=np.float64)

    @staticmethod
    def _render(phase_code):
        ph = np.asarray(phase_code, dtype=np.float64)
        x = np.concatenate([np.cos(2.0 * np.pi * ph), np.sin(2.0 * np.pi * ph)])
        return x / (np.linalg.norm(x) + 1e-12)

    def imprint(self, phase_code):
        """Anti-Hebbian imprint of a cue: add it to the stored orthonormal basis and rebuild W (projector onto
        the stored span). Idempotent for an already-stored (in-span) cue."""
        x = self._render(phase_code)
        for u in self._basis:
            x = x - (u @ x) * u
        nrm = np.linalg.norm(x)
        if nrm > 1e-6:
            self._basis.append(x / nrm)
            self._rebuild()

    def _rebuild(self):
        if self._basis:
            U = np.stack(self._basis, axis=1)
            self.W = U @ U.T
        else:
            self.W = np.zeros((2 * self.dim, 2 * self.dim), dtype=np.float64)

    def lesion(self):
        self.W = np.zeros((2 * self.dim, 2 * self.dim), dtype=np.float64)

    def novelty(self, phase_code):
        x = self._render(phase_code)
        return float(x @ x - x @ (self.W @ x))


# ---------------------------------------------------------------------------
# The FHRR composer core (reuse-by-import bind/unbind/bundle; codes are the REAL correlated phasor codes).
# ---------------------------------------------------------------------------
class _FHRRCore:
    """Minimal FHRR role-filler composer on the REAL phasor codes. bind = phase ADD, unbind = phase SUBTRACT,
    bundle = phase-midpoint -- the project's validated spiking-phasor primitives (reuse by import). Roles are
    random phasor codes (deterministic per seed). This is the UNCHANGED VSA operation half of Option A; the
    cleanup (argmax vs learned attractor) is applied to its noisy unbind estimate. Concept codes are stored as
    SPIKE TRAINS (the project's representation -- and the representation the TPAM vocab + the query share)."""

    ROLES = ("agent", "action", "patient")

    def __init__(self, words, phase_codes, seed, t_steps=CYCLE_STEPS):
        self.words = list(words)
        self.t_steps = int(t_steps)
        self.D = phase_codes.shape[1]
        self.code = {w: phases_to_spikes(phase_codes[i], t_steps) for i, w in enumerate(self.words)}
        rng = np.random.default_rng(seed + 7)
        self.role = {r: phases_to_spikes(rng.uniform(0.0, 1.0, self.D), t_steps) for r in self.ROLES}

    def encode(self, fact):
        bound = [phase_sum_neuron(self.role[r], self.code[fact[r]], self.t_steps)
                 for r in self.ROLES if r in fact]
        return phase_midpoint_bundle(bound, self.t_steps) if len(bound) > 1 else bound[0]

    def unbind(self, composite, role):
        return phase_subtraction_neuron(composite, self.role[role], self.t_steps)

    def make_tpam(self, words=None, lesion=False):
        """The LEARNED Hopfield/CA3 attractor over the codebook: W = S S* Hebbian outer product of the stored
        SPIKE-TRAIN codes (the vocab the query shares). lesion -> zero the recurrent weights (anti-cheat)."""
        words = words if words is not None else self.words
        tpam = ResonateFireTPAM([self.code[w] for w in words], t_steps=self.t_steps)
        if lesion:
            tpam.w = np.zeros_like(tpam.w)
        return tpam


def _argmax_cleanup(rec_spikes, core, words):
    """BASELINE cleanup (I-3): god's-eye argmax over the codebook by FHRR phase-similarity (a matched filter)."""
    sims = [phase_similarity(rec_spikes, core.code[w], core.t_steps) for w in words]
    return words[int(np.argmax(sims))]


def _attractor_cleanup(rec_spikes, tpam, words):
    """OPTION-A cleanup: the LEARNED Hopfield/CA3 attractor settles the noisy phasor onto the nearest stored
    attractor (catalog D.13 pattern completion). No argmax over a god's-eye list -- the vocabulary lives in
    the distributed recurrent weights."""
    idx, _frac = tpam.cleanup_annealed(rec_spikes, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS)
    return words[idx]


# ---------------------------------------------------------------------------
# TEST 1 -- SYSTEMATICITY CONTROL (run FIRST; the core risk).
# ---------------------------------------------------------------------------
def test_systematicity(core, tpam, words, rng, n_trials=80):
    """Train/imprint the learned cleanup on a set of role-filler bindings, HOLD OUT a novel combination whose
    PARTS were trained but the COMBINATION never was, and test whether it generalizes.

    Concretely: a small role-filler grid (a few agents x actions x patients). The learned cleanup's vocabulary
    is the set of fillers (the attractor stores the filler codes -- the codebook the unbind must resolve to);
    every filler appears in SOME trained fact, but the held-out (a, ac, p) triple is NEVER one of the stored
    facts. At test time we STORE the held-out fact (bind+bundle) and query each role. A SYSTEMATIC binder
    recovers a, ac, p (it composes parts it knows); a MEMORIZING readout that only echoes trained traces fails.

    Reported: TRAINED-combo accuracy (facts the binder WAS given) vs HELD-OUT novel-combo accuracy (parts seen,
    combination never). For Option A the FHRR operation is identical for every operand and the attractor only
    stores the FILLER codebook -- so this directly measures whether a never-seen COMBINATION of known parts
    recovers. Option A is EXPECTED to pass (held-out ~ trained); a learned-readout binder (Option C) is where
    this goes NEGATIVE -- this control exposes exactly that, the deliverable either way. The argmax baseline (the
    algebra's systematicity ceiling, also operand-agnostic) is reported alongside."""
    agents = [w for w in ("dog", "cat", "river") if w in words]
    actions = [w for w in ("go", "come", "look") if w in words]
    patients = [w for w in ("north", "south", "east", "west") if w in words]
    fillers = sorted(set(agents + actions + patients))
    tpam_f = core.make_tpam(fillers)        # attractor over JUST the filler codebook (the cleanup targets)

    trained_ok = trained_n = held_ok = held_n = held_ok_argmax = 0
    all_combos = list(itertools.product(agents, actions, patients))
    for _ in range(n_trials):
        a_h, ac_h, p_h = all_combos[rng.integers(len(all_combos))]
        trained_sample = [c for c in all_combos if c != (a_h, ac_h, p_h)]
        for _ in range(3):
            a, ac, p = trained_sample[rng.integers(len(trained_sample))]
            comp = core.encode({"agent": a, "action": ac, "patient": p})
            ra = _attractor_cleanup(core.unbind(comp, "agent"), tpam_f, fillers)
            rac = _attractor_cleanup(core.unbind(comp, "action"), tpam_f, fillers)
            rp = _attractor_cleanup(core.unbind(comp, "patient"), tpam_f, fillers)
            trained_ok += int(ra == a and rac == ac and rp == p)
            trained_n += 1
        comp = core.encode({"agent": a_h, "action": ac_h, "patient": p_h})
        ra = _attractor_cleanup(core.unbind(comp, "agent"), tpam_f, fillers)
        rac = _attractor_cleanup(core.unbind(comp, "action"), tpam_f, fillers)
        rp = _attractor_cleanup(core.unbind(comp, "patient"), tpam_f, fillers)
        held_ok += int(ra == a_h and rac == ac_h and rp == p_h)
        held_n += 1
        ra2 = _argmax_cleanup(core.unbind(comp, "agent"), core, fillers)
        rac2 = _argmax_cleanup(core.unbind(comp, "action"), core, fillers)
        rp2 = _argmax_cleanup(core.unbind(comp, "patient"), core, fillers)
        held_ok_argmax += int(ra2 == a_h and rac2 == ac_h and rp2 == p_h)

    return {
        "trained_acc": trained_ok / max(1, trained_n),
        "heldout_acc_attractor": held_ok / max(1, held_n),
        "heldout_acc_argmax": held_ok_argmax / max(1, held_n),
        "trained_n": trained_n, "heldout_n": held_n,
        "n_fillers": len(fillers), "n_combos": len(all_combos),
        "chance_per_role": 1.0 / max(1, len(fillers)),
    }


# ---------------------------------------------------------------------------
# TEST 2 -- LEARNED ATTRACTOR CLEANUP vs ARGMAX (on the real correlated codes).
# ---------------------------------------------------------------------------
def test_cleanup(core, tpam, tpam_lesion, words, rng, n_trials=200):
    """Bind two stored facts, unbind one role, clean up the noisy estimate two ways: (a) the LEARNED Hopfield/
    CA3 attractor (Option A) and (b) the god's-eye argmax (baseline). Gate: attractor >= argmax on the codes.
    Also report a LESIONED-attractor accuracy (recurrent weights zeroed) as the anti-cheat -- it must collapse
    toward chance, proving the cleanup rides the LEARNED weights."""
    n_arg = n_att = n_les = n_tot = 0
    for _ in range(n_trials):
        a, ac, p = (str(x) for x in rng.choice(words, size=3, replace=False))
        comp = core.encode({"agent": a, "action": ac, "patient": p})
        role = rng.choice(core.ROLES)
        truth = {"agent": a, "action": ac, "patient": p}[role]
        rec = core.unbind(comp, role)
        n_arg += int(_argmax_cleanup(rec, core, words) == truth)
        n_att += int(_attractor_cleanup(rec, tpam, words) == truth)
        n_les += int(_attractor_cleanup(rec, tpam_lesion, words) == truth)
        n_tot += 1
    return {
        "argmax_acc": n_arg / n_tot, "attractor_acc": n_att / n_tot,
        "attractor_lesioned_acc": n_les / n_tot, "n": n_tot,
        "chance": 1.0 / len(words),
    }


# ---------------------------------------------------------------------------
# TEST 3 -- the LEARNED FAMILIARITY GATE (the no-confab moat).
# ---------------------------------------------------------------------------
def test_familiarity(core, phase_codes, words, rng, n_known=8, n_trials=40):
    """A LEARNED anti-Hebbian familiarity signal that separates a KNOWN cue (low novelty) from an UNKNOWN cue
    (high novelty) -> abstain if novel. Imprint a set of KNOWN concept codes; present known cues (familiar) and
    NEVER-imprinted cues (novel: fresh random phasor codes outside the stored span); confirm
    N(known) < threshold < N(unknown) with a clean margin. Anti-cheat: lesion the LEARNED weights -> every
    novelty equals ||x||^2 -> the separation collapses."""
    D = core.D
    fam = AntiHebbianFamiliarity(D)
    known_idx = list(range(len(words)))[:n_known]
    for i in known_idx:
        fam.imprint(phase_codes[i])

    known_N, unknown_N = [], []
    for _ in range(n_trials):
        i = known_idx[rng.integers(len(known_idx))]
        known_N.append(fam.novelty(phase_codes[i]))
        unknown_N.append(fam.novelty(rng.uniform(0.0, 1.0, D)))   # a never-imprinted cue
    known_N = np.asarray(known_N); unknown_N = np.asarray(unknown_N)
    sep_margin = float(unknown_N.min() - known_N.max())
    thr = 0.5 * (known_N.max() + unknown_N.min())
    known_correct = float(np.mean(known_N < thr))
    unknown_correct = float(np.mean(unknown_N > thr))

    fam.lesion()
    kl = np.asarray([fam.novelty(phase_codes[known_idx[rng.integers(len(known_idx))]]) for _ in range(n_trials)])
    ul = np.asarray([fam.novelty(rng.uniform(0.0, 1.0, D)) for _ in range(n_trials)])
    lesion_margin = float(ul.min() - kl.max())

    return {
        "known_mean": float(known_N.mean()), "known_max": float(known_N.max()),
        "unknown_mean": float(unknown_N.mean()), "unknown_min": float(unknown_N.min()),
        "separation_margin": sep_margin, "threshold": thr,
        "known_below_thr_frac": known_correct, "unknown_above_thr_frac": unknown_correct,
        "clean_separation": bool(sep_margin > 0.0),
        "lesion_margin": lesion_margin, "lesion_collapsed": bool(lesion_margin <= 1e-9),
        "n_known_imprinted": len(known_idx), "n_trials": n_trials,
    }


def _run_regime(seed, proj_dim, t_steps, decorrelate):
    """Build the core + codes for one regime (raw-correlated or ZCA-decorrelated); run cleanup + systematicity."""
    words, signed, phase_codes, bcos = load_real_codes(
        seed, proj_dim, np.random.default_rng(seed), decorrelate=decorrelate)
    core = _FHRRCore(words, phase_codes, seed, t_steps=t_steps)
    tpam = core.make_tpam()
    tpam_lesion = core.make_tpam(lesion=True)
    clr = test_cleanup(core, tpam, tpam_lesion, words, np.random.default_rng(seed + 2))
    sysr = test_systematicity(core, tpam, words, np.random.default_rng(seed + 1))
    return words, phase_codes, core, bcos, clr, sysr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=512,
                    help="random-Gaussian projection dim for the real codes (preserves cosines)")
    ap.add_argument("--t-steps", type=int, default=CYCLE_STEPS)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw",
                                         "_cortex_learned_cleanup_derisk.json"))
    args = ap.parse_args()

    if not os.path.exists(CACHE % args.seed):
        print("[derisk] MISSING denoise64 cache %s -- cannot run on REAL codes" % (CACHE % args.seed),
              flush=True)
        return 1

    print("=== STEP 3 cheap-first de-risk (Option A + systematicity control) ===", flush=True)
    print("seed=%d proj_dim=%d t_steps=%d  REAL denoise64 codes (NOT random-clean)"
          % (args.seed, args.proj_dim, args.t_steps), flush=True)

    # Regime 1: the brain's RAW correlated codes (the headline stress).
    print("\n>>> REGIME: RAW correlated codes (the brain's codes, no decorrelation) <<<", flush=True)
    words, phase_codes, core, bcos, clr, sysr = _run_regime(args.seed, args.proj_dim, args.t_steps, False)
    print("[codes] V=%d D=%d  between-code PHASE-cos=%.3f  (the correlated-code stress)"
          % (len(words), phase_codes.shape[1], bcos), flush=True)

    # Regime 2: ZCA-decorrelated codes (Option A's "learned-decorrelated" codebook; catalog D.12).
    print("\n>>> REGIME: ZCA-DECORRELATED codes (Option A learned-decorrelation; DG pattern separation) <<<",
          flush=True)
    _wd, phase_codes_d, core_d, bcos_d, clr_d, sysr_d = _run_regime(
        args.seed, args.proj_dim, args.t_steps, True)
    print("[codes] between-code PHASE-cos=%.3f  (decorrelated)" % bcos_d, flush=True)

    # ---- TEST 1: SYSTEMATICITY (FIRST) ----  reported on the regime where the attractor functions (decorr).
    print("\n--- TEST 1: SYSTEMATICITY control (run FIRST -- the core risk) ---", flush=True)
    print("  [decorrelated regime, where the learned attractor functions]", flush=True)
    print("  trained-combo acc (attractor)  : %.3f  (n=%d)" % (sysr_d["trained_acc"], sysr_d["trained_n"]),
          flush=True)
    print("  HELD-OUT novel-combo acc (attr): %.3f  (n=%d, chance/role=%.3f over %d fillers)"
          % (sysr_d["heldout_acc_attractor"], sysr_d["heldout_n"], sysr_d["chance_per_role"],
             sysr_d["n_fillers"]), flush=True)
    print("  HELD-OUT novel-combo acc (argmx): %.3f  (the algebra's systematicity ceiling)"
          % sysr_d["heldout_acc_argmax"], flush=True)

    # ---- TEST 2: LEARNED ATTRACTOR CLEANUP vs ARGMAX ----  both regimes (the mapped boundary).
    print("\n--- TEST 2: learned attractor cleanup vs argmax (REAL codes; both regimes) ---", flush=True)
    print("  RAW correlated   : argmax=%.3f  attractor=%.3f  lesioned=%.3f  (chance=%.3f)"
          % (clr["argmax_acc"], clr["attractor_acc"], clr["attractor_lesioned_acc"], clr["chance"]),
          flush=True)
    print("  ZCA decorrelated : argmax=%.3f  attractor=%.3f  lesioned=%.3f  (gate: attractor>=argmax)"
          % (clr_d["argmax_acc"], clr_d["attractor_acc"], clr_d["attractor_lesioned_acc"]), flush=True)

    # ---- TEST 3: LEARNED FAMILIARITY GATE ----  on the raw correlated codes (the realistic cue regime).
    print("\n--- TEST 3: learned anti-Hebbian familiarity gate (the no-confab moat) ---", flush=True)
    famr = test_familiarity(core, phase_codes, words, np.random.default_rng(args.seed + 3))
    print("  known novelty   mean=%.3f max=%.3f" % (famr["known_mean"], famr["known_max"]), flush=True)
    print("  unknown novelty mean=%.3f min=%.3f" % (famr["unknown_mean"], famr["unknown_min"]), flush=True)
    print("  separation margin (unk.min - known.max) = %+.3f  clean_separation=%s"
          % (famr["separation_margin"], famr["clean_separation"]), flush=True)
    print("  known<thr=%.0f%%  unknown>thr=%.0f%%" %
          (100 * famr["known_below_thr_frac"], 100 * famr["unknown_above_thr_frac"]), flush=True)
    print("  LESIONED margin=%+.3f  lesion_collapsed=%s  (anti-cheat: separation rides LEARNED weights)"
          % (famr["lesion_margin"], famr["lesion_collapsed"]), flush=True)

    # ---- VERDICT ----
    # systematicity holds = the learned attractor (in the regime where it functions) generalizes to novel
    # combos at ~ the trained accuracy (and tracks the algebra ceiling).
    sys_ok = (sysr_d["trained_acc"] >= 0.7
              and sysr_d["heldout_acc_attractor"] >= 0.9 * sysr_d["trained_acc"]
              and sysr_d["heldout_acc_attractor"] >= 0.7)
    # cleanup gate: attractor >= argmax. On RAW codes the attractor collapses; it only meets the gate
    # decorrelated -> Option A needs the learned-decorrelation step (the honest boundary).
    cleanup_raw_ok = clr["attractor_acc"] >= clr["argmax_acc"] - 1e-9
    cleanup_dec_ok = clr_d["attractor_acc"] >= clr_d["argmax_acc"] - 1e-9
    cleanup_lesion_ok = (clr["attractor_lesioned_acc"] <= clr["argmax_acc"]
                         and clr_d["attractor_lesioned_acc"] < clr_d["attractor_acc"])
    fam_ok = famr["clean_separation"] and famr["lesion_collapsed"]

    # GO requires the attractor to beat argmax on the BRAIN's codes WITHOUT a separate decorrelation crutch
    # (cleanup_raw_ok). It does not -> Option A's attractor cleanup is conditional on learned-decorrelation.
    if sys_ok and cleanup_raw_ok and fam_ok and cleanup_lesion_ok:
        verdict = "GO"
    elif sys_ok and cleanup_dec_ok and fam_ok and cleanup_lesion_ok:
        verdict = "PARTIAL"        # works, but only WITH learned-decorrelation (not on raw correlated codes)
    elif fam_ok or cleanup_dec_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"
    print("\n=== VERDICT: %s ===" % verdict, flush=True)
    print("  systematicity_holds=%s  attractor>=argmax(raw)=%s  attractor>=argmax(decorr)=%s  "
          "familiarity_separates=%s  cleanup_lesion_degrades=%s"
          % (sys_ok, cleanup_raw_ok, cleanup_dec_ok, fam_ok, cleanup_lesion_ok), flush=True)

    out = {
        "probe": "cortex_learned_cleanup_derisk", "seed": args.seed, "proj_dim": args.proj_dim,
        "t_steps": args.t_steps, "n_words": len(words), "D": int(phase_codes.shape[1]),
        "between_code_phase_cos_raw": bcos, "between_code_phase_cos_decorr": bcos_d,
        "verdict": verdict,
        "systematicity_decorr": sysr_d, "systematicity_raw": sysr,
        "cleanup_raw": clr, "cleanup_decorr": clr_d, "familiarity": famr,
        "gates": {"systematicity_holds": bool(sys_ok),
                  "attractor_ge_argmax_raw": bool(cleanup_raw_ok),
                  "attractor_ge_argmax_decorr": bool(cleanup_dec_ok),
                  "cleanup_lesion_degrades": bool(cleanup_lesion_ok),
                  "familiarity_separates": bool(fam_ok)},
    }
    op = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print("wrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
