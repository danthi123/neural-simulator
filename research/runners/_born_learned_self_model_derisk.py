"""Lane A · SELF-SCHEMA (BORN adoption from the landscape survey CLOSEST-1) — a LEARNED bodily SELF-MODEL: an
efference->predicted-feedback FORWARD MODEL learned by synapses (Hebbian/Oja co-fire), plus a NEURAL comparator
(reafference cancellation) that emits a self/other AGENCY signal from PREDICTED-vs-ACTUAL feedback. The
"mirror-test" correlate: the brain recognises self-generated (predictable) sensory consequences vs
externally-generated (unpredicted) ones.

WHY THIS IS NOT ALREADY BUILT (re-anchor). Our self-schema lane already has (i) DR-3 self-schema region (reads its
own attention/confidence/authorship internal state) and (ii) the AGENCY / AUTHORSHIP tag GO (2026-08-01): a
FIXED-STRUCTURE corollary-discharge comparator that reads "was an efference-copy MARKER present?". Both LACK a
LEARNED forward model. The authorship-tag's own honest scope names "self-organized/learned wiring" as the un-built
follow-on. The genuine BORN step, built here: the forward model is LEARNED (action_k->predicted_k grown by co-fire,
NOT hand-wired), and agency = "does the ACTUAL feedback match what my forward model PREDICTED this action would
cause?" — NOT "was a marker present?". This is decisive on the DECOUPLED case (efference IS present but the world's
feedback is wrong): a marker/presence detector says SELF (wrong); the forward-model comparator says LOW agency
(right) — the Blakemore-Frith perturbed-reafference result, the true agency computation.

BIOLOGY (brain-based, non-negotiable):
  * FORWARD MODEL (Wolpert-Miall-Kawato internal model; von Holst-Mittelstaedt reafference; Sperry efference copy):
    action (efference) -> predicted sensory feedback, LEARNED by synapses. During development the body emits an
    action and the reafferent sensory feedback arrives; co-firing (Hebb/Oja) binds action_k -> predicted_k. After
    learning, the efference copy alone predicts the feedback.
  * COMPARATOR = REAFFERENCE CANCELLATION (Blakemore-Wolpert-Frith 1998; Frith 1992; the "why you can't tickle
    yourself" circuit): the PREDICTED feedback INHIBITS the actual sensory RESPONSE via a fast interneuron. When
    the prediction matches the actual feedback (same identity, self-caused) the response is CANCELLED (attenuated
    reafference) -> the residual response is LOW -> HIGH agency. No prediction (external) or a wrong prediction
    (decoupled) -> the response is UN-cancelled -> HIGH residual -> LOW agency. The agency read-out is a NEURAL
    population rate (residual sensory response), NOT a host abs(predicted-actual). [This cancellation form is used
    rather than a coincidence AND-gate because the synaptically-driven `predicted` pool fires as SYNCHRONOUS
    volleys — which a fast FS interneuron reads cleanly — while the tonically-driven `sensory` pool fires
    ASYNCHRONOUSLY; a symmetric summation AND-gate saturates on the predicted volley alone. Measured, see finding.]

MECHANISM (ONE numpy Izhikevich SimulationBridge, reuse-by-import, NO `sim/` edit; additive/default-off):
  Per-identity regions (K identities) so the topographic comparator wiring is clean:
    action_k, predicted_k, sensory_k, resp_k (sensory response), rinh_k (FS cancellation interneuron).
  * FORWARD MODEL: action_i -> predicted_j for ALL (i,j) (all-to-all across identities), Hebbian-plastic, ZERO
    init, Oja's rule (input-DEPENDENT fixed point -> selective), plasticity_gate "fwd". Learning selects the
    diagonal action_k->predicted_k from co-fire; a random/unlearned model keeps the off-diagonal (learning anti-cheat).
  * COMPARATOR (FIXED-structure, banked scope, as the affect/authorship GOs banked a hand-wired attractor):
    sensory_k -> resp_k (excite the response); predicted_k -> rinh_k -> resp_k (gaba, cancel the response).
    Agency = LOW residual resp. Readout `agency := -resp_rate` (higher = more self).
  * BODY/WORLD boundary (legit host): the action (efference) the body emits and the actual sensory feedback the
    world returns are external drive. Everything between — forward model + comparator + agency — is neurons/synapses.

CONDITIONS (per identity, REPS each, OU noise so rates vary -> AUC is meaningful):
  * SELF (self-caused, predictable):  drive action_k + sensory_k  -> predicted_k cancels resp_k -> LOW resp -> HIGH agency
  * EXTERNAL (not self-caused):        drive sensory_k only        -> no prediction, resp_k un-cancelled -> LOW agency
  * DECOUPLED (self action, wrong fb): drive action_k + sensory_j  -> predicted_k cancels the EMPTY resp_k; resp_j
                                       (sensory_j) un-cancelled -> HIGH resp -> LOW agency (despite efference present)

HEAD-TO-HEAD:
  (A) vs our EXISTING foundation (a PRESENCE detector: agency := was efference present?): SELF & DECOUPLED both have
      efference -> presence says SELF for BOTH -> it CANNOT separate self-predictable from decoupled. The learned
      forward-model comparator CAN. Gate: FM discriminates SELF-vs-DECOUPLED >> presence does. This is the concrete
      capability the LEARNED forward model adds that the foundation lacked.
  (B) vs anti-cheats below.

ANTI-CHEATS (all wired + INVOKED):
  (a) SELF-vs-OTHER CONTINGENCY. Identical sensory CONTENT (sensory_k) as SELF (action_k present) vs EXTERNAL (no
      action) MUST get different agency (else a sensory detector, not a self-model). And DECOUPLED (efference present,
      wrong feedback) MUST read LOW (agency requires prediction-match, not action-presence).
  (b) LEARNING-REQUIRED. A RANDOM-weights forward model (skip training; randomise action->predicted) -> predicted
      from action_k is un-aligned -> the wrong resp pool is cancelled -> the self response stays un-cancelled ->
      agency discrimination collapses to chance. Proves the LEARNED synaptic mapping (not the fixed comparator
      structure) carries the discrimination.
  (c) NEURAL COMPARATOR. Agency = residual `resp` POPULATION rate (a cancellation circuit), NOT a host
      abs(predicted-actual) formula.
  (d) 6-seed for any generalization claim (this file's --smoke is 1 seed = RUNS + arms live; the parent runs 6-seed).

Run (smoke):  SIM_BACKEND=numpy PYTHONPATH=$PWD python -u -m research.runners._born_learned_self_model_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy PYTHONPATH=$PWD python -u -m research.runners._born_learned_self_model_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  (passthrough on numpy)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_born_learned_self_model_6seed.json"

# ---- operating-point constants -----------------------------------------------------------------------------
K = 4                    # distinct identities (action/feedback pairs)
N_ACT = 40               # action (efference) pool per identity
N_PRED = 40              # predicted-feedback pool per identity
N_SENS = 40              # actual sensory-feedback pool per identity
N_RESP = 30              # residual sensory-response pool per identity (the agency read-out; low = self)
N_RINH = 15              # cancellation interneuron (FS) per identity
N_PWTA = 20              # SHARED lateral-inhibition pool for the predicted layer (competitive normalization)

W_SENS_RESP = float(os.environ.get("BORN_W_SR", 40.0))  # sensory_k -> resp_k (the response to be cancelled)
W_PRED_INH = float(os.environ.get("BORN_W_PI", 30.0))   # predicted_k -> rinh_k (prediction drives the canceller)
W_INH_RESP = float(os.environ.get("BORN_W_IR", 110.0))  # rinh_k -> resp_k (gaba; cancels the response)
# predicted-layer lateral inhibition (WTA / normalization): only the strongly-driven DIAGONAL prediction fires, so
# the weak OFF-diagonal FM leak (measured: decoupled resp 0.036 < external 0.041 without it) cannot fire predicted_j
# -> the decoupled sensory response stays UN-cancelled. This is the competitive companion process real cortex runs.
W_PRED_WTA = float(os.environ.get("BORN_W_PW", 18.0))   # predicted_k -> pwta (drive the shared inhibitor)
W_WTA_PRED = float(os.environ.get("BORN_W_WP", 6.0))    # pwta -> predicted_k (gaba; lateral suppression)
W_FM_MAX = 45.0          # Hebbian cap; Oja sets w* below it from the input correlation
W_FM_RANDOM = 12.0       # per-synapse magnitude of the RANDOM (unlearned) forward model control
W_FM_PERMUTED = 30.0     # per-synapse magnitude of the PERMUTED-SELECTIVE control on its mis-mapped diagonal
                         # (~the learned Oja diagonal ~30, so it is SELECTIVE like the learned model but routes
                         # action_i -> predicted_{perm[i]} for a derangement perm -> the decisive "is it the
                         # LEARNED MAPPING, not just any selective structure?" anti-cheat the parent added)

DRIVE_ACT_PA = 2600.0    # external drive to an action pool during an utterance (no-recurrence pool -> strong)
DRIVE_SENS_PA = 2600.0   # external drive to the actual sensory-feedback pool (the world's reafference)
DRIVE_TEACH_PA = 2600.0  # reafferent teacher drive to predicted_k DURING TRAINING ONLY (gated off at test)
OU_PA = 3.0              # OU noise (rate jitter for AUC; sub-dominant to the pool drive)

TRAIN_CYCLES = int(os.environ.get("BORN_CYC", 40))  # co-fire training passes over all K identities
TRAIN_ON = 25            # steps action_k + teacher(predicted_k) co-active per cycle
TRAIN_OFF = 10           # zero-input settle per cycle
TRIAL_STEPS = int(os.environ.get("BORN_TS", 40))   # test window (dt=1ms). Long windows accumulate post-inhibitory
                                                    # rebound in resp under strong gaba -> keep moderate (measured).
FLUSH_STEPS = 60         # inter-trial zero-input gap (drain delayed spikes; cf. authorship-tag runner)
REPS = int(os.environ.get("BORN_REPS", 12))         # repetitions per (identity x condition) test cell


# ===========================================================================================================
# The bodily self-model brain: efference -> LEARNED forward model -> reafference-cancellation agency comparator.
# ===========================================================================================================
class BodilySelfModel:
    def __init__(self, seed, forward="learned"):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway

        assert forward in ("learned", "random", "permuted")
        self.seed = int(seed)
        self.forward = forward
        # PERMUTED-SELECTIVE control: a derangement (no fixed points) so the forward model is selective but
        # MIS-MAPPED (action_i -> predicted_{perm[i]}, perm[i] != i). Tests that the CORRECT learned mapping,
        # not merely selective forward structure, carries the agency signal.
        self.perm = None
        if forward == "permuted":
            prng = np.random.default_rng(int(seed) + 4242)
            perm = np.arange(K)
            while True:
                prng.shuffle(perm)
                if not np.any(perm == np.arange(K)):
                    break
            self.perm = perm.copy()

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_neuromodulator_subsystem = False
        cfg.enable_nmda = False
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)                 # SEEDS THE SUBSTRATE (NOT actual_seed_used — the CLAUDE.md gotcha)
        cfg.enable_stdp = False              # STDP eligibility never applies w/o reward -> weight stays 0 (per _D note)
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = True   # the forward model is grown by Hebbian co-fire (LEARNED by synapses)
        cfg.hebbian_learning_rate = 0.05
        cfg.hebbian_min_weight = 0.0
        cfg.hebbian_max_weight = float(W_FM_MAX)
        # Oja's rule (config.py:408): fixed point w* = <a x_j>/<a^2> = the INPUT CORRELATION (input-DEPENDENT), so
        # the diagonal action_k->predicted_k (correlated at training) grows and the off-diagonal (uncorrelated)
        # does NOT — WITHOUT it the (w_max-w) rule's fixed point is w_max for EVERY gated synapse (input-independent)
        # and the off-diagonal drifts to the cap => no selectivity (measured: diag 44.6 / off 31.8 -> Oja 30.2 / 4.5).
        cfg.hebbian_oja = 1.0
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(OU_PA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        cfg.fast_spike_reset = True

        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
        FS = "IZH2007_FS_CORTICAL_INTERNEURON"

        def exc(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        def fs(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=FS)

        regions = [fs("pwta", N_PWTA)]  # shared predicted-layer lateral-inhibition pool
        for k in range(K):
            regions += [exc(f"action_{k}", N_ACT), exc(f"predicted_{k}", N_PRED),
                        exc(f"sensory_{k}", N_SENS), exc(f"resp_{k}", N_RESP), fs(f"rinh_{k}", N_RINH)]

        pathways = []
        # FORWARD MODEL: action_i -> predicted_j for ALL (i,j), Hebbian(Oja)-plastic, ZERO init. Learning selects
        # the diagonal from co-fire; a random/unlearned model keeps the off-diagonal (the learning anti-cheat).
        for i in range(K):
            for j in range(K):
                if forward == "random":
                    w0 = float(W_FM_RANDOM)
                elif forward == "permuted":
                    w0 = float(W_FM_PERMUTED) if j == int(self.perm[i]) else 0.0
                else:
                    w0 = 0.0
                pathways.append(RegionPathway(
                    from_region=f"action_{i}", to_region=f"predicted_{j}", density=1.0,
                    weight_mean=w0, weight_jitter=0.0,
                    plastic=(forward == "learned"),
                    plasticity_gate=("fwd" if forward == "learned" else None)))
        # COMPARATOR (FIXED): reafference cancellation. sensory_k excites its response; predicted_k drives a fast
        # interneuron that cancels that response -> a matched prediction attenuates the (self-caused) reafference.
        for k in range(K):
            pathways.append(RegionPathway(from_region=f"sensory_{k}", to_region=f"resp_{k}", density=0.8,
                                          weight_mean=W_SENS_RESP, weight_jitter=0.1, plastic=False))
            pathways.append(RegionPathway(from_region=f"predicted_{k}", to_region=f"rinh_{k}", density=0.8,
                                          weight_mean=W_PRED_INH, weight_jitter=0.1, plastic=False))
            pathways.append(RegionPathway(from_region=f"rinh_{k}", to_region=f"resp_{k}", density=0.8,
                                          weight_mean=W_INH_RESP, weight_jitter=0.1, plastic=False, receptor="gaba_a"))
            # predicted-layer lateral inhibition (competitive normalization): each predicted pool drives the shared
            # pwta pool, which inhibits all predicted pools -> the weak off-diagonal FM leak stays sub-threshold.
            pathways.append(RegionPathway(from_region=f"predicted_{k}", to_region="pwta", density=0.7,
                                          weight_mean=W_PRED_WTA, weight_jitter=0.1, plastic=False))
            pathways.append(RegionPathway(from_region="pwta", to_region=f"predicted_{k}", density=0.7,
                                          weight_mean=W_WTA_PRED, weight_jitter=0.1, plastic=False, receptor="gaba_a"))

        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        rm = self._bridge.region_manager
        self.idx = {f"{grp}_{k}": np.asarray(rm.indices(f"{grp}_{k}"), dtype=np.int64)
                    for grp in ("action", "predicted", "sensory", "resp", "rinh") for k in range(K)}

    # -- drive helpers -------------------------------------------------------------------------------------
    def _zero(self):
        self._bridge.cp_external_input_current[:] = 0.0

    def _add(self, key, pA):
        self._bridge.cp_external_input_current[self.idx[key]] = np.float32(pA)

    def _flush(self, n):
        for _ in range(int(n)):
            self._zero()
            self._bridge._run_one_simulation_step()

    # -- training: grow action_k -> predicted_k by co-fire (efference + reafferent teacher) ----------------
    def train(self):
        if self.forward != "learned":
            return
        b = self._bridge
        try:
            b.set_plasticity_gate("fwd", 1.0)
        except KeyError:
            pass
        rng_order = np.random.default_rng(self.seed + 777)
        for _ in range(TRAIN_CYCLES):
            ks = list(range(K)); rng_order.shuffle(ks)
            for k in ks:
                for _ in range(TRAIN_ON):
                    self._zero()
                    self._add(f"action_{k}", DRIVE_ACT_PA)      # efference
                    self._add(f"predicted_{k}", DRIVE_TEACH_PA)  # reafferent teacher (world feedback teaches FM)
                    b._run_one_simulation_step()
                self._flush(TRAIN_OFF)
        try:
            b.set_plasticity_gate("fwd", 0.0)                  # FREEZE the forward model before test
        except KeyError:
            pass
        self._flush(FLUSH_STEPS)

    # -- learned forward-model weight matrix (action_i -> predicted_j mean synapse) -------------------------
    def fm_matrix(self):
        M = to_host(self._bridge.cp_connections)
        try:
            M = M.toarray()
        except AttributeError:
            M = np.asarray(M)
        W = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                sub = M[np.ix_(self.idx[f"action_{i}"], self.idx[f"predicted_{j}"])]  # rows=pre=action, cols=post=predicted
                W[i, j] = float(np.abs(sub).mean())
        return W

    # -- one test trial. condition in {'self','external','decoupled'}. returns per-pool rates ---------------
    def trial(self, k, condition, j_decouple=None):
        b = self._bridge
        self._flush(FLUSH_STEPS)
        resp = action = 0.0
        sens_k = k if condition != "decoupled" else j_decouple  # which sensory identity actually arrives
        for _ in range(TRIAL_STEPS):
            self._zero()
            if condition in ("self", "decoupled"):
                self._add(f"action_{k}", DRIVE_ACT_PA)          # efference present
            self._add(f"sensory_{sens_k}", DRIVE_SENS_PA)       # the world's actual feedback
            b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            resp += float(sum(fs[self.idx[f"resp_{m}"]].sum() for m in range(K))) / (K * N_RESP)
            action += float(sum(fs[self.idx[f"action_{m}"]].sum() for m in range(K))) / (K * N_ACT)
        # residual response rate (surprise). agency = -resp (higher = more self-caused).
        return {"resp": resp / TRIAL_STEPS, "action": action / TRIAL_STEPS}


# ===========================================================================================================
# Protocol + metrics
# ===========================================================================================================
def _auc(pos, neg):
    """Rank-based AUC of separating pos (label 1) from neg (label 0). 0.5 = chance."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def _run_conditions(brain, rng):
    """Balanced (identity x condition x REPS). Returns per-condition agency (=-resp) + residual-resp + action rate."""
    agency = {"self": [], "external": [], "decoupled": []}
    resp = {"self": [], "external": [], "decoupled": []}
    action_rate = {"self": [], "external": [], "decoupled": []}
    order = []
    for k in range(K):
        for _ in range(REPS):
            order.append(("self", k, None))
            order.append(("external", k, None))
            j = (k + 1 + int(rng.integers(0, K - 1))) % K  # a wrong feedback identity (!= k)
            order.append(("decoupled", k, int(j)))
    rng.shuffle(order)
    for cond, k, j in order:
        r = brain.trial(k, cond, j_decouple=j)
        agency[cond].append(-r["resp"])   # agency: HIGHER (less-negative) = self-caused (cancelled response)
        resp[cond].append(r["resp"])
        action_rate[cond].append(r["action"])
    return agency, resp, action_rate


def run_seed(seed):
    rng = np.random.default_rng(seed)

    # --- LEARNED forward model ---
    brain = BodilySelfModel(seed, forward="learned")
    brain.train()
    W = brain.fm_matrix()
    diag = float(np.mean(np.diag(W)))
    offdiag = float((W.sum() - np.trace(W)) / (K * K - K))
    agency, resp, action_rate = _run_conditions(brain, rng)

    resp_mean = {c: float(np.mean(v)) for c, v in resp.items()}
    # discrimination: SELF (self-caused) vs {EXTERNAL, DECOUPLED} (not self-caused) by AGENCY (=-resp)
    auc_learned = _auc(agency["self"], agency["external"] + agency["decoupled"])
    # the DECISIVE head-to-head axis: SELF vs DECOUPLED (both have efference present)
    auc_self_vs_dec = _auc(agency["self"], agency["decoupled"])
    # PRESENCE detector (our existing-foundation analogue: agency := efference present) on SELF vs DECOUPLED
    auc_presence_self_vs_dec = _auc(action_rate["self"], action_rate["decoupled"])
    # contingency: identical content (sensory_k) as self vs external must differ
    auc_self_vs_ext = _auc(agency["self"], agency["external"])

    # --- RANDOM forward model (learning-required anti-cheat) ---
    brain_r = BodilySelfModel(seed, forward="random")   # no training; fixed random action->predicted
    agency_r, resp_r, _ = _run_conditions(brain_r, rng)
    auc_random = _auc(agency_r["self"], agency_r["external"] + agency_r["decoupled"])
    auc_random_self_vs_dec = _auc(agency_r["self"], agency_r["decoupled"])

    # --- PERMUTED-SELECTIVE forward model (mapping-specific anti-cheat: selective but MIS-mapped) ---
    brain_p = BodilySelfModel(seed, forward="permuted")  # fixed derangement action_i -> predicted_{perm[i]}
    agency_p, resp_p, _ = _run_conditions(brain_p, rng)
    auc_permuted = _auc(agency_p["self"], agency_p["external"] + agency_p["decoupled"])
    auc_permuted_self_vs_dec = _auc(agency_p["self"], agency_p["decoupled"])

    return {
        "seed": int(seed),
        "fm_diag": round(diag, 3), "fm_offdiag": round(offdiag, 3),
        "fm_diag_ratio": round(diag / (offdiag + 1e-9), 3),
        "resp_self": round(resp_mean["self"], 4), "resp_external": round(resp_mean["external"], 4),
        "resp_decoupled": round(resp_mean["decoupled"], 4),
        "agency_self": round(float(np.mean(agency["self"])), 4),
        "agency_external": round(float(np.mean(agency["external"])), 4),
        "agency_decoupled": round(float(np.mean(agency["decoupled"])), 4),
        "auc_learned": round(auc_learned, 4),
        "auc_self_vs_dec": round(auc_self_vs_dec, 4),
        "auc_presence_self_vs_dec": round(auc_presence_self_vs_dec, 4),
        "auc_self_vs_ext": round(auc_self_vs_ext, 4),
        "resp_random_self": round(float(np.mean(resp_r["self"])), 4),
        "resp_random_decoupled": round(float(np.mean(resp_r["decoupled"])), 4),
        "auc_random": round(auc_random, 4),
        "auc_random_self_vs_dec": round(auc_random_self_vs_dec, 4),
        "resp_permuted_self": round(float(np.mean(resp_p["self"])), 4),
        "resp_permuted_decoupled": round(float(np.mean(resp_p["decoupled"])), 4),
        "auc_permuted": round(auc_permuted, 4),
        "auc_permuted_self_vs_dec": round(auc_permuted_self_vs_dec, 4),
    }


def _aggregate_verdict(rows):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    diag, off = m("fm_diag"), m("fm_offdiag")
    r_self, r_ext, r_dec = m("resp_self"), m("resp_external"), m("resp_decoupled")
    auc_learned = m("auc_learned")
    auc_svd, auc_pres_svd = m("auc_self_vs_dec"), m("auc_presence_self_vs_dec")
    auc_sve = m("auc_self_vs_ext")
    auc_random = m("auc_random")
    auc_random_svd = m("auc_random_self_vs_dec")
    auc_permuted_svd = m("auc_permuted_self_vs_dec")

    checks = {
        # forward model actually LEARNED the diagonal action_k->predicted_k mapping (Oja selective)
        "forward_model_learned(diag>>offdiag)": diag >= off + 3.0,
        # (c) neural comparator does work: the prediction CANCELS the self response (resp_self << resp_ext)
        "comparator_cancels_on_self(ext-self>=0.01)": (r_ext - r_self) >= 0.01 and r_ext >= 0.02,
        # agency discrimination: SELF vs {external, decoupled}
        "agency_discriminates(auc>=0.90)": auc_learned >= 0.90,
        # (a) contingency: same content (sensory_k) self vs external -> different agency
        "contingency_self_vs_external(auc>=0.85)": auc_sve >= 0.85,
        # (a) decoupled decisive: efference present but wrong feedback -> LOW agency (self response cancelled, dec not)
        "decoupled_is_low(self<<dec, auc>=0.85)": auc_svd >= 0.85 and r_dec >= r_self + 0.01,
        # HEAD-TO-HEAD: the learned FM beats a presence/marker detector on SELF-vs-DECOUPLED (the foundation's limit)
        "beats_presence_on_decoupled(FM>>presence)": auc_svd >= auc_pres_svd + 0.25 and auc_pres_svd <= 0.65,
        # (b) learning-required, on the DECISIVE self-vs-decoupled axis: a RANDOM (all-to-all) forward model predicts
        # EVERY identity -> cancels indiscriminately -> it behaves like a PRESENCE detector (it can nail external,
        # which is why its POOLED auc is not chance, but it CANNOT do decoupled). So gate on self-vs-decoupled, where
        # only a LEARNED identity-specific prediction can separate them.
        "learning_required(random_fails_decoupled, <=0.65)": auc_random_svd <= 0.65 and auc_svd >= auc_random_svd + 0.25,
        # MAPPING-SPECIFIC (parent-added): a SELECTIVE-but-MIS-mapped forward model (derangement action_i->pred_{perm[i]})
        # must ALSO fail self-vs-decoupled -> proves it is the CORRECT learned mapping, not merely selective forward
        # structure + the cancellation/lateral-inhibition machinery, that carries the agency signal.
        "mapping_specific(permuted_fails_decoupled, <=0.65)": auc_permuted_svd <= 0.65 and auc_svd >= auc_permuted_svd + 0.25,
    }
    go = all(checks.values())
    means = {"fm_diag": diag, "fm_offdiag": off, "resp_self": r_self, "resp_external": r_ext,
             "resp_decoupled": r_dec, "auc_learned": auc_learned, "auc_self_vs_dec": auc_svd,
             "auc_presence_self_vs_dec": auc_pres_svd, "auc_self_vs_ext": auc_sve, "auc_random": auc_random,
             "auc_random_self_vs_dec": auc_random_svd, "auc_permuted_self_vs_dec": auc_permuted_svd}

    v = Verdict("learned bodily self-model: forward-model agency discrimination", chance=0.5)
    v.floor("agency AUC (self vs not-self) beats chance", measured=auc_learned, floor=0.5)
    v.control("learned vs RANDOM forward model on the decisive self-vs-decoupled axis", treatment=auc_svd,
              control=auc_random_svd, min_separation=0.25)
    v.control("forward-model comparator vs presence detector (self-vs-decoupled)", treatment=auc_svd,
              control=auc_pres_svd, min_separation=0.25)
    v.control("CORRECT learned mapping vs a PERMUTED-selective (mis-mapped) forward model on self-vs-decoupled",
              treatment=auc_svd, control=auc_permuted_svd, min_separation=0.25)
    v.reaches("forward model learned the diagonal (mean |w| action_k->predicted_k >> off-diagonal)",
              before=off, after=diag)
    v.reaches("prediction CANCELS the self reafference (resp_ext -> resp_self)", before=r_ext, after=r_self)
    v.require("comparator cancels the self response (resp_ext - resp_self >= 0.01)", r_ext - r_self,
              expect=lambda x: x >= 0.01)
    v.require("decoupled reads LOW despite efference present (resp_dec >= resp_self + 0.01)", r_dec - r_self,
              expect=lambda x: x >= 0.01)
    v.require("contingency: same content self-vs-external AUC >= 0.85", auc_sve, expect=lambda x: x >= 0.85)
    v.require("presence detector CANNOT do self-vs-decoupled (AUC <= 0.65)", auc_pres_svd,
              expect=lambda x: x <= 0.65)
    v.disabled("STDP / reward-mod / homeostasis / short-term & structural plasticity",
               why="the forward model is grown by Hebbian(Oja) co-fire; the comparator is a fixed-structure neural "
                   "reafference-cancellation circuit (the banked scope, as the affect/authorship GOs banked a "
                   "hand-wired attractor)")
    decided = v.decide(go=go, verbose=False)

    attributable_to("the LEARNED forward model (vs a random one) on decoupled", auc_svd, auc_random_svd)
    attributable_to("prediction-match (vs efference-presence) on decoupled", auc_svd, auc_pres_svd)
    attributable_to("the CORRECT learned mapping (vs a permuted-selective one) on decoupled", auc_svd, auc_permuted_svd)
    return go, checks, means, decided


def _diag(seed):
    """Operating-point probe: after learning, report per-pool firing + the cancellation levels."""
    b = BodilySelfModel(seed, forward="learned")
    b.train()
    W = b.fm_matrix()
    print(f"  [diag seed {seed}] W_SR={W_SENS_RESP} W_PI={W_PRED_INH} W_IR={W_INH_RESP} | "
          f"FM diag={np.mean(np.diag(W)):.3f} off={(W.sum()-np.trace(W))/(K*K-K):.3f}", flush=True)

    def probe(drive):
        b._flush(FLUSH_STEPS)
        acc = {g: 0.0 for g in ("resp", "predicted", "sensory", "action", "rinh")}
        npr = {"resp": N_RESP, "predicted": N_PRED, "sensory": N_SENS, "action": N_ACT, "rinh": N_RINH}
        for _ in range(TRIAL_STEPS):
            b._zero()
            for key, pa in drive:
                b._add(key, pa)
            b._bridge._run_one_simulation_step()
            fs = to_host(b._bridge.cp_firing_states)
            for g in acc:
                acc[g] += float(sum(fs[b.idx[f"{g}_{m}"]].sum() for m in range(K))) / (K * npr[g])
        return {g: v / TRIAL_STEPS for g, v in acc.items()}
    for label, drive in [
        ("EXTERNAL s0 (agency LOW->resp HIGH)", [('sensory_0', DRIVE_SENS_PA)]),
        ("SELF a0+s0 (agency HIGH->resp LOW) ", [('action_0', DRIVE_ACT_PA), ('sensory_0', DRIVE_SENS_PA)]),
        ("DECOUP a0+s1 (agency LOW->resp HIGH)", [('action_0', DRIVE_ACT_PA), ('sensory_1', DRIVE_SENS_PA)]),
    ]:
        r = probe(drive)
        print(f"    {label} -> resp={r['resp']:.3f} | rinh={r['rinh']:.3f} pred={r['predicted']:.3f} "
              f"sens={r['sensory']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed — proves it RUNS + every arm is live")
    ap.add_argument("--diag", action="store_true", help="operating-point probe (tune weights)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if a.diag:
        _diag(a.seeds[0])
        return 0
    if a.smoke:
        a.seeds = [a.seeds[0]]

    t0 = time.time()
    print(f"[born self-model] LEARNED forward model (action->predicted, Hebbian/Oja) + reafference-CANCELLATION "
          f"agency comparator. seeds={a.seeds} smoke={a.smoke}\n"
          f"  GATE: agency AUC(self vs not-self)>=0.90; decoupled LOW; beats presence on self-vs-dec; "
          f"random-FM fails; FM diagonal learned.", flush=True)
    rows = [run_seed(s) for s in a.seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] FM diag/off {r['fm_diag']:.2f}/{r['fm_offdiag']:.2f} (x{r['fm_diag_ratio']:.1f}) "
              f"|| resp self/ext/dec {r['resp_self']:.3f}/{r['resp_external']:.3f}/{r['resp_decoupled']:.3f} "
              f"|| AUC learned {r['auc_learned']:.3f} | self-vs-dec FM {r['auc_self_vs_dec']:.3f} vs presence "
              f"{r['auc_presence_self_vs_dec']:.3f} | self-vs-ext {r['auc_self_vs_ext']:.3f} || RANDOM auc "
              f"{r['auc_random']:.3f}", flush=True)

    go, checks, means, decided = _aggregate_verdict(rows)
    n = len(a.seeds)
    if go:
        verdict = (
            f"GO ({n}-seed) — LEARNED BODILY SELF-MODEL: a Hebb/Oja-learned forward model (action->predicted "
            f"feedback, diagonal {means['fm_diag']:.1f} >> off {means['fm_offdiag']:.1f}) + a NEURAL "
            f"reafference-cancellation comparator emit an AGENCY signal that discriminates self-caused from "
            f"not-self-caused feedback at AUC={means['auc_learned']:.3f} (chance 0.5). The prediction CANCELS the "
            f"self reafference (resp_self {means['resp_self']:.3f} << resp_ext {means['resp_external']:.3f}). "
            f"DECISIVE: on DECOUPLED trials (efference PRESENT but the world's feedback is WRONG) agency reads LOW "
            f"(resp_dec {means['resp_decoupled']:.3f}; self-vs-dec AUC {means['auc_self_vs_dec']:.3f}) while a "
            f"PRESENCE/marker detector — our existing foundation — CANNOT tell them apart "
            f"({means['auc_presence_self_vs_dec']:.3f}): the LEARNED forward model adds agency-discrimination the "
            f"foundation lacked. Contingency holds (same content self-vs-external AUC {means['auc_self_vs_ext']:.3f}); "
            f"learning is REQUIRED (a random all-to-all forward model predicts EVERY identity -> cancels "
            f"indiscriminately -> behaves like the presence detector: on the decisive self-vs-decoupled axis it "
            f"collapses to {means['auc_random_self_vs_dec']:.3f}). Brain-based "
            f"(synaptic forward model + neural cancellation comparator; NO host abs()); numpy-CPU; NO sim/ edit. "
            f"SMOKE-level — needs the parent's 6-seed for a generalization claim.")
    else:
        miss = [k for k, v in checks.items() if not v]
        verdict = (f"BOUNDARY (build-informative, {n}-seed) — agency AUC {means['auc_learned']:.3f}; self-vs-dec FM "
                   f"{means['auc_self_vs_dec']:.3f} vs presence {means['auc_presence_self_vs_dec']:.3f}; random-FM "
                   f"{means['auc_random']:.3f}; resp self/ext/dec {means['resp_self']:.3f}/{means['resp_external']:.3f}/"
                   f"{means['resp_decoupled']:.3f}; FM diag/off {means['fm_diag']:.2f}/{means['fm_offdiag']:.2f}. "
                   f"FAILED: {miss}. Tune the cancellation (W_SENS_RESP/W_PRED_INH/W_INH_RESP) or FM learning; the "
                   f"self-model is the next tune, not a wall.")

    summary = {
        "probe": "born_learned_self_model (Lane A self-schema; BORN CLOSEST-1 adoption)",
        "verdict": verdict, "GO": bool(go),
        "status": decided["status"], "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"], "undefined_reasons": decided["undefined_reasons"],
        "checks": checks, "means": means, "per_seed": rows,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "K": K, "N_ACT": N_ACT, "N_PRED": N_PRED, "N_SENS": N_SENS,
                   "N_RESP": N_RESP, "N_RINH": N_RINH, "N_PWTA": N_PWTA, "W_SENS_RESP": W_SENS_RESP,
                   "W_PRED_INH": W_PRED_INH, "W_INH_RESP": W_INH_RESP, "W_PRED_WTA": W_PRED_WTA,
                   "W_WTA_PRED": W_WTA_PRED, "W_FM_MAX": W_FM_MAX, "W_FM_RANDOM": W_FM_RANDOM,
                   "TRAIN_CYCLES": TRAIN_CYCLES, "TRIAL_STEPS": TRIAL_STEPS, "REPS": REPS, "OU_PA": OU_PA,
                   "hebbian_oja": 1.0, "hebbian_learning_rate": 0.05},
        "mechanism": "action_i->predicted_j all-to-all Hebbian(Oja)-plastic (zero init) grows the SELECTIVE diagonal "
                     "from co-fire (efference + reafferent teacher); fixed reafference-cancellation comparator "
                     "(sensory_k->resp_k excite; predicted_k->rinh_k->resp_k gaba cancel); agency = -resp (residual "
                     "response). self=action_k+sensory_k (predicted_k cancels resp_k -> LOW resp -> HIGH agency); "
                     "external=sensory_k only (no prediction -> HIGH resp -> LOW agency); decoupled=action_k+sensory_j "
                     "(predicted_k cancels empty resp_k; resp_j un-cancelled -> HIGH resp -> LOW agency).",
        "HONEST_NOTE": "numpy-CPU read on the real spiking Izhikevich bridge ('numpy' is the backend, not a host "
                       "shortcut). The forward model is LEARNED by synapses (Hebbian/Oja co-fire; Oja gives the "
                       "input-DEPENDENT fixed point that yields diagonal selectivity; STDP eligibility never applies "
                       "without reward on this bridge -> weight stays 0, per the _D heteroassoc note). The comparator "
                       "is a FIXED-structure neural reafference-cancellation circuit (banked scope, as the "
                       "affect/authorship GOs banked a hand-wired attractor); a self-organized comparator + a "
                       "reward-gated (three-factor) forward model are the named follow-ons. The cancellation form (vs "
                       "a coincidence AND-gate) is dictated by the substrate: the synaptically-driven `predicted` pool "
                       "fires SYNCHRONOUS volleys (which a fast FS interneuron reads) while the tonically-driven "
                       "`sensory` pool fires ASYNCHRONOUSLY -> a symmetric summation AND-gate saturates on the "
                       "predicted volley alone (measured). Biology: Wolpert-Miall-Kawato internal model; "
                       "von Holst-Mittelstaedt reafference; Sperry efference copy; Frith/Blakemore IPL-cerebellar "
                       "comparator; Hebb/Oja.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[born self-model] VERDICT: {verdict}", flush=True)
    print(f"[born self-model] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
