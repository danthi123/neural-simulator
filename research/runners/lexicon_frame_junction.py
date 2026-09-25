"""FRAME-JUNCTION referent lexicon: heard context reaches the noun/non-noun category pools only through spiking
coincidence units, one per (word before, word after) pair (language lane E, 2026-09-24; default OFF).

PRE-REGISTRATION: research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md (committed
before this file). Selected by `BRAIN_LEARNED_REFERENT_JUNCTION=1` inside
`lexicon_spiking_frame_category.get_lexicon()`; it only matters when `BRAIN_LEARNED_REFERENT_LEXICON=1` routes the
lexicon into the D6 referent parse. Unset -> this module is never imported.

WHY. The single-offset lexicon (`lexicon_spiking_frame_category.SpikingFrameCategoryLexicon`) learned its noun
evidence almost entirely from the LEFT neighbour ("the word before is *the* / *a*"; frame_proxy_s7.json: held-out
nouns +85.7 left vs -3.6 right). Any word that follows a determiner is pushed toward "noun" ('the MOST beautiful',
'a WONDERFUL day'), and closed-class words the curriculum never covered sit on the decision boundary. Infants
categorize by the FRAME, the joint (before, after) context (Mintz 2003, Cognition 90:91; Chemla et al. 2009, Dev Sci
12:396: the discontinuous two-sided frame is what makes it work). One side added to the other is a different
signal. This variant replaces the additive single-offset edge by a conjunction.

WHAT IT IS (every step between the heard corpus and the decision is neurons/synapses, as in v2):
  * ENVIRONMENT (host, unchanged): the same FrameEnvironment presentation: K sampled real occurrences of the word,
    each driving its <=4 frame afferents FR(offset, context word) for T_ON steps.
  * JUNCTION POOL FJ (new): C x C excitatory neurons. J(a, b) has exactly two input synapses, from FR(-1, a) and
    FR(+1, b), at a fixed weight W_J at which a lone afferent does not fire J and the two together do: a two-input
    threshold AND on a point neuron, the stand-in for a thin-dendrite conjunction subunit (Polsky, Mel & Schiller
    2004, Nat Neurosci 7:621). The -2/+2 afferents project nowhere in this variant.
  * LEARNED EDGE: FJ -> CN (referent pool) and FJ -> CX (non-referent pool), all-to-all, uniform-with-jitter start,
    the SAME synapse-local Oja rule and teacher curriculum as v2, with the junction RATES as the pre factor. There is
    NO FR -> CN/CX edge: every route from heard context to a category pool needs both neighbours at once.
  * COMPETITION + READ-OUT: unchanged (reciprocal FSI lateral inhibition; the host reads the winning pool with the
    v2 MIN_RATE / DEAD_MARGIN abstain rule -- "spiking with a host read-out").
Nothing names the closed class to the circuit; no word list is read.

LESIONS (`set_lesion`, each verified at measurement by the inherited weight-hash check in `decide()`):
  "learned_edge"  FJ -> CN/CX restored to the start weights (route runner R4).
  "competition"   the reciprocal inhibitory weights zeroed.
  "afferent_zero" FJ -> CN/CX zeroed (integrity smoke).
  "coincidence"   every FR -> FJ weight set to OR_LESION_FACTOR x W_J, so one afferent delivers what two did: the
                  AND becomes an OR. The learned FJ -> CN/CX weights are untouched (the G3 lesion).

CONSTANTS were set at DEV seed 7 (not an evaluation seed), by the rules of AMENDMENT 1 of the pre-registration:
  * T_ON_J = 50: the junction's first spike to a coincident pair comes 12-26 steps after onset at every tested weight,
    so no AND completes inside v2's 10-step occurrence (still ~5x shorter than a spoken word).
  * W_J, I_TONIC_J: the max-margin point of the `calibrate_and` grid (lone afferent held 150 steps -> 0 junction
    spikes; pair -> >= 1 spike within T_ON_J; on every sampled junction). and_calibration_s7.json.
  * DRIVE_MATCH_S: `measure_drive_ratio` (drive_ratio_s7.json); start weight, Oja rate and normalisation follow.

HONEST RESIDUALS (declared in the pre-registration): the junction WIRING is host-designed (exhaustive, fixed, one
unit per (-1,+1) pair over the C most-heard words), not grown by development; the AND is a somatic threshold, not a
dendritic plateau (the engine's plateau lasts ~80 ms, and its coincidence count needs same-step spikes); the
junction threshold is set by a constant hyperpolarizing current (I_TONIC_J), the stand-in for tonic inhibition;
only the immediate frame; teacher-driven curriculum; host read-out; noun-hood not referent-hood.

Calibration (dev seed 7): SIM_BACKEND=numpy python -m research.runners.lexicon_frame_junction --calibrate --seed 7 \
    --json research/findings/raw/_lexicon_closed_class/and_calibration_s7.json
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import lexicon_spiking_frame_category as L  # noqa: E402

# ── constants (dev seed 7 calibration; see the module docstring and AMENDMENT 1 of the pre-registration) ─────────
W_J = 300.0                # FR(-1,a) -> J(a,b) and FR(+1,b) -> J(a,b) weight (set by `calibrate_and`, target (i))
I_TONIC_J = -650.0         # constant hyperpolarizing current on every junction (pA): the tonic-inhibition stand-in
T_ON_J = 50                # steps each heard occurrence drives its afferents in THIS variant (v2 uses L.T_ON = 10)
OR_LESION_FACTOR = 2.0     # "coincidence" lesion: one afferent delivers what two did (fixed a priori, not calibrated)
# DRIVE MATCHING (AMENDMENT 1). Junctions fire far more sparsely than v2's frame afferents (one junction per
# occurrence, ~2 spikes per 50-step window), so v2's weight scale leaves the category pools silent after training.
# DRIVE_MATCH_S is MEASURED at dev seed 7 (`measure_drive_ratio`: v2 FR spikes/step over FJ spikes/step, mean over
# the untrained curriculum presentations; drive_ratio_s7.json). With W_J = S*W, x_J = x/S, the rescaling below maps
# the junction Oja update exactly onto v2's (dW_J = S*dW): start weight x S, rate x S^2, normalisation / S^2.
DRIVE_MATCH_S = 27.53
W_INIT_J = L.W_INIT * DRIVE_MATCH_S                   # FJ -> category start weight (uniform, jittered by L.W_JITTER)
ETA_J = L.ETA * DRIVE_MATCH_S ** 2                    # Oja rate
OJA_BETA_J = L.OJA_BETA / DRIVE_MATCH_S ** 2          # Oja normalisation strength

LESION_KINDS = L.LESION_KINDS + ("coincidence",)
_OFF_L, _OFF_R = L.OFFSETS.index(-1), L.OFFSETS.index(1)


def junction_enabled() -> bool:
    """`BRAIN_LEARNED_REFERENT_JUNCTION` in {1,true,yes,on}. DEFAULT OFF."""
    v = os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def build_junction_circuit(seed: int, n_frame: int, C: int, w_j: float = W_J, w_init: float = W_INIT_J):
    """FR (n_frame afferents) + FJ (C*C junctions) -> {CN0, CX0} with the v2 reciprocal FSI inhibition.

    The region framework builds every pathway except FR -> FJ (a RegionPathway cannot express "exactly these two
    presynaptic neurons"); the built connectivity is then re-installed verbatim together with the 2*C*C FR -> FJ
    synapses through `inject_explicit_wiring` (presynaptic polarity traits preserved)."""
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.backend import to_host
    from sim.enums import NeuronType
    from sim.regions import RegionPathway
    rs, fs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    regions = [L._region("FR", n_frame, exc_fraction=1.0, neuron_type=rs),
               L._region("FJ", C * C, exc_fraction=1.0, neuron_type=rs)]
    pathways = []
    for p in ("CN", "CX"):
        regions.append(L._region(f"{p}0", L.N_CAT, exc_fraction=1.0, neuron_type=rs))
    for p in ("IN", "IX"):
        regions.append(L._region(f"{p}0", L.N_FSI, exc_fraction=0.0, neuron_type=fs))
    for p in ("CN", "CX"):
        pathways.append(RegionPathway(from_region="FJ", to_region=f"{p}0", density=1.0, weight_mean=w_init,
                                      weight_jitter=L.W_JITTER, plastic=False))
    pathways.append(RegionPathway(from_region="CN0", to_region="IN0", density=1.0, weight_mean=L.TO_FSI_W,
                                  weight_jitter=0.05, plastic=False))
    pathways.append(RegionPathway(from_region="CX0", to_region="IX0", density=1.0, weight_mean=L.TO_FSI_W,
                                  weight_jitter=0.05, plastic=False))
    pathways.append(RegionPathway(from_region="IN0", to_region="CX0", density=1.0, weight_mean=L.CROSS_W,
                                  weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    pathways.append(RegionPathway(from_region="IX0", to_region="CN0", density=1.0, weight_mean=L.CROSS_W,
                                  weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b._initialize_simulation_data()
    rm = b.region_manager
    fr = np.asarray(list(rm.indices("FR")), dtype=np.int64)
    fj = np.asarray(list(rm.indices("FJ")), dtype=np.int64)
    coo = b.cp_connections.tocoo()
    pre0 = np.asarray(to_host(coo.row), dtype=np.int64)
    post0 = np.asarray(to_host(coo.col), dtype=np.int64)
    w0 = np.asarray(to_host(coo.data), dtype=np.float64)
    a_idx, b_idx = np.divmod(np.arange(C * C), C)                  # J(a, b) = fj[a*C + b]
    pre_l = fr[_OFF_L * C + a_idx]
    pre_r = fr[_OFF_R * C + b_idx]
    pre_j = np.concatenate([pre_l, pre_r])
    post_j = np.concatenate([fj, fj])
    plan = {
        "built": {"pre_indices": pre0.tolist(), "post_indices": post0.tolist(), "initial_weights": w0.tolist(),
                  "plastic": False, "conn_type": "region-framework pathways, re-installed verbatim"},
        "fr_fj": {"pre_indices": pre_j.tolist(), "post_indices": post_j.tolist(),
                  "initial_weights": [float(w_j)] * len(pre_j), "plastic": False,
                  "conn_type": "frame junction: FR(-1,a) and FR(+1,b) -> J(a,b)"},
    }
    b.inject_explicit_wiring(plan)
    return b


class FrameJunctionLexicon(L.SpikingFrameCategoryLexicon):
    """See the module docstring. Same public surface as v2: train / decide / classify / is_referent / set_lesion."""

    variant = "junction"

    def __init__(self, seed: int, env, n_replicas: int = 1, *, eta=ETA_J, oja_beta=OJA_BETA_J,
                 teacher_i=L.TEACHER_I, i_frame=L.I_FRAME, k_occ=L.K_OCC, t_on=T_ON_J, epochs=L.EPOCHS,
                 w_j=W_J, w_init=W_INIT_J, i_tonic=I_TONIC_J):
        from sim.backend import to_host
        if int(n_replicas) != 1:
            raise ValueError("FrameJunctionLexicon is the deployment variant: n_replicas must be 1")
        self.seed, self.env, self.R = int(seed), env, 1
        self.eta, self.beta, self.teacher_i, self.i_frame = float(eta), float(oja_beta), float(teacher_i), float(i_frame)
        self.k_occ, self.t_on, self.epochs = int(k_occ), int(t_on), int(epochs)
        self.C, self.w_j, self.i_tonic = int(env.C), float(w_j), float(i_tonic)
        self.b = build_junction_circuit(self.seed, env.n_frame, self.C, w_j=self.w_j, w_init=w_init)
        rm = self.b.region_manager
        self.fr = np.asarray(list(rm.indices("FR")), dtype=np.int64)
        self.fj = np.asarray(list(rm.indices("FJ")), dtype=np.int64)
        self.cn = [np.asarray(list(rm.indices("CN0")), dtype=np.int64)]
        self.cx = [np.asarray(list(rm.indices("CX0")), dtype=np.int64)]
        self.inn = [np.asarray(list(rm.indices("IN0")), dtype=np.int64)]
        self.ixx = [np.asarray(list(rm.indices("IX0")), dtype=np.int64)]
        self.n = int(self.b.core_config.num_neurons)
        self.post_groups = [self.cn[0], self.cx[0]]
        self.S = L._synapse_slots(self.b, self.fj, self.post_groups)             # (C*C, 2*N_CAT) learned edge
        self.S_inh = np.concatenate([L._synapse_slots(self.b, self.inn[0], [self.cx[0]]).ravel(),
                                     L._synapse_slots(self.b, self.ixx[0], [self.cn[0]]).ravel()])
        self.S_j = self._junction_input_slots()                                   # all 2*C*C FR -> FJ synapses
        data = np.asarray(to_host(self.b.cp_connections.data))
        self.W_init = data[self.S].astype(np.float64).copy()
        self.W = self.W_init.copy()
        self.inh_init = data[self.S_inh].astype(np.float64).copy()
        self.wj_init = data[self.S_j].astype(np.float64).copy()
        self.lesion = None
        self._cache = {}
        self._install()

    def _junction_input_slots(self):
        from sim.backend import to_host
        indptr = np.asarray(to_host(self.b.cp_connections.indptr))
        indices = np.asarray(to_host(self.b.cp_connections.indices))
        is_fj = np.zeros(self.n, dtype=bool)
        is_fj[self.fj] = True
        slots = []
        for p in self.fr:
            ks = np.arange(indptr[p], indptr[p + 1])
            slots.append(ks[is_fj[indices[ks]]])
        S_j = np.concatenate(slots)
        if len(S_j) != 2 * self.C * self.C:
            raise RuntimeError(f"expected {2 * self.C * self.C} FR->FJ synapses, found {len(S_j)}")
        return S_j

    # ── weights on the bridge ────────────────────────────────────────────────────────────────────────────────
    def _install(self):
        from sim.backend import to_host, from_host
        data = np.asarray(to_host(self.b.cp_connections.data)).copy()
        W = self.W
        if self.lesion == "learned_edge":
            W = self.W_init
        elif self.lesion == "afferent_zero":
            W = np.zeros_like(self.W)
        data[self.S] = W
        data[self.S_inh] = 0.0 if self.lesion == "competition" else self.inh_init
        data[self.S_j] = self.wj_init * (OR_LESION_FACTOR if self.lesion == "coincidence" else 1.0)
        self.b.cp_connections.data = from_host(data.astype(np.float32))
        self._w_hash = self.weight_hash()

    def set_lesion(self, kind):
        if kind not in (None,) + LESION_KINDS:
            raise ValueError(kind)
        if kind != self.lesion:
            self.lesion = kind
            self._install()

    # ── one presentation: v2's, plus the junctions' constant tonic current ───────────────────────────────────
    def present(self, word: str, teacher=None):
        """As v2 `present`, with I_TONIC_J added on every junction for the whole presentation."""
        from sim.backend import to_host, from_host
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None, 0
        self._reset_state()
        base = np.zeros(self.n, dtype=np.float64)
        base[self.fj] = self.i_tonic
        if teacher is not None:
            if teacher[0] > 0:
                base[self.cn[0]] = self.teacher_i
            elif teacher[0] < 0:
                base[self.cx[0]] = self.teacher_i
        counts = np.zeros(self.n, dtype=np.float64)
        steps = 0
        b = self.b
        for feats in occ:
            cur = base.copy()
            if feats:
                cur[self.fr[np.asarray(feats, dtype=np.int64)]] += self.i_frame
            dev = from_host(cur.astype(np.float32))
            for _ in range(self.t_on):
                b.cp_external_input_current[:] = dev
                b._run_one_simulation_step()
                counts += np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
                steps += 1
        b.cp_external_input_current[:] = 0.0
        return counts, steps

    # ── learning: the v2 Oja rule with the JUNCTION rates as the pre factor ──────────────────────────────────
    def hebbian_update(self, counts, steps):
        x = counts[self.fj] / steps
        y = np.concatenate([counts[g] for g in self.post_groups]) / steps
        self.W += self.eta * (np.outer(x, y) - self.beta * (y * y)[None, :] * self.W)

    def graded_drive(self, word: str):
        """Diagnostic only: the CN-minus-CX drive of the word's COMPLETE (-1,+1) frames through the installed
        weights, assuming every complete frame fires its junction (the AND) and nothing else fires."""
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None
        C, x = self.C, np.zeros(self.C * self.C)
        for feats in occ:
            left = [f - _OFF_L * C for f in feats if _OFF_L * C <= f < (_OFF_L + 1) * C]
            right = [f - _OFF_R * C for f in feats if _OFF_R * C <= f < (_OFF_R + 1) * C]
            for a in left:
                for bb in right:
                    x[a * C + bb] += 1
        data_w = {None: self.W, "learned_edge": self.W_init, "afferent_zero": 0 * self.W}.get(self.lesion, self.W)
        d = (x @ data_w).reshape(1, 2, L.N_CAT).mean(axis=2)
        return d[:, 0] - d[:, 1]

    # ── the AND, measured (integrity smoke + dev calibration) ────────────────────────────────────────────────
    def junction_response(self, pairs, which: str, steps=None):
        """Spikes of junction J(a,b) over `steps` (default one occurrence window, T_ON_J) from a washed-out state,
        with the tonic current on, when driving only FR(-1,a) ('left'), only FR(+1,b) ('right') or both ('both').
        Uses the installed (possibly lesioned) weights; plasticity is off. Returns (counts, first-spike steps)."""
        from sim.backend import to_host, from_host
        C, steps = self.C, int(steps or self.t_on)
        counts, first = [], []
        for a, bb in pairs:
            self._reset_state()
            cur = np.zeros(self.n, dtype=np.float64)
            cur[self.fj] = self.i_tonic
            if which in ("left", "both"):
                cur[self.fr[_OFF_L * C + a]] += self.i_frame
            if which in ("right", "both"):
                cur[self.fr[_OFF_R * C + bb]] += self.i_frame
            dev = from_host(cur.astype(np.float32))
            j, n_sp, t0 = self.fj[a * C + bb], 0, -1
            for t in range(steps):
                self.b.cp_external_input_current[:] = dev
                self.b._run_one_simulation_step()
                s = int(np.asarray(to_host(self.b.cp_firing_states))[j])
                if s and t0 < 0:
                    t0 = t
                n_sp += s
            self.b.cp_external_input_current[:] = 0.0
            counts.append(n_sp)
            first.append(t0)
        return np.asarray(counts), np.asarray(first)


SUSTAIN_STEPS = 150   # a lone afferent must stay silent for 3 occurrence windows, not just one


def and_smoke(lex, n_sample: int = 64, seed: int = 0):
    """The pre-registered AND integrity smoke on `lex`'s INSTALLED weights (target (i), AMENDMENT 1 wording): over a
    random sample of junctions, a lone left or right afferent held for SUSTAIN_STEPS fires no junction, and the pair
    fires every sampled junction within one occurrence window (T_ON_J)."""
    rng = np.random.default_rng(seed)
    C = lex.C
    pairs = [(int(a), int(b)) for a, b in zip(rng.integers(0, C, n_sample), rng.integers(0, C, n_sample))]
    left, _ = lex.junction_response(pairs, "left", SUSTAIN_STEPS)
    right, _ = lex.junction_response(pairs, "right", SUSTAIN_STEPS)
    both, first = lex.junction_response(pairs, "both", lex.t_on)
    return {"n": n_sample, "lone_left_fired": int((left > 0).sum()), "lone_right_fired": int((right > 0).sum()),
            "pair_fired_in_window": int((both > 0).sum()), "pair_spikes_mean": float(both.mean()),
            "pair_first_spike_max": int(first.max()) if (first >= 0).all() else -1,
            "and_holds": bool((left == 0).all() and (right == 0).all() and (both > 0).all())}


def and_population(lex, sustain: int = None):
    """The AND checked on EVERY junction at once. Junctions have no lateral or feedback input (only their two FR
    afferents; FJ -> CN/CX is feed-forward), so driving ALL left (-1) afferents tests every junction's lone-left
    response, ALL right (+1) afferents every lone-right response, and both sets together every pair, in parallel.
    Uses the installed (possibly lesioned) weights. Returns per-junction violation counts."""
    from sim.backend import to_host, from_host
    C, sustain = lex.C, int(sustain or SUSTAIN_STEPS)
    left = lex.fr[_OFF_L * C:(_OFF_L + 1) * C]
    right = lex.fr[_OFF_R * C:(_OFF_R + 1) * C]

    def run(drive, steps):
        lex._reset_state()
        cur = np.zeros(lex.n, dtype=np.float64)
        cur[lex.fj] = lex.i_tonic
        cur[drive] += lex.i_frame
        dev = from_host(cur.astype(np.float32))
        counts = np.zeros(lex.n)
        for _ in range(steps):
            lex.b.cp_external_input_current[:] = dev
            lex.b._run_one_simulation_step()
            counts += np.asarray(to_host(lex.b.cp_firing_states))
        lex.b.cp_external_input_current[:] = 0.0
        return counts[lex.fj], counts / steps

    lone_l, rl = run(left, sustain)
    lone_r, rr = run(right, sustain)
    pair, _ = run(np.concatenate([left, right]), lex.t_on)
    L2, R2 = (lone_l > 0).reshape(C, C), (lone_r > 0).reshape(C, C)     # [a, b]
    rate_l, rate_r = rl[left], rr[right]
    return {"n_junctions": int(len(lex.fj)), "lone_left_fired": int((lone_l > 0).sum()),
            "lone_right_fired": int((lone_r > 0).sum()), "pair_silent": int((pair == 0).sum()),
            "pair_spikes_mean": float(pair.mean()),
            "and_violations": int(((lone_l > 0) | (lone_r > 0) | (pair == 0)).sum()),
            "and_holds_all": bool(((lone_l == 0) & (lone_r == 0) & (pair > 0)).all()),
            # afferent-level structure: a whole row (left word a) / column (right word b) firing alone
            "lone_left_full_rows": [int(a) for a in np.nonzero(L2.sum(axis=1) == C)[0]],
            "lone_right_full_cols": [int(b) for b in np.nonzero(R2.sum(axis=0) == C)[0]],
            "left_afferent_rate": {"median": float(np.median(rate_l)), "max": float(rate_l.max()),
                                   "argmax": int(rate_l.argmax())},
            "right_afferent_rate": {"median": float(np.median(rate_r)), "max": float(rate_r.max()),
                                    "argmax": int(rate_r.argmax())}}


def calibrate_and(seed: int, env, weights, biases, n_sample: int = 64):
    """Dev-seed grid over (W_J, I_TONIC_J) on the UNTRAINED junction circuit: the AND smoke at each point.
    Selection rule (AMENDMENT 1): the weight whose feasible bias range (AND holds) is widest, and the middle bias
    of that range (a max-margin choice; ties -> the smaller weight)."""
    rows = []
    for w in weights:
        lex = FrameJunctionLexicon(seed, env, w_j=float(w))
        for bias in biases:
            lex.i_tonic = float(bias)
            rows.append({"w_j": float(w), "i_tonic": float(bias), **and_smoke(lex, n_sample)})
    feas = collections_defaultdict_list()
    for r in rows:
        if r["and_holds"]:
            feas[r["w_j"]].append(r["i_tonic"])
    best = None
    for w in sorted(feas):
        bs = sorted(feas[w])
        width = bs[-1] - bs[0]
        if best is None or width > best[1]:
            best = (w, width, bs[len(bs) // 2] if len(bs) % 2 else 0.5 * (bs[len(bs) // 2 - 1] + bs[len(bs) // 2]))
    choice = None if best is None else {"w_j": best[0], "i_tonic": best[2], "feasible_bias_width": best[1]}
    return {"rows": rows, "choice": choice}


def measure_drive_ratio(seed: int, env):
    """AMENDMENT 1 drive matching: mean afferent spikes per step into the category pools, v2 (FR) over this variant
    (FJ), over the UNTRAINED presentations of the full seed curriculum (no teacher)."""
    v2 = L.SpikingFrameCategoryLexicon(seed, env)
    jx = FrameJunctionLexicon(seed, env)
    words, _ = L.seed_curriculum(env)
    a, b = [], []
    for w in words:
        c, st = v2.present(w)
        a.append(float(c[v2.fr].sum() / st))
        c, st = jx.present(w)
        b.append(float(c[jx.fj].sum() / st))
    return {"seed": seed, "n_words": len(words), "v2_fr_spikes_per_step_mean": float(np.mean(a)),
            "junction_fj_spikes_per_step_mean": float(np.mean(b)), "ratio_of_means": float(np.mean(a) / np.mean(b)),
            "per_word_ratio_median": float(np.median(np.asarray(a) / np.maximum(np.asarray(b), 1e-12))),
            "w_j": jx.w_j, "i_tonic": jx.i_tonic, "t_on_j": jx.t_on}


def collections_defaultdict_list():
    import collections
    return collections.defaultdict(list)


if __name__ == "__main__":
    import argparse
    import json
    import time
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default="300,400,500")
    ap.add_argument("--biases", default="-500,-600,-700,-800,-900")
    ap.add_argument("--n-sample", type=int, default=64)
    ap.add_argument("--drive-ratio", action="store_true")
    ap.add_argument("--and-population", action="store_true",
                    help="the AND on every junction at each (W_J, I_TONIC_J) grid point (--weights x --biases)")
    ap.add_argument("--corpus", default=L._DEFAULT_CORPUS)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    class _Env:                       # the AND needs only the afferent geometry, not the corpus
        C = L.CTX_C
        n_frame = L.CTX_C * len(L.OFFSETS)
    t0 = time.time()
    if a.calibrate:
        out = calibrate_and(a.seed, _Env(), [float(x) for x in a.weights.split(",")],
                            [float(x) for x in a.biases.split(",")], n_sample=a.n_sample)
        out.update({"seed": a.seed, "t_on_j": T_ON_J, "sustain_steps": SUSTAIN_STEPS,
                    "backend": os.environ.get("SIM_BACKEND"), "elapsed_s": round(time.time() - t0, 1)})
        for r in out["rows"]:
            print(r, flush=True)
        print("choice:", out["choice"])
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    if a.and_population:
        rows = []
        for w in [float(x) for x in a.weights.split(",")]:
            lexp = FrameJunctionLexicon(a.seed, _Env(), w_j=w)
            for bias in [float(x) for x in a.biases.split(",")]:
                lexp.i_tonic = bias
                rows.append({"w_j": w, "i_tonic": bias, **and_population(lexp)})
                print(rows[-1], flush=True)
        out = {"seed": a.seed, "rows": rows, "t_on_j": T_ON_J, "sustain_steps": SUSTAIN_STEPS,
               "frozen": {"w_j": W_J, "i_tonic": I_TONIC_J}, "backend": os.environ.get("SIM_BACKEND"),
               "elapsed_s": round(time.time() - t0, 1)}
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    if a.drive_ratio:
        from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab
        tokens = load_tokens(a.corpus, 8_000_000)
        vocab, _ = build_vocab(tokens, 2000)
        env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
        out = measure_drive_ratio(a.seed, env)
        out.update({"backend": os.environ.get("SIM_BACKEND"), "corpus_path": a.corpus,
                    "elapsed_s": round(time.time() - t0, 1)})
        print(out)
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    print(f"{time.time() - t0:.1f}s")
