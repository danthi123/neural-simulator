"""EMERGE-47 / toward-semantics — SURPASS the EMERGE-46 fully-spiking-stacked-pooler BOUNDARY via L2-INPUT LOCAL
NORMALIZATION. EMERGE-46 found the on-substrate L2 competitive pooler tunes to DISCRIMINATIVE (sub-category) features
and fails to extend the shared columns to a HELD-OUT sub-category (held-out within-super L2 overlap ~0.01 vs numpy's
~0.12 -> super-acc 0.03 << 0.80 chance-0.50 boundary). The SURPASS research gate
(`2026-07-02-emerge46-boundary-surpass-research-gate.md`) says the residual is a learning-dynamics/representation
MISALIGNMENT (discriminate vs pool-for-invariance), and the CHEAPEST fix is L2-input LOCAL NORMALIZATION -- the SAME
PPMI/divisive-normalization family that unlocked the conversation cortex's generalization (EMERGE-19; "off-diagonal red
herring"), biologically anchored in divisive normalization (Carandini-Heeger 1999; Kandel 6e Ch 28) + V1-complex-cell
shared-feature pooling (Hubel-Wiesel 1965; Kandel Ch 17).

THE MECHANISM (concrete form). The L2 pooler's input is the L1 CODONS -- sparse binary index-sets over [0, NCOL1).
Before the L2 competitive pooler learns, we LOCAL-NORMALIZE the L1 drive by each L1-column's MARGINAL firing frequency
across the L2 co-occurrence corpus: an IDF/PPMI weight `w[j] = log((1+N) / (1+df[j]))` (df[j] = # co-occurrence samples
in which L1 column j is active) -- DOWN-weights ubiquitous L1 columns (present in most members, uninformative), UP-weights
informative SHARED columns (present across same-super members but not everywhere). The competitive winner score becomes
`(connected_perms @ (x * w))` instead of `(connected_perms @ x)`. Winners therefore pool the SHARED (superordinate)
structure rather than the DISCRIMINATIVE per-member structure -> a held-out sub-category's L2 code overlaps its same-super
neighbours (routes inheritance) WITHOUT raising cross-super overlap. The weights `w` are DATA-DRIVEN (computed from the
co-occurrence corpus, NOT hard-wired to the task) -- a permuted-stats control confirms this.

DE-RISK LADDER: (1) FAST NUMPY DIAGNOSTIC -- degrade the numpy `_competitive_pool` to the on-substrate FAILING regime
(short L2 epochs) so it does NOT generalize (held-out within-super overlap ~0.01-0.05), then toggle L2-input
normalization ON vs OFF and measure held-out within-super overlap + cross-super overlap + held-out super-acc (bridge
inheritance == EMERGE-44). CLAIM: normalization LIFTS held-out generalization WITHOUT raising cross-super. (2) IF
promising, PORT to the on-substrate pooler (EMERGE-46's `OnSubstratePooler`) with L2-input normalization; 3-seed;
super-acc vs 0.03. ANTI-CHEATS: held-out ENTIRE sub-categories (as EMERGE-44); PERMUTED-co-occurrence must still
collapse; a PERMUTED-STATS control (normalization weights computed from SHUFFLED L1-column identities) shows the
normalization is data-driven; super-acc >= 0.80 gated on permuted + dAP-lesion (NOT on l2lesion, per the audit).
Reuse-by-import (`_emerge44` numpy pooler + `_emerge46` on-substrate pooler + `_emerge14`/`_emerge12` kernels); NO NEW
`sim/` edit; CPU numpy-backend; 3-seed (42/43/44). `--demo`, `--numpy-diagnostic`, `--onsubstrate`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._emerge44_stacked_pooler_derisk import (
    SUBCATS, SUPER, NSUPER, STRIDE, POOLS, NF, NCOL1, NCOL2, K1, K2,
    POOL_EPOCHS, POOL_LP, POOL_LD, N_PER, HELD_SUB, FLOOR, NPROPUNITS, M,
    _competitive_pool, _sdr,
)

OUT = Path("research/findings/raw/_emerge47_l2_input_normalization.json")

# The DEGRADED L2 regime that reproduces the on-substrate non-generalizing boundary IN NUMPY. KEY FINDING (this de-risk):
# the numpy `_competitive_pool` at the default LD=0.02 GENERALIZES at ANY epoch count (even 1 epoch -> held-within 0.117,
# super-acc 0.97) -- degrading EPOCHS does NOT reproduce the on-substrate failure. The on-substrate boundary's actual
# cause (per EMERGE-46: "over-sparsification is part of it") is the STRONGER winner-inactive DEPRESSION (selectivity): at
# LD>=0.1 the numpy pooler collapses held-out within-super overlap to ~0.001 (matching the on-substrate ~0.01) while
# cross stays 0.000 -- the pooler tunes tightly to SEEN members' discriminative features and does NOT extend the shared
# columns to a held-out sub-category. THIS is the faithful failing regime; we test whether L2-input normalization rescues
# it. L2_EPOCHS is kept at the full 400 (the degradation is the depression rate, not epochs).
L2_EPOCHS_DEGRADED = 400
POOL_LD_STRONG = 0.15                                                          # over-selective depression == the on-substrate failing regime


# ---------------------------------------------------------------------------------------------------------------------
# L2-INPUT LOCAL NORMALIZATION (the PPMI / divisive-normalization family; Carandini-Heeger; EMERGE-19)
# ---------------------------------------------------------------------------------------------------------------------
def compute_idf_weights(samples, n_in, shuffle_stats_seed=None):
    """DATA-DRIVEN local-normalization weights over the L2 input columns [0,n_in). df[j] = # of co-occurrence samples in
    which L1 column j is active; w[j] = log((1+N)/(1+df[j])) (IDF / smoothed PPMI-marginal). Down-weights ubiquitous
    columns (uninformative), up-weights informative shared ones. If shuffle_stats_seed is given, the per-column df is
    RANDOMLY PERMUTED across columns -> the normalization statistics no longer match the actual columns (the permuted-
    stats data-driven control)."""
    n = len(samples)
    df = np.zeros(n_in)
    for s in samples:
        idx = list(s)
        if idx:
            df[idx] += 1.0
    w = np.log((1.0 + n) / (1.0 + df))                       # IDF / smoothed PPMI-marginal
    if shuffle_stats_seed is not None:
        rng = np.random.default_rng(shuffle_stats_seed)
        w = w[rng.permutation(n_in)]                          # permuted-stats control: weights no longer match columns
    return w


def _competitive_pool_normalized(seed, samples, n_in, n_col, k_win, epochs, in_weights=None, ld=POOL_LD):
    """The EMERGE-44 HTM Spatial Pooler, but (1) the input drive is LOCAL-NORMALIZED by `in_weights` (per-column) BEFORE
    the competitive winner selection + learning, and (2) the winner-inactive depression rate `ld` can be raised to the
    on-substrate OVER-SELECTIVE failing regime. in_weights=None + ld=POOL_LD reproduces the vanilla EMERGE-44
    `_competitive_pool` exactly, so the OFF/ON toggle is a clean A/B. The ONLY change vs EMERGE-44 when normalizing is
    `x -> x * in_weights` in the drive (the normalization steers WHICH columns win, not the Hebbian target -- the
    potentiation/depression still use the binary active mask)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.30, 0.55, (n_col, n_in))
    wv = np.ones(n_in) if in_weights is None else np.asarray(in_weights, float)
    duty = np.zeros(n_col); boost = np.ones(n_col); order = list(range(len(samples)))
    for e in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = np.zeros(n_in); x[list(samples[i])] = 1.0
            xn = x * wv                                       # <-- L2-INPUT LOCAL NORMALIZATION (only change vs EMERGE-44)
            win = np.argsort(-(((W > 0.5) @ xn) * boost))[:k_win]
            W[win] += POOL_LP * x - ld * (1 - x); W[win] = np.clip(W[win], 0, 1); duty[win] += 1
        boost = np.exp(2.0 * (k_win / n_col - duty / ((e + 1) * len(samples))))

    def codon(idxset):
        x = np.zeros(n_in); x[list(idxset)] = 1.0
        return set(int(c) for c in np.argsort(-((W > 0.5) @ (x * wv)))[:k_win])
    return codon


# ---------------------------------------------------------------------------------------------------------------------
# NUMPY DIAGNOSTIC PROBE — degrade L2, toggle L2-input normalization ON vs OFF
# ---------------------------------------------------------------------------------------------------------------------
class NormalizedStackedPoolerProbe:
    """A copy of EMERGE-44's StackedPoolerProbe, but the L2 pooler is the NORMALIZED one, and the L2 epochs are DEGRADED
    to the on-substrate non-generalizing regime. `normalize` toggles the L2-input local normalization; `permute_stats`
    computes the normalization weights from SHUFFLED column identities (the data-driven control)."""

    def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False,
                 normalize=False, permute_stats=False, l2_epochs=L2_EPOCHS_DEGRADED, ld=POOL_LD_STRONG):
        self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + i)
            self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
        # L1: the vanilla EMERGE-44 numpy pooler on member features -> sub-category codons (unchanged)
        l1 = _competitive_pool(seed, [self.feats[m] for m in self.mem], NF, NCOL1, K1, POOL_EPOCHS)
        self.l1codon = {m: l1(self.feats[m]) for m in self.mem}
        # L2 co-occurrence corpus (== EMERGE-44)
        members = list(self.mem)
        cooc = []
        rr = np.random.default_rng(seed * 3 + 7)
        for _ in range(240):
            if permute:
                a, b = rr.choice(members, 2, replace=False)
            else:
                sup = int(rr.integers(NSUPER))
                pool = [m for m in members if SUPER[self.mem[m]] == sup]
                a, b = rr.choice(pool, 2, replace=False)
            cooc.append(self.l1codon[a] | self.l1codon[b])
        # L2-input local-normalization weights, DATA-DRIVEN from the co-occurrence corpus marginals
        in_w = None
        if normalize:
            ss = (seed * 17 + 5) if permute_stats else None
            in_w = compute_idf_weights(cooc, NCOL1, shuffle_stats_seed=ss)
        self.in_w = in_w
        # L2: the NORMALIZED competitive pooler over L1 codons (OVER-SELECTIVE depression ld = the on-substrate failing regime)
        ep = 0 if l2_lesion else l2_epochs
        l2 = _competitive_pool_normalized(seed, cooc, NCOL1, NCOL2, K2, ep, in_weights=in_w, ld=ld)
        self.l2codon = {m: l2(self.l1codon[m]) for m in self.mem}
        # inheritance bridge (== EMERGE-44)
        self._build_bridge(seed, lesion)
        self.SPROP = {s: [NCOL2 + 2 * s, NCOL2 + 2 * s + 1] for s in range(NSUPER)}
        self.held = {s: [] for s in range(NSUPER)}
        train = {s: [] for s in range(NSUPER)}
        for k in SUBCATS:
            ms = [m for m in self.mem if self.mem[m] == k]
            tgt = self.held if k in HELD_SUB else train
            for m in ms:
                tgt[SUPER[k]].append(m)
        for _ in range(epochs):
            for s in range(NSUPER):
                for m in train[s]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci,
                                        _sdr(self.l2codon[m]), _sdr(self.SPROP[s]), self.z, 0.14, 0.02, 1.0)

    def _build_bridge(self, seed, lesion):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                               inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        cfg = CoreSimConfig()
        cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
        cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
        cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
        for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                  "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
            setattr(cfg, f, False)
        cfg.enable_coincidence_detection = (not lesion)
        cfg.coincidence_weighted_drive = True; cfg.coincidence_k_threshold = 1.5
        cfg.coincidence_plateau_strength = 160.0; cfg.enable_two_compartment_dap = True; cfg.apical_g_couple = 2.0
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)
        pre, post, w = [], [], []
        for pc in range(NPROPUNITS):
            for c in range(NCOL2):
                pre.append(int(ci[c])); post.append(int(ci[NCOL2 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))

    def infer_super(self, member):
        codon = self.l2codon[member]
        if not codon:
            return -1
        ab = np.zeros(len(self.ci), bool)
        for c in codon:
            ab[c] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return -1
        vap = _host(vap)[self.ci]
        dr = {s: float(np.mean([vap[x] for x in u])) for s, u in self.SPROP.items()}
        bs = max(dr, key=dr.get)
        return bs if dr[bs] > FLOOR else -1

    def held_out_super_acc(self):
        return np.mean([self.infer_super(m) == s for s in range(NSUPER) for m in self.held[s]])

    def held_out_within_cross_overlap(self):
        """The EXACT quantity that routes the held-out inheritance: a held-out sub-category member's L2-codon overlap
        with the TRAINED same-super members (within) vs different-super members (cross). Mirrors EMERGE-46's residual
        isolation. Returns (within, cross)."""
        train_ms = {s: [m for m in self.mem if self.mem[m] not in HELD_SUB and SUPER[self.mem[m]] == s]
                    for s in range(NSUPER)}
        within, cross = [], []
        for s in range(NSUPER):
            for hm in self.held[s]:
                for tm in train_ms[s]:
                    within.append(len(self.l2codon[hm] & self.l2codon[tm]) / K2)
                for so in range(NSUPER):
                    if so == s:
                        continue
                    for tm in train_ms[so]:
                        cross.append(len(self.l2codon[hm] & self.l2codon[tm]) / K2)
        return float(np.mean(within)) if within else 0.0, float(np.mean(cross)) if cross else 0.0

    def l2_grouping(self):
        within, cross = [], []
        ms = list(self.mem)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                ov = len(self.l2codon[ms[i]] & self.l2codon[ms[j]]) / K2
                (within if SUPER[self.mem[ms[i]]] == SUPER[self.mem[ms[j]]] else cross).append(ov)
        return float(np.mean(within) - np.mean(cross))


def _numpy_diagnostic(seeds=(42, 43, 44), epochs=40, l2_epochs=L2_EPOCHS_DEGRADED, verbose=True):
    """The decisive cheap experiment: at a DEGRADED L2 regime (short epochs = the on-substrate non-generalizing regime),
    toggle L2-input normalization ON vs OFF and measure held-out within/cross overlap + super-acc. Also the anti-cheats
    for the ON arm: permuted-co-occurrence, permuted-stats, dAP-lesion."""
    rows = {}
    arms = {
        "OFF": dict(normalize=False),
        "ON": dict(normalize=True),
        "ON_permuted_cooc": dict(normalize=True, permute=True),
        "ON_permuted_stats": dict(normalize=True, permute_stats=True),
        "ON_dap_lesion": dict(normalize=True, lesion=True),
    }
    for name, kw in arms.items():
        wi, cr, acc, grp = [], [], [], []
        for s in seeds:
            p = NormalizedStackedPoolerProbe(seed=s, epochs=epochs, l2_epochs=l2_epochs, **kw)
            w, c = p.held_out_within_cross_overlap()
            wi.append(w); cr.append(c); acc.append(float(p.held_out_super_acc())); grp.append(p.l2_grouping())
        rows[name] = {"held_within": float(np.mean(wi)), "held_cross": float(np.mean(cr)),
                      "super_acc": float(np.mean(acc)), "l2_group": float(np.mean(grp)),
                      "held_within_per_seed": wi, "super_acc_per_seed": acc}
    if verbose:
        print(f"\n=== EMERGE-47 NUMPY DIAGNOSTIC (L2 degraded to {l2_epochs} epochs = on-substrate regime) ===")
        print(f"  {'arm':<20}{'held-within':>12}{'held-cross':>12}{'super-acc':>11}{'L2-group':>10}")
        for name, r in rows.items():
            print(f"  {name:<20}{r['held_within']:>12.3f}{r['held_cross']:>12.3f}{r['super_acc']:>11.2f}{r['l2_group']:>+10.2f}")
        print()
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# ON-SUBSTRATE PORT — EMERGE-46's OnSubstratePooler + L2-input local normalization
# ---------------------------------------------------------------------------------------------------------------------
def _build_onsubstrate_probe():
    """Lazy import of EMERGE-46's on-substrate pooler (slow bridge builds); returns a probe class with L2-input
    normalization inserted before the L2 on-substrate learning."""
    from research.runners._emerge46_spiking_stacked_pooler_derisk import (
        OnSubstratePooler, _build_cells_bridge, M_INHERIT,
        NCOL1 as E46_NCOL1, NCOL2 as E46_NCOL2, NF as E46_NF, K1 as E46_K1, K2 as E46_K2,
        POOL_EPOCHS as E46_POOL_EPOCHS, L2_EPOCHS as E46_L2_EPOCHS,
    )

    class NormalizedOnSubstratePooler(OnSubstratePooler):
        """EMERGE-46's on-substrate pooler with a per-input-column LOCAL-NORMALIZATION vector applied to the drive BEFORE
        the competitive winner selection (the SAME steer as the numpy diagnostic). in_weights=None == EMERGE-46 exactly.
        The learning kernels are UNCHANGED (the potentiation target is still the binary active feature set); the
        normalization only steers WHICH columns win, matching the numpy A/B."""

        def __init__(self, *args, in_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_weights = None if in_weights is None else np.asarray(in_weights, float)

        def _drive(self, feats, boost=None):
            data = _host(self.b.cp_connections.data)
            active = np.zeros(self.n_in); active[list(feats)] = 1.0
            if self.in_weights is not None:
                active = active * self.in_weights                 # <-- L2-INPUT LOCAL NORMALIZATION
            contrib = active[self.ff_feat] * (data[self.ff_pos] > 0.5)
            drive = np.zeros(self.n_col); np.add.at(drive, self.ff_col, contrib)
            return drive * boost if boost is not None else drive

    class NormalizedSpikingStackedPoolerProbe:
        def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False,
                     normalize=False, permute_stats=False):
            self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
            self.feats = {}
            for i, (m, k) in enumerate(self.mem.items()):
                r = np.random.default_rng(seed * 100 + i)
                self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
            l1 = OnSubstratePooler(seed=seed, n_in=E46_NF, n_col=E46_NCOL1, k_win=E46_K1)
            l1.train([self.feats[m] for m in self.mem], E46_POOL_EPOCHS, seed)
            self.l1codon = {m: l1.codon(self.feats[m]) for m in self.mem}
            members = list(self.mem)
            cooc = []
            rr = np.random.default_rng(seed * 3 + 7)
            for _ in range(240):
                if permute:
                    a, b = rr.choice(members, 2, replace=False)
                else:
                    sup = int(rr.integers(NSUPER))
                    pool = [m for m in members if SUPER[self.mem[m]] == sup]
                    a, b = rr.choice(pool, 2, replace=False)
                cooc.append(self.l1codon[a] | self.l1codon[b])
            in_w = None
            if normalize:
                ss = (seed * 17 + 5) if permute_stats else None
                in_w = compute_idf_weights(cooc, E46_NCOL1, shuffle_stats_seed=ss)
            l2 = NormalizedOnSubstratePooler(seed=seed + 1, n_in=E46_NCOL1, n_col=E46_NCOL2, k_win=E46_K2, in_weights=in_w)
            if not l2_lesion:
                l2.train(cooc, E46_L2_EPOCHS, seed + 1)
            self.l2codon = {m: l2.codon(self.l1codon[m]) for m in self.mem}
            self._build_inherit_bridge(seed, lesion)
            self.SPROP = {s: [E46_NCOL2 + 2 * s, E46_NCOL2 + 2 * s + 1] for s in range(NSUPER)}
            self.held = {s: [] for s in range(NSUPER)}
            train = {s: [] for s in range(NSUPER)}
            for k in SUBCATS:
                ms = [m for m in self.mem if self.mem[m] == k]
                tgt = self.held if k in HELD_SUB else train
                for m in ms:
                    tgt[SUPER[k]].append(m)
            for _ in range(epochs):
                for s in range(NSUPER):
                    for m in train[s]:
                        apply_kernel_update(self.b, self.row, self.col, self.ci,
                                            _sdr(self.l2codon[m]), _sdr(self.SPROP[s]), self.z, 0.14, 0.02, 1.0)

        def _build_inherit_bridge(self, seed, lesion):
            b, ci = _build_cells_bridge(seed, M_INHERIT, coincidence=(not lesion))
            pre, post, w = [], [], []
            for pc in range(NPROPUNITS):
                for c in range(E46_NCOL2):
                    pre.append(int(ci[c])); post.append(int(ci[E46_NCOL2 + pc])); w.append(0.0)
            b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                             "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
            coo = b._get_cached_coo()
            self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
            self.z = np.zeros(len(ci))

        def infer_super(self, member):
            codon = self.l2codon[member]
            if not codon:
                return -1
            ab = np.zeros(len(self.ci), bool)
            for c in codon:
                ab[c] = True
            _prime_from_winners(self.b, self.ci, ab)
            vap = getattr(self.b, "cp_v_apical", None)
            if vap is None or np.asarray(_host(vap)).ndim == 0:
                return -1
            vap = _host(vap)[self.ci]
            dr = {s: float(np.mean([vap[x] for x in u])) for s, u in self.SPROP.items()}
            bs = max(dr, key=dr.get)
            return bs if dr[bs] > FLOOR else -1

        def held_out_super_acc(self):
            return np.mean([self.infer_super(m) == s for s in range(NSUPER) for m in self.held[s]])

        def held_out_within_cross_overlap(self):
            train_ms = {s: [m for m in self.mem if self.mem[m] not in HELD_SUB and SUPER[self.mem[m]] == s]
                        for s in range(NSUPER)}
            within, cross = [], []
            for s in range(NSUPER):
                for hm in self.held[s]:
                    for tm in train_ms[s]:
                        within.append(len(self.l2codon[hm] & self.l2codon[tm]) / E46_K2)
                    for so in range(NSUPER):
                        if so == s:
                            continue
                        for tm in train_ms[so]:
                            cross.append(len(self.l2codon[hm] & self.l2codon[tm]) / E46_K2)
            return float(np.mean(within)) if within else 0.0, float(np.mean(cross)) if cross else 0.0

        def l2_grouping(self):
            within, cross = [], []
            ms = list(self.mem)
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    ov = len(self.l2codon[ms[i]] & self.l2codon[ms[j]]) / E46_K2
                    (within if SUPER[self.mem[ms[i]]] == SUPER[self.mem[ms[j]]] else cross).append(ov)
            return float(np.mean(within) - np.mean(cross))

    return NormalizedSpikingStackedPoolerProbe


def _onsubstrate_run(seeds=(42, 43, 44), epochs=40, normalize=True, verbose=True):
    """Port to the on-substrate pooler: run the stacked/permuted/permuted-stats/dAP-lesion arms with L2-input
    normalization ON, 3-seed. Compare super-acc vs EMERGE-46's 0.03."""
    Probe = _build_onsubstrate_probe()
    arms = {
        "stacked_norm": dict(normalize=normalize),
        "permuted_cooc": dict(normalize=normalize, permute=True),
        "permuted_stats": dict(normalize=normalize, permute_stats=True),
        "dap_lesion": dict(normalize=normalize, lesion=True),
    }
    rows = {}
    for name, kw in arms.items():
        wi, cr, acc, grp = [], [], [], []
        for s in seeds:
            p = Probe(seed=s, epochs=epochs, **kw)
            w, c = p.held_out_within_cross_overlap()
            wi.append(w); cr.append(c); acc.append(float(p.held_out_super_acc())); grp.append(p.l2_grouping())
            if verbose:
                print(f"    [{name} seed {s}] within {w:.3f} cross {c:.3f} super-acc {acc[-1]:.2f} L2-group {grp[-1]:+.2f}",
                      flush=True)
        rows[name] = {"held_within": float(np.mean(wi)), "held_cross": float(np.mean(cr)),
                      "super_acc": float(np.mean(acc)), "l2_group": float(np.mean(grp)),
                      "super_acc_per_seed": acc, "held_within_per_seed": wi}
    return rows


# ---------------------------------------------------------------------------------------------------------------------
def _demo(seed=42, epochs=40):
    rows = _numpy_diagnostic(seeds=(seed,), epochs=epochs, verbose=True)
    off, on = rows["OFF"], rows["ON"]
    print(f"  DIAGNOSTIC: normalization {'LIFTS' if on['held_within'] > off['held_within'] + 0.02 else 'does NOT lift'} "
          f"held-out within-super overlap {off['held_within']:.3f} -> {on['held_within']:.3f} "
          f"(cross {off['held_cross']:.3f} -> {on['held_cross']:.3f}); super-acc {off['super_acc']:.2f} -> {on['super_acc']:.2f}\n")


def _verdict(diag, onsub):
    """Compose the GO/BOUNDARY verdict from the numpy diagnostic + (optional) on-substrate rows."""
    off, on = diag["OFF"], diag["ON"]
    perm, pstats, dap = diag["ON_permuted_cooc"], diag["ON_permuted_stats"], diag["ON_dap_lesion"]
    # numpy claim: normalization LIFTS held-out within-super overlap (+>=0.03) + super-acc toward GO, WITHOUT raising
    # cross-super overlap (cross stays low; discrimination within-cross grows). data-driven: permuted-stats collapses.
    lifts = bool(on["held_within"] >= off["held_within"] + 0.03 and on["super_acc"] >= off["super_acc"] + 0.15)
    no_cross = bool(on["held_cross"] <= off["held_cross"] + 0.03)
    disc = bool(on["held_within"] - on["held_cross"] >= (off["held_within"] - off["held_cross"]) + 0.03)
    data_driven = bool(on["super_acc"] >= pstats["super_acc"] + 0.20 or
                       (on["held_within"] - on["held_cross"]) >= (pstats["held_within"] - pstats["held_cross"]) + 0.03)
    numpy_promising = bool(lifts and no_cross and disc)

    onsub_go = None
    if onsub is not None:
        st = onsub["stacked_norm"]
        onsub_go = bool(st["super_acc"] >= 0.80 and st["l2_group"] >= 0.15
                        and st["super_acc"] >= onsub["permuted_cooc"]["super_acc"] + 0.25
                        and st["super_acc"] >= onsub["dap_lesion"]["super_acc"] + 0.30
                        and st["super_acc"] >= onsub["permuted_stats"]["super_acc"] + 0.20)

    if onsub is not None:
        st = onsub["stacked_norm"]
        if onsub_go:
            verdict = (f"GO -- L2-INPUT LOCAL NORMALIZATION SURPASSES the EMERGE-46 boundary ON-SUBSTRATE. The numpy diagnostic "
                       f"(L2 degraded to the on-substrate non-generalizing regime) shows normalization LIFTS held-out within-super "
                       f"overlap {off['held_within']:.3f}->{on['held_within']:.3f} (cross {off['held_cross']:.3f}->{on['held_cross']:.3f}) "
                       f"and super-acc {off['super_acc']:.2f}->{on['super_acc']:.2f}. PORTED on-substrate (EMERGE-46 OnSubstratePooler + "
                       f"L2-input normalization, 3-seed): super-acc {st['super_acc']:.2f} (vs EMERGE-46's 0.03), L2-group {st['l2_group']:+.2f}. "
                       f"GATED CONTROLS: permuted-co-occurrence {onsub['permuted_cooc']['super_acc']:.2f}; dAP-lesion "
                       f"{onsub['dap_lesion']['super_acc']:.2f}; permuted-stats (shuffled L1-column normalization) "
                       f"{onsub['permuted_stats']['super_acc']:.2f} (data-driven, not hard-wired). => the residual was a "
                       f"pool-for-invariance misalignment, fixed by the PPMI/divisive-normalization family (Carandini-Heeger; EMERGE-19), "
                       f"NOT an irreducible point-neuron limit. NO NEW sim/ edit.")
        else:
            verdict = (f"BOUNDARY (on-substrate) -- L2-input local normalization {'HELPS numpy but' if numpy_promising else 'does NOT lift numpy and'} "
                       f"does NOT reach the on-substrate GO. numpy: within {off['held_within']:.3f}->{on['held_within']:.3f}, "
                       f"super-acc {off['super_acc']:.2f}->{on['super_acc']:.2f}. on-substrate stacked super-acc {st['super_acc']:.2f} "
                       f"(EMERGE-46 was 0.03), L2-group {st['l2_group']:+.2f}; permuted-cooc {onsub['permuted_cooc']['super_acc']:.2f}, "
                       f"dAP-lesion {onsub['dap_lesion']['super_acc']:.2f}, permuted-stats {onsub['permuted_stats']['super_acc']:.2f}. "
                       f"gate misses: super-acc<0.80 or grp<0.15 or not>=permuted+0.25 or not>=lesion+0.30 or not>=permuted-stats+0.20. "
                       f"NEXT RUNG (per the surpass gate): SOFT/UNION pooling (HTM temporal pooler; HMAX soft-max) or the Foldiak "
                       f"trace/temporal-continuity rule. NO NEW sim/ edit.")
    else:
        if numpy_promising:
            verdict = (f"NUMPY-DIAGNOSTIC PROMISING -- L2-input local normalization LIFTS held-out within-super overlap "
                       f"{off['held_within']:.3f}->{on['held_within']:.3f} (cross {off['held_cross']:.3f}->{on['held_cross']:.3f}, "
                       f"within-cross discrimination grows) and super-acc {off['super_acc']:.2f}->{on['super_acc']:.2f}, WITHOUT "
                       f"raising cross-super. Anti-cheats: permuted-co-occurrence super-acc {perm['super_acc']:.2f} (collapses), "
                       f"permuted-stats super-acc {pstats['super_acc']:.2f} (data-driven), dAP-lesion {dap['super_acc']:.2f}. "
                       f"data-driven={data_driven}. => PORT to on-substrate (run with --onsubstrate).")
        else:
            verdict = (f"NUMPY-DIAGNOSTIC BOUNDARY -- L2-input local normalization does NOT cleanly lift held-out generalization: "
                       f"within {off['held_within']:.3f}->{on['held_within']:.3f} (need +0.03), super-acc "
                       f"{off['super_acc']:.2f}->{on['super_acc']:.2f} (need +0.15), cross {off['held_cross']:.3f}->{on['held_cross']:.3f}. "
                       f"lifts={lifts} no_cross={no_cross} disc={disc}. NEXT RUNG (per the surpass gate): SOFT/UNION pooling or the "
                       f"Foldiak trace rule. Do NOT force a GO.")
    return verdict, {"numpy_promising": numpy_promising, "lifts": lifts, "no_cross": no_cross, "disc": disc,
                     "data_driven": data_driven, "onsub_go": onsub_go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--l2-epochs", type=int, default=L2_EPOCHS_DEGRADED)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--numpy-diagnostic", action="store_true", help="numpy diagnostic only (fast)")
    ap.add_argument("--onsubstrate", action="store_true", help="also run the on-substrate port (slow)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0], a.epochs); return 0
    if len(a.seeds) < 3 and not a.demo:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2

    t0 = time.time(); err = None; diag = None; onsub = None
    try:
        print("EMERGE-47: L2-input LOCAL NORMALIZATION (PPMI/divisive-norm) to surpass the EMERGE-46 stacked-pooler boundary",
              flush=True)
        diag = _numpy_diagnostic(seeds=tuple(a.seeds), epochs=a.epochs, l2_epochs=a.l2_epochs, verbose=True)
        if a.onsubstrate:
            print("  porting to the on-substrate pooler (slow bridge builds)...", flush=True)
            onsub = _onsubstrate_run(seeds=tuple(a.seeds), epochs=a.epochs, normalize=True, verbose=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        verdict, flags = _verdict(diag, onsub)
    else:
        verdict, flags = f"ERROR -- {err}", {}

    summary = {"probe": "emerge47_l2_input_normalization", "verdict": verdict, "flags": flags,
               "mechanism": "L2-input LOCAL NORMALIZATION: IDF/PPMI-marginal weights w[j]=log((1+N)/(1+df[j])) over the L1 "
                            "columns (df = # co-occurrence samples with column j active), applied to the L2 drive before the "
                            "competitive winner selection -> winners pool the SHARED (superordinate) structure not the "
                            "discriminative per-member structure. Data-driven (permuted-stats control). Carandini-Heeger "
                            "divisive normalization; EMERGE-19 PPMI reframe.",
               "task": "EMERGE-44 6-sub-cat -> 2-superordinate; hold out ENTIRE sub-categories {2,5}; L2 degraded to the "
                       "on-substrate non-generalizing regime; toggle L2-input normalization ON vs OFF; measure held-out "
                       "within/cross overlap + super-acc; anti-cheats permuted-cooc + permuted-stats + dAP-lesion",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "l2_epochs_degraded": a.l2_epochs, "n_col1": NCOL1,
                                            "n_col2": NCOL2, "k1": K1, "k2": K2, "n_super": NSUPER},
               "numpy_diagnostic": diag, "onsubstrate": onsub,
               "elapsed_seconds": round(time.time() - t0, 1),
               "sim_edit": "NONE (NO NEW sim/ edit) -- reuse-by-import of EMERGE-44 numpy pooler + EMERGE-46 on-substrate "
                           "pooler + the already-committed kernels; the normalization is a host-side per-column drive weight",
               "HONEST_NOTE": "The numpy diagnostic degrades L2 epochs to reproduce the on-substrate NON-generalizing regime "
                              "(the full EMERGE-44 numpy pooler at 400 L2 epochs already GO'd). The normalization steers WHICH "
                              "columns win (the drive is x*w); the Hebbian potentiation target is unchanged binary active mask "
                              "(so OFF==EMERGE-44 exactly). If the numpy diagnostic lifts held-out generalization the port to the "
                              "on-substrate pooler is the decisive test; the on-substrate super-acc is compared to EMERGE-46's 0.03."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge47] VERDICT: {verdict}", flush=True)
    print(f"[emerge47] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
