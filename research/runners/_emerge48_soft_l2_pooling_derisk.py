"""EMERGE-48 / toward-semantics — SURPASS the EMERGE-46 fully-spiking-stacked-pooler BOUNDARY via SOFT / UNION POOLING at
the L2 layer (HTM temporal pooler, Hawkins-Ahmad 2016; HMAX soft-max, Serre 2005). EMERGE-47 root-caused the EMERGE-46
boundary: the on-substrate L2 competitive pooler is OVER-SELECTIVE — its winner-inactive DEPRESSION (`ld_wi`) tunes each
L2 column TIGHTLY to the SEEN members' discriminative features, so a HELD-OUT sub-category's L1 codon does NOT drive the
shared L2 columns and inheritance collapses (held-out within-super L2 overlap ~0.01 -> super-acc ~0.03 << 0.80 chance-0.50).
EMERGE-47's numpy sweep proved the DOMINANT lever is SOFTENING that depression: at the on-substrate-faithful over-selective
regime (ld=0.15) super-acc is 0.06, but softening to ld=0.02 recovers it to 0.97 (numpy). This de-risk PORTS that soft/union
pooling to the REAL substrate: the L2 `OnSubstratePooler` is built with a LOW winner-inactive depression rate `ld_wi`
(swept ~0.0-0.05; numpy sweet spot ~0.02), so multiple L2 columns strengthen on SIMILAR inputs (union pooling) and
same-superordinate members — INCLUDING a held-out sub-category — SHARE L2 columns. L1 stays at its normal `ld_wi` (L1
discrimination is fine; only L2 must pool for invariance). Optionally combined with EMERGE-47's L2-input local
normalization (an additive secondary lift).

THE KEY POINT: the winner-inactive kernel `fused_htm_winner_inactive_depression(w, pre_active, post_win, lam_dep_wi, ...)`
ALREADY takes `lam_dep_wi` as a soft rate parameter (0 = off). A lower L2 rate needs NO NEW `sim/` edit — it is a
constructor argument to the EMERGE-46 `OnSubstratePooler`. Biology: HTM temporal/union pooling (Hawkins-Ahmad 2016) +
HMAX soft-max complex-cell pooling (Serre-Poggio 2005; Kandel Ch 17 V1 complex cells pool over simple cells) — a gentler
selectivity that pools SHARED structure rather than discriminating per-member.

ANTI-CHEATS (mirror EMERGE-44/46/47 exactly; all must still hold): held out ENTIRE SUB-CATEGORIES {2,5} (a held-out
member can inherit ONLY via the L2-DISCOVERED grouping); PERMUTED-co-occurrence collapses (no superordinate structure);
if normalization is kept, PERMUTED-STATS collapses (data-driven weights); dAP-LESION collapses (coincidence read is
load-bearing); GATE: super-acc >= 0.80 AND >= permuted+0.25 AND >= dAP-lesion+0.30 AND L2-grouping >= 0.15; l2lesion is
REPORTED-not-gated (a fixed-random control, per the anti-cheat control-validity methodology). CRITICAL (the shortcut
guard): soft pooling must NOT just raise cross-super overlap equally — WITHIN-super held-out overlap must rise while
CROSS-super stays LOW (else it is indiscriminate collision, not generalization), and permuted must still collapse.

Reuse-by-import (`_emerge44` task constants + numpy pooler; `_emerge46` `OnSubstratePooler` + bridge; `_emerge47`
`compute_idf_weights` + `NormalizedOnSubstratePooler`; `_emerge14`/`_emerge12` kernels); NO NEW `sim/` edit; CPU
numpy-backend; 3-seed (42/43/44). `--demo`, `--numpy-sweep` (fast), `--onsubstrate` (the decisive port), `--l2-ld` /
`--normalize`.
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
    SUBCATS, SUPER, NSUPER, POOLS, NCOL1, NCOL2, K1, K2,
    POOL_EPOCHS, N_PER, HELD_SUB, FLOOR, NPROPUNITS,
    _competitive_pool, _sdr,
)
from research.runners._emerge47_l2_input_normalization_derisk import (
    compute_idf_weights, _competitive_pool_normalized, POOL_LD_STRONG,
)

OUT = Path("research/findings/raw/_emerge48_soft_l2_pooling.json")

# The SOFT L2 winner-inactive depression rate (union pooling). EMERGE-47's numpy sweep: super-acc 0.06 (ld=0.15,
# over-selective) -> 0.97 (ld=0.02, soft). The on-substrate committed default is 0.02, but its float32/kernel dynamics land
# HARDER than numpy (EMERGE-46 super-acc 0.03 at ld_wi=0.02), so we sweep DOWN into the soft/union regime.
L2_LD_SOFT_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05]
L2_LD_SOFT_DEFAULT = 0.005                                                        # the on-substrate soft/union default (swept below)


# =====================================================================================================================
# FAST NUMPY SWEEP — confirm the soft/union L2 depression recovers held-out generalization (EMERGE-47's key sweep,
# extended: L1 stays at the discriminative rate, only L2 softens; ON/OFF normalization)
# =====================================================================================================================
class SoftL2NumpyProbe:
    """EMERGE-44/47 stacked pooler, numpy, with an INDEPENDENT L2 winner-inactive depression rate `l2_ld` (soft/union
    pooling) and an OPTIONAL L2-input normalization. L1 uses the vanilla EMERGE-44 pooler (normal discriminative rate);
    only L2 softens. Reproduces the on-substrate faithful-degraded regime at l2_ld=POOL_LD_STRONG (0.15) and the soft
    recovery at low l2_ld. Held-out ENTIRE sub-categories {2,5}."""

    def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False,
                 normalize=False, permute_stats=False, l2_ld=L2_LD_SOFT_DEFAULT, l2_epochs=400):
        self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + i)
            self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
        # L1: the vanilla EMERGE-44 numpy pooler (normal discriminative rate) -> sub-category codons
        l1 = _competitive_pool(seed, [self.feats[m] for m in self.mem], max(c for cs in POOLS.values() for c in cs) + 1,
                               NCOL1, K1, POOL_EPOCHS)
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
        in_w = None
        if normalize:
            ss = (seed * 17 + 5) if permute_stats else None
            in_w = compute_idf_weights(cooc, NCOL1, shuffle_stats_seed=ss)
        # L2: NORMALIZED (optional) competitive pooler over L1 codons with the SOFT/UNION depression rate l2_ld
        ep = 0 if l2_lesion else l2_epochs
        l2 = _competitive_pool_normalized(seed, cooc, NCOL1, NCOL2, K2, ep, in_weights=in_w, ld=l2_ld)
        self.l2codon = {m: l2(self.l1codon[m]) for m in self.mem}
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
        M = NCOL2 + NPROPUNITS
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
        """The EXACT quantity that routes the held-out inheritance: a held-out member's L2 overlap with TRAINED
        same-super members (within) vs different-super members (cross). Returns (within, cross)."""
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


def _numpy_sweep(seeds=(42, 43, 44), epochs=40, normalize=False, verbose=True):
    """Sweep the L2 winner-inactive depression rate (over-selective 0.15 -> soft/union 0.0) x normalize ON/OFF (numpy,
    3-seed). CLAIM (from EMERGE-47): softening the L2 depression recovers held-out generalization to GO; within-super
    rises while cross-super stays low."""
    lds = [POOL_LD_STRONG] + L2_LD_SOFT_SWEEP
    rows = {}
    for ld in lds:
        wi, cr, acc, grp = [], [], [], []
        for s in seeds:
            p = SoftL2NumpyProbe(seed=s, epochs=epochs, l2_ld=ld, normalize=normalize)
            w, c = p.held_out_within_cross_overlap()
            wi.append(w); cr.append(c); acc.append(float(p.held_out_super_acc())); grp.append(p.l2_grouping())
        rows[f"ld={ld}"] = {"l2_ld": ld, "held_within": float(np.mean(wi)), "held_cross": float(np.mean(cr)),
                            "super_acc": float(np.mean(acc)), "l2_group": float(np.mean(grp)),
                            "super_acc_per_seed": acc}
    if verbose:
        print(f"\n=== EMERGE-48 NUMPY SWEEP (L2 soft/union depression, normalize={normalize}) ===")
        print(f"  {'l2_ld':<10}{'held-within':>12}{'held-cross':>12}{'super-acc':>11}{'L2-group':>10}")
        for name, r in rows.items():
            print(f"  {r['l2_ld']:<10.3f}{r['held_within']:>12.3f}{r['held_cross']:>12.3f}"
                  f"{r['super_acc']:>11.2f}{r['l2_group']:>+10.2f}")
        print()
    return rows


# =====================================================================================================================
# ON-SUBSTRATE PORT — EMERGE-46 OnSubstratePooler with a SOFT L2 ld_wi (+ optional EMERGE-47 normalization)
# =====================================================================================================================
def _build_onsubstrate_probe():
    """Lazy import of EMERGE-46's on-substrate pooler + EMERGE-47's normalized subclass; returns a probe class whose L2
    on-substrate pooler uses a SOFT winner-inactive depression rate (ld_wi) for union pooling."""
    from research.runners._emerge46_spiking_stacked_pooler_derisk import (
        OnSubstratePooler, _build_cells_bridge, M_INHERIT, POOL_LD_WI,
        NCOL1 as E46_NCOL1, NCOL2 as E46_NCOL2, NF as E46_NF, K1 as E46_K1, K2 as E46_K2,
        POOL_EPOCHS as E46_POOL_EPOCHS, L2_EPOCHS as E46_L2_EPOCHS,
    )
    from research.runners._emerge47_l2_input_normalization_derisk import _build_onsubstrate_probe as _e47_builder
    # reuse the NormalizedOnSubstratePooler defined inside EMERGE-47's builder closure by re-deriving it here (it only
    # overrides _drive to apply in_weights); simplest is to redefine the same subclass locally so we control ld_wi too.

    class SoftNormalizedOnSubstratePooler(OnSubstratePooler):
        """EMERGE-46's on-substrate pooler with (1) a per-input-column LOCAL-NORMALIZATION vector applied to the drive
        (EMERGE-47; in_weights=None disables it) and (2) the SOFT winner-inactive depression rate `ld_wi` from the
        constructor (soft/union pooling). The learning kernels are UNCHANGED — `ld_wi` is the committed kernel's
        `lam_dep_wi` argument. in_weights=None + ld_wi=POOL_LD_WI == EMERGE-46 exactly."""

        def __init__(self, *args, in_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_weights = None if in_weights is None else np.asarray(in_weights, float)

        def _drive(self, feats, boost=None):
            active = np.zeros(self.n_in); active[list(feats)] = 1.0
            if self.in_weights is not None:
                active = active * self.in_weights                 # <-- L2-INPUT LOCAL NORMALIZATION (EMERGE-47)
            data = _host(self.b.cp_connections.data)
            contrib = active[self.ff_feat] * (data[self.ff_pos] > 0.5)
            drive = np.zeros(self.n_col); np.add.at(drive, self.ff_col, contrib)
            return drive * boost if boost is not None else drive

    class SoftL2SpikingStackedPoolerProbe:
        def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False,
                     normalize=False, permute_stats=False, l2_ld=L2_LD_SOFT_DEFAULT):
            self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
            self.feats = {}
            for i, (m, k) in enumerate(self.mem.items()):
                r = np.random.default_rng(seed * 100 + i)
                self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
            # L1: on-substrate pooler at the NORMAL discriminative ld_wi (POOL_LD_WI) -> sub-category codons
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
            # L2: on-substrate pooler with the SOFT/UNION ld_wi (+ optional normalization)
            l2 = SoftNormalizedOnSubstratePooler(seed=seed + 1, n_in=E46_NCOL1, n_col=E46_NCOL2, k_win=E46_K2,
                                                 ld_wi=l2_ld, in_weights=in_w)
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

    return SoftL2SpikingStackedPoolerProbe


def _onsubstrate_run(seeds=(42, 43, 44), epochs=40, l2_ld=L2_LD_SOFT_DEFAULT, normalize=False, verbose=True):
    """Port to the on-substrate pooler: the soft/union L2 (low ld_wi) + optional normalization, with the full anti-cheat
    arms, 3-seed. Compare super-acc vs EMERGE-46's 0.03."""
    Probe = _build_onsubstrate_probe()
    arms = {
        "stacked_soft": dict(),
        "permuted_cooc": dict(permute=True),
        "dap_lesion": dict(lesion=True),
        "l2lesion": dict(l2_lesion=True),                                        # reported-not-gated
    }
    if normalize:
        arms["permuted_stats"] = dict(permute_stats=True)
    rows = {}
    for name, kw in arms.items():
        wi, cr, acc, grp = [], [], [], []
        for s in seeds:
            p = Probe(seed=s, epochs=epochs, l2_ld=l2_ld, normalize=normalize, **kw)
            w, c = p.held_out_within_cross_overlap()
            wi.append(w); cr.append(c); acc.append(float(p.held_out_super_acc())); grp.append(p.l2_grouping())
            if verbose:
                print(f"    [{name} seed {s}] within {w:.3f} cross {c:.3f} super-acc {acc[-1]:.2f} L2-group {grp[-1]:+.2f}",
                      flush=True)
        rows[name] = {"held_within": float(np.mean(wi)), "held_cross": float(np.mean(cr)),
                      "super_acc": float(np.mean(acc)), "l2_group": float(np.mean(grp)),
                      "super_acc_per_seed": acc, "held_within_per_seed": wi, "held_cross_per_seed": cr}
    return rows


# =====================================================================================================================
def _demo(seed=42, epochs=40):
    rows = _numpy_sweep(seeds=(seed,), epochs=epochs, verbose=True)
    strong, soft = rows[f"ld={POOL_LD_STRONG}"], rows[f"ld={L2_LD_SOFT_DEFAULT}"]
    print(f"  DEMO: softening L2 depression {POOL_LD_STRONG}->{L2_LD_SOFT_DEFAULT} lifts held-out super-acc "
          f"{strong['super_acc']:.2f} -> {soft['super_acc']:.2f} "
          f"(within {strong['held_within']:.3f}->{soft['held_within']:.3f}, cross {strong['held_cross']:.3f}->{soft['held_cross']:.3f})\n")


def _verdict(sweep, onsub, l2_ld, normalize):
    """Compose the GO/BOUNDARY verdict. The load-bearing test is the ON-SUBSTRATE stacked_soft arm vs 0.80 + controls."""
    strong = sweep.get(f"ld={POOL_LD_STRONG}") if sweep else None
    soft = sweep.get(f"ld={l2_ld}") if sweep else None

    onsub_go = None; flags = {}
    if onsub is not None:
        st = onsub["stacked_soft"]; perm = onsub["permuted_cooc"]; dap = onsub["dap_lesion"]
        pstats = onsub.get("permuted_stats", {"super_acc": 0.0, "held_within": 0.0, "held_cross": 0.0})
        acc = st["super_acc"]; grp = st["l2_group"]
        # the shortcut guard: within-super held-out overlap must exceed cross-super (generalization, not collision)
        disc = bool(st["held_within"] - st["held_cross"] >= 0.02)
        gate_go = bool(acc >= 0.80 and grp >= 0.15 and acc >= perm["super_acc"] + 0.25 and acc >= dap["super_acc"] + 0.30)
        if normalize:
            gate_go = gate_go and bool(acc >= pstats["super_acc"] + 0.20)
        onsub_go = bool(gate_go and disc)
        flags = {"acc": acc, "grp": grp, "perm": perm["super_acc"], "dap": dap["super_acc"],
                 "disc": disc, "within": st["held_within"], "cross": st["held_cross"],
                 "permuted_stats": pstats["super_acc"] if normalize else None}

    if onsub is not None:
        st = onsub["stacked_soft"]; perm = onsub["permuted_cooc"]; dap = onsub["dap_lesion"]
        l2l = onsub["l2lesion"]["super_acc"]
        pstats_txt = f"; permuted-stats {onsub['permuted_stats']['super_acc']:.2f}" if normalize else ""
        if onsub_go:
            verdict = (f"GO -- SOFT/UNION L2 POOLING SURPASSES the EMERGE-46 fully-spiking-stacked-pooler BOUNDARY ON-SUBSTRATE. "
                       f"Lowering the L2 winner-inactive depression rate to ld_wi={l2_ld}"
                       f"{' + L2-input normalization' if normalize else ''} makes multiple L2 columns strengthen on similar "
                       f"inputs (HTM temporal pooler / HMAX soft-max), so a HELD-OUT ENTIRE sub-category SHARES its "
                       f"same-superordinate columns and inherits: on-substrate super-acc {st['super_acc']:.2f} (vs EMERGE-46's "
                       f"0.03, chance {1/NSUPER:.2f}), held-out within-super L2 overlap {st['held_within']:.3f} > cross-super "
                       f"{st['held_cross']:.3f} (GENERALIZATION, not indiscriminate collision), L2-group {st['l2_group']:+.2f}. "
                       f"GATED CONTROLS: PERMUTED-co-occurrence {perm['super_acc']:.2f} (input-destruction collapses); "
                       f"dAP-LESION {dap['super_acc']:.2f} (coincidence read load-bearing){pstats_txt}. REPORTED-secondary: "
                       f"L1->L2 lesion {l2l:.2f}. => the EMERGE-46 residual was OVER-SELECTIVE winner-inactive depression (a "
                       f"pool-for-invariance misalignment), fixed by SOFT/UNION pooling, NOT an irreducible point-neuron limit. "
                       f"The winner-inactive kernel already takes lam_dep_wi as a soft rate -- NO NEW sim/ edit. 3-seed "
                       f"(42/43/44); 6-seed is a cheap confirmation follow-on.")
        else:
            miss = []
            if st["super_acc"] < 0.80: miss.append(f"super-acc {st['super_acc']:.2f} < 0.80")
            if st["l2_group"] < 0.15: miss.append(f"L2-group {st['l2_group']:+.2f} < 0.15")
            if st["super_acc"] < perm["super_acc"] + 0.25: miss.append(f"permuted didn't collapse ({st['super_acc']:.2f} vs {perm['super_acc']:.2f})")
            if st["super_acc"] < dap["super_acc"] + 0.30: miss.append(f"dAP-lesion didn't collapse ({st['super_acc']:.2f} vs {dap['super_acc']:.2f})")
            if st["held_within"] - st["held_cross"] < 0.02: miss.append(f"no within>cross discrimination (within {st['held_within']:.3f} vs cross {st['held_cross']:.3f} = collision, not generalization)")
            verdict = (f"BOUNDARY (on-substrate) -- SOFT/UNION L2 pooling (ld_wi={l2_ld}"
                       f"{' + normalization' if normalize else ''}) does NOT reach the on-substrate GO: " + "; ".join(miss) +
                       f". on-substrate stacked super-acc {st['super_acc']:.2f} (EMERGE-46 was 0.03), within {st['held_within']:.3f} "
                       f"cross {st['held_cross']:.3f}; permuted-cooc {perm['super_acc']:.2f}, dAP-lesion {dap['super_acc']:.2f}. "
                       f"NEXT RUNG: the Foldiak (1991) trace / temporal-continuity rule (a slow eligibility trace pooling "
                       f"features that co-occur in TIME; needs grouped/curriculum presentation). NO NEW sim/ edit.")
    else:
        # numpy-only sweep verdict
        best = max(sweep.values(), key=lambda r: r["super_acc"]) if sweep else None
        if best is not None and best["super_acc"] >= 0.80 and best["held_within"] > best["held_cross"] + 0.02:
            verdict = (f"NUMPY-SWEEP PROMISING -- softening the L2 depression recovers held-out generalization "
                       f"(over-selective ld={POOL_LD_STRONG} super-acc {strong['super_acc']:.2f} -> soft ld={best['l2_ld']} "
                       f"super-acc {best['super_acc']:.2f}, within {best['held_within']:.3f} > cross {best['held_cross']:.3f}). "
                       f"=> PORT to on-substrate (run with --onsubstrate).")
        else:
            verdict = (f"NUMPY-SWEEP BOUNDARY -- softening the L2 depression does NOT cleanly recover held-out generalization. "
                       f"Do NOT force a GO. NEXT RUNG: the Foldiak trace rule.")
    return verdict, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--l2-ld", type=float, default=L2_LD_SOFT_DEFAULT, help="soft/union L2 winner-inactive depression rate")
    ap.add_argument("--normalize", action="store_true", help="also apply EMERGE-47 L2-input normalization")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--numpy-sweep", action="store_true", help="numpy L2-ld sweep only (fast)")
    ap.add_argument("--onsubstrate", action="store_true", help="also run the on-substrate port (slow, DECISIVE)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0], a.epochs); return 0
    if len(a.seeds) < 3 and not a.demo:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2

    t0 = time.time(); err = None; sweep = None; onsub = None
    try:
        print(f"EMERGE-48: SOFT/UNION L2 POOLING (low ld_wi={a.l2_ld}, normalize={a.normalize}) to surpass the EMERGE-46 "
              f"stacked-pooler boundary", flush=True)
        sweep = _numpy_sweep(seeds=tuple(a.seeds), epochs=a.epochs, normalize=a.normalize, verbose=True)
        if a.onsubstrate:
            print("  porting to the on-substrate pooler (slow bridge builds)...", flush=True)
            onsub = _onsubstrate_run(seeds=tuple(a.seeds), epochs=a.epochs, l2_ld=a.l2_ld,
                                     normalize=a.normalize, verbose=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        verdict, flags = _verdict(sweep, onsub, a.l2_ld, a.normalize)
    else:
        verdict, flags = f"ERROR -- {err}", {}

    summary = {"probe": "emerge48_soft_l2_pooling", "verdict": verdict, "flags": flags,
               "mechanism": "SOFT/UNION L2 POOLING: the L2 OnSubstratePooler is built with a LOW winner-inactive depression "
                            "rate ld_wi (the committed fused_htm_winner_inactive_depression's lam_dep_wi argument, 0=off), so "
                            "multiple L2 columns strengthen on similar inputs (HTM temporal pooler, Hawkins-Ahmad 2016; HMAX "
                            "soft-max, Serre-Poggio 2005) -> same-superordinate members (incl. a held-out sub-category) SHARE "
                            "L2 columns -> inheritance routes. L1 stays at the normal discriminative ld_wi. Optionally combined "
                            "with EMERGE-47 L2-input local normalization (an additive secondary lift).",
               "task": "EMERGE-44/46 6-sub-cat -> 2-superordinate; hold out ENTIRE sub-categories {2,5}; L2 on-substrate pooler "
                       "with soft/union ld_wi; measure held-out within/cross overlap + super-acc; anti-cheats permuted-cooc + "
                       "dAP-lesion (+ permuted-stats if normalize), l2lesion reported-not-gated",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "l2_ld": a.l2_ld, "normalize": a.normalize,
                                            "n_col1": NCOL1, "n_col2": NCOL2, "k1": K1, "k2": K2, "n_super": NSUPER},
               "numpy_sweep": sweep, "onsubstrate": onsub,
               "elapsed_seconds": round(time.time() - t0, 1),
               "sim_edit": "NONE (NO NEW sim/ edit) -- the soft L2 depression is a LOWER value of the ALREADY-COMMITTED kernel's "
                           "lam_dep_wi argument; reuse-by-import of EMERGE-44/46/47 poolers; every existing sim/ path byte-unchanged",
               "HONEST_NOTE": "EMERGE-47 root-caused the EMERGE-46 boundary as OVER-SELECTIVE winner-inactive depression and "
                              "proved (numpy sweep) that softening it is the DOMINANT lever (super-acc 0.06 at ld=0.15 -> 0.97 at "
                              "ld=0.02). This de-risk PORTS the soft/union pooling to the real substrate (a lower L2 ld_wi). "
                              "The shortcut guard: within-super held-out overlap must EXCEED cross-super (generalization, not "
                              "indiscriminate collision), and permuted must still collapse. Winner SELECTION is a host top-k over "
                              "the on-substrate drive (EMERGE-41 has the spiking FS-WTA version). 3 seeds (42/43/44); 6-seed is a "
                              "cheap confirmation follow-on."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge48] VERDICT: {verdict}", flush=True)
    print(f"[emerge48] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
