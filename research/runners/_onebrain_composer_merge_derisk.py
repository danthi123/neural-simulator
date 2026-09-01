"""ONE-BRAIN MERGE de-risk — the RECALL COMPOSER bridge + the SURPRISE organ on ONE shared spiking substrate.

THE DEEPEST ONE-SUBSTRATE RUNG (this lane's mission)
----------------------------------------------------
Production now has 4 Gate-B organs sharing substrates across 2 pools (surprise+world-model,
metacog+pragmatic). But BOTH pools are separate from the **recall COMPOSER bridge** -- the central
spiking organ the whole `/api/brain-chat` turn is built around (the RF-phasor VSA composer:
`query_patient` / the no-confab moat). Merging an ORGAN with the composer is the deepest one-substrate
step, because the composer is what every organ reads. This runner DE-RISKS it (NOT a production flip):
can the SURPRISE organ share ONE `SimulationBridge` with the RF-phasor recall composer, byte-identical,
with the moat intact + a genuine cross-organ synapse?

THE TWO ORGANS, TWO CODES ON ONE POOL
-------------------------------------
* COMPOSER (recall): the production `RFPhasorComposer` (the `/api/brain-chat` recall organ). Its ops are
  RF-phasor resonate-and-fire: `rf_kick` / `rf_resonate_steps` / `rf_read_phases` over complex synapses,
  with a host-argmax cleanup + the no-confab MOAT (`query_patient` returns None when no stored fact's cue
  matches). On the shared bridge it runs on a masked SLICE (the `SharedBridgeComposer` index-shift port
  from the 2026-07-20 composer+WKV CAPSTONE): rebase the bind/unbind/bundle conns by `rf_base`, kick with
  `neuron_mask`=composer-slice, read the slice. The RF ops use the `rf_resonate_steps` FAST PATH, which
  BYPASSES `_run_one_simulation_step` entirely and (with the mask) writes ONLY the composer slice's v/u.
* SURPRISE (expectation-violation): the D2 organ (`_spiking_expectation_rpe_derisk`, 6/6 GO): cue --Hebbian
  topographic--> patient_expected(FS/PV, GABA_A) --> surprise <-- patient_asserted(exc). IZHIKEVICH spiking
  + Hebbian learning + homeostasis + the merge flags (`per_region_threshold_heterogeneity`,
  `per_region_homeostasis_isolation`). Its `_step` runs the full Izhikevich `_run_one_simulation_step`.

WHY BYTE-IDENTITY HOLDS (the mechanism)
---------------------------------------
The two organs read through DIFFERENT machinery on the SAME `cp_membrane_potential_v`:
  - The composer's RF ops (`rf_resonate_steps`) never call `_run_one_simulation_step`, never read
    `cp_neuron_firing_thresholds` / the Hebbian / homeostasis code, and (masked) write only the composer
    slice. Its complex weights (`cp_rf_w_*`) live in composer-region rows/cols only. So the composer recall
    is INVARIANT to the surprise organ's Izhikevich state + to Hebbian/homeostasis being ON -> byte-identical
    to a standalone `RFPhasorComposer` (its own per-op RF bridges).
  - The surprise organ's Izhikevich `_step` touches every neuron, but the composer region carries no
    pathway to/from the surprise regions (in the byte-identity config), stays at REST (undriven), and is
    FROZEN by `per_region_homeostasis_isolation`. So the surprise read is byte-identical to the standalone
    surprise organ (the rung-1 result, `2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md` +
    `...-homeostasis-GO.md`).

THE CROSS-ORGAN SYNAPSE (honest)
--------------------------------
A `composer -> surprise` edge in the shared `cp_connections` IS load-bearing WHEN its source (composer-
region) neurons emit Izhikevich SPIKES (current-driven): lesion it -> the interaction collapses (proves the
pool is genuinely ONE + a same-code synapse acts). BUT the composer's RF-phasor RECALL leaves those neurons
in a PHASE state (|Z|~1, not an Izhikevich spike train), and its `rf_resonate_steps` fast path never
traverses `cp_connections` -- so the composer's actual recall does NOT natively drive the edge. The precise
boundary is the RF-phasor <-> spike-rate CODE gap; the named engine feature to close it is a PHASE->SPIKE
TRANSDUCER region (generalize the composer's existing spiking-cleanup RF-membrane->Izhikevich-WTA read into
a first-class shared-bridge primitive) so the recall itself drives the cross-organ synapse.

VERDICT
-------
GO on the merge: one shared pool + composer recall byte-identical (delta 0.0) + moat preserved + surprise
read byte-identical (delta 0.0) + determinism, AND a load-bearing cross-organ synapse on the shared pool.
BOUNDARY on the recall-DRIVEN cross-organ interaction: the RF-phasor recall cannot natively drive an
Izhikevich cross-synapse (phase != rate) -> the phase->spike transducer is the named next feature.

NO `sim/` edit; reuse-by-import; CPU-friendly (numpy). Run:
    SIM_BACKEND=numpy python -m research.runners._onebrain_composer_merge_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_merge_6seed.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    train_expectation,
    measure_conditions,
    _idx,
    _install_block_diagonal,
    _step,
    _hard_reset,
    _host,
)
from research.runners.rf_phasor_composer import RFPhasorComposer

# The grounded facts + vocab the composer stores (the CAPSTONE panel).
FACTS = [("dog", "chase", "cat"), ("owl", "eat", "mouse"), ("wolf", "hunt", "deer")]
VOCAB = sorted({w for f in FACTS for w in f})
UNSTORED_CUE = ("lion", "roar")   # never stored -> the no-confab moat must abstain (None)

# Surprise organ build params (match the production surprise organ / rung-1).
_SURPRISE_KW = dict(n_trained=8, n_novel=4, blk=24, cue_blk=24, cue_to_expected_weight=0.8)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The shared-bridge composer (the CAPSTONE index-shift port).
# ─────────────────────────────────────────────────────────────────────────────────────────────
class SharedBridgeComposer(RFPhasorComposer):
    """`RFPhasorComposer` whose RF resonate ops run on a masked SLICE of a shared bridge. Every RF op
    (`_resonate`) rebases the connections by `rf_base`, kicks with `neuron_mask`=composer-slice, resonates
    (the masked fast path writes only the composer slice), and reads the slice. All other composer state
    (concept/role codes, kb, moat) is unchanged -> the recall + moat are byte-identical to a standalone
    `RFPhasorComposer` iff the masked RF ops reproduce a dedicated 100%-RF bridge (they do; CAPSTONE)."""

    def bind_to_shared(self, merged, cmp_idx):
        self._merged = merged
        cmp_idx = np.asarray(cmp_idx)
        self._rf_base = int(cmp_idx.min())
        self._rf_size = int(len(cmp_idx))
        n = int(merged.core_config.num_neurons)
        m = np.zeros(n, dtype=bool)
        m[cmp_idx] = True
        self._rf_mask = m

    def _resonate(self, n, conns, kick, period=None):
        per = self.period if period is None else int(period)   # finer-period "second look" (decode escalation)
        n = int(n)
        if n > self._rf_size:
            raise ValueError(f"RF op needs {n} neurons but composer region is {self._rf_size}")
        b = self._merged
        N = int(b.core_config.num_neurons)
        base = self._rf_base
        shifted = [(base + int(post), base + int(pre), w) for (post, pre, w) in conns]
        b.rf_set_complex_weights(shifted)
        full_kick = np.zeros(N, dtype=np.complex128)
        kk = np.asarray(kick, dtype=np.complex128).reshape(-1)
        full_kick[base:base + n] = kk[:n]
        b.rf_kick(full_kick, period=per, lam=0.0, neuron_mask=self._rf_mask)
        b.rf_resonate_steps(per + 8)
        phases = np.asarray(b.rf_read_phases())
        return phases[base:base + n]


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Build the merged bridge: surprise organ regions + a composer region (+ opt. cross-organ edge).
# ─────────────────────────────────────────────────────────────────────────────────────────────
_SURPRISE_REGIONS = ("cue", "patient_expected", "patient_asserted", "surprise")


def build_merged(seed, D_cmp, *, per_region_thresh=True, homeo=True, homeo_iso=True,
                 with_cross=False, cross_weight=0.0):
    """ONE `SimulationBridge` holding the surprise organ's 4 regions + a `composer` region (sized for the
    RF ops: max(7, 2*K)*D so a 6-role encode/bundle AND the K-fact batched moat scan both fit). Config
    replicates `build_expectation_circuit` exactly (Izhikevich, Hebbian, homeostasis, GABA_B inert) + the
    two merge flags. `with_cross` adds a `composer -> surprise` pathway (the cross-organ synapse; kept at
    `cross_weight`, block-diagonal composer-block-c -> surprise-block-c). NO `sim/` edit."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel

    # SPEC EXTRACTION (reuse-by-import): a throwaway standalone surprise build -> its real region/pathway
    # specs + meta. per-region-thresh on the throwaway is irrelevant (specs are flag-independent).
    _brS, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=per_region_thresh, **_SURPRISE_KW)
    blk = metaS["blk"]
    cmp_n = max(7, 2 * len(FACTS)) * D_cmp

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_region_threshold_heterogeneity = bool(per_region_thresh)   # merge flag #1 (INIT byte-identity)
    cfg.per_region_homeostasis_isolation = bool(homeo_iso)             # merge flag #2 (idle-drift byte-identity)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.06
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0
    cfg.enable_homeostasis = bool(homeo)

    cfg.brain_regions = list(cfgS.brain_regions) + [
        BrainRegion(name="composer", n_neurons=cmp_n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = list(cfgS.region_pathways)
    if with_cross:
        # THE CROSS-ORGAN SYNAPSE: composer -> surprise (built full, masked block-diagonal after build).
        cfg.region_pathways = cfg.region_pathways + [
            RegionPathway(from_region="composer", to_region="surprise",
                          density=1.0, weight_mean=float(cross_weight), weight_jitter=0.0, plastic=False),
        ]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Surprise organ's topographic block-diagonal wiring (built full then masked concept c -> block c).
    _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, metaS["W_exc"])
    _install_block_diagonal(bridge, "patient_expected", "surprise", blk, metaS["W_inh"])
    _install_block_diagonal(bridge, "cue", "patient_expected", blk, float(_SURPRISE_KW["cue_to_expected_weight"]))
    if with_cross:
        # composer block c -> surprise block c (topographic), weight cross_weight. blk of composer = surprise blk.
        _install_block_diagonal(bridge, "composer", "surprise", blk, float(cross_weight))
    bridge._blk = blk
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, metaS


def _surp_idx_map(bridge, xp):
    return {r: xp.asarray(_idx(bridge, r)) for r in _SURPRISE_REGIONS}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Read-isolation: a surprise read must leave NO footprint on the composer slice (and vice-versa).
# ─────────────────────────────────────────────────────────────────────────────────────────────
_PER_NEURON_STATE = (
    "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
    "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
    "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
)


@contextlib.contextmanager
def restore_composer_slice(bridge, cmp_idx, xp):
    """Snapshot the FULL per-neuron state, run the block, then RESTORE the composer slice (so a surprise
    read/train leaves the composer's neural state exactly as it was -- there is no surprise->composer edge in
    the byte-identity config, so this only guards against an incidental homeostatic footprint)."""
    keep = xp.ones(int(bridge.cp_membrane_potential_v.shape[0]), dtype=bool)
    keep[xp.asarray(cmp_idx)] = False   # False over composer neurons -> they get restored
    snaps = []
    for name in _PER_NEURON_STATE:
        arr = getattr(bridge, name, None)
        snaps.append(None if arr is None else arr.copy())
    try:
        yield
    finally:
        for name, snap in zip(_PER_NEURON_STATE, snaps):
            if snap is None:
                continue
            cur = getattr(bridge, name)
            setattr(bridge, name, xp.where(keep, cur, snap))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Hashing / helpers.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _arr_hash(a):
    if a is None:
        return "None"
    return hashlib.sha256(np.asarray(_host(a)).astype(np.float64).tobytes()).hexdigest()[:16]


def _maxerr_lists(m, s, keys):
    e = 0.0
    for k in keys:
        for a, b in zip(m[k], s[k]):
            e = max(e, abs(float(a) - float(b)))
    return e


def _install_full_pathway_weight(bridge, src, dst, weight):
    """Set EVERY src->dst edge to `weight` (used to LESION the cross-organ pathway to 0). CSR-orientation
    robust (mirrors _install_block_diagonal's detection)."""
    import scipy.sparse as sp
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    n_rows = M.shape[0]
    row_is_dst = row_is_src = 0
    for r in range(n_rows):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    n_set = 0
    for r in range(n_rows):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_idx and post in dst_idx:
                data[off] = float(weight); n_set += 1
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    return n_set


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Cross-organ interaction reads (a CONFIRM read of the surprise organ, with the composer block driven
#  either by injected CURRENT (a transducer stand-in) or holding its RF-phasor RECALL state).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _confirm_surprise_with_composer(bridge, surp_idx, meta, xp, cmp_idx, *, composer_drive,
                                    fact=0, cue_pa=600.0, assert_pa=600.0, hold=60, pre_steps=60,
                                    composer_pa=600.0):
    """CONFIRM read of surprise fact `i` (surprise is ~0 Hz when the prediction cancels the assertion),
    while the composer block-`i` is driven by `composer_drive`:
      - "none":    composer left at rest (baseline).
      - "current": composer block-i gets `composer_pa` external current (a SPIKING transducer stand-in).
      - "rf":      the composer block holds its post-RF-op PHASE state (|Z|~1, set on the slice) -- the
                   composer's native recall readout, NOT an Izhikevich spike train.
    Returns the surprise-pool mean Hz in the assertion window."""
    blk = bridge._blk
    _hard_reset(bridge)
    cmp_idx = np.asarray(cmp_idx)
    # optionally pre-load the composer slice with a phasor readout state (the "rf" mode).
    if composer_drive == "rf":
        # set composer block-i's v/u to a unit-magnitude phasor (|Z|=1), the RF read-out state.
        cblk = cmp_idx[fact * blk:(fact + 1) * blk]
        ph = xp.asarray(np.cos(np.linspace(0, 2 * np.pi, len(cblk), endpoint=False)), dtype=bridge.cp_membrane_potential_v.dtype)
        pi = xp.asarray(np.sin(np.linspace(0, 2 * np.pi, len(cblk), endpoint=False)), dtype=bridge.cp_recovery_variable_u.dtype)
        bridge.cp_membrane_potential_v[xp.asarray(cblk)] = ph
        bridge.cp_recovery_variable_u[xp.asarray(cblk)] = pi
    # PREDICTION phase: cue alone (settle the expectation).
    bridge.cp_external_input_current[:] = 0.0
    cue = surp_idx["cue"]
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    for _ in range(pre_steps):
        _step(bridge)
    # ASSERTION phase: cue + asserted TRUE patient i (confirm) + optional composer current.
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    pa = surp_idx["patient_asserted"]
    bridge.cp_external_input_current[pa[fact * blk:(fact + 1) * blk]] = xp.float32(assert_pa)
    if composer_drive == "current":
        cblk = cmp_idx[fact * blk:(fact + 1) * blk]
        bridge.cp_external_input_current[xp.asarray(cblk)] = xp.float32(composer_pa)
    surp = surp_idx["surprise"]
    counts = 0
    for _ in range(hold):
        _step(bridge)
        counts += int(bridge.cp_firing_states[surp].sum())
    bridge.cp_external_input_current[:] = 0.0
    return counts / max(len(_host(surp)), 1) / (hold * 1e-3)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One seed.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, *, D_cmp=64, n_reps=22, cross_weight=8.0, verbose=True):
    from sim.backend import get_backend
    from tools.lab import attributable_to
    xp, _ = get_backend()

    # ── DETERMINISM: two FRESH merged builds at the same seed -> identical substrate. ──
    d1, _, _ = build_merged(seed, D_cmp)
    d2, _, _ = build_merged(seed, D_cmp)
    det_ok = (_arr_hash(d1.cp_membrane_potential_v) == _arr_hash(d2.cp_membrane_potential_v)
              and _arr_hash(d1.cp_connections.tocsr().data) == _arr_hash(d2.cp_connections.tocsr().data)
              and _arr_hash(d1.cp_neuron_firing_thresholds) == _arr_hash(d2.cp_neuron_firing_thresholds))

    # ── THE MERGED BRIDGE (byte-identity config: NO cross edge). ──
    merged, cfg_m, meta = build_merged(seed, D_cmp)
    cmp_idx = _idx(merged, "composer")
    surp_idx = _surp_idx_map(merged, xp)
    n_all = int(merged.core_config.num_neurons)
    n_surp = sum(len(_host(surp_idx[r])) for r in surp_idx)
    n_cmp = len(cmp_idx)
    # ONE POOL: one cp_membrane_potential_v holds BOTH organs' neurons; the composer region is contiguous.
    v = merged.cp_membrane_potential_v
    one_pool = bool(int(v.shape[0]) == n_all and n_all >= n_surp + n_cmp
                    and int(cmp_idx.max()) < n_all
                    and all(int(_host(surp_idx[r]).max()) < n_all for r in surp_idx)
                    and (int(cmp_idx.max()) - int(cmp_idx.min()) + 1 == n_cmp))

    # ── (1) SURPRISE READ byte-identity: train + read on the merged bridge (composer idle at rest,
    #    frozen by homeo-iso) vs a STANDALONE surprise organ (same config path). ──
    train_expectation(merged, cfg_m, surp_idx, meta, xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False
    with restore_composer_slice(merged, cmp_idx, xp):
        resM = measure_conditions(merged, cfg_m, surp_idx, meta, xp)

    brS, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
    brS._blk = metaS["blk"]
    cfgS.enable_homeostasis = True
    cfgS.per_region_homeostasis_isolation = True
    idxS = _surp_idx_map(brS, xp)
    train_expectation(brS, cfgS, idxS, metaS, xp, n_reps=n_reps)
    cfgS.enable_hebbian_learning = False
    resS = measure_conditions(brS, cfgS, idxS, metaS, xp)

    surprise_maxerr = _maxerr_lists(resM, resS, ["confirm_per", "contradict_per", "novel_per"])
    surprise_byte_id = bool(surprise_maxerr <= 1e-9)
    surp_sep = resM["contradict_hz"] / max(resM["confirm_hz"], 1e-6)   # faculty must SEPARATE (not exact-of-dead)
    surp_alive = bool(surp_sep >= 5.0)

    # ── (2) COMPOSER RECALL + MOAT byte-identity: the shared-bridge composer vs a standalone RFPhasorComposer. ──
    iso = RFPhasorComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    for a, vb, p in FACTS:
        iso.store(a, vb, p)
    iso_ans = [iso.query_patient(a, vb) for a, vb, p in FACTS]
    iso_abstain = iso.query_patient(*UNSTORED_CUE)

    sh = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    sh.bind_to_shared(merged, cmp_idx)
    # INTERLEAVE ISOLATION (the CAPSTONE's strongest test): a composer store+query on the shared bridge must
    # leave the SURPRISE organ's neural slice byte-identical (the composer's masked RF ops write ONLY the
    # composer slice; there is no composer->surprise synapse here). Snapshot surprise-slice v/u/thresholds,
    # run the whole composer STORE+QUERY sequence, then measure the surprise slice's max delta.
    surp_all = np.concatenate([np.asarray(_host(surp_idx[r])) for r in _SURPRISE_REGIONS])
    _snap = {nm: np.asarray(_host(getattr(merged, nm)))[surp_all].copy()
             for nm in ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_neuron_firing_thresholds")
             if getattr(merged, nm, None) is not None}
    for a, vb, p in FACTS:
        sh.store(a, vb, p)
    sh_ans = [sh.query_patient(a, vb) for a, vb, p in FACTS]
    sh_abstain = sh.query_patient(*UNSTORED_CUE)
    interleave_maxerr = 0.0
    for nm, before in _snap.items():
        after = np.asarray(_host(getattr(merged, nm)))[surp_all]
        interleave_maxerr = max(interleave_maxerr, float(np.abs(after - before).max()))
    composer_op_isolated = bool(interleave_maxerr <= 1e-9)

    recall_byte_id = bool(sh_ans == iso_ans)
    moat_preserved = bool(sh_abstain is None and iso_abstain is None and sh_abstain == iso_abstain)
    recall_correct = bool(sh_ans == [p for _a, _v, p in FACTS])   # the recall is actually RIGHT (not exact-of-garbage)

    # ── (3) CROSS-ORGAN SYNAPSE (composer -> surprise) on a bridge WITH the edge. ──
    xbridge, cfg_x, meta_x = build_merged(seed, D_cmp, with_cross=True, cross_weight=cross_weight)
    xsurp = _surp_idx_map(xbridge, xp)
    xcmp = _idx(xbridge, "composer")
    train_expectation(xbridge, cfg_x, xsurp, meta_x, xp, n_reps=n_reps)
    cfg_x.enable_hebbian_learning = False
    # verify the shared-pool cp_connections actually carries composer->surprise edges.
    n_cross_edges = _install_full_pathway_weight(xbridge, "composer", "surprise", cross_weight)  # re-assert intact
    # A) current-driven source (a SPIKING transducer stand-in): the edge IS load-bearing.
    base_none = _confirm_surprise_with_composer(xbridge, xsurp, meta_x, xp, xcmp, composer_drive="none")
    intact_cur = _confirm_surprise_with_composer(xbridge, xsurp, meta_x, xp, xcmp, composer_drive="current")
    _install_full_pathway_weight(xbridge, "composer", "surprise", 0.0)                            # LESION
    lesion_cur = _confirm_surprise_with_composer(xbridge, xsurp, meta_x, xp, xcmp, composer_drive="current")
    _install_full_pathway_weight(xbridge, "composer", "surprise", cross_weight)                   # restore
    interaction_current = intact_cur - base_none
    interaction_current_lesion = lesion_cur - base_none
    cross_frac = attributable_to("composer->surprise interaction @ the cross synapse",
                                 interaction_current, interaction_current_lesion)
    cross_load_bearing = bool(abs(interaction_current) >= 1.0
                              and abs(interaction_current) >= 5.0 * max(abs(interaction_current_lesion), 1e-6)
                              and (cross_frac is None or cross_frac >= 0.8))
    # B) the composer's RF-PHASOR RECALL state does NOT natively drive the edge (the boundary).
    intact_rf = _confirm_surprise_with_composer(xbridge, xsurp, meta_x, xp, xcmp, composer_drive="rf")
    interaction_rf = intact_rf - base_none
    recall_drives_edge = bool(abs(interaction_rf) >= 1.0)   # expected FALSE -> the phase->spike gap

    merge_go = bool(one_pool and det_ok and surprise_byte_id and surp_alive
                    and recall_byte_id and moat_preserved and recall_correct and composer_op_isolated)

    res = {
        "seed": seed, "D_cmp": D_cmp,
        "one_shared_pool": one_pool, "n_all": n_all, "n_surp": n_surp, "n_cmp": n_cmp,
        "determinism_ok": det_ok,
        # (1) surprise read byte-identity
        "surprise_maxerr_hz": float(surprise_maxerr), "surprise_byte_identical": surprise_byte_id,
        "surprise_separation_ratio": float(surp_sep), "surprise_faculty_alive": surp_alive,
        "surprise_merged": {k: resM[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        "surprise_solo": {k: resS[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        # (2) composer recall + moat byte-identity
        "composer_recall_shared": sh_ans, "composer_recall_isolated": iso_ans,
        "composer_recall_byte_identical": recall_byte_id, "composer_recall_correct": recall_correct,
        "moat_shared_abstain": sh_abstain, "moat_isolated_abstain": iso_abstain,
        "moat_preserved": moat_preserved,
        "interleave_maxerr": float(interleave_maxerr), "composer_op_isolated": composer_op_isolated,
        # (3) cross-organ synapse (composer -> surprise)
        "cross_edges_in_pool": int(n_cross_edges),
        "cross_base_none_hz": float(base_none),
        "cross_intact_current_hz": float(intact_cur), "cross_lesion_current_hz": float(lesion_cur),
        "cross_interaction_current_hz": float(interaction_current),
        "cross_interaction_current_lesion_hz": float(interaction_current_lesion),
        "cross_attribution_frac": (float(cross_frac) if cross_frac is not None else None),
        "cross_load_bearing": cross_load_bearing,
        "cross_rf_recall_hz": float(intact_rf), "cross_interaction_rf_hz": float(interaction_rf),
        "recall_drives_edge_natively": recall_drives_edge,   # FALSE = the phase->spike boundary
        "merge_go": merge_go,
    }
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}={n_surp}surp+{n_cmp}cmp) det={det_ok} | "
              f"SURPRISE byte-id err={surprise_maxerr:.2e}({surprise_byte_id}) sep={surp_sep:.1f}x | "
              f"COMPOSER recall {sh_ans}=={iso_ans}->{recall_byte_id} moat={moat_preserved} "
              f"op-isolated={composer_op_isolated}(err={interleave_maxerr:.1e}) | "
              f"CROSS load-bearing(current) intact={interaction_current:+.2f} lesion={interaction_current_lesion:+.2f}Hz "
              f"frac={cross_frac} ({cross_load_bearing}) | RF-recall drives edge={recall_drives_edge} "
              f"(rf int={interaction_rf:+.2f}Hz) | MERGE-GO={merge_go}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--D-cmp", type=int, default=64)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cross-weight", type=float, default=8.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== ONE-BRAIN MERGE: the RECALL COMPOSER bridge + the SURPRISE organ on ONE shared substrate ===")
    results = [run_seed(s, D_cmp=args.D_cmp, n_reps=args.n_reps, cross_weight=args.cross_weight)
               for s in seeds]

    n = len(results)
    def cnt(k):
        return sum(1 for r in results if r[k])
    n_pool = cnt("one_shared_pool"); n_det = cnt("determinism_ok")
    n_surp = cnt("surprise_byte_identical"); n_alive = cnt("surprise_faculty_alive")
    n_recall = cnt("composer_recall_byte_identical"); n_correct = cnt("composer_recall_correct")
    n_moat = cnt("moat_preserved"); n_cross = cnt("cross_load_bearing")
    n_isol = cnt("composer_op_isolated")
    n_merge = cnt("merge_go")
    n_recall_edge = cnt("recall_drives_edge_natively")
    max_surp_err = max(r["surprise_maxerr_hz"] for r in results)
    _gate = lambda k: "GO" if ((n >= 6 and k >= 5) or (n < 6 and k == n)) else "BOUNDARY"

    print("\n=== VERDICT ===")
    print(f"  one shared neuron pool (composer + surprise):   {n_pool}/{n}")
    print(f"  determinism (cfg.seed incl. thresholds):        {n_det}/{n}")
    print(f"  SURPRISE read byte-identical (merged vs solo):  {n_surp}/{n}  -> {_gate(n_surp)}  (max err {max_surp_err:.2e} Hz)")
    print(f"    surprise faculty alive (contradict>>confirm): {n_alive}/{n}")
    print(f"  COMPOSER recall byte-identical (shared vs iso): {n_recall}/{n}  -> {_gate(n_recall)}")
    print(f"    composer recall CORRECT (== stored patients): {n_correct}/{n}")
    print(f"  no-confab MOAT preserved (unstored -> abstain): {n_moat}/{n}  -> {_gate(n_moat)}")
    print(f"  composer op byte-ISOLATED from surprise slice:  {n_isol}/{n}  -> {_gate(n_isol)}  (a composer store+query leaves the surprise organ's v/u/thresholds byte-identical)")
    print(f"  --> MERGE byte-identity GO:                     {n_merge}/{n}  -> {_gate(n_merge)}")
    print(f"  cross-organ synapse LOAD-BEARING (current src): {n_cross}/{n}  -> {_gate(n_cross)}  (the shared pool carries a load-bearing composer->surprise edge)")
    print(f"  composer RF-recall natively DRIVES the edge:    {n_recall_edge}/{n}  (expected 0 -> the RF-phasor <-> spike-rate CODE gap: the composer's recall is a PHASE, not an Izhikevich spike train)")
    print(f"  --> BOUNDARY: a RECALL-DRIVEN cross-organ interaction needs a PHASE->SPIKE TRANSDUCER region")
    print(f"      (generalize the composer's spiking-cleanup RF-membrane->Izhikevich-WTA read into a first-class")
    print(f"       shared-bridge primitive) so the composer's recall itself drives the cross-organ synapse.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "onebrain_composer_merge", "n_seeds": n,
                       "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                       "n_surprise_byte_identical": n_surp, "n_surprise_faculty_alive": n_alive,
                       "n_composer_recall_byte_identical": n_recall, "n_composer_recall_correct": n_correct,
                       "n_moat_preserved": n_moat, "n_composer_op_isolated": n_isol, "n_merge_go": n_merge,
                       "n_cross_load_bearing": n_cross, "n_recall_drives_edge_natively": n_recall_edge,
                       "max_surprise_maxerr_hz": max_surp_err,
                       "merge_verdict": _gate(n_merge), "cross_verdict": _gate(n_cross),
                       "cross_weight": args.cross_weight, "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
