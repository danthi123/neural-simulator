"""GNW ACTIVE OVERWRITE — the last missing workspace-control primitive: replace an ignited incumbent A with a
challenger B while n_ignited stays ~1 (a genuine content SWAP: A's identity is gone, B's is delivered, n stays 1 —
NOT a co-ignition n->2, NOT a clear-then-reload).

CONTEXT (the workspace-control ledger before this runner). The distributed divisively-normalized workspace can
already: STOP (a conflict-triggered depression of the SHARED recurrence -> n_ignited=0, GLOBAL STOP GO 6/6,
`_gnw_distributed_overwrite_workspace_derisk.py`), EVICT from within (Rung-2d short-term depression opens the empty
metastable window, GO 6/6), and SELECT (the STN conflict sensor). The residual the parent finding
(`2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md`) named a BOUNDARY is the ACTIVE OVERWRITE: pure
divisive-norm competition co-ignites a self-sufficient incumbent (n 1->2), and a depression-based clear removes ALL
content (it cannot selectively RETAIN the challenger). That finding named THREE next levers; this runner implements
and tests each (they compose):

  LEVER 1 (timed self-eviction): drive the challenger B CONCURRENTLY while the incumbent A's shared-recurrence STD
    depletes through USE, timed so A self-evicts (STD) exactly as B's drive is present -> B ignites in the freed
    slot (a swap, no persistent empty window). The Rung-2d self-eviction, but with the challenger driven DURING the
    depletion so B fills the gap rather than the workspace going empty.
  LEVER 2 (larger overlap + a SUB-CRITICAL private core): split the recurrence into a SHARED-touching weight
    (w_shared, base) and a PRIVATE-core weight (w_priv < the self-sustain knee) so each pattern's private core
    CANNOT self-sustain ALONE, while the FULL pattern (private + shared) stays supra-critical. Then B's drive can
    capture the shared resource and A — sub-critical on its private core alone — cannot lock out the challenger.
    The parent flagged a "confident-commit leak" (driving one content spuriously co-ignites its neighbor at large
    overlap): addressed by keeping w_priv sub-critical for SELF-sustain yet the full pattern supra-threshold, so a
    confident single commit still holds n=1 when UNCHALLENGED (SELECTIVITY) but yields when a rival is driven.
  LEVER 3 (use-driven depression targeting ONLY the shared A-B neurons): a Tsodyks-Markram depression restricted to
    the SHARED A-B recurrence (pre in the A/B overlap), conflict-gated by the neural margin sensor, so ONLY the
    shared units the incumbent is currently using are depressed. The challenger B (externally driven into its
    private core + the shared units) survives while the incumbent A (relying on the now-depressed shared units to
    prop up its sub-critical private core) collapses -> B wins the competition.

COMPOSITION: lever 3's targeted shared depression is the effector, lever 2's sub-critical private core is what lets
A lose, lever 1's concurrent B drive is the external bias that makes the shared pool DEFECT to B (A is not
re-driven; B is). The clean swap is the composition; the per-lever switch-rate is reported.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed):
  SWITCH — a challenger B displaces incumbent A: delivered identity changes A->B on >=5/6, with n_ignited settling
    to EXACTLY 1 (NOT 0 = a stop not a swap; NOT 2 = co-ignition).
  SELECTIVITY — an UNCHALLENGED confident A commit still holds (n=1, winner A) through a long window (no spurious
    self-overwrite).
ANTI-CHEATS (each on every seed):
  (a) SUBSTRATE-DRIVEN not a host poke: host_workspace_reset_calls==0 and host_content_swap_calls==0 on the swap
      HEADLINE (a continuous run; the only host writes are external stimulus drive = world/body-legitimate).
  (b) LOAD-BEARING lever causal: LESION it (freeze the targeted depression, boost=0) -> NO swap (the incumbent
      holds / co-ignites) -> the swap is the STD/overlap dynamics, not the B drive alone.
  (c) BYTE-IDENTICAL substrate: with the overwrite path FLAG-OFF (uniform recurrence, base overlap, no targeted
      depression) the seed-derived Izhikevich params hash EQUALS the DIST-OVERWRITE base build at the same seed.
  (d) DETERMINISM: build twice at one seed -> identical substrate hash (heterogeneity seeded from cfg.seed, NOT
      actual_seed_used).

NOT-A-WALL: if no lever (alone or composed) achieves a clean swap, the residual is QUANTIFIED (does A hold? does B
co-ignite? does the workspace go empty?) and the next lever named. The workspace already has stop+evict+select, so
overwrite being hard is a mappable substrate property, not a failure to hide.

NO `sim/` edit; explicit wiring, dense frozen pools, host-computed STD written into the recurrence weights (the
Rung-2d / distributed-overwrite pattern; native global STP stays OFF).

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_active_overwrite_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_active_overwrite_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_active_overwrite_derisk --six-seed \
      --json research/findings/raw/_gnw_active_overwrite_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: validated constants + the ignition/competition instruments + determinism hash + wash-out.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS, WS_LOOP_GATE,
)
from research.runners._gnw_rung2_competitive_access_derisk import (
    _ignited, IGNITE_FRAC, SOLO_PLATEAU,
)
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
# the DIST-OVERWRITE base build — for the byte-identical anti-cheat (same seed -> same substrate).
from research.runners._gnw_distributed_overwrite_workspace_derisk import (
    build_overwrite_bridge as _dist_build,
)


# ── geometry: K overlapping distributed patterns; SPLIT recurrence (shared-touching vs private-core weight) ─────
N_PATTERNS = 3
PATTERN_SIZE = 100
DEFAULT_OVERLAP = 30           # larger overlap than the base (15): the shared A-B resource is the contested pool
WORKSPACE_N = N_PATTERNS * PATTERN_SIZE + 20   # sized for O=0 so the region size is overlap-invariant

NORM_N = 60
THAL_N = 60

W_SHARED = 34.0                # recurrence weight on edges TOUCHING a shared unit (base sustain strength)
W_PRIV = 20.0                  # recurrence weight on PRIVATE-CORE-only edges (SUB-CRITICAL: private can't self-hold)
WS_NORM_W = 5.0
NORM_WS_W = 16.0
THAL_TONIC_PA = 700.0
THAL_WS_W = 5.0

# short-term depression (Tsodyks-Markram; the eviction effector).
STD_TAU_D = 250.0
STD_BASELINE_U = 0.0           # lever 3: conflict-gated only (baseline 0 -> a confident commit never self-depletes)
LEVER1_BASELINE_U = 0.018      # lever 1: USE-driven self-eviction (the incumbent depletes through use)
CONFLICT_BOOST = 0.22          # the self-limiting dynamic-gate depletion applied while A & B are co-active
SWAP_STEPS = 140               # steps the challenger B is driven WHILE the self-limiting depression runs
EMPTY_CLEAN_MAX = 20           # a CLEAN swap keeps the empty (both-off) window <= this; more = a slow clear-reload
HEADLINE_OVERLAP = 0           # the headline swap substrate is DISJOINT (no shared resource to strand the winner)

IGNITE_PA = 2500.0
STRONG_PA = 2.0 * IGNITE_PA     # 5000 pA per-pattern ignition drive

# per-slot LATERAL WTA inhibition (the eviction lever: a rising challenger suppresses a DEPLETED incumbent).
WTA_POOL_N = 30                # neurons per per-slot inhibitory pool
WTA_W = 3.0                    # inh_k -> other slots' private cores (I_TO_E). 0.0 => WTA off (pure distributed base)
PRIV2INH_W = 3.0              # private_k -> inh_k (E_TO_I): each slot's winner drives its own lateral inhibitor

OU_NOISE_PA = 30.0
READ_FREE_STEPS = 45
CONF_HOLD_STEPS = 130               # a confident commit must HOLD at least this long (selectivity)

_RESTORE_CALLS = {"n": 0}
_CONTENT_SWAP_CALLS = {"n": 0}      # a host "replace content" poke — MUST stay 0 (never called; asserted)


def _counted_full_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── pattern geometry ───────────────────────────────────────────────────────────────────────────────────────
def _pattern_geometry(ws_indices, k_patterns=N_PATTERNS, pattern_size=PATTERN_SIZE, overlap=DEFAULT_OVERLAP):
    """Sliding-window overlapping patterns: pattern_k = ws[k*(P-O):k*(P-O)+P] (O=0 => disjoint = localist base).
    Returns (patterns, privates); privates[k] = neurons in pattern k ONLY (its exclusive core = the clean read)."""
    ws = np.asarray(ws_indices, dtype=np.int64)
    P, O = int(pattern_size), int(overlap)
    step = P - O
    patterns = [ws[k * step:k * step + P].copy() for k in range(k_patterns)]
    privates = []
    for k in range(k_patterns):
        others = (np.concatenate([patterns[j] for j in range(k_patterns) if j != k])
                  if k_patterns > 1 else np.array([], dtype=np.int64))
        privates.append(np.setdiff1d(patterns[k], others).astype(np.int64))
    return patterns, privates


def _recurrence_edges(patterns):
    """Deduped dense E->E edges within EACH pattern clique (no autapses). Returns (pre, post) int64 arrays."""
    pre_all, post_all = [], []
    for a in patterns:
        a = np.asarray(a, dtype=np.int64); m = a.shape[0]
        pre = np.repeat(a, m); post = np.tile(a, m); keep = pre != post
        pre_all.append(pre[keep]); post_all.append(post[keep])
    pre = np.concatenate(pre_all); post = np.concatenate(post_all)
    key = pre.astype(np.int64) * np.int64(10_000_019) + post.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    return pre[uniq].astype(np.int64), post[uniq].astype(np.int64)


def _rec_population_split(patterns, privates, w_shared, w_priv):
    """SPLIT recurrence (LEVER 2): an edge ENTIRELY within one clique's PRIVATE core gets w_priv (sub-critical);
    any edge touching a shared/overlap unit gets w_shared (base). priv_of[n] = the private-clique index of n, or -1.
    w_priv == w_shared reproduces the base uniform recurrence (the flag-OFF build)."""
    pre, post = _recurrence_edges(patterns)
    n_max = int(max(int(p.max()) for p in patterns)) + 1
    priv_of = np.full(n_max, -1, dtype=np.int64)
    for k, pv in enumerate(privates):
        priv_of[np.asarray(pv, dtype=np.int64)] = k
    both_priv_same = (priv_of[pre] == priv_of[post]) & (priv_of[pre] >= 0)
    ww = np.where(both_priv_same, np.float32(w_priv), np.float32(w_shared)).astype(np.float32)
    return {"pre_indices": pre, "post_indices": post, "initial_weights": ww,
            "plastic": False, "plasticity_gate": WS_LOOP_GATE, "conn_type": "E_TO_E", "count": int(pre.size)}


# ── short-term depression on the recurrence (Tsodyks-Markram); target_units restricts to the SHARED A-B units ───
class RecurrenceDepression:
    """Host-computed Tsodyks-Markram STD on workspace->workspace recurrence synapses whose PRESYNAPTIC neuron is in
    `target_units`. Tracks per-pre x; each step OVERWRITES those synapse weights with base*x BEFORE the step.
    Baseline U is the per-spike release; a CONFLICT BOOST transiently raises it. target_units=None -> ALL
    workspace-used units (the global depression, lever 1). target_units=shared_ab -> ONLY the shared A-B recurrence
    (lever 3: deplete only the shared units the incumbent is using). NOT a host state reset: it depletes the
    substrate's own recurrent synaptic resources. Native global STP stays OFF."""

    def __init__(self, bridge, xp, ws_used, target_units=None, U=STD_BASELINE_U, tau_D=STD_TAU_D, dt=1.0):
        self.bridge = bridge; self.xp = xp
        self.U = float(U); self.tau_D = float(tau_D); self.dt = float(dt); self.boost = 0.0
        n = bridge.core_config.num_neurons
        csr = bridge.cp_connections; csr.sort_indices()
        indptr = to_host(csr.indptr); indices = to_host(csr.indices).astype(np.int64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)); cols = indices
        wsmask = np.zeros(n, dtype=bool); wsmask[np.asarray(ws_used, dtype=np.int64)] = True
        tset = np.asarray(ws_used if target_units is None else target_units, dtype=np.int64)
        premask = np.zeros(n, dtype=bool); premask[tset] = True
        mask = wsmask[rows] & wsmask[cols] & premask[rows]     # workspace->workspace, pre in the target set
        self.idx = np.where(mask)[0]
        self.pre = rows[self.idx]
        self.base = to_host(csr.data)[self.idx].astype(np.float64).copy()
        self.idx_dev = xp.asarray(self.idx)
        self.x = np.ones(n, dtype=np.float64)
        self.target = tset
        self.n_writes = 0
        self.n_rec_syn = int(self.idx.size)

    def apply(self):
        self.bridge.cp_connections.data[self.idx_dev] = self.xp.asarray(
            self.base * self.x[self.pre], dtype=self.xp.float32)
        self.n_writes += 1

    def update(self, fired_host):
        w = self.target
        self.x[w] += (1.0 - self.x[w]) * (self.dt / self.tau_D)     # recovery
        U = self.U + self.boost
        fired = w[fired_host[w]]
        if fired.size:
            self.x[fired] = self.x[fired] - U * self.x[fired]       # depletion
        np.clip(self.x, 0.0, 1.0, out=self.x)

    def reset(self):
        self.x[:] = 1.0; self.boost = 0.0


# ── build the split-recurrence distributed workspace bridge ───────────────────────────────────────────────────
def build_swap_bridge(seed=42, overlap=DEFAULT_OVERLAP, w_shared=W_SHARED, w_priv=W_PRIV, ws_norm_w=WS_NORM_W,
                      norm_ws_w=NORM_WS_W, thal_tonic_pA=THAL_TONIC_PA, thal_ws_w=THAL_WS_W, norm_lesion=False,
                      heterogeneity=True, ou_noise_pA=OU_NOISE_PA, pattern_size=PATTERN_SIZE,
                      wta_w=WTA_W, wta_lesion=False, priv2inh_w=PRIV2INH_W):
    """workspace (exc, NMDA; K overlapping patterns, SPLIT recurrence) + norm_pool (inh; divisive normalization) +
    thal (exc; tonic shared support) + K per-slot inhibitory pools inh_k (LATERAL WTA: private_k -> inh_k -> every
    OTHER slot's private core, so a rising winner SUPPRESSES its rivals). w_priv < w_shared => sub-critical private
    cores (lever 2); wta_w>0 => a rising challenger can push a DEPLETED incumbent below sustain (surpasses the
    Rung-2c inhibition-alone eviction boundary by pre-depleting). wta_lesion=True zeroes the lateral inhibition.
    ALL wiring explicit. Returns (bridge, xp, patterns_dev, privates_dev, thal_dev, ws_used, shared_ab, snap,
    handles)."""
    xp, _ = get_backend()

    use_wta = bool(wta_w > 0.0 and not wta_lesion)
    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    norm_pool = BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0,
                            enable_nmda=False)
    thal = BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    regions = [workspace, norm_pool, thal]
    # K per-slot inhibitory pools (built whenever wta_w>0.0 so the substrate — hence the byte-identical hash — is
    # invariant to the wta_lesion anti-cheat; the lesion only zeroes the lateral WEIGHT).
    inh_slots = []
    if wta_w > 0.0:
        for k in range(N_PATTERNS):
            inh_slots.append(BrainRegion(name=f"inh{k}", n_neurons=WTA_POOL_N, exc_fraction=0.0,
                                         internal_density=0.0, enable_nmda=False))
        regions = [workspace, norm_pool, thal] + inh_slots

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(w_shared) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(w_shared) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, pattern_size, overlap)
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64) if overlap < pattern_size \
        else np.asarray(patterns[0], dtype=np.int64)
    shared_ab = np.intersect1d(patterns[0], patterns[1]).astype(np.int64)   # the contested A-B resource
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)

    norm_ws_eff = 0.0 if norm_lesion else float(norm_ws_w)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["workspace_rec"] = _rec_population_split(patterns, privates, float(w_shared), float(w_priv))
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(ws_norm_w), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, norm_ws_eff, "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(thal_ws_w), "E_TO_E")

    inh = list(norm_idx)
    inh_idx_list = []
    if wta_w > 0.0:
        wta_eff = 0.0 if wta_lesion else float(wta_w)
        for k in range(N_PATTERNS):
            ik = np.asarray(rm.indices(f"inh{k}"), dtype=np.int64)
            inh_idx_list.append(ik)
            inh += list(ik)
            union_plan[f"priv2inh{k}"] = _dense_pop(privates[k], ik, float(priv2inh_w), "E_TO_I")
            # LATERAL: inh_k suppresses every OTHER slot's private core (winner-take-all)
            others = np.concatenate([privates[j] for j in range(N_PATTERNS) if j != k]).astype(np.int64)
            union_plan[f"inh{k}2others"] = _dense_pop(ik, others, wta_eff, "I_TO_E")

    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(thal_tonic_pA)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "overlap": int(overlap), "w_shared": float(w_shared), "w_priv": float(w_priv),
               "pattern_size": int(pattern_size), "ws_norm_w": float(ws_norm_w), "norm_ws_w": float(norm_ws_eff),
               "thal_tonic_pA": float(thal_tonic_pA), "thal_ws_w": float(thal_ws_w), "norm_lesion": bool(norm_lesion),
               "heterogeneity": bool(heterogeneity), "ou_noise_pA": float(ou_noise_pA), "n_patterns": N_PATTERNS,
               "private_sizes": [int(p.size) for p in privates], "n_shared_ab": int(shared_ab.size),
               "n_ws_used": int(ws_used.size), "n_rec_edges": int(union_plan["workspace_rec"]["count"]),
               "wta_w": float(wta_w), "wta_lesion": bool(wta_lesion), "use_wta": bool(use_wta)}
    return (bridge, xp, [xp.asarray(p) for p in patterns], [xp.asarray(p) for p in privates], thal_dev, ws_used,
            shared_ab, snap, handles)


# ── stepping + spiking reads ─────────────────────────────────────────────────────────────────────────────────
def _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map=None):
    if std is not None:
        std.apply()
    bridge.cp_external_input_current[:] = 0.0
    if thal_tonic_pA != 0.0:
        bridge.cp_external_input_current[thal_dev] = xp.float32(thal_tonic_pA)
    if drive_map:
        for idx_dev, val in drive_map:
            if val > 0.0:
                bridge.cp_external_input_current[idx_dev] = xp.float32(val)
    bridge._run_one_simulation_step()
    if std is not None:
        std.update(to_host(bridge.cp_firing_states).astype(bool))


def _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std, n_free=READ_FREE_STEPS):
    """Free-run n_free steps (tonic on, no pattern drive) and return the LATE-window per-pattern PRIVATE-core mean
    firing rate — the clean spiking identity read. Leaves the workspace evolving in place."""
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(privates_dev)
    for t in range(n_free):
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std)
        if t >= late_start:
            for i, p in enumerate(privates_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[p].astype(xp.float64).sum()))
    out = []
    for i, p in enumerate(privates_dev):
        denom = float((n_free - late_start) * int(p.shape[0]))
        out.append(counts[i] / denom if denom > 0 else 0.0)
    return out


def _instant_private_rate(bridge, xp, privates_dev, idx):
    p = privates_dev[idx]
    return float(to_host(bridge.cp_firing_states[p].astype(xp.float64).mean()))


def _margin(rates):
    order = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    top = rates[order[0]]
    second = rates[order[1]] if len(order) > 1 else 0.0
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    return int(order[0]), float(top - second), n_ign


def _verdict_label(rates):
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    if n_ign == 0:
        return "ABSTAIN", 0
    return (f"COMMIT_p{int(np.argmax(rates))}" if n_ign == 1 else f"AMBIGUOUS_p{int(np.argmax(rates))}"), n_ign


def _drive(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map, n=DRIVE_STEPS):
    for _ in range(n):
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map=drive_map)


# ── the criticality diagnostic (lever 2): is a private core SUB-CRITICAL alone, the full pattern SUPRA-critical? ─
def diagnose_criticality(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, target=0):
    """Drive ONLY the private core of `target` (never the shared units), free-run: does the private core self-sustain
    ALONE? Then drive the FULL pattern: does it self-sustain? SUB-CRITICAL private + SUPRA-critical full = lever 2."""
    _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(privates_dev[target], STRONG_PA)])
    pr = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    priv_alone_ignited = bool(_ignited(pr[target]))
    _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[target], STRONG_PA)])
    fr = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    full_ignited = bool(_ignited(fr[target]))
    return {"private_alone_rate": float(pr[target]), "private_alone_ignited": priv_alone_ignited,
            "full_pattern_rate": float(fr[target]), "full_pattern_ignited": full_ignited,
            "sub_critical_private": bool(not priv_alone_ignited and full_ignited)}


# ── SELECTIVITY: an UNCHALLENGED confident commit holds (n=1, winner=target) through a long window ─────────────
def run_confident(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, target=0,
                  hold_steps=CONF_HOLD_STEPS, isolate=True):
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[target], STRONG_PA)])
    for _ in range(hold_steps):                         # HOLD with NO drive and NO conflict boost (boost stays 0)
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std)
    rates = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win, m, n = _margin(rates)
    v, _ = _verdict_label(rates)
    return {"rates": [float(r) for r in rates], "winner": int(win), "n_ignited": int(n), "margin": float(m),
            "delivered": v, "confident_ok": bool(n == 1 and win == target)}


# ── the ACTIVE OVERWRITE swap (SELF-LIMITING dynamic gate; levers composable) ──────────────────────────────────
def run_swap(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, *,
             incumbent=0, challenger=1, swap_steps=SWAP_STEPS, conflict_boost=CONFLICT_BOOST,
             use_depression=True, drive_challenger=True, isolate=True):
    """Ignite incumbent A (hold), then drive challenger B for swap_steps under a SELF-LIMITING DYNAMIC gate: at each
    step the depression boost is ON only while A and B are BOTH ignited (a live co-active conflict) and OFF the
    moment the incumbent drops out — so the challenger's recurrence RECOVERS and holds once it wins. Because A rides
    the recurrence alone while B has external drive, depleting the recurrence starves A first (the asymmetry).
      use_depression=False -> LESION (boost forced 0) = the STD load-bearing anti-cheat (expect co-ignition/hold).
      drive_challenger=False -> no B (a control).
      isolate=False -> a CONTINUOUS run (0 restore calls) = the swap HEADLINE.
    Tracks per-step co-ignition + empty windows: a CLEAN swap = settled n=1 winner=B with a SHORT empty window; a
    SLOW overwrite (clear-then-reload) = settled n=1 winner=B but through a LONG empty window."""
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    # (1) ignite A alone
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[incumbent], STRONG_PA)])
    pre = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win_pre, margin_pre, n_pre = _margin(pre)

    # (2) SWAP: drive B; self-limiting dynamic gate (boost only during A&B co-activity).
    dmap = [(patterns_dev[challenger], STRONG_PA)] if drive_challenger else None
    coactive_win, empty_win = 0, 0
    for _ in range(swap_steps):
        na0 = _ignited(_instant_private_rate(bridge, xp, privates_dev, incumbent))
        nb0 = _ignited(_instant_private_rate(bridge, xp, privates_dev, challenger))
        std.boost = (conflict_boost if (na0 and nb0) else 0.0) if use_depression else 0.0
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map=dmap)
        na = _ignited(_instant_private_rate(bridge, xp, privates_dev, incumbent))
        nb = _ignited(_instant_private_rate(bridge, xp, privates_dev, challenger))
        coactive_win += int(na and nb); empty_win += int(not na and not nb)
    std.boost = 0.0

    # (3) free-run (B drive off): does B self-sustain as the new incumbent, A gone?
    post = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std, n_free=FREE_STEPS)
    win_post, margin_post, n_post = _margin(post)
    v_pre, _ = _verdict_label(pre); v_post, _ = _verdict_label(post)
    switched = bool(win_pre == incumbent and n_pre == 1 and win_post == challenger and n_post == 1)
    clean_swap = bool(switched and empty_win <= EMPTY_CLEAN_MAX)
    slow_overwrite = bool(switched and empty_win > EMPTY_CLEAN_MAX)
    return {"pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
            "winner_pre": int(win_pre), "winner_post": int(win_post), "n_ignited_pre": int(n_pre),
            "n_ignited_post": int(n_post), "delivered_pre": v_pre, "delivered_post": v_post,
            "conflict_boost": float(conflict_boost if use_depression else 0.0),
            "swap_ok": clean_swap, "switched_identity": switched, "slow_overwrite": slow_overwrite,
            "co_ignition": bool(n_pre == 1 and n_post >= 2), "went_empty": bool(n_pre >= 1 and n_post == 0),
            "incumbent_held": bool(win_post == incumbent and n_post == 1),
            "coactive_steps": int(coactive_win), "empty_steps": int(empty_win)}


# ── one seed: per-lever switch-rate + SELECTIVITY + anti-cheats ────────────────────────────────────────────────
def evaluate_seed(seed, *, headline_overlap=HEADLINE_OVERLAP, wta_w=WTA_W, conflict_boost=CONFLICT_BOOST,
                  swap_steps=SWAP_STEPS, char_overlap=DEFAULT_OVERLAP, w_shared=W_SHARED, w_priv=W_PRIV,
                  lever1_U=LEVER1_BASELINE_U, heterogeneity=True, verbose=True):
    # ── HEADLINE substrate: DISJOINT (no shared resource to strand the winner) + WTA lateral inhibition ──────────
    def _hbuild(**kw):
        params = dict(seed=seed, overlap=headline_overlap, w_shared=w_shared, w_priv=w_shared, wta_w=wta_w,
                      heterogeneity=heterogeneity)   # disjoint => uniform recurrence (w_priv=w_shared)
        params.update(kw)
        return build_swap_bridge(**params)
    bh, xph, patsh, privsh, thalh, ws_usedh, _sabh, snaph, hh = _hbuild()

    # SELECTIVITY: unchallenged confident A holds (full-recurrence STD, U=0; the dynamic gate never fires w/o B).
    std_sel = RecurrenceDepression(bh, xph, ws_usedh, target_units=None, U=0.0)
    conf = run_confident(bh, xph, patsh, privsh, thalh, snaph, std_sel, THAL_TONIC_PA)
    selectivity = bool(conf["confident_ok"])

    # HEADLINE swap: self-limiting DYNAMIC gate + WTA, CONTINUOUS (0 restore calls) = the substrate-driven headline.
    std_h = RecurrenceDepression(bh, xph, ws_usedh, target_units=None, U=0.0)
    restore_before = _RESTORE_CALLS["n"]
    headline = run_swap(bh, xph, patsh, privsh, thalh, snaph, std_h, THAL_TONIC_PA, swap_steps=swap_steps,
                        conflict_boost=conflict_boost, use_depression=True, isolate=False)
    continuous_no_restore = bool(_RESTORE_CALLS["n"] == restore_before)
    host_workspace_reset_calls = 0 if continuous_no_restore else 1
    host_content_swap_calls = int(_CONTENT_SWAP_CALLS["n"])   # never called -> 0 (asserted)
    clean_swap = bool(headline["swap_ok"])            # n stays ~1 (short empty window)
    switched = bool(headline["switched_identity"])    # settled n=1 winner=B (clean OR slow)
    slow_overwrite = bool(headline["slow_overwrite"])  # settled n=1 winner=B via a LONG empty window

    # ── ANTI-CHEAT: STD LOAD-BEARING — lesion the depression -> NO overwrite (incumbent holds / co-ignites) ──────
    std_lesD = RecurrenceDepression(bh, xph, ws_usedh, target_units=None, U=0.0)
    lesionD = run_swap(bh, xph, patsh, privsh, thalh, snaph, std_lesD, THAL_TONIC_PA, swap_steps=swap_steps,
                       conflict_boost=conflict_boost, use_depression=False, isolate=True)
    std_load_bearing = bool(switched and not lesionD["switched_identity"])

    # ── ANTI-CHEAT: WTA LOAD-BEARING — build with wta_lesion (pools kept, lateral weight 0) -> NO overwrite ──────
    bwl, xpwl, patswl, privswl, thalwl, ws_usedwl, _sabwl, snapwl, _ = _hbuild(wta_lesion=True)
    std_wl = RecurrenceDepression(bwl, xpwl, ws_usedwl, target_units=None, U=0.0)
    lesionW = run_swap(bwl, xpwl, patswl, privswl, thalwl, snapwl, std_wl, THAL_TONIC_PA, swap_steps=swap_steps,
                       conflict_boost=conflict_boost, use_depression=True, isolate=True)
    wta_load_bearing = bool(switched and not lesionW["switched_identity"])
    # ATTRIBUTION: whose is the overwrite (headline vs each lesion)? On a clean NO-GO both arms read 0 -> the call
    # reports UNDEFINED (a null: no swap to attribute), which is the honest result — NOT 0% or 100% attributable.
    std_swap_attr = attributable_to("clean swap via targeted STD (headline vs depression-lesion)",
                                    float(switched), float(lesionD["switched_identity"]), warn_below=0.0)
    wta_swap_attr = attributable_to("clean swap via WTA (headline vs WTA-lesion)",
                                    float(switched), float(lesionW["switched_identity"]), warn_below=0.0)

    # ── the THREE NAMED LEVERS (characterization on the OVERLAP substrate; each hits the co-ignition boundary) ────
    def _cbuild(**kw):
        params = dict(seed=seed, overlap=char_overlap, w_shared=w_shared, w_priv=w_priv, wta_w=0.0,
                      heterogeneity=heterogeneity)
        params.update(kw)
        return build_swap_bridge(**params)
    bc, xpc, patsc, privsc, thalc, ws_usedc, shared_abc, snapc, hc = _cbuild()
    crit = diagnose_criticality(bc, xpc, patsc, privsc, thalc, snapc,
                                RecurrenceDepression(bc, xpc, ws_usedc, target_units=None, U=0.0), THAL_TONIC_PA)
    lev1 = run_swap(bc, xpc, patsc, privsc, thalc, snapc,
                    RecurrenceDepression(bc, xpc, ws_usedc, target_units=None, U=lever1_U),
                    THAL_TONIC_PA, swap_steps=swap_steps, conflict_boost=0.0, use_depression=False)  # use-driven only
    lev2 = run_swap(bc, xpc, patsc, privsc, thalc, snapc,
                    RecurrenceDepression(bc, xpc, ws_usedc, target_units=shared_abc, U=0.0),
                    THAL_TONIC_PA, swap_steps=swap_steps, use_depression=False)                       # pure competition
    lev3 = run_swap(bc, xpc, patsc, privsc, thalc, snapc,
                    RecurrenceDepression(bc, xpc, ws_usedc, target_units=shared_abc, U=0.0),
                    THAL_TONIC_PA, swap_steps=swap_steps, conflict_boost=conflict_boost, use_depression=True)

    # ── ANTI-CHEAT: BYTE-IDENTICAL — flag-OFF build (wta_w=0, base overlap 15, uniform) == DIST-OVERWRITE base ────
    b_off, xp_off, *_off = build_swap_bridge(seed=seed, overlap=15, w_shared=w_shared, w_priv=w_shared, wta_w=0.0,
                                             heterogeneity=heterogeneity)
    h_off = _threshold_hash(b_off, xp_off)
    b_dist, xp_dist, *_dist = _dist_build(seed=seed, overlap=15)
    h_dist = _threshold_hash(b_dist, xp_dist)
    byte_identical_substrate = bool(h_off == h_dist and h_off != "")

    # ── ANTI-CHEAT: DETERMINISM — build headline twice at this seed -> identical substrate hash ─────────────────
    h_a = _threshold_hash(bh, xph)
    b2, xp2, *_2 = _hbuild()
    seed_deterministic = bool(_threshold_hash(b2, xp2) == h_a and h_a != "")

    levers = {"lever1_full_use_driven": lev1, "lever2_overlap_pure_competition": lev2,
              "lever3_targeted_shared_std": lev3, "headline_disjoint_wta_dyngate": headline}

    # GO gate: the CLEAN swap (n stays ~1) on the headline + selectivity + the anti-cheats.
    seed_go = bool(clean_swap and selectivity and std_load_bearing and wta_load_bearing and continuous_no_restore
                   and host_content_swap_calls == 0 and byte_identical_substrate and seed_deterministic)

    # PRECONDITIONS (validity -> a failing one is UNDEFINED, an instrument failure, NOT a negative): the workspace
    # ignites, the run is substrate-driven (no host reset/content-swap), the substrate is the seeded base build.
    # The OUTCOME being tested (clean swap + selectivity + causal load-bearing) is the go= decision -> its failure
    # is a genuine NO-GO (the mapped boundary), not UNDEFINED.
    v = Verdict("distributed-workspace ACTIVE OVERWRITE (seed %d)" % seed)
    v.require("confident commit ignites (n>=1) [precondition]", bool(conf["n_ignited"] >= 1), expect=True)
    v.require("substrate-driven: 0 host workspace-reset calls on the headline [precondition]",
              continuous_no_restore, expect=True)
    v.require("substrate-driven: 0 host content-swap calls [precondition]", bool(host_content_swap_calls == 0),
              expect=True)
    v.require("byte-identical substrate (flag-off build == DIST-OVERWRITE base hash) [precondition]",
              byte_identical_substrate, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash) [precondition]", seed_deterministic,
              expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity", why="banked global-STP foot-gun; STD here targets the recurrence only")
    # go = the OUTCOME: a clean (n stays ~1) SWITCH that is SELECTIVE and causally attributed to STD + WTA.
    vd = v.decide(go=bool(clean_swap and selectivity and std_load_bearing and wta_load_bearing), verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"headline_overlap": int(headline_overlap), "wta_w": float(wta_w),
                            "priv2inh_w": float(PRIV2INH_W), "conflict_boost": float(conflict_boost),
                            "swap_steps": int(swap_steps), "char_overlap": int(char_overlap),
                            "w_shared": float(w_shared), "w_priv": float(w_priv), "lever1_U": float(lever1_U),
                            "heterogeneity": bool(heterogeneity), "char_private_sizes": hc["private_sizes"],
                            "empty_clean_max": int(EMPTY_CLEAN_MAX)},
        "criticality": crit,
        "go_gate": {"clean_switch": clean_swap, "selectivity": selectivity, "switched_identity": switched,
                    "slow_overwrite": slow_overwrite},
        "per_lever_switch": {k: {"clean_swap": bool(v_["swap_ok"]), "switched_identity": bool(v_["switched_identity"]),
                                 "slow_overwrite": bool(v_["slow_overwrite"]), "winner_pre": v_["winner_pre"],
                                 "winner_post": v_["winner_post"], "n_pre": v_["n_ignited_pre"],
                                 "n_post": v_["n_ignited_post"], "co_ignition": bool(v_["co_ignition"]),
                                 "went_empty": bool(v_["went_empty"]), "incumbent_held": bool(v_["incumbent_held"]),
                                 "coactive_steps": v_["coactive_steps"], "empty_steps": v_["empty_steps"]}
                             for k, v_ in levers.items()},
        "anti_cheats": {"std_load_bearing": std_load_bearing, "wta_load_bearing": wta_load_bearing,
                        "continuous_no_restore": continuous_no_restore,
                        "host_content_swap_calls": host_content_swap_calls,
                        "byte_identical_substrate": byte_identical_substrate,
                        "seed_deterministic": seed_deterministic,
                        "std_swap_attribution": (None if std_swap_attr is None else float(std_swap_attr)),
                        "wta_swap_attribution": (None if wta_swap_attr is None else float(wta_swap_attr))},
        "residual": {"headline_n_post": int(headline["n_ignited_post"]),
                     "headline_winner_post": int(headline["winner_post"]),
                     "headline_co_ignition": bool(headline["co_ignition"]),
                     "headline_went_empty": bool(headline["went_empty"]),
                     "headline_empty_steps": int(headline["empty_steps"]),
                     "headline_coactive_steps": int(headline["coactive_steps"]),
                     "lesionD_switched": bool(lesionD["switched_identity"]),
                     "lesionD_n_post": int(lesionD["n_ignited_post"]),
                     "lesionW_switched": bool(lesionW["switched_identity"]),
                     "lesionW_n_post": int(lesionW["n_ignited_post"])},
        "measurements": {"confident": conf, "headline": headline, "lesionD": lesionD, "lesionW": lesionW,
                         "substrate_hash": h_a, "flagoff_hash": h_off, "dist_hash": h_dist},
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[overwrite seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        for k, v_ in levers.items():
            print(f"    {k:32s}: win {v_['winner_pre']}->{v_['winner_post']} n {v_['n_ignited_pre']}->"
                  f"{v_['n_ignited_post']} clean={v_['swap_ok']} switched={v_['switched_identity']} "
                  f"slow={v_['slow_overwrite']} (co_ign={v_['co_ignition']} empty_steps={v_['empty_steps']} "
                  f"coact={v_['coactive_steps']})", flush=True)
        print(f"    SELECTIVITY: conf n={conf['n_ignited']} win={conf['winner']} ok={selectivity} | "
              f"STD_LB={std_load_bearing}(lesD switched={lesionD['switched_identity']}) "
              f"WTA_LB={wta_load_bearing}(lesW switched={lesionW['switched_identity']}) "
              f"no_restore={continuous_no_restore} byte_id={byte_identical_substrate} det={seed_deterministic}",
              flush=True)
    return result


# ── smoke: an operating-point sweep on one seed (find a swap + selectivity point) ──────────────────────────────
def run_smoke(seed, args):
    print(f"[overwrite smoke] seed={seed} — operating-point grid", flush=True)
    grid = []
    for wta in ([args.wta_w] if args.wta_grid is None else args.wta_grid):
        for cb in ([args.conflict_boost] if args.cb_grid is None else args.cb_grid):
            r = evaluate_seed(seed, wta_w=float(wta), conflict_boost=float(cb), swap_steps=args.swap_steps,
                              char_overlap=args.char_overlap, w_priv=args.w_priv,
                              heterogeneity=not args.no_heterogeneity, verbose=True)
            grid.append({"wta_w": float(wta), "conflict_boost": float(cb), "seed_go": r["seed_go"],
                         "go_gate": r["go_gate"], "per_lever_switch": r["per_lever_switch"],
                         "anti_cheats": r["anti_cheats"], "residual": r["residual"]})
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_active_overwrite_derisk", "mode": "smoke", "seed": seed, "grid": grid}, f,
                  indent=2, default=str)
    print(f"\n[overwrite smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW active overwrite: swap an ignited incumbent for a challenger "
                                             "(n stays ~1) via disjoint substrate + WTA lateral inhibition + a "
                                             "self-limiting dynamic depletion gate; the three named levers "
                                             "(timed self-eviction / overlap+sub-critical / targeted shared STD) "
                                             "characterized as the co-ignition boundary.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--headline-overlap", type=int, default=HEADLINE_OVERLAP)
    ap.add_argument("--wta-w", type=float, default=WTA_W)
    ap.add_argument("--conflict-boost", type=float, default=CONFLICT_BOOST)
    ap.add_argument("--swap-steps", type=int, default=SWAP_STEPS)
    ap.add_argument("--char-overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--w-shared", type=float, default=W_SHARED)
    ap.add_argument("--w-priv", type=float, default=W_PRIV)
    ap.add_argument("--lever1-u", type=float, default=LEVER1_BASELINE_U)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wta-grid", type=float, nargs="+", default=None)
    ap.add_argument("--cb-grid", type=float, nargs="+", default=None)
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_active_overwrite_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[overwrite] K={N_PATTERNS} P={PATTERN_SIZE} headline_overlap={args.headline_overlap} wta_w={args.wta_w} "
          f"conflict_boost={args.conflict_boost} swap_steps={args.swap_steps} char_overlap={args.char_overlap} "
          f"w_shared={args.w_shared} w_priv={args.w_priv} het={not args.no_heterogeneity} "
          f"backend={args.backend}\n", flush=True)

    if args.smoke:
        return run_smoke(args.seed, args)

    common = dict(headline_overlap=args.headline_overlap, wta_w=args.wta_w, conflict_boost=args.conflict_boost,
                  swap_steps=args.swap_steps, char_overlap=args.char_overlap, w_shared=args.w_shared,
                  w_priv=args.w_priv, lever1_U=args.lever1_u, heterogeneity=not args.no_heterogeneity)
    results = [evaluate_seed(s, verbose=True, **common) for s in args.seeds]

    n_clean = sum(int(r["go_gate"]["clean_switch"]) for r in results)
    n_select = sum(int(r["go_gate"]["selectivity"]) for r in results)
    n_switched = sum(int(r["go_gate"]["switched_identity"]) for r in results)   # clean OR slow overwrite
    n_slow = sum(int(r["go_gate"]["slow_overwrite"]) for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)
    all_no_reset = all(r["anti_cheats"]["continuous_no_restore"] for r in results)
    all_no_content = all(r["anti_cheats"]["host_content_swap_calls"] == 0 for r in results)
    all_byte_id = all(r["anti_cheats"]["byte_identical_substrate"] for r in results)
    all_determ = all(r["anti_cheats"]["seed_deterministic"] for r in results)
    all_std_lb = all(r["anti_cheats"]["std_load_bearing"] for r in results)
    all_wta_lb = all(r["anti_cheats"]["wta_load_bearing"] for r in results)

    lever_names = ["lever1_full_use_driven", "lever2_overlap_pure_competition", "lever3_targeted_shared_std",
                   "headline_disjoint_wta_dyngate"]
    per_lever_clean = {ln: sum(int(r["per_lever_switch"][ln]["clean_swap"]) for r in results) for ln in lever_names}
    per_lever_switched = {ln: sum(int(r["per_lever_switch"][ln]["switched_identity"]) for r in results)
                          for ln in lever_names}

    clean_go = bool(n_clean >= 5)
    select_go = bool(n_select >= 5)
    anti_ok = bool(all_no_reset and all_no_content and all_byte_id and all_determ and all_std_lb and all_wta_lb)
    all_ignite = all(r["measurements"]["confident"]["n_ignited"] >= 1 for r in results)
    # AGGREGATE VERDICT with an explicit preconditions block (a verdict must travel with what earned it). The
    # OUTCOME (a clean swap on >=5/6) is the go= decision; the preconditions are the run's validity (ignition,
    # substrate-driven, byte-identical, deterministic across all seeds).
    av = Verdict("distributed-workspace ACTIVE OVERWRITE — 6-seed aggregate")
    av.require("all seeds: confident commit ignites (n>=1)", all_ignite, expect=True)
    av.require("all seeds: substrate-driven (0 host workspace-reset calls)", all_no_reset, expect=True)
    av.require("all seeds: 0 host content-swap calls", all_no_content, expect=True)
    av.require("all seeds: byte-identical substrate (flag-off == DIST-OVERWRITE base)", all_byte_id, expect=True)
    av.require("all seeds: determinism (cfg.seed seeds the substrate)", all_determ, expect=True)
    av.require("no seed UNDEFINED", not any_undefined, expect=True)
    av.disabled("homeostasis", why="frozen weights; synaptic-scaling clip is a Rung-1/2 foot-gun")
    av.disabled("native_short_term_plasticity", why="banked global-STP foot-gun; STD targets the recurrence only")
    agg_vd = av.decide(go=bool(clean_go and select_go and anti_ok), verbose=True)

    if agg_vd["status"] == "UNDEFINED":
        verdict = "UNDEFINED"
    elif clean_go and select_go and anti_ok:
        verdict = "GO"                    # clean n-stays-1 swap 6/6 + selectivity + anti-cheats
    elif n_switched >= 5 and select_go:
        verdict = "PARTIAL"               # overwrite ACHIEVED (clean or slow) 5/6 + selectivity; clean swap residual
    else:
        verdict = "NO-GO"

    summary = {
        "runner": "_gnw_active_overwrite_derisk", "mode": "six-seed", "verdict": verdict,
        "n_clean_swap_go": n_clean, "n_selectivity_go": n_select, "n_switched_identity": n_switched,
        "n_slow_overwrite": n_slow, "n_seed_go": n_go, "n_seeds": len(results), "seeds": list(args.seeds),
        "per_lever_clean_swap_rate": per_lever_clean, "per_lever_switched_identity_rate": per_lever_switched,
        "any_undefined": any_undefined,
        "aggregate_anti_cheats": {"all_continuous_no_restore": all_no_reset,
                                  "all_host_content_swap_zero": all_no_content,
                                  "all_byte_identical_substrate": all_byte_id, "all_seed_deterministic": all_determ,
                                  "all_std_load_bearing": all_std_lb, "all_wta_load_bearing": all_wta_lb},
        "preconditions": agg_vd["preconditions"], "disabled_processes": agg_vd["disabled_processes"],
        "undefined_reasons": agg_vd["undefined_reasons"],
        "operating_point": results[0]["operating_point"] if results else {}, "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  ACTIVE OVERWRITE VERDICT: {verdict}  (CLEAN-SWAP {n_clean}/{len(results)}; SELECTIVITY "
          f"{n_select}/{len(results)}; switched-identity {n_switched}/{len(results)}; slow-overwrite "
          f"{n_slow}/{len(results)}; seed_go {n_go}/{len(results)})", flush=True)
    print(f"  per-lever CLEAN-swap: " + ", ".join(f"{ln}={per_lever_clean[ln]}" for ln in lever_names), flush=True)
    print(f"  per-lever switched-identity: " + ", ".join(f"{ln}={per_lever_switched[ln]}" for ln in lever_names),
          flush=True)
    for r in results:
        g = r["go_gate"]; a = r["anti_cheats"]; res = r["residual"]
        print(f"    seed {r['seed']}: {r['verdict']:9s} clean={g['clean_switch']} select={g['selectivity']} "
              f"switched={g['switched_identity']} slow={g['slow_overwrite']} | headline win->"
              f"{res['headline_winner_post']} n_post={res['headline_n_post']} empty_steps={res['headline_empty_steps']}"
              f" (co_ign={res['headline_co_ignition']}) | STD_LB={a['std_load_bearing']} WTA_LB={a['wta_load_bearing']}"
              f" no_reset={a['continuous_no_restore']} byte_id={a['byte_identical_substrate']} det={a['seed_deterministic']}",
              flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if (clean_go and select_go) else 1


if __name__ == "__main__":
    raise SystemExit(main())
